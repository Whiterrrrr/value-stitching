import copy
import functools
import math
from typing import Any, Dict, Tuple
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue
from utils.encoders import GCEncoder, encoder_modules

# AWR weights are capped at 100, applied to the log-advantage so exp() cannot
# overflow on the way there. A plain float, so importing this module does not
# initialize a JAX device.
_AWR_LOG_CAP = math.log(100.0)

class VASTAgent(flax.struct.PyTreeNode):
    """VAST agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    @staticmethod
    def asymmetric_l2_loss(pred: jnp.ndarray, target: jnp.ndarray, expectile: float) -> jnp.ndarray:
        """Compute the asymmetric L2 loss for expectile regression."""
        diff = pred - target
        weight = jnp.where(target > pred, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def g_value_loss(self, batch, grad_params):
        """Compute the high-level SARSA value loss."""
        observations, goal_k, goal_j = batch["observations"], batch["goal_k"], batch["goal_j"]
        # Same scale as the critic. G, V, and Q must match or the composition below breaks.
        k_discounted_returns = batch["k_discounted_returns"] * self.config.get("reward_scale", 1.0)
        # mc_mask: k >= 1. comp_mask: k > j >= 1. Both losses are normalized by the valid count.
        mc_mask, comp_mask = batch["mc_mask"], batch["comp_mask"]
        ks, js = batch["k"], batch["j"]
        
        def g_mc_loss() -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            """
            Monte-Carlo endpoint supervision:
            G(s_t, s_{t+k}) ~= sum_{j=0}^{k-1} gamma^j r_{t+j}
            """
            pred = self.network.select("g_value")(observations, goal_k, ks[:, None], params=grad_params)

            loss = ((pred - k_discounted_returns) ** 2 * mc_mask).sum() / jnp.maximum(mc_mask.sum(), 1.0)
            info = {
                "g_mc/loss": loss,
                "g_mc/pred_mean": jnp.mean(pred),
            }
            return loss, info

        def g_comp_loss() -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            """
            Compositional consistency:
            G(s_t, s_{t+k}) ~= G(s_t, s_{t+j}) + gamma^j * G(s_{t+j}, s_{t+k})
            """
            g_tk = self.network.select("g_value")(observations, goal_k, ks[:, None], params=grad_params)
            g_tj = self.network.select("g_value")(observations, goal_j, js[:, None], params=grad_params)
            jk_input = ks - js
            g_jk = self.network.select("g_value")(goal_j, goal_k, jk_input[:, None], params=grad_params)

            rhs = g_tj + (self.config['discount'] ** js) * g_jk
            loss = ((g_tk - rhs) ** 2 * comp_mask).sum() / jnp.maximum(comp_mask.sum(), 1.0)
            
            info = {"g_comp/loss": loss}
            return loss, info

        mc_loss, info = g_mc_loss()
        comp_loss, comp_info = g_comp_loss()
        
        total_g_loss = mc_loss + self.config["comp_coef"] * comp_loss

        info.update(comp_info)
        return total_g_loss, info
    
    def critic_loss(self, batch, grad_params, rng):
        observations, actions, next_observations = batch['observations'], batch['actions'], batch['next_observations']
        masks = batch['masks']
        # OGBench rewards are successes - 1. `reward_scale` (often 1 - discount) rescales
        # the return onto a smaller range without changing the greedy policy.
        rewards = batch['rewards'] * self.config.get('reward_scale', 1.0)

        high_rng, next_high_rng, low_rng = jax.random.split(rng, 3)
        sub_goals, ks = self.sample_high_actions(observations, seed=high_rng)
        next_sub_goals, _ = self.sample_high_actions(sub_goals, seed=next_high_rng, is_encoded=True)
        next_low_actions = self.sample_low_actions_train(
            sub_goals, goals=next_sub_goals, seed=low_rng,
            obs_encoded=True, goal_encoded=True
        )
        ks = ks.astype(jnp.float32)

        next_q = self.network.select('target_critic')(sub_goals, actions=next_low_actions, is_encoded=True)
        if self.config.get('q_agg', 'min') == 'min':
            next_q = jnp.min(next_q, axis=0)
        else:
            next_q = jnp.mean(next_q, axis=0)

        g_t_k = self.network.select('g_value')(observations, sub_goals, ks[:, None], goal_encoded=True)
        v_target = g_t_k + (self.config['discount'] ** ks) * next_q
        v = self.network.select('value')(observations, params=grad_params)
        next_v = self.network.select('value')(next_observations)
        q_target = rewards + next_v * self.config['discount'] * masks
        q1, q2 = self.network.select('critic')(observations, actions=actions, params=grad_params)

        # Do not mask by mc_mask: v_target uses on-policy subgoals, not the batch's k/j.
        v_loss = self.asymmetric_l2_loss(v, v_target, expectile=self.config['expectile']).mean()

        # Do not mask by `masks`. It already zeros the bootstrap in q_target, so terminal
        # transitions anchor Q = 0. Dropping them leaves the critic free to drift.
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return critic_loss, v_loss, {
            'critic_loss': critic_loss,
            'value_loss': v_loss,
            'q1_mean': q1.mean(),
            'v': v.mean(),
            'q_target_mean': q_target.mean(),
            'terminal_frac': (1.0 - masks).mean(),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level flow actor loss with value g + q."""
        observations = batch['observations']
        sub_goals, ks, next_low_actions = batch['goal_k'], batch['k'], batch['actions_k']
        
        qs = self.network.select("critic")(sub_goals, actions=next_low_actions)
        q = jnp.mean(qs, axis=0)
        g = self.network.select("g_value")(observations, sub_goals, ks[:, None])
        value = g + (self.config['discount'] ** ks) * q
        v = self.network.select('value')(observations)
        adv = value - v
        # Clamp before exp. Unscaled advantages overflow float32, and once most
        # weights sit at the cap or at 0, AWR collapses to BC on part of the batch.
        log_weight = jnp.minimum(adv * self.config['scale'], _AWR_LOG_CAP)
        weight_value = jnp.exp(log_weight)
        
        def flow_policy_loss_with_weight():
            x_rng, t_rng = jax.random.split(rng, 2)
            latent_dim = self.config['latent_dim']
            batch_size = observations.shape[0]
            
            if self.config.get('encoder', None) is not None:
                sub_goals_latent = self.network.select('state_encoder')(sub_goals)
            else:
                sub_goals_latent = sub_goals
            
            x_0 = jax.random.normal(x_rng, (batch_size, latent_dim + 1))
            x_1 = jnp.concatenate([sub_goals_latent, ks[:, None]], axis=-1)
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0
            
            pred = self.network.select('high_actor_flow')(observations, actions=x_t, times=t, params=grad_params)
            return (((pred - y) ** 2).mean(axis=-1) * weight_value).mean()
        
        actor_loss = flow_policy_loss_with_weight()        
        actor_info = {
            'high_actor_loss': actor_loss,
            'high_actor_adv': jnp.mean(adv),
            'high_actor_adv_max': adv.max(),
            'high_actor_adv_min': adv.min(),
            'high_actor_weight_value': jnp.mean(weight_value),
            'high_actor_weight_at_cap': jnp.mean(log_weight >= _AWR_LOG_CAP),
            'high_actor_weight_underflow': jnp.mean(weight_value <= 1e-6),
        }

        return actor_loss, actor_info

    def low_actor_loss(self, batch, grad_params, rng):
        """Compute the low-level flow BC actor loss."""
        observations, actions, low_actor_goals = batch['observations'], batch['actions'], batch['low_actor_goals']
        
        def flow_policy_loss():
            x_rng, t_rng = jax.random.split(rng, 2)

            batch_size, action_dim = actions.shape
            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = actions
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select('low_actor_flow')(observations, goals=low_actor_goals, actions=x_t, times=t, params=grad_params)
            return jnp.mean((pred - y) ** 2)

        actor_loss = flow_policy_loss()
        actor_info = {
            'low_actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 4)

        batch = dict(batch)
        g_value_loss, g_value_info = self.g_value_loss(batch, grad_params)
        for k, v in g_value_info.items():
            info[f'g_value/{k}'] = v

        critic_loss, v_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = g_value_loss + critic_loss + v_loss + high_actor_loss + low_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary.

        Logs per-module gradient norms. With ``encoder=None`` the five losses touch
        disjoint parameters; a shared visual encoder is the exception.
        """
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        grads, info = jax.grad(loss_fn, has_aux=True)(self.network.params)

        leaf_norms = jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.linalg.norm, grads))
        info.update(
            {
                'grad/max': jnp.max(
                    jnp.stack([jnp.max(g) for g in jax.tree_util.tree_leaves(grads)])
                ),
                'grad/min': jnp.min(
                    jnp.stack([jnp.min(g) for g in jax.tree_util.tree_leaves(grads)])
                ),
                'grad/norm': jnp.linalg.norm(jnp.stack(leaf_norms), ord=1),
                'grad/global_norm': optax.global_norm(grads),
            }
        )
        for module_name, subtree in grads.items():
            info[f'grad_norm/{module_name}'] = optax.global_norm(subtree)

        new_network = self.network.apply_gradients(grads=grads)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info
    
    @functools.partial(jax.jit, static_argnames=['is_encoded'])
    def sample_high_actions(
        self,
        observations,
        seed=None,
        is_encoded=False,
    ):
        """Sample a latent subgoal and horizon with flow matching.

        Set `is_encoded` when `observations` is already a latent vector.
        """
        latent_dim = self.config['latent_dim']
        batch_size = observations.shape[0]
        goals = jax.random.normal(seed, (batch_size, latent_dim + 1))
        for i in range(self.config['flow_steps']):
            t = jnp.full((batch_size, 1), i / self.config['flow_steps'])
            vels = self.network.select('high_actor_flow')(observations, actions=goals, times=t, is_encoded=is_encoded)
            goals = goals + vels / self.config['flow_steps']
            
        goals, ks = goals[..., :-1], goals[..., -1]
        return goals, ks
      
    @functools.partial(jax.jit, static_argnames=['obs_encoded', 'goal_encoded'])
    def sample_low_actions_train(
        self,
        observations,
        goals=None,
        seed=None,
        obs_encoded=False,
        goal_encoded=False,
    ):
        """Sample low-level actions with flow matching.

        `obs_encoded` and `goal_encoded` skip the corresponding encoder.
        """
        batch_size = observations.shape[0]
        actions = jax.random.normal(seed, (batch_size, self.config['action_dim']))
        is_encoded = obs_encoded
        for i in range(self.config['flow_steps']):
            t = jnp.full((batch_size, 1), i / self.config['flow_steps'])
            vels = self.network.select('low_actor_flow')(
                observations, goals=goals, actions=actions, times=t,
                is_encoded=is_encoded, goal_encoded=goal_encoded
            )
            actions = actions + vels / self.config['flow_steps']
            
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
    ):
        """Sample an action. A single observation is batched internally and squeezed on return."""
        single_input = (
            observations.ndim == 1
            or (self.config.get('encoder', None) is not None and observations.ndim == 3)
        )
        if single_input:
            observations = observations[None, ...]

        high_seed, low_seed = jax.random.split(seed)
        subgoals, _ = self.sample_high_actions(observations, seed=high_seed)
        
        batch_size = observations.shape[0]
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_goals = jnp.repeat(jnp.expand_dims(subgoals, 0), self.config['num_samples'], axis=0)
        n_actions = jax.random.normal(
            low_seed, (self.config['num_samples'], batch_size, self.config['action_dim'])
        )
        for i in range(self.config['flow_steps']):
            t = jnp.full((self.config['num_samples'], batch_size, 1), i / self.config['flow_steps'])
            vels = self.network.select('low_actor_flow')(n_observations, goals=n_goals, actions=n_actions, times=t, goal_encoded=True)
            n_actions = n_actions + vels / self.config['flow_steps']
            
        q = self.network.select('critic')(n_observations, actions=n_actions).min(axis=0)
        actions = n_actions[jnp.argmax(q)]
        
        actions = jnp.clip(actions, -1, 1)
        if single_input:
            actions = actions[0]

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations, ex_actions, ex_goals = example_batch['observations'], example_batch['actions'], example_batch['goal_k']
        ex_k_feat = example_batch['k'][:, None]
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        gc_encoder, obs_encoder = None, None
        state_encoder_def = None
        if config.get('encoder', None) is not None:
            encoder_def = encoder_modules[config['encoder']](
                layer_norm=config.get('encoder_layer_norm', False),
            )
            gc_encoder = GCEncoder(state_encoder=encoder_def, goal_encoder=encoder_def)
            obs_encoder = GCEncoder(state_encoder=encoder_def)
            state_encoder_def = GCEncoder(state_encoder=encoder_def)

            _enc_vars = state_encoder_def.init(jax.random.PRNGKey(42), ex_observations)
            latent_dim = state_encoder_def.apply(_enc_vars, ex_observations).shape[-1]
        else:
            latent_dim = ex_goals.shape[-1]

        # Zeros only fix the flow network's input width at init.
        ex_goals_latent = jnp.zeros((ex_goals.shape[0], latent_dim))
        ex_goals_with_k = jnp.concatenate([ex_goals_latent, ex_k_feat], axis=-1)

        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            gc_encoder=obs_encoder,
        )
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            gc_encoder=obs_encoder,
        )
        g_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            gc_encoder=gc_encoder,
        )

        high_actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=latent_dim + 1,
            layer_norm=config['layer_norm'],
            gc_encoder=obs_encoder,
        )
        
        low_actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            gc_encoder=gc_encoder,
        )

        network_info = dict(
            g_value=(g_value_def, (ex_observations, ex_goals, ex_k_feat)),
            critic=(critic_def, {'observations': ex_observations, 'actions': ex_actions}),
            target_critic=(copy.deepcopy(critic_def), {'observations': ex_observations, 'actions': ex_actions}),
            value=(value_def, (ex_observations,)),
            high_actor_flow=(high_actor_flow_def, (ex_observations, None, ex_goals_with_k, ex_times)),
            low_actor_flow=(low_actor_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
        )

        if state_encoder_def is not None:
            network_info['state_encoder'] = (state_encoder_def, (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(optax.clip_by_global_norm(config['grad_clip']),optax.adam(learning_rate=config['lr']))

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['action_dim'], config['goal_dim'], config['latent_dim'] = action_dim, latent_dim, latent_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='vast',
            action_dim=ml_collections.config_dict.placeholder(int),
            goal_dim=ml_collections.config_dict.placeholder(int),
            latent_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=512,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            q_agg='mean',  # Critic bootstrap aggregation: 'min' or 'mean'.
            expectile=0.9,
            # Multiplies rewards and k-step returns. Use `1 - discount` to map OGBench
            # returns onto [-1, 0], then retune `scale` by 1 / reward_scale.
            reward_scale=1.0,
            scale=1.0,  # Advantage temperature in the high-level AWR weights.
            flow_steps=10,
            num_samples=32,  # Best-of-n samples at action selection.
            comp_coef=0.5,  # Weight on the compositional G loss.
            dataset_class='HGMCDataset',
            max_horizon=25,
            grad_clip=1.0,
            low_actor_p_curgoal=0.1,
            low_actor_p_trajgoal=0.8,
            low_actor_p_randomgoal=0.1,
            low_actor_geom_sample=False,
            encoder=ml_collections.config_dict.placeholder(str),  # e.g. 'impala'; None for states.
            encoder_layer_norm=False,
        )
    )
    return config
