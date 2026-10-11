import dataclasses
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


@dataclasses.dataclass
class GCDataset:
    """Trajectory boundaries and goal-index sampling used by `HGMCDataset`."""

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        assert self.terminal_locs[-1] == self.size - 1

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config['discount']

        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Same trajectory, excluding the current state unless it is already terminal.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            offsets = np.random.geometric(p=1 - discount, size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            denom = p_trajgoal + p_randomgoal
            traj_prob = p_trajgoal / denom if denom > 0 else 0.0
            goal_idxs = np.where(np.random.rand(batch_size) < traj_prob, traj_goal_idxs, random_goal_idxs)
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def get_observations(self, idxs):
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])

    def get_actions(self, idxs):
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['actions'])

@dataclasses.dataclass
class HGMCDataset(GCDataset):
    """Samples variable-horizon Monte-Carlo returns for hierarchical training."""
    def discounted_return(self, rewards, discount):
        discounts = discount ** np.arange(rewards.shape[1])
        discounted_returns = rewards * discounts
        return np.sum(discounted_returns, axis=1)

    def sample(self, batch_size: int, idxs=None):
        idxs = self.dataset.get_random_idxs(batch_size) if idxs is None else idxs
        batch = self.dataset.sample(batch_size, idxs)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        has_oracle = 'oracle_reps' in self.dataset

        low_actor_goal_idxs = self.sample_goals(
            idxs, 
            self.config['low_actor_p_curgoal'], 
            self.config['low_actor_p_trajgoal'], 
            self.config['low_actor_p_randomgoal'], 
            self.config['low_actor_geom_sample']
        )
        
        if has_oracle:
            batch['low_actor_goals'] = self.dataset['oracle_reps'][low_actor_goal_idxs]
        else:
            batch['low_actor_goals'] = self.get_observations(low_actor_goal_idxs)
        
        max_horizon = np.minimum(self.config['max_horizon'], final_state_idxs - idxs)
        k_step = np.where(max_horizon >= 1, np.floor(np.random.rand(batch_size) * max_horizon).astype(np.int32) + 1, 1)
        j_step = np.where(k_step >= 2, np.floor(np.random.rand(batch_size) * np.maximum(k_step - 1, 1)).astype(np.int32) + 1, 0)
    
        goal_k_idxs = np.minimum(idxs + k_step, final_state_idxs)
        goal_j_idxs = np.minimum(idxs + j_step, goal_k_idxs)
        # MC target is valid for any k >= 1. False only when s_t is already terminal.
        valid_mc = (final_state_idxs > idxs).astype(np.float32)
        # Compositional identity needs k > j >= 1.
        valid_comp = ((k_step >= 2) & (j_step >= 1)).astype(np.float32)
        
        rewards = np.array(self.dataset['rewards'])
        reward_idx_matrix = idxs[:, None] + np.arange(self.config['max_horizon'])[None, :]
        valid_condition = (reward_idx_matrix <= final_state_idxs[:, None]) & \
                            (reward_idx_matrix < idxs[:, None] + k_step[:, None]) & \
                                (reward_idx_matrix < rewards.shape[0])
        valid_indices = reward_idx_matrix[valid_condition]
            
        k_horizon_reward_sequence = np.zeros_like(reward_idx_matrix, dtype=np.float32)
        k_horizon_reward_sequence[valid_condition] = rewards[valid_indices]
        k_horizon_discounted_returns = self.discounted_return(k_horizon_reward_sequence, self.config['discount']).astype(np.float32)
        
        batch['k_discounted_returns'] = k_horizon_discounted_returns
        batch['goal_k'], batch['k'] = self.get_observations(goal_k_idxs), k_step
        batch['actions_k'] = self.get_actions(goal_k_idxs)
        batch['goal_j'], batch['j'] = self.get_observations(goal_j_idxs), j_step
        batch['mc_mask'] = valid_mc
        batch['comp_mask'] = valid_comp

        return batch
