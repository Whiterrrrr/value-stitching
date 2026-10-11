from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm, trange


def supply_rng(f, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def _info_for_env(vector_infos, env_idx):
    """Pull one env's info out of a gymnasium vector `infos` mapping."""
    out = {}
    for key, value in vector_infos.items():
        if key.startswith('_'):
            continue
        mask = vector_infos.get(f'_{key}')
        if mask is not None and not bool(mask[env_idx]):
            continue
        if isinstance(value, dict):
            nested = _info_for_env(value, env_idx)
            if nested:
                out[key] = nested
        else:
            out[key] = value[env_idx]
    return out


def _make_vmapped_actor(agent):
    """vmap single-observation `sample_actions`.

    A batched call shares one best-of-n index across envs and changes the policy.
    """
    sample = agent.sample_actions

    def _one(observation, seed):
        return sample(observations=observation, seed=seed)

    return jax.jit(jax.vmap(_one))


def _split_seeds(rng, n):
    rng, *keys = jax.random.split(rng, n + 1)
    return rng, jnp.stack(keys)


def _evaluate_wave(
    batched_actor_fn,
    vec_env,
    wave_size,
    eval_gaussian,
    rng,
    dummy_action,
    desc,
):
    """One wave: up to `wave_size` workers, each completing exactly one episode."""
    stats = defaultdict(list)
    num_envs = vec_env.num_envs
    wave_seed = int(np.random.randint(0, 2**31 - 1))
    obs, _ = vec_env.reset(seed=wave_seed)
    alive = np.zeros(num_envs, dtype=bool)
    alive[:wave_size] = True
    pbar = tqdm(desc=desc, leave=False)

    while alive.any():
        # Finished workers get a dummy action; their transitions are ignored.
        actions = np.broadcast_to(dummy_action, (num_envs,) + dummy_action.shape).copy()
        rng, seeds = _split_seeds(rng, num_envs)
        sampled = np.asarray(jax.device_get(batched_actor_fn(obs, seeds)))
        actions[alive] = sampled[alive]

        if eval_gaussian is not None:
            noise_mask = alive.reshape((-1,) + (1,) * (actions.ndim - 1))
            actions = np.where(
                noise_mask,
                np.random.normal(actions, eval_gaussian),
                actions,
            )
        actions = np.clip(actions, -1, 1)

        obs, _, terminated, truncated, infos = vec_env.step(actions)
        finished = alive & (terminated | truncated)
        for i in np.flatnonzero(finished):
            add_to(stats, flatten(_info_for_env(infos, int(i))))
            alive[i] = False
        pbar.update(1)
        pbar.set_postfix(left=int(alive.sum()))
    pbar.close()
    return stats, rng


def _evaluate_parallel(
    batched_actor_fn,
    vec_env,
    num_eval_episodes,
    eval_gaussian,
    rng,
):
    """Roll out episodes on a vector env. Extra episodes run in later waves."""
    stats = defaultdict(list)
    num_envs = vec_env.num_envs
    if num_envs <= 0:
        raise ValueError(f'vec_env.num_envs must be positive, got {num_envs}')
    dummy_action = np.zeros(vec_env.single_action_space.shape, dtype=np.float32)

    completed = 0
    while completed < num_eval_episodes:
        wave_size = min(num_envs, num_eval_episodes - completed)
        wave_stats, rng = _evaluate_wave(
            batched_actor_fn,
            vec_env,
            wave_size,
            eval_gaussian,
            rng,
            dummy_action,
            desc=f'eval {completed + 1}-{completed + wave_size}/{num_eval_episodes} x{wave_size}',
        )
        for key, values in wave_stats.items():
            stats[key].extend(values)
        completed += wave_size

    return stats, []


def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_gaussian=None,
    vec_env=None,
):
    """Evaluate the agent.

    Video episodes are rendered and excluded from the statistics. When `vec_env`
    is set, metric episodes step in parallel; video episodes stay sequential.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    use_vec = vec_env is not None and num_eval_episodes > 0
    if use_vec:
        batched_actor_fn = _make_vmapped_actor(agent)
        actor_rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
        stats, trajs = _evaluate_parallel(
            batched_actor_fn,
            vec_env,
            num_eval_episodes,
            eval_gaussian,
            actor_rng,
        )
        video_start = 0
        sequential_episodes = num_video_episodes
    else:
        video_start = num_eval_episodes
        sequential_episodes = num_eval_episodes + num_video_episodes

    for i in trange(sequential_episodes):
        traj = defaultdict(list)
        should_render = i >= video_start

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation)
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < video_start:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
