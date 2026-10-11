"""JAX-free eval env constructors.

AsyncVectorEnv workers are spawned into a fresh interpreter. This module must
not import JAX (or anything that initializes a GPU) so the rollout
processes only load MuJoCo / OGBench.
"""

from functools import partial

import ogbench
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from envs.episode_monitor import EpisodeMonitor


def make_eval_env(env_name):
    """Create one OGBench eval env. Safe to call inside a spawned worker."""
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    return EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])


def make_vec_eval_env(env_name, num_envs):
    """Create `num_envs` parallel eval workers via spawn (CUDA-parent safe)."""
    if num_envs <= 0:
        raise ValueError(f'num_envs must be positive, got {num_envs}')
    env_fns = [partial(make_eval_env, env_name) for _ in range(num_envs)]
    # Disabled so a finished worker does not start a second episode mid-wave.
    return AsyncVectorEnv(
        env_fns,
        shared_memory=True,
        context='spawn',
        autoreset_mode=AutoresetMode.DISABLED,
    )
