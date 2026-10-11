from pathlib import Path

import numpy as np
import ogbench

from envs.episode_monitor import EpisodeMonitor
from utils.datasets import Dataset


def make_env_and_datasets(dataset_name, dataset_path, dataset_only=False):
    """Make an OGBench environment and its train/validation datasets."""
    dataset_dir = str(Path(dataset_path).parent)
    if dataset_only:
        _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_dir=dataset_dir, compact_dataset=True
        )
    else:
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_dir=dataset_dir, compact_dataset=True
        )
        eval_env = ogbench.make_env_and_datasets(dataset_name, env_only=True)

        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    # Keep actions strictly inside (-1, 1).
    eps = 1e-5
    train_dataset = train_dataset.copy(
        add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps))
    )
    val_dataset = val_dataset.copy(add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + eps, 1 - eps)))

    if dataset_only:
        return train_dataset, val_dataset
    else:
        env.reset()
        eval_env.reset()
        return env, eval_env, train_dataset, val_dataset
