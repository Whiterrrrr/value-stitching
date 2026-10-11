"""Roll out a trained agent from a checkpoint, print stats, and save a video per episode.

Example:
    python utils/test.py \
        --model_path=exp/.../params_1000000.pkl \
        --env_name=scene-play-singletask-task1-v0 \
        --dataset_dir=/path/to/ogbench/data \
        --num_episodes=10 \
        --video_path=videos/eval.mp4
"""

import glob
import os
import pickle
import random
import re
from collections import defaultdict

import flax
import imageio
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from tqdm import trange

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import HGMCDataset
from utils.evaluation import flatten, supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None, 'Path to the checkpoint .pkl file.', required=True)
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to run.')
flags.DEFINE_string('video_path', 'videos/eval.mp4', 'Output path for the rendered video.')
flags.DEFINE_string('env_name', 'scene-play-singletask-task1-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory (used to build the example batch).')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('video_fps', 15, 'FPS of the saved video.')
config_flags.DEFINE_config_file('agent', 'agents/vast.py', lock_config=False)


def build_agent(config, env_name, dataset_dir, seed):
    """Create the agent (and eval env) with the same setup as training."""
    datasets = [f for f in sorted(glob.glob(f'{dataset_dir}/*.npz')) if '-val.npz' not in f]
    assert len(datasets) > 0, f'No dataset found in {dataset_dir}'

    _, eval_env, train_dataset, _ = make_env_and_datasets(env_name, dataset_path=datasets[0])

    dataset_class = {'HGMCDataset': HGMCDataset}[config['dataset_class']]
    train_dataset = dataset_class(train_dataset, config)
    example_batch = train_dataset.sample(1)

    agent = agents[config['agent_name']].create(seed, example_batch, config)
    return agent, eval_env


def load_checkpoint(agent, model_path):
    """Load agent parameters from a .pkl checkpoint file."""
    with open(model_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'Restored from {model_path}')
    return agent


def run_episodes(agent, env, num_episodes, video_frame_skip):
    """Roll out episodes, collecting per-episode stats and rendered frames."""
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    stats = defaultdict(list)
    episode_frames = []
    for _ in trange(num_episodes, desc='Episodes'):
        observation, info = env.reset()
        done = False
        step = 0
        frames = []
        while not done:
            action = np.array(actor_fn(observations=observation))
            action = np.clip(action, -1, 1)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if step % video_frame_skip == 0 or done:
                frames.append(env.render().copy())

        for k, v in flatten(info).items():
            stats[k].append(v)
        episode_frames.append(frames)

    return dict(stats), episode_frames


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    agent, eval_env = build_agent(FLAGS.agent, FLAGS.env_name, FLAGS.dataset_dir, FLAGS.seed)
    agent = load_checkpoint(agent, FLAGS.model_path)

    stats, episode_frames = run_episodes(
        agent, eval_env, FLAGS.num_episodes, FLAGS.video_frame_skip
    )

    print('===== Evaluation results =====')
    for k in ['success', 'episode.return', 'episode.length']:
        if k in stats:
            print(f'{k}: {np.mean(stats[k]):.4f}')

    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.video_path)), exist_ok=True)
    root, ext = os.path.splitext(FLAGS.video_path)
    env_suffix = re.sub(r'[^A-Za-z0-9_-]+', '-', FLAGS.env_name).strip('-')
    task_match = re.search(r'(task\d+)', FLAGS.env_name)
    task_suffix = task_match.group(1) if task_match is not None else 'task-unknown'
    returns = stats.get('episode.return', [None] * len(episode_frames))
    for i, frames in enumerate(episode_frames):
        suffix = f'_ret{returns[i]:.2f}' if returns[i] is not None else ''
        episode_path = f'{root}_ep{i:03d}{suffix}_{env_suffix}_{task_suffix}{ext}'
        imageio.mimsave(episode_path, frames, fps=FLAGS.video_fps)
        print(f'Saved video to {episode_path}')


if __name__ == '__main__':
    app.run(main)
