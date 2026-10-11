import glob
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from envs.eval_env import make_vec_eval_env
from utils.datasets import HGMCDataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('agent_name', 'Debug', 'log wandb')
flags.DEFINE_string('project_name', 'VAST-1.0.1', 'log wandb')
flags.DEFINE_string('env_name', 'puzzle-4x5-play-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('dataset_replace_interval', 0, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets to use.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 2000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_integer('eval_num_envs', 25, 'Max parallel eval workers. Episodes beyond this run in later waves. 0 uses eval_episodes workers.',)
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
config_flags.DEFINE_config_file('agent', 'agents/vast.py', lock_config=False)


def main(_):
    exp_name = FLAGS.agent_name
    setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    if FLAGS.dataset_dir is None:
        datasets = [None]
    else:
        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
    if FLAGS.num_datasets is not None:
        datasets = datasets[: FLAGS.num_datasets]
    dataset_idx = 0
    goal_conditioned = 'singletask' not in FLAGS.env_name
    
    env, eval_env, train_dataset, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx])

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_class = {'HGMCDataset': HGMCDataset}[config['dataset_class']]
    train_dataset = dataset_class(train_dataset, config)
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    first_time = time.time()
    last_time = time.time()

    vec_eval_env = None
    # At most `eval_num_envs` workers, one episode each per wave. Extra episodes start a new wave.
    n_eval_envs = FLAGS.eval_episodes if FLAGS.eval_num_envs <= 0 else min(FLAGS.eval_num_envs, FLAGS.eval_episodes)
    if n_eval_envs > 1:
        n_waves = (FLAGS.eval_episodes + n_eval_envs - 1) // n_eval_envs
        print(
            f'Creating {n_eval_envs} parallel eval envs '
            f'({FLAGS.eval_episodes} episodes, {n_waves} wave{"s" if n_waves > 1 else ""})...',
            flush=True,
        )
        vec_eval_env = make_vec_eval_env(FLAGS.env_name, n_eval_envs)

    try:
        for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
                train_metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                wandb.log(train_metrics, step=i)

            if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                task_infos = eval_env.unwrapped.task_infos if hasattr(eval_env.unwrapped, 'task_infos') else eval_env.task_infos

                if not goal_conditioned:
                    task_id = eval_env.spec.kwargs['reward_task_id']
                    task_name = task_infos[task_id - 1]['task_name']

                    eval_info, _, cur_renders = evaluate(
                        agent=agent,
                        env=eval_env,
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                        eval_gaussian=FLAGS.eval_gaussian,
                        vec_env=vec_eval_env,
                    )
                    renders.extend(cur_renders)
                    metric_names = ['episode.return', 'episode.length', 'success']
                    eval_metrics.update(
                        {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)
                else:
                    raise NotImplementedError('Goal-conditioned evaluation is not implemented yet.')

                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                if FLAGS.video_episodes > 0:
                    video = get_wandb_video(renders=renders, n_cols=3)
                    eval_metrics['video'] = video

                wandb.log(eval_metrics, step=i)

            if i % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, i)

            if FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0 and len(datasets) > 1:
                dataset_idx = (dataset_idx + 1) % len(datasets)
                train_dataset, _ = make_env_and_datasets(
                    FLAGS.env_name, dataset_path=datasets[dataset_idx], dataset_only=True,
                )
                train_dataset = dataset_class(train_dataset, config)
    finally:
        if vec_eval_env is not None:
            vec_eval_env.close()


if __name__ == '__main__':
    app.run(main)
