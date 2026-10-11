<div align="center">

# [Horizon Adaptive Offline Policy Learning via <br>Value Stitching](https://arxiv.org/abs/2606.21136)
## [Paper](https://arxiv.org/abs/2606.21136) &emsp;[Project Page](https://whiterrrrr.github.io/value-stitching/)
<img src="assets/value-stitching.png" width="100%">
</div>

This repository contains the official implementation of ***VAST*** (*Horizon Adaptive Offline Policy Learning via <u>**VA**</u>lue <u>**ST**</u>itching*) designed for long-horizon, complex offline RL tasks.

## Overview

Traditional TD-based value learning relies on fixed-step backups, however, often fail to capture the complex temporal structure of long-horizon, multi-stage tasks. *VAST* overcomes this limitation by coupling value optimization with
- a future-conditioned **auxiliary value function**,
- a **stitching policy** that optimally selects the reward maximizing future.

*VAST* enables direct estimation and compositional *"stitching"* of variable-length returns grounded in actionable sub-goal states, providing an accurate and greedily exploitable value-supervision signal for offline policy optimization.

## Setup

### Requirements

The pinned stack in `requirements.txt` includes JAX 0.4.28 with the CUDA 12 plugins, MuJoCo, OGBench, and Weights & Biases. Install it with:

```bash
pip install -r requirements.txt
```

### Running Experiments

The main entry point is  `main.py`. Example runing `cube-double-play`:

```bash
python main.py --seed=0 \
    --env_name=cube-double-play-singletask-task1-v0 \
    --run_group=cube-double-play \
    --dataset_dir=/path/to/ogbench/data \
    --agent_name=vast-reproducing \
    --project_name=VAST \
    --agent=agents/vast.py \
    --agent.discount=0.99 \
    --agent.comp_coef=0.5 \
    --agent.max_horizon=50 \
    --agent.scale=0.08 \
    --agent.num_samples=4 \
    --agent.q_agg=mean \
    --agent.low_actor_p_curgoal=0.1 \
    --agent.low_actor_p_trajgoal=0.8 \
    --agent.low_actor_p_randomgoal=0.1
```
To reproduce the paper, use the following hyperparameters:

<details>
<summary><b>Hyperparameters by domain</b></summary>

| Domain | `K` | `β` | `N_rej` | `γ` | `λ` | Clip Q | Subgoal `[p_cur, p_traj, p_rand]` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `scene-play` | 50 | 0.03 | 4 | 0.995 | 0.25 | false | 0.1, 0.8, 0.1 |
| `cube-single-play` | 50 | 0.08 | 4 | 0.99 | 0.5 | false | 0.1, 0.8, 0.1 |
| `cube-double-play` | 50 | 0.08 | 4 | 0.99 | 0.5 | false | 0.1, 0.8, 0.1 |
| `puzzle-4x4-play` | 10 | 0.01 | 4 (task 2, 5); 16 (others) | 0.995 | 0.5 | false | 0.1, 0.8, 0.1 |
| `antmaze-large-navigate` | 50 | 0.05 | 4 | 0.995 | 0.5 | false | 0, 1, 0 |
| `antmaze-giant-navigate` | 50 | 0.05 | 4 | 0.995 | 0.5 | true | 0, 1, 0 |
| `cube-triple-play` | 25 | 0.001 | 4 | 0.999 | 0.5 | false | 0.1, 0.8, 0.1 |
| `cube-quadruple-play` | 10 | 0.001 | 16 | 0.999 | 0.1 | false | 0.1, 0.8, 0.1 |
| `puzzle-4x5-play` | 10 | 0.005 | 32 | 0.995 | 0.5 | false | 0.1, 0.8, 0.1 |
| `puzzle-4x6-play` | 10 | 0.005 | 32 | 0.995 | 0.5 | false | 0.1, 0.8, 0.1 |

</details>

## BibTeX

```bibtex
@inproceedings{zheng2026vast,
  title     = {Horizon Adaptive Offline Policy Learning via Value Stitching},
  author    = {Kexin Zheng and Xianyuan Zhan and Xintao Yan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgments
This codebase is built on top of [SHARSA](https://github.com/seohongpark/horizon-reduction). We thank the research community for all contributions from prior studies.