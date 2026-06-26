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

## To Do List

The code will be released gradually.
- [x] arxiv & project page released
- [x] initial repo