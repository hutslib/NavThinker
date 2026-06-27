<div align="center">
  <h1>NavThinker: Action-Conditioned World Models for<br>Coupled Prediction and Planning in Social Navigation</h1>
  <h4>
    <a href="https://hutslib.github.io/">Tianshuai Hu</a><sup>1</sup>,
    <a href="https://zeying-gong.github.io/">Zeying Gong</a><sup>2</sup>,
    <a href="https://ldkong.com/">Lingdong Kong</a><sup>3</sup>,
    <a href="https://xmei-hk.github.io/">XiaoDong Mei</a><sup>1</sup>,
    Yiyi Ding<sup>2</sup>,
    <br>
    Qi Zeng<sup>2</sup>,
    <a href="https://alanliangc.github.io/">Ao Liang</a><sup>4</sup>,
    <a href="https://rongli.tech/">Rong Li</a><sup>2</sup>,
    Yangyi Zhong<sup>2</sup>,
    <a href="https://junweiliang.me/">Junwei Liang</a><sup>1,2,&#8224;</sup>
  </h4>

  <p>
    <sup>1</sup>HKUST &nbsp;&nbsp;
    <sup>2</sup>HKUST (Guangzhou) &nbsp;&nbsp;
    <sup>3</sup>NUS &nbsp;&nbsp;
    <sup>4</sup>UCAS
    <br>
    <sup>&#8224;</sup>Corresponding author
  </p>

  <p>
    <a href="https://hutslib.github.io/NavThinker/">Project Page</a> |
    <a href="https://arxiv.org/abs/2603.15359">Paper (arXiv)</a>
  </p>

  <!-- Badges -->
  <p>
    <a href="https://hutslib.github.io/NavThinker/">
      <img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page Badge">
    </a>
    <a href="https://arxiv.org/abs/2603.15359">
      <img src="https://img.shields.io/badge/cs.RO-arXiv:2603.15359-b31b1b.svg" alt="arXiv Paper Badge">
    </a>
    <a href="https://github.com/facebookresearch/habitat-sim">
      <img src="https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational" alt="Habitat Sim Badge">
    </a>
    <a href="https://github.com/hutslib/socialnav-wm/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License Badge">
    </a>
  </p>
</div>

> ### 🚧 Code is still under cleanup
> This repository is being prepared for open-source release. The code is **not yet
> complete** — APIs, configs, and file layout may change. See the [release checklist](#release-checklist) below for status.

## :sparkles: Overview

To move safely and efficiently through crowds, a robot should not just react to what it
sees now — it should **think ahead** about how the scene and the people in it will evolve
*under the actions it is considering*. In social navigation, the robot's motion and human
motion are mutually coupled: where the robot goes changes how people move, and vice versa.

**NavThinker** tackles this with an **action-conditioned world model** that is tightly
coupled with a reinforcement-learning policy. The world model operates in the **patch
feature space of a frozen [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
encoder** and autoregressively predicts the future, with multi-head decoders that
reconstruct **future depth maps**, **human trajectories**, and **reward**. These
action-conditioned future features are fused back into the policy's observation embedding
and turned into **social reward shaping**, so the policy learns to weigh how each candidate
action would reshape the future before committing to it.

The policy is trained with **DD-PPO** on real environment interactions and stays
**image-based** throughout — the world model acts as an on-line *advisor* (foresight via
feature fusion and reward shaping) rather than replacing real experience. This keeps the
robustness of model-free training while injecting model-based lookahead.

We evaluate on the **[Social-HM3D & Social-MP3D](#3-downloading-the-social-hm3d--social-mp3d-datasets)**
SocialNav benchmark — large-scale photo-realistic indoor scenes populated with humans
following natural movement patterns, with
zero-shot transfer to Social-MP3D.

### Highlights

- **Unified scene-and-interaction world model.** Future depth (scene dynamics) and human
  trajectories (interaction) are predicted from the *same* patch-level latent state, sharing
  one dynamics model so the two predictions reinforce each other.
- **Coupled prediction & planning.** The world model influences the policy through two
  channels: **feature fusion** of predictive features and **action-conditioned imagination
  reward shaping** that converts predicted human motion into social-safety penalties.


## Release Checklist

This is a staged open-source release. Current status:

**Done**
- [x] README
- [x] Policy **training + evaluation** code (NavThinker, with the coupled online world model)
- [x] Clean `configs/` and launcher `scripts/` for Social-HM3D

**To do**
- [ ] Full code cleanup, comments, and API stabilization
- [ ] Pretrained checkpoints (policy + world model)
- [ ] End-to-end verified install & dataset/episode download guide
- [ ] Baseline configs/scripts (A*, ORCA, Habita-official, Falcon)


## :hammer_and_wrench: Installation

This codebase is built on top of [Habitat-Lab / Habitat-Sim 3.0](https://github.com/facebookresearch/habitat-lab)
and the [Falcon](https://github.com/Zeying-Gong/Falcon) social-navigation framework.

### 1. Prepare the conda environment

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed:

```bash
conda_env_name=navthinker
conda create -n $conda_env_name python=3.9 cmake=3.14.0
conda activate $conda_env_name
```

### 2. Install habitat-sim & habitat-lab

Following [Habitat-Lab](https://github.com/facebookresearch/habitat-lab.git)'s instructions:

```bash
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat
```

If you hit network problems, you can manually download the conda package from
[this link](https://anaconda.org/aihabitat/habitat-sim/0.3.1/download/linux-64/habitat-sim-0.3.1-py3.9_headless_bullet_linux_3d6d67d6deae4ab2472cc84df7a3cef1503f606d.tar.bz2)
and install it via `conda install --use-local /path/to/xxx.tar.bz2`.

Then clone this repository and install the bundled Habitat packages and dependencies:

```bash
git clone https://github.com/hutslib/socialnav-wm.git
cd socialnav-wm
pip install -e habitat-lab
pip install -e habitat-baselines
pip install -r requirements.txt
```

The frozen Depth Anything V2 encoder weights are also required for the world model — see the
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) repository.

### 3. Downloading the Social-HM3D & Social-MP3D datasets

- **Scene datasets.** Follow the instructions for **HM3D** and **MatterPort3D** in Habitat-Lab's
  [DATASETS.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).

- **Episode datasets.** Download the SocialNav episodes for the test scenes from
  [this link](https://drive.google.com/drive/folders/1V0a8PYeMZimFcHgoJGMMTkvscLhZeKzD?usp=drive_link),
  unzip them, and place them under the default location:

  ```bash
  unzip <episodes>.zip -d data/datasets/pointnav
  ```

- **Leg animation data.**

  ```bash
  wget https://github.com/facebookresearch/habitat-lab/files/12502177/spot_walking_trajectory.csv \
    -O data/robots/spot_data/spot_walking_trajectory.csv
  ```

- **Multi-agent assets.**

  ```bash
  python -m habitat_sim.utils.datasets_download \
    --uids hab3-episodes habitat_humanoids hab3_bench_assets hab_spot_arm
  ```

The resulting file structure should look like this:

```
socialnav-wm/
└── data/
    ├── datasets/
    │   └── pointnav/
    │       ├── social-hm3d/{train,val}/{content,*.json.gz}
    │       └── social-mp3d/{train,val}/{content,*.json.gz}
    ├── scene_datasets/
    ├── robots/
    ├── humanoids/
    ├── versioned_data/
    └── hab3_bench_assets/
```

> **Note:** the definition of SocialNav here differs from the original task in
> [Habitat 3.0](https://arxiv.org/abs/2310.13724); it follows the Social-HM3D / Social-MP3D
> benchmark introduced in Falcon.


## :rocket: Training

Experiment configs live in [`configs/`](configs) and the launcher scripts in
[`scripts/`](scripts). Train the NavThinker policy with DD-PPO:

```bash
# bash scripts/train.sh <config_name> <num_gpus>
bash scripts/train.sh navthinker_hm3d.yaml 4
```

Re-running resumes automatically and writes logs/checkpoints under
`experiments/<config_name>/`.

The world model is **co-trained online with the policy**: the policy is optimized on real
interactions with DD-PPO, while the world model is optimized on a replay buffer by a separate
optimizer and feeds detached foresight back to the policy via feature fusion and social reward
shaping. (The standalone offline world-model training pipeline is not part of this release.)

## :arrow_forward: Evaluation

Evaluate a trained checkpoint on the val split (writes videos/TensorBoard under
`eval_experiments/navthinker/`):

```bash
# bash scripts/eval.sh <config_name> <checkpoint.pth> [num_envs]
bash scripts/eval.sh navthinker_hm3d_eval.yaml experiments/navthinker/checkpoints/ckpt.100.pth 1
```

For zero-shot transfer, switch the benchmark task in
[`configs/navthinker_hm3d_eval.yaml`](configs/navthinker_hm3d_eval.yaml) to the Social-MP3D
task and evaluate the same Social-HM3D-trained checkpoint.

### Baselines

For reference, the benchmark also includes two classic rule-based planners and a learning-based baseline:

- **[A*](https://ieeexplore.ieee.org/document/4082128)** — shortest-path planning with a heuristic.
- **[ORCA](https://gamma.cs.unc.edu/ORCA/publications/ORCA.pdf)** — reciprocal collision-free multi-agent navigation.

```bash
python -u -m habitat_baselines.run --config-name=social_nav_v2/astar_hm3d.yaml
python -u -m habitat_baselines.run --config-name=social_nav_v2/orca_hm3d.yaml
```


## :black_nib: Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@article{hu2026navthinker,
  title={NavThinker: Action-Conditioned World Models for Coupled Prediction and Planning in Social Navigation},
  author={Hu, Tianshuai and Gong, Zeying and Kong, Lingdong and Mei, XiaoDong and Ding, Yiyi and Zeng, Qi and Liang, Ao and Li, Rong and Zhong, Yangyi and Liang, Junwei},
  journal={arXiv preprint arXiv:2603.15359},
  year={2026}
}
```

This work builds on our earlier social-navigation framework, Falcon:

```bibtex
@article{gong2024cognition,
  title={From Cognition to Precognition: A Future-Aware Framework for Social Navigation},
  author={Gong, Zeying and Hu, Tianshuai and Qiu, Ronghe and Liang, Junwei},
  journal={arXiv preprint arXiv:2409.13244},
  year={2024}
}
```


## :pray: Acknowledgments

We thank the following projects, on which this work builds:

- [Falcon](https://github.com/Zeying-Gong/Falcon) — social-navigation framework and Social-HM3D / Social-MP3D benchmark
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) & [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [DreamerV3](https://github.com/danijar/dreamerv3) and [DINO-WM](https://github.com/gaoyuezhou/dino_wm) — world-model design inspiration


## :scroll: License

This project is released under the [MIT License](LICENSE).
