#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from habitat.config.default_structured_configs import SimulatorSensorConfig

cs = ConfigStore.instance()


@dataclass
class HabitatBaselinesBaseConfig:
    pass


@dataclass
class WBConfig(HabitatBaselinesBaseConfig):
    """Weights and Biases config"""

    # The name of the project on W&B.
    project_name: str = ""
    # Logging entity (like your username or team name)
    entity: str = ""
    # The group ID to assign to the run. Optional to specify.
    group: str = ""
    # The run name to assign to the run. If not specified,
    # W&B will randomly assign a name.
    run_name: str = ""


@dataclass
class EvalConfig(HabitatBaselinesBaseConfig):
    # The split to evaluate on
    split: str = "val"
    use_ckpt_config: bool = True
    should_load_ckpt: bool = True
    # The number of time to run each episode through evaluation.
    # Only works when evaluating on all episodes.
    evals_per_ep: int = 1
    video_option: List[str] = field(
        # available options are "disk" and "tensorboard"
        default_factory=list
    )
    save_images: bool = False
    image_dir: Optional[str] = None
    extra_sim_sensors: Dict[str, SimulatorSensorConfig] = field(
        default_factory=dict
    )


@dataclass
class PreemptionConfig(HabitatBaselinesBaseConfig):
    # Append the slurm job ID to the resume state filename if running
    # a slurm job. This is useful when you want to have things from a different
    # job but the same checkpoint dir not resume.
    append_slurm_job_id: bool = False
    # Number of gradient updates between saving the resume state
    save_resume_state_interval: int = 100
    # Save resume states only when running with slurm
    # This is nice if you don't want debug jobs to resume
    save_state_batch_only: bool = False


@dataclass
class ActionDistributionConfig(HabitatBaselinesBaseConfig):
    use_log_std: bool = True
    use_softplus: bool = False
    std_init: float = MISSING
    log_std_init: float = 0.0
    # If True, the std will be a parameter not conditioned on state
    use_std_param: bool = False
    # If True, the std will be clamped to the specified min and max std values
    clamp_std: bool = True
    min_std: float = 1e-6
    max_std: int = 1
    min_log_std: int = -5
    max_log_std: int = 2
    # For continuous action distributions (including gaussian):
    action_activation: str = "tanh"  # ['tanh', '']
    scheduled_std: bool = False


@dataclass
class ObsTransformConfig(HabitatBaselinesBaseConfig):
    type: str = MISSING


@dataclass
class CenterCropperConfig(ObsTransformConfig):
    type: str = "CenterCropper"
    height: int = 256
    width: int = 256
    channels_last: bool = True
    trans_keys: Tuple[str, ...] = (
        "rgb",
        "depth",
        "semantic",
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms",
    name="center_cropper_base",
    node=CenterCropperConfig,
)


@dataclass
class ResizeShortestEdgeConfig(ObsTransformConfig):
    type: str = "ResizeShortestEdge"
    size: int = 256
    channels_last: bool = True
    trans_keys: Tuple[str, ...] = (
        "rgb",
        "depth",
        "semantic",
    )
    semantic_key: str = "semantic"


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms",
    name="resize_shortest_edge_base",
    node=ResizeShortestEdgeConfig,
)


@dataclass
class Cube2EqConfig(ObsTransformConfig):
    type: str = "CubeMap2Equirect"
    height: int = 256
    width: int = 512
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms",
    name="cube_2_eq_base",
    node=Cube2EqConfig,
)


@dataclass
class Cube2FishConfig(ObsTransformConfig):
    type: str = "CubeMap2Fisheye"
    height: int = 256
    width: int = 256
    fov: int = 180
    params: Tuple[float, ...] = (0.2, 0.2, 0.2)
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms",
    name="cube_2_fish_base",
    node=Cube2FishConfig,
)


@dataclass
class AddVirtualKeysConfig(ObsTransformConfig):
    type: str = "AddVirtualKeys"
    virtual_keys: Dict[str, int] = field(default_factory=dict)


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms",
    name="add_virtual_keys_base",
    node=AddVirtualKeysConfig,
)


@dataclass
class Eq2CubeConfig(ObsTransformConfig):
    type: str = "Equirect2CubeMap"
    height: int = 256
    width: int = 256
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms",
    name="eq_2_cube_base",
    node=Eq2CubeConfig,
)


@dataclass
class HrlDefinedSkillConfig(HabitatBaselinesBaseConfig):
    """
    Defines a low-level skill to be used in the hierarchical policy.
    """

    skill_name: str = MISSING
    name: str = "PointNavResNetPolicy"
    action_distribution_type: str = "gaussian"
    load_ckpt_file: str = ""
    max_skill_steps: int = 200
    # If true, the stop action will be called if the skill times out.
    force_end_on_timeout: bool = True
    # Overrides the config file of a neural network skill rather than loading
    # the config file from the checkpoint file.
    force_config_file: str = ""
    at_resting_threshold: float = 0.15
    # If true, this will apply the post-conditions of the skill after it
    # terminates.
    apply_postconds: bool = False

    # If true, do not call grip_actions automatically when calling high level skills.
    # Do not check either if an arm action necessarily exists.
    ignore_grip: bool = False
    obs_skill_inputs: List[str] = field(default_factory=list)
    obs_skill_input_dim: int = 3
    start_zone_radius: float = 0.3
    # For the oracle navigation skill
    action_name: str = "base_velocity"
    stop_thresh: float = 0.001
    # For the reset_arm_skill
    reset_joint_state: List[float] = MISSING
    # The set of PDDL action names (as defined in the PDDL domain file) that
    # map to this skill. If not specified,the name of the skill must match the
    # PDDL action name.
    pddl_action_names: Optional[List[str]] = None
    turn_power_x: float = 0.0
    turn_power_y: float = 0.0
    # Additional skill data to be passed to the skill. Included so extending to
    # new skills doesn't require adding new Hydra dataclass configs.
    skill_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalPolicyConfig(HabitatBaselinesBaseConfig):
    high_level_policy: Dict[str, Any] = MISSING
    # Names of the skills to not load.
    ignore_skills: List[str] = field(default_factory=list)
    defined_skills: Dict[str, HrlDefinedSkillConfig] = field(
        default_factory=dict
    )
    use_skills: Dict[str, str] = field(default_factory=dict)


@dataclass
class PolicyConfig(HabitatBaselinesBaseConfig):
    name: str = "PointNavResNetPolicy"
    action_distribution_type: str = "categorical"  # or 'gaussian'
    # If the list is empty, all keys will be included.
    # For gaussian action distribution:
    action_dist: ActionDistributionConfig = ActionDistributionConfig()
    obs_transforms: Dict[str, ObsTransformConfig] = field(default_factory=dict)
    hierarchical_policy: HierarchicalPolicyConfig = MISSING


@dataclass
class PPOConfig(HabitatBaselinesBaseConfig):
    """Proximal policy optimization config"""

    clip_param: float = 0.2
    ppo_epoch: int = 4
    num_mini_batch: int = 2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 2.5e-4
    eps: float = 1e-5
    max_grad_norm: float = 0.5
    num_steps: int = 5
    use_gae: bool = True
    use_linear_lr_decay: bool = False
    use_linear_clip_decay: bool = False
    gamma: float = 0.99
    tau: float = 0.95
    reward_window_size: int = 50
    use_normalized_advantage: bool = False
    hidden_size: int = 512
    entropy_target_factor: float = 0.0
    use_adaptive_entropy_pen: bool = False
    use_clipped_value_loss: bool = True
    # Use double buffered sampling, typically helps
    # when environment time is similar or larger than
    # policy inference time during rollout generation
    # Not that this does not change the memory requirements
    use_double_buffered_sampler: bool = False


@dataclass
class VERConfig(HabitatBaselinesBaseConfig):
    """Variable experience rollout config"""

    variable_experience: bool = True
    num_inference_workers: int = 2
    overlap_rollouts_and_learn: bool = False


@dataclass
class AuxLossConfig(HabitatBaselinesBaseConfig):
    max_human_num: int = 6
    position_dim: int = 2
    loss_scale: float = 0.1
    future_step: int = 4
    # pass


@dataclass
class CPCALossConfig(AuxLossConfig):
    """Action-conditional contrastive predictive coding loss"""

    k: int = 20
    time_subsample: int = 6
    future_subsample: int = 2
    loss_scale: float = 0.1


@dataclass
class DDPPOConfig(HabitatBaselinesBaseConfig):
    """Decentralized distributed proximal policy optimization config"""

    sync_frac: float = 0.6
    distrib_backend: str = "GLOO"
    rnn_type: str = "GRU"
    num_recurrent_layers: int = 1
    backbone: str = "resnet18"
    resnet_baseplanes: int = 32
    # Visual encoder backbone
    pretrained_weights: str = "data/ddppo-models/gibson-2plus-resnet50.pth"
    # Initialize with pretrained weights
    pretrained: bool = False
    # Loads just the visual encoder backbone weights
    pretrained_encoder: bool = False
    # Whether the visual encoder backbone will be trained
    train_encoder: bool = True
    # Whether to reset the critic linear layer
    reset_critic: bool = True
    # Forces distributed mode for testing
    force_distributed: bool = False
    # Param/buffer name substrings to exclude from DDP allreduce (e.g. decoupled WM).
    # Example: [".net.world_model."] so params never receiving grad in PPO don't trigger reducer error.
    ddp_ignore_param_patterns: List[str] = field(default_factory=list)


@dataclass
class AgentAccessMgrConfig(HabitatBaselinesBaseConfig):
    type: str = "SingleAgentAccessMgr"
    ###############################
    # Population play configuration
    num_agent_types: int = 1
    num_active_agents_per_type: List[int] = field(default_factory=lambda: [1])
    num_pool_agents_per_type: List[int] = field(default_factory=lambda: [1])
    agent_sample_interval: int = 20
    force_partner_sample_idx: int = -1
    # A value of -1 means not configured.
    behavior_latent_dim: int = -1
    # Configuration option for evaluating BDP. If True, then include all
    # behavior agent IDs in the batch. If False, then we will randomly sample IDs.
    force_all_agents: bool = False
    discrim_reward_weight: float = 1.0
    allow_self_play: bool = False
    self_play_batched: bool = False
    # If specified, this will load the policies for the type 1 population from
    # the checkpoint file at the start of training. Used to independently train
    # the type 1 population, and then train a separate against this population.
    load_type1_pop_ckpts: Optional[List[str]] = None
    ###############################


@dataclass
class RLConfig(HabitatBaselinesBaseConfig):
    """Reinforcement learning config"""

    agent: AgentAccessMgrConfig = AgentAccessMgrConfig()
    preemption: PreemptionConfig = PreemptionConfig()
    policy: Dict[str, PolicyConfig] = field(
        default_factory=lambda: {"main_agent": PolicyConfig()}
    )
    ppo: PPOConfig = PPOConfig()
    ddppo: DDPPOConfig = DDPPOConfig()
    ver: VERConfig = VERConfig()
    auxiliary_losses: Dict[str, AuxLossConfig] = field(default_factory=dict)


@dataclass
class ProfilingConfig(HabitatBaselinesBaseConfig):
    capture_start_step: int = -1
    num_steps_to_capture: int = -1


@dataclass
class VectorEnvFactoryConfig(HabitatBaselinesBaseConfig):
    """
    `_target_` points to the `VectorEnvFactory` to setup the vectorized
    environment. Defaults to the Habitat vectorized environment setup.
    """

    _target_: str = "habitat_baselines.common.HabitatVectorEnvFactory"


@dataclass
class EvaluatorConfig(HabitatBaselinesBaseConfig):
    """
    `_target_` points to the `Evaluator` class to instantiate to evaluate the
    policy during evaluation mode.
    """

    _target_: str = (
        "habitat_baselines.rl.ppo.habitat_evaluator.HabitatEvaluator"
    )


@dataclass
class HydraCallbackConfig(HabitatBaselinesBaseConfig):
    """
    Generic callback option for Hydra. Used to create the `_target_` class or
    call the `_target_` method.
    """

    _target_: Optional[str] = None


@dataclass
class WorldModelConfig(HabitatBaselinesBaseConfig):
    """World Model 配置：模型结构 + 训练参数"""

    # 开关
    enabled: bool = False
    train_world_model: bool = False

    # 特征融合（policy 侧）
    fusion_mode: str = "concat"

    # Encoder / 视觉
    encoder_type: str = "dreamer"
    cnn_depth: int = 32
    mlp_units: int = 256
    kernel_size: int = 4
    minres: int = 4

    # NavThinker encoder（当 encoder_type=navthinker 时使用，与 policy 的 ddppo 一致）
    navthinker_backbone: str = "resnet50"
    navthinker_baseplanes: int = 32
    navthinker_ngroups: int = 16
    navthinker_use_projection: bool = True
    navthinker_target_dim: int = 512
    navthinker_pretrained_path: Optional[str] = None
    navthinker_strict_load: bool = False
    freeze_wm_encoder: bool = False  # 冻结 WM encoder，仅训练 RSSM + decoders

    # Depth Anything V2 encoder (encoder_type=depth_anything)
    da_vit_model: str = "vits"
    da_pretrained_path: Optional[str] = None
    da_projection_dim: int = 512
    da_vit_input_size: int = 518

    # Dynamics type: gru / dino_wm
    dynamics_type: str = "gru"
    # DINO-WM Causal ViT Predictor (dynamics_type=dino_wm)
    dino_wm_depth: int = 6
    dino_wm_heads: int = 16
    dino_wm_mlp_dim: int = 2048
    dino_wm_dim_head: int = 64
    dino_wm_num_hist: int = 3
    dino_wm_dropout: float = 0.1
    dino_wm_emb_dropout: float = 0.0

    # DPT depth head (used with dino_wm)
    dpt_head_features: int = 64

    # ViT / patch feature decoder
    feat_decoder_hidden: int = 512
    feat_decoder_layers: int = 2
    feature_loss_scale: float = 0.0

    # RSSM
    dyn_stoch: int = 30
    dyn_deter: int = 200
    dyn_hidden: int = 200
    dyn_discrete: int = 32
    dyn_mean_act: str = "none"
    dyn_std_act: str = "softplus"
    dyn_min_std: float = 0.1
    unimix_ratio: float = 0.01
    initial: str = "learned"

    # 通用
    act: str = "SiLU"
    norm: bool = True
    num_actions: int = 4
    visual_keys: Optional[List[str]] = None
    fuse_keys_1d: Optional[List[str]] = None
    human_state_goal_keys: Optional[List[str]] = None
    num_humans: int = 6
    pred_horizon: int = 4
    use_goal_conditioning: bool = True
    state_goal_dim: int = 8
    residual: bool = True

    pretrained_wm_checkpoint: Optional[str] = None

    # 训练策略
    wm_train_ratio: float = 0.1
    wm_warmup_updates: int = 1000
    wm_grad_clip: float = 100.0

    # WM 优化器与 batch
    wm_lr: float = 3e-4
    opt_eps: float = 1e-5
    weight_decay: float = 0.0
    wm_batch_size: int = 16
    wm_sequence_length: int = 50
    wm_epochs_per_update: int = 1

    # Replay buffer
    replay_buffer_size: int = 100000
    replay_buffer_warmup: int = 5000

    # Loss 权重
    depth_loss_scale: float = 1.0
    traj_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0
    kl_loss_scale: float = 0.1
    kl_free_bits: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1

    # 可视化间隔（每 N 次 WM update 输出一次可视化，0=禁用）
    wm_vis_interval: int = 20

    # Imagination (social-aware latent planning)
    imagine_enabled: bool = False
    imagine_horizon: int = 15
    imagine_social_coeff: float = 0.1
    imagine_safety_margin: float = 0.5
    imagine_intrinsic_coeff: float = 0.01

    # Imagination loss (multi-step prior supervision for WM training)
    imag_loss_scale: float = 0.0
    imag_loss_depth_scale: float = 1.0
    imag_loss_reward_scale: float = 1.0
    imag_loss_traj_scale: float = 1.0
    imag_loss_horizon_gamma: float = 0.95
    imag_loss_horizon: int = 5

    # Replay buffer 独立存储路径（用于阶段 1 采集 + 阶段 2 离线训练）
    # 阶段 1：正常训练时定期把 replay buffer 保存到此路径
    # 阶段 2：wm_offline_only=True 时从此路径加载数据，纯训练 WM
    replay_buffer_path: Optional[str] = None
    replay_buffer_save_interval: int = 50  # 每 N 次 policy update 保存一次 replay buffer

    # 纯采集模式：只收集 replay buffer 到 replay_buffer_path，不训练 WM
    # 需要 enabled=True（构建 encoder 用于特征提取），但 train_world_model 可以为 False
    collect_replay_only: bool = False
    collect_replay_episodes: int = 0  # 采集目标 episode 数（0=不限，由 total_num_steps 控制）
    save_failed_episodes: bool = False  # 是否保存非 success 的 episode（含撞人等）

    # 采集时视觉数据保存开关
    save_depth_images: bool = True
    save_depth_video: bool = True
    save_rgb_images: bool = True
    save_rgb_video: bool = True
    save_third_rgb_images: bool = True
    save_third_rgb_video: bool = True
    save_topdown_images: bool = True
    save_combined_video: bool = True  # eval 风格多视角合成视频
    save_combined_images: bool = False  # 每帧 combined frame 单独保存为 PNG

    # 离线 WM 训练模式：不创建环境，直接从 replay_buffer_path 加载数据训练 WM
    wm_offline_only: bool = False
    wm_offline_updates: int = 10000  # 离线训练总更新次数

    # Depth prediction loss (predictor→DPT→depth end-to-end supervision)
    depth_pred_loss_scale: float = 0.0

    # Lookahead Planning
    lookahead_enabled: bool = False
    lookahead_steps: int = 1
    lookahead_dim: int = 128

    # 分布式
    ddp: bool = True


@dataclass
class PolicyEncoderConfig(HabitatBaselinesBaseConfig):
    """Policy visual encoder configuration."""
    type: str = "resnet"
    da_vit_model: str = "vits"
    da_pretrained_path: Optional[str] = None
    da_projection_dim: int = 512
    da_vit_input_size: int = 224


@dataclass
class HabitatBaselinesConfig(HabitatBaselinesBaseConfig):
    # task config can be a list of configs like "A.yaml,B.yaml"
    # If habitat_baselines.evaluate is true, the run will be in evaluation mode
    # replaces --run-type eval when true
    evaluate: bool = False
    trainer_name: str = "ppo"
    updater_name: str = "PPO"
    distrib_updater_name: str = "DDPPO"
    torch_gpu_id: int = 0
    tensorboard_dir: str = "tb"
    writer_type: str = "tb"
    video_dir: str = "video_dir"
    video_fps: int = 10
    test_episode_count: int = -1
    # path to ckpt or path to ckpts dir
    eval_ckpt_path_dir: str = "data/checkpoints"
    num_environments: int = 16
    num_processes: int = -1  # deprecated
    rollout_storage_name: str = "RolloutStorage"
    checkpoint_folder: str = "data/checkpoints"
    num_updates: int = 10000
    num_checkpoints: int = 10
    # Number of model updates between checkpoints
    checkpoint_interval: int = -1
    total_num_steps: float = -1.0
    log_interval: int = 10
    log_file: str = "train.log"
    force_blind_policy: bool = False
    verbose: bool = True
    # Creates the vectorized environment.
    vector_env_factory: VectorEnvFactoryConfig = VectorEnvFactoryConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    eval_keys_to_include_in_name: List[str] = field(default_factory=list)
    # For our use case, the CPU side things are mainly memory copies
    # and nothing of substantive compute. PyTorch has been making
    # more and more memory copies parallel, but that just ends up
    # slowing those down dramatically and reducing our perf.
    # This forces it to be single threaded.  The default
    # value is left as false as it's different from how
    # PyTorch normally behaves, but all configs we provide
    # set it to true and yours likely should too
    force_torch_single_threaded: bool = False
    # Weights and Biases config
    wb: WBConfig = WBConfig()
    # When resuming training or evaluating, will use the original
    # training config if load_resume_state_config is True
    load_resume_state_config: bool = True
    eval: EvalConfig = EvalConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    # Whether to log the infos that are only logged to a single process to the
    # CLI along with the other metrics.
    should_log_single_proc_infos: bool = False
    # Called every time a checkpoint is saved.
    # Function signature: fn(save_file_path: str) -> None
    # If not specified, there is no callback.
    on_save_ckpt_callback: Optional[HydraCallbackConfig] = None
    # Policy visual encoder 配置
    policy_encoder: PolicyEncoderConfig = field(default_factory=PolicyEncoderConfig)
    # World Model 配置（模型结构 + 训练参数）
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)


@dataclass
class HabitatBaselinesRLConfig(HabitatBaselinesConfig):
    rl: RLConfig = RLConfig()


@dataclass
class HabitatBaselinesILConfig(HabitatBaselinesConfig):
    il: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HabitatBaselinesSPAConfig(HabitatBaselinesConfig):
    sense_plan_act: Any = MISSING


# Register configs to config store
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=HabitatBaselinesRLConfig(),
)
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_il_config_base",
    node=HabitatBaselinesILConfig,
)
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_spa_config_base",
    node=HabitatBaselinesSPAConfig,
)
cs.store(
    group="habitat_baselines/rl/policy", name="policy_base", node=PolicyConfig
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.cpca",
    group="habitat_baselines/rl/auxiliary_losses",
    name="cpca",
    node=CPCALossConfig,
)


from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HabitatBaselinesConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://habitat_baselines/config/",
        )
