from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotJointDataConfig
from openpi.training.config import TrainConfig
from openpi.training.config import pi0_config

# rlbench_4 alone (Franka Panda, 30 fps) in its own joint space: 7 joints and the gripper. rlbench_4_joint is rlbench_4
# with the open-jar gripper labels fixed (swx/qz/eef_tools/make_rlbench_joint.py). History frames come from the front
# camera: the episode's first frame and one frame per second over the last 8 seconds. Attention is plain XLA because
# the A100 (Ampere) cannot run the default xla_cudnn.
my_config = TrainConfig(
    name="v1.0_rlbench",
    model=pi0_config.Pi0Config(pi05=True, attention_impl="xla"),
    data=LeRobotJointDataConfig(
        repo_id="/robot/robot-research-exp-0/user/swx/qz/data/rlbench_4_joint",
        base_camera="observation.images.front",
        left_wrist_camera="observation.images.wrist",
        arm_joints=(7,),
        use_quantile_norm=True,
        base_config=DataConfig(
            prompt_from_task=True,
            extra_frames=(0, *range(-8 * 30, 0, 30)),
            extra_frame_keys=("observation.images.front",),
        ),
        assets=AssetsConfig(assets_dir="assets", asset_id="v1.0_rlbench"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/SLA_base/params"),
    num_train_steps=30_000,
    save_interval=10_000,
    batch_size=64,
    num_workers=32,
    fsdp_devices=8,
    state_dropout_prob=0.25,
    ema_decay=0.999,
)
