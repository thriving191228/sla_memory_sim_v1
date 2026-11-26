from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotJointDataConfig
from openpi.training.config import TrainConfig
from openpi.training.config import pi0_config

# robodojo_18 alone (the competition's RoboDojo simulation data, two arms, 25 fps) in its own joint space: per arm 6
# joints and a gripper. Besides the three current views the model sees 9 history frames from cam_high: the episode's
# first frame and one frame per second over the last 8 seconds. Attention is plain XLA because the A100 (Ampere)
# cannot run the default xla_cudnn.
my_config = TrainConfig(
    name="v1.0_robodojo",
    model=pi0_config.Pi0Config(pi05=True, attention_impl="xla"),
    data=LeRobotJointDataConfig(
        repo_id="/robot/robot-research-exp-0/user/swx/qz/data/robodojo_18",
        base_camera="observation.images.cam_high",
        left_wrist_camera="observation.images.cam_left_wrist",
        right_wrist_camera="observation.images.cam_right_wrist",
        arm_joints=(6, 6),
        use_quantile_norm=True,
        base_config=DataConfig(prompt_from_task=True, extra_frames=(0, *range(-8 * 25, 0, 25))),
        assets=AssetsConfig(assets_dir="assets", asset_id="v1.0_robodojo"),
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
