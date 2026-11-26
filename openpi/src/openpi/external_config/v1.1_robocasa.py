from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotSingleArmDataConfig
from openpi.training.config import TrainConfig
from openpi.training.config import pi0_config

# robocasa_7 alone (Panda on a mobile base, 20 fps), end-effector actions; the end-effector version was converted by
# swx/qz/eef_tools/convert_to_eef.py. Actions are the arm's 7 dims followed by the 5 base-motion commands (base motion
# 4, control mode 1), which stay absolute. History frames come from agentview_left: the episode's first frame and one
# frame per second over the last 8 seconds. Attention is xla_cudnn for the H200 (Hopper): cuDNN for the prefix rows, XLA
# for the action-token rows.
my_config = TrainConfig(
    name="v1.1_robocasa",
    model=pi0_config.Pi0Config(pi05=True, attention_impl="xla_cudnn"),
    data=LeRobotSingleArmDataConfig(
        repo_id="/robot/robot-research-exp-0/user/swx/qz/data/robocasa_7_ee",
        base_camera="observation.images.robot0_agentview_left",
        wrist_camera="observation.images.robot0_eye_in_hand",
        extra_action_dims=5,
        use_quantile_norm=True,
        base_config=DataConfig(
            prompt_from_task=True,
            extra_frames=(0, *range(-8 * 20, 0, 20)),
            extra_frame_keys=("observation.images.robot0_agentview_left",),
        ),
        assets=AssetsConfig(assets_dir="assets", asset_id="v1.1_robocasa"),
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
