import pathlib

from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotRobodojoDataConfig
from openpi.training.config import LeRobotSingleArmDataConfig
from openpi.training.config import MixedDataSourceConfig
from openpi.training.config import MixedLeRobotDataConfig
from openpi.training.config import TrainConfig
from openpi.training.config import pi0_config

# End-effector versions of the three competition datasets (converted by swx/qz/eef_tools/convert_to_eef.py).
_DATA_ROOT = pathlib.Path("/robot/robot-research-exp-0/user/swx/qz/data")

# robodojo_18 (two arms, 25 fps), rlbench_4 (Panda, 30 fps) and robocasa_7 (Panda on a mobile base, 20 fps; the base is
# not learned) trained together, end-effector actions. Each source gets 9 history frames from its base view: the
# episode's first frame and one frame per second over the last 8 seconds, so the stride follows its fps. Sources are
# mixed by frame count (every frame once per epoch).
my_config = TrainConfig(
    name="v1.0_init",
    model=pi0_config.Pi0Config(pi05=True),
    data=MixedLeRobotDataConfig(
        repo_id="v1.0_init",
        sources=(
            MixedDataSourceConfig(
                name="robodojo_18",
                weight=1.0,
                data_factory=LeRobotRobodojoDataConfig(
                    repo_id=str(_DATA_ROOT / "robodojo_18_ee"),
                    use_eef_action=True,
                    use_so3_rpy=True,
                    base_config=DataConfig(prompt_from_task=True, extra_frames=(0, *range(-8 * 25, 0, 25))),
                ),
            ),
            MixedDataSourceConfig(
                name="rlbench_4",
                weight=1.0,
                data_factory=LeRobotSingleArmDataConfig(
                    repo_id=str(_DATA_ROOT / "rlbench_4_ee"),
                    base_camera="observation.images.front",
                    wrist_camera="observation.images.wrist",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        extra_frames=(0, *range(-8 * 30, 0, 30)),
                        extra_frame_keys=("observation.images.front",),
                    ),
                ),
            ),
            MixedDataSourceConfig(
                name="robocasa_7",
                weight=1.0,
                data_factory=LeRobotSingleArmDataConfig(
                    repo_id=str(_DATA_ROOT / "robocasa_7_ee"),
                    base_camera="observation.images.robot0_agentview_left",
                    wrist_camera="observation.images.robot0_eye_in_hand",
                    base_config=DataConfig(
                        prompt_from_task=True,
                        extra_frames=(0, *range(-8 * 20, 0, 20)),
                        extra_frame_keys=("observation.images.robot0_agentview_left",),
                    ),
                ),
            ),
        ),
        use_quantile_norm=True,
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(assets_dir="assets", asset_id="v1.0_init"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/SLA_base/params"),
    num_train_steps=60_000,
    save_interval=10_000,
    # 512 samples per step. On one node of 80 GB GPUs pass `--batch-size 256 --gradient-accumulation-steps 2`.
    batch_size=512,
    num_workers=32,
    fsdp_devices=8,
    state_dropout_prob=0.25,
    ema_decay=0.999,
)
