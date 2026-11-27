"""Run the robodojo_18, rlbench_4 and robocasa_7 checkpoints in a simulator, one control step at a time.

Observations and actions use the formats of these training datasets: the `observation.state` vector and the camera
images under their LeRobot keys go in, one `action` vector comes out. The robocasa model was trained on the
end-effector version robocasa_7_ee, so its state and actions are converted here.

The history frames (the episode's first frame and one frame per second over the last 8 seconds) are counted in calls,
so call `act` once per control step at the dataset's frame rate (robodojo 25, rlbench 30, robocasa 20 per second)
and `reset` before every episode.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.policies import policy_config
from openpi.policies.robodojo_memory import RobodojoMemoryBuffer
from openpi.training import config as _config

# Actions run from each 50-step chunk before the model predicts again, by benchmark (the config name's suffix).
# robodojo runs the whole chunk, as the XPolicyLab RoboDojo evaluation does.
REPLAN_STEPS = {"robodojo": 50, "rlbench": 10, "robocasa": 16}


def robocasa_state(state: np.ndarray) -> np.ndarray:
    """robocasa_7 state (16) -> robocasa_7_ee state (8), as robocasa_7_ee was converted.

    robocasa_7: base position 3, base quaternion 4, hand position 3 and quaternion 4 (x, y, z, w) in the base frame,
    two finger positions 2. robocasa_7_ee: the hand [x, y, z, qw, qx, qy, qz] and its opening (1 open, 0 closed).
    """
    xyzw = Rotation.from_quat(state[10:14]).as_quat()
    opening = np.clip(state[14] - state[15], 0.0, 0.08) / 0.08
    return np.concatenate([state[7:10], xyzw[3:], xyzw[:3], [opening]])


def robocasa_action(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """One robocasa_7_ee action (13) and the current robocasa_7 state (16) -> one robocasa_7 action (12, -1~1).

    robocasa_7_ee: the hand's absolute target [x, y, z, qw, qx, qy, qz, opening] in the base frame, then the 5 base
    dims as in robocasa_7. robocasa_7: base motion 4, control mode 1 (-1 or 1), robosuite OSC_POSE deltas from the
    current hand pose (target position = current + 0.05 a, target rotation = Exp(0.5 a) current), gripper (-1 open,
    1 closed).
    """
    turn = Rotation.from_quat([*action[4:7], action[3]]) * Rotation.from_quat(state[10:14]).inv()
    arm = np.concatenate([(action[:3] - state[7:10]) / 0.05, turn.as_rotvec() / 0.5, [1.0 - 2.0 * action[7]]])
    out = np.clip(np.concatenate([action[8:13], arm]), -1.0, 1.0)
    out[4] = 1.0 if out[4] > 0 else -1.0
    return out


class SimPolicy:
    """obs = {"observation.state": ..., "observation.images.<camera>": [H, W, 3] uint8 RGB, "prompt": instruction}.

    The cameras are the ones the config was trained on (`base_camera` and the wrist cameras in `external_config/`).
    The model predicts 50 actions at a time and predicts again after `replan_steps` calls (default: `REPLAN_STEPS`).
    """

    def __init__(self, config_name: str, checkpoint_dir: str, replan_steps: int | None = None):
        benchmark = config_name.rsplit("_", 1)[-1]
        replan_steps = REPLAN_STEPS[benchmark] if replan_steps is None else replan_steps
        train_config = _config.get_config(config_name)
        if not 1 <= replan_steps <= train_config.model.action_horizon:
            raise ValueError(f"replan_steps must be in [1, {train_config.model.action_horizon}], got {replan_steps}")
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        # Model slot -> LeRobot camera key, as in training, e.g. {"base": "observation.images.cam_high", ...}.
        self._cameras = data_config.repack_transforms.inputs[0].structure["images"]
        # extra_frames is (0, -8 * fps, ..., -fps) in all three configs.
        self._fps = -data_config.extra_frames[-1]
        self._seconds = len(data_config.extra_frames) - 1
        self._robocasa = benchmark == "robocasa"
        self._replan_steps = replan_steps
        self._policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
        self.reset()

    def reset(self) -> None:
        self._history = RobodojoMemoryBuffer(fps=self._fps, history_seconds=self._seconds)
        self._chunk = None
        self._step = 0

    def act(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["observation.state"], dtype=np.float64)
        images = {slot: obs[key] for slot, key in self._cameras.items()}
        # The buffer stacks the history of the camera it calls cam_high; here that is the base camera.
        images["cam_high"] = images.pop("base")
        model_state = robocasa_state(state) if self._robocasa else state
        self._history.update({"images": images, "state": model_state.astype(np.float32), "prompt": obs["prompt"]})
        if self._chunk is None or self._step == self._replan_steps:
            model_obs = self._history.build_observation()
            model_obs["images"]["base"] = model_obs["images"].pop("cam_high")
            self._chunk = np.asarray(self._policy.infer(model_obs)["actions"])
            self._step = 0
        action = self._chunk[self._step]
        self._step += 1
        return robocasa_action(action, state) if self._robocasa else action
