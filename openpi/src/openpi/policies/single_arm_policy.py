"""One arm stored as [x, y, z, qw, qx, qy, qz, gripper], e.g. the end-effector versions of rlbench_4 and robocasa_7.

The model's slots are laid out for two arms. The arm goes into the left-arm slots and its wrist camera into the
left-wrist slot; the right-arm slots stay zero and the right-wrist image is black and masked out.
"""

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.policies.robodojo_policy import _parse_image
from openpi.policies.robodojo_policy import local_arm_to_target_pose
from openpi.policies.robodojo_policy import target_to_local_arm_pose

# These grippers mostly point down, about 180 degrees from the base frame, where a rotation vector jumps between +pi
# and -pi. Turning the tool frame by a fixed 180 degrees about its x axis maps "pointing down" close to identity.
# Relative actions (R_action R_state^T) do not change; the outputs turn the tool frame back.
_TOOL_FLIP = Rotation.from_euler("x", np.pi)


def _turn_tool(pose: np.ndarray, rotation: Rotation) -> np.ndarray:
    """[..., xyz, wxyz, rest] -> the same pose with the tool frame right-multiplied by `rotation`."""
    xyzw = np.concatenate([pose[..., 4:7], pose[..., 3:4]], axis=-1).reshape(-1, 4)
    turned = (Rotation.from_quat(xyzw) * rotation).as_quat().reshape(*pose.shape[:-1], 4)
    return np.concatenate([pose[..., :3], turned[..., 3:4], turned[..., :3], pose[..., 7:]], axis=-1)


@dataclasses.dataclass(frozen=True)
class SingleArmInputs(transforms.DataTransformFn):
    # Dims stored after the arm's 8 in each action, e.g. robocasa_7_ee's 5 base-motion commands. They are appended
    # unchanged after the arm's 7, and the state gets as many zeros so that state and actions have the same length,
    # which the rotation and delta-action transforms require.
    extra_action_dims: int = 0

    def __call__(self, data: dict) -> dict:
        # `delta_timestamps` stacks [*extra frames, current frame] onto the base camera (the data config's
        # `extra_frame_keys`); the wrist camera stays a single frame.
        frames = np.asarray(data["images"]["base"])
        extra_images = [_parse_image(frame) for frame in frames[:-1]] if frames.ndim == 4 else []
        base = _parse_image(frames[-1] if frames.ndim == 4 else frames)
        # [xyz, wxyz, gripper] -> [xyz, rpy, gripper]; with no arm offset only the rotation format changes.
        state = local_arm_to_target_pose(
            _turn_tool(np.asarray(data["state"])[..., :8], _TOOL_FLIP), is_left=True, arm_base_distance=0.0
        )
        inputs = {
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": _parse_image(data["images"]["wrist"]),
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
            "state": np.concatenate([state, np.zeros((*state.shape[:-1], self.extra_action_dims))], axis=-1),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            arm = local_arm_to_target_pose(_turn_tool(actions[..., :8], _TOOL_FLIP), is_left=True, arm_base_distance=0.0)
            inputs["actions"] = np.concatenate([arm, actions[..., 8 : 8 + self.extra_action_dims]], axis=-1)
        if extra_images:
            inputs["extra_images"] = np.stack(extra_images)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SingleArmOutputs(transforms.DataTransformFn):
    # Must match the `SingleArmInputs` the checkpoint was trained with.
    extra_action_dims: int = 0

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        arm = target_to_local_arm_pose(actions[..., :7], is_left=True, arm_base_distance=0.0)
        extra = actions[..., 7 : 7 + self.extra_action_dims]
        return {"actions": np.concatenate([_turn_tool(arm, _TOOL_FLIP.inv()), extra], axis=-1)}
