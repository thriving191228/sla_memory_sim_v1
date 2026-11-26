"""Joint-space arms stored as [joints..., gripper] per arm, e.g. robodojo_18 (two arms) and rlbench_4_joint (one arm).

State and actions go into the model as they are (the model transforms pad them to the model's action dim). Images go
into the base, left-wrist and right-wrist slots; without a right-wrist camera that slot is black and masked out.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.robodojo_policy import _parse_image


@dataclasses.dataclass(frozen=True)
class JointInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # `delta_timestamps` stacks [*extra frames, current frame] onto the base camera (the data config's
        # `extra_frame_keys`); the wrist cameras stay single frames.
        images = data["images"]
        frames = np.asarray(images["base"])
        extra_images = [_parse_image(frame) for frame in frames[:-1]] if frames.ndim == 4 else []
        base = _parse_image(frames[-1] if frames.ndim == 4 else frames)
        has_right_wrist = "right_wrist" in images
        inputs = {
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": _parse_image(images["left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["right_wrist"]) if has_right_wrist else np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.bool_(has_right_wrist),
            },
            "state": np.asarray(data["state"]),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if extra_images:
            inputs["extra_images"] = np.stack(extra_images)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class JointOutputs(transforms.DataTransformFn):
    # Number of real action dims; the model's output is padded to its own action dim.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
