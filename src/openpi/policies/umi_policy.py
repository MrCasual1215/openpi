import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_umi_example() -> dict:
    """Creates a random input example for the UMI policy."""
    return {
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/ee_pose": np.random.rand(6),
        "observation/gripper_width": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """Maps UMI observations to the model input format.

    Expected inputs:
    - observation/wrist_image: wrist RGB image
    - observation/ee_pose: 6D end-effector pose [x, y, z, rx, ry, rz]
    - observation/gripper_width: scalar gripper width
    - actions: optional [action_horizon, 7] training targets
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(data["observation/wrist_image"])
        ee_pose = np.asarray(data["observation/ee_pose"])
        gripper_width = np.asarray(data["observation/gripper_width"])
        if gripper_width.ndim == 0:
            gripper_width = gripper_width[np.newaxis]

        state = np.concatenate([ee_pose, gripper_width], axis=0)

        inputs = {
            "state": state,
            "image": {
                # UMI only provides a wrist camera, so the unused image slots are masked out.
                "base_0_rgb": np.zeros_like(wrist_image),
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    """Maps model outputs back to UMI actions."""

    def __call__(self, data: dict) -> dict:
        # UMI actions are [dpose(6), gripper_abs(1)].
        return {"actions": np.asarray(data["actions"][:, :7])}
