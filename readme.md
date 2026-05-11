## pi0.5 部署修改内容

- `1. 数据转换脚本`
  - 参考 [convert\_libero\_data\_to\_lerobot.py](file:///home/sunpeng/sp/pi/openpi/examples/libero/convert_libero_data_to_lerobot.py#L37-L96)。
  - 你需要把原始机器人日志转成 LeRobot 数据集，至少包含：
    - `image` 或你自定义的 RGB 键
    - `state` 或拆开的 `ee_pose` / `gripper_width`
    - `actions`
    - `task` 或 `prompt`
    - 正确的 `fps`
  - `fps` 非常重要，因为训练时 action chunk 的时间间隔是按数据集 `fps` 算的，见 [data\_loader.py](file:///home/sunpeng/sp/pi/openpi/src/openpi/training/data_loader.py#L140-L145)。
- `2. 自定义 policy 适配文件`
  - 新建一个类似 `src/openpi/policies/my_robot_policy.py` 的文件。
  - 这个文件要模仿 [libero\_policy.py](file:///home/sunpeng/sp/pi/openpi/src/openpi/policies/libero_policy.py#L29-L100) 或 [droid\_policy.py](file:///home/sunpeng/sp/pi/openpi/src/openpi/policies/droid_policy.py#L30-L80)。
  - 它的职责只有两个：
    - `Inputs`: 把你原始字段映射成统一的 `state/image/image_mask/prompt/actions`
    - `Outputs`: 从模型输出里截取你真实需要的动作维度
- `3. 自定义 DataConfig`
  - 在 [config.py](file:///home/sunpeng/sp/pi/openpi/src/openpi/training/config.py#L282-L355) 的风格上，新增一个 `LeRobotMyRobotDataConfig`。
  - 这里要定义：
    - `repack_transform`: 你的数据集键名怎么映射到推理时的键名
    - `data_transforms`: 用你的 `MyRobotInputs/MyRobotOutputs`
    - 是否加 `DeltaActions`
    - `prompt_from_task`
    - `repo_id`
- `4. 自定义 TrainConfig`
  - 在 [config.py](file:///home/sunpeng/sp/pi/openpi/src/openpi/training/config.py#L651-L918) 这种写法里新增一个训练配置，比如 `pi05_my_robot`。
  - 这里指定：
    - 用 `pi0` 还是 `pi05`
    - `action_horizon`
    - `repo_id`
    - `weight_loader`
    - batch size / steps
  - 训练入口 `scripts/train.py` 本身不用改。

<br />

<br />

<br />



### 数据集转换
``` bash
uv run examples/umi/convert_umi_data_to_lerobot.py \
  --data-dir /home/sunpeng/sp/pi/dataset/single/20260428 \
  --repo-id sp/umi_dataset \
  --task "pick up the red block and place it into the white rectangular tray"
```
### 归一化
修改 pi05_umi 里的：
```
repo_id="sp/umi_dataset"
```
```
uv run scripts/compute_norm_stats.py --config-name pi05_umi
```