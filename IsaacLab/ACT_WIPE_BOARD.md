# ACT 擦黑板训练与部署

## 记录一次正常 ACT 擦黑板轨迹

这条路径使用现有的 WebSocket 推理与随机白板字迹环境，不启用 A-B benchmark。启动 ACT 服务：

```bash
cd /home/user/IsaacLab
./deploy_act.sh
```

另开终端运行一次回合并导出轨迹：

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd /home/user/IsaacLab
./isaaclab.sh -p scripts/my_projects/dagger728.py \
  --policy_state_dim 20 \
  --enable_cameras \
  --record_trajectory trajectory_results/act_wipe_episode.json
```

`--record_trajectory` 会将普通 ACT 擦除回合限制为 1 回合，并持续运行到字迹擦除完成；不设置
仿真时间上限。JSON 以 50 Hz 保存基座系 TCP 位姿矩阵、命令位姿、关节角度/速度/扭矩/加速度/加加速度、接触力和擦除进度。

目标：保持 Isaac Sim 擦黑板场景和 URDF 不变，使用与官方 DP 相同的 777 个 episode
训练 ACT，然后通过与 LingBotVLA 相同的 5556 WebSocket 接口部署。LingBotVLA 仍默认
使用 14D 状态；ACT 使用 20D 状态 `pose7 + joint7 + wrench6`。

## 已准备完成

- LeRobot 环境：`/home/user/miniconda3/envs/lerobot`，训练入口已验证。
- 数据集：`/mnt/bigdrive/datasets/wipe_board_act_lerobot_v3`。
- 数据规模：777 episodes、245817 frames、两路 256x256 RGB、20D state、14D action。
- ResNet18 ImageNet 预训练权重已经缓存，无需在训练时联网下载。
- 旧的 LingBotVLA 代码和旧 ACT 数据目录均未修改。

## 1. 训练 ACT

```bash
cd /home/user/IsaacLab
./train_act.sh
```

脚本训练 100000 step，每 10000 step 保存一次。再次执行同一命令时，如果已经存在
`checkpoints/last`，会自动续训。

显存不足时降低 batch size：

```bash
./train_act.sh --batch_size=4
```

仍然显存不足时依次改为 2、1，不要修改状态和动作维度。

训练输出：

```text
/mnt/bigdrive/output_models/wipe_board_act/checkpoints/010000/pretrained_model
/mnt/bigdrive/output_models/wipe_board_act/checkpoints/last/pretrained_model
```

`last` 是 LeRobot 自动维护的符号链接，始终指向最近一次保存的 checkpoint。

## 2. 启动 ACT 服务

```bash
cd /home/user/IsaacLab
./deploy_act.sh
```

服务启动后会从 checkpoint 读取 ACT 配置、模型权重以及训练时保存的归一化处理器。服务端
保留 ACT 的 50 步动作缓存，但每次向 Isaac 客户端返回一个 `[1, 14]` 动作。

## 3. 启动擦黑板场景

另开终端运行：

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd /home/user/IsaacLab
./isaaclab.sh -p scripts/my_projects/dagger728.py --policy_state_dim 20 --enable_cameras
```

场景继续使用原来的两路 256x256 图像、固定擦拭四元数、动作控制器、URDF 和擦除判定。

## 4. 切回 LingBotVLA

LingBotVLA 服务命令不变。场景不传新参数时默认仍为 14D，也可以显式运行：

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/my_projects/dagger728.py --policy_state_dim 14
```
