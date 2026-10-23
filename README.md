# ACT + 导纳控制：机械臂擦白板

使用双臂平台的右侧 7 自由度机械臂，在 Isaac Sim 中擦除原有不规则白板字迹。
这里整理了已经实现的仿真、专家采集、ACT 数据转换、训练、WebSocket 部署与 TCP 轨迹记录。
控制频率为仿真时间 50 Hz，ACT chunk=50、执行=50。实际墙钟运行速度由硬件和渲染负载决定。

## 下载哪些内容

代码、本地修改的 LeRobot 和示例曲线随 Git 仓库提供。
机器人 USD/URDF/mesh、模型和大数据存放于 [Releases](https://github.com/fk15856871430-hash/act-whiteboard-wiping/releases/tag/v1.0.0)，不使用付费 Git LFS 配额。
每个附件小于 2 GiB；下载工具支持续传、SHA256 校验及分卷解包。

| 下载组 | 用途 |
| --- | --- |
| `robot-assets` | 原部署机器人完整 USD/URDF/mesh，约 109 MiB |
| `runtime` | 现成 ACT checkpoint、归一化器、训练状态、dataset metadata、ResNet18 缓存 |
| `environments` | 本机两个独立 Python 环境压缩包及审计记录，约 9.3 GiB |
| `training-data` | 已转换的 777 回合、245817 帧 LeRobot 训练数据，约 15 GiB |
| `raw-demonstrations` | 原始 785 个 HDF5，约 56 GiB |

只运行现成模型，下载前三组即可；重训再下载训练数据，重新转换再下载原始示教。
环境为 Ubuntu 22.04 x86_64 打包；原机 RTX 4090 24 GiB、NVIDIA 驱动 580.167.08。

## 在另一台电脑运行

先安装 Git、Python 3、tar、NVIDIA 驱动，并准备约 70 GiB 空间用于推理安装；全部数据和分卷同时保留建议至少 250 GiB。

```bash
git clone https://github.com/fk15856871430-hash/act-whiteboard-wiping.git
cd act-whiteboard-wiping
python3 release_tools/download_assets.py --groups robot-assets runtime environments
```

通过 [NVIDIA 官方渠道](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html) 安装 **Isaac Sim 5.1.0**。
本机的精确 VERSION 为 `5.1.0-rc.19+release.26219.9c81211b.gl`；不同构建需要实际验证。
本仓库不公开分发 Isaac Sim 安装程序。

```bash
# 将下面路径替换成你实际安装的位置；它应包含 setup_conda_env.sh。
ln -s /absolute/path/to/isaacsim IsaacLab/_isaac_sim

# 环境路径使用你有写权限的绝对路径，且不能是已有环境目录。
bash IsaacLab/whiteboard_repro/restore_environments.sh /absolute/path/to/whiteboard_envs
cp IsaacLab/whiteboard_repro/local.env.example IsaacLab/whiteboard_repro/local.env
```

编辑 `IsaacLab/whiteboard_repro/local.env`，填写：

```bash
WB_SIM_PYTHON=/absolute/path/to/whiteboard_envs/env_isaaclab/bin/python
LEROBOT_PYTHON=/absolute/path/to/whiteboard_envs/lerobot/bin/python
```

两个环境分别为 Python 3.11 + torch 2.7.0/cu128、Python 3.12 + torch 2.11.0/cu130。
不要混用环境。然后检查：

```bash
cd IsaacLab
bash whiteboard_repro/run.sh check
```

终端 1：

```bash
cd act-whiteboard-wiping/IsaacLab
bash whiteboard_repro/run.sh server
```

终端 2（带图形桌面的会话）：

```bash
cd act-whiteboard-wiping/IsaacLab
bash whiteboard_repro/run.sh sim
# 或记录一个完整正常擦除回合：
# bash whiteboard_repro/run.sh record
```

Simple_Room 房间依赖在线 Isaac 资产；离线运行须准备房间的完整本地依赖，并在 local.env 设置 `WHITEBOARD_ROOM_USD`。
验收时应看到双相机可用、WebSocket 握手成功、右臂接触白板且字迹逐步消失。
本地恢复环境及连续 51 步 ACT 推理已验证；尚未在另一台物理电脑完成带渲染的完整擦除。

## 采集、训练与轨迹导出

在仓库根目录下载所需数据：

```bash
python3 release_tools/download_assets.py --groups training-data
# 如需从原始示教重新转换：
# python3 release_tools/download_assets.py --groups raw-demonstrations
```

进入 `IsaacLab` 后使用：

```bash
bash whiteboard_repro/run.sh train                # 使用现成训练数据重训，写入独立 runs
bash whiteboard_repro/run.sh convert              # 冻结的 777 个原始回合重新转换
bash whiteboard_repro/run.sh csv                  # 单回合记录导出 CSV
bash whiteboard_repro/run.sh plot                 # 绘制曲线
```

采集时先停止 ACT 服务，在两个终端依次运行 `run.sh collect-env` 与 `run.sh collect-planner`；两套通信共用 5556 端口。
完整流程、数据维度、坐标系、参数和问题排查见 [详细说明](IsaacLab/whiteboard_repro/README.md)。
其中关于“完整迁移副本自带 isaacsim/”的段落仅适用于原机私下迁移包；本 GitHub 公开版以本首页安装方式为准。

## 示例与来源

[examples/](examples/) 含一次正常擦除的 JSON、335 行数据 CSV、关节量与 TCP 曲线，记录的擦除率为 100%。
TCP 记录包含基座坐标系下 4×4 位姿矩阵及速度/加速度/加加速度；关节记录包含角度、角速度、扭矩等。
训练状态采用的世界系末端参考点与记录的实际擦布 TCP 不同，具体定义见详细说明。
查看 [本机验证范围](IsaacLab/whiteboard_repro/VALIDATION.md) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。
