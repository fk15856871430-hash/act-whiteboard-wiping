# 机械臂擦白板：迁移与复现说明

这份说明对应本机已经实现的 **双臂平台右臂擦白板、ACT WebSocket 推理、Z 向导纳/力反馈修正**。
主入口为 `scripts/my_projects/dagger728.py`，使用场景中原有的不规则随机字迹。
把整个迁移目录传到另一台 Linux 电脑后，先恢复环境，再启动 ACT 服务和仿真。

## 1. 交付目录及本机原件

默认完整交付目录：`/mnt/bigdrive/whiteboard_reproduction_20261023`。
它是独立副本，不依赖原电脑的 `/home/user` 或 `/mnt/bigdrive` 路径。

```text
whiteboard_reproduction_20261023/
├── README.md                       本说明
├── SHA256SUMS                      所有普通文件的 SHA256
├── bundle_manifest.json            文件大小、hash、符号链接清单
├── IsaacLab/                       本机源码快照，包含未提交的任务代码
│   ├── _isaac_sim -> ../isaacsim    相对符号链接
│   ├── scripts/dual_arm/           机器人 USD 分层文件、URDF、meshes
│   ├── scripts/my_projects/        采集和推理主程序
│   ├── official_act/               原始数据 → LeRobot 转换
│   ├── policy_servers/             ACT WebSocket 适配与通信
│   └── whiteboard_repro/           启动、检查、环境恢复及版本快照
├── lerobot/                        本机实际使用、有修改的 LeRobot 源码
├── isaacsim/                       本机 Isaac Sim 程序副本（构建时可选）
├── data/
│   ├── raw_hdf5/                   原始采集 785 个 HDF5，约 56 GiB
│   └── wipe_board_act_lerobot_v3/   已转换 777 回合数据，约 15 GiB
├── models/wipe_board_act/          checkpoint 及训练状态，约 1.2 GiB
├── cache/torch/                    ResNet18 ImageNet 初始权重
├── environments/                   两个 Conda 环境压缩包（单独打包步骤生成）
├── examples/                       已有正常 ACT 单回合 JSON、CSV 与曲线
├── archive/                        历史场景脚本、机器人源资料、另一份 RL 环境代码
└── runs/                           新采集/新训练输出，运行时创建
```

`demo` 构建模式仅有 metadata 和当前 checkpoint，不含训练图像、原始数据；本次完整迁移使用 `full`。
不要只复制 Git 仓库：任务代码原本有很多未跟踪文件，数据与模型也存放在仓库外。

| 环节 | 原电脑文件/目录 | 迁移后的用途 |
| --- | --- | --- |
| 仿真与擦除判定 | `scripts/my_projects/dagger728.py` | ACT 推理、双相机、右臂控制、导纳修正、正常擦除与轨迹记录 |
| 采集场景 | `scripts/my_projects/wipe728.py` | ZMQ 同步环境、双相机、状态与动作写入 HDF5 |
| 专家采集器 | `scripts/my_projects/planner728.py` | 根据剩余字迹进行自动擦除、接触力反馈，成功保存、失败重采 |
| 原始示教 | `/home/user/IsaacLab/vla_dataset` | 已有示教，不要被新采集覆盖 |
| ACT 数据转换 | `official_act/convert_dataset.py` | 两相机裁剪、20D state、14D action、50 Hz LeRobot v3 |
| 训练数据 | `/mnt/bigdrive/datasets/wipe_board_act_lerobot_v3` | 777 回合、245817 帧，可直接训练 |
| 训练 | `train_act.sh` | ACT 100000 step，chunk=50，执行=50，batch=8 |
| 模型 | `/mnt/bigdrive/output_models/wipe_board_act` | 保留 010000、100000 两个 checkpoint；`last -> 100000` |
| 部署 | `deploy_act.sh`、`policy_servers/lerobot_policy_server.py` | 加载权重、归一化器和数据 metadata，端口 5556 |
| 外部训练源码 | `/home/user/catkin_ws_ENCOS擦白板/catkin_ws_ENCOS/lerobot` | 本地修改过的 LeRobot 0.6.2，随包附带，不能换成任意最新版 |
| TCP 数据导出 | `scripts/export_act_wipe_csv.py`、`scripts/plot_act_wipe_trajectory.py` | 单回合 JSON → CSV、曲线 |

## 2. 另一台电脑准备条件

尽量使用 **Ubuntu 22.04 x86_64 + NVIDIA RTX GPU**。原机是 RTX 4090 24 GiB，驱动 580.167.08。
这只是已经使用的硬件基线；没有在另一种显卡上验证最低配置。重训显存不足可减小 batch。
建议目标磁盘至少空出 **150 GiB**，为解压环境和新训练留余量。

已记录的实际软件版本：

| 项目 | 仿真环境 `env_isaaclab` | ACT 环境 `lerobot` |
| --- | --- | --- |
| Python | 3.11.15 | 3.12.13 |
| PyTorch | 2.7.0+cu128 | 2.11.0+cu130 |
| torchvision | 0.22.0+cu128 | 0.26.0+cu130 |
| 框架 | IsaacLab 0.54.3 | LeRobot 0.6.2（附带修改源码） |
| Isaac Sim | 5.1.0-rc.19+release.26219.9c81211b.gl（本机 VERSION 原文） | 独立进程，不加载 Isaac 的 torch |

完整版本、源码 commit、未提交修改、每个环境自己的包列表位于 `IsaacLab/whiteboard_repro/snapshots/`。
旧 `ACT_WIPE_BOARD.md` / server 注释中出现的 LeRobot 0.5.x 是历史文字，以这次快照的 **0.6.2** 为准。
两个环境必须独立，避免把 Isaac 的 PYTHONPATH 注入 ACT 服务；统一启动器会清理 PYTHONPATH。

### 场景资产的联网要求

机器人本体依赖的五个 USD layer 已位于 `scripts/dual_arm/urdf/`，静态解析只剩
Isaac 自带的 `OmniPBR.mdl` 材质依赖。不要只拷贝单个 `dual_arm.usd`。

**Simple_Room 房间仍使用 Isaac 的远程资产地址**，没有把完整 NVIDIA 资产库打进本包。
有网时使用现有默认地址；若目标电脑离线，先准备相同版本的 Isaac 资产包及房间引用的材质/贴图，
然后在 `local.env` 设置 `WHITEBOARD_ROOM_USD=/绝对路径/Environments/Simple_Room/simple_room.usd`。
只拷贝房间 USD 本身通常不够，必须保留其相对依赖目录。没有完成房间资产准备前，不应宣称完全离线复现。

## 3. 如何将整套目录传过去

本次已经在大硬盘生成独立迁移副本；先查看包内 `build_options.json` 和 `environments/` 是否齐全。
用移动硬盘直接复制整个目录即可；推荐 ext4 文件系统，保留符号链接和执行权限。
也可以在原电脑上替换下面的用户名、IP 和目标目录后执行：

```bash
rsync -a --info=progress2 --partial \
  /mnt/bigdrive/whiteboard_reproduction_20261023/ \
  目标用户名@目标IP:/data/whiteboard_reproduction_20261023/
```

不要加 `--delete`。传输中断后重跑同一命令可续传。
目标机进入收到的目录检查完整性：

```bash
cd /data/whiteboard_reproduction_20261023
sha256sum -c SHA256SUMS --quiet
```

`SHA256SUMS` 校验文件内容，`bundle_manifest.json` 同时记录 symlink。
如果工具把 symlink 变成普通文本，需重新创建 `_isaac_sim -> ../isaacsim` 和 checkpoint 的 `last -> 100000`。

## 4. 在目标机恢复两个 Python 环境

优先用附带的 `environments/*.tar.gz` 恢复，不直接拖拽原 Conda 环境目录。
压缩包在原电脑用 conda-pack 生成，恢复后重新安装两个仓库的 editable 源码。
驱动、系统图形库和远程房间资产不包含在 Conda 包里。

```bash
cd /data/whiteboard_reproduction_20261023/IsaacLab
bash whiteboard_repro/restore_environments.sh /data/whiteboard_envs
cp whiteboard_repro/local.env.example whiteboard_repro/local.env
```

用文本编辑器打开 `whiteboard_repro/local.env`，填写：

```bash
WB_SIM_PYTHON=/data/whiteboard_envs/env_isaaclab/bin/python
LEROBOT_PYTHON=/data/whiteboard_envs/lerobot/bin/python
```

其他模型、数据路径会从迁移目录自动推导，无须重建原机目录。
若未附带 Isaac Sim，则在目标机安装相同版本，并在 `IsaacLab` 目录建立
`ln -s /你安装的/isaacsim _isaac_sim`；已有正确链接时不用重复建立。
从二进制副本运行 Isaac Sim 可能需要在目标机执行官方 `post_install.sh` 完成系统集成。

恢复脚本拒绝覆盖已有环境。如果没有环境压缩包，回到原机执行第 9 节打包命令。
`*.conda-explicit.txt` 可重建基础 Conda 包，`*.constraints.txt` 用于约束项目依赖，
但它们本身不是离线 wheel 仓库，也不保证所有历史版本还可下载。不要对包含其他工具的整个环境盲目执行 pip 全量重装。

## 5. 最先验证：用现成 ACT 完成擦白板

所有命令都在迁移后的 `IsaacLab` 目录运行；启动器会选择正确 Python，无须在 base 环境直接导入 Isaac。

```bash
bash whiteboard_repro/run.sh check
```

终端 1，启动模型服务：

```bash
cd /data/whiteboard_reproduction_20261023/IsaacLab
bash whiteboard_repro/run.sh server
```

终端 2，启动正常擦白板场景：

```bash
cd /data/whiteboard_reproduction_20261023/IsaacLab
bash whiteboard_repro/run.sh sim
```

启动器对应你原来的 `./deploy_act.sh` 及
`./isaaclab.sh -p scripts/my_projects/dagger728.py --policy_state_dim 20 --enable_cameras`。
运行的是随机字迹擦除。相机开关必需，policy 状态维度是 20。

可另开终端检查 `curl http://127.0.0.1:5556/healthz`，应返回 `OK`。
验收应看到：双相机正常、服务成功握手、右臂接触白板、字迹逐步消失。
仅启动服务器或通过静态检查，还不能证明目标机仿真已经成功。

记录一次完整擦除，用下面命令替换 `sim`：

```bash
bash whiteboard_repro/run.sh record
# 回合完成、JSON 写出后：
bash whiteboard_repro/run.sh csv
bash whiteboard_repro/run.sh plot
```

默认输出 `trajectory_results/act_wipe_episode.json`、同名 CSV，曲线输出位置由 plot 程序打印。
随包 `examples/` 另有一次历史正常擦除的完整记录（335 帧，擦除率 100%）及导出的 CSV/曲线，便于核对格式。
这个记录模式运行一个普通擦除回合到完成，不设置人工仿真时长上限。
TCP 以机器人基座为参考系保存 4×4 位姿矩阵，另存 world_T_base、姿态、命令位姿、
关节角度/角速度/扭矩/加速度/加加速度、TCP 速度/加速度/加加速度、接触力和擦除进度。
扭矩是仿真关节量，采集 wrench 的后三维为占位零值，不能当作真实六维力传感器测量。
训练 `pose7` 使用世界系/末端连杆参考点；轨迹记录的实际擦布 TCP、基座系与它不同，不能混用。

频率解释：控制步长 0.02 s，即仿真时间 50 Hz；ACT chunk 50 步、执行 50 步，
每次 WS 请求返回一步动作。模型每 50 步重新计算一块，约每 1 秒仿真时间计算一次。
实际墙钟速度由渲染、推理和显卡负载决定，不保证实时 50 Hz。

## 6. 重新采集示教

先停止 ACT 服务和推理场景。采集 ZMQ 和 ACT WebSocket 都占用 **5556**，不能同时启动。
终端 1：

```bash
bash whiteboard_repro/run.sh collect-env
```

等采集仿真准备就绪，终端 2：

```bash
bash whiteboard_repro/run.sh collect-planner
```

专家控制器读取剩余字迹、完成擦除后保存，失败回合废弃重采。
现有采集器内部默认计划 500 个成功回合、每回合墙钟 90 s 失败重试；这是原采集保护逻辑，
与第 5 节正常 ACT 单回合记录的运行时间无关。若只需要少量示教，可在保存提示出现后停止采集器。
控制器最后发送 shutdown，但环境未实现专门退出分支，结束后手动 Ctrl+C 关闭两个终端即可。

新数据统一保存到迁移目录下 `runs/new_demonstrations/`，两进程共享
`WHITEBOARD_RAW_DATA`，不会覆盖随包原始数据。每回合 HDF5 包含两路 JPEG、主相机深度、
pose7、7 关节位置、6D wrench、7D target_pose 和 language_instruction。
控制时间间隔是 20 ms；墙钟录制速度可能低于 50 FPS。

## 7. 数据转换及重训

**直接重训优先使用随包现成数据集**，无需再次做 777 回合转换。

```bash
bash whiteboard_repro/run.sh train
# 显存不足：
# bash whiteboard_repro/run.sh train --batch_size=4
```

默认输出 `runs/wipe_board_act_new`。这里从头训练，不会覆盖随包原模型。
同一新输出目录已有 last checkpoint 时，train 脚本会续训。
若想启动另一组从头训练，在 `local.env` 改用一个不存在的 `OUTPUT_DIR`。

如需从随包原始示教重建与原任务一致的数据快照：

```bash
bash whiteboard_repro/run.sh convert
```

程序按 `snapshots/raw_episodes.json` 冻结的 777 个文件名转换，输出到
`data/wipe_board_act_reconverted`；不会把后来新增的 HDF5 混进来。
原电脑共有 785 文件，原转换器按文件名字典序取前 777 个，末尾是 `episode_1429.hdf5`。
这里只记录现存文件与转换规则，历史数据没有额外的来源签名。

对新采集数据训练时，使用原转换器并显式指定实际回合数量 N：

```bash
source whiteboard_repro/config.sh
env -u PYTHONPATH "${LEROBOT_PYTHON}" official_act/convert_dataset.py \
  --input_dir "${WHITEBOARD_RAW_DATA}" \
  --output_path "${WB_BUNDLE_ROOT}/data/my_new_dataset" \
  --repo_id wipe_board_act_lerobot_v3 --episode_count N --image_size 256 --fps 50
```

把 N 替换成正整数，再在 `local.env` 将 `DATASET_ROOT` 指向新输出路径，同时更换新的 `OUTPUT_DIR`。
目标输出存在时转换器会拒绝，避免误删；不要随意加 `--overwrite`。

转换必须保持：主相机右侧正方形裁剪、腕部居中裁剪、256×256 RGB；
state=`pose7(7)+joint_pos(7)+wrench(6)`；action=14D，前 7 维来自 target_pose，后 7 维补零，
其中 action 四元数被原转换器固定为 `[1,0,0,0]`。部署又使用固定擦拭姿态与 Z 向力反馈修正。
因此是 ACT 与现有控制逻辑共同完成任务，不能把输出解释为任意 6D 姿态自由跟踪。
metadata 中 `robot_type=dual_arm_ur5` 是旧标签；实际平台以随包双臂 URDF、关节名为准，
右臂有 7 个关节，不要因此改用标准 UR5 资产。

继续训练原 100000 step 模型，需明确提高总步数，例如：

```bash
bash whiteboard_repro/run.sh resume --steps=110000
```

它从 `MODEL_PATH` 旁的 training_state 恢复，并覆盖配置中的旧 dataset/output 路径；输出仍到新 runs 目录。
总步数不提高时，已完成的 100000 step 训练不会自动多训。
重训随机性与 GPU/驱动差异可能使结果不完全逐位一致；复现现有模型应优先加载随包 checkpoint。

## 8. 常见启动问题

| 现象 | 处理 |
| --- | --- |
| `No module named isaaclab` | 用恢复后的 env_isaaclab，重新 editable 安装 `source/isaaclab`；不要使用 base |
| 相机/渲染异常 | 保持 `--enable_cameras`，检查 NVIDIA 驱动与图形会话 |
| `No module named lerobot`、导入错误 | 从随包 lerobot 源码重新安装到 Python 3.12 环境，不混用 Isaac torch |
| 连接 5556 失败 | 先看服务是否模型加载完；确认没有采集 ZMQ 或其他策略占用端口 |
| 找不到 robot USD | 保持整个 scripts/dual_arm 目录结构，路径已改为相对代码定位 |
| 找不到 Simple_Room/材质/贴图 | 目标机联网或设置完整本地房间资产路径，见第 2 节 |
| 有权重但服务仍读取 dataset | 适配器需 dataset meta/stats 以构建策略，不能删 metadata |
| 续训仍引用原机路径 | 使用 run.sh resume 或新版 train_act.sh，已显式覆盖 dataset.root/output_dir |

## 9. 原机重新制作迁移包（以后代码更新时）

生成新目录，不会覆盖已有交付：

```bash
cd /home/user/IsaacLab
python whiteboard_repro/build_bundle.py \
  --output /mnt/bigdrive/whiteboard_reproduction_另一个日期 \
  --profile full --include-sim
bash whiteboard_repro/pack_environments.sh /mnt/bigdrive/whiteboard_reproduction_另一个日期
python whiteboard_repro/build_bundle.py \
  --output /mnt/bigdrive/whiteboard_reproduction_另一个日期 --rehash
```

环境打包工具临时安装到 mktemp 目录，不修改两个工作环境。
本机存在 packaging/setuptools 被 pip 替换、旧 Conda 记录未更新的历史情况。
`pack_runtime.py` 仅在实际版本及 pip RECORD 文件完整性通过核对后，打包这两项的当前运行文件，
跳过过时的 Conda 记录，审计结果保存在 `environments/*.audit.json`。
它不修改原环境；出现其他缺失包会停止，不会一概忽略缺失文件。
打包必须避开正在训练/写数据的时段，避免文件在复制或计算 hash 时变化。

## 10. 已有的其他资料放在哪里

以下补充或历史资料不作为当前 ACT 任务主入口；代码已另存 archive，大体积历史数据的取舍逐项列出：

| 原机位置 | 内容及处理 |
| --- | --- |
| `vla_test_data/`，约 272 GiB | 推理/接管/失败历史数据；如要全部归档，另用 rsync 复制，迁移盘需额外空间 |
| `trajectory_results/`，约 385 MiB | 现有曲线、CSV、回合记录及历史测试输出；按文件选取归档，运行本任务可重新生成 |
| `scripts/my_projects/` 其他脚本 | 多次历史版本；完整代码另存 `archive/my_projects/`，实际运行使用主链路 728 三脚本 |
| 根目录 `dual_arm/`，约 812 MiB | 另一份机器人源资料，另存 `archive/robot_source/`；实际部署引用 scripts/dual_arm |
| `source/isaaclab_tasks/.../whiteboard_erasing` | 另一个 manager-based/RL 环境，另存 `archive/whiteboard_erasing_rl/`；本次启动器使用 self-contained 双臂 ACT 场景 |
| `official_diffusion_policy/`、LingBot 相关目录 | 其他策略实现；本次 ACT 独立运行不依赖 |
| `scripts/my_projects/dagger728.py` 接管分支 | 历史 DAgger 功能仍在原脚本中，单回合 record 走正常 ACT 擦除路径 |

本交付的范围是把现有 ACT 白板链路带到另一台电脑运行、采集和训练。
本机已执行的校验见同目录 `VALIDATION.md`；目标机还需实际完成第 5 节的带渲染擦除验收。
