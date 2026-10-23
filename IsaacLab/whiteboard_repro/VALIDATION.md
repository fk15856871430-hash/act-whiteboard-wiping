# 本次迁移整理的本机验证

日期：2026-10-23。验证范围是本机的独立迁移副本及从环境压缩包恢复出的新路径。
没有在另一台物理电脑上执行，也没有重新训练或重新运行带渲染的完整擦除回合。

通过的检查：

- 新建/修改的 shell 脚本通过 bash 语法检查，Python 文件通过编译检查。
- 785 个原始 HDF5 与冻结清单的文件大小逐项匹配，其中 777 个属于原转换规则选集。
- 当前 checkpoint 的 last 相对链接正确指向 100000，checkpoint 含归一化器及 training_state。
- 机器人 USD 的五个 layer 均来自本地 scripts/dual_arm；独立 USD 解析只剩 Isaac 的 OmniPBR.mdl 系统材质依赖。
- 迁移目录符号链接均为相对路径、可解析，Isaac Sim 的原机自引用绝对链接已在副本中改为 `.`。
- 两个环境均已从 conda-pack 压缩包解到独立新路径，完成 conda-unpack 和随包源码 editable 重装。
- 恢复后的 Python 3.11/3.12 环境均通过模块、CUDA、资产、checkpoint 和 dataset metadata 检查。
- 恢复后的环境分别执行仿真入口及采集入口 `--help`，均成功，并包含 enable_cameras 参数。
- 使用恢复出的 ACT 环境、随包服务器源码、随包 checkpoint、随包数据 metadata 与缓存，
  在 HF_HUB_OFFLINE=1 下启动独立测试端口 18556：healthz 返回 OK；WebSocket metadata 是
  ACT / state 20 / action 14 / 51594126 参数；reset 返回空动作；连续 51 次推理均得到有限值的 (1,14) 动作，跨过一次 chunk 更新。
- 测试服务已关闭，不占用正式 5556 端口。

环境审计：原环境内 packaging/setuptools 的 Conda 记录已经过时，与实际运行的 pip 版本不同。
打包器只接受已核对的四组替换版本，并验证实际 pip RECORD 文件完整，封装当前运行文件，
跳过旧 Conda 记录。原 Conda 环境和仓库外 LeRobot 工作区没有被重装或改写。
细节见交付目录 `environments/*.audit.json`。

交付目录的 SHA256SUMS 和 bundle_manifest.json 覆盖普通文件内容、大小与相对符号链接。
目标机完成传输后仍需执行 SHA256 校验，以及 README 第 5 节的带渲染擦除验收。
Simple_Room 在线资产或本地完整资产副本仍须由目标机能访问。
