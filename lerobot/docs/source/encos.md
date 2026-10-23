# ENCOS dual-arm integration

This workspace exposes the existing ROS 2 controller through the LeRobot device API. The bridge uses
seven joints per arm, in radians. Set `arm_mode=right` for a right-arm-only dataset and policy, or use
the default `both` for 14-DOF bimanual operation. Observations include position, velocity and `iq` for
each selected joint. Camera configs use the regular LeRobot camera syntax.

## Runtime prerequisites

Run these commands in the same shell as LeRobot so Python can import the generated ROS messages:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
cd lerobot
uv sync
```

Start the ENCOS controller nodes first. Check the topics with `ros2 topic echo /dexbotCtl/midMonitor_msg_send`.

## Record a right-arm whiteboard task

The existing `robot_control_hw` node remains the command owner. LeRobot records its interpolated target and
does not publish a second command source. Replace the two serial-number placeholders with the output of
`rs-enumerate-devices -s`:

```bash
lerobot-record \
  --robot.type=encos_dual_arm \
  --robot.command_mode=disabled \
  --robot.arm_mode=right \
  --teleop.type=encos_feedback \
  --teleop.arm_mode=right \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "D405_SERIAL", width: 640, height: 480, fps: 30, use_depth: true}, head: {type: intelrealsense, serial_number_or_name: "D435_SERIAL", width: 640, height: 480, fps: 30, use_depth: true} }' \
  --dataset.repo_id=local/encos-whiteboard-right \
  --dataset.root=./data/encos-whiteboard-right \
  --dataset.fps=20 \
  --dataset.num_episodes=50 \
  --dataset.single_task="erase whiteboard" \
  --dataset.depth_encoder.depth_min=0.07 \
  --dataset.depth_encoder.depth_max=3.0 \
  --dataset.depth_encoder.use_log=true
```

This dataset contains these task-relevant features on every frame:

- `observation.state`: right-arm joint position, velocity, and current for 7 joints.
- `action`: 7 right-arm joint targets from the existing VR control chain.
- `observation.images.wrist` and `observation.images.head`: RGB frames.
- `observation.images.wrist_depth` and `observation.images.head_depth`: depth maps.

Keep the VR publishers and `robot_control_hw` running during recording. The D405 and D435 serial numbers
must identify individual devices; do not use a shared model name when both cameras are connected.
The cameras stream at 30 FPS because this is a standard RealSense profile; LeRobot samples the latest frames
into the dataset at the requested 20 FPS.

The recorded dataset therefore contains both RGB and depth. ACT should initially train on the two RGB views
and the 21-dimensional right-arm state; do not infer `policy.input_features` from this dataset because that
would also include the single-channel depth streams. Depth remains stored in the dataset for inspection,
calibration, and a later depth-aware policy.

## Direct LeRobot control

For whiteboard policy deployment, keep `whiteboard_control_hw` (including
admittance) as the low-level command owner and use its TCP adapter:

```bash
ros2 launch whiteboard_control_hw whiteboard_control.launch.py enable_admittance:=true
lerobot-rollout \
  --robot.type=encos_whiteboard \
  --robot.command_mode=direct \
  --robot.max_relative_target_m=0.008 \
  --policy.path=outputs/encos-whiteboard-act/checkpoints/last/pretrained_model \
  --task="wipe the whiteboard"
```

Direct recording still requires a separate teleoperator that produces actions. For the existing VR
chain, use the recording command above with `command_mode=disabled` so there is only one command owner.

## Train ACT

The resulting dataset is a standard LeRobot dataset. A minimal ACT run is:

```bash
lerobot-train \
  --dataset.repo_id=local/encos-whiteboard-right \
  --policy.type=act \
  --policy.input_features='{ observation.state: {type: STATE, shape: [21]}, observation.images.wrist: {type: VISUAL, shape: [3, 480, 640]}, observation.images.head: {type: VISUAL, shape: [3, 480, 640]} }' \
  --output_dir=outputs/encos-whiteboard-act \
  --job_name=encos-whiteboard-act \
  --steps=100000
```

For deployment, use `lerobot-rollout` with the same `encos_dual_arm` robot config and the trained checkpoint.
