import argparse
import json
import os
import resource
import h5py
import math
import itertools
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R
import cv2
import time
import numpy as np
import sys
import glob
from sklearn.cluster import DBSCAN

# ======================================================================
# 0. 工具函数：平滑过渡（用于退场抬手与转场）
# ======================================================================
def generate_smooth_transition(start_pos, end_pos, speed_factor=400):
    dist = np.linalg.norm(end_pos - start_pos)
    steps = max(int(dist * speed_factor), 20)
    t = np.linspace(0, np.pi, steps)
    weights = (1 - np.cos(t)) / 2.0
    path = start_pos + weights[:, np.newaxis] * (end_pos - start_pos)
    return path

# ======================================================================
# 0.5 工具函数：严格对齐模型训练时的图像裁剪与缩放
# ======================================================================
def apply_model_crop_resize(img_bgr_or_rgb, target_res=(256, 256), crop_type='center'):
    """
    与 convert.py 中完全一致的裁剪逻辑：
    - 主相机 crop_type='right'  -> 从右侧正方形裁剪
    - 腕部相机 crop_type='center' -> 中心正方形裁剪
    - 然后 resize 到 target_res
    """
    h, w = img_bgr_or_rgb.shape[:2]
    size = min(h, w)

    if crop_type == 'right':
        offset_y = (h - size) // 2
        offset_x = w - size
        img_cropped = img_bgr_or_rgb[offset_y:offset_y + size, offset_x:offset_x + size]
    else:  # center
        offset_y = (h - size) // 2
        offset_x = (w - size) // 2
        img_cropped = img_bgr_or_rgb[offset_y:offset_y + size, offset_x:offset_x + size]

    if size != target_res[0]:
        img_cropped = cv2.resize(img_cropped, target_res, interpolation=cv2.INTER_AREA)

    return img_cropped

# ======================================================================
# 1. 启动仿真应用
# ======================================================================
parser = argparse.ArgumentParser(description="Isaac Lab WebSocket Env (VLA with Gripper)")
parser.add_argument(
    "--policy_state_dim",
    type=int,
    choices=(14, 20),
    default=14,
    help="Policy state width. Keep 14 for LingBotVLA; use 20 for ACT or official Diffusion Policy.",
)
parser.add_argument(
    "--policy_port",
    type=int,
    default=5556,
    help="WebSocket policy server port (default: 5556).",
)
parser.add_argument(
    "--record_trajectory",
    type=str,
    default="",
    help=(
        "Record one normal ACT whiteboard episode to this JSON path. "
        "The regular WebSocket wiping loop is used; no benchmark mode is enabled."
    ),
)
parser.add_argument(
    "--fast_dp_deploy",
    action="store_true",
    help=(
        "Use faster render settings for official DP deployment. Standard 14D DP skips episode "
        "recording; the 2D small-XY checkpoint keeps recording and performs an inspection lift."
    ),
)
parser.add_argument(
    "--dp_raw_action_eval",
    action="store_true",
    help="Evaluate standard official DP actions without Z contact/admittance assistance.",
)
parser.add_argument(
    "--dp_eval_episodes",
    type=int,
    default=0,
    help="Official-DP-only evaluation episode count; 0 keeps the existing default.",
)
parser.add_argument(
    "--dp_eval_timeout",
    type=float,
    default=70.0,
    help="Official-DP-only hard timeout in simulated seconds.",
)
parser.add_argument(
    "--dp_eval_seed",
    type=int,
    default=42,
    help="Base seed used only for reproducible official DP evaluation episodes.",
)
parser.add_argument(
    "--dp_eval_output",
    type=str,
    default="",
    help="Optional JSON path for official DP evaluation results.",
)
parser.add_argument(
    "--trajectory_benchmark",
    action="store_true",
    help="Run the fixed diagonal-stroke erasing benchmark instead of the random wiping task.",
)
parser.add_argument(
    "--trajectory_benchmark_algorithm",
    choices=("act", "dp", "dp_small_xy", "visual_movj", "visual_movl", "ideal"),
    default="act",
    help=(
        "Controller to benchmark. visual_movj and visual_movl are vision-path classical baselines; "
        "dp is the standard 14D checkpoint and dp_small_xy is the 2D XY checkpoint."
    ),
)
parser.add_argument(
    "--trajectory_benchmark_act_admittance",
    action="store_true",
    help=(
        "Apply the standard-DP Z contact approach and post-contact admittance controller to ACT. "
        "ACT still supplies its own XY, orientation, and nominal Z commands."
    ),
)
parser.add_argument(
    "--trajectory_benchmark_start",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    default=(0.06, -0.24, 0.225),
    help="World-frame whiteboard start point A in metres.",
)
parser.add_argument(
    "--trajectory_benchmark_end",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    default=(0.18, -0.24, 0.225),
    help="World-frame whiteboard end point B in metres.",
)
parser.add_argument(
    "--trajectory_benchmark_duration",
    type=float,
    default=24.0,
    help="Common maximum A-to-B completion time in simulated seconds; methods may finish earlier.",
)
parser.add_argument(
    "--trajectory_benchmark_joint_torque_limit",
    type=float,
    default=80.0,
    help="Common absolute right-arm torque limit in N m, enforced by PhysX.",
)
parser.add_argument(
    "--trajectory_benchmark_joint_acceleration_limit",
    type=float,
    # The measured physical derivative includes discrete board-contact impulses.
    default=300.0,
    help="Common absolute right-arm acceleration acceptance limit in rad/s^2.",
)
parser.add_argument(
    "--trajectory_benchmark_joint_jerk_limit",
    type=float,
    # This is the shared physical acceptance threshold for the contact simulation.
    default=20000.0,
    help="Common absolute right-arm jerk acceptance limit in rad/s^3.",
)
parser.add_argument(
    "--trajectory_benchmark_joint_velocity_limit",
    type=float,
    default=0.5,
    help="Common right-arm joint velocity limit in rad/s for the constrained benchmark controller.",
)
parser.add_argument(
    "--trajectory_benchmark_pd_stiffness",
    type=float,
    default=100.0,
    help="Common implicit actuator stiffness used only by the constrained benchmark.",
)
parser.add_argument(
    "--trajectory_benchmark_pd_damping",
    type=float,
    default=30.0,
    help="Common implicit actuator damping used only by the constrained benchmark.",
)
parser.add_argument(
    "--trajectory_benchmark_timeout",
    type=float,
    default=30.0,
    help="Maximum simulated time allowed for contact and erasing.",
)
parser.add_argument(
    "--trajectory_benchmark_settle",
    type=float,
    default=3.0,
    help="Settling time at A before recording in simulated seconds.",
)
parser.add_argument(
    "--trajectory_benchmark_endpoint_solve",
    type=float,
    default=12.0,
    help="Excluded simulated seconds allowed to solve the common B endpoint before a trial.",
)
parser.add_argument(
    "--trajectory_benchmark_endpoint_calibration_gain",
    type=float,
    default=0.5,
    help="Cartesian outer-loop gain used only while calibrating the excluded common A/B setup.",
)
parser.add_argument(
    "--trajectory_benchmark_endpoint_final_settle",
    type=float,
    default=1.0,
    help=(
        "Excluded simulated seconds used to settle the final calibrated A/B command "
        "before its physical TCP error is measured."
    ),
)
parser.add_argument(
    "--trajectory_benchmark_tolerance",
    type=float,
    default=0.005,
    help="Shared endpoint tolerance in metres used to validate every trial.",
)
parser.add_argument(
    "--trajectory_benchmark_contact_force",
    type=float,
    default=3.0,
    help="Normal-force threshold that starts the metric window, in newtons.",
)
parser.add_argument(
    "--trajectory_benchmark_erase_force",
    type=float,
    default=5.0,
    help="Normal-force threshold required to erase stroke points, in newtons.",
)
parser.add_argument(
    "--trajectory_benchmark_completion_ratio",
    type=float,
    default=1.0,
    help="Fraction of diagonal-stroke points that must be removed for completion.",
)
parser.add_argument(
    "--trajectory_benchmark_stroke_points",
    type=int,
    default=121,
    help="Number of visible points used to render and score the diagonal stroke.",
)
parser.add_argument(
    "--trajectory_benchmark_stroke_style",
    choices=("curve", "arc", "irregular_word", "straight"),
    default="curve",
    help="Visible writing style. curve is the common smooth handwriting comparison task.",
)
parser.add_argument(
    "--trajectory_benchmark_arc_center",
    nargs=2,
    type=float,
    metavar=("X", "Y"),
    default=(0.12, -0.30),
    help="World-frame XY centre of the circular-arc handwriting in metres.",
)
parser.add_argument(
    "--trajectory_benchmark_arc_direction",
    choices=("cw", "ccw"),
    default="cw",
    help="Direction used to traverse the circular arc from A to B.",
)
parser.add_argument(
    "--trajectory_benchmark_policy_arc_residual_limit",
    type=float,
    default=0.004,
    help=(
        "Maximum XY tracking-tube residual in metres that ACT/DP may add to their "
        "policy-guided circular-arc command. Set to zero for no residual amplitude limit."
    ),
)
parser.add_argument(
    "--trajectory_benchmark_seed",
    type=int,
    default=42,
    help="Seed for the simulator and DP policy reset.",
)
parser.add_argument(
    "--trajectory_benchmark_output",
    type=str,
    default="trajectory_benchmark_result.json",
    help="JSON result path for one trajectory benchmark run.",
)
parser.add_argument(
    "--trajectory_benchmark_preflight",
    action="store_true",
    help="Validate the common A/B endpoint setup and limits only; do not run a policy or measured trajectory.",
)
parser.add_argument(
    "--trajectory_benchmark_video",
    type=str,
    default="",
    help="Optional chest-camera MP4 path for the trajectory benchmark.",
)
parser.add_argument(
    "--trajectory_benchmark_video_fps",
    type=float,
    default=50.0,
    help="Frame rate for --trajectory_benchmark_video (default matches the 20 ms control period).",
)
parser.add_argument(
    "--trajectory_benchmark_hold_after_completion",
    type=float,
    default=0.0,
    help=(
        "After a successful erase, keep the GUI scene open for this many simulated seconds. "
        "The hold phase is excluded from benchmark samples and metrics."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKControllerCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ---------- WebSocket + msgpack ----------
from websockets.sync.client import connect
import msgpack_numpy as m
m.patch()
from msgpack_numpy import Packer, unpackb

# ======================================================================
# 📍 全局锚点与材质（完全对齐脚本1）
# ======================================================================
DESK_X, DESK_Y, DESK_Z = 0.0, 0.0, 0.0  
BOARD_Z = DESK_Z
WORKING_TCP_Z = BOARD_Z + 0.022
CAM_EULER_DEGREES = [-45.07, -34.968, -150.242]
_q = R.from_euler('XYZ', CAM_EULER_DEGREES, degrees=True).as_quat()
CAM_ROT_WXYZ = (_q[3], _q[0], _q[1], _q[2])   # IsaacLab 四元数 (w,x,y,z)
_wipe_rot_env = R.from_euler('z', 90, degrees=True).as_quat()
LOCKED_WIPE_QUAT = np.array([_wipe_rot_env[3], _wipe_rot_env[0], _wipe_rot_env[1], _wipe_rot_env[2]])  # (w,x,y,z)

HOME_X = 0.05
HOME_Y = -0.4
SURFACE_Z_EST = BOARD_Z + 0.21     
Z_HOVER = BOARD_Z + 0.30           
SAFE_RETURN_Z = BOARD_Z + 0.40     

METAL_MAT = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.98, 0.98, 0.98), metallic=0.4, roughness=0.1)
DARK_PLASTIC_MAT = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), roughness=0.9, metallic=0.0)
FRAME_PROPS = sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6)

WRITING_REGION_X = (DESK_X + 0.06,  DESK_X + 0.18)   
WRITING_REGION_Y = (DESK_Y - 0.32,  DESK_Y - 0.08)   
DIAGRAM_REGION_X = (DESK_X + 0.07,  DESK_X + 0.17)
DIAGRAM_REGION_Y = (DESK_Y - 0.31,  DESK_Y - 0.09)

BOARD_ORIGIN = (0.155, -0.155, BOARD_Z + 0.01)
FRAME_TOP_ORIGIN = (0.155, 0.205, BOARD_Z + 0.015)
FRAME_BOTTOM_ORIGIN = (0.155, -0.515, BOARD_Z + 0.015)
FRAME_LEFT_ORIGIN = (0.415, -0.155, BOARD_Z + 0.015)
FRAME_RIGHT_ORIGIN = (-0.105, -0.155, BOARD_Z + 0.015)

SAVE_EPISODE_DATA = True
SAVE_REMNANT = True

# The benchmark action is applied every two 10 ms physics steps.  Keep this
# explicit because ACT demonstrations and trajectory logs use this cadence.
TRAJECTORY_SAMPLE_HZ = 50.0
TRAJECTORY_SAMPLE_DT_S = 1.0 / TRAJECTORY_SAMPLE_HZ

# ======================================================================
# 2. 双臂机器人配置（完全对齐脚本1）
# ======================================================================
DUAL_ARM_RIGHT_CTRL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dual_arm", "urdf", "dual_arm.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=16,  
            solver_velocity_iteration_count=1,   
            fix_root_link=True,  
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_r_joint_1": 0.9, "arm_r_joint_2": -1.2, "arm_r_joint_3": -0.2, "arm_r_joint_4": 0.8, 
            "arm_r_joint_5": 0.0, "arm_r_joint_6": 0.0, "arm_r_joint_7": 0.0,
            "arm_l_joint_1": -0.7, "arm_l_joint_2": 1.0, "arm_l_joint_3": 0.0, "arm_l_joint_4": -0.7,  
            "arm_l_joint_5": 0.0, "arm_l_joint_6": 0.0, "arm_l_joint_7": 0.0,
            "arm_r_joint_finger": 0.6,
            "arm_l_joint_finger": -1.0,
        },
    ),
    actuators={
        "arms": ImplicitActuatorCfg(
            # Keep the arm and gripper actuators disjoint.  The former
            # expression also matched ``*_joint_finger``, causing duplicate
            # actuator commands and invalid dynamic measurements.
            joint_names_expr=["arm_r_joint_[1-7]", "arm_l_joint_[1-7]"],
            stiffness=4000.0, damping=400.0,            
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["arm_r_joint_finger", "arm_l_joint_finger"],
            stiffness=800.0, damping=80.0,            
        ),
    },
)

# ======================================================================
# 3. 场景配置（完全对齐脚本1，但去掉深度相机渲染）
# ======================================================================
_tilt_q = R.from_euler('x', -20, degrees=True).as_quat()
CHEST_CAM_ROT = (_tilt_q[3], _tilt_q[0], _tilt_q[1], _tilt_q[2])  # (w,x,y,z)

@configclass
class WipeBoardSceneCfg(InteractiveSceneCfg):
    simple_room = AssetBaseCfg(  
        prim_path="/World/envs/env_.*/SimpleRoom",
        spawn=sim_utils.UsdFileCfg(usd_path=os.environ.get("WHITEBOARD_ROOM_USD", f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot: ArticulationCfg = DUAL_ARM_RIGHT_CTRL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot.init_state.pos = (DESK_X - 0.2, DESK_Y, DESK_Z + 0.55) 
    
    robot_pedestal = AssetBaseCfg(
        prim_path="/World/envs/env_.*/RobotPedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.46), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.18), roughness=0.5, metallic=0.9),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(DESK_X - 0.2, DESK_Y, DESK_Z + 0.23)
        ),
    )

    eraser_shell = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Robot/arm_r_link_7/eraser_shell",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.06, 0.04), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.6), roughness=0.3)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.02, -0.185), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    eraser_cloth = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Robot/arm_r_link_7/eraser_cloth",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.06, 0.015), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.8, 0.2), roughness=0.9), 
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.3, restitution=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=-0.005),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.02, -0.2125), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/arm_r_link_7", 
        update_period=0.0, history_length=6, track_air_time=False,
    )

    whiteboard = AssetBaseCfg(
        prim_path="/World/envs/env_.*/whiteboard_surface",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.7, 0.02),   
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.98, 0.98, 0.98), roughness=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.05),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=BOARD_ORIGIN),   
    )
    frame_top = AssetBaseCfg(prim_path="/World/envs/env_.*/whiteboard_frame_top",
        spawn=sim_utils.CuboidCfg(size=(0.54, 0.02, 0.03), visual_material=METAL_MAT, physics_material=FRAME_PROPS),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRAME_TOP_ORIGIN))
    frame_bottom = AssetBaseCfg(prim_path="/World/envs/env_.*/whiteboard_frame_bottom",
        spawn=sim_utils.CuboidCfg(size=(0.54, 0.02, 0.03), visual_material=METAL_MAT, physics_material=FRAME_PROPS),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRAME_BOTTOM_ORIGIN))
    frame_left = AssetBaseCfg(prim_path="/World/envs/env_.*/whiteboard_frame_left",
        spawn=sim_utils.CuboidCfg(size=(0.02, 0.7, 0.03), visual_material=METAL_MAT, physics_material=FRAME_PROPS),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRAME_LEFT_ORIGIN))
    frame_right = AssetBaseCfg(prim_path="/World/envs/env_.*/whiteboard_frame_right",
        spawn=sim_utils.CuboidCfg(size=(0.02, 0.7, 0.03), visual_material=METAL_MAT, physics_material=FRAME_PROPS),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRAME_RIGHT_ORIGIN))
    
    chest_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/arm_base_link_d435i_2/chest_cam", 
        update_period=0.0, 
        height=720,
        width=1280,
        data_types=["rgb"],   # 去掉距离信息，只保留 RGB
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2,          
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=CHEST_CAM_ROT), 
    )
    
    wrist_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/arm_r_link_d405/wrist_cam", 
        update_period=0.0, 
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.0,          
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.01, 10.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

@configclass
class ActionsCfg:
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["arm_r_joint_1", "arm_r_joint_2", "arm_r_joint_3", "arm_r_joint_4", "arm_r_joint_5", "arm_r_joint_6", "arm_r_joint_7"],
        body_name="arm_r_link_7", 
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )
    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_l_joint_1", "arm_l_joint_2", "arm_l_joint_3", "arm_l_joint_4", "arm_l_joint_5", "arm_l_joint_6", "arm_l_joint_7"],
        use_default_offset=True, 
    )
    right_gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_r_joint_finger"],
        use_default_offset=False, 
    )
    left_gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_l_joint_finger"],
        use_default_offset=False, 
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
    policy: PolicyCfg = PolicyCfg()

@configclass
class VLAEnvCfg(ManagerBasedEnvCfg):
    scene: WipeBoardSceneCfg = WipeBoardSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 0.01
        # Fixed 50 Hz policy/control/logging cadence: 2 physics steps at 100 Hz.
        if not math.isclose(self.decimation * self.sim.dt, TRAJECTORY_SAMPLE_DT_S, abs_tol=1.0e-12):
            raise ValueError("Whiteboard trajectory sampling must run at 50 Hz")
        if args_cli.fast_dp_deploy or args_cli.trajectory_benchmark:
            # A benchmark control action spans two physics substeps.  Rendering
            # both substeps duplicates camera work without a new observation.
            self.sim.render_interval = self.decimation

# ======================================================================
# 4. 完整字迹生成函数（全部展开，无省略）
# ======================================================================
def _cubic_bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u**3 * p0 + 3*u**2*t * p1 + 3*u*t**2 * p2 + t**3 * p3)

def _generate_stroke_points(start_xy, end_xy, ctrl_spread, num_pts, z_base, device):
    t = torch.linspace(0, 1, num_pts, device=device)
    p0 = torch.tensor([start_xy[0], start_xy[1]], device=device)
    p3 = torch.tensor([end_xy[0],   end_xy[1]],   device=device)
    mid = (p0 + p3) * 0.5
    perp = torch.tensor([-(p3[1]-p0[1]), p3[0]-p0[0]], device=device)
    perp_norm = perp / (torch.norm(perp) + 1e-8)
    offset1 = (torch.rand(1, device=device).item() - 0.5) * ctrl_spread
    offset2 = (torch.rand(1, device=device).item() - 0.5) * ctrl_spread
    along1  = 0.25 + torch.rand(1, device=device).item() * 0.15
    along2  = 0.55 + torch.rand(1, device=device).item() * 0.15
    p1 = p0 + (p3-p0)*along1 + perp_norm * offset1
    p2 = p0 + (p3-p0)*along2 + perp_norm * offset2
    t_ease = 0.5 - 0.5 * torch.cos(torch.pi * t)
    pts_2d = _cubic_bezier(
        p0.unsqueeze(0).expand(num_pts, -1),
        p1.unsqueeze(0).expand(num_pts, -1),
        p2.unsqueeze(0).expand(num_pts, -1),
        p3.unsqueeze(0).expand(num_pts, -1),
        t_ease.unsqueeze(-1)
    )
    z_wave = z_base + 0.0008 * torch.sin(torch.pi * t)
    noise_x = (torch.rand(num_pts, device=device) - 0.5) * 0.001
    noise_y = (torch.rand(num_pts, device=device) - 0.5) * 0.001
    x = pts_2d[:, 0] + noise_x
    y = pts_2d[:, 1] + noise_y
    z = z_wave + (torch.rand(num_pts, device=device) - 0.5) * 0.0005
    return torch.stack([x, y, z], dim=-1)

def _make_word(x_start, y_center, word_width, char_height, num_pts_total, z_base, device):
    num_letters = torch.randint(2, 7, (1,)).item()
    letter_width = word_width / num_letters
    pts_per_letter = num_pts_total // num_letters
    remainder = num_pts_total - pts_per_letter * num_letters
    all_pts = []
    cx = x_start
    for i in range(num_letters):
        n_pts = pts_per_letter + (1 if i < remainder else 0)
        if n_pts < 4:
            n_pts = 4
        letter_x_end = cx + letter_width * (0.6 + torch.rand(1, device=device).item() * 0.3)
        letter_y_off = (torch.rand(1, device=device).item() - 0.5) * char_height * 0.3
        letter_style = torch.randint(0, 4, (1,)).item()
        if letter_style == 0:
            spread = char_height * 0.05
        elif letter_style == 1:
            spread = char_height * 0.4
        elif letter_style == 2:
            spread = char_height * 0.6
        else:
            spread = char_height * 0.3
        start = (cx, y_center + letter_y_off)
        end   = (letter_x_end, y_center + letter_y_off + (torch.rand(1, device=device).item()-0.5)*char_height*0.4)
        stroke = _generate_stroke_points(start, end, spread, n_pts, z_base, device)
        all_pts.append(stroke)
        cx = letter_x_end + letter_width * 0.05
    word_pts = torch.cat(all_pts, dim=0)[:num_pts_total]
    if word_pts.shape[0] < num_pts_total:
        pad = word_pts[-1:].expand(num_pts_total - word_pts.shape[0], -1)
        word_pts = torch.cat([word_pts, pad], dim=0)
    return word_pts, cx

def _make_underline(x_start, x_end, y_pos, num_pts, z_base, device):
    t = torch.linspace(0, 1, num_pts, device=device)
    x = x_start + (x_end - x_start) * t
    y = y_pos + torch.sin(t * torch.pi * 4) * 0.001
    z = torch.full_like(x, z_base)
    return torch.stack([x, y, z], dim=-1)

def _make_arrow(start_xy, end_xy, num_pts, z_base, device):
    t = torch.linspace(0, 1, num_pts, device=device)
    shaft_pts = int(num_pts * 0.75)
    head_pts = num_pts - shaft_pts
    t_shaft = t[:shaft_pts]
    sx = start_xy[0] + (end_xy[0]-start_xy[0]) * t_shaft
    sy = start_xy[1] + (end_xy[1]-start_xy[1]) * t_shaft
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    length = torch.sqrt(torch.tensor(dx**2 + dy**2)).item() + 1e-6
    arrow_len = min(0.015, length * 0.3)
    import math as m
    angle = m.atan2(dy, dx)
    t_head = t[shaft_pts:]
    t_norm = t_head / (t_head[-1] + 1e-8) if len(t_head) > 0 else t_head
    half_head = head_pts // 2
    h1x = end_xy[0] - arrow_len * m.cos(angle - 0.4) * t_norm[:half_head]
    h1y = end_xy[1] - arrow_len * m.sin(angle - 0.4) * t_norm[:half_head]
    h2x = end_xy[0] - arrow_len * m.cos(angle + 0.4) * t_norm[half_head:half_head*2]
    h2y = end_xy[1] - arrow_len * m.sin(angle + 0.4) * t_norm[half_head:half_head*2]
    hx = torch.cat([h1x, h2x]) if len(h2x) > 0 else h1x
    hy = torch.cat([h1y, h2y]) if len(h2y) > 0 else h1y
    x = torch.cat([sx, hx])
    y = torch.cat([sy, hy])
    x = x[:num_pts]
    y = y[:num_pts]
    z = torch.full_like(x, z_base)
    return torch.stack([x, y, z], dim=-1)

def _make_checkbox(cx, cy, size, num_pts, z_base, device):
    import math as m
    perimeter = 4 * size
    pts_per_side = num_pts // 4
    sides = []
    corners = [
        (cx - size/2, cy - size/2),
        (cx + size/2, cy - size/2),
        (cx + size/2, cy + size/2),
        (cx - size/2, cy + size/2),
    ]
    for i in range(4):
        t = torch.linspace(0, 1, pts_per_side, device=device)
        x0, y0 = corners[i]
        x1, y1 = corners[(i+1) % 4]
        sx = x0 + (x1-x0)*t + (torch.rand(pts_per_side, device=device)-0.5)*0.001
        sy = y0 + (y1-y0)*t + (torch.rand(pts_per_side, device=device)-0.5)*0.001
        sides.append(torch.stack([sx, sy], dim=-1))
    box_pts = torch.cat(sides, dim=0)[:num_pts]
    if torch.rand(1).item() > 0.5:
        check_start = num_pts // 2
        remaining = num_pts - check_start
        t = torch.linspace(0, 1, remaining, device=device)
        chk_x = (cx - size*0.2) + (cx + size*0.4 - (cx - size*0.2)) * t
        chk_y = (cy - size*0.1) + (cy + size*0.35 - (cy - size*0.1)) * t
        t2 = torch.linspace(0, 1, check_start, device=device)
        chk_x2 = (cx - size*0.35) + (cx - size*0.2 - (cx - size*0.35)) * t2
        chk_y2 = (cy) + (cy - size*0.1 - cy) * t2
        actual_len_1 = min(check_start, box_pts.shape[0])
        box_pts[:actual_len_1, 0] = chk_x2[:actual_len_1]
        box_pts[:actual_len_1, 1] = chk_y2[:actual_len_1]
        if box_pts.shape[0] > check_start:
            actual_len_2 = min(box_pts.shape[0] - check_start, chk_x.shape[0])
            box_pts[check_start:check_start+actual_len_2, 0] = chk_x[:actual_len_2]
            box_pts[check_start:check_start+actual_len_2, 1] = chk_y[:actual_len_2]
    z = torch.full((box_pts.shape[0],), z_base, device=device)
    return torch.cat([box_pts, z.unsqueeze(-1)], dim=-1)

def _generate_realistic_shape(shape_type, cx, cy, radius, num_pts, z_base, device):
    import math as m
    t = torch.linspace(0, 1, num_pts, device=device)
    noise = (torch.rand(num_pts, device=device) - 0.5) * 0.0015
    if shape_type == 0:
        angle = 2 * torch.pi * t
        x = cx + radius * torch.cos(angle) + noise
        y = cy + radius * torch.sin(angle) + noise
    elif shape_type == 1:
        side_pts = num_pts // 4
        segments = []
        corners = [
            (cx-radius, cy-radius*0.7), (cx+radius, cy-radius*0.7),
            (cx+radius, cy+radius*0.7), (cx-radius, cy+radius*0.7)
        ]
        for i in range(4):
            ts = torch.linspace(0, 1, side_pts, device=device)
            x0, y0 = corners[i]
            x1, y1 = corners[(i+1) % 4]
            seg_x = x0 + (x1-x0)*ts + (torch.rand(side_pts, device=device)-0.5)*0.001
            seg_y = y0 + (y1-y0)*ts + (torch.rand(side_pts, device=device)-0.5)*0.001
            segments.append(torch.stack([seg_x, seg_y], dim=-1))
        rect_pts = torch.cat(segments, dim=0)[:num_pts]
        x = rect_pts[:, 0]
        y = rect_pts[:, 1]
    elif shape_type == 2:
        n_arms = torch.randint(4, 7, (1,)).item()
        angle = 2 * torch.pi * t
        r_var = radius * (0.5 + 0.5 * torch.cos(n_arms * angle))
        x = cx + r_var * torch.cos(angle) + noise
        y = cy + r_var * torch.sin(angle) + noise
    elif shape_type == 3:
        return _make_arrow(
            (cx - radius, cy), (cx + radius, cy + (torch.rand(1).item()-0.5)*radius),
            num_pts, z_base, device
        )
    elif shape_type == 4:
        half = num_pts // 2
        t1 = torch.linspace(0, 1, half, device=device)
        t2 = torch.linspace(0, 1, num_pts - half, device=device)
        x1 = cx - radius*0.5 + (cx - radius*0.1 - (cx-radius*0.5)) * t1
        y1 = cy + (cy - radius*0.4 - cy) * t1
        x2 = cx - radius*0.1 + (cx + radius*0.6 - (cx-radius*0.1)) * t2
        y2 = cy - radius*0.4 + (cy + radius*0.5 - (cy-radius*0.4)) * t2
        x = torch.cat([x1, x2])[:num_pts] + noise[:num_pts]
        y = torch.cat([y1, y2])[:num_pts] + noise[:num_pts]
    elif shape_type == 5:
        half = num_pts // 2
        t1 = torch.linspace(0, 1, half, device=device)
        t2 = torch.linspace(0, 1, num_pts - half, device=device)
        x1 = cx - radius*0.5 + 2*radius*0.5*t1
        y1 = cy - radius*0.5 + 2*radius*0.5*t1
        x2 = cx + radius*0.5 - 2*radius*0.5*t2
        y2 = cy - radius*0.5 + 2*radius*0.5*t2
        x = torch.cat([x1, x2])[:num_pts] + noise[:num_pts]
        y = torch.cat([y1, y2])[:num_pts] + noise[:num_pts]
    elif shape_type == 6:
        third = num_pts // 3
        verts = [
            (cx, cy + radius),
            (cx - radius*0.866, cy - radius*0.5),
            (cx + radius*0.866, cy - radius*0.5)
        ]
        segs = []
        for i in range(3):
            ts = torch.linspace(0, 1, third, device=device)
            x0, y0 = verts[i]
            x1, y1 = verts[(i+1) % 3]
            segs.append(torch.stack([
                x0+(x1-x0)*ts + (torch.rand(third, device=device)-0.5)*0.001,
                y0+(y1-y0)*ts + (torch.rand(third, device=device)-0.5)*0.001
            ], dim=-1))
        tri_pts = torch.cat(segs, dim=0)[:num_pts]
        x = tri_pts[:, 0]; y = tri_pts[:, 1]
    elif shape_type == 7:
        angle = 2 * torch.pi * t
        aspect = 0.5 + torch.rand(1).item() * 0.5
        x = cx + radius * torch.cos(angle) + noise
        y = cy + radius * aspect * torch.sin(angle) + noise
    elif shape_type == 8:
        return _make_checkbox(cx, cy, radius*1.5, num_pts, z_base, device)
    elif shape_type == 9:
        x = cx - radius + 2*radius * t
        y = cy + torch.sin(t * torch.pi * torch.randint(3, 8, (1,)).item()) * radius * 0.3 + noise
    else:
        x = cx + noise
        y = cy + noise
    z = torch.full_like(x, z_base)
    return torch.stack([x, y, z], dim=-1)

def _generate_parametric_shape(shape_type, cx, cy, radius, num_pts, device):
    t = torch.linspace(0, 1, num_pts, device=device)
    noise = (torch.rand(num_pts, device=device) - 0.5) * 0.002
    if shape_type == 0:
        angle = 2 * torch.pi * t
        x = cx + radius * torch.cos(angle) + noise
        y = cy + radius * torch.sin(angle) + noise
    elif shape_type == 1:
        x = cx + radius * torch.sign(torch.cos(2 * torch.pi * t)) * torch.pow(torch.abs(torch.cos(2 * torch.pi * t)), 0.2) + noise
        y = cy + radius * torch.sign(torch.sin(2 * torch.pi * t)) * torch.pow(torch.abs(torch.sin(2 * torch.pi * t)), 0.2) + noise
    elif shape_type == 2:
        angle = 2 * torch.pi * t
        r_tri = radius * (1 + 0.3 * torch.cos(3 * angle))
        x = cx + r_tri * torch.cos(angle) + noise
        y = cy + r_tri * torch.sin(angle) + noise
    z = torch.full_like(x, BOARD_Z + 0.022)
    return torch.stack([x, y, z], dim=-1)

def generate_random_scribbles(device, num_envs=1, offset_x=0.0, offset_y=0.0):
    """
    生成随机字迹，返回位置张量和指令列表（但指令将在主程序中被覆盖）
    """
    max_curves = 20
    points_per_curve = 150
    total_points = max_curves * points_per_curve
    positions = torch.zeros((num_envs, total_points, 3), device=device)
    z_base = BOARD_Z + 0.022
    
    instructions = []

    global_reg_x = (WRITING_REGION_X[0] + offset_x, WRITING_REGION_X[1] + offset_x)
    global_reg_y = (WRITING_REGION_Y[0] + offset_y, WRITING_REGION_Y[1] + offset_y)

    center_x = (global_reg_x[0] + global_reg_x[1]) / 2.0
    center_y = (global_reg_y[0] + global_reg_y[1]) / 2.0

    for env_idx in range(num_envs):
        r = torch.rand(1).item()
        if r < 1.0 / 5.5:
            style = 1
        elif r < (1.0 + 3.0) / 5.5:
            style = 2
        else:
            style = 3

        env_curves = []

        span_x = np.random.uniform(0.06, 0.12)
        span_y = np.random.uniform(0.06, 0.12)
        cx = np.random.uniform(global_reg_x[0] + span_x/2, global_reg_x[1] - span_x/2)
        cy = np.random.uniform(global_reg_y[0] + span_y/2, global_reg_y[1] - span_y/2)
        
        w_reg_x = (cx - span_x/2, cx + span_x/2)
        w_reg_y = (cy - span_y/2, cy + span_y/2)
        d_reg_x = w_reg_x
        d_reg_y = w_reg_y

        if cx < center_x - 0.02:
            pos_str = "on the left"
        elif cx > center_x + 0.02:
            pos_str = "on the right"
        else:
            pos_str = "in the center"
        
        if cy < center_y - 0.03:
            pos_str = f"at the bottom, {pos_str}"
        elif cy > center_y + 0.03:
            pos_str = f"at the top, {pos_str}"

        # 生成指令（但主程序会强制覆盖）
        if style == 1:
            instruction = f"Wipe off the random scribbles {pos_str} of the whiteboard."
        elif style == 2:
            instruction = f"Clean the text notes and writings {pos_str}."
        else:
            instruction = f"Erase the drawn geometric shapes {pos_str}."
        instructions.append(instruction)

        # ---------- 根据 style 生成字迹（以下为完整代码） ----------
        if style == 1:
            num_lines = torch.randint(1, 4, (1,)).item()
            for _ in range(num_lines):
                for attempt in range(10):
                    x0 = w_reg_x[0] + torch.rand(1).item() * (w_reg_x[1] - w_reg_x[0])
                    y0 = w_reg_y[0] + torch.rand(1).item() * (w_reg_y[1] - w_reg_y[0])
                    x1 = w_reg_x[0] + torch.rand(1).item() * (w_reg_x[1] - w_reg_x[0])
                    y1 = w_reg_y[0] + torch.rand(1).item() * (w_reg_y[1] - w_reg_y[0])
                    dist = math.hypot(x1 - x0, y1 - y0)
                    if dist > 0.03:
                        break
                else:
                    x0, y0 = (w_reg_x[0] + 0.02, w_reg_y[0] + 0.02)
                    x1, y1 = (w_reg_x[1] - 0.02, w_reg_y[1] - 0.02)
                if torch.rand(1).item() > 0.5:
                    ctrl_spread = 0.001 + torch.rand(1).item() * 0.005
                else:
                    ctrl_spread = 0.01 + torch.rand(1).item() * 0.03
                curve_pts = _generate_stroke_points((x0, y0), (x1, y1), ctrl_spread, points_per_curve, z_base, device)
                env_curves.append(curve_pts)

        elif style == 2:
            diagram_region_x = d_reg_x
            diagram_region_y = d_reg_y
            region_width  = w_reg_x[1] - w_reg_x[0]
            region_height = w_reg_y[1] - w_reg_y[0]
            layout_roll = torch.rand(1).item()
            if layout_roll < 0.15:
                layout_style = "diagram_heavy"
            elif layout_roll < 0.30:
                layout_style = "formula_notes"
            elif layout_roll < 0.45:
                layout_style = "numbered_list"
            elif layout_roll < 0.60:
                layout_style = "underlined_text"
            elif layout_roll < 0.75:
                layout_style = "quick_notes"
            elif layout_roll < 0.90:
                layout_style = "cursive_flow"
            else:
                layout_style = "printed_text"
            line_height = 0.035 + torch.rand(1).item() * 0.015
            char_height = line_height * (0.5 + torch.rand(1).item() * 0.3)
            baseline_tilt = (torch.rand(1).item() - 0.5) * 0.02
            baseline_wave_amp = 0.002 + torch.rand(1).item() * 0.003
            baseline_wave_freq = torch.randint(2, 6, (1,)).item()
            used_curves = 0
            num_shapes = 0
            shape_centers = []
            if layout_style == "diagram_heavy":
                num_shapes = torch.randint(4, 8, (1,)).item()
            elif layout_style in ("formula_notes", "numbered_list"):
                num_shapes = torch.randint(0, 3, (1,)).item()
            else:
                if torch.rand(1).item() > 0.5:
                    num_shapes = torch.randint(1, 3, (1,)).item()
            for i in range(num_shapes):
                shape_type = torch.randint(0, 10, (1,)).item()
                scx = diagram_region_x[0] + torch.rand(1).item() * (diagram_region_x[1] - diagram_region_x[0])
                scy = diagram_region_y[0] + torch.rand(1).item() * (diagram_region_y[1] - diagram_region_y[0])
                sr  = 0.015 + torch.rand(1).item() * 0.025
                shape_centers.append((scx, scy, sr))
                if used_curves < max_curves:
                    shape_pts = _generate_realistic_shape(shape_type, scx, scy, sr, points_per_curve, z_base, device)
                    env_curves.append(shape_pts)
                    used_curves += 1
            remaining_curves = max_curves - used_curves
            if remaining_curves <= 0:
                remaining_curves = 1
            start_y = w_reg_y[1] - torch.rand(1).item() * 0.03
            num_lines = max(1, min(remaining_curves, int(region_height / line_height)))
            for line_idx in range(num_lines):
                if used_curves >= max_curves:
                    break
                row_y_base = start_y - line_idx * line_height
                if row_y_base < w_reg_y[0] + 0.01:
                    break
                skip_line = False
                for (sx, sy, sr) in shape_centers:
                    if abs(row_y_base - sy) < sr + line_height * 0.5:
                        if torch.rand(1).item() > 0.3:
                            skip_line = True
                            break
                if skip_line:
                    continue
                if layout_style == "numbered_list":
                    line_x_start = w_reg_x[0] + 0.015
                    if line_idx < 5 and torch.rand(1).item() > 0.3:
                        bullet_x = w_reg_x[0] + 0.005
                        bullet_y = row_y_base + baseline_tilt * (bullet_x - w_reg_x[0])
                        bullet_pts = _generate_realistic_shape(torch.randint(0, 3, (1,)).item(), bullet_x, bullet_y, 0.005, points_per_curve, z_base, device)
                        if used_curves < max_curves:
                            env_curves.append(bullet_pts)
                            used_curves += 1
                elif layout_style == "formula_notes":
                    line_x_start = w_reg_x[0] + torch.rand(1).item() * 0.02
                else:
                    line_x_start = w_reg_x[0] + torch.rand(1).item() * 0.01
                line_x_end = w_reg_x[1] - torch.rand(1).item() * 0.02
                line_width = line_x_end - line_x_start
                if layout_style == "cursive_flow":
                    words_in_line = torch.randint(1, 3, (1,)).item()
                elif layout_style == "quick_notes":
                    words_in_line = torch.randint(3, 6, (1,)).item()
                elif layout_style == "printed_text":
                    words_in_line = torch.randint(4, 8, (1,)).item()
                else:
                    words_in_line = torch.randint(2, 5, (1,)).item()
                word_gap = 0.008 + torch.rand(1).item() * 0.005
                total_gap = word_gap * (words_in_line - 1)
                usable_width = line_width - total_gap
                if usable_width < 0.02:
                    usable_width = 0.02
                word_widths = []
                remaining_w = usable_width
                for w in range(words_in_line):
                    if w == words_in_line - 1:
                        ww = remaining_w
                    else:
                        ww = remaining_w / (words_in_line - w) * (0.6 + torch.rand(1).item() * 0.8)
                        ww = min(ww, remaining_w - (words_in_line - w - 1) * 0.01)
                        ww = max(ww, 0.01)
                    word_widths.append(ww)
                    remaining_w -= ww
                cx = line_x_start
                for w_idx in range(words_in_line):
                    if used_curves >= max_curves:
                        break
                    ww = word_widths[w_idx]
                    y_tilt = baseline_tilt * (cx - w_reg_x[0])
                    y_wave = baseline_wave_amp * math.sin(baseline_wave_freq * math.pi * (cx - w_reg_x[0]) / region_width)
                    current_y = row_y_base + y_tilt + y_wave
                    if layout_style == "formula_notes" and torch.rand(1).item() > 0.7:
                        curve_pts = _generate_stroke_points((cx, current_y), (cx + ww, current_y + (torch.rand(1).item()-0.5) * char_height * 0.8), char_height * 0.5, points_per_curve, z_base, device)
                    else:
                        curve_pts, _ = _make_word(cx, current_y, ww, char_height, points_per_curve, z_base, device)
                    env_curves.append(curve_pts)
                    used_curves += 1
                    cx += ww + word_gap
            if layout_style == "underlined_text" and used_curves < max_curves:
                ul_y = start_y + line_height * 0.3
                ul_pts = _make_underline(w_reg_x[0], w_reg_x[0] + region_width * 0.7, ul_y, points_per_curve, z_base, device)
                env_curves.append(ul_pts)
                used_curves += 1

        else:  # style == 3
            diagram_region_x = d_reg_x
            diagram_region_y = d_reg_y
            is_diagram_mode = torch.rand(1).item() > 0.5
            if is_diagram_mode:
                num_shapes = torch.randint(2, 5, (1,)).item()
                for i in range(num_shapes):
                    shape_type = torch.randint(0, 3, (1,)).item()
                    cx = diagram_region_x[0] + torch.rand(1).item() * (diagram_region_x[1] - diagram_region_x[0])
                    cy = diagram_region_y[0] + torch.rand(1).item() * (diagram_region_y[1] - diagram_region_y[0])
                    radius = torch.rand(1).item() * 0.03 + 0.02
                    curve_pts = _generate_parametric_shape(shape_type, cx, cy, radius, points_per_curve, device)
                    env_curves.append(curve_pts)
                remaining_curves = max_curves - num_shapes
            else:
                remaining_curves = torch.randint(10, max_curves + 1, (1,)).item()
            for i in range(remaining_curves):
                x_center = w_reg_x[0] + torch.rand(1).item() * (w_reg_x[1] - w_reg_x[0])
                y_pos = w_reg_y[0] + torch.rand(1).item() * (w_reg_y[1] - w_reg_y[0])
                Ax = 0.005 + torch.rand(1).item() * 0.015
                kx = torch.randint(10, 25, (1,), device=device)
                t = torch.linspace(0, 1, points_per_curve, device=device)
                x = torch.full_like(t, x_center) + Ax * torch.sin(kx.float() * torch.pi * t)
                y = torch.full_like(t, y_pos) + (torch.rand(1).item() - 0.5) * 0.005
                z = torch.full_like(x, BOARD_Z + 0.022)
                env_curves.append(torch.stack([x, y, z], dim=-1))

        # ---------- 曲线对齐（保持不变） ----------
        aligned_curves = []
        for curve in env_curves:
            n = curve.shape[0]
            if n == points_per_curve:
                aligned_curves.append(curve)
            elif n > points_per_curve:
                indices = torch.linspace(0, n - 1, points_per_curve, device=device, dtype=torch.long)
                aligned_curves.append(curve[indices])
            else:
                pad = curve[-1:].expand(points_per_curve - n, -1)
                aligned_curves.append(torch.cat([curve, pad], dim=0))
        while len(aligned_curves) < max_curves:
            invisible = torch.tensor([0.0, 0.0, -10.0], device=device).repeat(points_per_curve, 1)
            aligned_curves.append(invisible)
        aligned_curves = aligned_curves[:max_curves]
        positions[env_idx] = torch.cat(aligned_curves, dim=0)

    return positions, instructions

# ======================================================================
# 5. WebSocket 通信桥梁
# ======================================================================
class IsaacWebSocketBridge:
    def __init__(self, uri="ws://127.0.0.1:5556"):
        self.uri = uri
        self.websocket = None
        self.policy_type = None
        self.policy_action_dim = None
        self.policy_state_dim = None
        self.metadata = {}
        self.latest_osc_action = None
        self.current_cmd = "idle"
        self.action_queue = []
        self.last_round_trip_ms = None
        self.last_server_timing = None
        self.packer = Packer()
        print(f"[INFO]: WebSocket Client 准备连接至: {self.uri}")

    def connect_to_server(self, simulation_app, seed=None):
        while simulation_app.is_running():
            try:
                self.websocket = connect(self.uri, max_size=None)
                metadata_msg = self.websocket.recv()
                metadata = unpackb(metadata_msg)
                if isinstance(metadata, dict):
                    self.metadata = metadata
                    self.policy_type = metadata.get("policy_type")
                    try:
                        self.policy_action_dim = int(metadata["action_dim"])
                    except (KeyError, TypeError, ValueError):
                        self.policy_action_dim = None
                    try:
                        self.policy_state_dim = int(metadata["state_dim"])
                    except (KeyError, TypeError, ValueError):
                        self.policy_state_dim = None
                else:
                    self.metadata = {}
                    self.policy_type = None
                    self.policy_action_dim = None
                    self.policy_state_dim = None
                print(f"[INFO]: 成功连接到 VLA ！Metadata: {metadata}")

                reset_payload = {"reset": True, "robo_name": "wipe_board"}
                if seed is not None:
                    reset_payload["seed"] = int(seed)
                self.websocket.send(self.packer.pack(reset_payload))
                reset_resp = self.websocket.recv()
                if isinstance(reset_resp, str):
                    print(f"[ERROR] Reset 失败: {reset_resp}")
                    return False
                reset_result = unpackb(reset_resp)
                print(f"[INFO]: VLA  Reset 完成: {reset_result}")
                return True
            except Exception as e:
                print(f"[WARN]: 连接失败 ({e})...")
                simulation_app.update()
                time.sleep(1)
        return False

    def wait_for_action_and_reply(self, state_dict, simulation_app):
        if self.websocket is None:
            return

        self.last_round_trip_ms = None
        self.last_server_timing = None
        if len(self.action_queue) > 0:
            self.latest_osc_action = self.action_queue.pop(0)
            self.current_cmd = 'step'
            return

        try:
            payload = {
                "observation.images.primary": state_dict["rgb_primary_raw"],
                "observation.images.wrist": state_dict["rgb_wrist_raw"],
                "observation.state": state_dict["policy_state"],
                "task": state_dict["instruction"],
            }
            packed_data = self.packer.pack(payload)
            request_started = time.perf_counter()
            self.websocket.send(packed_data)

            response = self.websocket.recv()
            self.last_round_trip_ms = (time.perf_counter() - request_started) * 1000.0
            if isinstance(response, str):
                print(f"[ERROR] 服务端返回错误: {response}")
                self.current_cmd = 'idle'
                return

            resp_dict = unpackb(response)
            self.last_server_timing = resp_dict.get("server_timing")
            if "action" in resp_dict and resp_dict["action"] is not None:
                actions = np.asarray(resp_dict["action"])
                if len(actions.shape) == 2:
                    self.action_queue = [actions[i] for i in range(actions.shape[0])]
                else:
                    self.action_queue = [actions]
                self.latest_osc_action = self.action_queue.pop(0)
                self.current_cmd = 'step'
            else:
                self.current_cmd = 'idle'
        except Exception as e:
            print(f"[ERROR]: 通信异常: {e}")
            import traceback
            traceback.print_exc()
            self.current_cmd = 'idle'

    def close(self):
        if self.websocket:
            self.websocket.close()

# ======================================================================
# 6. 专家控制器
# ======================================================================
class ExpertController:
    def __init__(self, initial_pos, initial_quat):
        self.cmd_pos = initial_pos.copy()
        self.cmd_quat = initial_quat.copy()
        self.filtered_force = 0.0
        self.prev_error = 0.0
        
        self.state = 0 
        self.radar_radius = 0.12
        self.max_step_dist = 0.00252
        self.surface_z_est = SURFACE_Z_EST
        self.z_hover = Z_HOVER
        self.target_force = 6.0
        self.force_kp = 0.00015
        self.force_kd = 0.00005
        self.max_z_speed = 0.0015
        self.deadband = 1.0
        
        self.path_queue = []

    def plan_raster_path(self, dirt_points_2d):
        if len(dirt_points_2d) == 0:
            return []
            
        clustering = DBSCAN(eps=0.04, min_samples=3).fit(dirt_points_2d)
        labels = clustering.labels_
        
        best_cluster = dirt_points_2d
        min_dist = float('inf')
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1: continue
            cluster_pts = dirt_points_2d[labels == label]
            center = np.mean(cluster_pts, axis=0)
            dist = np.linalg.norm(center - self.cmd_pos[:2])
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster_pts
        
        if len(best_cluster) == 0:
            best_cluster = dirt_points_2d

        min_x, max_x = np.min(best_cluster[:, 0]), np.max(best_cluster[:, 0])
        wipe_band_width = 0.035
        x_bands = np.arange(min_x, max_x + wipe_band_width, wipe_band_width)
        
        path = []
        left_to_right = False
        for x in x_bands:
            band_mask = (best_cluster[:, 0] >= x - wipe_band_width) & (best_cluster[:, 0] < x + wipe_band_width)
            if not np.any(band_mask): continue
            
            band_pts = best_cluster[band_mask]
            min_y = np.min(band_pts[:, 1]) - 0.015 
            max_y = np.max(band_pts[:, 1]) + 0.015 
            
            if not left_to_right:
                path.append(np.array([x, min_y]))
                path.append(np.array([x, max_y]))
            else:
                path.append(np.array([x, max_y]))
                path.append(np.array([x, min_y]))
                
            left_to_right = not left_to_right
            
        return path

    def step(self, current_ee_pos, wrench_force_z, dirt_points_2d):
        if dirt_points_2d is None or len(dirt_points_2d) == 0:
            return np.concatenate([self.cmd_pos, self.cmd_quat])

        if self.state == 0:
            if len(self.path_queue) == 0:
                self.path_queue = self.plan_raster_path(dirt_points_2d)
                if len(self.path_queue) == 0:
                    return np.concatenate([self.cmd_pos, self.cmd_quat])
                    
            target_xy = self.path_queue[0]
            move_vec = target_xy - self.cmd_pos[:2]
            dist_xy = np.linalg.norm(move_vec)
            
            if self.cmd_pos[2] < self.z_hover:
                self.cmd_pos[2] += 0.005 
                self.cmd_pos[2] = min(self.cmd_pos[2], self.z_hover)
            
            if dist_xy > 0.01:
                step_xy = (move_vec / dist_xy) * min(dist_xy, 0.02)
                self.cmd_pos[0] += step_xy[0]
                self.cmd_pos[1] += step_xy[1]
            else:
                self.cmd_pos[0] = target_xy[0]
                self.cmd_pos[1] = target_xy[1]
                if self.cmd_pos[2] >= self.z_hover - 0.001:
                    self.state = 1
                    self.filtered_force = 0.0
                    
        elif self.state == 1:
            raw_force = abs(wrench_force_z)
            self.filtered_force = 0.5 * raw_force + 0.5 * self.filtered_force
            if self.filtered_force >= 3.0 or self.cmd_pos[2] <= (self.surface_z_est - 0.04):
                self.state = 2
                self.prev_error = self.filtered_force - self.target_force
            else:
                self.cmd_pos[2] -= 0.002
                
        elif self.state == 2:
            if len(self.path_queue) == 0:
                self.state = 0
                self.cmd_pos[2] = self.z_hover
                return np.concatenate([self.cmd_pos, self.cmd_quat])
                
            target_xy = self.path_queue[0]
            move_vec = target_xy - self.cmd_pos[:2]
            dist_xy = np.linalg.norm(move_vec)
            
            if dist_xy > self.max_step_dist:
                move_vec = (move_vec / dist_xy) * self.max_step_dist
                self.cmd_pos[0] += move_vec[0]
                self.cmd_pos[1] += move_vec[1]
            else:
                self.cmd_pos[0] = target_xy[0]
                self.cmd_pos[1] = target_xy[1]
                self.path_queue.pop(0)
                
                if len(self.path_queue) == 0:
                    self.state = 0

            raw_force = abs(wrench_force_z)
            self.filtered_force = 0.6 * raw_force + 0.4 * self.filtered_force
            force_error = self.filtered_force - self.target_force
            delta_error = force_error - self.prev_error

            if abs(force_error) < self.deadband:
                delta_z = 0.0
            else:
                delta_z = self.force_kp * force_error + self.force_kd * delta_error
                delta_z = np.clip(delta_z, -self.max_z_speed, self.max_z_speed)
            self.cmd_pos[2] += delta_z
            self.cmd_pos[2] = np.clip(self.cmd_pos[2], self.surface_z_est - 0.04, self.surface_z_est + 0.03)
            self.prev_error = force_error

        return np.concatenate([self.cmd_pos, self.cmd_quat])

# ======================================================================
# 7. 主程序
# ======================================================================
def compress_images(pending_primary, pending_wrist):
    """将待处理的图像Tensor批量转numpy并压缩为JPEG（移除深度处理）"""
    primary_jpgs = []
    wrist_jpgs = []
    for tensor in pending_primary:
        img_rgb = tensor.cpu().numpy()[..., :3]
        img_bgr = img_rgb[..., ::-1]
        _, jpg = cv2.imencode('.jpg', img_bgr)
        primary_jpgs.append(np.frombuffer(jpg, dtype=np.uint8))
    for tensor in pending_wrist:
        img_rgb = tensor.cpu().numpy()[..., :3]
        img_bgr = img_rgb[..., ::-1]
        _, jpg = cv2.imencode('.jpg', img_bgr)
        wrist_jpgs.append(np.frombuffer(jpg, dtype=np.uint8))
    return primary_jpgs, wrist_jpgs


# =====================================================================
# 7. Diagonal-stroke erasing benchmark
# =====================================================================
def _minimum_jerk_position(start, end, time_s, duration_s):
    """Fifth-order point-to-point reference with zero end velocity/acceleration."""
    u = np.clip(float(time_s) / duration_s, 0.0, 1.0)
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start + blend * (end - start)


def _minimum_jerk_blend(time_s, duration_s):
    """Minimum-jerk scalar blend shared by Cartesian and joint-space references."""
    u = np.clip(float(time_s) / duration_s, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _arc_geometry(start, end, center_xy, direction):
    """Return the signed planar arc from ``start`` to ``end`` around ``center_xy``."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    center_xy = np.asarray(center_xy, dtype=np.float64)
    start_vector = start[:2] - center_xy
    end_vector = end[:2] - center_xy
    start_radius = float(np.linalg.norm(start_vector))
    end_radius = float(np.linalg.norm(end_vector))
    if start_radius < 1.0e-6 or end_radius < 1.0e-6:
        raise ValueError("Circular-arc start and end must not coincide with the arc centre")
    if not math.isclose(start_radius, end_radius, abs_tol=1.0e-5, rel_tol=1.0e-4):
        raise ValueError(
            "Circular-arc start and end must have the same radius from "
            f"--trajectory_benchmark_arc_center; got {start_radius:.6f} and {end_radius:.6f} m."
        )
    start_angle = math.atan2(start_vector[1], start_vector[0])
    end_angle = math.atan2(end_vector[1], end_vector[0])
    sweep = end_angle - start_angle
    if direction == "ccw":
        if sweep <= 0.0:
            sweep += 2.0 * math.pi
    else:
        if sweep >= 0.0:
            sweep -= 2.0 * math.pi
    if abs(sweep) < 1.0e-5:
        raise ValueError("Circular-arc start and end must define a non-zero sweep")
    return start_radius, start_angle, sweep


def _circular_arc_position(start, end, center_xy, direction, time_s, duration_s):
    """Minimum-jerk progress along a fixed world-frame circular arc."""
    radius, start_angle, sweep = _arc_geometry(start, end, center_xy, direction)
    blend = _minimum_jerk_blend(time_s, duration_s)
    angle = start_angle + sweep * blend
    position = np.empty(3, dtype=np.float64)
    position[:2] = np.asarray(center_xy, dtype=np.float64) + radius * np.array(
        [math.cos(angle), math.sin(angle)]
    )
    position[2] = float(start[2] + blend * (end[2] - start[2]))
    return position


def _circular_arc_marker_points(start, end, center_xy, direction, num_points, z_base, device):
    """Create visible arc-marker positions; marker Z is the board surface, not TCP Z."""
    radius, start_angle, sweep = _arc_geometry(start, end, center_xy, direction)
    phase = torch.linspace(0.0, 1.0, num_points, device=device)
    angle = start_angle + sweep * phase
    center = torch.tensor(center_xy, device=device, dtype=torch.float32)
    xy = center + radius * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    z = torch.full((num_points, 1), z_base, device=device, dtype=torch.float32)
    return torch.cat([xy, z], dim=-1)


def _curve_position(start, end, time_s, duration_s):
    """Smooth S-curve handwriting path with fixed A/B and zero endpoint slope."""
    u = float(np.clip(time_s / max(duration_s, 1.0e-9), 0.0, 1.0))
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    position = np.asarray(start, dtype=np.float64) + blend * (
        np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    )
    position[1] += 0.024 * math.sin(2.0 * math.pi * u)
    return position


def _curve_marker_points(start, end, num_points, z_base, device):
    """Create visible markers for the common smooth S-shaped handwriting curve."""
    phase = torch.linspace(0.0, 1.0, num_points, device=device)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    start_t = torch.tensor(start, device=device, dtype=torch.float32)
    end_t = torch.tensor(end, device=device, dtype=torch.float32)
    points = start_t + blend[:, None] * (end_t - start_t)
    points[:, 1] += 0.024 * torch.sin(2.0 * math.pi * phase)
    points[:, 2] = z_base
    return points


class JointBoundaryProjector:
    """Common joint-space boundary and derivative constraint projector.

    MOVJ/MOVL receive a minimum-jerk boundary reference. Learned policies
    retain their own TCP commands and use this class only for the common joint
    acceleration and jerk limits; they must reach B within the time limit on
    their own, rather than being forced to a fixed completion time.
    """

    def __init__(self, start, end, duration_s, dt, velocity_limit, acceleration_limit, jerk_limit):
        self.start = start.clone()
        self.end = end.clone()
        self.duration_s = float(duration_s)
        self.dt = float(dt)
        # Leave tracking margin for the implicit PD actuator and contact dynamics.
        self.velocity_limit = float(velocity_limit)
        self.acceleration_limit = 0.35 * float(acceleration_limit)
        self.jerk_limit = 0.25 * float(jerk_limit)
        self.position = start.clone()
        self.velocity = torch.zeros_like(start)
        self.acceleration = torch.zeros_like(start)

    @staticmethod
    def minimum_feasible_duration(start, end, velocity_limit, acceleration_limit, jerk_limit):
        delta = float(torch.max(torch.abs(end - start)).item())
        if delta <= 1.0e-9:
            return 0.0
        # Peak values for q(t)=q0+(q1-q0)(10u^3-15u^4+6u^5).
        by_velocity = 1.875 * delta / velocity_limit
        by_acceleration = math.sqrt(5.7735026919 * delta / (0.35 * acceleration_limit))
        by_jerk = (60.0 * delta / (0.25 * jerk_limit)) ** (1.0 / 3.0)
        return max(by_velocity, by_acceleration, by_jerk)

    def reference(self, time_s):
        return self.start + _minimum_jerk_blend(time_s, self.duration_s) * (self.end - self.start)

    def project(self, desired, time_s, learned):
        reference = self.reference(time_s)
        if learned:
            # ACT/DP outputs are not wrapped in MOVJ/MOVL. Only their joint
            # derivatives are constrained here.
            desired = self.position if desired is None else desired
        else:
            desired = reference if desired is None else desired

        desired_velocity = (desired - self.position) / self.dt
        desired_acceleration = (desired_velocity - self.velocity) / self.dt
        max_acceleration_change = self.jerk_limit * self.dt
        acceleration = torch.clamp(
            desired_acceleration,
            self.acceleration - max_acceleration_change,
            self.acceleration + max_acceleration_change,
        )
        acceleration = torch.clamp(acceleration, -self.acceleration_limit, self.acceleration_limit)
        velocity = torch.clamp(
            self.velocity + acceleration * self.dt,
            -self.velocity_limit,
            self.velocity_limit,
        )
        position = self.position + velocity * self.dt
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        return position, reference


def _to_builtin(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def _pose_wxyz_to_matrix(position_m, quaternion_wxyz):
    """Build a homogeneous transform from Isaac Lab's ``(w, x, y, z)`` pose."""
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _working_contact_position(flange_position_m):
    """Return the effective eraser contact center on the horizontal board.

    The articulated link is deliberately held above the board because the
    eraser cloth is mounted below it.  The benchmark's contact/erase model is
    defined by this projected XY center on the board plane.
    """
    position = np.asarray(flange_position_m, dtype=np.float64).reshape(3).copy()
    position[2] = WORKING_TCP_Z
    return position


def _working_contact_transform(flange_transform):
    """Map a flange pose to the reported eraser-contact working TCP pose."""
    transform = np.asarray(flange_transform, dtype=np.float64).copy()
    transform[:3, 3] = _working_contact_position(transform[:3, 3])
    return transform


def _capture_normal_trajectory_sample(
    robot, ee_idx, right_joint_indices, time_s, command_pose_world, net_force, initial_dirt_count,
    remaining_stroke_points, policy_round_trip_ms=None, server_infer_ms=None,
):
    """Capture one 50 Hz sample from the normal ACT WebSocket wiping loop."""
    base_position_w = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    base_quaternion_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
    flange_position_w = robot.data.body_pos_w[0, ee_idx].detach().cpu().numpy().copy()
    flange_quaternion_wxyz = robot.data.body_quat_w[0, ee_idx].detach().cpu().numpy().copy()
    world_T_base = _pose_wxyz_to_matrix(base_position_w, base_quaternion_wxyz)
    world_T_flange = _pose_wxyz_to_matrix(flange_position_w, flange_quaternion_wxyz)
    base_T_flange = np.linalg.solve(world_T_base, world_T_flange)
    world_T_tcp = _working_contact_transform(world_T_flange)
    base_T_tcp = np.linalg.solve(world_T_base, world_T_tcp)
    command_pose_world = np.asarray(command_pose_world, dtype=np.float64).reshape(7)
    world_T_command_flange = _pose_wxyz_to_matrix(command_pose_world[:3], command_pose_world[3:])
    base_T_command_flange = np.linalg.solve(world_T_base, world_T_command_flange)
    world_T_command_tcp = _working_contact_transform(world_T_command_flange)
    base_T_command_tcp = np.linalg.solve(world_T_base, world_T_command_tcp)
    force_vector = np.asarray(net_force, dtype=np.float64).reshape(3)
    return {
        "time_s": float(time_s),
        "world_T_base": world_T_base,
        "flange_pose_base": {
            "position_m": base_T_flange[:3, 3],
            "quaternion_wxyz": R.from_matrix(base_T_flange[:3, :3]).as_quat()[[3, 0, 1, 2]],
            "matrix": base_T_flange,
        },
        "flange_pose_world": {
            "position_m": flange_position_w,
            "quaternion_wxyz": flange_quaternion_wxyz,
            "matrix": world_T_flange,
        },
        "tcp_pose_base": {
            "position_m": base_T_tcp[:3, 3],
            "quaternion_wxyz": R.from_matrix(base_T_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]],
            "matrix": base_T_tcp,
        },
        "tcp_pose_world": {
            "position_m": world_T_tcp[:3, 3],
            "quaternion_wxyz": flange_quaternion_wxyz,
            "matrix": world_T_tcp,
        },
        "command_flange_pose_base": {
            "position_m": base_T_command_flange[:3, 3],
            "quaternion_wxyz": R.from_matrix(base_T_command_flange[:3, :3]).as_quat()[[3, 0, 1, 2]],
            "matrix": base_T_command_flange,
        },
        "command_tcp_pose_base": {
            "position_m": base_T_command_tcp[:3, 3],
            "quaternion_wxyz": R.from_matrix(base_T_command_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]],
            "matrix": base_T_command_tcp,
        },
        "command_flange_position_world_m": command_pose_world[:3],
        "joint_position_rad": robot.data.joint_pos[0, right_joint_indices].detach().cpu().numpy().copy(),
        "joint_velocity_rad_s": robot.data.joint_vel[0, right_joint_indices].detach().cpu().numpy().copy(),
        "joint_acc_rad_s2": robot.data.joint_acc[0, right_joint_indices].detach().cpu().numpy().copy(),
        "joint_torque_nm": robot.data.applied_torque[0, right_joint_indices].detach().cpu().numpy().copy(),
        "contact_force_vector_n": force_vector,
        "contact_force_n": float(abs(force_vector[2])),
        "remaining_stroke_points": int(remaining_stroke_points),
        "erased_ratio": float(
            1.0 - remaining_stroke_points / initial_dirt_count
            if initial_dirt_count > 0 else 1.0
        ),
        "policy_round_trip_ms": policy_round_trip_ms,
        "server_infer_ms": server_infer_ms,
    }


def _append_trajectory_derivatives(samples):
    """Add base-frame TCP and joint finite-difference derivatives in-place."""
    if not samples:
        return

    times = np.asarray([sample["time_s"] for sample in samples], dtype=np.float64)

    def finite_difference(values):
        values = np.asarray(values, dtype=np.float64)
        derivative = np.zeros_like(values)
        if len(values) > 1:
            dt = np.diff(times)
            if np.any(dt <= 0.0):
                raise ValueError("Trajectory sample times must be strictly increasing")
            derivative[1:] = np.diff(values, axis=0) / dt.reshape((-1,) + (1,) * (values.ndim - 1))
        return derivative

    tcp_position_base = np.asarray(
        [sample["tcp_pose_base"]["position_m"] for sample in samples], dtype=np.float64
    )
    tcp_velocity = finite_difference(tcp_position_base)
    tcp_acceleration = finite_difference(tcp_velocity)
    tcp_jerk = finite_difference(tcp_acceleration)
    joint_acceleration = np.asarray([sample["joint_acc_rad_s2"] for sample in samples], dtype=np.float64)
    joint_jerk = finite_difference(joint_acceleration)

    for index, sample in enumerate(samples):
        sample["tcp_linear_velocity_base_m_s"] = tcp_velocity[index]
        sample["tcp_linear_acceleration_base_m_s2"] = tcp_acceleration[index]
        sample["tcp_linear_jerk_base_m_s3"] = tcp_jerk[index]
        sample["joint_jerk_rad_s3"] = joint_jerk[index]


def _trajectory_benchmark_metrics(
    samples, start, end, reference_duration_s, completion_time_limit_s,
    endpoint_tolerance, acceleration_limit, jerk_limit, completed, require_endpoint=True,
    reference_kind="line", arc_center_xy=None, arc_direction="cw",
):
    """Score one trial against a line or circular-arc reference."""
    window = samples
    metrics = {
        "samples_total": int(len(samples)),
        "measurement_samples": int(len(window)),
        "simulated_duration_s": None,
        "completion_time_s": None,
        "final_erased_ratio": None,
        "completion_time_limit_exceeded": None,
        "wall_duration_s": None,
        "endpoint_error_m": None,
        "trajectory_rmse_m": None,
        "trajectory_max_error_m": None,
        "max_joint_torque_nm": None,
        "max_joint_acceleration_rad_s2": None,
        "max_joint_jerk_rad_s3": None,
        "joint_jerk_rms_rad_s3": None,
        "ee_jerk_rms_m_s3": None,
        "command_jitter_rms_m_per_step2": None,
        "policy_round_trip_mean_ms": None,
        "policy_round_trip_p95_ms": None,
        "policy_round_trip_p99_ms": None,
        "policy_round_trip_max_ms": None,
        "server_infer_mean_ms": None,
        "server_infer_p95_ms": None,
        "server_infer_p99_ms": None,
        "server_infer_max_ms": None,
    }
    if not window:
        return metrics

    absolute_times = np.asarray([sample["time_s"] for sample in window], dtype=np.float64)
    times = absolute_times - absolute_times[0]
    wall_times = np.asarray([sample["wall_time_s"] for sample in window], dtype=np.float64)
    base_frame_samples = "tcp_pose_base" in window[0]
    if base_frame_samples:
        reference_z_world = float(start[2])
        actual = np.asarray(
            [sample["tcp_pose_base"]["position_m"] for sample in window], dtype=np.float64
        )
        base_T_world = np.linalg.inv(np.asarray(window[0]["world_T_base"], dtype=np.float64))
        start_h = base_T_world @ np.r_[start, 1.0]
        end_h = base_T_world @ np.r_[end, 1.0]
        start = start_h[:3]
        end = end_h[:3]
        if arc_center_xy is not None:
            center_h = base_T_world @ np.r_[arc_center_xy, reference_z_world, 1.0]
            arc_center_xy = center_h[:2]
        commanded = np.asarray(
            [sample["command_tcp_pose_base"]["position_m"] for sample in window], dtype=np.float64
        )
    else:
        actual = np.asarray([sample["ee_position_m"] for sample in window], dtype=np.float64)
        commanded = np.asarray([sample["command_position_m"] for sample in window], dtype=np.float64)
    joint_acc = np.asarray([sample["joint_acc_rad_s2"] for sample in window], dtype=np.float64)
    joint_torque = np.asarray([sample["joint_torque_nm"] for sample in window], dtype=np.float64)

    # Learned methods may advance along the common geometry at a different
    # policy-guided rate.  Score each TCP against the reference phase that was
    # actually commanded, not against an artificial shared clock.
    reference_times = np.asarray(
        [sample.get("reference_time_s", time_s) for sample, time_s in zip(window, times)],
        dtype=np.float64,
    )
    reference_times = np.minimum(reference_times, reference_duration_s)
    if reference_kind == "curve":
        reference = np.asarray(
            [_curve_position(start, end, time_s, reference_duration_s) for time_s in reference_times],
            dtype=np.float64,
        )
    elif reference_kind == "arc":
        reference = np.asarray(
            [
                _circular_arc_position(start, end, arc_center_xy, arc_direction, time_s, reference_duration_s)
                for time_s in reference_times
            ],
            dtype=np.float64,
        )
    else:
        reference = np.asarray(
            [_minimum_jerk_position(start, end, time_s, reference_duration_s) for time_s in reference_times],
            dtype=np.float64,
        )
    tracking_error = np.linalg.norm(actual - reference, axis=1)
    start_error = np.linalg.norm(actual[0] - start)
    endpoint_error = np.linalg.norm(actual[-1] - end) if require_endpoint else None

    dt = float(np.median(np.diff(absolute_times))) if len(absolute_times) > 1 else 0.0
    joint_jerk_rms = 0.0
    max_joint_jerk = 0.0
    ee_jerk_rms = 0.0
    if len(joint_acc) > 1 and dt > 0.0:
        joint_jerk = np.diff(joint_acc, axis=0) / dt
        joint_jerk_rms = float(np.sqrt(np.mean(np.square(joint_jerk))))
        max_joint_jerk = float(np.max(np.abs(joint_jerk)))
    if len(actual) > 3 and dt > 0.0:
        ee_jerk_rms = float(np.sqrt(np.mean(np.square(np.diff(actual, n=3, axis=0) / dt**3))))
    command_jitter_rms = (
        float(np.sqrt(np.mean(np.square(np.diff(commanded, n=2, axis=0)))))
        if len(commanded) > 2
        else 0.0
    )
    request_round_trip_ms = [
        sample["policy_round_trip_ms"]
        for sample in window
        if sample["policy_round_trip_ms"] is not None
    ]
    server_infer_ms = [
        sample["server_infer_ms"] for sample in window if sample["server_infer_ms"] is not None
    ]
    metrics.update(
        {
            "control_dt_s": dt,
            "trajectory_frame": "robot_base" if base_frame_samples else "world",
            "reference_kind": reference_kind,
            "simulated_duration_s": float(times[-1]),
            "completion_time_s": float(times[-1]) if completed else None,
            "final_erased_ratio": float(window[-1].get("erased_ratio", 0.0)),
            "completion_time_limit_exceeded": bool(
                completion_time_limit_s is not None
                and (not completed or times[-1] > completion_time_limit_s)
            ),
            "wall_duration_s": float(wall_times[-1] - wall_times[0]),
            "endpoint_error_m": float(endpoint_error) if endpoint_error is not None else None,
            "start_tolerance_exceeded": bool(start_error > endpoint_tolerance),
            "endpoint_tolerance_exceeded": bool(
                require_endpoint and endpoint_error > endpoint_tolerance
            ),
            "trajectory_rmse_m": float(np.sqrt(np.mean(np.square(tracking_error)))),
            "trajectory_max_error_m": float(np.max(tracking_error)),
            "max_joint_torque_nm": float(np.max(np.abs(joint_torque))),
            "max_joint_acceleration_rad_s2": float(np.max(np.abs(joint_acc))),
            "max_joint_jerk_rad_s3": max_joint_jerk,
            "joint_jerk_rms_rad_s3": joint_jerk_rms,
            "ee_jerk_rms_m_s3": ee_jerk_rms,
            "command_jitter_rms_m_per_step2": command_jitter_rms,
            "joint_acceleration_limit_exceeded": bool(np.max(np.abs(joint_acc)) > acceleration_limit),
            "joint_jerk_limit_exceeded": bool(max_joint_jerk > jerk_limit),
            "policy_round_trip_mean_ms": float(np.mean(request_round_trip_ms)) if request_round_trip_ms else None,
            "policy_round_trip_p95_ms": float(np.percentile(request_round_trip_ms, 95)) if request_round_trip_ms else None,
            "policy_round_trip_p99_ms": float(np.percentile(request_round_trip_ms, 99)) if request_round_trip_ms else None,
            "policy_round_trip_max_ms": float(np.max(request_round_trip_ms)) if request_round_trip_ms else None,
            "server_infer_mean_ms": float(np.mean(server_infer_ms)) if server_infer_ms else None,
            "server_infer_p95_ms": float(np.percentile(server_infer_ms, 95)) if server_infer_ms else None,
            "server_infer_p99_ms": float(np.percentile(server_infer_ms, 99)) if server_infer_ms else None,
            "server_infer_max_ms": float(np.max(server_infer_ms)) if server_infer_ms else None,
        }
    )
    return metrics


def run_trajectory_benchmark():
    """Run a constrained whiteboard-erasing benchmark."""
    algorithm = args_cli.trajectory_benchmark_algorithm
    start = np.asarray(args_cli.trajectory_benchmark_start, dtype=np.float64)
    end = np.asarray(args_cli.trajectory_benchmark_end, dtype=np.float64)
    # Commands and IK use these flange poses.  The reported/evaluated TCP is
    # the eraser contact center directly above the whiteboard surface.
    working_start = _working_contact_position(start)
    working_end = _working_contact_position(end)
    stroke_style = args_cli.trajectory_benchmark_stroke_style
    is_timed_stroke = stroke_style in {"straight", "arc", "curve"}
    arc_center_xy = np.asarray(args_cli.trajectory_benchmark_arc_center, dtype=np.float64)
    if is_timed_stroke and args_cli.trajectory_benchmark_duration <= 0.0:
        raise ValueError("--trajectory_benchmark_duration must be positive")
    if args_cli.trajectory_benchmark_joint_torque_limit <= 0.0:
        raise ValueError("--trajectory_benchmark_joint_torque_limit must be positive")
    if args_cli.trajectory_benchmark_joint_acceleration_limit <= 0.0:
        raise ValueError("--trajectory_benchmark_joint_acceleration_limit must be positive")
    if args_cli.trajectory_benchmark_joint_jerk_limit <= 0.0:
        raise ValueError("--trajectory_benchmark_joint_jerk_limit must be positive")
    if args_cli.trajectory_benchmark_joint_velocity_limit <= 0.0:
        raise ValueError("--trajectory_benchmark_joint_velocity_limit must be positive")
    if args_cli.trajectory_benchmark_pd_stiffness <= 0.0 or args_cli.trajectory_benchmark_pd_damping <= 0.0:
        raise ValueError("Benchmark PD stiffness and damping must be positive")
    if args_cli.trajectory_benchmark_hold_after_completion < 0.0:
        raise ValueError("--trajectory_benchmark_hold_after_completion must be non-negative")
    if args_cli.trajectory_benchmark_policy_arc_residual_limit < 0.0:
        raise ValueError("--trajectory_benchmark_policy_arc_residual_limit must be non-negative")
    if args_cli.trajectory_benchmark_endpoint_solve <= 0.0:
        raise ValueError("--trajectory_benchmark_endpoint_solve must be positive")
    if not 0.0 < args_cli.trajectory_benchmark_endpoint_calibration_gain <= 1.0:
        raise ValueError("--trajectory_benchmark_endpoint_calibration_gain must be in (0, 1]")
    if args_cli.trajectory_benchmark_endpoint_final_settle <= 0.0:
        raise ValueError("--trajectory_benchmark_endpoint_final_settle must be positive")
    if args_cli.trajectory_benchmark_stroke_points < 2:
        raise ValueError("--trajectory_benchmark_stroke_points must be at least 2")
    if not 0.0 < args_cli.trajectory_benchmark_completion_ratio <= 1.0:
        raise ValueError("--trajectory_benchmark_completion_ratio must be in (0, 1]")
    if args_cli.trajectory_benchmark_contact_force < 0.0:
        raise ValueError("--trajectory_benchmark_contact_force must be non-negative")
    if args_cli.trajectory_benchmark_erase_force < args_cli.trajectory_benchmark_contact_force:
        raise ValueError("--trajectory_benchmark_erase_force must be at least the contact-force threshold")
    if is_timed_stroke and np.linalg.norm(end - start) < 1e-6:
        raise ValueError("Trajectory benchmark requires distinct A and B points")
    if stroke_style == "arc":
        _arc_geometry(start, end, arc_center_xy, args_cli.trajectory_benchmark_arc_direction)
    np.random.seed(args_cli.trajectory_benchmark_seed)
    torch.manual_seed(args_cli.trajectory_benchmark_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.trajectory_benchmark_seed)

    env_cfg = VLAEnvCfg()
    use_cameras = algorithm not in {"visual_movj", "visual_movl", "ideal"} or bool(
        args_cli.trajectory_benchmark_video
    )
    if not use_cameras:
        # Analytic visual baselines are driven by the already recognized A-B
        # stroke. Do not instantiate RTX sensors that they never read.
        env_cfg.scene.chest_camera = None
        env_cfg.scene.wrist_camera = None
    else:
        # The policy preprocessing resizes both observations to 256x256. A
        # benchmark-only 640x360 sensor keeps the same RGB modality and field
        # of view while avoiding the 1280x720 render cost at every control step.
        env_cfg.scene.chest_camera.width = 640
        env_cfg.scene.chest_camera.height = 360
        env_cfg.scene.wrist_camera.width = 640
        env_cfg.scene.wrist_camera.height = 360
    if algorithm in {"visual_movj", "visual_movl", "ideal"}:
        # These analytic baselines do not consume camera observations. Keep
        # the camera-enabled Kit experience for scene compatibility, but do
        # not render sensor frames during their physics-only benchmark.
        env_cfg.sim.render_interval = 1_000_000
    # For implicit actuators the effort limit must be present in the actuator
    # configuration before articulation construction; writing it afterwards
    # does not update the actuator's internally cached limit.
    env_cfg.scene.robot.actuators["arms"].effort_limit_sim = args_cli.trajectory_benchmark_joint_torque_limit
    env_cfg.scene.robot.actuators["arms"].effort_limit = args_cli.trajectory_benchmark_joint_torque_limit
    env_cfg.scene.robot.actuators["arms"].velocity_limit_sim = args_cli.trajectory_benchmark_joint_velocity_limit
    env_cfg.scene.robot.actuators["arms"].velocity_limit = args_cli.trajectory_benchmark_joint_velocity_limit
    env_cfg.scene.robot.actuators["arms"].stiffness = args_cli.trajectory_benchmark_pd_stiffness
    env_cfg.scene.robot.actuators["arms"].damping = args_cli.trajectory_benchmark_pd_damping
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    chest_camera = env.scene["chest_camera"] if use_cameras else None
    wrist_camera = env.scene["wrist_camera"] if use_cameras else None
    body_indices, _ = robot.find_bodies("arm_r_link_7")
    ee_idx = body_indices[0]
    right_joint_indices, _ = robot.find_joints("arm_r_joint_[1-7]")
    right_joint_names = [robot.joint_names[int(index)] for index in right_joint_indices]
    left_arm_fixed = torch.zeros((1, 7), device=env.device)
    gripper_2d_target = torch.tensor([[0.6, -1.0]], device=env.device, dtype=torch.float32)
    locked_quat = torch.tensor([LOCKED_WIPE_QUAT], device=env.device, dtype=torch.float32)
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    if not math.isclose(step_dt, TRAJECTORY_SAMPLE_DT_S, abs_tol=1.0e-12):
        raise RuntimeError(f"Expected a 50 Hz control period, got {1.0 / step_dt:.6f} Hz")
    if stroke_style == "irregular_word":
        # Reuse the same handwritten-word generator used by the normal
        # whiteboard task. The seed makes the visible word reproducible.
        stroke_points, _ = _make_word(
            x_start=WRITING_REGION_X[0] + 0.005,
            y_center=(WRITING_REGION_Y[0] + WRITING_REGION_Y[1]) * 0.5,
            word_width=(WRITING_REGION_X[1] - WRITING_REGION_X[0]) * 0.95,
            char_height=(WRITING_REGION_Y[1] - WRITING_REGION_Y[0]) * 0.75,
            num_pts_total=args_cli.trajectory_benchmark_stroke_points,
            z_base=BOARD_Z + 0.022,
            device=env.device,
        )
    elif stroke_style == "curve":
        stroke_points = _curve_marker_points(
            start, end, args_cli.trajectory_benchmark_stroke_points,
            BOARD_Z + 0.022, env.device,
        )
    elif stroke_style == "arc":
        stroke_points = _circular_arc_marker_points(
            start,
            end,
            arc_center_xy,
            args_cli.trajectory_benchmark_arc_direction,
            args_cli.trajectory_benchmark_stroke_points,
            BOARD_Z + 0.022,
            env.device,
        )
    else:
        stroke_t = torch.linspace(
            0.0, 1.0, args_cli.trajectory_benchmark_stroke_points, device=env.device
        )
        stroke_start = torch.tensor(start, device=env.device, dtype=torch.float32)
        stroke_end = torch.tensor(end, device=env.device, dtype=torch.float32)
        stroke_points = stroke_start + stroke_t[:, None] * (stroke_end - stroke_start)
        # Legacy straight benchmark: A/B are end-effector contact coordinates.
        stroke_points[:, 2] = BOARD_Z + 0.022
    dirt_positions = stroke_points.unsqueeze(0)
    initial_dirt_count = int(dirt_positions.shape[1])
    dirt_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TrajectoryBenchmarkStroke",
        markers={
            "dirt": sim_utils.CuboidCfg(
                size=(0.004, 0.004, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.05, 0.1), roughness=1.0
                ),
            )
        },
    )
    dirt_markers = VisualizationMarkers(dirt_marker_cfg)
    dirt_markers.visualize(translations=dirt_positions[0])
    video_writer = None
    video_path = ""
    if args_cli.trajectory_benchmark_video:
        video_path = os.path.abspath(args_cli.trajectory_benchmark_video)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        frame = chest_camera.data.output["rgb"][0].cpu().numpy()[..., :3]
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args_cli.trajectory_benchmark_video_fps,
            (width, height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open trajectory benchmark video: {video_path}")

    def make_action(target_position):
        target_pos_w = torch.tensor([target_position], device=env.device, dtype=torch.float32)
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.clone(),
            robot.data.root_quat_w.clone(),
            target_pos_w,
            locked_quat,
        )
        return torch.cat([torch.cat([pos_b, quat_b], dim=-1), left_arm_fixed, gripper_2d_target], dim=-1)

    def apply_right_arm_gravity_compensation():
        """Apply shared model-based gravity feed-forward to the right arm."""
        gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, right_joint_indices]
        robot.set_joint_effort_target(gravity, right_joint_indices)

    def step_setup_without_render(action):
        """Advance the excluded A/B setup phase without camera rendering."""
        env.action_manager.process_action(action)
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env.action_manager.apply_action()
            apply_right_arm_gravity_compensation()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

    def settle_cartesian_endpoint(target, duration_s):
        """Find a command bias whose physical TCP settles at ``target``.

        Implicit PD control has a gravity-induced static position error. This
        outer loop runs only before measurement and calibrates the shared A/B
        physical endpoints; it does not alter any method's measured path.
        """
        command = np.asarray(target, dtype=np.float64).copy()
        correction_steps = max(1, int(round(0.20 / step_dt)))
        total_steps = max(1, int(np.ceil(duration_s / step_dt)))
        # Keep correcting through the excluded setup window: with this
        # gravity-loaded implicit actuator, a fixed Cartesian target can
        # otherwise drift back outside the common 5 mm endpoint tolerance.
        calibration_steps = total_steps
        # Some board poses need a sizable Cartesian command bias to offset
        # gravity-induced joint error. This setup-only guard prevents the
        # calibration from running without bound while preserving a reachable
        # workspace around the requested physical endpoint.
        max_bias_m = 0.15
        for step_index in range(calibration_steps):
            step_setup_without_render(make_action(command))
            if (step_index + 1) % correction_steps == 0:
                actual = robot.data.body_pos_w[0, ee_idx].detach().cpu().numpy()
                command += args_cli.trajectory_benchmark_endpoint_calibration_gain * (target - actual)
                command = target + np.clip(command - target, -max_bias_m, max_bias_m)
        # The last correction above has not yet affected the physical robot.
        # Settle it before evaluating A/B; otherwise preflight measures an old
        # pose and falsely rejects a reachable endpoint.
        final_settle_steps = max(
            1, int(np.ceil(args_cli.trajectory_benchmark_endpoint_final_settle / step_dt))
        )
        for _ in range(final_settle_steps):
            step_setup_without_render(make_action(command))
        error = float(
            torch.linalg.vector_norm(
                robot.data.body_pos_w[0, ee_idx] - torch.tensor(target, device=env.device)
            ).item()
        )
        return command, error

    # The timed stroke tasks share calibrated physical A/B endpoints. The
    # irregular-handwriting mode retains its one-start-pose behavior.
    print("[wipe] Settling robot start pose...", flush=True)
    endpoint_setup_duration = max(
        args_cli.trajectory_benchmark_settle, args_cli.trajectory_benchmark_endpoint_solve
    ) if is_timed_stroke else args_cli.trajectory_benchmark_settle
    start_command, _ = settle_cartesian_endpoint(start, endpoint_setup_duration)
    if is_timed_stroke:
        print("[benchmark] Solving endpoint B...", flush=True)
        end_command, endpoint_solver_error = settle_cartesian_endpoint(
            end, args_cli.trajectory_benchmark_endpoint_solve
        )
        if endpoint_solver_error > args_cli.trajectory_benchmark_tolerance and not args_cli.trajectory_benchmark_preflight:
            raise RuntimeError(
                "The shared endpoint solve did not reach B within tolerance: "
                f"{endpoint_solver_error:.4f} m > {args_cli.trajectory_benchmark_tolerance:.4f} m."
            )
        endpoint_solver_peak_torque = float(
            torch.max(torch.abs(robot.data.applied_torque[0, right_joint_indices])).item()
        )
        joint_end = robot.data.joint_pos[0, right_joint_indices].detach().clone()
        env.reset()
        start_command, _ = settle_cartesian_endpoint(start, endpoint_setup_duration)
    else:
        # There is no B pose in a normal handwriting task. Learned Cartesian
        # actions remain free; the projector only needs a valid initial state.
        end_command = start_command.copy()
        endpoint_solver_error = 0.0
        endpoint_solver_peak_torque = 0.0
        joint_end = robot.data.joint_pos[0, right_joint_indices].detach().clone()
        env.reset()
        start_command, _ = settle_cartesian_endpoint(start, endpoint_setup_duration)
    start_solver_error = float(
        torch.linalg.vector_norm(
            robot.data.body_pos_w[0, ee_idx] - torch.tensor(start, device=env.device)
        ).item()
    )
    joint_start = robot.data.joint_pos[0, right_joint_indices].detach().clone()
    start_solver_peak_torque = float(
        torch.max(torch.abs(robot.data.applied_torque[0, right_joint_indices])).item()
    )
    if use_cameras:
        # ACT/DP need a valid initial RGB observation, but setup does not need
        # hundreds of redundant camera renders.
        env.sim.render()
    print("[benchmark] Starting measured trajectory window...", flush=True)
    minimum_duration = JointBoundaryProjector.minimum_feasible_duration(
        joint_start,
        joint_end,
        args_cli.trajectory_benchmark_joint_velocity_limit,
        args_cli.trajectory_benchmark_joint_acceleration_limit,
        args_cli.trajectory_benchmark_joint_jerk_limit,
    )
    arc_radius = None
    arc_sweep = None
    if stroke_style == "arc":
        arc_radius, _, arc_sweep = _arc_geometry(
            start, end, arc_center_xy, args_cli.trajectory_benchmark_arc_direction
        )
        chord_length = float(np.linalg.norm(end[:2] - start[:2]))
        # The endpoint-only joint bound must cover MOVL's longer Cartesian
        # arc. This conservative scale is common to all four controllers.
        minimum_duration *= max(1.0, abs(arc_sweep) * arc_radius / max(chord_length, 1.0e-9))
    if args_cli.trajectory_benchmark_preflight:
        preflight_ok = (
            endpoint_solver_error <= args_cli.trajectory_benchmark_tolerance
            and start_solver_error <= args_cli.trajectory_benchmark_tolerance
            and args_cli.trajectory_benchmark_duration >= minimum_duration
        )
        preflight = {
            "schema_version": 7,
            "benchmark": "curve_constraint_preflight" if stroke_style == "curve" else ("circular_arc_constraint_preflight" if stroke_style == "arc" else "a_to_b_constraint_preflight"),
            "start_A_world_m": start,
            "end_B_world_m": end,
            "start_A_command_m": start_command,
            "end_B_command_m": end_command,
            "completion_time_limit_s": args_cli.trajectory_benchmark_duration,
            "constraints": {
                "endpoint_tolerance_m": args_cli.trajectory_benchmark_tolerance,
                "joint_torque_limit_nm": args_cli.trajectory_benchmark_joint_torque_limit,
                "joint_acceleration_limit_rad_s2": args_cli.trajectory_benchmark_joint_acceleration_limit,
                "joint_jerk_limit_rad_s3": args_cli.trajectory_benchmark_joint_jerk_limit,
                "joint_velocity_limit_rad_s": args_cli.trajectory_benchmark_joint_velocity_limit,
                "joint_pd_stiffness": args_cli.trajectory_benchmark_pd_stiffness,
                "joint_pd_damping": args_cli.trajectory_benchmark_pd_damping,
            },
            "endpoint_B_error_m": endpoint_solver_error,
            "endpoint_A_error_m": start_solver_error,
            "endpoint_B_peak_joint_torque_nm": endpoint_solver_peak_torque,
            "endpoint_A_peak_joint_torque_nm": start_solver_peak_torque,
            "minimum_visual_baseline_duration_s": minimum_duration,
            "preflight_passed": preflight_ok,
        }
        if stroke_style == "arc":
            preflight["arc"] = {
                "center_world_xy_m": arc_center_xy,
                "direction": args_cli.trajectory_benchmark_arc_direction,
                "radius_m": arc_radius,
                "sweep_rad": arc_sweep,
            }
        output_path = os.path.abspath(args_cli.trajectory_benchmark_output)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(_to_builtin(preflight), output_file, indent=2)
        print(f"[benchmark] Preflight {'passed' if preflight_ok else 'failed'}: {output_path}", flush=True)
        print(f"[benchmark] B error: {endpoint_solver_error * 1000.0:.2f} mm", flush=True)
        print(f"[benchmark] A error: {start_solver_error * 1000.0:.2f} mm", flush=True)
        print(f"[benchmark] B peak torque: {endpoint_solver_peak_torque:.2f} N m", flush=True)
        print(f"[benchmark] A peak torque: {start_solver_peak_torque:.2f} N m", flush=True)
        print(f"[benchmark] Minimum visual-baseline duration: {minimum_duration:.2f} s", flush=True)
        # Kit may leave non-daemon shutdown work alive after a completed
        # headless app.  The result file is already closed above, so use a
        # process exit to return the launcher and release GPU resources.
        if getattr(args_cli, "headless", False):
            os._exit(0)
        simulation_app.close()
        return
    if is_timed_stroke and args_cli.trajectory_benchmark_duration < minimum_duration:
        raise ValueError(
            "Requested --trajectory_benchmark_duration is below the common completion-time limit required by MOVJ/MOVL. "
            f"Use at least {minimum_duration:.2f} s for the selected acceleration/jerk limits."
        )
    # Each method has the same hard deadline, but not a forced completion time.
    # The physical Cartesian loop needs a settling tail after the geometric
    # reference reaches B.  Reserve that tail for every method; learned
    # policies still retain their own phase-rate and lateral corrections.
    baseline_duration = (
        max(step_dt, minimum_duration, 0.85 * args_cli.trajectory_benchmark_duration)
        if is_timed_stroke
        else max(step_dt, minimum_duration, 1.0)
    )
    joint_projector = JointBoundaryProjector(
        joint_start,
        joint_end,
        baseline_duration,
        step_dt,
        args_cli.trajectory_benchmark_joint_velocity_limit,
        args_cli.trajectory_benchmark_joint_acceleration_limit,
        args_cli.trajectory_benchmark_joint_jerk_limit,
    )

    bridge = None
    if algorithm not in {"ideal", "visual_movj", "visual_movl"}:
        bridge = IsaacWebSocketBridge(uri=f"ws://127.0.0.1:{args_cli.policy_port}")
        if not bridge.connect_to_server(simulation_app, seed=args_cli.trajectory_benchmark_seed):
            raise RuntimeError("Could not connect to the requested policy server")
        expected_policy = "act" if algorithm == "act" else "official_diffusion_policy"
        if bridge.policy_type != expected_policy:
            raise RuntimeError(
                f"Benchmark algorithm '{algorithm}' requires '{expected_policy}', "
                f"but server reports '{bridge.policy_type}'"
            )
        if bridge.policy_state_dim not in (14, 20):
            raise RuntimeError(f"Unsupported benchmark policy state width: {bridge.policy_state_dim}")
        if algorithm == "dp_small_xy" and bridge.policy_action_dim != 2:
            raise RuntimeError(
                f"dp_small_xy requires a 2D action checkpoint, got action_dim={bridge.policy_action_dim}"
            )
        if algorithm == "dp" and bridge.policy_action_dim == 2:
            raise RuntimeError(
                "dp requires the standard non-2D checkpoint; use --trajectory_benchmark_algorithm dp_small_xy"
            )

    samples = []
    cpu_started = time.process_time()
    rss_started_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    total_steps = max(1, int(np.ceil(args_cli.trajectory_benchmark_duration / step_dt))) if is_timed_stroke else None
    completed = False
    # The measured window begins at the calibrated physical A pose.
    held_policy_position = start_command.copy()
    filtered_policy_phase_rate = 1.0
    arc_phase_time_s = 0.0
    final_command_position = held_policy_position.copy()
    # Match the standard-DP deployment: retain the policy's XY/pose command,
    # while this common external controller finds and regulates board contact in Z.
    z_admittance_enabled = algorithm in {"act", "dp"}
    z_contact_acquired = False
    # Under the common low-stiffness actuator the calibrated Cartesian command
    # differs from the physical TCP pose. Start admittance in command space.
    z_contact_cmd = float(start_command[2])
    z_command_floor = float(min(start_command[2], end_command[2]) - 0.02)
    accumulated_z_offset = 0.0
    target_force_z = 6.0
    contact_enter_force = 3.0
    contact_lost_force = 0.5
    admittance_k = 0.0003
    max_z_offset = 0.08
    coarse_approach_z_step = 0.002
    fine_approach_z_step = 0.0005
    fine_approach_start_z = BOARD_Z + 0.245

    def reference_position(time_s):
        if stroke_style == "curve":
            return _curve_position(start, end, time_s, baseline_duration)
        if stroke_style == "arc":
            return _circular_arc_position(
                start,
                end,
                arc_center_xy,
                args_cli.trajectory_benchmark_arc_direction,
                time_s,
                baseline_duration,
            )
        return _minimum_jerk_position(start, end, time_s, baseline_duration)

    def calibrated_reference_command(time_s):
        """Map the physical reference through the excluded endpoint command biases."""
        blend = _minimum_jerk_blend(time_s, baseline_duration)
        bias = (1.0 - blend) * (start_command - start) + blend * (end_command - end)
        return reference_position(time_s) + bias

    def read_sample(
        time_s, command_position, policy_round_trip_ms=None, server_infer_ms=None, joint_target=None,
        reference_time_s=None,
    ):
        net_force = env.scene["contact_forces"].data.net_forces_w[0, 0]
        remaining = int(torch.sum(dirt_positions[0, :, 2] > 0.0).item())
        base_position_w = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        base_quaternion_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
        flange_position_w = robot.data.body_pos_w[0, ee_idx].detach().cpu().numpy().copy()
        flange_quaternion_wxyz = robot.data.body_quat_w[0, ee_idx].detach().cpu().numpy().copy()
        world_T_base = _pose_wxyz_to_matrix(base_position_w, base_quaternion_wxyz)
        world_T_flange = _pose_wxyz_to_matrix(flange_position_w, flange_quaternion_wxyz)
        base_T_flange = np.linalg.solve(world_T_base, world_T_flange)
        world_T_tcp = _working_contact_transform(world_T_flange)
        base_T_tcp = np.linalg.solve(world_T_base, world_T_tcp)
        world_T_command_flange = _pose_wxyz_to_matrix(command_position, LOCKED_WIPE_QUAT)
        base_T_command_flange = np.linalg.solve(world_T_base, world_T_command_flange)
        world_T_command_tcp = _working_contact_transform(world_T_command_flange)
        base_T_command_tcp = np.linalg.solve(world_T_base, world_T_command_tcp)
        return {
            "time_s": float(time_s),
            "wall_time_s": float(time.perf_counter() - wall_started),
            # Kept for schema-v4 readers. It now means the working TCP.
            "ee_position_m": world_T_tcp[:3, 3],
            "ee_quaternion_wxyz": flange_quaternion_wxyz,
            "world_T_base": world_T_base,
            "flange_pose_base": {
                "position_m": base_T_flange[:3, 3],
                "quaternion_wxyz": R.from_matrix(base_T_flange[:3, :3]).as_quat()[[3, 0, 1, 2]],
                "matrix": base_T_flange,
            },
            "flange_pose_world": {
                "position_m": flange_position_w,
                "quaternion_wxyz": flange_quaternion_wxyz,
                "matrix": world_T_flange,
            },
            "tcp_pose_base": {
                "position_m": base_T_tcp[:3, 3],
                "quaternion_wxyz": R.from_matrix(base_T_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]],
                "matrix": base_T_tcp,
            },
            "tcp_pose_world": {
                "position_m": world_T_tcp[:3, 3],
                "quaternion_wxyz": flange_quaternion_wxyz,
                "matrix": world_T_tcp,
            },
            "command_flange_pose_base": {
                "position_m": base_T_command_flange[:3, 3],
                "quaternion_wxyz": R.from_matrix(base_T_command_flange[:3, :3]).as_quat()[[3, 0, 1, 2]],
                "matrix": base_T_command_flange,
            },
            "command_tcp_pose_base": {
                "position_m": base_T_command_tcp[:3, 3],
                "quaternion_wxyz": R.from_matrix(base_T_command_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]],
                "matrix": base_T_command_tcp,
            },
            "command_flange_position_world_m": np.asarray(command_position, dtype=np.float64).copy(),
            "command_position_m": np.asarray(command_position, dtype=np.float64).copy(),
            "reference_time_s": float(time_s if reference_time_s is None else reference_time_s),
            # Keep the checkpoint's historical articulation ordering for policy input.
            "policy_joint_position_rad": robot.data.joint_pos[0, :7].detach().cpu().numpy().copy(),
            "joint_position_rad": robot.data.joint_pos[0, right_joint_indices].detach().cpu().numpy().copy(),
            "joint_target_rad": (
                robot.data.joint_pos[0, right_joint_indices].detach().cpu().numpy().copy()
                if joint_target is None
                else joint_target.detach().cpu().numpy().copy()
            ),
            "joint_velocity_rad_s": robot.data.joint_vel[0, right_joint_indices].detach().cpu().numpy().copy(),
            "joint_acc_rad_s2": robot.data.joint_acc[0, right_joint_indices].detach().cpu().numpy().copy(),
            "joint_torque_nm": robot.data.applied_torque[0, right_joint_indices].detach().cpu().numpy().copy(),
            "contact_force_n": float(abs(net_force[2].item())),
            "remaining_stroke_points": remaining,
            "erased_ratio": float(1.0 - remaining / initial_dirt_count),
            "policy_round_trip_ms": policy_round_trip_ms,
            "server_infer_ms": server_infer_ms,
        }

    def step_right_joint_target(action, joint_target):
        """Apply a projected right-arm target while retaining shared left/gripper actions."""
        env.action_manager.process_action(action)
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env.action_manager.apply_action()
            # The projected target is written after differential IK so it is
            # the right arm's final command for this physics substep.
            robot.set_joint_position_target(joint_target.unsqueeze(0), right_joint_indices)
            apply_right_arm_gravity_compensation()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)

    def step_cartesian_target(action):
        """Execute a Cartesian IK target without converting MOVL to MOVJ."""
        env.action_manager.process_action(action)
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env.action_manager.apply_action()
            apply_right_arm_gravity_compensation()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)

    def erase_contacted_stroke():
        net_force = env.scene["contact_forces"].data.net_forces_w[0, 0]
        if abs(net_force[2].item()) < args_cli.trajectory_benchmark_erase_force:
            return
        # The board is planar and the eraser is held normal to it. Resolve the
        # physical contact patch in board/world XY rather than the TCP's local
        # coordinates, whose roll may otherwise make a contacting flat eraser
        # miss visible marker points.
        ee_pos_w = robot.data.body_pos_w[0, ee_idx].unsqueeze(0)
        planar_delta = dirt_positions[..., :2] - ee_pos_w.unsqueeze(1)[..., :2]
        erase_mask = (
            (torch.abs(planar_delta[..., 0]) < 0.04)
            & (torch.abs(planar_delta[..., 1]) < 0.03)
            & (dirt_positions[..., 2] > 0.0)
        )
        if torch.any(erase_mask):
            dirt_positions[erase_mask] = torch.tensor(
                [0.0, 0.0, -10.0], device=env.device, dtype=dirt_positions.dtype
            )
            dirt_markers.visualize(translations=dirt_positions[0])

    # The settled start pose is the first measured sample. Timed tasks require
    # both the requested erase ratio and endpoint B before T_max.
    samples.append(read_sample(0.0, start_command, reference_time_s=0.0))
    step_iterator = range(total_steps) if total_steps is not None else itertools.count()
    for step in step_iterator:
        if not simulation_app.is_running():
            break
        time_s = step * step_dt
        planned_time_s = min(arc_phase_time_s, baseline_duration)
        current = samples[-1]
        # Policy checkpoints were trained with the articulated flange pose.
        # Recording and scoring use the separate, physical eraser-contact TCP.
        ee_position = np.asarray(current["flange_pose_world"]["position_m"])
        ee_quat = np.asarray(current["flange_pose_world"]["quaternion_wxyz"])
        policy_joint_pos = np.asarray(current["policy_joint_position_rad"])
        net_force = env.scene["contact_forces"].data.net_forces_w[0, 0]
        wrench_6d = torch.cat(
            [net_force, torch.zeros(3, device=env.device, dtype=net_force.dtype)]
        ).cpu().numpy()

        policy_round_trip_ms = None
        server_infer_ms = None
        desired_joint_target = None
        if algorithm in {"ideal", "visual_movl"}:
            command_position = calibrated_reference_command(planned_time_s)
        elif algorithm == "visual_movj":
            # The displayed command stays on the shared arc, while MOVJ
            # receives its own endpoint-only joint-space interpolation below.
            command_position = calibrated_reference_command(planned_time_s)
        else:
            if bridge.action_queue:
                # ACT returns a 50-step action chunk. Reuse the queued actions
                # without copying or resizing two camera frames every step.
                bridge.wait_for_action_and_reply(None, simulation_app)
            else:
                pose7 = np.concatenate([ee_position, ee_quat]).astype(np.float32)
                policy_state = np.concatenate([pose7, policy_joint_pos, wrench_6d]).astype(np.float32)
                if bridge.policy_state_dim == 14:
                    policy_state = policy_state[:14]
                primary = apply_model_crop_resize(
                    chest_camera.data.output["rgb"][0].cpu().numpy()[..., :3],
                    target_res=(256, 256), crop_type="right",
                )
                wrist = apply_model_crop_resize(
                    wrist_camera.data.output["rgb"][0].cpu().numpy()[..., :3],
                    target_res=(256, 256), crop_type="center",
                )
                bridge.wait_for_action_and_reply(
                    {
                        "instruction": "wipe the whiteboard clean",
                        "rgb_primary_raw": primary,
                        "rgb_wrist_raw": wrist,
                        "policy_state": policy_state,
                    },
                    simulation_app,
                )
            policy_round_trip_ms = bridge.last_round_trip_ms
            if bridge.last_server_timing:
                server_infer_ms = bridge.last_server_timing.get("infer_ms")
            if bridge.current_cmd == "step":
                raw_action = np.asarray(bridge.latest_osc_action, dtype=np.float32).reshape(-1)
                if raw_action.size >= 3:
                    held_policy_position = raw_action[:3].astype(np.float64, copy=True)
                elif raw_action.size == 2 and bridge.policy_type == "official_diffusion_policy":
                    held_policy_position[:2] = raw_action
                else:
                    raise RuntimeError(f"Unsupported policy action shape in benchmark: {raw_action.shape}")
            if stroke_style in {"arc", "curve"}:
                # The reference supplies the common A-to-B task phase.  ACT
                # and DP then contribute their raw lateral policy output.
                # With a zero residual limit this is deliberately neither
                # clipped, filtered, nor faded at the endpoints: the measured
                # curve can expose the two trained models' actual behavior.
                # Endpoint error is reported rather than corrected away.
                arc_command = calibrated_reference_command(planned_time_s)
                raw_residual_xy = held_policy_position[:2] - arc_command[:2]
                residual_norm = float(np.linalg.norm(raw_residual_xy))
                residual_limit = args_cli.trajectory_benchmark_policy_arc_residual_limit
                if residual_norm > residual_limit > 0.0:
                    raw_residual_xy *= residual_limit / residual_norm
                command_position = arc_command.copy()
                command_position[:2] += raw_residual_xy
            else:
                command_position = held_policy_position.copy()

        if z_admittance_enabled:
            contact_force_z = float(abs(net_force[2].item()))
            if contact_force_z >= contact_enter_force:
                z_contact_acquired = True
            elif z_contact_acquired and contact_force_z < contact_lost_force:
                z_contact_acquired = False
                accumulated_z_offset = 0.0

            if not z_contact_acquired:
                z_contact_cmd = min(z_contact_cmd, float(command_position[2]))
                approach_step = (
                    coarse_approach_z_step
                    if z_contact_cmd > fine_approach_start_z
                    else fine_approach_z_step
                )
                z_contact_cmd = max(z_contact_cmd - approach_step, z_command_floor)
                command_position[2] = z_contact_cmd
            else:
                # Do not let a new policy action abruptly lift a contacted eraser.
                z_contact_cmd = min(z_contact_cmd, float(command_position[2]))
                command_position[2] = min(float(command_position[2]), z_contact_cmd)

            if contact_force_z > contact_lost_force:
                force_error = contact_force_z - target_force_z
                adaptive_k = admittance_k
                if contact_force_z > 10.0:
                    adaptive_k *= 1.0 + (contact_force_z - 10.0) * 0.2
                accumulated_z_offset += force_error * adaptive_k * step_dt
            accumulated_z_offset = float(
                np.clip(accumulated_z_offset, -max_z_offset, max_z_offset)
            )
            command_position[2] = max(
                command_position[2] + accumulated_z_offset, z_command_floor
            )

        control_action = make_action(command_position)
        projected_joint_target = None
        if algorithm == "visual_movj":
            # MOVJ follows a sequence of arc waypoints in joint space.  It is
            # intentionally different from MOVL's Cartesian interpolation,
            # while still making its TCP trace the requested handwriting.
            env.action_manager.process_action(control_action)
            env.action_manager.apply_action()
            desired_joint_target = robot.data.joint_pos_target[0, right_joint_indices].detach().clone()
            if is_timed_stroke and planned_time_s >= baseline_duration:
                desired_joint_target = joint_end.clone()
            projected_joint_target, _ = joint_projector.project(
                desired_joint_target, planned_time_s, learned=False,
            )
            step_right_joint_target(control_action, projected_joint_target)
        else:
            # ACT, DP, and MOVL retain Cartesian arc tracking.  Their actual
            # TCP and physical joint derivatives are independently measured
            # against the same limits; no endpoint-only joint projector can
            # turn their paths into the same MOVJ motion.
            step_cartesian_target(control_action)
        erase_contacted_stroke()
        final_command_position = command_position.copy()
        sample = read_sample(
            (step + 1) * step_dt,
            command_position,
            policy_round_trip_ms,
            server_infer_ms,
            projected_joint_target,
            planned_time_s,
        )
        samples.append(sample)
        if video_writer is not None:
            frame_bgr = chest_camera.data.output["rgb"][0].cpu().numpy()[..., :3][..., ::-1]
            video_writer.write(np.ascontiguousarray(frame_bgr))
        if stroke_style == "irregular_word":
            remaining_limit = int(
                np.ceil(initial_dirt_count * (1.0 - args_cli.trajectory_benchmark_completion_ratio))
            )
            completed = int(sample["remaining_stroke_points"]) <= remaining_limit
        elif stroke_style in {"arc", "curve"}:
            # This comparison has a shared time upper bound, not a hard B
            # acceptance gate.  It ends when the policy-guided A-to-B phase
            # is exhausted; erase ratio and physical endpoint error stay as
            # independent outcomes.  Otherwise the common reference would
            # hide a learned policy's lateral/terminal behavior by forcing it
            # to keep running until it happened to enter a tight B tube.
            completed = arc_phase_time_s >= baseline_duration
        else:
            completed = (
                np.linalg.norm(np.asarray(sample["ee_position_m"]) - working_end)
                <= args_cli.trajectory_benchmark_tolerance
            )
        if completed:
            break
        if is_timed_stroke and arc_phase_time_s < baseline_duration:
            if algorithm in {"act", "dp"}:
                next_reference = calibrated_reference_command(
                    min(planned_time_s + step_dt, baseline_duration)
                )
                tangent_xy = next_reference[:2] - arc_command[:2]
                tangent_norm = float(np.linalg.norm(tangent_xy))
                if tangent_norm > 1.0e-9:
                    tangent_xy /= tangent_norm
                    policy_tangent_error = float(
                        np.dot(held_policy_position[:2] - arc_command[:2], tangent_xy)
                    )
                    target_phase_rate = float(np.clip(1.0 + 8.0 * policy_tangent_error, 0.85, 1.15))
                    filtered_policy_phase_rate = (
                        0.90 * filtered_policy_phase_rate + 0.10 * target_phase_rate
                    )
            arc_phase_time_s = min(
                baseline_duration, arc_phase_time_s + step_dt * filtered_policy_phase_rate
            )
        if (step + 1) % max(1, int(round(1.0 / step_dt))) == 0:
            elapsed_text = (
                f"{((step + 1) * step_dt):.1f}s"
                if total_steps is None
                else f"{((step + 1) * step_dt):.1f}/{args_cli.trajectory_benchmark_duration:.1f}s"
            )
            print(f"[wipe] Running {elapsed_text}; erased={sample['erased_ratio']:.1%}", flush=True)
    # GUI inspection is intentionally outside the fixed-duration interval.
    if args_cli.trajectory_benchmark_hold_after_completion > 0.0:
        hold_steps = int(np.ceil(args_cli.trajectory_benchmark_hold_after_completion / step_dt))
        for _ in range(hold_steps):
            if not simulation_app.is_running():
                break
            env.step(make_action(final_command_position))
            if video_writer is not None:
                frame_bgr = chest_camera.data.output["rgb"][0].cpu().numpy()[..., :3][..., ::-1]
                video_writer.write(np.ascontiguousarray(frame_bgr))

    if not samples:
        if bridge:
            bridge.close()
        simulation_app.close()
        raise RuntimeError("Simulation closed before any trajectory benchmark sample was recorded")
    _append_trajectory_derivatives(samples)
    if is_timed_stroke:
        # Persist the common reference in the same robot-base frame as the
        # measured TCP. This makes arc tracking and all requested TCP plots
        # inspectable without reconstructing scene transforms externally.
        for sample in samples:
            reference_world = _working_contact_position(
                reference_position(sample.get("reference_time_s", sample["time_s"]))
            )
            world_T_reference = _pose_wxyz_to_matrix(reference_world, LOCKED_WIPE_QUAT)
            base_T_reference = np.linalg.solve(
                np.asarray(sample["world_T_base"], dtype=np.float64), world_T_reference
            )
            sample["reference_tcp_pose_base"] = {
                "position_m": base_T_reference[:3, 3],
                "quaternion_wxyz": R.from_matrix(base_T_reference[:3, :3]).as_quat()[[3, 0, 1, 2]],
                "matrix": base_T_reference,
            }
            sample["reference_position_world_m"] = reference_world
    metric_reference_end = working_end if is_timed_stroke else working_start
    metrics = _trajectory_benchmark_metrics(
        samples,
        working_start,
        metric_reference_end,
        baseline_duration,
        (
            args_cli.trajectory_benchmark_duration if is_timed_stroke else None
        ),
        args_cli.trajectory_benchmark_tolerance,
        args_cli.trajectory_benchmark_joint_acceleration_limit,
        args_cli.trajectory_benchmark_joint_jerk_limit,
        completed,
        require_endpoint=is_timed_stroke,
        reference_kind="curve" if stroke_style == "curve" else ("arc" if stroke_style == "arc" else "line"),
        arc_center_xy=arc_center_xy if stroke_style == "arc" else None,
        arc_direction=args_cli.trajectory_benchmark_arc_direction,
    )
    metrics["joint_torque_limit_exceeded"] = bool(
        metrics["max_joint_torque_nm"] > args_cli.trajectory_benchmark_joint_torque_limit
    )
    hard_constraint_keys = [
        "start_tolerance_exceeded",
        "completion_time_limit_exceeded",
        "joint_torque_limit_exceeded",
        "joint_acceleration_limit_exceeded",
        "joint_jerk_limit_exceeded",
    ]
    # For curve/arc wiping the terminal TCP error is an evaluation metric,
    # not a pass/fail constraint.  The requested A/B task and common T_max
    # remain unchanged, while raw ACT/DP end behavior remains observable.
    if stroke_style == "straight":
        hard_constraint_keys.append("endpoint_tolerance_exceeded")
    strict_valid = not any(metrics[key] for key in hard_constraint_keys)
    if stroke_style == "irregular_word" and not completed:
        strict_valid = False
    resources = {
        "client_cpu_time_s": float(time.process_time() - cpu_started),
        "client_peak_rss_delta_mib": float(
            max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_started_kib) / 1024.0
        ),
        "client_peak_cuda_allocated_mib": (
            float(torch.cuda.max_memory_allocated() / (1024.0**2)) if torch.cuda.is_available() else 0.0
        ),
        "client_peak_cuda_reserved_mib": (
            float(torch.cuda.max_memory_reserved() / (1024.0**2)) if torch.cuda.is_available() else 0.0
        ),
    }
    learned_policy = bridge is not None
    base_T_world_at_start = np.linalg.inv(np.asarray(samples[0]["world_T_base"], dtype=np.float64))
    start_A_base = (base_T_world_at_start @ np.r_[working_start, 1.0])[:3]
    end_B_base = (base_T_world_at_start @ np.r_[working_end, 1.0])[:3]
    flange_start_A_base = (base_T_world_at_start @ np.r_[start, 1.0])[:3]
    flange_end_B_base = (base_T_world_at_start @ np.r_[end, 1.0])[:3]
    result = {
        "schema_version": 7 if stroke_style in {"arc", "curve"} else 5,
        "benchmark": (
            "time_limited_a_to_b_erasing_motion"
            if stroke_style == "straight"
            else "curve_erasing_motion"
            if stroke_style == "curve"
            else "circular_arc_erasing_motion"
            if stroke_style == "arc"
            else "irregular_handwriting_erasing"
        ),
        "algorithm": algorithm,
        "trial_seed": args_cli.trajectory_benchmark_seed,
        "policy_type": bridge.policy_type if bridge else "vision_path_planner",
        "policy_state_dim": bridge.policy_state_dim if bridge else None,
        "policy_action_dim": bridge.policy_action_dim if bridge else None,
        "model_parameter_count": int(bridge.metadata.get("parameter_count", 0)) if bridge else 0,
        "training_data": {
            "episodes": 777 if learned_policy else 0,
            "frames": 245817 if learned_policy else 0,
        },
        "resources": resources,
        "controller_assistance": (
            "common_gravity_compensation+policy_guided_arc_tracking_raw_unbounded_lateral_output+"
            "z_contact_approach_and_admittance"
            if z_admittance_enabled
            else "common_gravity_compensation+arc_waypoint_visual_baseline"
        ),
        "sampling": {
            "frequency_hz": TRAJECTORY_SAMPLE_HZ,
            "control_period_s": step_dt,
            "physics_dt_s": env_cfg.sim.dt,
            "physics_steps_per_sample": env_cfg.decimation,
        },
        "coordinate_frames": {
            "trajectory_reference": "robot_base",
            "robot_base": {
                "description": "Isaac Lab robot articulation root; this is the robot-base frame for this platform.",
                "platform": "dual_arm",
                "robot_asset": "scripts/dual_arm/urdf/dual_arm.usd",
                "body": "Robot articulation root",
                "pose_in_world_field": "world_T_base",
            },
            "working_tcp": {
                "description": "Eraser contact center projected onto the whiteboard working plane.",
                "world_z_m": WORKING_TCP_Z,
                "position_rule": "flange world XY with fixed whiteboard-contact Z",
            },
            "flange_body": "arm_r_link_7",
            "pose_matrix_convention": "T_A_B maps homogeneous coordinates from frame B into frame A.",
            "quaternion_convention": "wxyz",
        },
        "tcp_semantics": "working_eraser_contact_center_v2",
        "joint_names": right_joint_names,
        "start_A_world_m": working_start,
        "end_B_world_m": working_end,
        "start_A_robot_base_m": start_A_base,
        "end_B_robot_base_m": end_B_base,
        "flange_start_A_world_m": start,
        "flange_end_B_world_m": end,
        "flange_start_A_robot_base_m": flange_start_A_base,
        "flange_end_B_robot_base_m": flange_end_B_base,
        "flange_start_A_command_m": start_command,
        "flange_end_B_command_m": end_command,
        "working_tcp_start_A_command_world_m": _working_contact_position(start_command),
        "working_tcp_end_B_command_world_m": _working_contact_position(end_command),
        # Historical aliases: these remain flange commands for schema-v7
        # consumers. Use the explicit flange_* fields in new analyses.
        "start_A_command_m": start_command,
        "end_B_command_m": end_command,
        "stroke": stroke_style,
        "stroke_generator": (
            "_make_word->_generate_stroke_points"
            if stroke_style == "irregular_word"
            else "smooth_s_curve"
            if stroke_style == "curve"
            else "minimum_jerk_circular_arc"
            if stroke_style == "arc"
            else "minimum_jerk_straight_line"
        ),
        "stroke_marker_count": initial_dirt_count,
        "stroke_marker_z_world_m": BOARD_Z + 0.022,
        "reference": (
            "nominal_A_to_B_setup_pose_only"
            if stroke_style == "irregular_word"
            else "policy_guided_s_curve_tracking"
            if stroke_style == "curve"
            else "policy_guided_circular_arc_tracking"
            if stroke_style == "arc"
            else "minimum_jerk_straight_line_for_visual_baselines"
        ),
        "completion_time_limit_s": (
            args_cli.trajectory_benchmark_duration if is_timed_stroke else None
        ),
        "visual_baseline_planned_duration_s": baseline_duration if is_timed_stroke else None,
        "completed_within_time_limit": completed,
        "hold_after_completion_s": args_cli.trajectory_benchmark_hold_after_completion,
        "constraints": {
            "endpoint_tolerance_m": args_cli.trajectory_benchmark_tolerance,
            "endpoint_tolerance_mode": "report_only" if stroke_style in {"arc", "curve"} else "hard",
            "joint_torque_limit_nm": args_cli.trajectory_benchmark_joint_torque_limit,
            "joint_acceleration_limit_rad_s2": args_cli.trajectory_benchmark_joint_acceleration_limit,
            "joint_jerk_limit_rad_s3": args_cli.trajectory_benchmark_joint_jerk_limit,
            "joint_command_acceleration_limit_rad_s2": args_cli.trajectory_benchmark_joint_acceleration_limit,
            "joint_command_jerk_limit_rad_s3": args_cli.trajectory_benchmark_joint_jerk_limit,
            "joint_velocity_limit_rad_s": args_cli.trajectory_benchmark_joint_velocity_limit,
            "joint_pd_stiffness": args_cli.trajectory_benchmark_pd_stiffness,
            "joint_pd_damping": args_cli.trajectory_benchmark_pd_damping,
            "learned_policy_arc_residual_limit_m": args_cli.trajectory_benchmark_policy_arc_residual_limit,
        },
        "strict_constraints_satisfied": strict_valid,
        "metrics": metrics,
        "samples": samples,
    }
    if stroke_style == "arc":
        arc_radius, arc_start_angle, arc_sweep = _arc_geometry(
            start, end, arc_center_xy, args_cli.trajectory_benchmark_arc_direction
        )
        result["arc"] = {
            "center_world_xy_m": arc_center_xy,
            "radius_m": arc_radius,
            "start_angle_rad": arc_start_angle,
            "sweep_rad": arc_sweep,
            "direction": args_cli.trajectory_benchmark_arc_direction,
        }
        result["completion_condition"] = "arc_erased_to_ratio_and_tcp_at_B"
    if stroke_style == "irregular_word":
        # Keep the normal wiping result independent from the old A/B schema.
        result["task"] = "wipe_whiteboard_irregular_handwriting"
        result["completion_condition"] = "all_handwriting_points_erased"
        result.pop("completion_time_limit_s")
        result.pop("visual_baseline_planned_duration_s")
        result["start_pose_world_m"] = result.pop("start_A_world_m")
        result.pop("end_B_world_m")
        result.pop("start_A_robot_base_m")
        result.pop("end_B_robot_base_m")
        result["start_pose_command_world_m"] = result.pop("start_A_command_m")
        result.pop("end_B_command_m")
        result["reference"] = "none"
    output_path = os.path.abspath(args_cli.trajectory_benchmark_output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_output_path = f"{output_path}.tmp"
    with open(temporary_output_path, "w", encoding="utf-8") as output_file:
        json.dump(_to_builtin(result), output_file, indent=2)
    os.replace(temporary_output_path, output_path)
    if video_writer is not None:
        video_writer.release()
    print(
        "\nIrregular handwriting wipe complete"
        if stroke_style == "irregular_word"
        else "\nTime-limited circular-arc trajectory benchmark complete"
        if stroke_style == "arc"
        else "\nTime-limited A-to-B trajectory benchmark complete"
    )
    print(f"  Algorithm: {algorithm}")
    print(f"  Completion time: {metrics['completion_time_s'] if completed else 'timeout'} s")
    if is_timed_stroke:
        print(f"  Time limit: {args_cli.trajectory_benchmark_duration:.2f} s")
    else:
        print("  Time limit: none (runs until the handwriting is erased)")
    print(f"  Trajectory RMSE: {metrics['trajectory_rmse_m'] * 1000.0:.2f} mm")
    print(f"  Peak joint acceleration: {metrics['max_joint_acceleration_rad_s2']:.2f} rad/s^2")
    print(f"  Joint jerk RMS: {metrics['joint_jerk_rms_rad_s3']:.2f} rad/s^3")
    print(f"  Result JSON: {output_path}")
    if video_path:
        print(f"  Video MP4: {video_path}")
    if not strict_valid:
        failed_constraints = [key for key in hard_constraint_keys if metrics[key]]
        print(f"  Invalid trial, violated: {', '.join(failed_constraints)}")
    if bridge:
        bridge.close()
    if getattr(args_cli, "headless", False):
        # The diagnostic JSON is the benchmark result even when a method
        # misses its endpoint or time target.  Return success so a batch can
        # collect comparable failures from every method and repeat.
        os._exit(0)
    # Kit shutdown can hang after a finite headless benchmark. Ensure the
    # outer launcher regains control and can stop the policy-server child.
    import threading

    exit_watchdog = threading.Timer(10.0, os._exit, args=(0,))
    exit_watchdog.daemon = True
    exit_watchdog.start()
    simulation_app.close()
    exit_watchdog.cancel()


def main():
    if args_cli.trajectory_benchmark:
        try:
            run_trajectory_benchmark()
        except BaseException:
            # Endpoint/constraint validation may fail before the normal
            # benchmark shutdown path.  Always release Isaac Sim resources so
            # a rejected trial cannot retain the GPU for the next one.
            simulation_app.close()
            raise
        return
    success_times = []

    BASE_SAVE_DIR = "vla_test_data"
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    for sub in ["success", "slight_timeout", "timeout", "collision"]:
        os.makedirs(os.path.join(BASE_SAVE_DIR, sub), exist_ok=True)
    FOR_TRAIN_DIR = os.path.join(BASE_SAVE_DIR, "for_train")
    os.makedirs(FOR_TRAIN_DIR, exist_ok=True)
    if SAVE_REMNANT:
        os.makedirs(os.path.join(BASE_SAVE_DIR, "remnants"), exist_ok=True)

    env_cfg = VLAEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    obs, info = env.reset()

    left_indices, _ = env.scene["robot"].find_joints("arm_l_joint_[1-7]")
    right_joint_indices, _ = env.scene["robot"].find_joints("arm_r_joint_[1-7]")
    left_arm_fixed = torch.zeros((1, 7), device=env.device)
    gripper_2d_target = torch.tensor([[0.6, -1.0]], device=env.device, dtype=torch.float32)

    ws_bridge = IsaacWebSocketBridge(uri=f"ws://127.0.0.1:{args_cli.policy_port}")

    robot = env.scene["robot"]
    chest_camera = env.scene["chest_camera"]
    wrist_camera = env.scene["wrist_camera"]

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DirtMarkers",
        markers={"dirt": sim_utils.CuboidCfg(size=(0.004, 0.004, 0.001), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.1), roughness=1.0))},
    )
    dirt_markers = VisualizationMarkers(marker_cfg)

    body_indices, _ = robot.find_bodies("arm_r_link_7")
    ee_idx = body_indices[0]

    def latch_current_pose():
        r_pos_w = robot.data.root_pos_w.clone()
        r_quat_w = robot.data.root_quat_w.clone()
        e_pos_w = robot.data.body_pos_w[:, ee_idx, :].clone()
        e_quat_w = robot.data.body_quat_w[:, ee_idx, :].clone()
        e_pos_b, e_quat_b = math_utils.subtract_frame_transforms(r_pos_w, r_quat_w, e_pos_w, e_quat_w)
        r_arm_target = torch.cat([e_pos_b, e_quat_b], dim=-1)
        return torch.cat([r_arm_target, left_arm_fixed, gripper_2d_target], dim=-1)

    print("[INFO]: 仿真器就绪，正在尝试连接 VLA ...")
    ws_bridge.connect_to_server(simulation_app)
    if args_cli.record_trajectory and ws_bridge.policy_type != "act":
        raise RuntimeError(
            "--record_trajectory requires the ACT WebSocket server; "
            f"received policy_type={ws_bridge.policy_type!r}"
        )
    if args_cli.record_trajectory and ws_bridge.policy_state_dim != 20:
        raise RuntimeError(
            "--record_trajectory expects the ACT 20D pose7+joint7+wrench6 state; "
            f"received state_dim={ws_bridge.policy_state_dim}"
        )
    fast_dp_deploy = (
        args_cli.fast_dp_deploy
        and ws_bridge.policy_type == "official_diffusion_policy"
    )
    small_xy_dp = fast_dp_deploy and ws_bridge.policy_action_dim == 2
    if args_cli.dp_raw_action_eval and ws_bridge.policy_type != "official_diffusion_policy":
        raise RuntimeError("--dp_raw_action_eval requires an official_diffusion_policy server")
    if args_cli.dp_raw_action_eval and not fast_dp_deploy:
        raise RuntimeError("--dp_raw_action_eval also requires --fast_dp_deploy")
    dp_raw_action_eval = args_cli.dp_raw_action_eval
    # The small-XY policy has no Z output. Keep its recordings, including the
    # controller-generated inspection segment, so its behaviour remains visible.
    save_episode_data = SAVE_EPISODE_DATA and (not fast_dp_deploy or small_xy_dp)
    if fast_dp_deploy:
        if small_xy_dp:
            print(
                "[INFO]: Official DP small-XY 部署：保留回合录制；"
                "擦除进度到 80% 后将抬手观察并重新建立接触。专家接管已禁用"
            )
        elif dp_raw_action_eval:
            print("[INFO]: Official DP 原始动作评估：关闭专家接管、Z 接触辅助和导纳力控")
        else:
            print("[INFO]: Official DP 部署：DP 控制 XYZ，叠加 Z 向导纳力控；专家接管已禁用")

    record_trajectory = bool(args_cli.record_trajectory)
    TOTAL_EPISODES = (
        1
        if record_trajectory
        else (
        args_cli.dp_eval_episodes
        if fast_dp_deploy and args_cli.dp_eval_episodes > 0
        else 500
        )
    )
    SOFT_TIMEOUT = 60.0
    dp_eval_records = []
    
    success_count = 0
    expert_takeover_success_count = 0
    timeout_fail_count = 0
    slight_timeout_fail_count = 0
    expert_fail_count = 0

    TARGET_FORCE_Z = 6.0
    ADMITTANCE_K = 0.0003
    MAX_Z_OFFSET = 0.08
    DP_CONTACT_ENTER_FORCE = 3.0
    DP_CONTACT_LOST_FORCE = 0.5
    DP_APPROACH_Z_STEP = 0.0005
    DP_COARSE_APPROACH_Z_STEP = 0.002
    DP_FINE_APPROACH_START_Z = BOARD_Z + 0.245
    DP_MIN_LINK_Z = BOARD_Z + 0.185
    SMALL_XY_INSPECTION_TRIGGER_RATIO = 0.80
    SMALL_XY_INSPECTION_HOLD_S = 0.40
    SMALL_XY_INSPECTION_LIFT_STEP = 0.004
    SMALL_XY_INSPECTION_DESCENT_STEP = 0.0015

    episode_counter = 2000

    for episode_idx in range(TOTAL_EPISODES):
        print(f"\n{'='*50}")
        print(f"[回合 {episode_idx + 1}/{TOTAL_EPISODES}] 开始初始化")
        print('='*50)

        ws_bridge.action_queue.clear()
        ws_bridge.latest_osc_action = None
        ws_bridge.current_cmd = "idle"

        episode_seed = None
        if fast_dp_deploy:
            episode_seed = args_cli.dp_eval_seed + episode_idx
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)

        env.reset()
        dirt_positions, instr_list = generate_random_scribbles(env.device, env.num_envs)
        # 强制固定语言指令
        current_instruction = "wipe the whiteboard clean"
        dirt_markers.visualize(translations=dirt_positions[0])

        initial_dirt_count = torch.sum(dirt_positions[0][:, 2] > 0).item()
        remaining_history = []
        expert_takeover = False
        
        lift_hand_start_time = None
        current_hard_timeout = (
            None
            if record_trajectory
            else (args_cli.dp_eval_timeout if fast_dp_deploy else 70.0)
        )

        # 飞向 Home 位（使用 LOCKED_WIPE_QUAT）
        home_quat_w = torch.tensor([LOCKED_WIPE_QUAT], device=env.device, dtype=torch.float)
        for _ in range(40):
            target_pos_w = torch.tensor([[HOME_X, HOME_Y, BOARD_Z + 0.35]], device=env.device, dtype=torch.float)
            target_quat_w = home_quat_w
            root_pos_w = robot.data.root_pos_w.clone()
            root_quat_w = robot.data.root_quat_w.clone()
            pos_b, quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
            action = torch.cat([torch.cat([pos_b, quat_b], dim=-1), left_arm_fixed, gripper_2d_target], dim=-1)
            env.step(action)

        try:
            reset_payload = {"reset": True, "robo_name": "wipe_board"}
            if fast_dp_deploy:
                reset_payload["seed"] = episode_seed
            ws_bridge.websocket.send(ws_bridge.packer.pack(reset_payload))
            reset_resp = ws_bridge.websocket.recv()
            print(f"[回合 {episode_idx + 1}] VLA 模型端记忆已清理")
        except Exception as e:
            print(f"[回合 {episode_idx + 1}] 警告：VLA 重置通信异常: {e}")

        locked_home_pose = latch_current_pose()
        for _ in range(10):
            env.step(locked_home_pose)
        ws_bridge.locked_target = locked_home_pose

        ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        ee_quat = robot.data.body_quat_w[0, ee_idx].cpu().numpy()
        expert_controller = ExpertController(ee_pos.copy(), ee_quat.copy())

        ep_step_count = 0
        sim_elapsed = 0.0
        episode_success = False
        fail_reason = ""
        episode_buffer = {
            "image_primary_jpgs": [],
            "image_wrist_jpgs": [],
            # 已移除 depth_primary_mm
            "joint_pos": [],
            "pose7": [],
            "wrench": [],
            "action": []
        }
        trajectory_samples = []

        current_base_pose_w = np.concatenate([ee_pos, ee_quat])
        current_base_pose_w[2] -= 0.01
        accumulated_z_offset = 0.0
        expert_action_7d = None
        dp_contact_time = None
        dp_contact_acquired = False
        dp_contact_z_cmd = float(ee_pos[2])
        progress_times = {"90": None, "95": None, "99": None}
        small_xy_inspection_state = "wipe"
        small_xy_inspection_complete = False
        small_xy_inspection_hold_start = None

        if record_trajectory:
            trajectory_samples.append(
                _capture_normal_trajectory_sample(
                    robot,
                    ee_idx,
                    right_joint_indices,
                    0.0,
                    np.concatenate([ee_pos, ee_quat]),
                    np.zeros(3, dtype=np.float64),
                    initial_dirt_count,
                    int(initial_dirt_count),
                )
            )

        pending_primary_tensors = []
        pending_wrist_tensors = []
        episode_wall_start = time.perf_counter()
        # 移除 pending_depth_tensors

        def flush_images():
            if pending_primary_tensors:
                jpgs_p, jpgs_w = compress_images(pending_primary_tensors, pending_wrist_tensors)
                episode_buffer["image_primary_jpgs"].extend(jpgs_p)
                episode_buffer["image_wrist_jpgs"].extend(jpgs_w)
                pending_primary_tensors.clear()
                pending_wrist_tensors.clear()

        while simulation_app.is_running():
            sim_elapsed = ep_step_count * env_cfg.sim.dt * env_cfg.decimation

            if current_hard_timeout is not None and sim_elapsed > current_hard_timeout:
                episode_success = False
                if expert_takeover:
                    fail_reason = "专家接管后依然超时"
                    expert_fail_count += 1
                else:
                    fail_reason = "自主硬超时未完成"
                    timeout_fail_count += 1
                print(f"[回合 {episode_idx + 1}] ❌ {fail_reason} | 仿真用时: {sim_elapsed:.2f}s / {current_hard_timeout}s")
                break

            all_body_pos_w = robot.data.body_pos_w[0]
            ee_pos_w = all_body_pos_w[ee_idx].unsqueeze(0)
            ee_quat_w = robot.data.body_quat_w[:, ee_idx, :]
            pose7_world = np.concatenate([ee_pos_w.cpu().numpy()[0], ee_quat_w.cpu().numpy()[0]])
            joint_pos = robot.data.joint_pos[:, :7].cpu().numpy()[0]

            net_forces = env.scene["contact_forces"].data.net_forces_w[0, 0]
            net_torques = torch.zeros(3, device=env.device, dtype=net_forces.dtype)
            wrench_6d = torch.cat([net_forces, net_torques]).cpu().numpy()
            contact_force_z = abs(net_forces[2].item())
            if (
                ws_bridge.policy_type == "official_diffusion_policy"
                and dp_contact_time is None
                and contact_force_z >= DP_CONTACT_ENTER_FORCE
            ):
                dp_contact_time = sim_elapsed

            valid_dirt = dirt_positions[0][dirt_positions[0][:, 2] > 0]
            remaining_count = len(valid_dirt)

            erased_ratio_now = (
                1.0 - remaining_count / initial_dirt_count
                if initial_dirt_count > 0
                else 1.0
            )
            if fast_dp_deploy and initial_dirt_count > 0:
                for threshold_name, threshold in (("90", 0.90), ("95", 0.95), ("99", 0.99)):
                    if progress_times[threshold_name] is None and erased_ratio_now >= threshold:
                        progress_times[threshold_name] = sim_elapsed

            rgb_primary_tensor = chest_camera.data.output["rgb"][0].clone()
            rgb_wrist_tensor = wrist_camera.data.output["rgb"][0].clone()
            # 移除 depth_primary_tensor

            if save_episode_data:
                pending_primary_tensors.append(rgb_primary_tensor)
                pending_wrist_tensors.append(rgb_wrist_tensor)
            # 不再追加深度

            if ep_step_count % 150 == 0:
                flush_images()

            # ---- 退场逻辑 ----
            if remaining_count == 0:
                # This observation was captured before discovering success and
                # has no matching policy action. Drop it before recording the
                # controller-driven exit trajectory so HDF5 fields stay aligned.
                if save_episode_data:
                    if pending_primary_tensors:
                        pending_primary_tensors.pop()
                        pending_wrist_tensors.pop()
                    elif episode_buffer["image_primary_jpgs"]:
                        episode_buffer["image_primary_jpgs"].pop()
                        episode_buffer["image_wrist_jpgs"].pop()

                if expert_takeover:
                    episode_success = True
                    print(f"[回合 {episode_idx + 1}] 🦾 专家救场擦除完成！(用时 {sim_elapsed:.2f}s) 准备存入 DAgger")
                else:
                    if record_trajectory or sim_elapsed <= SOFT_TIMEOUT:
                        episode_success = True
                        success_count += 1
                        success_times.append(sim_elapsed)
                        print(f"[回合 {episode_idx + 1}] ✅ 自主擦除完成，任务成功！")
                    else:
                        episode_success = False
                        fail_reason = "略超时"
                        slight_timeout_fail_count += 1
                        print(f"[回合 {episode_idx + 1}] ⏱️ 擦除完成但仿真用时 {sim_elapsed:.2f}s 超过 {SOFT_TIMEOUT}s，判定为略超时失败")
                
                # ----- 退场录制（含 record_and_step，已移除深度） -----
                if save_episode_data and episode_success:
                    print("🛫 正在录制【主动避挡退场】动作...")
                    
                    def record_and_step(tgt_pose, act_tensor):
                        nonlocal ep_step_count
                        # 1. 抓取当前物理状态
                        curr_body_pos = robot.data.body_pos_w[0]
                        curr_ee_pos = curr_body_pos[ee_idx].unsqueeze(0)
                        curr_ee_quat = robot.data.body_quat_w[:, ee_idx, :]
                        curr_pose7 = np.concatenate([curr_ee_pos.cpu().numpy()[0], curr_ee_quat.cpu().numpy()[0]])
                        curr_joint = robot.data.joint_pos[:, :7].cpu().numpy()[0]
                        
                        curr_net_f = env.scene["contact_forces"].data.net_forces_w[0, 0]
                        curr_net_t = torch.zeros(3, device=env.device, dtype=curr_net_f.dtype)
                        curr_wrench = torch.cat([curr_net_f, curr_net_t]).cpu().numpy()
                        
                        # 2. 抓取当前相机画面（仅 RGB，无深度）
                        pending_primary_tensors.append(chest_camera.data.output["rgb"][0].clone())
                        pending_wrist_tensors.append(wrist_camera.data.output["rgb"][0].clone())
                        
                        # 3. 写入缓冲区
                        episode_buffer["joint_pos"].append(curr_joint)
                        episode_buffer["pose7"].append(curr_pose7)
                        episode_buffer["wrench"].append(curr_wrench)
                        episode_buffer["action"].append(tgt_pose)
                        
                        # 4. 内存管理与环境步进
                        if ep_step_count % 150 == 0:
                            flush_images()
                            
                        env.step(act_tensor)
                        ep_step_count += 1

                    # --- 退场阶段 1：垂直拔高 ---
                    start_pos = ee_pos_w[0].cpu().numpy().copy()
                    start_quat = ee_quat_w[0].cpu().numpy().copy()
                    lift_pos = start_pos.copy()
                    lift_pos[2] = SAFE_RETURN_Z
                    for pt in generate_smooth_transition(start_pos, lift_pos, speed_factor=560):
                        target_pose = np.concatenate([pt, start_quat])
                        right_arm_target = torch.tensor([target_pose], device=env.device, dtype=torch.float32)
                        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
                            robot.data.root_pos_w.clone(), robot.data.root_quat_w.clone(), right_arm_target[:, :3], right_arm_target[:, 3:]
                        )
                        action = torch.cat([torch.cat([target_pos_b, target_quat_b], dim=-1), left_arm_fixed, gripper_2d_target], dim=-1)
                        record_and_step(target_pose, action)
                        
                    # --- 退场阶段 2：飞回右侧 Home 点 ---
                    home_pos = np.array([HOME_X, HOME_Y, SAFE_RETURN_Z])
                    for pt in generate_smooth_transition(lift_pos, home_pos, speed_factor=400):
                        target_pose = np.concatenate([pt, start_quat])
                        right_arm_target = torch.tensor([target_pose], device=env.device, dtype=torch.float32)
                        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
                            robot.data.root_pos_w.clone(), robot.data.root_quat_w.clone(), right_arm_target[:, :3], right_arm_target[:, 3:]
                        )
                        action = torch.cat([torch.cat([target_pos_b, target_quat_b], dim=-1), left_arm_fixed, gripper_2d_target], dim=-1)
                        record_and_step(target_pose, action)
                        
                    # --- 退场阶段 3：悬停驻留 ---
                    for _ in range(10): 
                        record_and_step(target_pose, action)

                flush_images()
                break

            # ===== 专家接管触发判断 =====
            if not expert_takeover and not fast_dp_deploy and not record_trajectory:
                if contact_force_z < 0.5:
                    if lift_hand_start_time is None:
                        lift_hand_start_time = sim_elapsed
                else:
                    lift_hand_start_time = None

                if sim_elapsed >= 15.0:
                    remaining_history.append((sim_elapsed, remaining_count))
                    while remaining_history and sim_elapsed - remaining_history[0][0] > 10.0:
                        remaining_history.pop(0)
                
                if sim_elapsed >= 15.0:
                    trigger = False
                    trigger_reason = ""
                    if sim_elapsed > 60.0 and remaining_count > 0:
                        trigger = True
                        trigger_reason = "60秒未完成"
                    elif lift_hand_start_time is not None and (sim_elapsed - lift_hand_start_time) > 15.0:
                        trigger = True
                        trigger_reason = "抬手空转超15秒"
                    elif len(remaining_history) >= 2:
                        t0, cnt0 = remaining_history[0]
                        t1, cnt1 = remaining_history[-1]
                        if t1 - t0 >= 9.9:
                            erased_in_window = cnt0 - cnt1
                            erase_ratio = erased_in_window / initial_dirt_count if initial_dirt_count > 0 else 0
                            if erase_ratio < 0.10:
                                trigger = True
                                trigger_reason = f"10秒内擦除效率过低({erase_ratio*100:.1f}%)"
                    
                    if trigger:
                        expert_takeover = True
                        current_hard_timeout = 90.0
                        print(f"🚨 自主失败，触发专家接管！原因: {trigger_reason} | 剩余 {remaining_count} 点")
                        ws_bridge.current_cmd = 'idle'
                        for key in episode_buffer:
                            if key not in ["image_primary_jpgs", "image_wrist_jpgs"]:
                                episode_buffer[key].clear()
                        pending_primary_tensors.clear()
                        pending_wrist_tensors.clear()
                        episode_buffer["image_primary_jpgs"].clear()
                        episode_buffer["image_wrist_jpgs"].clear()
                        
                        expert_controller.cmd_pos = ee_pos_w[0].cpu().numpy().copy()
                        expert_controller.cmd_quat = ee_quat_w[0].cpu().numpy().copy()
                        expert_controller.filtered_force = contact_force_z
                        expert_controller.prev_error = contact_force_z - TARGET_FORCE_Z
                        
                        dirt_2d = valid_dirt.cpu().numpy()[:, :2] if len(valid_dirt) > 0 else np.empty((0,2))
                        if len(dirt_2d) > 0:
                            dists_to_expert = np.linalg.norm(dirt_2d - expert_controller.cmd_pos[:2], axis=1)
                            if contact_force_z > 3.0 and np.any(dists_to_expert < expert_controller.radar_radius):
                                expert_controller.state = 2
                                print("🦾 专家就绪：贴地起步，直接开始擦拭！")
                            else:
                                expert_controller.state = 0
                                print("🦾 专家就绪：距离较远，开始平滑转场...")
                        else:
                            expert_controller.state = 0

            if (
                small_xy_dp
                and not dp_raw_action_eval
                and not small_xy_inspection_complete
                and small_xy_inspection_state == "wipe"
                and remaining_count > 0
                and erased_ratio_now >= SMALL_XY_INSPECTION_TRIGGER_RATIO
            ):
                # A 2D checkpoint cannot request a Z lift itself. This short
                # controller phase exposes the remaining board to both cameras,
                # then returns to force-regulated wiping if dirt remains.
                small_xy_inspection_state = "lift"
                small_xy_inspection_complete = True
                small_xy_inspection_hold_start = None
                dp_contact_acquired = False
                accumulated_z_offset = 0.0
                print(
                    f"🧐 small-XY DP 擦除 {erased_ratio_now * 100:.1f}%，"
                    f"抬手检查剩余 {remaining_count} 点"
                )

            # ---- 控制模式 ----
            if not expert_takeover:
                # 获取原始图像（用于保存）
                rgb_primary_raw = rgb_primary_tensor.cpu().numpy()[..., :3]
                rgb_wrist_raw = rgb_wrist_tensor.cpu().numpy()[..., :3]
                
                # 💥 核心修复：在发送给模型前，严格执行与训练数据相同的非对称裁剪和缩放！
                model_input_primary = apply_model_crop_resize(rgb_primary_raw, target_res=(256, 256), crop_type='right')
                model_input_wrist = apply_model_crop_resize(rgb_wrist_raw, target_res=(256, 256), crop_type='center')

                if args_cli.policy_state_dim == 20:
                    policy_state = np.concatenate([pose7_world, joint_pos, wrench_6d]).astype(np.float32)
                else:
                    policy_state = np.concatenate([pose7_world, joint_pos]).astype(np.float32)
                state_dict = {
                    "instruction": current_instruction,
                    "pose7": pose7_world,
                    "joint_pos": joint_pos,
                    "rgb_primary_raw": model_input_primary,  # 发送裁剪后的 256x256 图像
                    "rgb_wrist_raw": model_input_wrist,      # 发送裁剪后的 256x256 图像
                    "policy_state": policy_state,
                    "dirt_points": valid_dirt.cpu().numpy() if len(valid_dirt) > 0 else None,
                }
                ws_bridge.wait_for_action_and_reply(state_dict, simulation_app)
                if ws_bridge.current_cmd == 'step':
                    action_raw = np.asarray(
                        ws_bridge.latest_osc_action, dtype=np.float32
                    ).reshape(-1)
                    # The separate small-XY checkpoint intentionally outputs only XY;
                    # keep its legacy compatibility while standard 14D DP follows ACT.
                    if (
                        ws_bridge.policy_type == "official_diffusion_policy"
                        and action_raw.shape == (2,)
                    ):
                        current_action_7d = np.concatenate(
                            [pose7_world[:3], LOCKED_WIPE_QUAT]
                        ).astype(np.float32)
                        current_action_7d[:2] = action_raw
                    elif action_raw.size >= 7:
                        current_action_7d = action_raw[:7].copy()
                    else:
                        raise ValueError(
                            f"Unsupported policy action shape: {action_raw.shape}"
                        )
                    # 强制使用完美擦除姿态
                    current_action_7d[3:7] = LOCKED_WIPE_QUAT
                    expert_action_7d = None
                else:
                    current_action_7d = current_base_pose_w.copy()
            else:
                current_ee_xyz = ee_pos_w[0].cpu().numpy()
                dirt_2d = valid_dirt.cpu().numpy()[:, :2] if len(valid_dirt) > 0 else np.empty((0,2))
                expert_action_7d = expert_controller.step(current_ee_xyz, contact_force_z, dirt_2d)
                current_action_7d = expert_action_7d

            # ---- 导纳力控 ----
            pos_w = current_action_7d[:3].copy()
            quat_wxyz = current_action_7d[3:7]
            small_xy_is_inspecting = small_xy_dp and small_xy_inspection_state != "wipe"

            if not expert_takeover and not dp_raw_action_eval:
                if ws_bridge.policy_type == "official_diffusion_policy":
                    if small_xy_is_inspecting:
                        current_xyz = ee_pos_w[0].cpu().numpy()
                        # Keep the eraser above the same patch while the cameras
                        # observe it, rather than sweeping sideways in the air.
                        pos_w[:2] = current_xyz[:2]
                        if small_xy_inspection_state == "lift":
                            pos_w[2] = min(
                                current_xyz[2] + SMALL_XY_INSPECTION_LIFT_STEP,
                                Z_HOVER,
                            )
                            if pos_w[2] >= Z_HOVER - 1e-6:
                                small_xy_inspection_state = "hold"
                                small_xy_inspection_hold_start = sim_elapsed
                                print("🧐 small-XY DP 已抬手，正在获取检查画面")
                        elif small_xy_inspection_state == "hold":
                            pos_w[2] = Z_HOVER
                            if sim_elapsed - small_xy_inspection_hold_start >= SMALL_XY_INSPECTION_HOLD_S:
                                small_xy_inspection_state = "reacquire"
                                print("🧽 small-XY DP 检查结束，重新下探擦除")
                        else:  # reacquire
                            pos_w[2] = max(
                                current_xyz[2] - SMALL_XY_INSPECTION_DESCENT_STEP,
                                DP_MIN_LINK_Z,
                            )
                            if contact_force_z >= DP_CONTACT_ENTER_FORCE:
                                small_xy_inspection_state = "wipe"
                                dp_contact_acquired = True
                                dp_contact_z_cmd = float(pos_w[2])
                                accumulated_z_offset = 0.0
                                print("🧽 small-XY DP 已重新建立接触，继续擦除")
                    else:
                        if contact_force_z >= DP_CONTACT_ENTER_FORCE:
                            dp_contact_acquired = True
                        elif dp_contact_acquired and contact_force_z < DP_CONTACT_LOST_FORCE:
                            # Re-acquire contact using Z only if a chunk lifts the eraser.
                            dp_contact_acquired = False
                            accumulated_z_offset = 0.0

                        if not dp_contact_acquired:
                            # Preserve the policy's XY exactly. This is only a startup/lost-contact
                            # fallback for models whose raw Z target remains above the board.
                            dp_contact_z_cmd = min(dp_contact_z_cmd, float(pos_w[2]))
                            approach_step = (
                                DP_COARSE_APPROACH_Z_STEP
                                if dp_contact_z_cmd > DP_FINE_APPROACH_START_Z
                                else DP_APPROACH_Z_STEP
                            )
                            dp_contact_z_cmd = max(
                                dp_contact_z_cmd - approach_step,
                                DP_MIN_LINK_Z,
                            )
                            pos_w[2] = dp_contact_z_cmd
                        elif not small_xy_dp:
                            # Standard 14D DP keeps its contact guard. Small-XY
                            # has no raw Z command and is allowed to use the
                            # inspection controller above.
                            dp_contact_z_cmd = min(dp_contact_z_cmd, float(pos_w[2]))
                            pos_w[2] = min(float(pos_w[2]), dp_contact_z_cmd)

                if not small_xy_is_inspecting:
                    if contact_force_z > 0.5:
                        force_error = contact_force_z - TARGET_FORCE_Z
                        if contact_force_z > 10.0:
                            adaptive_k = ADMITTANCE_K * (1.0 + (contact_force_z - 10.0) * 0.2)
                        else:
                            adaptive_k = ADMITTANCE_K
                        step_offset = force_error * adaptive_k * (env_cfg.sim.dt * env_cfg.decimation)
                        accumulated_z_offset += step_offset
                    accumulated_z_offset = np.clip(accumulated_z_offset, -MAX_Z_OFFSET, MAX_Z_OFFSET)
                    pos_w[2] += accumulated_z_offset
                    if pos_w[2] < BOARD_Z + 0.185:
                        pos_w[2] = BOARD_Z + 0.185

                if ws_bridge.policy_type == "official_diffusion_policy" and ep_step_count % 100 == 0:
                    diagnostic_prefix = "[DP RAW] " if dp_raw_action_eval else "[DP XYZ] "
                    print(
                        diagnostic_prefix +
                        f"raw=({current_action_7d[0]:.3f},{current_action_7d[1]:.3f},{current_action_7d[2]:.3f}) "
                        f"cmd=({pos_w[0]:.3f},{pos_w[1]:.3f},{pos_w[2]:.3f}) "
                        f"force={contact_force_z:.2f}N contact={dp_contact_acquired} "
                        f"inspection={small_xy_inspection_state if small_xy_dp else 'off'}"
                    )
                
            target_pose = np.concatenate([pos_w, quat_wxyz])

            right_arm_target = torch.tensor([target_pose], device=env.device, dtype=torch.float32)
            target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
                robot.data.root_pos_w.clone(), robot.data.root_quat_w.clone(), right_arm_target[:, :3], right_arm_target[:, 3:]
            )
            ws_bridge.locked_target = torch.cat([torch.cat([target_pos_b, target_quat_b], dim=-1), left_arm_fixed, gripper_2d_target], dim=-1)

            obs, info = env.step(ws_bridge.locked_target)
            ep_step_count += 1

            if save_episode_data:
                episode_buffer["joint_pos"].append(joint_pos)
                episode_buffer["pose7"].append(pose7_world)
                episode_buffer["wrench"].append(wrench_6d)
                if expert_takeover:
                    episode_buffer["action"].append(expert_action_7d)
                elif small_xy_is_inspecting:
                    episode_buffer["action"].append(target_pose)
                else:
                    episode_buffer["action"].append(current_action_7d)

            # 擦除检测（完全对齐脚本1）
            if ws_bridge.current_cmd == 'step' or expert_takeover:
                ee_quat_inv = ee_quat_w.clone()
                ee_quat_inv[:, 1:] = -ee_quat_inv[:, 1:]  
                rel_pos_w = dirt_positions - ee_pos_w.unsqueeze(1)
                ee_quat_inv_expanded = ee_quat_inv.unsqueeze(1).expand(-1, dirt_positions.shape[1], -1)
                local_pos = math_utils.quat_apply(ee_quat_inv_expanded, rel_pos_w)
                mask_x = torch.abs(local_pos[..., 0]) < 0.04
                mask_y = torch.abs(local_pos[..., 1] - 0.02) < 0.03
                mask_z = (local_pos[..., 2] < -0.20) & (local_pos[..., 2] > -0.225)
                FORCE_THRESHOLD = 5.0
                if contact_force_z > FORCE_THRESHOLD:
                    erase_mask = mask_x & mask_y & mask_z
                    dirt_positions[erase_mask] = torch.tensor([0.0, 0.0, -10.0], device=env.device)
                    dirt_markers.visualize(translations=dirt_positions[0])

            if record_trajectory:
                remaining_after_step = int(torch.sum(dirt_positions[0, :, 2] > 0.0).item())
                post_step_force = env.scene["contact_forces"].data.net_forces_w[0, 0]
                server_timing = ws_bridge.last_server_timing or {}
                trajectory_samples.append(
                    _capture_normal_trajectory_sample(
                        robot,
                        ee_idx,
                        right_joint_indices,
                        ep_step_count * env_cfg.sim.dt * env_cfg.decimation,
                        target_pose,
                        post_step_force.detach().cpu().numpy(),
                        initial_dirt_count,
                        remaining_after_step,
                        ws_bridge.last_round_trip_ms,
                        server_timing.get("infer_ms"),
                    )
                )

        flush_images()

        # ---------- 后处理与保存 ----------
        if expert_takeover and episode_success:
            expert_takeover_success_count += 1

        if not episode_success and SAVE_REMNANT:
            valid_points = dirt_positions[0][dirt_positions[0][:, 2] > 0].cpu().numpy()
            if len(valid_points) > 0:
                remnant_dir = os.path.join(BASE_SAVE_DIR, "remnants")
                remnant_filename = f"remnant_ep{episode_idx+1:03d}_{fail_reason}.hdf5"
                remnant_path = os.path.join(remnant_dir, remnant_filename)
                with h5py.File(remnant_path, 'w') as f:
                    f.create_dataset('points', data=valid_points)
                    f.attrs['episode'] = episode_idx + 1
                    f.attrs['elapsed_time'] = sim_elapsed
                    f.attrs['fail_reason'] = fail_reason
                    f.attrs['num_points'] = len(valid_points)
                print(f"[保存残局] 已保存 {len(valid_points)} 个剩余字迹点至 {remnant_filename}")

        # ---- 保存数据 ----
        if save_episode_data and len(episode_buffer["image_primary_jpgs"]) > 0:
            if expert_takeover and episode_success:
                save_dir = FOR_TRAIN_DIR
                filename = f"episode_{episode_counter:04d}.hdf5"
                episode_counter += 1
            elif expert_takeover and not episode_success:
                save_dir = os.path.join(BASE_SAVE_DIR, "timeout")
                filename = f"episode_{episode_idx+1:03d}.hdf5"
            elif episode_success:
                save_dir = os.path.join(BASE_SAVE_DIR, "success")
                filename = f"episode_{episode_idx+1:03d}.hdf5"
            else:
                if fail_reason == "略超时":
                    subdir = "slight_timeout"
                elif fail_reason == "自主硬超时未完成":
                    subdir = "timeout"
                elif fail_reason == "碰撞":
                    subdir = "collision"
                else:
                    subdir = "other"
                save_dir = os.path.join(BASE_SAVE_DIR, subdir)
                filename = f"episode_{episode_idx+1:03d}.hdf5"
            
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            with h5py.File(filepath, 'w') as f:
                grp_obs = f.create_group('obs')
                grp_action = f.create_group('action')
                primary_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
                wrist_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
                
                grp_obs.create_dataset('image_primary', data=np.array(episode_buffer["image_primary_jpgs"], dtype=object), dtype=primary_dtype)
                grp_obs.create_dataset('image_wrist', data=np.array(episode_buffer["image_wrist_jpgs"], dtype=object), dtype=wrist_dtype)
                # 不再保存 distance_to_camera
                
                grp_obs.create_dataset('joint_pos', data=np.array(episode_buffer["joint_pos"]))
                grp_obs.create_dataset('pose7', data=np.array(episode_buffer["pose7"]))
                grp_obs.create_dataset('wrench', data=np.array(episode_buffer["wrench"]))
                grp_action.create_dataset('target_pose', data=np.array(episode_buffer["action"]))
                f.attrs['instruction'] = current_instruction
                f.attrs['success'] = episode_success
                f.attrs['elapsed_time'] = sim_elapsed
                f.attrs['episode_index'] = episode_idx + 1
                f.attrs['fail_reason'] = fail_reason if not episode_success else "none"
                f.attrs['num_steps'] = ep_step_count
                f.attrs['expert_takeover'] = expert_takeover
            print(f"[保存] 回合数据已保存至 {filepath}")
        elif save_episode_data:
            print("[警告] 回合无图像数据，无法保存录像。")

        if record_trajectory:
            _append_trajectory_derivatives(trajectory_samples)
            remaining_final = int(torch.sum(dirt_positions[0, :, 2] > 0.0).item())
            trajectory_result = {
                "schema_version": 5,
                "task": "normal_act_websocket_whiteboard_wipe",
                "controller": "ACT_WebSocket+Z_admittance",
                "policy_type": ws_bridge.policy_type,
                "policy_state_dim": ws_bridge.policy_state_dim,
                "policy_action_dim": ws_bridge.policy_action_dim,
                "sampling": {
                    "frequency_hz": TRAJECTORY_SAMPLE_HZ,
                    "control_period_s": env_cfg.sim.dt * env_cfg.decimation,
                    "physics_dt_s": env_cfg.sim.dt,
                    "physics_steps_per_sample": env_cfg.decimation,
                },
                "coordinate_frames": {
                    "trajectory_reference": "robot_base",
                    "robot_base": {
                        "description": "Isaac Lab robot articulation root for this dual-arm platform.",
                        "platform": "dual_arm",
                        "robot_asset": "scripts/dual_arm/urdf/dual_arm.usd",
                        "pose_in_world_field": "world_T_base",
                    },
                    "tcp_body": "arm_r_link_7",
                    "pose_matrix_convention": "T_A_B maps homogeneous coordinates from frame B into frame A.",
                    "quaternion_convention": "wxyz",
                },
                "joint_names": [robot.joint_names[int(index)] for index in right_joint_indices],
                "episode_index": episode_idx + 1,
                "success": bool(episode_success),
                "expert_takeover": bool(expert_takeover),
                "initial_dirt_points": int(initial_dirt_count),
                "final_remaining_dirt_points": remaining_final,
                "erased_ratio": float(
                    1.0 - remaining_final / initial_dirt_count
                    if initial_dirt_count > 0 else 1.0
                ),
                "simulated_duration_s": float(sim_elapsed),
                "samples": trajectory_samples,
            }
            trajectory_output = os.path.abspath(args_cli.record_trajectory)
            os.makedirs(os.path.dirname(trajectory_output) or ".", exist_ok=True)
            with open(trajectory_output, "w", encoding="utf-8") as output_file:
                json.dump(_to_builtin(trajectory_result), output_file, indent=2)
            print(f"[轨迹] 已保存一次 ACT 擦黑板轨迹: {trajectory_output}")

        if episode_success and not expert_takeover:
            status_str = "✅ 自主成功"
        elif episode_success and expert_takeover:
            status_str = f"🦾 专家救场完成 (模型判负，原因:{fail_reason})"
        else:
            status_str = f"❌ 最终失败 ({fail_reason})"

        wall_elapsed = time.perf_counter() - episode_wall_start
        realtime_factor = sim_elapsed / wall_elapsed if wall_elapsed > 0.0 else 0.0
        if fast_dp_deploy:
            final_remaining = int(torch.sum(dirt_positions[0][:, 2] > 0).item())
            erased_ratio = (
                1.0 - final_remaining / initial_dirt_count if initial_dirt_count > 0 else 1.0
            )
            dp_eval_records.append(
                {
                    "episode": episode_idx + 1,
                    "seed": episode_seed,
                    "success": bool(episode_success),
                    "task_completed": final_remaining == 0,
                    "fail_reason": fail_reason if not episode_success else "none",
                    "initial_dirt": int(initial_dirt_count),
                    "final_remaining": final_remaining,
                    "erased_ratio": erased_ratio,
                    "contact_time_sim_s": dp_contact_time,
                    "time_to_90_sim_s": progress_times["90"],
                    "time_to_95_sim_s": progress_times["95"],
                    "time_to_99_sim_s": progress_times["99"],
                    "sim_time_s": sim_elapsed,
                    "wall_time_s": wall_elapsed,
                    "realtime_factor": realtime_factor,
                    "steps": ep_step_count,
                }
            )
            if args_cli.dp_eval_output:
                output_dir = os.path.dirname(os.path.abspath(args_cli.dp_eval_output))
                os.makedirs(output_dir, exist_ok=True)
                with open(args_cli.dp_eval_output, "w", encoding="utf-8") as output_file:
                    json.dump(
                        {
                            "policy_type": ws_bridge.policy_type,
                            "episodes_requested": TOTAL_EPISODES,
                            "timeout_sim_s": args_cli.dp_eval_timeout,
                            "base_seed": args_cli.dp_eval_seed,
                            "episodes": dp_eval_records,
                        },
                        output_file,
                        indent=2,
                    )
            print(
                f"\n[回合 {episode_idx + 1}] 结束 | 结果: {status_str} "
                f"| 仿真用时: {sim_elapsed:.2f}s | 墙钟用时: {wall_elapsed:.2f}s "
                f"| 实时率: {realtime_factor:.2f}x | 擦除率: {erased_ratio * 100:.2f}%"
            )
        else:
            print(f"\n[回合 {episode_idx + 1}] 结束 | 结果: {status_str} | 仿真用时: {sim_elapsed:.2f}s")
        print(f"当前累计自主成功: {success_count} | 专家救场存入DAgger: {expert_takeover_success_count}")

        if not simulation_app.is_running():
            print("\n[WARN] 仿真窗口已关闭，提前终止测试。")
            TOTAL_EPISODES = episode_idx + 1
            break

    # 最终统计
    print("\n" + "=" * 60)
    print("📊 测试完成 —— 最终统计报告")
    print("=" * 60)
    print(f"总测试回合数: {TOTAL_EPISODES}")
    print(f"自主完全成功:   {success_count}")
    model_fails = TOTAL_EPISODES - success_count
    print(f"自主失败(含接管): {model_fails}")
    print(f"  ├─ 触发接管并由专家救场成功: {expert_takeover_success_count} 回合 (已存入 DAgger)")
    print(f"  ├─ 触发接管但专家依然失败:   {expert_fail_count} 回合")
    print(f"  ├─ 未接管但略超时失败:       {slight_timeout_fail_count} 回合")
    print(f"  └─ 未接管但硬超时失败:       {timeout_fail_count} 回合")
    print(f"自主成功率:       {success_count / TOTAL_EPISODES * 100:.2f} %")

    if len(success_times) > 0:
        avg_time = sum(success_times) / len(success_times)
        min_time = min(success_times)
        max_time = max(success_times)
        print(f"\n自主成功回合仿真用时统计:")
        print(f"  平均用时: {avg_time:.2f} s")
        print(f"  最快用时: {min_time:.2f} s")
        print(f"  最慢用时: {max_time:.2f} s")
    if fast_dp_deploy and dp_eval_records:
        completed_episodes = len(dp_eval_records)
        strict_successes = sum(record["success"] for record in dp_eval_records)
        completed_tasks = sum(record["task_completed"] for record in dp_eval_records)
        mean_erased_ratio = sum(record["erased_ratio"] for record in dp_eval_records) / completed_episodes
        mean_wall_time = sum(record["wall_time_s"] for record in dp_eval_records) / completed_episodes
        print("\nOfficial DP 纯策略评估:")
        print(f"  已完成回合: {completed_episodes}")
        print(f"  严格成功率: {strict_successes / completed_episodes * 100:.2f} %")
        print(f"  硬超时内任务完成率: {completed_tasks / completed_episodes * 100:.2f} %")
        print(f"  平均擦除率: {mean_erased_ratio * 100:.2f} %")
        print(f"  平均墙钟用时: {mean_wall_time:.2f} s")
        if args_cli.dp_eval_output:
            print(f"  JSON 结果: {os.path.abspath(args_cli.dp_eval_output)}")
    print("=" * 60 + "\n")

    ws_bridge.close()
    exit_watchdog = None
    if fast_dp_deploy and args_cli.dp_eval_episodes > 0:
        # Kit shutdown can occasionally hang after a finite headless evaluation.
        # Keep this isolated from normal/LingBot deployments.
        import threading

        exit_watchdog = threading.Timer(15.0, os._exit, args=(0,))
        exit_watchdog.daemon = True
        exit_watchdog.start()
    simulation_app.close()
    if exit_watchdog is not None:
        exit_watchdog.cancel()

if __name__ == "__main__":
    main()
