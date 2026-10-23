import argparse
import os      
import h5py  
import math
import glob  
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R
import cv2
import time
import numpy as np
import zmq

# ======================================================================
# 1. 启动仿真应用
# ======================================================================
parser = argparse.ArgumentParser(description="Isaac Lab ZMQ OSC Env with Visuals (Dual Camera for VLA)")
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

# ======================================================================
# 📍 全局锚点与材质
# ======================================================================
DESK_X, DESK_Y, DESK_Z = 0.0, 0.0, 0.0  
BOARD_Z = DESK_Z
CAM_EULER_DEGREES = [-45.07, -34.968, -150.242]
_q = R.from_euler('XYZ', CAM_EULER_DEGREES, degrees=True).as_quat()
CAM_ROT_WXYZ = (_q[3], _q[0], _q[1], _q[2])


_wipe_rot_env = R.from_euler('z', 90, degrees=True).as_quat()
LOCKED_WIPE_QUAT = np.array([_wipe_rot_env[3], _wipe_rot_env[0], _wipe_rot_env[1], _wipe_rot_env[2]])

HOME_X = 0.05
HOME_Y = -0.4

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

# ======================================================================
# 2. 双臂机器人配置 (应用了新 USD 及两指夹爪)
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
            joint_names_expr=["arm_r_joint_.*", "arm_l_joint_.*"],
            stiffness=4000.0, damping=400.0,            
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["arm_r_joint_finger", "arm_l_joint_finger"],
            stiffness=800.0, damping=80.0,            
        ),
    },
)

# ======================================================================
# 3. 场景配置 (恢复 simple_room，挂载相机)
# ======================================================================
_tilt_q = R.from_euler('x', -20, degrees=True).as_quat()  # SciPy 返回 [x, y, z, w]
CHEST_CAM_ROT = (_tilt_q[3], _tilt_q[0], _tilt_q[1], _tilt_q[2])  # 转为 IsaacLab 需要的 (w, x, y, z)
@configclass
class WipeBoardSceneCfg(InteractiveSceneCfg):
    # 💥 完全恢复你原来的 simple_room 设定
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
    
    # 💥 全局视角：胸口主摄 (RealSense D435i 真实参数)
    # 💥 修改 offset 里的 rot
    chest_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/arm_base_link_d435i_2/chest_cam", 
        update_period=0.0, 
        height=720,  
        width=1280,  
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2,          
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 100.0)
        ),
        # 👇 只有这里变了，应用向下倾斜的四元数
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=CHEST_CAM_ROT), 
    )
    
    # 💥 局部视角：腕部特写 (RealSense D405 真实参数)
    # 还原 1280x720 分辨率与 87度 极广微距FOV
    wrist_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/arm_r_link_d405/wrist_cam", 
        update_period=0.0, 
        height=720,  # 真实高度
        width=1280,  # 真实宽度
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.0,          # 对应水平FOV 87°
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
    # 💥 将夹爪拆分为 Right 和 Left，强制固定维度顺序，完美对接你的 [[1.0, -1.0]] 目标！
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

# ======================================================================
# 4. 字迹生成函数
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

def generate_random_scribbles(device, num_envs=1):
    max_curves = 20
    points_per_curve = 150
    total_points = max_curves * points_per_curve
    positions = torch.zeros((num_envs, total_points, 3), device=device)
    z_base = BOARD_Z + 0.022

    for env_idx in range(num_envs):
        r = torch.rand(1).item()
        if r < 1.0 / 5.5:
            style = 1
        elif r < (1.0 + 3.0) / 5.5:
            style = 2
        else:
            style = 3

        env_curves = []

        if style == 1:
            num_lines = torch.randint(3, 6, (1,)).item()   
            for _ in range(num_lines):
                for attempt in range(20):  
                    x0 = WRITING_REGION_X[0] + torch.rand(1).item() * (WRITING_REGION_X[1] - WRITING_REGION_X[0])
                    y0 = WRITING_REGION_Y[0] + torch.rand(1).item() * (WRITING_REGION_Y[1] - WRITING_REGION_Y[0])
                    x1 = WRITING_REGION_X[0] + torch.rand(1).item() * (WRITING_REGION_X[1] - WRITING_REGION_X[0])
                    y1 = WRITING_REGION_Y[0] + torch.rand(1).item() * (WRITING_REGION_Y[1] - WRITING_REGION_Y[0])
                    dist = math.hypot(x1 - x0, y1 - y0)
                    if dist > 0.05:
                        break
                else:
                    x0, y0 = (WRITING_REGION_X[0] + 0.02, WRITING_REGION_Y[0] + 0.02)
                    x1, y1 = (WRITING_REGION_X[1] - 0.02, WRITING_REGION_Y[1] - 0.02)

                if torch.rand(1).item() > 0.5:
                    ctrl_spread = 0.001 + torch.rand(1).item() * 0.005
                else:
                    ctrl_spread = 0.01 + torch.rand(1).item() * 0.03

                curve_pts = _generate_stroke_points(
                    (x0, y0), (x1, y1), ctrl_spread, points_per_curve, z_base, device
                )
                env_curves.append(curve_pts)

        elif style == 2:
            diagram_region_x = DIAGRAM_REGION_X
            diagram_region_y = DIAGRAM_REGION_Y
            region_width  = WRITING_REGION_X[1] - WRITING_REGION_X[0]
            region_height = WRITING_REGION_Y[1] - WRITING_REGION_Y[0]

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
                num_shapes = torch.randint(5, 8, (1,)).item()   
            elif layout_style in ("formula_notes", "numbered_list"):
                num_shapes = torch.randint(1, 3, (1,)).item()   
            else:
                if torch.rand(1).item() > 0.5:
                    num_shapes = torch.randint(2, 4, (1,)).item()  
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

            start_y = WRITING_REGION_Y[1] - torch.rand(1).item() * 0.03
            min_lines = 2 if region_height > line_height * 2 else 1
            num_lines = max(min_lines, min(remaining_curves, int(region_height / line_height)))

            for line_idx in range(num_lines):
                if used_curves >= max_curves:
                    break
                row_y_base = start_y - line_idx * line_height
                if row_y_base < WRITING_REGION_Y[0] + 0.01:
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
                    line_x_start = WRITING_REGION_X[0] + 0.015
                    if line_idx < 5 and torch.rand(1).item() > 0.3:
                        bullet_x = WRITING_REGION_X[0] + 0.005
                        bullet_y = row_y_base + baseline_tilt * (bullet_x - WRITING_REGION_X[0])
                        bullet_pts = _generate_realistic_shape(
                            torch.randint(0, 3, (1,)).item(),
                            bullet_x, bullet_y, 0.005, points_per_curve, z_base, device
                        )
                        if used_curves < max_curves:
                            env_curves.append(bullet_pts)
                            used_curves += 1
                elif layout_style == "formula_notes":
                    line_x_start = WRITING_REGION_X[0] + torch.rand(1).item() * 0.02
                else:
                    line_x_start = WRITING_REGION_X[0] + torch.rand(1).item() * 0.01

                line_x_end = WRITING_REGION_X[1] - torch.rand(1).item() * 0.02
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
                    y_tilt = baseline_tilt * (cx - WRITING_REGION_X[0])
                    y_wave = baseline_wave_amp * math.sin(
                        baseline_wave_freq * math.pi * (cx - WRITING_REGION_X[0]) / region_width
                    )
                    current_y = row_y_base + y_tilt + y_wave

                    if layout_style == "formula_notes" and torch.rand(1).item() > 0.7:
                        curve_pts = _generate_stroke_points(
                            (cx, current_y),
                            (cx + ww, current_y + (torch.rand(1).item()-0.5) * char_height * 0.8),
                            char_height * 0.5, points_per_curve, z_base, device
                        )
                    else:
                        curve_pts, _ = _make_word(
                            cx, current_y, ww, char_height, points_per_curve, z_base, device
                        )
                    env_curves.append(curve_pts)
                    used_curves += 1
                    cx += ww + word_gap

            if layout_style == "underlined_text" and used_curves < max_curves:
                ul_y = start_y + line_height * 0.3
                ul_pts = _make_underline(
                    WRITING_REGION_X[0], WRITING_REGION_X[0] + region_width * 0.7,
                    ul_y, points_per_curve, z_base, device
                )
                env_curves.append(ul_pts)
                used_curves += 1

        else:  # style == 3
            diagram_region_x = DIAGRAM_REGION_X
            diagram_region_y = DIAGRAM_REGION_Y
            is_diagram_mode = torch.rand(1).item() > 0.5

            if is_diagram_mode:
                num_shapes = torch.randint(3, 6, (1,)).item()
                for i in range(num_shapes):
                    shape_type = torch.randint(0, 3, (1,)).item()
                    cx = diagram_region_x[0] + torch.rand(1).item() * (diagram_region_x[1] - diagram_region_x[0])
                    cy = diagram_region_y[0] + torch.rand(1).item() * (diagram_region_y[1] - diagram_region_y[0])
                    radius = torch.rand(1).item() * 0.03 + 0.02
                    curve_pts = _generate_parametric_shape(shape_type, cx, cy, radius, points_per_curve, device)
                    env_curves.append(curve_pts)
                remaining_curves = max_curves - num_shapes
            else:
                remaining_curves = torch.randint(15, max_curves + 1, (1,)).item()
            for i in range(remaining_curves):
                x_center = WRITING_REGION_X[0] + torch.rand(1).item() * (WRITING_REGION_X[1] - WRITING_REGION_X[0])
                y_pos = WRITING_REGION_Y[0] + torch.rand(1).item() * (WRITING_REGION_Y[1] - WRITING_REGION_Y[0])
                Ax = 0.005 + torch.rand(1).item() * 0.015
                kx = torch.randint(10, 25, (1,), device=device)
                t = torch.linspace(0, 1, points_per_curve, device=device)
                x = torch.full_like(t, x_center) + Ax * torch.sin(kx.float() * torch.pi * t)
                y = torch.full_like(t, y_pos) + (torch.rand(1).item() - 0.5) * 0.005
                z = torch.full_like(x, BOARD_Z + 0.022)
                env_curves.append(torch.stack([x, y, z], dim=-1))

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

    return positions

# ======================================================================
# 5. ZMQ 通信桥梁
# ======================================================================
class IsaacZMQBridge:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://127.0.0.1:5556")
        self.latest_osc_action = None 
        self.current_cmd = "idle"            
        self.episode_name = "default_ep"     
        print("[INFO]: ZMQ Bridge Initialized on port 5556")

    def wait_for_action_and_reply(self, state_dict, simulation_app):
        self.socket.send_pyobj(state_dict)
        while simulation_app.is_running():
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                if isinstance(msg, dict) and 'cmd' in msg:
                    self.current_cmd = msg['cmd']
                    if self.current_cmd == 'step':
                        self.latest_osc_action = msg['target_pose']
                    elif self.current_cmd == 'save':
                        self.episode_name = msg.get('episode_name', 'default_ep')
                else:
                    if 'target_pose' in msg:
                        self.current_cmd = 'step'
                        self.latest_osc_action = msg['target_pose']
                break 
            except zmq.Again:
                simulation_app.update()

    def close(self):
        self.socket.close()
        self.context.term()

# ======================================================================
# 6. 主程序
# ======================================================================
def main():
    env_cfg = VLAEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    obs, info = env.reset()
    zmq_bridge = IsaacZMQBridge()
    robot = env.scene["robot"]
    
    chest_camera = env.scene["chest_camera"] 
    wrist_camera = env.scene["wrist_camera"]
    
    DATASET_DIR = os.environ.get("WHITEBOARD_RAW_DATA", "vla_dataset")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    episode_buffer = {"image_primary": [], "image_wrist": [], "depth_primary_mm": [], "joint_pos": [], "pose7": [], "wrench": [], "action": []}
    
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DirtMarkers",
        markers={"dirt": sim_utils.CuboidCfg(size=(0.004, 0.004, 0.001), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.1), roughness=1.0))},
    )
    dirt_markers = VisualizationMarkers(marker_cfg)

    current_dirt_positions = None

    def move_scene_objects(board_off_x, board_off_y):
        board_pos = (BOARD_ORIGIN[0] + board_off_x, BOARD_ORIGIN[1] + board_off_y, BOARD_ORIGIN[2])
        env.scene["whiteboard"].set_world_poses(
            torch.tensor([board_pos], device=env.device, dtype=torch.float),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float)
        )
        frame_names = ["frame_top", "frame_bottom", "frame_left", "frame_right"]
        origins = [FRAME_TOP_ORIGIN, FRAME_BOTTOM_ORIGIN, FRAME_LEFT_ORIGIN, FRAME_RIGHT_ORIGIN]
        for name, (ox, oy, oz) in zip(frame_names, origins):
            new_pos = (ox + board_off_x, oy + board_off_y, oz)
            env.scene[name].set_world_poses(
                torch.tensor([new_pos], device=env.device, dtype=torch.float),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float)
            )

    def refresh_environment_state():
        nonlocal current_dirt_positions
        board_off_x = np.random.uniform(-0.08, 0.08)
        board_off_y = np.random.uniform(-0.1, 0.1)

        move_scene_objects(board_off_x, board_off_y)

        new_dirt = generate_random_scribbles(env.device, env.num_envs)
        current_dirt_positions = new_dirt.clone()

        current_dirt_positions[..., 0] += board_off_x
        current_dirt_positions[..., 1] += board_off_y
        dirt_markers.visualize(translations=current_dirt_positions[0])

    refresh_environment_state()

    body_indices, _ = robot.find_bodies("arm_r_link_7") 
    ee_idx = body_indices[0]
    
    arm_links = [f"arm_{side}_link_{i}" for side in ['r', 'l'] for i in range(2, 7)]
    check_indices, _ = robot.find_bodies(arm_links)
    
    ep_step_count = 0

    l_arm_target = torch.zeros((1, 7), device=env.device)
    gripper_2d_target = torch.tensor([[0.6, -1.0]], device=env.device, dtype=torch.float32)

    def latch_current_pose():
        r_pos_w = robot.data.root_pos_w.clone()
        r_quat_w = robot.data.root_quat_w.clone()
        e_pos_w = robot.data.body_pos_w[:, ee_idx, :].clone()
        e_quat_w = robot.data.body_quat_w[:, ee_idx, :].clone()
        e_pos_b, e_quat_b = math_utils.subtract_frame_transforms(r_pos_w, r_quat_w, e_pos_w, e_quat_w)
        r_arm_target = torch.cat([e_pos_b, e_quat_b], dim=-1)
        return torch.cat([r_arm_target, l_arm_target, gripper_2d_target], dim=-1)

    def fly_to_home_pose():
        target_pos_w = torch.tensor([[HOME_X, HOME_Y, BOARD_Z + 0.35]], device=env.device, dtype=torch.float)
        target_quat_w = torch.tensor([LOCKED_WIPE_QUAT], device=env.device, dtype=torch.float)
        for _ in range(40):
            root_pos_w = robot.data.root_pos_w.clone()
            root_quat_w = robot.data.root_quat_w.clone()
            pos_b, quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
            action = torch.cat([torch.cat([pos_b, quat_b], dim=-1), l_arm_target, gripper_2d_target], dim=-1)
            env.step(action)

    fly_to_home_pose()
    zmq_bridge.locked_target = latch_current_pose()

    print("[INFO]: OSC Environment Ready. Awaiting synchronous commands...")
    connected = False
    while simulation_app.is_running() and not connected:
        try:
            msg = zmq_bridge.socket.recv_pyobj(flags=zmq.NOBLOCK)
            connected = True
            print("[INFO]: Controller connected! Simulation is LIVE.")
        except zmq.Again:
            env.step(zmq_bridge.locked_target)

    while simulation_app.is_running():
        all_body_pos_w = robot.data.body_pos_w[0]
        
        ee_pos_w = all_body_pos_w[ee_idx].unsqueeze(0)
        ee_quat_w = robot.data.body_quat_w[:, ee_idx, :]
        pose7_world = np.concatenate([ee_pos_w.cpu().numpy()[0], ee_quat_w.cpu().numpy()[0]])
        joint_pos = robot.data.joint_pos[:, :7].cpu().numpy()[0]
        
        net_forces = env.scene["contact_forces"].data.net_forces_w[0, 0]
        net_torques = torch.zeros(3, device=env.device, dtype=net_forces.dtype)
        wrench_6d = torch.cat([net_forces, net_torques]).cpu().numpy()
        contact_force_z = abs(net_forces[2].item())
        
        is_collision = False
        if ep_step_count > 150:
            check_pos = all_body_pos_w[check_indices] 
            hit_table = torch.any(check_pos[:, 2] < 0.03).item()
            hit_pedestal_x = (check_pos[:, 0] > -0.295) & (check_pos[:, 0] < -0.105)
            hit_pedestal_y = (check_pos[:, 1] > -0.095) & (check_pos[:, 1] < 0.095)
            hit_pedestal_z = (check_pos[:, 2] < 0.42)
            hit_pedestal = torch.any(hit_pedestal_x & hit_pedestal_y & hit_pedestal_z).item()
            hit_board = torch.any(check_pos[:, 0] > 0.27).item()
            is_collision = hit_table or hit_pedestal or hit_board
        
        valid_dirt = current_dirt_positions[0][current_dirt_positions[0][:, 2] > 0]
        
        rgb_primary_data = chest_camera.data.output["rgb"][0].cpu().numpy()[..., :3]
        rgb_wrist_data = wrist_camera.data.output["rgb"][0].cpu().numpy()[..., :3]
        
        state_data = {
            "pose7": pose7_world,
            "joint_pos": joint_pos,
            "wrench": wrench_6d,
            "rgb_primary": rgb_primary_data,   
            "rgb_wrist": rgb_wrist_data,
            "dirt_points": valid_dirt.cpu().numpy() if len(valid_dirt) > 0 else None,
            "collision": is_collision 
        }
        
        zmq_bridge.wait_for_action_and_reply(state_data, simulation_app)

        if zmq_bridge.current_cmd == 'step':
            osc_target_w = torch.tensor(np.array([zmq_bridge.latest_osc_action]), device=env.device, dtype=torch.float32)
            target_pos_w = osc_target_w[:, :3]
            target_quat_w = osc_target_w[:, 3:]
            root_pos_w = robot.data.root_pos_w.clone()
            root_quat_w = robot.data.root_quat_w.clone()
            
            target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
            osc_target_b = torch.cat([target_pos_b, target_quat_b], dim=-1)
            
            zmq_bridge.locked_target = torch.cat([osc_target_b, l_arm_target, gripper_2d_target], dim=-1)
            
            _, jpg_primary = cv2.imencode('.jpg', rgb_primary_data[..., ::-1])
            _, jpg_wrist = cv2.imencode('.jpg', rgb_wrist_data[..., ::-1])
            
            depth_array = chest_camera.data.output["distance_to_camera"][0].cpu().numpy()
            depth_mm = np.clip(depth_array * 1000.0, 0, 65535).astype(np.uint16)
            
            episode_buffer["image_primary"].append(np.frombuffer(jpg_primary, dtype=np.uint8))
            episode_buffer["image_wrist"].append(np.frombuffer(jpg_wrist, dtype=np.uint8))
            episode_buffer["depth_primary_mm"].append(depth_mm)
            
            episode_buffer["joint_pos"].append(joint_pos)
            episode_buffer["pose7"].append(pose7_world)
            episode_buffer["wrench"].append(wrench_6d)
            episode_buffer["action"].append(zmq_bridge.latest_osc_action)
            
        elif zmq_bridge.current_cmd == 'save':
            print(f"[DATA]: 正在将多模态数据写入硬盘 -> {zmq_bridge.episode_name}.hdf5 ...")
            file_path = os.path.join(DATASET_DIR, f"{zmq_bridge.episode_name}.hdf5")
            with h5py.File(file_path, 'w') as f:
                vlen_uint8 = h5py.vlen_dtype(np.dtype('uint8'))
                f.create_dataset('obs/image_primary', data=episode_buffer["image_primary"], dtype=vlen_uint8)
                f.create_dataset('obs/image_wrist', data=episode_buffer["image_wrist"], dtype=vlen_uint8)
                f.create_dataset('obs/distance_to_camera', data=np.array(episode_buffer["depth_primary_mm"]), compression="gzip")
                
                f.create_dataset('obs/joint_pos', data=np.array(episode_buffer["joint_pos"], dtype=np.float32))
                f.create_dataset('obs/pose7', data=np.array(episode_buffer["pose7"], dtype=np.float32))
                f.create_dataset('obs/wrench', data=np.array(episode_buffer["wrench"], dtype=np.float32))
                f.create_dataset('action/target_pose', data=np.array(episode_buffer["action"], dtype=np.float32))
                f.create_dataset('obs/ft_sensor_pose_fb', data=np.array(episode_buffer["pose7"], dtype=np.float32))
                f.create_dataset('language_instruction', data="wipe the whiteboard clean")
                
            print(f"[DATA]: 保存完成！共录制了 {len(episode_buffer['image_primary'])} 帧。")
            for key in episode_buffer: episode_buffer[key].clear()
            
            ep_step_count = 0
            refresh_environment_state()          
            fly_to_home_pose()                   
            zmq_bridge.locked_target = latch_current_pose()
            zmq_bridge.current_cmd = 'idle'
            print("[INFO]: 场景已热刷新（白板已随机化，字迹更新），已回到 Home Pose，等待新指令！")
            continue

        elif zmq_bridge.current_cmd == 'abort':
            print(f"[WARN]: ⚠️ 收到控制端重置信号！废弃当前数据，立刻刷新白板字迹！")
            for key in episode_buffer: episode_buffer[key].clear()
            env.reset()
            ep_step_count = 0
            refresh_environment_state()
            fly_to_home_pose()
            zmq_bridge.locked_target = latch_current_pose()
            zmq_bridge.current_cmd = 'idle'
            continue

        obs, info = env.step(zmq_bridge.locked_target)
        ep_step_count += 1
        
        if zmq_bridge.current_cmd == 'step':
            ee_quat_inv = ee_quat_w.clone()
            ee_quat_inv[:, 1:] = -ee_quat_inv[:, 1:]  
            
            rel_pos_w = current_dirt_positions - ee_pos_w.unsqueeze(1)
            ee_quat_inv_expanded = ee_quat_inv.unsqueeze(1).expand(-1, current_dirt_positions.shape[1], -1)
            local_pos = math_utils.quat_apply(ee_quat_inv_expanded, rel_pos_w)
            
            mask_x = torch.abs(local_pos[..., 0]) < 0.04
            mask_y = torch.abs(local_pos[..., 1] - 0.02) < 0.03
            mask_z = (local_pos[..., 2] < -0.20) & (local_pos[..., 2] > -0.225)
            
            FORCE_THRESHOLD = 5.0 
            if contact_force_z > FORCE_THRESHOLD:
                erase_mask = mask_x & mask_y & mask_z
                current_dirt_positions[erase_mask] = torch.tensor([0.0, 0.0, -10.0], device=env.device)
                dirt_markers.visualize(translations=current_dirt_positions[0])

    zmq_bridge.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
