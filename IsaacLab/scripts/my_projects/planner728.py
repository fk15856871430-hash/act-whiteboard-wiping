import zmq
import numpy as np
import sys 
import time
import os      
import glob
from scipy.spatial.transform import Rotation as R

# =========================================================
# 🚨 异常状态定义
# =========================================================
class CollisionException(Exception): pass
class TimeoutException(Exception): pass

# =========================================================
# 📍 全局锁死姿态与原点
# =========================================================
# 已经应用 90 度偏航补偿
_wipe_rot = R.from_euler('z', 90, degrees=True).as_quat()  
LOCKED_WIPE_QUAT = np.array([_wipe_rot[3], _wipe_rot[0], _wipe_rot[1], _wipe_rot[2]])  

HOME_X = 0.05
HOME_Y = -0.4
SURFACE_Z_EST = 0.225      
SAFE_RETURN_Z = 0.40      

def get_next_episode_index(dataset_dir="vla_dataset"):
    if not os.path.exists(dataset_dir):
        return 0
    files = glob.glob(os.path.join(dataset_dir, "episode_*"))
    if not files:
        return 0
    indices = []
    for f in files:
        try:
            base = os.path.basename(f)
            num_str = base.split('_')[1].split('.')[0]
            if num_str.isdigit():
                indices.append(int(num_str))
        except (IndexError, ValueError):
            continue
    return max(indices) + 1 if indices else 0

class IsaacZMQClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:5556")
        self.socket.send_pyobj({'cmd': 'sync'})
        self.initial_state = self.socket.recv_pyobj()

    def step(self, pose7):
        self.socket.send_pyobj({'cmd': 'step', 'target_pose': pose7})
        return self.socket.recv_pyobj()

    def save_and_reset(self, episode_name):
        self.socket.send_pyobj({'cmd': 'save', 'episode_name': episode_name})
        return self.socket.recv_pyobj()

    def abort_and_reset(self):
        self.socket.send_pyobj({'cmd': 'abort'})
        return self.socket.recv_pyobj()

def generate_smooth_transition(start_pos, end_pos, speed_factor=400):
    dist = np.linalg.norm(end_pos - start_pos)
    steps = max(int(dist * speed_factor), 20) 
    t = np.linspace(0, np.pi, steps)
    weights = (1 - np.cos(t)) / 2.0
    return start_pos + weights[:, np.newaxis] * (end_pos - start_pos)

# =========================================================
# 3. 自动化数据采集引擎
# =========================================================
def main():
    MAX_EPISODES = 500 
    MAX_EPISODE_DURATION = 90.0 
    
    TARGET_FORCE = 6.0       
    FORCE_KP = 0.00015       
    FORCE_KD = 0.00005       
    MAX_Z_SPEED = 0.0015     
    DEADBAND = 1.0           
    MAX_STEP_DIST = 0.00252   
    
    report_stats = {'success': 0, 'collision': 0, 'timeout': 0}
    
    print("⏳ 正在连接双臂仿真引擎...")
    client = IsaacZMQClient()
    print("✅ 自动打图流水线启动！【动态弓字形引力引擎】已就绪！")
    
    state = client.initial_state
    current_quat = LOCKED_WIPE_QUAT
    cmd_pos = state['pose7'][:3].copy()
    
    for _ in range(10): 
        client.step(np.concatenate([cmd_pos, current_quat]))
    
    DATASET_DIR = os.environ.get("WHITEBOARD_RAW_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vla_dataset"))
    start_ep = get_next_episode_index(DATASET_DIR)
    
    current_ep = start_ep
    target_ep = start_ep + MAX_EPISODES

    while current_ep < target_ep:
        ep_name = f"episode_{current_ep:04d}"
        print(f"\n" + "="*50 + f"\n🎬 开始录制动态抗遮挡数据: {ep_name}\n" + "="*50)

        episode_start_time = time.time()  
        is_down = False  
        sweep_dir = 1    # 1 代表向 +Y 方向擦拭，-1 代表向 -Y 方向擦拭

        def safe_step(pos):
            nonlocal state
            if time.time() - episode_start_time > MAX_EPISODE_DURATION:
                raise TimeoutException()
            st = client.step(np.concatenate([pos, LOCKED_WIPE_QUAT]))
            if st.get('collision', False):
                raise CollisionException()
            state = st 
            return st

        try:
            filtered_force = 0.0
            prev_error = 0.0

            while True:
                dirt_points = state.get('dirt_points')
                
                # ==========================================
                # 🎓 完美退场逻辑
                # ==========================================
                if dirt_points is None or len(dirt_points) == 0:
                    print("\n🎉 白板已绝对洁净！执行退场动作...")
                    
                    safe_return_pos = cmd_pos.copy()
                    safe_return_pos[2] = SAFE_RETURN_Z 
                    for pt in generate_smooth_transition(cmd_pos, safe_return_pos, speed_factor=150):
                        state = safe_step(pt)
                        cmd_pos = pt
                        
                    home_pos = np.array([HOME_X, HOME_Y, SAFE_RETURN_Z])
                    for pt in generate_smooth_transition(cmd_pos, home_pos, speed_factor=100):
                        state = safe_step(pt)
                        cmd_pos = pt
                        
                    for _ in range(5):
                        safe_step(cmd_pos)
                        
                    state = client.save_and_reset(ep_name)
                    cmd_pos = state['pose7'][:3].copy()
                    for _ in range(5): safe_step(cmd_pos)
                    
                    report_stats['success'] += 1
                    current_ep += 1
                    break 

                # ==========================================
                # 🔥 动态引力寻路 (垂直弓字形：前后擦拭，从右往左推进)
                # ==========================================
                dirt_xy = dirt_points[:, :2]
                
                # 提取白板擦的“真实物理坐标”
                eraser_offset_world = np.array([-0.02, 0.0]) 
                eraser_xy = cmd_pos[:2] + eraser_offset_world
                
                # 1. 动态剥离出“最右侧”的污渍带 (Y轴坐标最小的一带)
                min_y = np.min(dirt_xy[:, 1])
                band_width = 0.045  # 单次纵向扫描的带度 (4.5 厘米宽)
                band_mask = dirt_xy[:, 1] < (min_y + band_width)
                band_points = dirt_xy[band_mask]
                
                # 2. 锁定这一带的前后极值(X轴)和中心Y
                band_min_x = np.min(band_points[:, 0])
                band_max_x = np.max(band_points[:, 0])
                band_mean_y = np.mean(band_points[:, 1])
                
                # 3. 弓字形引力状态机：沿着 X 轴 (前后) 来回擦拭
                if sweep_dir == 1:
                    # 往 +X 极点拉扯（推向远端，略微超出去保证边缘擦净）
                    target_xy = np.array([band_max_x + 0.015, band_mean_y])
                    # 如果海绵已经擦到最远端，立刻换向
                    if eraser_xy[0] >= band_max_x - 0.005:
                        sweep_dir = -1
                else:
                    # 往 -X 极点拉扯（拉回近端）
                    target_xy = np.array([band_min_x - 0.015, band_mean_y])
                    # 如果海绵已经拉到最近端，立刻换向
                    if eraser_xy[0] <= band_min_x + 0.005:
                        sweep_dir = 1
                # ==========================================
                # 🚄 平滑移动与力控
                # ==========================================
                move_vec = target_xy - eraser_xy
                dist_to_target = np.linalg.norm(move_vec)

                # XY 平面追踪
                if dist_to_target > MAX_STEP_DIST:
                    step_vec = (move_vec / dist_to_target) * MAX_STEP_DIST
                else:
                    step_vec = move_vec
                
                cmd_pos[0] += step_vec[0]
                cmd_pos[1] += step_vec[1]

                # Z 轴力控 PID 与探底
                wrench = state.get('wrench')
                raw_force = abs(wrench[2]) if wrench is not None else 0.0
                filtered_force = 0.6 * raw_force + 0.4 * filtered_force

                if not is_down:
                    if filtered_force >= 3.0 or cmd_pos[2] <= (SURFACE_Z_EST - 0.04):
                        is_down = True
                        prev_error = filtered_force - TARGET_FORCE
                    else:
                        cmd_pos[2] -= 0.0015
                else:
                    force_error = filtered_force - TARGET_FORCE
                    delta_error = force_error - prev_error

                    if abs(force_error) > DEADBAND:
                        delta_z = FORCE_KP * force_error + FORCE_KD * delta_error
                        delta_z = max(min(delta_z, MAX_Z_SPEED), -MAX_Z_SPEED)
                        cmd_pos[2] += delta_z
                        
                    cmd_pos[2] = max(min(cmd_pos[2], SURFACE_Z_EST + 0.03), SURFACE_Z_EST - 0.04)
                    prev_error = force_error

                state = safe_step(cmd_pos)
                sys.stdout.write(f"\r🔴 [动态弓字扫描] 剩余字迹: {len(dirt_points):>4} | 实时接触力: {filtered_force:>5.2f} N")
                sys.stdout.flush()

        except CollisionException:
            print("\n🚨🚨 [碰撞] 当前回合废弃，立刻热重置场景...")
            report_stats['collision'] += 1
            state = client.abort_and_reset()
            cmd_pos = state['pose7'][:3].copy()
            for _ in range(5): client.step(np.concatenate([cmd_pos, LOCKED_WIPE_QUAT]))
            continue
            
        except TimeoutException:
            print(f"\n❌ [超时] 耗时过长，作废该回合...")
            report_stats['timeout'] += 1
            state = client.abort_and_reset()
            cmd_pos = state['pose7'][:3].copy()
            for _ in range(5): client.step(np.concatenate([cmd_pos, LOCKED_WIPE_QUAT]))
            continue
                
    client.socket.send_pyobj({'cmd': 'shutdown'})
    client.socket.recv_pyobj()
    print("\n流水线安全关闭。")

if __name__ == "__main__":
    main()
