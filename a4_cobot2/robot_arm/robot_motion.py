# ============================================================
# robot_arm/robot_motion.py
# 역할:
#   - Doosan DSR_ROBOT2 API와 직접 연결되는 저수준 로봇 동작 함수 모음입니다.
# 현재 상태:
#   - 실제 pick/place는 아직 없고, Z축 소폭 이동 테스트 함수만 있습니다.
# 주의:
#   - 이 파일은 로봇 API를 직접 호출하므로 안전 정지, 단위(mm/m), 좌표계 변환을 가장 조심해야 합니다.
# ============================================================
# 동작 테스트용 임시파일

# robot_motion.py

import os

import numpy as np
import rclpy
import DR_init
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation





GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"


# Doosan bringup에서 사용하는 로봇 namespace/model과 맞춰야 합니다.
ROBOT_ID = 'dsr01'
ROBOT_MODEL = 'm0609'

# 테스트 모션 속도/가속도입니다. 실제 pick/place 연결 시 물체와 환경에 맞게 낮게 시작하세요.
VELOCITY = 50
ACC = 50

# pick/place 시 물체 바로 위에서 접근/후퇴하기 위한 높이 오프셋(mm)
APPROACH_Z_OFFSET_MM = 50.0

# pick 시 최종 하강 높이 보정(mm). 물건 위를 살짝 잡으면(감지 z가 높으면) 값을 키워 더 내려가 잡는다.
PICK_Z_OFFSET_MM = 30.0

# 탑다운 파지 시 그리퍼 자세(posx의 rx, ry, rz).
# 임시 값이므로 실제 집기 자세로 반드시 교체해야 한다.
GRASP_ORIENTATION = [90, 180, 90]

# gripper<-camera 캘리브레이션 행렬(mm 단위). eye-in-hand 카메라 외부 파라미터입니다.
NPY_PATH = os.path.join(
    get_package_share_directory('a4_cobot2'), 'resource', 'T_gripper2camera.npy'
)

# 작업공간을 나눠 찍는 3개 관측 자세(joint, degree): [중앙, 왼쪽, 오른쪽]
SCAN_POSES_DEG = [
    [3.86, 30.35, 38.15, -0.07, 111.245, -88.86],
    [3.9, 30.69, 37.61, -0.14, 126.49, -88.94],
    [3.86, 30.35, 38.15, -0.07, 91.71, -87.99],
]

# dsr:
#   import DSR_ROBOT2 as dsr_module 한 뒤 저장되는 모듈 객체입니다.
# _node:
#   DSR_ROBOT2 내부 service/client 생성을 위해 필요한 별도 ROS2 node입니다.
# _emergency:
#   이 파일 내부의 소프트 stop 플래그입니다. 이미 실행 중인 로봇 API 명령을 즉시 끊는 실제 E-stop은 아닙니다.
dsr = None
_node = None
_emergency = False
gripper = None


# 비상정지 요청 상태로 바꾸는 함수
def request_stop():
    global _emergency
    _emergency = True


# 비상정지 요청 상태를 해제하는 함수
def clear_stop():
    global _emergency
    _emergency = False


# 현재 비상정지 요청 상태인지 확인하는 함수
def is_stopped():
    return _emergency


# 비상정지 상태에서 movel/movej가 실행되지 않도록 감싸는 함수
def _wrap_motion_guard():
    global dsr

    _orig_movel = dsr.movel
    _orig_movej = dsr.movej

    def movel_guarded(*args, **kwargs):
        if _emergency:
            return None
        return _orig_movel(*args, **kwargs)

    def movej_guarded(*args, **kwargs):
        if _emergency:
            return None
        return _orig_movej(*args, **kwargs)

    dsr.movel = movel_guarded
    dsr.movej = movej_guarded


# DSR_ROBOT2 전용 ROS2 노드를 만들고 Doosan 로봇 API를 연결하는 함수
def connect():
    global dsr, _node, gripper

    if dsr is not None:
        return dsr

    if not rclpy.ok():
        raise RuntimeError('rclpy.init() 이후에 robot_motion.connect()를 호출해야 합니다.')

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    # 중요:
    # DSR_ROBOT2는 내부에서 DR_init.__dsr__node.create_client(...)를 사용하므로
    # import DSR_ROBOT2 전에 반드시 __dsr__node를 넣어야 한다.
    _node = rclpy.create_node('robot_motion_dsr', namespace=ROBOT_ID)
    DR_init.__dsr__node = _node

    import DSR_ROBOT2 as dsr_module
    from robot_arm.onrobot import RG

    dsr = dsr_module
    gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
    _wrap_motion_guard()

    try:
        dsr.DR_BASE = 0
        dsr.DR_TOOL = 1
    except Exception:
        pass

    _node.get_logger().info('DSR_ROBOT2 연결 완료')
    return dsr


# posx [x, y, z, rx, ry, rz](mm, ZYZ deg)를 base<-gripper 4x4 변환 행렬로 만드는 함수
def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    rotation = Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [x, y, z]
    return matrix


# 현재 로봇 자세에서 base<-camera 4x4 변환 행렬을 계산하는 함수
def get_base_to_camera_matrix():
    if dsr is None:
        raise RuntimeError('robot_motion.connect()가 먼저 호출되지 않았습니다.')

    gripper2cam = np.load(NPY_PATH)
    base2gripper = get_robot_pose_matrix(*dsr.get_current_posx()[0])
    return base2gripper @ gripper2cam


# 지정한 관측 자세(joint, degree)로 이동하는 함수
def move_to_scan_pose(pose_deg):
    if dsr is None:
        raise RuntimeError('robot_motion.connect()가 먼저 호출되지 않았습니다.')

    if _emergency:
        return False
    grip_open()
    dsr.movej(pose_deg, vel=VELOCITY, acc=ACC)
    return not _emergency


# 그리퍼를 여는 함수 (실제 그리퍼 제어 API 연결 필요)
def grip_open(force=200):
    gripper.open_gripper(force)
    pass


# 그리퍼를 닫는 함수 (실제 그리퍼 제어 API 연결 필요)
def grip_close(force=200):
    gripper.close_gripper(force)
    pass


# 물체를 pick_position에서 집어 place_position으로 옮기는 pick-and-place 함수.
# pick_position, place_position은 robot base 좌표계(mm) {x, y, z} dict이다.
def pick_and_place_object(object_name, pick_position, place_position, object_angle):
    if dsr is None:
        raise RuntimeError('robot_motion.connect()가 먼저 호출되지 않았습니다.')

    if _emergency:
        return False

    if pick_position is None or place_position is None:
        raise ValueError(f'{object_name}의 pick 또는 place 위치가 없습니다.')

    # base 좌표(mm) {x, y, z}에 고정 그리퍼 자세를 붙여 posx 6요소를 만든다.
    pick_pose = [
        float(pick_position['x']), float(pick_position['y']), float(pick_position['z'])
    ] + GRASP_ORIENTATION
    place_pose = [
        float(place_position['x']), float(place_position['y']), float(pick_position['z'])
    ] + GRASP_ORIENTATION

    # 감지 z가 살짝 높아 물건 위를 잡을 때, 이 값만큼 더 내려가서 잡는다. (mm)
    pick_pose[2] -= PICK_Z_OFFSET_MM

    # 물체 바로 위 접근 지점(집기 전/놓기 전 안전 높이)
    pick_approach = pick_pose.copy()
    pick_approach[2] += APPROACH_Z_OFFSET_MM

    place_approach = place_pose.copy()
    place_approach[2] += APPROACH_Z_OFFSET_MM

    _node.get_logger().info(f'[pick_and_place] {object_name} 시작')

    grip_open()

    # ① 물체 위로 접근 (절대, 고정 자세)
    dsr.movel(pick_approach, vel=VELOCITY, acc=ACC, ref=dsr.DR_BASE)
    if _emergency:
        return False

    # ② 접근 위치에서 각도에 맞춰 손목 회전 (툴Z 상대, movel=동기)
    if object_angle is not None:

        rot = object_angle - 90 if object_angle >= 0 else object_angle + 90
        dsr.amovel([0, 0, 0, 0, 0, rot], vel=VELOCITY, acc=ACC, ref=dsr.DR_TOOL)

        if _emergency:
            return False

    # ③ 상대로 하강 (회전 유지) — 툴Z로 접근높이만큼 내려감
    dsr.movel([0, 0, APPROACH_Z_OFFSET_MM, 0, 0, 0], vel=VELOCITY - 20, acc=ACC - 20, ref=dsr.DR_TOOL)
    if _emergency:
        return False

    grip_close()
    dsr.wait(2.0)
    # 들어 올린 뒤 목표 위치 위로 이동 → 내려놓기
    dsr.movel(pick_approach, vel=VELOCITY, acc=ACC, ref=dsr.DR_BASE)
    if _emergency:
        return False

    dsr.movel(place_approach, vel=VELOCITY, acc=ACC, ref=dsr.DR_BASE)
    if _emergency:
        return False

    dsr.movel(place_pose, vel=VELOCITY, acc=ACC, ref=dsr.DR_BASE)
    if _emergency:
        return False

    grip_open()

    # 안전 높이로 복귀
    dsr.movel(place_approach, vel=VELOCITY, acc=ACC, ref=dsr.DR_BASE)

    return not _emergency


# 실제 로봇 이동을 중단하기 위한 stop 플래그를 세우는 함수
def safe_stop():
    request_stop()


# DSR 전용 노드를 종료하는 함수
def shutdown():
    global _node

    if _node is not None:
        _node.destroy_node()
        _node = None