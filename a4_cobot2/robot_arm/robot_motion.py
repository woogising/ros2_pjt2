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

import rclpy
import DR_init

# Doosan bringup에서 사용하는 로봇 namespace/model과 맞춰야 합니다.
ROBOT_ID = 'dsr01'
ROBOT_MODEL = 'm0609'

# 테스트 모션 속도/가속도입니다. 실제 pick/place 연결 시 물체와 환경에 맞게 낮게 시작하세요.
VELOCITY = 10
ACC = 10

# test_small_assist_motion()에서 현재 TCP를 Z축으로 몇 mm 올릴지 정하는 값입니다.
TEST_Z_OFFSET_MM = 5.0

# dsr:
#   import DSR_ROBOT2 as dsr_module 한 뒤 저장되는 모듈 객체입니다.
# _node:
#   DSR_ROBOT2 내부 service/client 생성을 위해 필요한 별도 ROS2 node입니다.
# _emergency:
#   이 파일 내부의 소프트 stop 플래그입니다. 이미 실행 중인 로봇 API 명령을 즉시 끊는 실제 E-stop은 아닙니다.
dsr = None
_node = None
_emergency = False


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
    global dsr, _node

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

    dsr = dsr_module

    _wrap_motion_guard()

    try:
        dsr.DR_BASE = 0
        dsr.DR_TOOL = 1
    except Exception:
        pass

    _node.get_logger().info('DSR_ROBOT2 연결 완료')
    return dsr


# 현재 TCP 위치에서 Z축으로 아주 조금 올라갔다가 다시 내려오는 테스트 모션 함수
def test_small_assist_motion(object_name='unknown_object'):
    if dsr is None:
        raise RuntimeError('robot_motion.connect()가 먼저 호출되지 않았습니다.')

    if _emergency:
        return False

    current_pose = list(dsr.get_current_posx()[0])

    up_pose = current_pose.copy()
    up_pose[2] += TEST_Z_OFFSET_MM

    _node.get_logger().info(
        f'[test_motion] {object_name}: current_z={current_pose[2]:.2f}, '
        f'target_z={up_pose[2]:.2f}'
    )

    dsr.movel(up_pose, vel=VELOCITY, acc=ACC, ref=dsr.DR_BASE)

    if _emergency:
        return False

    dsr.movel(current_pose, vel=VELOCITY, acc=ACC, ref=dsr.DR_BASE)

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