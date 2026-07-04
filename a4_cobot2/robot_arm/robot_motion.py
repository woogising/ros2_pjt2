# 동작 테스트용 임시파일

# robot_motion.py

import rclpy
import DR_init

ROBOT_ID = 'dsr01'
ROBOT_MODEL = 'm0609'

VELOCITY = 10
ACC = 10
TEST_Z_OFFSET_MM = 5.0

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