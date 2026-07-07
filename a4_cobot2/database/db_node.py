# ============================================================
# database/db_node.py
# 역할:
#   - 기존 ROS2 topic을 구독해서 SQLite DB에 작업 이력을 저장합니다.
#   - DB 저장을 위해 새 topic을 만들지 않습니다.
# 구독 topic:
#   - /task_command_raw
#   - /task_command
#   - /task_status
#   - /user_notice
#   - /safety_command
#   - /safety_state
#   - /emergency_stop
# 저장 방식:
#   - /task_command가 들어오면 run_id를 만들고 task_runs를 시작합니다.
#   - 이후 상태/안내/safety 이벤트를 current_run_id에 연결합니다.
#   - /task_status가 idle 또는 종료성 상태가 되면 task_runs를 종료 처리합니다.
# ============================================================
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, String

from database.db_manager import DBManager

try:
    from task_manager.task_config import (
        TOPIC_TASK_COMMAND,
        TOPIC_TASK_STATUS,
        TOPIC_SAFETY_COMMAND,
        TOPIC_USER_NOTICE,
        TOPIC_TASK_COMMAND_RAW,
        TOPIC_SAFETY_STATE,
        TOPIC_EMERGENCY_STOP
    )
except Exception:
    TOPIC_TASK_COMMAND = '/task_command'
    TOPIC_TASK_STATUS = '/task_status'
    TOPIC_SAFETY_COMMAND = '/safety_command'
    TOPIC_USER_NOTICE = '/user_notice'
    TOPIC_TASK_COMMAND_RAW = '/task_command_raw'
    TOPIC_SAFETY_STATE = '/safety_state'
    TOPIC_EMERGENCY_STOP = '/emergency_stop'





START_COMMANDS = {
    'check_workspace',
    'start_organize',
}


START_STATUSES = {
    'check_workspace_requested': 'check_workspace',
    'start_organize_requested': 'start_organize',
    'recheck_workspace_requested': 'recheck_workspace',
}


# idle은 task_manager가 큰 작업을 끝내고 대기 상태로 돌아왔다는 의미로 처리합니다.
IDLE_STATUS = 'idle'


# idle이 안 들어오더라도 단독 명령 또는 오류 흐름에서 작업을 끝낼 수 있는 상태들입니다.
TERMINAL_STATUSES = {
    'unknown_command',
    'busy',
    'shutdown_requested',
    'stop_requested',
    'check_workspace_stopped',
    'workspace_detection_stopped',
    'object_detection_service_unavailable',
    'judge_workspace_service_unavailable',
    'workspace_judgement_failed',
    'workspace_judgement_json_error',
    'workspace_judgement_response_error',
    'robot_arm_action_unavailable',
    'robot_organize_goal_rejected',
    'robot_organize_failed',
    'robot_organize_result_error',
    'robot_organize_cancel_accepted',
    'robot_organize_cancel_rejected',
    'robot_organize_cancel_error',
}


# 이 단어들이 들어간 상태는 시스템 처리 실패 또는 중단으로 간주합니다.
FAILURE_KEYWORDS = [
    'failed',
    'failure',
    'error',
    'unavailable',
    'rejected',
    'stopped',
    'cancel',
    'remaining',
    'unknown_command',
    'busy',
]


DEFAULT_BAG_TOPICS = [
    '/task_command',
    '/task_command_raw',
    '/task_status',
    '/user_notice',
    '/safety_command',
    '/safety_state',
    '/emergency_stop',
    '/rosout',
    '/camera/camera/color/image_raw',
    '/camera/camera/aligned_depth_to_color/image_raw',
    '/camera/camera/color/camera_info',
]


class DBNode(Node):
    # db_node를 초기화하고 기존 topic 구독자와 SQLite DB를 준비하는 함수
    def __init__(self):
        super().__init__('db_node')

        self.declare_parameter('db_path', '~/a4_cobot2_ws/a4_cobot2_log/cobot2_log.db')
        self.declare_parameter('bag_path', '~/a4_cobot2_ws/a4_cobot2_log/bags')
        self.declare_parameter('bag_topics', DEFAULT_BAG_TOPICS)

        self.db_path = self.get_parameter('db_path').get_parameter_value().string_value
        self.bag_path = self.get_parameter('bag_path').get_parameter_value().string_value
        self.bag_topics = self._get_string_array_parameter('bag_topics', DEFAULT_BAG_TOPICS)

        self.db = DBManager(self.db_path)

        # current_run_id:
        #   현재 진행 중인 작업 ID입니다. /task_command가 들어올 때 생성됩니다.
        self.current_run_id: Optional[str] = None
        self.current_command: Optional[str] = None
        self.current_bag_id: Optional[str] = None
        self.last_non_idle_status: Optional[str] = None

        # pending_raw_text:
        #   command_input_node는 /task_command_raw를 먼저 발행하고 /task_command를 뒤이어 발행합니다.
        #   이 값을 잠깐 보관해 parsed command와 같은 commands row에 저장합니다.
        self.pending_raw_text: Optional[str] = None

        self.default_qos = QoSProfile(depth=10)
        self.latched_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.raw_command_sub = self.create_subscription(
            String,
            TOPIC_TASK_COMMAND_RAW,
            self.raw_command_callback,
            self.default_qos,
        )
        self.task_command_sub = self.create_subscription(
            String,
            TOPIC_TASK_COMMAND,
            self.task_command_callback,
            self.default_qos,
        )
        self.task_status_sub = self.create_subscription(
            String,
            TOPIC_TASK_STATUS,
            self.task_status_callback,
            self.default_qos,
        )
        self.user_notice_sub = self.create_subscription(
            String,
            TOPIC_USER_NOTICE,
            self.user_notice_callback,
            self.default_qos,
        )
        self.safety_command_sub = self.create_subscription(
            String,
            TOPIC_SAFETY_COMMAND,
            self.safety_command_callback,
            self.default_qos,
        )
        self.safety_state_sub = self.create_subscription(
            String,
            TOPIC_SAFETY_STATE,
            self.safety_state_callback,
            self.latched_state_qos,
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            TOPIC_EMERGENCY_STOP,
            self.emergency_stop_callback,
            self.latched_state_qos,
        )

        self.get_logger().info(f'DBNode started. db_path={Path(self.db_path).expanduser()}')
        if self.bag_path.strip():
            self.get_logger().info(f'rosbag metadata will be linked with bag_path={self.bag_path}')

    # 문자열 배열 파라미터를 안전하게 가져오는 함수
    def _get_string_array_parameter(self, name: str, default_value: List[str]) -> List[str]:
        try:
            value = self.get_parameter(name).get_parameter_value().string_array_value
            if value:
                return list(value)
        except Exception:
            pass

        return default_value

    # 현재 시간을 ISO 문자열로 반환하는 함수
    def now_iso(self) -> str:
        return datetime.now().astimezone().isoformat(timespec='milliseconds')

    # DB에서 사용할 작업 ID를 생성하는 함수
    def make_run_id(self, command: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_command = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in command)
        return f'{timestamp}_{safe_command}'

    # 상태 문자열을 보고 성공 여부를 추정하는 함수
    def is_success_status(self, status: str) -> bool:
        if status is None:
            return False

        normalized = status.lower()

        for keyword in FAILURE_KEYWORDS:
            if keyword in normalized:
                return False

        return True

    # command 이름을 기준으로 새 task_run을 시작하는 함수
    def start_new_run(self, command: str, raw_text: Optional[str], timestamp: Optional[str] = None):
        if timestamp is None:
            timestamp = self.now_iso()

        if self.current_run_id is not None:
            self.finish_current_run(
                final_status=self.last_non_idle_status or 'interrupted_by_new_run',
                success=False,
                memo='new command arrived before previous run finished',
            )

        self.current_run_id = self.make_run_id(command)
        self.current_command = command
        self.current_bag_id = None
        self.last_non_idle_status = None

        bag_path = self.bag_path.strip() or None

        self.db.create_task_run(
            run_id=self.current_run_id,
            command=command,
            raw_text=raw_text,
            started_at=timestamp,
            bag_path=bag_path,
        )

        if bag_path is not None:
            self.current_bag_id = f'bag_{self.current_run_id}'
            self.db.upsert_bag_record(
                bag_id=self.current_bag_id,
                run_id=self.current_run_id,
                bag_path=bag_path,
                started_at=timestamp,
                ended_at=None,
                topics=self.bag_topics,
            )

        self.get_logger().info(f'Started task run: {self.current_run_id}, command={command}')

    # 현재 task_run을 종료 처리하는 함수
    def finish_current_run(self, final_status: str, success: bool, memo: Optional[str] = None):
        if self.current_run_id is None:
            return

        ended_at = self.now_iso()

        self.db.finish_task_run(
            run_id=self.current_run_id,
            ended_at=ended_at,
            final_status=final_status,
            success=success,
            memo=memo,
        )

        if self.current_bag_id is not None and self.bag_path.strip():
            self.db.upsert_bag_record(
                bag_id=self.current_bag_id,
                run_id=self.current_run_id,
                bag_path=self.bag_path.strip(),
                started_at=None,
                ended_at=ended_at,
                topics=self.bag_topics,
            )

        self.get_logger().info(
            f'Finished task run: {self.current_run_id}, final_status={final_status}, success={success}'
        )

        self.current_run_id = None
        self.current_command = None
        self.current_bag_id = None
        self.last_non_idle_status = None

    # /task_command_raw를 받아 다음 /task_command와 연결하기 위해 임시 저장하는 함수
    def raw_command_callback(self, msg: String):
        raw_text = msg.data.strip()
        if raw_text == '':
            return

        self.pending_raw_text = raw_text
        self.get_logger().info(f'Received raw command: {raw_text}')

    # /task_command를 받아 작업 실행 기록을 시작하거나 현재 작업에 명령 기록을 추가하는 함수
    def task_command_callback(self, msg: String):
        command = msg.data.strip()
        timestamp = self.now_iso()
        raw_text = self.pending_raw_text
        self.pending_raw_text = None

        if command in START_COMMANDS:
            self.start_new_run(command=command, raw_text=raw_text, timestamp=timestamp)

        elif self.current_run_id is None:
            # stop, shutdown, unknown처럼 독립적으로 들어올 수 있는 명령도 기록에 남깁니다.
            self.start_new_run(command=command, raw_text=raw_text, timestamp=timestamp)

        elif raw_text is not None:
            self.db.update_task_raw_text(self.current_run_id, raw_text)

        self.db.insert_command(
            run_id=self.current_run_id,
            timestamp=timestamp,
            raw_text=raw_text,
            parsed_command=command,
        )

        self.get_logger().info(f'Saved command: run_id={self.current_run_id}, command={command}')

    # /task_status를 저장하고 작업 종료 상태인지 판단하는 함수
    def task_status_callback(self, msg: String):
        status = msg.data.strip()
        timestamp = self.now_iso()

        if status in START_STATUSES and self.current_run_id is None:
            # DB 노드가 /task_command를 놓쳤거나 중간에 켜진 경우를 대비한 보정입니다.
            self.start_new_run(
                command=START_STATUSES[status],
                raw_text=None,
                timestamp=timestamp,
            )

        self.db.insert_task_status(
            run_id=self.current_run_id,
            timestamp=timestamp,
            status=status,
        )

        if self.current_run_id is None:
            return

        if status != IDLE_STATUS:
            self.last_non_idle_status = status
            self.db.update_task_status(self.current_run_id, status)

        if status == IDLE_STATUS:
            final_status = self.last_non_idle_status or IDLE_STATUS
            self.finish_current_run(
                final_status=final_status,
                success=self.is_success_status(final_status),
            )
            return

        if status in TERMINAL_STATUSES:
            self.finish_current_run(
                final_status=status,
                success=self.is_success_status(status),
            )

    # /user_notice 문장을 저장하는 함수
    def user_notice_callback(self, msg: String):
        notice = msg.data.strip()
        if notice == '':
            return

        self.db.insert_user_notice(
            run_id=self.current_run_id,
            timestamp=self.now_iso(),
            notice=notice,
        )

    # /safety_command 이벤트를 저장하는 함수
    def safety_command_callback(self, msg: String):
        command = msg.data.strip()
        if command == '':
            return

        self.db.insert_safety_event(
            run_id=self.current_run_id,
            timestamp=self.now_iso(),
            event_type='safety_command',
            value=command,
        )

    # /safety_state 이벤트를 저장하는 함수
    def safety_state_callback(self, msg: String):
        state = msg.data.strip()
        if state == '':
            return

        self.db.insert_safety_event(
            run_id=self.current_run_id,
            timestamp=self.now_iso(),
            event_type='safety_state',
            value=state,
        )

    # /emergency_stop Bool 이벤트를 저장하는 함수
    def emergency_stop_callback(self, msg: Bool):
        self.db.insert_safety_event(
            run_id=self.current_run_id,
            timestamp=self.now_iso(),
            event_type='emergency_stop',
            value=str(bool(msg.data)).lower(),
        )

    # 노드 종료 전에 열려 있는 작업과 DB 연결을 정리하는 함수
    def close(self):
        if self.current_run_id is not None:
            self.finish_current_run(
                final_status=self.last_non_idle_status or 'db_node_shutdown',
                success=False,
                memo='db_node closed while run was active',
            )

        self.db.close()


# ROS2 db_node를 실행하고 topic 로그를 DB에 계속 기록하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = DBNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 db_node를 종료합니다.')

    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
