import rclpy

from rclpy.node import Node
from std_msgs.msg import String

from notification.notice_utils import make_task_status_notice
from notification.notice_utils import make_safety_state_notice


class StatusNotifierNode(Node):
    # status_notifier_node를 초기화하고 작업 상태, 안전 상태, 사용자 안내 문장 구독자를 준비하는 함수
    def __init__(self):
        super().__init__('status_notifier_node')

        self.declare_parameter('use_tts', False)
        self.use_tts = self.get_parameter('use_tts').get_parameter_value().bool_value

        self.tts = None
        self.last_notice = None

        if self.use_tts:
            self.initialize_tts()

        self.task_status_sub = self.create_subscription(
            String,
            '/task_status',
            self.task_status_callback,
            10
        )

        self.safety_state_sub = self.create_subscription(
            String,
            '/safety_state',
            self.safety_state_callback,
            10
        )

        self.user_notice_sub = self.create_subscription(
            String,
            '/user_notice',
            self.user_notice_callback,
            10
        )

        self.get_logger().info('StatusNotifierNode started.')
        self.notify('상태 안내 노드가 시작되었습니다.', speak=False)

    # use_tts 파라미터가 true일 때 TTS 객체를 준비하는 함수
    def initialize_tts(self):
        try:
            from voice.tts import TTS

            self.tts = TTS()
            self.get_logger().info('TTS enabled.')

        except Exception as e:
            self.tts = None
            self.use_tts = False
            self.get_logger().warn(f'TTS 초기화 실패. 콘솔 출력만 사용합니다: {e}')

    # /task_status로 들어온 상태 코드를 사용자 안내 문장으로 바꾸는 함수
    def task_status_callback(self, msg: String):
        status = msg.data.strip()
        notice = make_task_status_notice(status)

        if notice is None:
            return

        self.notify(notice)

    # /safety_state로 들어온 안전 상태를 사용자 안내 문장으로 바꾸는 함수
    def safety_state_callback(self, msg: String):
        safety_state = msg.data.strip()
        notice = make_safety_state_notice(safety_state)

        if notice is None:
            return

        self.notify(notice)

    # /user_notice로 들어온 사용자 안내 문장을 그대로 출력하는 함수
    def user_notice_callback(self, msg: String):
        notice = msg.data.strip()

        if notice == '':
            return

        self.notify(notice)

    # 안내 문장을 콘솔에 출력하고 설정된 경우 TTS로 말하는 함수
    def notify(self, notice: str, speak: bool = True):
        if notice is None or notice.strip() == '':
            return

        if notice == self.last_notice:
            return

        self.last_notice = notice

        self.get_logger().info(f'USER NOTICE: {notice}')
        print(f'[USER NOTICE] {notice}')

        if speak and self.use_tts and self.tts is not None:
            self.tts.speak(notice)


# ROS2 status_notifier_node를 실행하고 상태 안내 callback을 계속 처리하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = StatusNotifierNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
