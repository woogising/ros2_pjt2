# ============================================================
# voice/command_input_node.py
# 역할:
#   - wakeword 감지 -> STT -> LLM 명령 분류 -> /task_command 발행을 담당합니다.
# 주요 출력 topic:
#   - /task_command: task_manager_node가 구독하는 내부 명령어
#   - /task_command_raw: STT 원문 확인용 디버그 topic
# 명령어 의미:
#   - stop: 로봇 동작/작업 정지 요청
#   - shutdown: 명령 입력 노드 종료 요청
# ============================================================
import time
import rclpy
import pyaudio

from rclpy.node import Node
from std_msgs.msg import String

from voice.MicController import MicController, MicConfig
from voice.wakeup_word import WakeupWord
from voice.stt import STT
from voice.tts import TTS
from voice.command_classifier import CommandClassifier


class CommandInputNode(Node):
    # command_input_node를 초기화하고 wakeword, STT, TTS, LLM 분류기, ROS2 publisher를 준비하는 함수
    def __init__(self):
        super().__init__('command_input_node')
        # task_manager_node가 실제로 처리하는 내부 명령어를 발행합니다.
        # 예: 'check_workspace', 'start_organize', 'stop', 'shutdown'
        self.command_pub = self.create_publisher(String, '/task_command', 10)

        # STT가 알아들은 원문 문장을 디버깅하기 위한 topic입니다.
        # task_manager_node는 이 topic을 사용하지 않고, 사람이 로그 확인용으로 씁니다.
        self.raw_command_pub = self.create_publisher(String, '/task_command_raw', 10)

        # 음성 입력 처리 객체들입니다.
        # STT: 실제 음성 -> 텍스트
        # TTS: 사용자에게 안내 멘트 출력/음성 출력
        # CommandClassifier: 텍스트 -> 내부 명령어 분류
        self.stt = STT()
        self.tts = TTS()
        self.command_classifier = CommandClassifier()
        # wakeword 감지용 마이크 설정입니다.
        # rate=48000: 실제 마이크 입력 샘플링 레이트
        # buffer_size=24000: WakeupWord가 한 번에 읽는 버퍼 크기
        # WakeupWord 내부에서 16000Hz로 resample해서 모델에 넣습니다.
        self.mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            fmt=pyaudio.paInt16,
            buffer_size=24000
        )
        self.mic_controller = MicController(config=self.mic_config)
        self.wakeup_word = WakeupWord(self.mic_config.buffer_size)

        self.get_logger().info('CommandInputNode started.')

    # TTS 출력이 STT 녹음에 섞이지 않도록 잠깐 대기하는 함수
    def wait_after_tts(self, seconds: float = 1.0):
        time.sleep(seconds)

    # hello, rokey 시동어가 감지될 때까지 마이크 stream을 열고 대기하는 함수
    def wait_for_wakeup(self) -> bool:
        self.get_logger().info('"hello, rokey" 시동어 대기중')

        try:
            self.mic_controller.open_stream()
            self.wakeup_word.set_stream(self.mic_controller.stream)

            while rclpy.ok():
                if self.wakeup_word.is_wakeup():
                    self.get_logger().info('Wakeword detected.')
                    return True

        except OSError:
            self.get_logger().error('마이크 stream을 열 수 없습니다. device_index를 확인하세요.')
            return False

        finally:
            self.mic_controller.close_stream()

        return False

    # 원본 STT 문장과 분류된 내부 명령어를 각각 ROS2 토픽으로 발행하는 함수
    def publish_command(self, raw_text: str, command: str):
        raw_msg = String()
        raw_msg.data = raw_text

        command_msg = String()
        command_msg.data = command

        self.raw_command_pub.publish(raw_msg)
        self.command_pub.publish(command_msg)

        self.get_logger().info(f'Raw command: {raw_text}')
        self.get_logger().info(f'Parsed command: {command}')

    # 분류된 명령어에 따라 사용자에게 TTS 피드백을 제공하는 함수
    def speak_command_feedback(self, command: str):
        if command == 'check_workspace':
            self.tts.speak('작업공간 확인 명령을 받았습니다.')
        elif command == 'start_organize':
            self.tts.speak('정리 시작 명령을 받았습니다.')
        elif command == 'stop':
            self.tts.speak('동작 정지 명령을 받았습니다.')
        elif command == 'shutdown':
            self.tts.speak('명령 입력 노드를 종료합니다.')
        else:
            self.tts.speak('명령을 이해하지 못했습니다.')

    # wakeword 감지 후 음성 명령을 한 번 받아 STT, LLM 분류, 토픽 발행, TTS 피드백을 수행하는 함수
    def process_voice_command_once(self):
        self.tts.speak('동작을 말씀해주세요.')
        self.wait_after_tts(1.0)

        raw_text = self.stt.speech2text()

        if raw_text is None or raw_text.strip() == '':
            self.get_logger().warn('STT 결과가 비어 있습니다.')
            self.tts.speak('음성을 인식하지 못했습니다.')
            return None

        command = self.command_classifier.classify(raw_text)

        if command == 'unknown':
            self.get_logger().warn(f'유효하지 않은 명령입니다: {raw_text}')
            self.tts.speak('명령을 이해하지 못했습니다. 다시 호출 후 말씀해주세요.')
            return None

        self.publish_command(raw_text, command)
        self.speak_command_feedback(command)

        return command


# ROS2 노드를 실행하고, 최초 인사 후 wakeword가 감지될 때마다 음성 명령을 한 번 처리하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = CommandInputNode()

    try:
        node.tts.speak('반갑습니다.')
        node.wait_after_tts(0.8)

        while rclpy.ok():
            wakeup_detected = node.wait_for_wakeup()

            if not wakeup_detected:
                continue

            command = node.process_voice_command_once()
            if command == 'shutdown':
                break

            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()