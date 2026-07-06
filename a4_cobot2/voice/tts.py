# ============================================================
# voice/tts.py
# 역할:
#   - 사용자 안내 문장을 콘솔에 출력하고, spd-say가 있으면 음성으로 읽습니다.
# 사용 위치:
#   - command_input_node: 명령 입력 피드백
#   - status_notifier_node: use_tts=True일 때 상태 안내
# ============================================================
import shutil
import subprocess


class TTS:
    # 시스템에 spd-say가 설치되어 있는지 확인하고 TTS 실행 준비를 하는 함수
    def __init__(self):
        self.spd_say_path = shutil.which('spd-say')

    # 입력된 문장을 터미널에 출력하고 가능하면 음성으로 읽어주는 함수
    def speak(self, text: str):
        print(f'TTS: {text}')

        if self.spd_say_path is None:
            return

        try:
            subprocess.Popen(
                ['spd-say', text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f'TTS 실행 실패: {e}')