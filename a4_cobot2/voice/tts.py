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
    # spd-say 음성 설정 (원하는 목소리로 바꾸세요)
    #   VOICE: 'spd-say -L' 목록 중 하나. 예: 'Korean', 'Korean+Annie', 'Korean+Boris'
    #   RATE : 말 속도 (-100 ~ 100), PITCH: 음높이 (-100 ~ 100)
    VOICE = 'Korean+Boris'
    RATE = 50
    PITCH = 50

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
                [
                    'spd-say',
                    '-y', self.VOICE,
                    '-r', str(self.RATE),
                    '-p', str(self.PITCH),
                    text,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f'TTS 실행 실패: {e}')