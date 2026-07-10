# ============================================================
# voice/wakeup_word.py
# 역할:
#   - openWakeWord 모델로 "hello rokey" 시동어를 감지합니다.
# 입력:
#   - MicController가 열어둔 PyAudio stream
# 출력:
#   - confidence가 기준 이상이면 True 반환
# 주의:
#   - 마이크 입력 rate는 48000Hz이고, 모델 입력용으로 16000Hz로 resample합니다.
# ============================================================
import os
import numpy as np
from openwakeword.model import Model
from scipy.signal import resample
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

MODEL_NAME = "hello_rokey_8332_32.tflite"
MODEL_PATH = os.path.join(PACKAGE_PATH, f"resource/{MODEL_NAME}")

class WakeupWord:
    def __init__(self, buffer_size):
        # model:
        #   set_stream()에서 실제 openWakeWord 모델이 로드됩니다.
        self.model = None

        # model_name:
        #   predict() 결과 dict에서 confidence를 꺼낼 때 사용하는 key입니다.
        self.model_name = MODEL_NAME.split(".", maxsplit=1)[0]

        # stream:
        #   MicController.open_stream()으로 열린 PyAudio stream입니다.
        self.stream = None

        # buffer_size:
        #   stream.read()가 한 번에 읽는 샘플 수입니다.
        self.buffer_size = buffer_size

    # 마이크 stream에서 오디오 chunk를 읽고 hello_rokey confidence가 기준 이상인지 판단하는 함수
    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        print("confidence: ", confidence)
        # Wakeword 탐지
        if confidence > 0.3:
            print("Wakeword detected!")
            return True
        return False
    
    # 외부에서 열린 마이크 stream을 wakeword 모델에 연결하는 함수
    def set_stream(self, stream):
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.stream = stream
