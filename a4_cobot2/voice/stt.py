import os
import tempfile

import sounddevice as sd
import scipy.io.wavfile as wav
from openai import OpenAI
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory


class STT:
    # .env에서 OpenAI API 키를 읽고 Whisper STT 클라이언트를 준비하는 함수
    def __init__(self):
        package_name = 'a4_cobot2'
        package_path = get_package_share_directory(package_name)

        env_path = os.path.join(package_path, 'resource', '.env')
        load_dotenv(dotenv_path=env_path)

        openai_api_key = os.getenv('OPENAI_API_KEY')

        if openai_api_key is None:
            raise RuntimeError('OPENAI_API_KEY가 .env 파일에 없습니다.')

        self.client = OpenAI(api_key=openai_api_key)

        self.duration = 3 # 녹음 기간 설정(단위: 초)
        self.samplerate = 16000 # 1초에 오디오 샘플을 몇 개 기록할지 정하는 값

    # 마이크로 5초 동안 음성을 녹음하고 Whisper API를 이용해 텍스트로 변환하는 함수
    def speech2text(self) -> str:
        print(f'음성 녹음을 시작합니다. {self.duration}초 동안 말해주세요.')

        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1, # 1채널 모노 음성(소리 입력 통로가 1개)
            dtype='int16'
        )

        sd.wait()
        print('녹음 완료. Whisper에 전송 중...')

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)

            with open(temp_wav.name, 'rb') as f:
                transcript = self.client.audio.transcriptions.create(
                    model='whisper-1',
                    file=f
                )

        print(f'STT 결과: {transcript.text}')
        return transcript.text