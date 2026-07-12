# ============================================================
# voice/tts3.py
# 역할:
#   - Microsoft edge-tts로 문장을 자연스러운 음성으로 빠르게 읽어줍니다.
#   - 무료이며 API 키가 필요 없습니다. (Microsoft Edge 온라인 TTS)
#   - 기존 voice/tts.py와 인터페이스(speak)는 동일합니다.
# 필요:
#   - pip install edge-tts
#   - ffplay(ffmpeg) 로 mp3 재생 (시스템에 설치돼 있어야 함)
#   - 인터넷 연결 (edge-tts는 온라인 서비스, 다만 빠름)
# ============================================================
import asyncio
import os
import subprocess
import tempfile
import threading

import edge_tts


class TTS:
    # edge-tts 음성 설정 (원하는 목소리로 바꾸세요)
    #   VOICE: ko-KR-SunHiNeural(여성), ko-KR-InJoonNeural(남성),
    #          ko-KR-HyunsuNeural(남성) 등. 'edge-tts --list-voices | grep ko-KR' 로 확인
    #   RATE : 말 속도. 예) '+0%', '+20%', '-10%'
    #   PITCH: 음높이. 예) '+0Hz', '+20Hz', '-10Hz'
    VOICE = 'ko-KR-InJoonNeural'
    RATE = '+20%'
    PITCH = '+30Hz'

    def __init__(self):
        pass

    # 입력된 문장을 터미널에 출력하고 edge-tts로 읽어주는 함수.
    # 합성+재생은 백그라운드 스레드에서 하여 호출한 노드를 막지 않는다.
    def speak(self, text: str):
        print(f'TTS3: {text}')

        if not text or text.strip() == '':
            return

        threading.Thread(
            target=self._synthesize_and_play,
            args=(text,),
            daemon=True,
        ).start()

    # edge-tts로 mp3를 합성해 임시 파일로 받고 ffplay로 재생하는 함수
    def _synthesize_and_play(self, text: str):
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                path = tmp.name

            asyncio.run(self._synthesize(text, path))

            # ffplay: 창 없이, 재생 끝나면 자동 종료, 로그 숨김
            subprocess.run(
                ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', path],
                check=False,
            )

        except Exception as e:
            print(f'TTS3 실행 실패: {e}')

        finally:
            if path is not None and os.path.exists(path):
                os.remove(path)

    # edge-tts 비동기 합성 (Communicate.save는 coroutine)
    async def _synthesize(self, text: str, path: str):
        communicate = edge_tts.Communicate(
            text,
            self.VOICE,
            rate=self.RATE,
            pitch=self.PITCH,
        )
        await communicate.save(path)
