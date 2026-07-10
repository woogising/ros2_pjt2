# ============================================================
# voice/command_classifier.py
# 역할:
#   - STT가 변환한 자연어 문장을 내부 명령어로 분류합니다.
# 입력 예:
#   - "작업공간 확인해줘" -> check_workspace
#   - "정리 시작해줘" -> start_organize
# 출력:
#   - task_manager_node가 이해하는 명령 문자열 하나
# ============================================================
import os

from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class CommandClassifier:
    # .env에서 API 키를 읽고 LLM 기반 명령 분류 체인을 준비하는 함수
    def __init__(self):
        package_name = 'a4_cobot2'
        package_path = get_package_share_directory(package_name)

        env_path = os.path.join(package_path, 'resource', '.env')
        load_dotenv(dotenv_path=env_path)

        openai_api_key = os.getenv('OPENAI_API_KEY')

        if openai_api_key is None:
            raise RuntimeError('OPENAI_API_KEY가 .env 파일에 없습니다.')

        self.llm = ChatOpenAI(
            model='gpt-4o',
            temperature=0,
            openai_api_key=openai_api_key
        )

        prompt_content = """
        모델의 입력 출력 예시)

        사용자의 문장을 보고 로봇 명령을 하나로 분류하세요.

        가능한 출력은 반드시 아래 중 하나만 사용하세요.
        - check_workspace
        - start_organize
        - stop
        - shutdown
        - unknown

        예시:
        입력: "로봇아 작업공간 확인해줘"
        출력: check_workspace

        입력: "정리 시작해줘"
        출력: start_organize

        입력: "멈춰"
        출력: stop

        입력: "정지해"
        출력: stop

        입력: "명령 노드 종료해"
        출력: shutdown

        입력: "시스템 꺼"
        출력: shutdown

        입력: "오늘 날씨 어때?"
        출력: unknown

        사용자 입력:
        "{user_input}"

        출력:
        """

        self.prompt_template = PromptTemplate(
            input_variables=['user_input'],
            template=prompt_content
        )

        self.chain = self.prompt_template | self.llm

    # 사용자 자연어 문장을 LLM에 전달하고 내부 명령어 하나로 변환하는 함수
    def classify(self, user_input: str) -> str:
        response = self.chain.invoke({'user_input': user_input})
        command = response.content.strip()

        allowed_commands = [
            'check_workspace',
            'start_organize',
            'stop',
            'shutdown',
            'unknown'
        ]

        if command not in allowed_commands:
            return 'unknown'

        return command