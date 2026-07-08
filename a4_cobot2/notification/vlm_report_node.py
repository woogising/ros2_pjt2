# ============================================================
# notification/vlm_report_node.py
# 역할:
#   - 최종 재검증 결과를 사용자에게 보고할 문장을 생성하는 VLM 보고 노드입니다.
#   - TaskManagerNode가 /generate_final_report service를 호출하면,
#     최신 RGB 이미지, YOLO annotated 이미지, workspace judgement JSON을 함께 사용해
#     GPT-4o 기반 최종 보고문을 생성합니다.
#
# 입력 topic:
#   - /camera/camera/color/image_raw: 최종 작업공간 원본 RGB 이미지
#   - /yolo_detection_image: YOLO bbox/mask/label이 그려진 이미지
#
# Service:
#   - /generate_final_report
#
# 주의:
#   - 이 노드는 로봇 동작 판단을 대신하지 않습니다.
#   - 로봇 이동/정리 여부는 기존 workspace_judge_node의 좌표 기반 판단을 사용합니다.
#   - VLM은 사용자 보고문 생성과 보조 시각 확인 설명용으로만 사용합니다.
# ============================================================
import base64
import json
import os
from typing import Any, Dict, Optional

import cv2
import rclpy

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from dotenv import load_dotenv
from openai import OpenAI
from rclpy.node import Node
from sensor_msgs.msg import Image

from od_msg.srv import GenerateReport
from notification.notice_utils import make_recheck_remaining_notice
from workspace.workspace_judge_utils import get_default_zone_rules


PACKAGE_NAME = 'a4_cobot2'


class VLMReportNode(Node):
    # VLMReportNode를 초기화하고 이미지 구독자, OpenAI client, 보고문 생성 service를 준비하는 함수
    def __init__(self):
        super().__init__('vlm_report_node')

        self.declare_parameter('use_vlm', True)
        self.declare_parameter('model', 'gpt-4o')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('annotated_image_topic', '/yolo_detection_image')
        self.declare_parameter('max_image_width', 960)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('openai_timeout_sec', 15.0)

        self.use_vlm = self.get_parameter('use_vlm').get_parameter_value().bool_value
        self.model = self.get_parameter('model').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.annotated_image_topic = self.get_parameter('annotated_image_topic').get_parameter_value().string_value
        self.max_image_width = self.get_parameter('max_image_width').get_parameter_value().integer_value
        self.jpeg_quality = self.get_parameter('jpeg_quality').get_parameter_value().integer_value
        self.openai_timeout_sec = self.get_parameter('openai_timeout_sec').get_parameter_value().double_value

        self.bridge = CvBridge()

        # latest_raw_frame:
        #   최종 보고에 사용할 최신 원본 카메라 이미지입니다.
        self.latest_raw_frame = None

        # latest_annotated_frame:
        #   YOLO bbox/mask/label이 그려진 최신 이미지입니다.
        self.latest_annotated_frame = None

        self.zone_rules = get_default_zone_rules()

        self.client = self._create_openai_client()

        self.raw_image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.raw_image_callback,
            10,
        )

        self.annotated_image_sub = self.create_subscription(
            Image,
            self.annotated_image_topic,
            self.annotated_image_callback,
            10,
        )

        self.report_srv = self.create_service(
            GenerateReport,
            '/generate_final_report',
            self.handle_generate_final_report,
        )

        self.get_logger().info('VLMReportNode started.')
        self.get_logger().info(f'use_vlm={self.use_vlm}, model={self.model}')
        self.get_logger().info(f'image_topic={self.image_topic}')
        self.get_logger().info(f'annotated_image_topic={self.annotated_image_topic}')

    # .env에서 OPENAI_API_KEY를 읽어 OpenAI client를 만드는 함수
    def _create_openai_client(self):
        if not self.use_vlm:
            self.get_logger().warn('use_vlm=False 이므로 VLM 호출 없이 fallback 보고문만 사용합니다.')
            return None

        try:
            package_path = get_package_share_directory(PACKAGE_NAME)
            env_path = os.path.join(package_path, 'resource', '.env')
            load_dotenv(dotenv_path=env_path)

            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None or api_key.strip() == '':
                self.get_logger().warn('OPENAI_API_KEY가 없어 fallback 보고문만 사용합니다.')
                self.use_vlm = False
                return None

            return OpenAI(
                api_key=api_key,
                timeout=self.openai_timeout_sec,
            )

        except Exception as exc:
            self.get_logger().warn(f'OpenAI client 초기화 실패. fallback 보고문만 사용합니다: {exc}')
            self.use_vlm = False
            return None

    # 원본 RGB 이미지 topic을 받아 최신 frame으로 저장하는 함수
    def raw_image_callback(self, msg: Image):
        try:
            self.latest_raw_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )
        except Exception as exc:
            self.get_logger().warn(f'원본 이미지 변환 실패: {exc}')

    # YOLO annotated 이미지 topic을 받아 최신 frame으로 저장하는 함수
    def annotated_image_callback(self, msg: Image):
        try:
            self.latest_annotated_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )
        except Exception as exc:
            self.get_logger().warn(f'annotated 이미지 변환 실패: {exc}')

    # /generate_final_report service 요청을 받아 최종 사용자 보고문을 생성하는 함수
    def handle_generate_final_report(self, request, response):
        try:
            request_payload = json.loads(request.report_request_json)

        except json.JSONDecodeError as exc:
            response.success = False
            response.report_text = ''
            response.report_json = json.dumps(
                {
                    'source': 'error',
                    'reason': 'invalid_report_request_json',
                },
                ensure_ascii=False,
            )
            response.message = f'invalid report_request_json: {exc}'
            return response

        fallback_notice = self.make_fallback_report(request_payload)

        if not self.use_vlm or self.client is None:
            response.success = True
            response.report_text = fallback_notice
            response.report_json = json.dumps(
                {
                    'source': 'fallback',
                    'reason': 'vlm_disabled_or_unavailable',
                    'model': self.model,
                },
                ensure_ascii=False,
            )
            response.message = 'fallback report generated'
            return response

        if self.latest_raw_frame is None and self.latest_annotated_frame is None:
            response.success = True
            response.report_text = fallback_notice
            response.report_json = json.dumps(
                {
                    'source': 'fallback',
                    'reason': 'no_image_received',
                    'model': self.model,
                },
                ensure_ascii=False,
            )
            response.message = 'no image received, fallback report generated'
            return response

        try:
            report_text = self.generate_vlm_report(request_payload, fallback_notice)

            response.success = True
            response.report_text = report_text
            response.report_json = json.dumps(
                {
                    'source': 'vlm',
                    'model': self.model,
                    'used_raw_image': self.latest_raw_frame is not None,
                    'used_annotated_image': self.latest_annotated_frame is not None,
                },
                ensure_ascii=False,
            )
            response.message = 'vlm report generated'
            return response

        except Exception as exc:
            self.get_logger().error(f'VLM 보고문 생성 실패: {exc}')

            response.success = True
            response.report_text = fallback_notice
            response.report_json = json.dumps(
                {
                    'source': 'fallback',
                    'reason': 'vlm_exception',
                    'error': str(exc),
                    'model': self.model,
                },
                ensure_ascii=False,
            )
            response.message = f'vlm failed, fallback report generated: {exc}'
            return response

    # judgement payload만으로 기존 방식의 안전한 fallback 보고문을 만드는 함수
    def make_fallback_report(self, request_payload: Dict[str, Any]) -> str:
        fallback_notice = request_payload.get('fallback_notice')
        if isinstance(fallback_notice, str) and fallback_notice.strip() != '':
            return fallback_notice.strip()

        judgement_payload = request_payload.get('judgement_payload', {})
        result = judgement_payload.get('result', 'unknown')

        if result == 'all_clear':
            return '정리가 완료되었습니다. 재검증 결과, 모든 물건이 지정된 구역에 배치되었습니다.'

        if result == 'misplaced_found':
            return make_recheck_remaining_notice(judgement_payload)

        if result == 'unknown_rule_found':
            unknown_objects = judgement_payload.get('unknown_rule_objects', [])
            names = [
                obj.get('name', '알 수 없는 물체')
                for obj in unknown_objects
            ]
            object_text = ', '.join(names) if names else '일부 물체'
            return (
                f'정리 후 재검증을 했지만 일부 물체의 배치 규칙을 찾을 수 없습니다. '
                f'확인이 필요한 물체는 {object_text}입니다.'
            )

        if result == 'no_objects':
            return '재검증 중 감지된 물체가 없습니다. 카메라 시야 또는 작업공간을 확인해주세요.'

        return '정리 후 작업공간 상태를 정확히 판단하지 못했습니다. 확인이 필요합니다.'

    # OpenAI VLM에 보낼 최종 보고 프롬프트를 만드는 함수
    def build_prompt(self, request_payload: Dict[str, Any], fallback_notice: str) -> str:
        compact_payload = {
            'report_mode': request_payload.get('report_mode', 'final_recheck_report'),
            'detected_objects': request_payload.get('detected_objects', []),
            'judgement_payload': request_payload.get('judgement_payload', {}),
            'zone_rules': self.zone_rules,
            'fallback_notice': fallback_notice,
        }

        return f"""
너는 협동로봇 작업공간 정리 시스템의 최종 보고 담당자다.

중요한 원칙:
1. 로봇의 공식 판단은 JSON의 judgement_payload를 우선한다.
2. 이미지는 보조 확인용이다. 이미지와 JSON이 다르면 확정적으로 단정하지 말고 "시각적으로는 확인이 필요합니다"처럼 말한다.
3. 로봇 동작 좌표, 픽셀 좌표, 내부 JSON 키 이름은 사용자에게 자세히 말하지 않는다.
4. 한국어로 짧고 자연스럽게 보고한다.
5. 사용자가 듣는 TTS 문장이므로 2~4문장 정도로 말한다.
6. 안전 문제나 불확실성이 있으면 마지막 문장에 확인 필요성을 말한다.

입력 데이터:
{json.dumps(compact_payload, ensure_ascii=False, indent=2)}

출력 형식:
- 최종 사용자 보고문만 출력한다.
- 제목, bullet, JSON, markdown은 쓰지 않는다.
""".strip()

    # cv2 frame을 OpenAI API에 넣을 base64 data URL로 변환하는 함수
    def frame_to_data_url(self, frame) -> Optional[str]:
        if frame is None:
            return None

        image = frame.copy()
        height, width = image.shape[:2]

        if self.max_image_width > 0 and width > self.max_image_width:
            scale = self.max_image_width / float(width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(self.jpeg_quality),
        ]

        ok, encoded = cv2.imencode('.jpg', image, encode_params)
        if not ok:
            return None

        image_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        return f'data:image/jpeg;base64,{image_b64}'

    # 최신 이미지와 JSON payload를 이용해 VLM 최종 보고문을 생성하는 함수
    def generate_vlm_report(self, request_payload: Dict[str, Any], fallback_notice: str) -> str:
        prompt = self.build_prompt(request_payload, fallback_notice)

        user_content = [
            {
                'type': 'text',
                'text': prompt,
            }
        ]

        raw_image_url = self.frame_to_data_url(self.latest_raw_frame)
        annotated_image_url = self.frame_to_data_url(self.latest_annotated_frame)

        if raw_image_url is not None:
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': raw_image_url,
                        'detail': 'low',
                    },
                }
            )

        if annotated_image_url is not None:
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': annotated_image_url,
                        'detail': 'low',
                    },
                }
            )

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=300,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        '너는 로봇 작업공간 정리 결과를 사용자에게 짧고 정확하게 보고하는 한국어 안내자다.'
                    ),
                },
                {
                    'role': 'user',
                    'content': user_content,
                },
            ],
        )

        report_text = completion.choices[0].message.content

        if report_text is None or report_text.strip() == '':
            return fallback_notice

        return report_text.strip()


# ROS2 vlm_report_node를 실행하고 보고문 생성 service callback을 계속 처리하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = VLMReportNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()