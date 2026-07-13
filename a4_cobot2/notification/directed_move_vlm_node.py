# ============================================================
# notification/directed_move_vlm_node.py
# 역할:
#   - "왼쪽 바나나를 초록칸으로 옮겨줘" 같은 지정 이동 명령을 해석하는 VLM 노드입니다.
#   - TaskManagerNode가 /resolve_directed_move service를 호출하면,
#     최신 RGB 이미지 + 검출 물체 목록 + 명령 문장을 GPT-4o에 주고
#     "어느 물체(target_id)를 어느 구역(place_zone)으로" 옮길지 판단합니다.
#
# 입력 topic:
#   - /camera/camera/color/image_raw: 현재 작업공간 원본 RGB 이미지(장면 확인용)
#
# Service:
#   - /resolve_directed_move
#
# 주의:
#   - VLM은 좌표(mm)를 계산하지 않습니다. target_id가 가리키는 검출 결과의 base 좌표를
#     TaskManagerNode가 그대로 사용합니다.
#   - "종류당 1개" 전제를 두므로 물체 목록을 id↔name 텍스트로만 제공합니다.
# ============================================================
import base64
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import rclpy

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from dotenv import load_dotenv
from openai import OpenAI
from rclpy.node import Node
from sensor_msgs.msg import Image

from od_msg.srv import ResolveDirectedMove
from workspace.workspace_judge_utils import get_default_zone_rules


PACKAGE_NAME = 'a4_cobot2'


class DirectedMoveVLMNode(Node):
    # DirectedMoveVLMNode를 초기화하고 이미지 구독자, OpenAI client, 해석 service를 준비하는 함수
    def __init__(self):
        super().__init__('directed_move_vlm_node')

        self.declare_parameter('use_vlm', True)
        self.declare_parameter('model', 'gpt-4o')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('max_image_width', 960)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('openai_timeout_sec', 15.0)
        self.declare_parameter('confidence_threshold', 0.4)

        self.use_vlm = self.get_parameter('use_vlm').get_parameter_value().bool_value
        self.model = self.get_parameter('model').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.max_image_width = self.get_parameter('max_image_width').get_parameter_value().integer_value
        self.jpeg_quality = self.get_parameter('jpeg_quality').get_parameter_value().integer_value
        self.openai_timeout_sec = self.get_parameter('openai_timeout_sec').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.bridge = CvBridge()

        # latest_raw_frame:
        #   지정 이동 판단에 사용할 최신 원본 카메라 이미지입니다.
        self.latest_raw_frame = None

        self.zone_rules = get_default_zone_rules()

        self.client = self._create_openai_client()

        self.raw_image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.raw_image_callback,
            10,
        )

        self.resolve_srv = self.create_service(
            ResolveDirectedMove,
            '/resolve_directed_move',
            self.handle_resolve_directed_move,
        )

        self.get_logger().info('DirectedMoveVLMNode started.')
        self.get_logger().info(f'use_vlm={self.use_vlm}, model={self.model}')
        self.get_logger().info(f'image_topic={self.image_topic}')

    # .env에서 OPENAI_API_KEY를 읽어 OpenAI client를 만드는 함수
    def _create_openai_client(self):
        if not self.use_vlm:
            self.get_logger().warn('use_vlm=False 이므로 지정 이동 해석을 사용할 수 없습니다.')
            return None

        try:
            package_path = get_package_share_directory(PACKAGE_NAME)
            env_path = os.path.join(package_path, 'resource', '.env')
            load_dotenv(dotenv_path=env_path)

            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None or api_key.strip() == '':
                self.get_logger().warn('OPENAI_API_KEY가 없어 지정 이동 해석을 사용할 수 없습니다.')
                self.use_vlm = False
                return None

            return OpenAI(
                api_key=api_key,
                timeout=self.openai_timeout_sec,
            )

        except Exception as exc:
            self.get_logger().warn(f'OpenAI client 초기화 실패: {exc}')
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

    # /resolve_directed_move 요청을 받아 target_id/place_zone을 판단하는 함수
    def handle_resolve_directed_move(self, request, response):
        if not self.use_vlm or self.client is None:
            return self._fail_response(
                response,
                reason='VLM을 사용할 수 없어 지정 이동을 수행할 수 없습니다.',
                message='vlm_disabled_or_unavailable',
            )

        try:
            detected_payload = json.loads(request.detected_objects_json)
            objects = detected_payload.get('objects', [])
        except json.JSONDecodeError as exc:
            return self._fail_response(
                response,
                reason='물체 목록을 해석하지 못했습니다.',
                message=f'invalid detected_objects_json: {exc}',
            )

        if not isinstance(objects, list) or len(objects) == 0:
            return self._fail_response(
                response,
                reason='작업공간에서 감지된 물체가 없습니다.',
                message='no_detected_objects',
            )

        if self.latest_raw_frame is None:
            return self._fail_response(
                response,
                reason='카메라 이미지를 받지 못해 지정 이동을 수행할 수 없습니다.',
                message='no_camera_image',
            )

        try:
            result = self.resolve_with_vlm(request.command, objects)
        except Exception as exc:
            self.get_logger().error(f'지정 이동 VLM 해석 실패: {exc}')
            return self._fail_response(
                response,
                reason='지정 이동 명령 해석 중 오류가 발생했습니다.',
                message=f'vlm_exception: {exc}',
            )

        return self._build_response(response, result, objects)

    # VLM 응답 dict와 물체 목록으로 service response를 채우는 함수
    def _build_response(self, response, result: Dict[str, Any], objects: List[Dict[str, Any]]):
        target_id = result.get('target_id', -1)
        place_zone = result.get('place_zone', '')
        confidence = float(result.get('confidence', 0.0))
        need_confirmation = bool(result.get('need_confirmation', False))
        reason = str(result.get('reason', '')).strip()

        valid_zones = self.zone_rules.get('zones', {})

        if not isinstance(target_id, int) or target_id < 0 or target_id >= len(objects):
            return self._fail_response(
                response,
                reason=reason or '말씀하신 물체를 찾지 못했습니다.',
                message='target_not_found',
            )

        if place_zone not in valid_zones:
            return self._fail_response(
                response,
                reason=reason or '옮길 구역을 이해하지 못했습니다.',
                message='invalid_place_zone',
            )

        if need_confirmation or confidence < self.confidence_threshold:
            return self._fail_response(
                response,
                reason=reason or '명령이 명확하지 않아 확인이 필요합니다.',
                message='low_confidence_or_need_confirmation',
            )

        response.success = True
        response.target_id = target_id
        response.place_zone = place_zone
        response.confidence = confidence
        response.need_confirmation = need_confirmation
        response.reason = reason
        response.message = 'resolved'
        return response

    # 실패 응답을 채우는 함수
    def _fail_response(self, response, reason: str, message: str):
        response.success = False
        response.target_id = -1
        response.place_zone = ''
        response.confidence = 0.0
        response.need_confirmation = False
        response.reason = reason
        response.message = message
        return response

    # 명령 문장과 물체 목록으로 GPT-4o에 보낼 프롬프트를 만드는 함수
    def build_prompt(self, command: str, objects: List[Dict[str, Any]]) -> str:
        object_lines = []
        for index, obj in enumerate(objects):
            name = obj.get('name', 'unknown') if isinstance(obj, dict) else 'unknown'
            object_lines.append(f'- id {index}: {name}')
        object_text = '\n'.join(object_lines)

        # 구역 이름 → 그 구역에 속하는 물체 클래스(의미 힌트)
        class_to_zone = self.zone_rules.get('class_to_zone', {})
        zone_names = list(self.zone_rules.get('zones', {}).keys())
        zone_hint_lines = []
        for zone_name in zone_names:
            classes = [cls for cls, z in class_to_zone.items() if z == zone_name]
            class_text = ', '.join(classes) if classes else '(지정된 물체 없음)'
            zone_hint_lines.append(f'- {zone_name}: {class_text}')
        zone_text = '\n'.join(zone_hint_lines)

        return f"""
너는 협동로봇에게 "어떤 물체를 어느 구역으로 옮길지"를 정해주는 판단자다.

사용자 명령:
"{command}"

현재 작업공간에서 감지된 물체 목록(id는 정수):
{object_text}

옮길 수 있는 구역(color 이름과 그 구역에 배치되는 물체 종류):
{zone_text}

함께 제공되는 이미지는 현재 작업공간 사진이다. 명령과 이미지를 함께 보고 판단하라.

규칙:
1. target_id는 위 목록의 id 중 하나여야 한다. 명령에 맞는 물체가 없으면 -1.
2. place_zone은 위 구역 color 이름({', '.join(zone_names)}) 중 하나여야 한다. 못 정하면 빈 문자열.
3. 명령이 모호하거나(예: 같은 물체가 여러 개로 보임) 이미지에서 대상 물체가 안 보이면
   need_confirmation을 true로 한다.
4. confidence는 0.0~1.0 사이 확신도다.

반드시 아래 JSON 형식으로만 답하라. 다른 텍스트, 설명, markdown은 절대 쓰지 마라.
{{
  "target_id": <정수>,
  "place_zone": "<color 이름 또는 빈 문자열>",
  "confidence": <0.0~1.0>,
  "need_confirmation": <true 또는 false>,
  "reason": "<짧은 한국어 설명>"
}}
""".strip()

    # 최신 이미지와 프롬프트로 GPT-4o를 호출해 판단 결과 dict를 반환하는 함수
    def resolve_with_vlm(self, command: str, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self.build_prompt(command, objects)

        user_content = [
            {'type': 'text', 'text': prompt},
        ]

        image_url = self.frame_to_data_url(self.latest_raw_frame)
        if image_url is not None:
            user_content.append({
                'type': 'image_url',
                'image_url': {
                    'url': image_url,
                    'detail': 'low',
                },
            })

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=200,
            response_format={'type': 'json_object'},
            messages=[
                {
                    'role': 'system',
                    'content': (
                        '너는 로봇 지정 이동 명령을 해석해 target_id와 place_zone을 JSON으로만 '
                        '출력하는 판단자다.'
                    ),
                },
                {
                    'role': 'user',
                    'content': user_content,
                },
            ],
        )

        content = completion.choices[0].message.content
        return json.loads(content)

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


# ROS2 directed_move_vlm_node를 실행하고 해석 service callback을 계속 처리하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = DirectedMoveVLMNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
