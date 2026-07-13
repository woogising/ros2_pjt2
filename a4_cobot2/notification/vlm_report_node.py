# ============================================================
# notification/vlm_report_node.py
# 역할:
#   - 최종 재검증 결과를 사용자에게 보고할 문장을 생성합니다.
#   - 현재 재검증 3자세 raw 이미지를 CLIP으로 임베딩하여 과거 사례를 검색합니다.
#   - 현재 이미지 + annotated 이미지 + 유사 사례를 VLM에 함께 입력합니다.
#   - VLM의 객관적 묘사와 보고문을 현재 recheck metadata.json에 저장합니다.
#   - 보고 생성이 끝난 뒤 현재 사례를 VectorDB에 추가합니다.
#
# 중요:
#   - 로봇 동작 판단은 기존 workspace_judge_node의 공식 판단을 계속 사용합니다.
#   - VectorDB와 VLM 묘사는 보고문 보조 자료이며 로봇 이동 판단을 덮어쓰지 않습니다.
# ============================================================
import base64
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from dotenv import load_dotenv
from openai import OpenAI
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

from od_msg.srv import GenerateReport
from notification.image_case_memory import (
    ImageCaseMemory,
    extract_case_id,
    metadata_path_for_case,
    write_json_atomic,
)
from notification.notice_utils import make_recheck_remaining_notice
from workspace.workspace_judge_utils import get_default_zone_rules


PACKAGE_NAME = 'a4_cobot2'
ALLOWED_LAYOUT_STATUS = {
    'normal_clean',
    'normal_untidy',
    'misplaced',
    'overlap',
    'boundary',
    'occluded',
    'unknown',
}


class VLMReportNode(Node):
    def __init__(self):
        super().__init__('vlm_report_node')

        # 기존 VLM 파라미터
        self.declare_parameter('use_vlm', True)
        self.declare_parameter('model', 'gpt-4o')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('annotated_image_topic', '/yolo_detection_image')
        self.declare_parameter('max_image_width', 960)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('openai_timeout_sec', 15.0)

        # 사례 메모리 파라미터. launch 파일을 수정하지 않아도 이 기본값으로 동작합니다.
        self.declare_parameter('memory_enabled', True)
        self.declare_parameter('memory_dir', '')
        self.declare_parameter('memory_embedding_model', 'openai/clip-vit-base-patch32')
        self.declare_parameter('memory_device', 'auto')
        self.declare_parameter('memory_top_k', 1)
        self.declare_parameter('memory_min_similarity', 0.85)
        self.declare_parameter('memory_reference_image_limit', 1)
        self.declare_parameter('memory_auto_index', True)
        self.declare_parameter('memory_require_three_raw_images', True)
        self.declare_parameter('memory_backfill_existing_metadata', True)

        self.use_vlm = self.get_parameter('use_vlm').value
        self.model = self.get_parameter('model').value
        self.image_topic = self.get_parameter('image_topic').value
        self.annotated_image_topic = self.get_parameter('annotated_image_topic').value
        self.max_image_width = int(self.get_parameter('max_image_width').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.openai_timeout_sec = float(self.get_parameter('openai_timeout_sec').value)

        self.memory_enabled = bool(self.get_parameter('memory_enabled').value)
        self.memory_dir_param = str(self.get_parameter('memory_dir').value).strip()
        self.memory_embedding_model = str(self.get_parameter('memory_embedding_model').value)
        self.memory_device = str(self.get_parameter('memory_device').value)
        self.memory_top_k = int(self.get_parameter('memory_top_k').value)
        self.memory_min_similarity = float(self.get_parameter('memory_min_similarity').value)
        self.memory_reference_image_limit = int(
            self.get_parameter('memory_reference_image_limit').value
        )
        self.memory_auto_index = bool(self.get_parameter('memory_auto_index').value)
        self.memory_require_three_raw_images = bool(
            self.get_parameter('memory_require_three_raw_images').value
        )
        self.memory_backfill_existing_metadata = bool(
            self.get_parameter('memory_backfill_existing_metadata').value
        )

        self.bridge = CvBridge()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.zone_rules = get_default_zone_rules()
        self.client = self._create_openai_client()

        # 첫 재검증 요청에서 이미지 저장 위치를 보고 자동으로 생성합니다.
        self.case_memory: Optional[ImageCaseMemory] = None
        self.case_memory_dir: Optional[str] = None
        self._backfilled_metadata_dirs = set()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.raw_image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.raw_image_callback,
            image_qos,
        )
        self.annotated_image_sub = self.create_subscription(
            Image,
            self.annotated_image_topic,
            self.annotated_image_callback,
            image_qos,
        )
        self.report_srv = self.create_service(
            GenerateReport,
            '/generate_final_report',
            self.handle_generate_final_report,
        )

        self.get_logger().info('VLMReportNode with image case memory started.')
        self.get_logger().info(f'use_vlm={self.use_vlm}, model={self.model}')
        self.get_logger().info(
            f'memory_enabled={self.memory_enabled}, '
            f'embedding_model={self.memory_embedding_model}, '
            f'top_k={self.memory_top_k}, min_similarity={self.memory_min_similarity}'
        )

    def _create_openai_client(self):
        if not self.use_vlm:
            self.get_logger().warn('use_vlm=False 이므로 fallback 보고문만 사용합니다.')
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

            return OpenAI(api_key=api_key, timeout=self.openai_timeout_sec)
        except Exception as exc:
            self.get_logger().warn(
                f'OpenAI client 초기화 실패. fallback 보고문만 사용합니다: {exc}'
            )
            self.use_vlm = False
            return None

    def raw_image_callback(self, msg: Image):
        try:
            self.latest_raw_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'원본 이미지 변환 실패: {exc}')

    def annotated_image_callback(self, msg: Image):
        try:
            self.latest_annotated_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as exc:
            self.get_logger().warn(f'annotated 이미지 변환 실패: {exc}')

    # --------------------------------------------------------
    # 서비스 처리
    # --------------------------------------------------------
    def handle_generate_final_report(self, request, response):
        try:
            request_payload = json.loads(request.report_request_json)
        except json.JSONDecodeError as exc:
            response.success = False
            response.report_text = ''
            response.report_json = json.dumps(
                {'source': 'error', 'reason': 'invalid_report_request_json'},
                ensure_ascii=False,
            )
            response.message = f'invalid report_request_json: {exc}'
            return response

        fallback_notice = self.make_fallback_report(request_payload)
        available_scan_image_count = self.count_available_scan_images(request_payload)
        has_latest_topic_image = (
            self.latest_raw_frame is not None or self.latest_annotated_frame is not None
        )

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

        if available_scan_image_count == 0 and not has_latest_topic_image:
            response.success = True
            response.report_text = fallback_notice
            response.report_json = json.dumps(
                {
                    'source': 'fallback',
                    'reason': 'no_scan_or_latest_image_received',
                    'model': self.model,
                    'available_scan_image_count': available_scan_image_count,
                },
                ensure_ascii=False,
            )
            response.message = 'no scan/latest image received, fallback report generated'
            return response

        memory_context = self.prepare_memory_context(request_payload)

        try:
            vlm_analysis = self.generate_vlm_report(
                request_payload=request_payload,
                fallback_notice=fallback_notice,
                retrieved_cases=memory_context.get('retrieved_cases', []),
            )
            report_text = vlm_analysis['report_text']

            persistence = self.persist_case(
                request_payload=request_payload,
                vlm_analysis=vlm_analysis,
                memory_context=memory_context,
                analysis_source='vlm',
            )

            response.success = True
            response.report_text = report_text
            response.report_json = json.dumps(
                {
                    'source': 'vlm',
                    'model': self.model,
                    'available_scan_image_count': available_scan_image_count,
                    'used_scan_images': available_scan_image_count > 0,
                    'used_latest_topic_images_as_fallback': available_scan_image_count == 0,
                    'used_case_memory': bool(memory_context.get('retrieved_cases')),
                    'retrieved_case_count': len(memory_context.get('retrieved_cases', [])),
                    'retrieved_cases': self.compact_retrieved_cases(
                        memory_context.get('retrieved_cases', [])
                    ),
                    'case_id': memory_context.get('case_id'),
                    'metadata_path': persistence.get('metadata_path'),
                    'memory_index_status': persistence.get('index_status'),
                    'memory_index_size': persistence.get('index_size'),
                    'memory_error': memory_context.get('error'),
                },
                ensure_ascii=False,
            )
            response.message = 'vlm report generated and case memory updated'
            return response

        except Exception as exc:
            self.get_logger().error(f'VLM 보고문 생성 실패: {exc}')

            # VLM이 실패해도 현재 사례의 공식 판단과 fallback 결과는 metadata에 남깁니다.
            fallback_analysis = {
                'report_text': fallback_notice,
                'description': fallback_notice,
                'layout_status': 'unknown',
                'confidence': 0.0,
            }
            persistence = self.persist_case(
                request_payload=request_payload,
                vlm_analysis=fallback_analysis,
                memory_context=memory_context,
                analysis_source='fallback',
                force_search_disabled=True,
            )

            response.success = True
            response.report_text = fallback_notice
            response.report_json = json.dumps(
                {
                    'source': 'fallback',
                    'reason': 'vlm_exception',
                    'error': str(exc),
                    'model': self.model,
                    'case_id': memory_context.get('case_id'),
                    'metadata_path': persistence.get('metadata_path'),
                    'memory_index_status': 'not_indexed_vlm_failed',
                    'memory_error': memory_context.get('error'),
                },
                ensure_ascii=False,
            )
            response.message = f'vlm failed, fallback report generated: {exc}'
            return response

    # --------------------------------------------------------
    # 사례 메모리 준비/저장
    # --------------------------------------------------------
    def collect_scan_image_paths(
        self, request_payload: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        scan_images = request_payload.get('scan_images', [])
        if not isinstance(scan_images, list):
            return [], []

        sortable_items = []
        for order, item in enumerate(scan_images):
            if not isinstance(item, dict):
                continue
            raw_index = item.get('index', order)
            try:
                index = int(raw_index)
            except Exception:
                index = order
            sortable_items.append((index, item))

        sortable_items.sort(key=lambda pair: pair[0])

        raw_paths = []
        annotated_paths = []
        for _, item in sortable_items:
            raw_path = item.get('raw_image_path')
            if raw_path and os.path.exists(str(raw_path)):
                raw_paths.append(os.path.abspath(str(raw_path)))

            annotated_path = item.get('annotated_image_path')
            if annotated_path and os.path.exists(str(annotated_path)):
                annotated_paths.append(os.path.abspath(str(annotated_path)))

        return raw_paths, annotated_paths

    def resolve_memory_dir(self, raw_paths: Sequence[str]) -> str:
        if self.memory_dir_param:
            return os.path.abspath(os.path.expanduser(self.memory_dir_param))

        first_path = os.path.abspath(raw_paths[0])
        scan_dir = os.path.dirname(first_path)
        if os.path.basename(scan_dir) == 'scan_images':
            return os.path.join(os.path.dirname(scan_dir), 'vector_db')
        return os.path.join(scan_dir, 'vector_db')

    def ensure_case_memory(self, raw_paths: Sequence[str]) -> ImageCaseMemory:
        resolved_dir = self.resolve_memory_dir(raw_paths)

        if self.case_memory is not None and self.case_memory_dir == resolved_dir:
            return self.case_memory

        self.case_memory = ImageCaseMemory(
            root_dir=resolved_dir,
            model_name=self.memory_embedding_model,
            device=self.memory_device,
            logger=self.get_logger(),
        )
        self.case_memory_dir = resolved_dir
        self.get_logger().info(
            f'재검증 사례 메모리를 준비했습니다: {self.case_memory.stats()}'
        )
        return self.case_memory

    def prepare_memory_context(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_paths, annotated_paths = self.collect_scan_image_paths(request_payload)
        context: Dict[str, Any] = {
            'enabled': self.memory_enabled,
            'raw_paths': raw_paths,
            'annotated_paths': annotated_paths,
            'case_id': extract_case_id(raw_paths + annotated_paths),
            'embedding': None,
            'retrieved_cases': [],
            'error': None,
        }

        if not self.memory_enabled:
            return context
        if not raw_paths:
            context['error'] = 'no_raw_scan_images'
            return context

        if context['case_id'] is None:
            context['case_id'] = f'recheck_unknown_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        try:
            memory = self.ensure_case_memory(raw_paths)

            metadata_dir = os.path.dirname(os.path.abspath(raw_paths[0]))
            if (
                self.memory_backfill_existing_metadata
                and metadata_dir not in self._backfilled_metadata_dirs
            ):
                backfill_result = memory.backfill_from_metadata_dir(
                    metadata_dir=metadata_dir,
                    require_three_raw_images=self.memory_require_three_raw_images,
                )
                self._backfilled_metadata_dirs.add(metadata_dir)
                self.get_logger().info(
                    f'기존 정상 사례 VectorDB 동기화 완료: {backfill_result}'
                )

            embedding = memory.encode_case(raw_paths)
            retrieved_cases = memory.search(
                query_embedding=embedding,
                top_k=max(0, self.memory_top_k),
                min_similarity=self.memory_min_similarity,
                exclude_case_id=context['case_id'],
            )

            context['embedding'] = embedding
            context['retrieved_cases'] = retrieved_cases

            self.get_logger().info(
                '유사 재검증 사례 검색 완료: '
                f'case_id={context["case_id"]}, '
                f'retrieved={[(item.get("case_id"), round(item.get("similarity", 0.0), 4)) for item in retrieved_cases]}'
            )
        except Exception as exc:
            context['error'] = str(exc)
            self.get_logger().warn(
                f'사례 메모리를 사용할 수 없어 현재 이미지로만 VLM 보고를 생성합니다: {exc}'
            )

        return context

    def compact_retrieved_cases(self, retrieved_cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                'case_id': item.get('case_id'),
                'similarity': round(float(item.get('similarity', 0.0)), 6),
                'reference_status': item.get('reference_status', 'normal'),
                'reference_visual_status': item.get(
                    'reference_visual_status', item.get('layout_status', 'unknown')
                ),
                'reference_source': item.get('reference_source', 'unknown'),
                'description': item.get('description', ''),
            }
            for item in retrieved_cases
        ]

    def infer_reference_label(
        self,
        judgement_payload: Dict[str, Any],
        vlm_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """현재 사례가 정상 기준 이미지로 검색 가능한지 보수적으로 분류합니다."""
        result = str(judgement_payload.get('result', 'unknown'))
        layout_status = str(vlm_analysis.get('layout_status', 'unknown'))
        placement_status = 'unknown'
        source = 'official_judgement'

        if result == 'all_clear':
            placement_status = 'normal'
        elif result == 'misplaced_found':
            misplaced = judgement_payload.get('misplaced_objects', [])
            reasons = [
                str(item.get('reason', ''))
                for item in misplaced
                if isinstance(item, dict)
            ]
            if reasons and all(reason == 'untidy_in_zone' for reason in reasons):
                # 구역 자체는 맞고 정돈 기준만 부족한 경우는 정상 배치 참고 사례로 사용합니다.
                placement_status = 'normal'
                layout_status = 'normal_untidy'
                source = 'official_judgement_untidy_in_zone'
            else:
                placement_status = 'abnormal'

        return {
            'placement_status': placement_status,
            'visual_status': layout_status,
            'source': source,
            'description': str(vlm_analysis.get('description', '')).strip(),
        }

    def persist_case(
        self,
        request_payload: Dict[str, Any],
        vlm_analysis: Dict[str, Any],
        memory_context: Dict[str, Any],
        analysis_source: str,
        force_search_disabled: bool = False,
    ) -> Dict[str, Any]:
        raw_paths = memory_context.get('raw_paths', [])
        annotated_paths = memory_context.get('annotated_paths', [])
        case_id = memory_context.get('case_id')

        if not case_id or not raw_paths:
            return {
                'metadata_path': None,
                'index_status': 'not_saved_no_case_images',
                'index_size': None,
            }

        judgement_payload = request_payload.get('judgement_payload', {})
        if not isinstance(judgement_payload, dict):
            judgement_payload = {}

        required_count = 3 if self.memory_require_three_raw_images else 1
        reference_label = self.infer_reference_label(
            judgement_payload=judgement_payload,
            vlm_analysis=vlm_analysis,
        )

        search_enabled = (
            self.memory_enabled
            and self.memory_auto_index
            and not force_search_disabled
            and analysis_source == 'vlm'
            and len(raw_paths) >= required_count
            and reference_label.get('placement_status') == 'normal'
            and memory_context.get('embedding') is not None
            and self.case_memory is not None
        )

        metadata_path = metadata_path_for_case(case_id, raw_paths)
        metadata = {
            'schema_version': 1,
            'case_id': case_id,
            'created_at': self.created_at_from_case_id(case_id),
            'images': {
                'raw': [os.path.basename(path) for path in raw_paths],
                'annotated': [os.path.basename(path) for path in annotated_paths],
            },
            'official_judgement': judgement_payload,
            # VectorDB 검색에는 공식 원본 판정과 별도로 reference_label을 사용합니다.
            # 사람이 metadata를 확인해 human_confirmed로 바꾸면 그 값이 최우선입니다.
            'reference_label': reference_label,
            'detected_objects': request_payload.get('detected_objects', []),
            'detected_objects_frame': request_payload.get('detected_objects_frame'),
            'vlm_analysis': {
                'source': analysis_source,
                'model': self.model,
                'layout_status': vlm_analysis.get('layout_status', 'unknown'),
                'description': vlm_analysis.get('description', ''),
                'report_text': vlm_analysis.get('report_text', ''),
                'confidence': float(vlm_analysis.get('confidence', 0.0)),
            },
            'retrieved_cases': self.compact_retrieved_cases(
                memory_context.get('retrieved_cases', [])
            ),
            'vector_memory': {
                'enabled': self.memory_enabled,
                'embedding_model': self.memory_embedding_model,
                'backend': self.case_memory.backend_name if self.case_memory else None,
                'store_dir': self.case_memory_dir,
                'search_enabled': search_enabled,
                'added_to_index': False,
                'memory_error': memory_context.get('error'),
            },
            # 자동 누적은 하되, 사람이 나중에 확인/수정할 수 있게 상태를 분리합니다.
            'review_status': 'unreviewed',
        }

        write_json_atomic(metadata_path, metadata)

        index_status = 'metadata_only_search_disabled'
        index_size = None
        if search_enabled:
            try:
                result = self.case_memory.add_case(
                    case_id=case_id,
                    embedding=memory_context['embedding'],
                    metadata_path=metadata_path,
                    raw_image_paths=raw_paths,
                    annotated_image_paths=annotated_paths,
                )
                index_status = result.get('status', 'added')
                index_size = result.get('index_size')
                metadata['vector_memory']['added_to_index'] = True
                metadata['vector_memory']['index_status'] = index_status
                metadata['vector_memory']['index_size'] = index_size
                write_json_atomic(metadata_path, metadata)
            except Exception as exc:
                index_status = f'index_failed: {exc}'
                metadata['vector_memory']['index_error'] = str(exc)
                write_json_atomic(metadata_path, metadata)
                self.get_logger().warn(f'현재 사례 VectorDB 추가 실패: {exc}')

        self.get_logger().info(
            f'재검증 사례 metadata 저장: {metadata_path}, index_status={index_status}'
        )
        return {
            'metadata_path': metadata_path,
            'index_status': index_status,
            'index_size': index_size,
        }

    def created_at_from_case_id(self, case_id: str) -> str:
        match = re.fullmatch(r'recheck_(\d{8})_(\d{6})', case_id)
        if match:
            try:
                return datetime.strptime(
                    f'{match.group(1)}{match.group(2)}', '%Y%m%d%H%M%S'
                ).isoformat()
            except Exception:
                pass
        return datetime.now().isoformat(timespec='seconds')

    # --------------------------------------------------------
    # 보고문 생성
    # --------------------------------------------------------
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
            names = [obj.get('name', '알 수 없는 물체') for obj in unknown_objects]
            object_text = ', '.join(names) if names else '일부 물체'
            return (
                '정리 후 재검증을 했지만 일부 물체의 배치 규칙을 찾을 수 없습니다. '
                f'확인이 필요한 물체는 {object_text}입니다.'
            )
        if result == 'no_objects':
            return '재검증 중 감지된 물체가 없습니다. 카메라 시야 또는 작업공간을 확인해주세요.'
        return '정리 후 작업공간 상태를 정확히 판단하지 못했습니다. 확인이 필요합니다.'

    def build_prompt(
        self,
        request_payload: Dict[str, Any],
        fallback_notice: str,
        retrieved_cases: Sequence[Dict[str, Any]],
    ) -> str:
        scan_images = request_payload.get('scan_images', [])
        scan_image_summary = []

        if isinstance(scan_images, list):
            for item in scan_images:
                if not isinstance(item, dict):
                    continue
                scan_image_summary.append(
                    {
                        'index': item.get('index'),
                        'total': item.get('total'),
                        'scan_mode': item.get('scan_mode'),
                        'annotation_source': item.get('annotation_source', 'unknown'),
                        'has_raw_image': bool(item.get('raw_image_path')),
                        'has_annotated_image': bool(item.get('annotated_image_path')),
                    }
                )

        reference_summary = self.compact_retrieved_cases(retrieved_cases)
        compact_payload = {
            'report_mode': request_payload.get(
                'report_mode', 'final_recheck_visual_check'
            ),
            'detected_objects': request_payload.get('detected_objects', []),
            'detected_objects_frame': request_payload.get('detected_objects_frame'),
            'judgement_payload': request_payload.get('judgement_payload', {}),
            'scan_image_summary': scan_image_summary,
            'retrieved_reference_cases': reference_summary,
            'zone_rules': self.zone_rules,
            'fallback_notice': fallback_notice,
        }

        return f"""
너는 협동로봇 작업공간 정리 시스템의 최종 재검증 보고 담당자다.

현재 이미지:
- 원본 이미지는 로봇 정리 후 3자세에서 실제로 촬영한 작업공간이다.
- annotated 이미지는 YOLO와 RT-DETR 융합 bbox 및 SAM2 또는 YOLO fallback mask를 보여준다.

과거 유사 사례:
- retrieved_reference_cases와 뒤에 첨부된 참고 이미지는 정상 배치로 승인된 과거 사례다.
- similarity는 시각적 유사도이며 정상 확률이나 비정상 확률이 아니다.
- 유사도가 기준 이상인 가장 가까운 정상 사례 하나만 제공된다. 참고 사례의 설명을 그대로 복사하지 말고 현재 이미지에서 실제로 확인되는 내용만 사용한다.

반드시 지킬 원칙:
1. 로봇의 공식 정상/비정상 판단은 judgement_payload를 최우선으로 한다.
2. 과거 사례와 VLM의 시각 묘사는 공식 판단을 덮어쓰지 않는다.
3. 공식 판단과 현재 이미지가 명확히 충돌하면 단정하지 말고 추가 확인이 필요하다고 표현한다.
4. 물체 간격, 겹침, 경계 근접, 가림, 전반적인 정돈 상태는 현재 이미지를 기준으로 객관적으로 묘사한다.
5. 좌표, 내부 JSON 키, 이미지 경로, 모델 내부 처리 방식은 사용자에게 말하지 않는다.
6. report_text는 TTS로 읽기 좋은 한국어 2~4문장으로 작성한다.
7. description은 metadata에 저장할 객관적인 한두 문장으로 작성한다.
8. layout_status는 아래 값 중 하나만 사용한다.
   - normal_clean: 규칙을 만족하고 정돈 상태도 좋음
   - normal_untidy: 규칙은 만족하지만 간격/방향/정돈 개선이 필요함
   - misplaced: 공식 판단상 잘못된 구역 배치가 있음
   - overlap: 물체끼리 겹침이 두드러짐
   - boundary: 구역 경계에 매우 가까워 확인 필요
   - occluded: 가림 때문에 시각 확인이 어려움
   - unknown: 판단하기 어려움
9. confidence는 시각 묘사에 대한 확신도로 0.0~1.0 숫자다. 공식 판정 확률이 아니다.

입력 데이터:
{json.dumps(compact_payload, ensure_ascii=False, indent=2)}

반드시 아래 JSON 객체만 출력한다. markdown 코드 블록은 쓰지 않는다.
{{
  "report_text": "사용자에게 말할 최종 보고문",
  "description": "현재 배치의 객관적인 시각 묘사",
  "layout_status": "normal_clean|normal_untidy|misplaced|overlap|boundary|occluded|unknown",
  "confidence": 0.0
}}
""".strip()

    def generate_vlm_report(
        self,
        request_payload: Dict[str, Any],
        fallback_notice: str,
        retrieved_cases: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(request_payload, fallback_notice, retrieved_cases)
        user_content = [{'type': 'text', 'text': prompt}]

        used_scan_image_count = self.append_scan_images_to_user_content(
            user_content=user_content,
            request_payload=request_payload,
        )

        if used_scan_image_count == 0:
            raw_image_url = self.frame_to_data_url(self.latest_raw_frame)
            annotated_image_url = self.frame_to_data_url(self.latest_annotated_frame)

            if raw_image_url is not None:
                user_content.extend(
                    [
                        {
                            'type': 'text',
                            'text': '저장된 재검증 파일을 사용할 수 없어 최신 원본 이미지를 대신 사용합니다.',
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': raw_image_url, 'detail': 'low'},
                        },
                    ]
                )

            if annotated_image_url is not None:
                user_content.extend(
                    [
                        {
                            'type': 'text',
                            'text': '저장된 재검증 파일을 사용할 수 없어 최신 최종 앙상블 이미지를 대신 사용합니다.',
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': annotated_image_url,
                                'detail': 'low',
                            },
                        },
                    ]
                )

        self.append_reference_cases_to_user_content(user_content, retrieved_cases)

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=450,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        '너는 로봇 작업공간 정리 결과를 공식 판단 JSON, 현재 재검증 이미지, '
                        '과거 유사 사례를 바탕으로 짧고 정확하게 보고하는 한국어 안내자다. '
                        '응답은 사용자가 요구한 JSON 객체 형식만 따른다.'
                    ),
                },
                {'role': 'user', 'content': user_content},
            ],
        )

        content = completion.choices[0].message.content
        return self.parse_vlm_analysis(content, fallback_notice)

    def parse_vlm_analysis(
        self, content: Optional[str], fallback_notice: str
    ) -> Dict[str, Any]:
        if content is None or content.strip() == '':
            raise ValueError('VLM 응답이 비어 있습니다.')

        text = content.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)

        payload = None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            first = text.find('{')
            last = text.rfind('}')
            if first >= 0 and last > first:
                payload = json.loads(text[first:last + 1])

        if not isinstance(payload, dict):
            # 구버전처럼 일반 문장만 응답한 경우에도 최종 보고는 살립니다.
            return {
                'report_text': text or fallback_notice,
                'description': text or fallback_notice,
                'layout_status': 'unknown',
                'confidence': 0.0,
            }

        report_text = str(payload.get('report_text', '')).strip()
        description = str(payload.get('description', '')).strip()
        layout_status = str(payload.get('layout_status', 'unknown')).strip()

        if report_text == '':
            report_text = fallback_notice
        if description == '':
            description = report_text
        if layout_status not in ALLOWED_LAYOUT_STATUS:
            layout_status = 'unknown'

        try:
            confidence = float(payload.get('confidence', 0.0))
        except Exception:
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        return {
            'report_text': report_text,
            'description': description,
            'layout_status': layout_status,
            'confidence': confidence,
        }

    # --------------------------------------------------------
    # 이미지 입력 구성
    # --------------------------------------------------------
    def frame_to_data_url(self, frame) -> Optional[str]:
        if frame is None:
            return None

        image = frame.copy()
        height, width = image.shape[:2]

        if self.max_image_width > 0 and width > self.max_image_width:
            scale = self.max_image_width / float(width)
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
            )

        ok, encoded = cv2.imencode(
            '.jpg',
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            return None

        image_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        return f'data:image/jpeg;base64,{image_b64}'

    def image_path_to_data_url(self, image_path: str) -> Optional[str]:
        if image_path is None or str(image_path).strip() == '':
            return None
        if not os.path.exists(image_path):
            self.get_logger().warn(f'VLM 입력 이미지 파일이 없습니다: {image_path}')
            return None

        frame = cv2.imread(image_path)
        if frame is None:
            self.get_logger().warn(f'VLM 입력 이미지 파일을 읽지 못했습니다: {image_path}')
            return None
        return self.frame_to_data_url(frame)

    def count_available_scan_images(self, request_payload: Dict[str, Any]) -> int:
        raw_paths, annotated_paths = self.collect_scan_image_paths(request_payload)
        return len(raw_paths) + len(annotated_paths)

    def append_scan_images_to_user_content(
        self, user_content: list, request_payload: Dict[str, Any]
    ) -> int:
        used_count = 0
        scan_images = request_payload.get('scan_images', [])
        if not isinstance(scan_images, list):
            return 0

        sortable_items = []
        for order, item in enumerate(scan_images):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get('index', order))
            except Exception:
                index = order
            sortable_items.append((index, item))

        for index, item in sorted(sortable_items, key=lambda pair: pair[0]):
            total = item.get('total', len(sortable_items))
            pose_number = index + 1

            raw_url = self.image_path_to_data_url(item.get('raw_image_path'))
            if raw_url is not None:
                user_content.extend(
                    [
                        {
                            'type': 'text',
                            'text': f'현재 최종 재검증 자세 {pose_number}/{total} 원본 이미지입니다.',
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': raw_url, 'detail': 'low'},
                        },
                    ]
                )
                used_count += 1

            annotated_url = self.image_path_to_data_url(
                item.get('annotated_image_path')
            )
            if annotated_url is not None:
                user_content.extend(
                    [
                        {
                            'type': 'text',
                            'text': (
                                f'현재 최종 재검증 자세 {pose_number}/{total} 최종 앙상블 표시 이미지입니다. '
                                'bbox는 YOLO와 RT-DETR 융합 결과이며 mask는 SAM2 또는 YOLO fallback 결과입니다.'
                            ),
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': annotated_url, 'detail': 'low'},
                        },
                    ]
                )
                used_count += 1

        return used_count

    def append_reference_cases_to_user_content(
        self,
        user_content: list,
        retrieved_cases: Sequence[Dict[str, Any]],
    ) -> int:
        """가장 유사한 정상 사례 하나의 raw pose 1/2/3을 모두 VLM에 넣습니다."""
        if not retrieved_cases:
            return 0

        case_limit = max(0, self.memory_reference_image_limit)
        if case_limit <= 0:
            return 0

        used_count = 0
        for rank, case in enumerate(retrieved_cases[:case_limit], start=1):
            similarity = float(case.get('similarity', 0.0))
            if similarity < self.memory_min_similarity:
                continue

            raw_paths = case.get('raw_image_paths', [])
            if not isinstance(raw_paths, list):
                raw_paths = []

            description = str(case.get('description', '')).strip()
            visual_status = case.get(
                'reference_visual_status', case.get('layout_status', 'unknown')
            )
            reference_source = case.get('reference_source', 'unknown')

            user_content.append(
                {
                    'type': 'text',
                    'text': (
                        f'과거 정상 기준 사례 {rank}입니다. '
                        f'현재 사례와의 유사도={similarity:.4f}, '
                        f'시각 상태={visual_status}, 승인 출처={reference_source}, '
                        f'기준 설명={description or "설명 없음"}. '
                        '이어지는 세 장은 동일 사례의 자세 1/3, 2/3, 3/3 원본 이미지입니다. '
                        '현재 장면과 비교하는 참고 자료로만 사용하세요.'
                    ),
                }
            )

            total = len(raw_paths)
            for pose_index, image_path in enumerate(raw_paths, start=1):
                image_url = self.image_path_to_data_url(image_path)
                if image_url is None:
                    continue
                user_content.extend(
                    [
                        {
                            'type': 'text',
                            'text': (
                                f'과거 정상 기준 사례 {rank}의 원본 자세 '
                                f'{pose_index}/{total} 이미지입니다.'
                            ),
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': image_url, 'detail': 'low'},
                        },
                    ]
                )
                used_count += 1

        return used_count



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
