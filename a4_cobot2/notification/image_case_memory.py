# ============================================================
# notification/image_case_memory.py
# 역할:
#   - 최종 재검증 3자세 raw 이미지를 CLIP 임베딩 하나로 변환합니다.
#   - 과거 재검증 사례를 FAISS IndexFlatIP에 저장하고 유사 사례를 검색합니다.
#   - FAISS가 설치되지 않은 경우에는 동일한 정규화 벡터에 대해 NumPy 정확 검색을 사용합니다.
#
# 저장 구조:
#   notification/vector_db/
#     ├── recheck_cases.faiss
#     ├── case_catalog.json
#     └── embeddings/
#         └── recheck_YYYYMMDD_HHMMSS.npy
# ============================================================
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


CASE_ID_PATTERN = re.compile(
    r'(recheck_\d{8}_\d{6})_pose_\d+_of_\d+_(?:raw|annotated)\.(?:jpg|jpeg|png)$',
    re.IGNORECASE,
)


def _pose_sort_key(path: str):
    """pose_N_of_M 파일명을 자세 번호 순서로 정렬합니다."""
    name = os.path.basename(str(path))
    match = re.search(r'_pose_(\d+)_of_(\d+)_', name, re.IGNORECASE)
    if match:
        return (int(match.group(1)), name)
    return (10**9, name)


def sort_pose_paths(paths: Sequence[str]) -> List[str]:
    return sorted([str(path) for path in paths], key=_pose_sort_key)


def extract_case_id(image_paths: Sequence[str]) -> Optional[str]:
    """재검증 이미지 파일명에서 recheck_YYYYMMDD_HHMMSS case id를 추출합니다."""
    for image_path in image_paths:
        name = os.path.basename(str(image_path))
        match = CASE_ID_PATTERN.search(name)
        if match:
            return match.group(1)
    return None


def metadata_path_for_case(case_id: str, image_paths: Sequence[str]) -> str:
    """현재 이미지와 같은 scan_images 디렉터리에 metadata JSON 경로를 만듭니다."""
    for image_path in image_paths:
        if image_path:
            return os.path.join(os.path.dirname(os.path.abspath(image_path)), f'{case_id}_metadata.json')
    return os.path.abspath(f'{case_id}_metadata.json')


def write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """중간에 프로세스가 종료돼도 JSON이 반쯤 쓰이지 않도록 원자적으로 저장합니다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=f'.{target.name}.',
        suffix='.tmp',
        dir=str(target.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj)
    except Exception:
        return default


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError('임베딩 벡터의 norm이 0입니다.')
    return vector / norm


class ImageCaseMemory:
    """CLIP 이미지 임베딩과 FAISS를 이용하는 재검증 사례 메모리입니다."""

    def __init__(
        self,
        root_dir: str,
        model_name: str = 'openai/clip-vit-base-patch32',
        device: str = 'auto',
        logger: Any = None,
    ):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.model_name = model_name
        self.device_requested = device
        self.logger = logger

        self.catalog_path = os.path.join(self.root_dir, 'case_catalog.json')
        self.index_path = os.path.join(self.root_dir, 'recheck_cases.faiss')
        self.embeddings_dir = os.path.join(self.root_dir, 'embeddings')

        os.makedirs(self.embeddings_dir, exist_ok=True)

        self._lock = threading.RLock()
        self._catalog: List[Dict[str, Any]] = self._load_catalog()
        self._active_records: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._index = None

        self._torch = None
        self._processor = None
        self._model = None
        self._device = None

        self._rebuild_index()

    @property
    def backend_name(self) -> str:
        return 'faiss_index_flat_ip' if faiss is not None else 'numpy_cosine_fallback'

    def _log_info(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _log_warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warn(message)

    def _load_catalog(self) -> List[Dict[str, Any]]:
        payload = read_json(self.catalog_path, default=[])
        if not isinstance(payload, list):
            self._log_warn('case_catalog.json 형식이 올바르지 않아 빈 catalog로 시작합니다.')
            return []

        records = []
        seen = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get('case_id', '')).strip()
            embedding_path = str(item.get('embedding_path', '')).strip()
            if not case_id or not embedding_path or case_id in seen:
                continue
            seen.add(case_id)
            records.append(item)
        return records

    def _save_catalog(self) -> None:
        write_json_atomic(self.catalog_path, {'records': self._catalog})
        # 이전 버전에서 list 형식으로 읽도록 만들어진 경우를 피하기 위해 바로 list로 다시 저장합니다.
        # 위 write는 원자 저장 함수를 재사용하기 위한 중간 단계이고, 최종 파일은 list입니다.
        target = Path(self.catalog_path)
        fd, temp_path = tempfile.mkstemp(
            prefix=f'.{target.name}.', suffix='.tmp', dir=str(target.parent), text=True
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file_obj:
                json.dump(self._catalog, file_obj, ensure_ascii=False, indent=2)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _ensure_encoder(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError(
                '이미지 임베딩 의존성이 없습니다. '
                'requirements_vector_memory.txt를 설치해주세요. '
                f'원인: {exc}'
            ) from exc

        if self.device_requested == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device_requested

        self._log_info(f'이미지 임베딩 모델을 로드합니다: {self.model_name}, device={device}')

        processor = CLIPProcessor.from_pretrained(self.model_name)
        model = CLIPModel.from_pretrained(self.model_name)
        model.eval()
        model.to(device)

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device

        self._log_info('이미지 임베딩 모델 로드가 완료되었습니다.')

    def encode_case(self, image_paths: Sequence[str]) -> np.ndarray:
        """여러 자세 raw 이미지 임베딩을 평균하여 사례 벡터 하나를 반환합니다."""
        valid_paths = [
            os.path.abspath(str(path))
            for path in image_paths
            if path and os.path.exists(str(path))
        ]
        if not valid_paths:
            raise ValueError('임베딩할 raw 이미지가 없습니다.')

        self._ensure_encoder()

        from PIL import Image

        images = []
        try:
            for image_path in valid_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert('RGB').copy())

            inputs = self._processor(images=images, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(self._device)

            with self._torch.inference_mode():
                features = self._model.get_image_features(pixel_values=pixel_values)
                features = self._torch.nn.functional.normalize(features, p=2, dim=-1)
                case_feature = features.mean(dim=0, keepdim=True)
                case_feature = self._torch.nn.functional.normalize(case_feature, p=2, dim=-1)

            return case_feature[0].detach().cpu().numpy().astype(np.float32)
        finally:
            for image in images:
                image.close()

    def _rebuild_index(self) -> None:
        with self._lock:
            vectors = []
            active_records = []

            for record in self._catalog:
                embedding_path = str(record.get('embedding_path', ''))
                if not embedding_path or not os.path.exists(embedding_path):
                    continue
                try:
                    vector = normalize_vector(np.load(embedding_path))
                except Exception as exc:
                    self._log_warn(
                        f'사례 임베딩을 읽지 못해 제외합니다: '
                        f'{record.get("case_id")}, {exc}'
                    )
                    continue
                vectors.append(vector)
                active_records.append(record)

            self._active_records = active_records
            self._index = None

            if not vectors:
                self._matrix = None
                return

            matrix = np.stack(vectors).astype(np.float32)
            self._matrix = matrix

            if faiss is not None:
                index = faiss.IndexFlatIP(matrix.shape[1])
                index.add(matrix)
                self._index = index
                try:
                    faiss.write_index(index, self.index_path)
                except Exception as exc:
                    self._log_warn(f'FAISS index 파일 저장 실패: {exc}')

    def contains_case(self, case_id: str) -> bool:
        return any(record.get('case_id') == case_id for record in self._catalog)

    def add_case(
        self,
        case_id: str,
        embedding: np.ndarray,
        metadata_path: str,
        raw_image_paths: Sequence[str],
        annotated_image_paths: Sequence[str],
    ) -> Dict[str, Any]:
        """현재 사례를 저장합니다. 같은 case id는 중복 추가하지 않고 경로만 갱신합니다."""
        vector = normalize_vector(embedding)

        with self._lock:
            embedding_path = os.path.join(self.embeddings_dir, f'{case_id}.npy')
            np.save(embedding_path, vector)

            record = {
                'case_id': case_id,
                'embedding_path': os.path.abspath(embedding_path),
                'metadata_path': os.path.abspath(metadata_path),
                'raw_image_paths': [os.path.abspath(path) for path in raw_image_paths],
                'annotated_image_paths': [
                    os.path.abspath(path) for path in annotated_image_paths
                ],
            }

            status = 'added'
            for index, old_record in enumerate(self._catalog):
                if old_record.get('case_id') == case_id:
                    self._catalog[index] = record
                    status = 'updated'
                    break
            else:
                self._catalog.append(record)

            self._save_catalog()
            self._rebuild_index()

            return {
                'status': status,
                'case_id': case_id,
                'index_size': len(self._active_records),
                'backend': self.backend_name,
            }

    def _reference_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """사례의 검색용 기준 라벨을 반환합니다. 사람 확인 라벨을 가장 우선합니다."""
        reference = metadata.get('reference_label', {})
        if isinstance(reference, dict) and str(reference.get('placement_status', '')).strip():
            return {
                'placement_status': str(reference.get('placement_status', 'unknown')).strip(),
                'visual_status': str(reference.get('visual_status', 'unknown')).strip(),
                'description': str(reference.get('description', '')).strip(),
                'source': str(reference.get('source', 'metadata')).strip(),
            }

        official = metadata.get('official_judgement', {})
        if not isinstance(official, dict):
            official = {}
        vlm_analysis = metadata.get('vlm_analysis', {})
        if not isinstance(vlm_analysis, dict):
            vlm_analysis = {}

        result = str(official.get('result', 'unknown'))
        placement_status = 'unknown'
        visual_status = str(vlm_analysis.get('layout_status', 'unknown'))

        if result == 'all_clear':
            placement_status = 'normal'
        elif result == 'misplaced_found':
            misplaced = official.get('misplaced_objects', [])
            reasons = [
                str(item.get('reason', ''))
                for item in misplaced
                if isinstance(item, dict)
            ]
            if reasons and all(reason == 'untidy_in_zone' for reason in reasons):
                placement_status = 'normal'
                visual_status = 'normal_untidy'
            else:
                placement_status = 'abnormal'

        return {
            'placement_status': placement_status,
            'visual_status': visual_status,
            'description': str(vlm_analysis.get('description', '')).strip(),
            'source': 'inferred_from_original_metadata',
        }

    def _metadata_allows_search(self, metadata: Dict[str, Any]) -> bool:
        vector_memory = metadata.get('vector_memory', {})
        if isinstance(vector_memory, dict) and vector_memory.get('search_enabled') is False:
            return False

        reference = self._reference_info(metadata)
        return reference.get('placement_status') == 'normal'

    def _record_to_result(self, record: Dict[str, Any], similarity: float) -> Dict[str, Any]:
        metadata_path = str(record.get('metadata_path', ''))
        metadata = read_json(metadata_path, default={})
        if not isinstance(metadata, dict):
            metadata = {}

        official = metadata.get('official_judgement', {})
        if not isinstance(official, dict):
            official = {}

        vlm_analysis = metadata.get('vlm_analysis', {})
        if not isinstance(vlm_analysis, dict):
            vlm_analysis = {}

        reference = self._reference_info(metadata)

        raw_paths = record.get('raw_image_paths', [])
        if not isinstance(raw_paths, list):
            raw_paths = []
        valid_raw_paths = sort_pose_paths(
            [path for path in raw_paths if path and os.path.exists(path)]
        )

        representative_path = None
        if valid_raw_paths:
            representative_path = valid_raw_paths[len(valid_raw_paths) // 2]

        return {
            'case_id': record.get('case_id'),
            'similarity': float(similarity),
            'metadata_path': metadata_path,
            'representative_image_path': representative_path,
            'raw_image_paths': valid_raw_paths,
            'official_result': official.get('result', 'unknown'),
            'reference_status': reference.get('placement_status', 'unknown'),
            'reference_visual_status': reference.get('visual_status', 'unknown'),
            'reference_source': reference.get('source', 'unknown'),
            'layout_status': reference.get('visual_status', vlm_analysis.get('layout_status', 'unknown')),
            'description': reference.get('description') or vlm_analysis.get('description', ''),
            'report_text': vlm_analysis.get('report_text', ''),
            'review_status': metadata.get('review_status', 'unreviewed'),
            '_search_enabled': self._metadata_allows_search(metadata),
        }

    def backfill_from_metadata_dir(
        self,
        metadata_dir: str,
        require_three_raw_images: bool = True,
    ) -> Dict[str, Any]:
        """기존 정상 metadata와 raw 이미지를 찾아 VectorDB에 누락 사례를 추가합니다."""
        root = Path(metadata_dir)
        if not root.is_dir():
            return {'scanned': 0, 'added': 0, 'skipped': 0, 'errors': []}

        scanned = 0
        added = 0
        skipped = 0
        errors: List[str] = []

        for metadata_path_obj in sorted(root.glob('recheck_*_metadata.json')):
            scanned += 1
            metadata_path = str(metadata_path_obj)
            metadata = read_json(metadata_path, default={})
            if not isinstance(metadata, dict):
                skipped += 1
                continue
            if not self._metadata_allows_search(metadata):
                skipped += 1
                continue

            case_id = str(metadata.get('case_id', '')).strip()
            if not case_id or self.contains_case(case_id):
                skipped += 1
                continue

            images = metadata.get('images', {})
            if not isinstance(images, dict):
                skipped += 1
                continue

            def resolve_list(values):
                resolved = []
                if not isinstance(values, list):
                    return resolved
                for value in values:
                    value = str(value).strip()
                    if not value:
                        continue
                    path = value if os.path.isabs(value) else str(root / value)
                    if os.path.exists(path):
                        resolved.append(os.path.abspath(path))
                return sort_pose_paths(resolved)

            raw_paths = resolve_list(images.get('raw', []))
            annotated_paths = resolve_list(images.get('annotated', []))
            required = 3 if require_three_raw_images else 1
            if len(raw_paths) < required:
                skipped += 1
                continue

            try:
                embedding = self.encode_case(raw_paths)
                result = self.add_case(
                    case_id=case_id,
                    embedding=embedding,
                    metadata_path=metadata_path,
                    raw_image_paths=raw_paths,
                    annotated_image_paths=annotated_paths,
                )
                vector_memory = metadata.setdefault('vector_memory', {})
                if not isinstance(vector_memory, dict):
                    vector_memory = {}
                    metadata['vector_memory'] = vector_memory
                vector_memory.update({
                    'search_enabled': True,
                    'added_to_index': True,
                    'memory_error': None,
                    'index_status': result.get('status', 'added'),
                    'index_size': result.get('index_size'),
                    'backend': self.backend_name,
                    'store_dir': self.root_dir,
                })
                write_json_atomic(metadata_path, metadata)
                added += 1
            except Exception as exc:
                errors.append(f'{case_id}: {exc}')

        return {
            'scanned': scanned,
            'added': added,
            'skipped': skipped,
            'errors': errors,
            'index_size': len(self._active_records),
        }

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        min_similarity: float = -1.0,
        exclude_case_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """정규화된 현재 사례와 가장 비슷한 과거 사례를 반환합니다."""
        query = normalize_vector(query_embedding)

        with self._lock:
            if self._matrix is None or not self._active_records or top_k <= 0:
                return []

            candidate_count = len(self._active_records)

            if self._index is not None:
                scores, indices = self._index.search(
                    query.reshape(1, -1).astype(np.float32),
                    candidate_count,
                )
                score_list = scores[0].tolist()
                index_list = indices[0].tolist()
            else:
                scores = self._matrix @ query
                indices = np.argsort(-scores)
                score_list = scores[indices].tolist()
                index_list = indices.tolist()

            results = []
            for score, record_index in zip(score_list, index_list):
                if record_index < 0 or record_index >= len(self._active_records):
                    continue

                record = self._active_records[record_index]
                if exclude_case_id and record.get('case_id') == exclude_case_id:
                    continue
                if float(score) < float(min_similarity):
                    continue

                result = self._record_to_result(record, float(score))
                if not result.pop('_search_enabled', True):
                    continue

                results.append(result)
                if len(results) >= top_k:
                    break

            return results

    def stats(self) -> Dict[str, Any]:
        return {
            'root_dir': self.root_dir,
            'model_name': self.model_name,
            'device': self._device or self.device_requested,
            'backend': self.backend_name,
            'catalog_size': len(self._catalog),
            'index_size': len(self._active_records),
            'faiss_available': faiss is not None,
        }
