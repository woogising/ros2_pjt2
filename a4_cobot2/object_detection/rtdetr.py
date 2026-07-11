# ============================================================
# object_detection/rtdetr.py
# 역할:
#   - 파인튜닝한 RT-DETR 모델을 로드합니다.
#   - 여러 프레임의 RT-DETR bbox를 그대로 반환하지 않고,
#     같은 클래스 + 비슷한 위치(IoU)의 결과를 프레임 간 집계합니다.
#   - 최종적으로 YOLO와 같은 detection dict 형식의 안정된 후보만 반환합니다.
#
# 이번 수정의 핵심:
#   1. 각 detection에 frame_index를 기록합니다.
#   2. 같은 프레임의 동일 클래스 bbox는 같은 그룹에 두 번 들어가지 못하게 합니다.
#      따라서 같은 클래스 물체가 실제로 두 개 있을 때 서로 다른 track으로 유지할 수 있습니다.
#   3. 여러 프레임에서 반복 감지된 bbox만 남깁니다.
#      순간적인 1프레임 오검출은 제거됩니다.
#   4. 그룹별 confidence 가중 평균 bbox와 평균 confidence를 반환합니다.
#
# 사용 위치:
#   - ensemble_detector.py에서 YOLO-seg 결과와 RT-DETR 결과를 bbox/class 레벨로 병합합니다.
#
# 주의:
#   - RT-DETR은 mask를 만들지 않습니다.
#   - 최종 3D grasp 계산에 쓰려면 SAM2.1 또는 YOLO mask가 필요합니다.
# ============================================================
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from ament_index_python.packages import get_package_share_directory

try:
    # Ultralytics 버전에 따라 RTDETR class가 있을 수 있습니다.
    from ultralytics import RTDETR as _RTDETRModel
except Exception:  # pragma: no cover - 환경별 fallback
    _RTDETRModel = None

from ultralytics import YOLO


PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

RTDETR_MODEL_FILENAME = os.getenv("A4_COBOT2_RTDETR_MODEL", "rtdetr_best.pt")
RTDETR_CLASS_NAME_JSON = os.getenv(
    "A4_COBOT2_RTDETR_CLASS_JSON",
    "class_name_tool.json",
)

RTDETR_MODEL_PATH = os.path.join(
    PACKAGE_PATH,
    "resource",
    RTDETR_MODEL_FILENAME,
)
RTDETR_JSON_PATH = os.path.join(
    PACKAGE_PATH,
    "resource",
    RTDETR_CLASS_NAME_JSON,
)


class RTDETRModel:
    # RT-DETR weight, class mapping, 프레임 간 집계 기준을 준비합니다.
    def __init__(
        self,
        model_path: Optional[str] = None,
        class_json_path: Optional[str] = None,
        aggregation_iou_threshold: float = 0.50,
        min_frame_ratio: float = 0.10,
        min_frame_support: int = 2,
    ):
        self.model_path = model_path or RTDETR_MODEL_PATH
        self.class_json_path = class_json_path or RTDETR_JSON_PATH
        self.enabled = False
        self.model = None

        # 같은 실제 물체로 묶기 위한 bbox IoU 기준입니다.
        self.aggregation_iou_threshold = float(aggregation_iou_threshold)

        # 전체 frame 중 최소 몇 비율에서 감지되어야 최종 후보로 남길지 정합니다.
        # 예: 31 frame이면 10% 기준으로 최소 4 frame에서 감지되어야 합니다.
        self.min_frame_ratio = float(min_frame_ratio)

        # frame 수가 적을 때도 최소 이 횟수 이상 감지되어야 합니다.
        self.min_frame_support = max(1, int(min_frame_support))

        self.class_id_to_name: Dict[int, str] = {}
        self.reversed_class_dict: Dict[str, int] = {}

        self._load_class_names()
        self._load_model()

    # class_name_tool.json을 읽어 class_id <-> class_name 매핑을 만듭니다.
    def _load_class_names(self):
        with open(self.class_json_path, "r", encoding="utf-8") as file:
            class_dict = json.load(file)
            self.class_id_to_name = {
                int(key): value
                for key, value in class_dict.items()
            }
            self.reversed_class_dict = {
                value: int(key)
                for key, value in class_dict.items()
            }

    # Ultralytics RT-DETR 모델을 로드합니다.
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(
                f"[RTDETRModel] weight not found. "
                f"RT-DETR disabled: {self.model_path}",
                flush=True,
            )
            self.enabled = False
            return

        try:
            if _RTDETRModel is not None:
                self.model = _RTDETRModel(self.model_path)
            else:
                # 일부 Ultralytics 버전은 YOLO(...)가 RT-DETR .pt도 자동 로드합니다.
                self.model = YOLO(self.model_path)

            self.enabled = True
            print(
                f"[RTDETRModel] loaded: {self.model_path}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[RTDETRModel] load failed. "
                f"RT-DETR disabled: {exc}",
                flush=True,
            )
            self.enabled = False
            self.model = None

    # target_names를 class_id set으로 바꿉니다.
    def _target_ids_from_names(
        self,
        target_names: Optional[Iterable[str]],
    ):
        if not target_names:
            return None

        target_ids = set()

        for name in target_names:
            if name in self.reversed_class_dict:
                target_ids.add(self.reversed_class_dict[name])
            else:
                print(
                    f"[RTDETRModel] unknown target class ignored: {name}",
                    flush=True,
                )

        return target_ids

    # 여러 frame에 RT-DETR inference를 수행한 뒤 프레임 간 중복을 집계합니다.
    def get_all_detections_from_frames(
        self,
        frames: Sequence,
        target_names: Optional[Iterable[str]] = None,
        confidence_threshold: float = 0.50,
    ) -> List[dict]:
        if not self.enabled or self.model is None:
            return []

        if not frames:
            return []

        target_ids = self._target_ids_from_names(target_names)

        try:
            results = self.model(frames, verbose=False)

        except Exception as exc:
            print(
                f"[RTDETRModel] inference failed: {exc}",
                flush=True,
            )
            return []

        raw: List[dict] = []

        # frame_index를 반드시 기록해야 고유 frame 감지 횟수를 계산할 수 있습니다.
        for frame_index, result in enumerate(results):
            if result.boxes is None:
                continue

            for box, score, label in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            ):
                class_id = int(label)
                confidence = float(score)

                if confidence < confidence_threshold:
                    continue

                if target_ids is not None and class_id not in target_ids:
                    continue

                raw.append({
                    "name": self.class_id_to_name.get(
                        class_id,
                        f"unknown_{class_id}",
                    ),
                    "class_id": class_id,
                    "box": [float(value) for value in box],
                    "confidence": confidence,
                    "mask": None,
                    "source": "rtdetr",
                    "frame_index": int(frame_index),
                })

        aggregated = self._aggregate_across_frames(
            raw=raw,
            frame_count=len(frames),
        )

        # 디버깅에 필요한 핵심 정보만 한 줄로 출력합니다.
        class_counts = Counter(
            detection["name"]
            for detection in aggregated
        )
        required_support = self._required_frame_support(len(frames))

        print(
            f"[RTDETRModel] frames={len(frames)}, "
            f"raw={len(raw)}, "
            f"aggregated={len(aggregated)}, "
            f"min_support={required_support}, "
            f"classes={dict(class_counts)}",
            flush=True,
        )

        return aggregated

    # 같은 클래스이면서 위치가 유사한 bbox를 실제 물체 단위 track으로 묶습니다.
    def _aggregate_across_frames(
        self,
        raw: List[dict],
        frame_count: int,
    ) -> List[dict]:
        if not raw:
            return []

        tracks: List[dict] = []

        # frame 순서로 처리하고 같은 frame 안에서는 confidence가 높은 후보를 먼저 처리합니다.
        ordered = sorted(
            raw,
            key=lambda item: (
                int(item["frame_index"]),
                -float(item["confidence"]),
            ),
        )

        for detection in ordered:
            frame_index = int(detection["frame_index"])
            class_id = int(detection["class_id"])

            best_track_index = None
            best_iou = 0.0

            for track_index, track in enumerate(tracks):
                if int(track["class_id"]) != class_id:
                    continue

                # 같은 frame의 두 bbox가 같은 track에 들어가지 않도록 합니다.
                # 이 조건이 있어야 같은 클래스 물체가 여러 개일 때 분리 상태를 유지할 수 있습니다.
                if frame_index in track["frame_indices"]:
                    continue

                iou = self._iou(
                    detection["box"],
                    track["reference_box"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_index = track_index

            if (
                best_track_index is not None
                and best_iou >= self.aggregation_iou_threshold
            ):
                track = tracks[best_track_index]
                track["detections"].append(detection)
                track["frame_indices"].add(frame_index)
                track["reference_box"] = self._confidence_weighted_box(
                    track["detections"]
                )

            else:
                tracks.append({
                    "class_id": class_id,
                    "detections": [detection],
                    "frame_indices": {frame_index},
                    "reference_box": list(detection["box"]),
                })

        required_support = self._required_frame_support(frame_count)
        aggregated: List[dict] = []

        for track in tracks:
            detections = track["detections"]
            frame_support = len(track["frame_indices"])

            # 여러 frame에서 반복 확인되지 않은 순간 오검출은 제거합니다.
            if frame_support < required_support:
                continue

            boxes = np.asarray(
                [item["box"] for item in detections],
                dtype=np.float32,
            )
            scores = np.asarray(
                [item["confidence"] for item in detections],
                dtype=np.float32,
            )

            best = max(
                detections,
                key=lambda item: float(item["confidence"]),
            )

            # bbox는 confidence가 높은 frame에 조금 더 큰 비중을 둔 평균을 사용합니다.
            weight_sum = float(scores.sum())
            if weight_sum > 0.0:
                final_box = np.average(
                    boxes,
                    axis=0,
                    weights=scores,
                )
            else:
                final_box = boxes.mean(axis=0)

            aggregated.append({
                "name": best["name"],
                "class_id": int(best["class_id"]),
                "box": final_box.astype(float).tolist(),
                "confidence": float(scores.mean()),
                "mask": None,
                "source": "rtdetr",

                # 아래 값은 디버깅 및 앙상블 상태 확인용 메타데이터입니다.
                "frame_support": int(frame_support),
                "frame_ratio": float(
                    frame_support / max(1, frame_count)
                ),
                "raw_group_count": int(len(detections)),
                "representative_frame_index": int(
                    best["frame_index"]
                ),
            })

        aggregated.sort(
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )

        return aggregated

    # 현재 frame 수에 필요한 최소 고유 frame 감지 횟수를 계산합니다.
    def _required_frame_support(self, frame_count: int) -> int:
        ratio_support = int(
            math.ceil(
                max(0, frame_count) * self.min_frame_ratio
            )
        )

        required = max(
            self.min_frame_support,
            ratio_support,
        )

        if frame_count > 0:
            required = min(required, frame_count)

        return max(1, required)

    # track의 현재 대표 bbox를 confidence 가중 평균으로 계산합니다.
    def _confidence_weighted_box(
        self,
        detections: List[dict],
    ) -> List[float]:
        boxes = np.asarray(
            [item["box"] for item in detections],
            dtype=np.float32,
        )
        scores = np.asarray(
            [item["confidence"] for item in detections],
            dtype=np.float32,
        )

        if float(scores.sum()) > 0.0:
            box = np.average(
                boxes,
                axis=0,
                weights=scores,
            )
        else:
            box = boxes.mean(axis=0)

        return box.astype(float).tolist()

    # bbox [x1, y1, x2, y2] 두 개의 IoU를 계산합니다.
    def _iou(
        self,
        box1,
        box2,
    ) -> float:
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        intersection = (
            max(0.0, x2 - x1)
            * max(0.0, y2 - y1)
        )

        area1 = (
            max(0.0, float(box1[2]) - float(box1[0]))
            * max(0.0, float(box1[3]) - float(box1[1]))
        )
        area2 = (
            max(0.0, float(box2[2]) - float(box2[0]))
            * max(0.0, float(box2[3]) - float(box2[1]))
        )

        union = area1 + area2 - intersection
        return intersection / union if union > 0.0 else 0.0
