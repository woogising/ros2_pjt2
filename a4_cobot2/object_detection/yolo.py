# ============================================================
# object_detection/yolo.py
# 역할:
#   - YOLO-seg 모델을 로드합니다.
#   - 여러 RGB frame의 bbox/mask 결과를 실제 물체 단위로 집계합니다.
#   - 외부에는 class/bbox/confidence/mask가 포함된 detection dict를 반환합니다.
#
# 이번 수정의 핵심:
#   1. 각 YOLO detection에 frame_index를 기록합니다.
#   2. 같은 클래스 + 비슷한 위치(IoU)의 detection을 하나의 track으로 묶습니다.
#   3. 같은 frame의 동일 클래스 bbox 두 개는 같은 track에 들어가지 못하게 합니다.
#      따라서 같은 클래스 물체가 실제로 여러 개 있어도 서로 분리될 수 있습니다.
#   4. 여러 frame에서 일정 횟수 이상 반복 감지된 track만 남깁니다.
#   5. 최종 detection에 yolo_frame_support / yolo_frame_ratio를 포함합니다.
#   6. RT-DETR과 같은 형식의 집계 요약 로그를 출력합니다.
#
# 출력 로그 예:
#   [YoloModel] frames=31, raw=62, aggregated=2,
#   min_support=4, classes={'pocari': 1, 'hammer': 1}
#
# mask 처리:
#   - bbox는 track 전체의 confidence 가중 평균을 사용합니다.
#   - mask는 track 안에서 confidence가 가장 높은 대표 frame의 mask를 사용합니다.
#   - 이후 EnsembleDetector에서 SAM2가 성공하면 SAM2 mask로 교체됩니다.
# ============================================================
import json
import math
import os
import time
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

YOLO_MODEL_FILENAME = "yolo_seg_best_v4.pt"
YOLO_CLASS_NAME_JSON = "class_name_tool.json"

YOLO_MODEL_PATH = os.path.join(
    PACKAGE_PATH,
    "resource",
    YOLO_MODEL_FILENAME,
)
YOLO_JSON_PATH = os.path.join(
    PACKAGE_PATH,
    "resource",
    YOLO_CLASS_NAME_JSON,
)


class YoloModel:
    # YOLO 모델과 프레임 간 집계 기준을 준비합니다.
    def __init__(
        self,
        aggregation_iou_threshold: float = 0.50,
        min_frame_ratio: float = 0.10,
        min_frame_support: int = 2,
        confidence_threshold: float = 0.60,
    ):
        self.model = YOLO(YOLO_MODEL_PATH)

        with open(
            YOLO_JSON_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            class_dict = json.load(file)
            self.class_id_to_name = {
                int(key): value
                for key, value in class_dict.items()
            }
            self.reversed_class_dict = {
                value: int(key)
                for key, value in class_dict.items()
            }

        # 같은 실제 물체로 묶기 위한 bbox IoU 기준입니다.
        self.aggregation_iou_threshold = float(
            aggregation_iou_threshold
        )

        # 전체 frame 중 최소 몇 비율에서 감지되어야 남길지 정합니다.
        # 31 frame에서 0.10이면 ceil(3.1)=4 frame 이상입니다.
        self.min_frame_ratio = float(min_frame_ratio)

        # frame 수가 적어도 최소 이 횟수 이상 반복 감지되어야 합니다.
        self.min_frame_support = max(
            1,
            int(min_frame_support),
        )

        # YOLO raw detection을 집계에 넣기 위한 confidence 기준입니다.
        # 0.8은 실제 있는 물체를 놓칠 수 있어 0.6으로 낮췄습니다.
        # 순간 오검출은 min_frame_support 기반 반복 감지 필터로 제거합니다.
        self.confidence_threshold = float(
            confidence_threshold
        )

        # 최근 집계 상태를 디버깅 또는 외부 확인에 사용할 수 있습니다.
        self.last_aggregation_stats = {
            "frame_count": 0,
            "raw_count": 0,
            "aggregated_count": 0,
            "required_support": 0,
        }

    # 일정 시간 동안 timestamp가 서로 다른 RGB frame만 수집합니다.
    def get_frames(
        self,
        img_node,
        duration=1.0,
    ):
        end_time = time.time() + duration
        frames = {}

        while time.time() < end_time:
            img_node.spin_once(timeout_sec=0.01)
            frame = img_node.get_color_frame()
            stamp = img_node.get_color_frame_stamp()

            if frame is not None:
                frames[stamp] = frame

            time.sleep(0.01)

        if not frames:
            print(
                f"No frames captured in {duration:.2f} seconds",
                flush=True,
            )

        print(
            f"{len(frames)} frames captured",
            flush=True,
        )

        return list(frames.values())

    # 특정 target 하나를 찾는 기존 호환/디버깅 함수입니다.
    def get_best_detection(
        self,
        img_node,
        target,
    ):
        img_node.spin_once(timeout_sec=0.01)
        frames = self.get_frames(img_node)

        if not frames:
            return None, None

        if target not in self.reversed_class_dict:
            print(
                f"Unknown target class: {target}",
                flush=True,
            )
            return None, None

        results = self.model(
            frames,
            verbose=False,
            retina_masks=True,
        )
        detections = self._aggregate_detections(
            results
        )

        label_id = self.reversed_class_dict[target]
        matches = [
            detection
            for detection in detections
            if int(detection["label"]) == label_id
        ]

        if not matches:
            print(
                "No matches found for the target label.",
                flush=True,
            )
            return None, None

        best_detection = max(
            matches,
            key=lambda item: float(item["score"]),
        )

        return (
            best_detection["box"],
            best_detection["score"],
        )

    # target_names를 class_id set으로 바꿉니다.
    def _target_ids_from_names(
        self,
        target_names: Optional[Iterable[str]] = None,
    ) -> Optional[Set[int]]:
        if not target_names:
            return None

        target_ids: Set[int] = set()

        for name in target_names:
            if name in self.reversed_class_dict:
                target_ids.add(
                    self.reversed_class_dict[name]
                )
            else:
                print(
                    f"[YoloModel] unknown target class ignored: {name}",
                    flush=True,
                )

        return target_ids

    # 이미 캡처된 여러 frame에서 YOLO-seg 추론과 프레임 간 집계를 수행합니다.
    def get_all_detections_from_frames(
        self,
        frames: Sequence,
        target_names: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        if not frames:
            print(
                "[YoloModel] no frames available for detection",
                flush=True,
            )
            return []

        target_ids = self._target_ids_from_names(
            target_names
        )

        results = self.model(
            frames,
            verbose=False,
            retina_masks=True,
        )

        detections = self._aggregate_detections(
            results=results,
            target_ids=target_ids,
        )

        converted: List[dict] = []

        for detection in detections:
            class_id = int(detection["label"])
            name = self.class_id_to_name.get(
                class_id,
                f"unknown_{class_id}",
            )

            converted.append({
                "name": name,
                "class_id": class_id,
                "box": detection["box"],
                "confidence": float(
                    detection["score"]
                ),
                "mask": detection.get("mask"),
                "source": "yolo",
                "yolo_confidence": float(
                    detection["score"]
                ),

                # YOLO가 전체 frame 중 몇 frame에서 이 물체를 확인했는지 보존합니다.
                "yolo_frame_support": detection.get(
                    "frame_support"
                ),
                "yolo_frame_ratio": detection.get(
                    "frame_ratio"
                ),
                "yolo_raw_group_count": detection.get(
                    "raw_group_count"
                ),
                "yolo_representative_frame_index": (
                    detection.get(
                        "representative_frame_index"
                    )
                ),
            })

        converted.sort(
            key=lambda item: float(
                item["confidence"]
            ),
            reverse=True,
        )

        class_counts = Counter(
            item["name"]
            for item in converted
        )
        stats = self.last_aggregation_stats

        print(
            f"[YoloModel] "
            f"frames={stats['frame_count']}, "
            f"raw={stats['raw_count']}, "
            f"aggregated={stats['aggregated_count']}, "
            f"min_support={stats['required_support']}, "
            f"classes={dict(class_counts)}",
            flush=True,
        )

        return converted

    # ImgNode에서 frame을 직접 캡처하는 기존 호환 함수입니다.
    def get_all_detections(
        self,
        img_node,
        target_names=None,
    ):
        img_node.spin_once(timeout_sec=0.01)
        frames = self.get_frames(img_node)

        return self.get_all_detections_from_frames(
            frames,
            target_names=target_names,
        )

    # 기존 코드 호환을 위해 유지하는 이미지 변환 함수입니다.
    @staticmethod
    def PCI_MAP(
        in_img,
        ratio,
    ):
        in_img = in_img.astype(np.float32)

        if len(in_img.shape) != 3:
            return in_img

        vertical, horizontal, depth = in_img.shape
        sample_count = round(
            ratio * vertical * horizontal
        )
        working = np.copy(in_img)

        for channel in range(depth):
            channel_sum = np.sum(
                working[:, :, channel]
            )
            if channel_sum > 0:
                working[:, :, channel] /= channel_sum

        scaled = sample_count * working
        means = []
        variances = []

        for channel in range(depth):
            means.append(
                np.mean(scaled[:, :, channel])
            )
            variances.append(
                np.var(scaled[:, :, channel])
            )

        means = np.asarray(means)
        variances = np.asarray(variances)

        alpha = np.where(
            variances != 0,
            means ** 2 / variances,
            0,
        )
        beta = np.where(
            variances != 0,
            means / variances,
            0,
        )

        poisson = np.zeros_like(scaled)

        for channel in range(depth):
            poisson[:, :, channel] = np.random.poisson(
                scaled[:, :, channel]
            )

        mapped = np.zeros_like(scaled)

        for channel in range(depth):
            mask = scaled[:, :, channel] > 0
            mapped[:, :, channel] = (
                poisson[:, :, channel]
                + alpha[channel] * mask
            ) / (
                1 + beta[channel] * mask
            )

        mapped = mapped / max(1, sample_count)

        if np.max(mapped) > 0:
            mapped = mapped / np.max(mapped)

        return (mapped * 255).astype(np.uint8)

    # 여러 frame의 raw detection을 실제 물체 단위 track으로 집계합니다.
    def _aggregate_detections(
        self,
        results,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        target_ids: Optional[Set[int]] = None,
    ) -> List[dict]:
        confidence_threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        iou_threshold = (
            self.aggregation_iou_threshold
            if iou_threshold is None
            else float(iou_threshold)
        )

        results = list(results)
        frame_count = len(results)
        raw: List[dict] = []

        # frame_index를 함께 저장해야 실제 고유 frame 감지 횟수를 계산할 수 있습니다.
        for frame_index, result in enumerate(results):
            if result.boxes is None:
                continue

            masks = (
                result.masks.data.cpu().numpy()
                if result.masks is not None
                else None
            )

            for detection_index, (
                box,
                score,
                label,
            ) in enumerate(
                zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                )
            ):
                class_id = int(label)
                confidence = float(score)

                if confidence < confidence_threshold:
                    continue

                if (
                    target_ids is not None
                    and class_id not in target_ids
                ):
                    continue

                mask = (
                    masks[detection_index]
                    if (
                        masks is not None
                        and detection_index < len(masks)
                    )
                    else None
                )

                raw.append({
                    "box": [
                        float(value)
                        for value in box
                    ],
                    "score": confidence,
                    "label": class_id,
                    "mask": mask,
                    "frame_index": int(
                        frame_index
                    ),
                })

        tracks: List[dict] = []

        # frame 순서로 처리하고 같은 frame 안에서는 confidence가 높은 후보를 우선 배치합니다.
        ordered = sorted(
            raw,
            key=lambda item: (
                int(item["frame_index"]),
                -float(item["score"]),
            ),
        )

        for detection in ordered:
            frame_index = int(
                detection["frame_index"]
            )
            class_id = int(
                detection["label"]
            )

            best_track_index = None
            best_iou = 0.0

            for track_index, track in enumerate(
                tracks
            ):
                if int(track["label"]) != class_id:
                    continue

                # 같은 frame에서 나온 두 bbox를 같은 실제 물체 track으로 합치지 않습니다.
                # 이 조건 덕분에 동일 클래스 물체 여러 개를 서로 다른 track으로 유지할 수 있습니다.
                if frame_index in track["frame_indices"]:
                    continue

                iou = self._iou(
                    detection["box"],
                    track["reference_box"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_index = (
                        track_index
                    )

            if (
                best_track_index is not None
                and best_iou >= iou_threshold
            ):
                track = tracks[
                    best_track_index
                ]
                track["detections"].append(
                    detection
                )
                track["frame_indices"].add(
                    frame_index
                )
                track["reference_box"] = (
                    self._confidence_weighted_box(
                        track["detections"]
                    )
                )

            else:
                tracks.append({
                    "label": class_id,
                    "detections": [
                        detection
                    ],
                    "frame_indices": {
                        frame_index
                    },
                    "reference_box": list(
                        detection["box"]
                    ),
                })

        required_support = (
            self._required_frame_support(
                frame_count
            )
        )
        final: List[dict] = []

        for track in tracks:
            group = track["detections"]
            frame_support = len(
                track["frame_indices"]
            )

            # 한두 frame에서만 나타난 순간 오검출은 제거합니다.
            if frame_support < required_support:
                continue

            boxes = np.asarray(
                [
                    item["box"]
                    for item in group
                ],
                dtype=np.float32,
            )
            scores = np.asarray(
                [
                    item["score"]
                    for item in group
                ],
                dtype=np.float32,
            )

            best = max(
                group,
                key=lambda item: float(
                    item["score"]
                ),
            )

            if float(scores.sum()) > 0.0:
                final_box = np.average(
                    boxes,
                    axis=0,
                    weights=scores,
                )
            else:
                final_box = boxes.mean(
                    axis=0
                )

            final.append({
                "box": final_box.astype(
                    float
                ).tolist(),
                "score": float(
                    scores.mean()
                ),
                "label": int(
                    best["label"]
                ),

                # bbox는 여러 frame 평균이지만 mask는 가장 신뢰도가 높은 대표 frame을 사용합니다.
                "mask": best.get("mask"),

                # 프레임 간 안정성을 확인하기 위한 메타데이터입니다.
                "frame_support": int(
                    frame_support
                ),
                "frame_ratio": float(
                    frame_support
                    / max(1, frame_count)
                ),
                "raw_group_count": int(
                    len(group)
                ),
                "representative_frame_index": int(
                    best["frame_index"]
                ),
            })

        final.sort(
            key=lambda item: float(
                item["score"]
            ),
            reverse=True,
        )

        self.last_aggregation_stats = {
            "frame_count": int(
                frame_count
            ),
            "raw_count": int(
                len(raw)
            ),
            "aggregated_count": int(
                len(final)
            ),
            "required_support": int(
                required_support
            ),
        }

        return final

    # 현재 frame 수에 필요한 최소 고유 frame 감지 횟수를 계산합니다.
    def _required_frame_support(
        self,
        frame_count: int,
    ) -> int:
        ratio_support = int(
            math.ceil(
                max(0, frame_count)
                * self.min_frame_ratio
            )
        )

        required = max(
            self.min_frame_support,
            ratio_support,
        )

        if frame_count > 0:
            required = min(
                required,
                frame_count,
            )

        return max(
            1,
            required,
        )

    # track 대표 bbox를 confidence 가중 평균으로 계산합니다.
    @staticmethod
    def _confidence_weighted_box(
        detections: List[dict],
    ) -> List[float]:
        boxes = np.asarray(
            [
                item["box"]
                for item in detections
            ],
            dtype=np.float32,
        )
        scores = np.asarray(
            [
                item["score"]
                for item in detections
            ],
            dtype=np.float32,
        )

        if float(scores.sum()) > 0.0:
            box = np.average(
                boxes,
                axis=0,
                weights=scores,
            )
        else:
            box = boxes.mean(
                axis=0
            )

        return box.astype(
            float
        ).tolist()

    # bbox [x1, y1, x2, y2] 두 개의 IoU를 계산합니다.
    @staticmethod
    def _iou(
        box1,
        box2,
    ) -> float:
        x1 = max(
            float(box1[0]),
            float(box2[0]),
        )
        y1 = max(
            float(box1[1]),
            float(box2[1]),
        )
        x2 = min(
            float(box1[2]),
            float(box2[2]),
        )
        y2 = min(
            float(box1[3]),
            float(box2[3]),
        )

        intersection = (
            max(0.0, x2 - x1)
            * max(0.0, y2 - y1)
        )

        area1 = (
            max(
                0.0,
                float(box1[2])
                - float(box1[0]),
            )
            * max(
                0.0,
                float(box1[3])
                - float(box1[1]),
            )
        )
        area2 = (
            max(
                0.0,
                float(box2[2])
                - float(box2[0]),
            )
            * max(
                0.0,
                float(box2[3])
                - float(box2[1]),
            )
        )

        union = (
            area1
            + area2
            - intersection
        )

        return (
            intersection / union
            if union > 0.0
            else 0.0
        )
