# ============================================================
# object_detection/ensemble_detector.py
# 역할:
#   - YOLO-seg, RT-DETR, SAM2.1을 object_detection 내부에서 결합합니다.
#   - 외부에는 기존 YoloModel.get_all_detections()와 같은 detection dict 형식으로 반환합니다.
#
# 설계:
#   - YOLO-seg: 기본 class/bbox/mask 후보
#   - RT-DETR: 프레임 간 집계된 bbox/class 후보로 검증 및 누락 보완
#   - SAM2.1: 최종 bbox prompt 기반 mask refinement
#
# 이번 수정의 핵심:
#   1. RT-DETR 집계 결과의 frame_support 정보를 보존합니다.
#   2. 수십~수백 줄짜리 FINAL 상세 로그 대신 한 줄 요약 로그를 기본 사용합니다.
#   3. SAM2 실패로 제거된 detection도 개별 출력하지 않고 이유별 개수로 요약합니다.
#   4. detailed_debug=True일 때만 개별 detection 상세 로그를 출력합니다.
#
# 주의:
#   - ROS2 topic/service/action은 변경하지 않습니다.
#   - detection.py는 self.model.get_all_detections(img_node)만 호출하면 됩니다.
# ============================================================
from collections import Counter
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .yolo import YoloModel
from .rtdetr import RTDETRModel
from .sam2_refiner import SAM2Refiner


class EnsembleDetector:
    # YOLO, RT-DETR, SAM2.1 wrapper를 초기화합니다.
    def __init__(
        self,
        frame_capture_sec: float = 1.0,
        fuse_iou_threshold: float = 0.50,
        rtdetr_only_conf_threshold: float = 0.80,
        yolo_weight: float = 0.60,
        rtdetr_weight: float = 0.40,
        detailed_debug: bool = False,
    ):
        self.yolo = YoloModel()
        self.rtdetr = RTDETRModel()
        self.sam2 = SAM2Refiner()

        # detection.py의 preview는 기존처럼 YOLO 원본 모델을 사용합니다.
        self.model = self.yolo.model

        self.frame_capture_sec = frame_capture_sec
        self.fuse_iou_threshold = fuse_iou_threshold
        self.rtdetr_only_conf_threshold = rtdetr_only_conf_threshold
        self.yolo_weight = yolo_weight
        self.rtdetr_weight = rtdetr_weight

        # False가 기본값입니다.
        # 필요할 때만 True로 바꾸면 detection별 상세 로그를 다시 볼 수 있습니다.
        self.detailed_debug = bool(detailed_debug)

    # 현재 프레임들에서 YOLO+RT-DETR 후보를 만들고 SAM2 mask로 정제합니다.
    def get_all_detections(
        self,
        img_node,
        target_names: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        frames = self.yolo.get_frames(
            img_node,
            duration=self.frame_capture_sec,
        )

        if not frames:
            print(
                "[EnsembleDetector] no frames captured",
                flush=True,
            )
            return []

        yolo_dets = self._get_yolo_detections(
            frames,
            target_names=target_names,
        )

        rtdetr_dets = self.rtdetr.get_all_detections_from_frames(
            frames,
            target_names=target_names,
        )

        fused = self._fuse_yolo_rtdetr(
            yolo_dets,
            rtdetr_dets,
        )

        # 현재 구조에서는 마지막 frame에 fused bbox를 prompt로 넣습니다.
        # 카메라와 물체가 고정된 작업공간 스캔을 전제로 합니다.
        frame_for_sam = frames[-1]

        final_dets: List[dict] = []
        dropped_reasons = Counter()

        for detection in fused:
            detection = dict(detection)
            yolo_mask = detection.get("mask")

            sam_mask, sam_score = self.sam2.predict_mask_from_box(
                frame_for_sam,
                detection.get("box"),
            )

            # SAM2가 실패하더라도 점수와 실패 원인을 잃지 않고 보존합니다.
            detection["sam2_score"] = sam_score
            detection["sam2_reason"] = self.sam2.last_reason

            if sam_mask is not None:
                detection["mask"] = sam_mask
                detection["mask_source"] = "sam2"
                detection["sam2_status"] = "success"

            elif yolo_mask is not None:
                detection["mask"] = yolo_mask
                detection["mask_source"] = "yolo"
                detection["sam2_status"] = "fallback_yolo"

            else:
                # RT-DETR-only 후보는 자체 mask가 없으므로
                # SAM2까지 실패하면 3D cloud를 만들 수 없어 제거합니다.
                reason = str(
                    detection.get("sam2_reason") or "unknown"
                )
                dropped_reasons[reason] += 1
                continue

            final_dets.append(detection)

        final_dets.sort(
            key=lambda item: float(
                item.get("confidence", 0.0)
            ),
            reverse=True,
        )

        print(
            f"[EnsembleDetector] "
            f"yolo={len(yolo_dets)}, "
            f"rtdetr={len(rtdetr_dets)}, "
            f"fused={len(fused)}, "
            f"final={len(final_dets)}, "
            f"dropped={sum(dropped_reasons.values())}",
            flush=True,
        )

        self._print_detection_summary(
            stage="FINAL",
            detections=final_dets,
            dropped_reasons=dropped_reasons,
        )

        if self.detailed_debug:
            self._print_detection_debug(
                "FINAL",
                final_dets,
            )

        return final_dets

    # 기본 로그: detection별 출력 대신 클래스/출처/mask/SAM 상태를 한 줄로 요약합니다.
    def _print_detection_summary(
        self,
        stage: str,
        detections: List[dict],
        dropped_reasons: Optional[Counter] = None,
    ):
        class_counts = Counter(
            str(item.get("name", "unknown"))
            for item in detections
        )
        source_counts = Counter(
            str(item.get("source", "unknown"))
            for item in detections
        )
        mask_counts = Counter(
            str(item.get("mask_source", "none"))
            for item in detections
        )
        sam2_reason_counts = Counter(
            str(item.get("sam2_reason", "none"))
            for item in detections
        )

        # YOLO가 실제로 몇 frame에서 확인된 후보인지 표시합니다.
        yolo_supports = [
            (
                str(item.get("name", "unknown")),
                int(item.get("yolo_frame_support", 0) or 0),
            )
            for item in detections
            if item.get("yolo_frame_support") is not None
        ]

        # RT-DETR이 실제로 몇 frame에서 확인된 후보인지 표시합니다.
        rtdetr_supports = [
            (
                str(item.get("name", "unknown")),
                int(
                    item.get(
                        "rtdetr_frame_support",
                        item.get("frame_support", 0),
                    )
                    or 0
                ),
            )
            for item in detections
            if (
                item.get("rtdetr_frame_support") is not None
                or item.get("frame_support") is not None
            )
        ]

        print(
            f"[{stage}] "
            f"count={len(detections)}, "
            f"classes={dict(class_counts)}, "
            f"sources={dict(source_counts)}, "
            f"masks={dict(mask_counts)}, "
            f"sam2_reasons={dict(sam2_reason_counts)}, "
            f"yolo_supports={yolo_supports}, "
            f"rtdetr_supports={rtdetr_supports}, "
            f"dropped_reasons={dict(dropped_reasons or {})}",
            flush=True,
        )

    # 선택 로그: detailed_debug=True일 때만 detection별 상세 정보를 출력합니다.
    def _print_detection_debug(
        self,
        stage,
        detections,
    ):
        print(
            f"\n[{stage}] count={len(detections)}",
            flush=True,
        )

        for index, detection in enumerate(detections):
            mask = detection.get("mask")

            if mask is None:
                mask_shape = None
                mask_pixels = 0
            else:
                mask_array = np.asarray(mask)
                mask_shape = tuple(mask_array.shape)
                mask_pixels = int(
                    np.count_nonzero(mask_array > 0)
                )

            confidence = detection.get("confidence")
            confidence_text = (
                f"{float(confidence):.4f}"
                if confidence is not None
                else "None"
            )

            print(
                f"[{stage} {index}] "
                f"name={detection.get('name')}, "
                f"class_id={detection.get('class_id')}, "
                f"confidence={confidence_text}, "
                f"box={detection.get('box')}, "
                f"source={detection.get('source')}, "
                f"yolo_conf={detection.get('yolo_confidence')}, "
                f"rtdetr_conf={detection.get('rtdetr_confidence')}, "
                f"fusion_iou={detection.get('fusion_iou')}, "
                f"yolo_support={detection.get('yolo_frame_support')}, "
                f"rtdetr_support={detection.get('rtdetr_frame_support', detection.get('frame_support'))}, "
                f"mask_source={detection.get('mask_source')}, "
                f"sam2_status={detection.get('sam2_status')}, "
                f"sam2_reason={detection.get('sam2_reason')}, "
                f"sam2_score={detection.get('sam2_score')}, "
                f"mask_shape={mask_shape}, "
                f"mask_pixels={mask_pixels}",
                flush=True,
            )

    # YOLO wrapper가 frame list를 직접 받을 수 있으면 그 함수를 사용합니다.
    def _get_yolo_detections(
        self,
        frames: Sequence,
        target_names: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        if hasattr(
            self.yolo,
            "get_all_detections_from_frames",
        ):
            return self.yolo.get_all_detections_from_frames(
                frames,
                target_names=target_names,
            )

        # yolo.py가 아직 패치되지 않은 환경을 위한 fallback입니다.
        target_ids = None

        if target_names:
            target_ids = set()

            for name in target_names:
                if name in self.yolo.reversed_class_dict:
                    target_ids.add(
                        self.yolo.reversed_class_dict[name]
                    )

        results = self.yolo.model(
            frames,
            verbose=False,
            retina_masks=True,
        )
        detections = self.yolo._aggregate_detections(
            results
        )

        converted = []

        for detection in detections:
            class_id = int(detection["label"])

            if (
                target_ids is not None
                and class_id not in target_ids
            ):
                continue

            name = self.yolo.class_id_to_name.get(
                class_id,
                f"unknown_{class_id}",
            )

            converted.append({
                "name": name,
                "class_id": class_id,
                "box": detection["box"],
                "confidence": float(detection["score"]),
                "mask": detection.get("mask"),
                "source": "yolo",
                "yolo_confidence": float(
                    detection["score"]
                ),
                "yolo_frame_support": detection.get(
                    "frame_support"
                ),
                "yolo_frame_ratio": detection.get(
                    "frame_ratio"
                ),
                "yolo_representative_frame_index": (
                    detection.get(
                        "representative_frame_index"
                    )
                ),
            })

        converted.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return converted

    # YOLO와 집계된 RT-DETR bbox 후보를 같은 class + IoU 기준으로 병합합니다.
    def _fuse_yolo_rtdetr(
        self,
        yolo_dets: List[dict],
        rtdetr_dets: List[dict],
    ) -> List[dict]:
        fused = []
        used_rtdetr = set()

        for yolo_detection in yolo_dets:
            best_index = None
            best_iou = 0.0

            for index, rtdetr_detection in enumerate(
                rtdetr_dets
            ):
                if index in used_rtdetr:
                    continue

                if (
                    int(
                        yolo_detection.get(
                            "class_id",
                            -1,
                        )
                    )
                    != int(
                        rtdetr_detection.get(
                            "class_id",
                            -2,
                        )
                    )
                ):
                    continue

                iou = self._iou(
                    yolo_detection["box"],
                    rtdetr_detection["box"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if (
                best_index is not None
                and best_iou >= self.fuse_iou_threshold
            ):
                rtdetr_detection = rtdetr_dets[
                    best_index
                ]
                used_rtdetr.add(best_index)

                fused.append(
                    self._merge_pair(
                        yolo_detection,
                        rtdetr_detection,
                        best_iou,
                    )
                )

            else:
                output = dict(yolo_detection)
                output["source"] = "yolo"
                output["yolo_confidence"] = float(
                    yolo_detection.get(
                        "confidence",
                        0.0,
                    )
                )
                output["rtdetr_confidence"] = None
                output["rtdetr_frame_support"] = None
                output["rtdetr_frame_ratio"] = None
                fused.append(output)

        # YOLO가 놓쳤지만 RT-DETR이 안정적으로 반복 감지한 후보를 보완합니다.
        for index, rtdetr_detection in enumerate(
            rtdetr_dets
        ):
            if index in used_rtdetr:
                continue

            if (
                float(
                    rtdetr_detection.get(
                        "confidence",
                        0.0,
                    )
                )
                < self.rtdetr_only_conf_threshold
            ):
                continue

            output = dict(rtdetr_detection)
            output["source"] = "rtdetr_only"
            output["yolo_confidence"] = None
            output["rtdetr_confidence"] = float(
                rtdetr_detection.get(
                    "confidence",
                    0.0,
                )
            )

            # rtdetr.py가 만든 frame support 정보를 명시적인 키로도 보존합니다.
            output["rtdetr_frame_support"] = (
                rtdetr_detection.get("frame_support")
            )
            output["rtdetr_frame_ratio"] = (
                rtdetr_detection.get("frame_ratio")
            )

            fused.append(output)

        return fused

    # 매칭된 YOLO/RT-DETR pair를 confidence 가중 평균으로 합칩니다.
    def _merge_pair(
        self,
        yolo_detection: dict,
        rtdetr_detection: dict,
        iou: float,
    ) -> dict:
        yolo_confidence = float(
            yolo_detection.get(
                "confidence",
                0.0,
            )
        )
        rtdetr_confidence = float(
            rtdetr_detection.get(
                "confidence",
                0.0,
            )
        )

        box = self._weighted_box(
            yolo_detection["box"],
            rtdetr_detection["box"],
            y_weight=max(
                1e-6,
                self.yolo_weight * yolo_confidence,
            ),
            r_weight=max(
                1e-6,
                self.rtdetr_weight * rtdetr_confidence,
            ),
        )

        confidence = (
            self.yolo_weight * yolo_confidence
            + self.rtdetr_weight * rtdetr_confidence
        )

        return {
            "name": (
                yolo_detection.get("name")
                or rtdetr_detection.get("name")
            ),
            "class_id": int(
                yolo_detection.get(
                    "class_id",
                    rtdetr_detection.get("class_id"),
                )
            ),
            "box": box,
            "confidence": float(confidence),

            # SAM2가 실패하면 기존 YOLO mask로 fallback합니다.
            "mask": yolo_detection.get("mask"),

            "source": "yolo+rtdetr",
            "yolo_confidence": yolo_confidence,
            "rtdetr_confidence": rtdetr_confidence,
            "fusion_iou": float(iou),

            # YOLO 프레임 간 집계 상태를 최종 결과까지 보존합니다.
            "yolo_frame_support": yolo_detection.get(
                "yolo_frame_support"
            ),
            "yolo_frame_ratio": yolo_detection.get(
                "yolo_frame_ratio"
            ),
            "yolo_representative_frame_index": (
                yolo_detection.get(
                    "yolo_representative_frame_index"
                )
            ),

            # RT-DETR 프레임 간 집계 상태를 최종 결과까지 보존합니다.
            "rtdetr_frame_support": rtdetr_detection.get(
                "frame_support"
            ),
            "rtdetr_frame_ratio": rtdetr_detection.get(
                "frame_ratio"
            ),
            "rtdetr_representative_frame_index": (
                rtdetr_detection.get(
                    "representative_frame_index"
                )
            ),
        }

    # 두 bbox를 confidence 가중 평균합니다.
    def _weighted_box(
        self,
        yolo_box,
        rtdetr_box,
        y_weight: float,
        r_weight: float,
    ):
        yolo_array = np.asarray(
            yolo_box,
            dtype=np.float32,
        )
        rtdetr_array = np.asarray(
            rtdetr_box,
            dtype=np.float32,
        )

        return (
            (
                yolo_array * y_weight
                + rtdetr_array * r_weight
            )
            / (y_weight + r_weight)
        ).astype(float).tolist()

    # bbox 두 개의 IoU를 계산합니다.
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
