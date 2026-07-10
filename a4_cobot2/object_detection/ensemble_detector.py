# ============================================================
# object_detection/ensemble_detector.py
# 역할:
#   - YOLO-seg, RT-DETR, SAM2.1을 object_detection 내부에서 결합합니다.
#   - 외부에는 기존 YoloModel.get_all_detections()와 같은 detection dict 형식으로 반환합니다.
# 설계:
#   - YOLO-seg: 기본 class/bbox/mask 후보
#   - RT-DETR: bbox/class 검증 및 YOLO 누락 후보 보완
#   - SAM2.1: 최종 bbox prompt 기반 mask refinement
# 주의:
#   - ROS2 topic/service/action은 변경하지 않습니다.
#   - detection.py는 self.model.get_all_detections(img_node)만 호출하면 됩니다.
# ============================================================
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .yolo import YoloModel
from .rtdetr import RTDETRModel
from .sam2_refiner import SAM2Refiner


class EnsembleDetector:
    # YOLO, RT-DETR, SAM2.1 wrapper를 초기화하는 함수
    def __init__(
        self,
        frame_capture_sec: float = 1.0,
        fuse_iou_threshold: float = 0.50,
        rtdetr_only_conf_threshold: float = 0.60,
        yolo_weight: float = 0.60,
        rtdetr_weight: float = 0.40,
    ):
        self.yolo = YoloModel()
        self.rtdetr = RTDETRModel()
        self.sam2 = SAM2Refiner()

        # detection.py의 publish_detection_image(), save_scan_pose_images()는
        # self.model.model(frame)을 호출한다. 기존 preview를 깨지 않도록 YOLO 원본 모델을 노출한다.
        self.model = self.yolo.model

        self.frame_capture_sec = frame_capture_sec
        self.fuse_iou_threshold = fuse_iou_threshold
        self.rtdetr_only_conf_threshold = rtdetr_only_conf_threshold
        self.yolo_weight = yolo_weight
        self.rtdetr_weight = rtdetr_weight

    # 현재 프레임들에서 YOLO+RT-DETR 후보를 만들고 SAM2 mask로 정제한 detection list를 반환하는 함수
    def get_all_detections(self, img_node, target_names: Optional[Iterable[str]] = None) -> List[dict]:
        frames = self.yolo.get_frames(img_node, duration=self.frame_capture_sec)
        if not frames:
            print("[EnsembleDetector] no frames captured")
            return []

        yolo_dets = self._get_yolo_detections(frames, target_names=target_names)
        rtdetr_dets = self.rtdetr.get_all_detections_from_frames(
            frames,
            target_names=target_names,
        )

        fused = self._fuse_yolo_rtdetr(yolo_dets, rtdetr_dets)
        frame_for_sam = frames[-1]

        final_dets = []
        for det in fused:
            det = dict(det)
            yolo_mask = det.get("mask")

            sam_mask, sam_score = self.sam2.predict_mask_from_box(frame_for_sam, det.get("box"))
            if sam_mask is not None:
                det["mask"] = sam_mask
                det["sam2_score"] = sam_score
                det["mask_source"] = "sam2"
            elif yolo_mask is not None:
                det["mask"] = yolo_mask
                det["sam2_score"] = None
                det["mask_source"] = "yolo"
            else:
                # 현재 detection.py는 mask가 없으면 3D cloud를 못 만들기 때문에 버린다.
                continue

            final_dets.append(det)

        final_dets.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        print(
            f"[EnsembleDetector] yolo={len(yolo_dets)}, rtdetr={len(rtdetr_dets)}, "
            f"final={len(final_dets)}"
        )
        return final_dets

    # YOLO wrapper가 frame list를 직접 받을 수 있으면 그 함수를 사용하고, 없으면 기존 로직을 최대한 재현하는 함수
    def _get_yolo_detections(self, frames: Sequence, target_names: Optional[Iterable[str]] = None) -> List[dict]:
        if hasattr(self.yolo, "get_all_detections_from_frames"):
            return self.yolo.get_all_detections_from_frames(frames, target_names=target_names)

        # yolo.py를 아직 패치하지 않았을 때의 fallback입니다.
        target_ids = None
        if target_names:
            target_ids = set()
            for name in target_names:
                if name in self.yolo.reversed_class_dict:
                    target_ids.add(self.yolo.reversed_class_dict[name])

        results = self.yolo.model(frames, verbose=False, retina_masks=True)
        detections = self.yolo._aggregate_detections(results)

        converted = []
        for det in detections:
            class_id = int(det["label"])
            if target_ids is not None and class_id not in target_ids:
                continue
            name = self.yolo.class_id_to_name.get(class_id, f"unknown_{class_id}")
            converted.append({
                "name": name,
                "class_id": class_id,
                "box": det["box"],
                "confidence": float(det["score"]),
                "mask": det.get("mask"),
                "source": "yolo",
                "yolo_confidence": float(det["score"]),
            })
        converted.sort(key=lambda x: x["confidence"], reverse=True)
        return converted

    # YOLO와 RT-DETR bbox 후보를 같은 class + IoU 기준으로 병합하는 함수
    def _fuse_yolo_rtdetr(self, yolo_dets: List[dict], rtdetr_dets: List[dict]) -> List[dict]:
        fused = []
        used_rtdetr = set()

        for ydet in yolo_dets:
            best_idx = None
            best_iou = 0.0
            for idx, rdet in enumerate(rtdetr_dets):
                if idx in used_rtdetr:
                    continue
                if int(ydet.get("class_id", -1)) != int(rdet.get("class_id", -2)):
                    continue

                iou = self._iou(ydet["box"], rdet["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= self.fuse_iou_threshold:
                rdet = rtdetr_dets[best_idx]
                used_rtdetr.add(best_idx)
                fused.append(self._merge_pair(ydet, rdet, best_iou))
            else:
                out = dict(ydet)
                out["source"] = "yolo"
                out["yolo_confidence"] = float(ydet.get("confidence", 0.0))
                out["rtdetr_confidence"] = None
                fused.append(out)

        for idx, rdet in enumerate(rtdetr_dets):
            if idx in used_rtdetr:
                continue
            if float(rdet.get("confidence", 0.0)) < self.rtdetr_only_conf_threshold:
                continue
            out = dict(rdet)
            out["source"] = "rtdetr_only"
            out["yolo_confidence"] = None
            out["rtdetr_confidence"] = float(rdet.get("confidence", 0.0))
            # RT-DETR only는 mask가 없으므로 SAM2 성공 시에만 최종 detection으로 살아남는다.
            fused.append(out)

        return fused

    # 매칭된 YOLO/RT-DETR pair를 하나의 detection으로 합치는 함수
    def _merge_pair(self, ydet: dict, rdet: dict, iou: float) -> dict:
        yconf = float(ydet.get("confidence", 0.0))
        rconf = float(rdet.get("confidence", 0.0))

        box = self._weighted_box(
            ydet["box"],
            rdet["box"],
            y_weight=max(1e-6, self.yolo_weight * yconf),
            r_weight=max(1e-6, self.rtdetr_weight * rconf),
        )

        confidence = (self.yolo_weight * yconf) + (self.rtdetr_weight * rconf)

        return {
            "name": ydet.get("name") or rdet.get("name"),
            "class_id": int(ydet.get("class_id", rdet.get("class_id"))),
            "box": box,
            "confidence": float(confidence),
            "mask": ydet.get("mask"),  # SAM2 실패 시 fallback으로 사용
            "source": "yolo+rtdetr",
            "yolo_confidence": yconf,
            "rtdetr_confidence": rconf,
            "fusion_iou": float(iou),
        }

    # 두 bbox를 confidence 가중 평균하는 함수
    def _weighted_box(self, ybox, rbox, y_weight: float, r_weight: float):
        y = np.asarray(ybox, dtype=np.float32)
        r = np.asarray(rbox, dtype=np.float32)
        return ((y * y_weight + r * r_weight) / (y_weight + r_weight)).astype(float).tolist()

    # 두 bbox의 IoU를 계산하는 함수
    def _iou(self, box1, box2) -> float:
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
