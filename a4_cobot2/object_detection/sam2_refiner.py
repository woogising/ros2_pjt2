# ============================================================
# object_detection/sam2_refiner.py
# 역할:
#   - YOLO/RT-DETR bbox를 SAM2.1 box prompt로 넣어 최종 mask를 정교화합니다.
# 사용 위치:
#   - ensemble_detector.py에서 bbox fusion 이후 final detection의 mask를 보정할 때 사용합니다.
# 주의:
#   - SAM2.1 config/checkpoint 경로는 환경마다 다릅니다.
#   - 기본값은 resource/sam2.1_hiera_l.yaml, resource/sam2_1_finetuned.pt 입니다.
#   - 실제 파일명에 맞게 환경변수 또는 상수를 수정하세요.
# ============================================================
import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ament_index_python.packages import get_package_share_directory


PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

SAM2_CONFIG_PATH = os.getenv(
    "A4_COBOT2_SAM2_CONFIG",
    os.path.join(PACKAGE_PATH, "resource", "sam2.1_hiera_b+.yaml"),
)
SAM2_CHECKPOINT_PATH = os.getenv(
    "A4_COBOT2_SAM2_CHECKPOINT",
    os.path.join(PACKAGE_PATH, "resource", "sam2_1_finetuned.pt"),
)


class SAM2Refiner:
    # SAM2.1 image predictor를 준비하는 함수
    def __init__(self, config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
        self.config_path = config_path or SAM2_CONFIG_PATH
        self.checkpoint_path = checkpoint_path or SAM2_CHECKPOINT_PATH
        self.enabled = False
        self.predictor = None
        self.last_image_id = None

        self._load_predictor()

    # sam2 패키지와 checkpoint/config를 로드하는 함수
    def _load_predictor(self):
        if not os.path.exists(self.checkpoint_path):
            print(f"[SAM2Refiner] checkpoint not found. SAM2 disabled: {self.checkpoint_path}")
            return
        if not os.path.exists(self.config_path) and os.path.isabs(self.config_path):
            print(f"[SAM2Refiner] config not found. SAM2 disabled: {self.config_path}")
            return

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_sam2(self.config_path, self.checkpoint_path, device=device)
            self.predictor = SAM2ImagePredictor(model)
            self.enabled = True
            print(f"[SAM2Refiner] loaded: {self.checkpoint_path} on {device}")
        except Exception as exc:
            print(f"[SAM2Refiner] load failed. SAM2 disabled: {exc}")
            self.enabled = False
            self.predictor = None

    # BGR frame을 SAM2가 기대하는 RGB image로 변환하고 predictor에 설정하는 함수
    def set_image(self, frame_bgr):
        if not self.enabled or self.predictor is None:
            return False
        if frame_bgr is None:
            return False

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        self.last_image_id = id(frame_bgr)
        return True

    # bbox prompt를 이용해 SAM2 mask를 반환하는 함수
    def predict_mask_from_box(self, frame_bgr, box: Sequence[float]) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if not self.enabled or self.predictor is None:
            return None, None
        if frame_bgr is None or box is None or len(box) != 4:
            return None, None

        try:
            self.set_image(frame_bgr)
            input_box = np.array(box, dtype=np.float32)

            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=True,
            )

            if masks is None or len(masks) == 0:
                return None, None

            scores_np = np.asarray(scores, dtype=np.float32)
            best_idx = int(np.argmax(scores_np)) if len(scores_np) else 0
            mask = masks[best_idx]
            score = float(scores_np[best_idx]) if len(scores_np) else None

            mask = self._normalize_mask(mask, frame_bgr.shape[:2])
            if not self._is_reasonable_mask(mask, box):
                return None, score

            return mask, score

        except Exception as exc:
            print(f"[SAM2Refiner] predict failed: {exc}")
            return None, None

    # SAM2 mask를 HxW uint8/bool 형태로 정규화하는 함수
    def _normalize_mask(self, mask, target_hw):
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask = mask.astype(np.uint8)

        h, w = target_hw
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return (mask > 0).astype(np.uint8)

    # mask가 너무 작거나 bbox와 거의 겹치지 않는 비정상 결과인지 확인하는 함수
    def _is_reasonable_mask(self, mask: np.ndarray, box: Sequence[float]) -> bool:
        if mask is None:
            return False

        area = int(mask.sum())
        if area < 20:
            return False

        h, w = mask.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return False

        box_area = max(1, (x2 - x1) * (y2 - y1))
        inside_area = int(mask[y1:y2, x1:x2].sum())

        # mask 대부분이 bbox 밖에 있거나 bbox 대비 너무 큰 경우를 방지합니다.
        if inside_area / max(1, area) < 0.50:
            return False
        if area > box_area * 4.0:
            return False

        return True
