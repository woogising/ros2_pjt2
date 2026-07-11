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
        self.last_reason = "not_run"

        self._load_predictor()

    # sam2 패키지와 checkpoint/config를 로드하는 함수
    def _load_predictor(self):
        import traceback

        print("[SAM2Refiner] load start", flush=True)
        print(f"[SAM2Refiner] config_path={self.config_path}", flush=True)
        print(f"[SAM2Refiner] checkpoint_path={self.checkpoint_path}", flush=True)
        print(
            f"[SAM2Refiner] config_exists={os.path.isfile(self.config_path)}",
            flush=True,
        )
        print(
            f"[SAM2Refiner] checkpoint_exists={os.path.isfile(self.checkpoint_path)}",
            flush=True,
        )

        if not os.path.isfile(self.checkpoint_path):
            print(
                f"[SAM2Refiner] checkpoint not found. "
                f"SAM2 disabled: {self.checkpoint_path}",
                flush=True,
            )
            self.enabled = False
            return

        if not os.path.isfile(self.config_path):
            print(
                f"[SAM2Refiner] config not found. "
                f"SAM2 disabled: {self.config_path}",
                flush=True,
            )
            self.enabled = False
            return

        try:
            import torch

            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = "cuda" if torch.cuda.is_available() else "cpu"

            config_path = os.path.abspath(self.config_path)
            config_dir = os.path.dirname(config_path)
            config_name = os.path.basename(config_path)

            print(f"[SAM2Refiner] device={device}", flush=True)
            print(f"[SAM2Refiner] config_dir={config_dir}", flush=True)
            print(f"[SAM2Refiner] config_name={config_name}", flush=True)

            if GlobalHydra.instance().is_initialized():
                print(
                    "[SAM2Refiner] clearing existing Hydra instance",
                    flush=True,
                )
                GlobalHydra.instance().clear()

            with initialize_config_dir(
                config_dir=config_dir,
                version_base="1.2",
            ):
                print(
                    "[SAM2Refiner] Hydra initialized with resource directory",
                    flush=True,
                )

                model = build_sam2(
                    config_file=config_name,
                    ckpt_path=self.checkpoint_path,
                    device=device,
                )

            self.predictor = SAM2ImagePredictor(model)
            self.enabled = True

            print(
                f"[SAM2Refiner] loaded successfully: "
                f"checkpoint={self.checkpoint_path}, device={device}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[SAM2Refiner] load failed. "
                f"type={type(exc).__name__}, error={exc}",
                flush=True,
            )
            traceback.print_exc()

            self.enabled = False
            self.predictor = None

    # BGR frame을 SAM2가 기대하는 RGB image로 변환하고 predictor에 설정하는 함수
    def set_image(self, frame_bgr):
        if not self.enabled or self.predictor is None:
            return False

        if frame_bgr is None:
            return False

        current_image_id = id(frame_bgr)

        # 같은 numpy frame이면 image embedding을 다시 계산하지 않는다.
        if self.last_image_id == current_image_id:
            return True

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        self.last_image_id = current_image_id

        print(
            f"[SAM2Refiner] set_image completed: "
            f"shape={frame_bgr.shape}, image_id={current_image_id}",
            flush=True,
        )

        return True

    # bbox prompt를 이용해 SAM2 mask를 반환하는 함수
    def predict_mask_from_box(
        self,
        frame_bgr,
        box: Sequence[float],
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:

        self.last_reason = None

        if not self.enabled:
            self.last_reason = "sam2_disabled"
            return None, None

        if self.predictor is None:
            self.last_reason = "predictor_is_none"
            return None, None

        if frame_bgr is None:
            self.last_reason = "frame_is_none"
            return None, None

        if box is None or len(box) != 4:
            self.last_reason = "invalid_box"
            return None, None

        try:
            # 기존 self.set_image(frame_bgr)를 이것으로 교체
            if not self.set_image(frame_bgr):
                self.last_reason = "set_image_failed"
                return None, None

            input_box = np.array(box, dtype=np.float32)

            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=True,
            )

            if masks is None or len(masks) == 0:
                self.last_reason = "no_mask_returned"
                return None, None

            scores_np = np.asarray(scores, dtype=np.float32)
            best_idx = int(np.argmax(scores_np)) if len(scores_np) else 0

            mask = masks[best_idx]
            score = float(scores_np[best_idx]) if len(scores_np) else None

            mask = self._normalize_mask(mask, frame_bgr.shape[:2])

            if not self._is_reasonable_mask(mask, box):
                self.last_reason = "mask_rejected"
                return None, score

            self.last_reason = "success"
            return mask, score

        except Exception as exc:
            self.last_reason = f"predict_exception:{type(exc).__name__}"

            print(
                f"[SAM2Refiner] predict failed: "
                f"type={type(exc).__name__}, error={exc}",
                flush=True,
            )

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
