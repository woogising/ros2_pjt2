# ============================================================
# object_detection/rtdetr.py
# 역할:
#   - 파인튜닝한 RT-DETR 모델을 로드하고, YOLO와 같은 detection dict 형식으로
#     bbox/class/confidence 후보를 반환합니다.
# 사용 위치:
#   - ensemble_detector.py에서 YOLO-seg 결과와 RT-DETR 결과를 bbox/class 레벨로 병합할 때 사용합니다.
# 주의:
#   - RT-DETR은 mask를 만들지 않으므로, 최종 3D grasp 계산에 쓰려면 SAM2.1 또는 YOLO mask가 필요합니다.
#   - 기본 weight 파일명은 resource/rtdetr_best.pt입니다. 실제 파일명에 맞게 아래 상수나 환경변수를 바꾸세요.
# ============================================================
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

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
RTDETR_CLASS_NAME_JSON = os.getenv("A4_COBOT2_RTDETR_CLASS_JSON", "class_name_tool.json")

RTDETR_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", RTDETR_MODEL_FILENAME)
RTDETR_JSON_PATH = os.path.join(PACKAGE_PATH, "resource", RTDETR_CLASS_NAME_JSON)


class RTDETRModel:
    # RT-DETR weight와 class mapping을 로드하는 함수
    def __init__(self, model_path: Optional[str] = None, class_json_path: Optional[str] = None):
        self.model_path = model_path or RTDETR_MODEL_PATH
        self.class_json_path = class_json_path or RTDETR_JSON_PATH
        self.enabled = False
        self.model = None

        self.class_id_to_name: Dict[int, str] = {}
        self.reversed_class_dict: Dict[str, int] = {}
        self._load_class_names()
        self._load_model()

    # class_name_tool.json을 읽어 class_id <-> class_name 매핑을 만드는 함수
    def _load_class_names(self):
        with open(self.class_json_path, "r", encoding="utf-8") as file:
            class_dict = json.load(file)
            self.class_id_to_name = {int(k): v for k, v in class_dict.items()}
            self.reversed_class_dict = {v: int(k) for k, v in class_dict.items()}

    # Ultralytics RT-DETR 모델을 로드하는 함수
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[RTDETRModel] weight not found. RT-DETR disabled: {self.model_path}")
            self.enabled = False
            return

        try:
            if _RTDETRModel is not None:
                self.model = _RTDETRModel(self.model_path)
            else:
                # 일부 Ultralytics 버전은 YOLO(...)가 RT-DETR .pt도 자동 로드합니다.
                self.model = YOLO(self.model_path)
            self.enabled = True
            print(f"[RTDETRModel] loaded: {self.model_path}")
        except Exception as exc:
            print(f"[RTDETRModel] load failed. RT-DETR disabled: {exc}")
            self.enabled = False

    # target_names를 class_id set으로 바꾸는 함수
    def _target_ids_from_names(self, target_names: Optional[Iterable[str]]):
        if not target_names:
            return None

        target_ids = set()
        for name in target_names:
            if name in self.reversed_class_dict:
                target_ids.add(self.reversed_class_dict[name])
            else:
                print(f"[RTDETRModel] unknown target class ignored: {name}")
        return target_ids

    # 여러 frame에 대해 RT-DETR inference를 수행하고 YOLO와 같은 detection dict 형식으로 반환하는 함수
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
            print(f"[RTDETRModel] inference failed: {exc}")
            return []

        raw = []
        for res in results:
            if res.boxes is None:
                continue
            for box, score, label in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                class_id = int(label)
                if float(score) < confidence_threshold:
                    continue
                if target_ids is not None and class_id not in target_ids:
                    continue

                raw.append({
                    "name": self.class_id_to_name.get(class_id, f"unknown_{class_id}"),
                    "class_id": class_id,
                    "box": [float(v) for v in box],
                    "confidence": float(score),
                    "mask": None,
                    "source": "rtdetr",
                })

        raw.sort(key=lambda x: x["confidence"], reverse=True)
        return raw
