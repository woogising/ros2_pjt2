# ============================================================
# object_detection/yolo.py
# 역할:
#   - YOLO 모델을 로드하고, 여러 RGB frame에서 탐지 결과를 모아 안정적인 detection 목록을 만듭니다.
# 주요 함수:
#   - get_all_detections(): 작업공간 전체 스캔용
#   - get_best_detection(): 특정 target 하나만 찾는 구버전/디버깅용
# 후처리:
#   - 여러 frame의 bbox를 IoU 기준으로 묶고 평균 bbox/confidence를 사용합니다.
# ============================================================
########## YoloModel ##########
import os
import json
import time
from collections import Counter

import rclpy
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
import numpy as np


PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

YOLO_MODEL_FILENAME = "yolo_seg_best_v2.pt"
YOLO_CLASS_NAME_JSON = "class_name_tool.json"

YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_MODEL_FILENAME)
YOLO_JSON_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_CLASS_NAME_JSON)


class YoloModel:
    def __init__(self):
        # YOLO_MODEL_PATH의 학습된 모델 파일을 로드합니다.
        self.model = YOLO(YOLO_MODEL_PATH)

        # class_name_tool.json을 읽어 class_id <-> class_name 매핑을 만듭니다.
        # get_all_detections()는 최종적으로 name과 class_id를 둘 다 반환합니다.
        with open(YOLO_JSON_PATH, "r", encoding="utf-8") as file:
            class_dict = json.load(file)
            self.class_id_to_name = {int(k): v for k, v in class_dict.items()}
            self.reversed_class_dict = {v: int(k) for k, v in class_dict.items()}

    def get_frames(self, img_node, duration=1.0):
        """get frames while target_time"""
        end_time = time.time() + duration
        frames = {}

        while time.time() < end_time:
            rclpy.spin_once(img_node)
            frame = img_node.get_color_frame()
            stamp = img_node.get_color_frame_stamp()
            if frame is not None:
                frames[stamp] = frame
            time.sleep(0.01)

        if not frames:
            print(f"No frames captured in {duration:.2f} seconds")

        print(f"{len(frames)} frames captured")
        return list(frames.values())

    def get_best_detection(self, img_node, target):
        rclpy.spin_once(img_node)
        frame = self.get_frames(img_node)

        frames = self.PCI_MAP(frame, 2.0)

        if not frames:  # Check if frames are empty
            return None, None
        if target not in self.reversed_class_dict:
            print(f"Unknown target class: {target}")
            return None, None

        results = self.model(frames, verbose=False)
        print("classes: ")
        print(results[0].names)
        detections = self._aggregate_detections(results)
        label_id = self.reversed_class_dict[target]
        print("label_id: ", label_id)
        print("detections: ", detections)

        matches = [d for d in detections if d["label"] == label_id]
        if not matches:
            print("No matches found for the target label.")
            return None, None
        best_det = max(matches, key=lambda x: x["score"])
        return best_det["box"], best_det["score"]

    def get_all_detections(self, img_node, target_names=None):
        """
        현재 카메라 프레임에서 탐지된 모든 물체를 반환한다.

        반환 형식:
        [
            {
                "name": "hammer",
                "class_id": 1,
                "box": [x1, y1, x2, y2],
                "confidence": 0.89
            }
        ]
        """
        rclpy.spin_once(img_node)
        frames = self.get_frames(img_node)

        if not frames:
            print("No frames available for detection.")
            return []

        target_ids = None
        if target_names:
            target_ids = set()
            for name in target_names:
                if name in self.reversed_class_dict:
                    target_ids.add(self.reversed_class_dict[name])
                else:
                    print(f"Unknown target class ignored: {name}")

        results = self.model(frames, verbose=False, retina_masks=True)
        detections = self._aggregate_detections(results)

        converted = []
        for det in detections:
            class_id = int(det["label"])

            if target_ids is not None and class_id not in target_ids:
                continue

            name = self.class_id_to_name.get(class_id, f"unknown_{class_id}")

            converted.append(
                {
                    "name": name,
                    "class_id": class_id,
                    "box": det["box"],
                    "confidence": float(det["score"]),
                    "mask": det.get("mask"),
                }
            )

        converted.sort(key=lambda x: x["confidence"], reverse=True)
        return converted
    
    def PCI_MAP(in_img, ratio):
        in_img = in_img.astype(np.float32)
        
        if len(in_img.shape) == 3:  
            v, h, d = in_img.shape

        Np = round(ratio * v * h)
        a1 = np.copy(in_img)

        a1[:, :, 0] /= np.sum(a1[:, :, 0]) 
        a1[:, :, 1] /= np.sum(a1[:, :, 1])
        a1[:, :, 2] /= np.sum(a1[:, :, 2])
                
        b = Np * a1 
        mu = []
        s2 = []
        for i in range(d):
            mu.append(np.mean(b[:, :, i])) 
            s2.append(np.var(b[:, :, i])) 
        mu = np.array(mu)
        s2 = np.array(s2)
        
    

        alpha = np.where(s2 != 0, mu**2 / s2, 0)
        beta = np.where(s2 != 0, mu / s2, 0)

        c = np.zeros_like(b)
        for i in range(d):
            c[:, :, i] = np.random.poisson(b[:, :, i])

        MAP = np.zeros_like(b)
        for i in range(d):
            mask = b[:, :, i] > 0
            MAP[:, :, i] = (c[:, :, i] + alpha[i] * mask) / (1 + beta[i] * mask)

        MAP1 = MAP / Np
        if np.max(MAP1) > 0:
            out_img = MAP1 / np.max(MAP1)
        else:
            out_img = MAP1  

        out_img = (out_img * 255).astype(np.uint8)  
        return out_img

    def _aggregate_detections(self, results, confidence_threshold=0.8, iou_threshold=0.5):
        """
        Fuse raw detection boxes across frames using IoU-based grouping
        and majority voting for robust final detections.
        """
        raw = []
        for res in results:
            masks = res.masks.data.cpu().numpy() if res.masks is not None else None
            for k, (box, score, label) in enumerate(zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            )):
                if score >= confidence_threshold:
                    mask = masks[k] if (masks is not None and k < len(masks)) else None
                    raw.append({"box": box, "score": score, "label": int(label), "mask": mask})

        final = []
        used = [False] * len(raw)

        for i, det in enumerate(raw):
            if used[i]:
                continue
            group = [det]
            used[i] = True
            for j, other in enumerate(raw):
                if not used[j] and other["label"] == det["label"]:
                    if self._iou(det["box"], other["box"]) >= iou_threshold:
                        group.append(other)
                        used[j] = True

            boxes = np.array([g["box"] for g in group])
            scores = np.array([g["score"] for g in group])
            labels = [g["label"] for g in group]
            best = max(group, key=lambda g: g["score"])

            final.append(
                {
                    "box": boxes.mean(axis=0).tolist(),
                    "score": float(scores.mean()),
                    "label": Counter(labels).most_common(1)[0][0],
                    "mask": best["mask"],
                }
            )

        return final

    def _iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
        """
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
