# YOLO + RT-DETR + SAM2.1 Ensemble Integration

## 넣을 위치

아래 파일을 `a4_cobot2/object_detection/` 폴더에 넣습니다.

```text
object_detection/
  detection.py              # 기존 파일 대체
  yolo.py                   # 기존 파일 대체
  rtdetr.py                 # 새 파일 추가
  sam2_refiner.py           # 새 파일 추가
  ensemble_detector.py      # 새 파일 추가
```

## resource에 넣을 파일

기본 코드 기준 파일명은 아래와 같습니다.

```text
resource/
  yolo_seg_best_v2.pt
  class_name_tool.json
  rtdetr_best.pt
  sam2_1_finetuned.pt
  sam2.1_hiera_l.yaml
```

실제 파일명이 다르면 환경변수로 지정할 수 있습니다.

```bash
export A4_COBOT2_RTDETR_MODEL="네_rtdetr_weight.pt"
export A4_COBOT2_RTDETR_CLASS_JSON="class_name_tool.json"
export A4_COBOT2_SAM2_CHECKPOINT="/absolute/path/to/sam2_finetuned.pt"
export A4_COBOT2_SAM2_CONFIG="/absolute/path/to/sam2.1_hiera_l.yaml"
```

## 동작 방식

1. `detection.py`의 기본 모델이 `ensemble`로 바뀝니다.
2. `EnsembleDetector`가 YOLO frame capture를 한 번 수행합니다.
3. 같은 frame list로 YOLO와 RT-DETR inference를 수행합니다.
4. 같은 class + IoU 기준으로 bbox를 병합합니다.
5. 병합된 bbox를 SAM2.1 box prompt로 넣어 mask를 정교화합니다.
6. 최종 detection dict는 기존 YOLO와 같은 형식으로 반환합니다.

```python
{
    "name": "hammer",
    "class_id": 0,
    "box": [x1, y1, x2, y2],
    "confidence": 0.91,
    "mask": final_mask,
}
```

따라서 `task_manager`, `workspace_judge`, `robot_arm` 통신은 수정하지 않아도 됩니다.

## 빌드 전 확인

```bash
python3 -c "from object_detection.ensemble_detector import EnsembleDetector; print('ensemble ok')"
python3 -c "from object_detection.rtdetr import RTDETRModel; print('rtdetr ok')"
python3 -c "from object_detection.sam2_refiner import SAM2Refiner; print('sam2 refiner ok')"
```

## 빌드

```bash
cd ~/a4_cobot2_ws
colcon build --packages-select a4_cobot2
source install/setup.bash
```

## 주의

- `rtdetr_best.pt`가 없으면 RT-DETR은 자동 비활성화되고 YOLO+SAM2 fallback처럼 동작합니다.
- SAM2 checkpoint/config가 없거나 `import sam2`가 안 되면 SAM2는 자동 비활성화되고 YOLO mask fallback을 사용합니다.
- RT-DETR only 후보는 mask가 없기 때문에 SAM2 mask 생성에 실패하면 최종 detection에서 제외됩니다.
- `/yolo_detection_image` preview는 기존처럼 YOLO만 사용합니다. 실제 3자세 scan에서만 앙상블 detection이 사용됩니다.
