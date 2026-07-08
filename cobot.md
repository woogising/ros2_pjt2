# a4_cobot2 — 음성 명령 기반 작업공간 정리 로봇

두산 협동로봇(m0609) + RealSense + YOLO segmentation으로, **음성 명령을 받아 물체를 인식하고 올바른 구역으로 정리**하는 ROS2 시스템.

---

## 전체 흐름

```
음성("작업공간 확인해줘")
  → command_input_node (STT + LLM 분류) → /task_command: check_workspace
  → task_manager_node → /start_workspace_scan
  → robot_arm_node : 3자세(중앙/좌/우)로 이동하며 각 자세의 base←camera 변환 발행
  → object_detection_node : YOLO seg로 감지 → base 좌표/각도 계산 → 3자세 병합
  → /scanned_objects_base → task_manager → /judge_workspace
  → workspace_judge_node : 물체가 올바른 구역(zone)에 있는지 판정 (정상/오배치)

음성("정리 시작해줘")
  → /task_command: start_organize
  → task_manager → /organize_objects (action) → robot_arm_node
  → 오배치 물체를 각도에 맞춰 집어 해당 구역 중앙에 놓음
```

---

## 노드 구성 (8개)

| 노드 | 역할 |
|------|------|
| **command_input_node** (voice) | 웨이크워드 "hello rokey" → OpenAI Whisper STT → LLM 명령 분류 → `/task_command` 발행 |
| **task_manager_node** | 전체 흐름 제어. 스캔 시작 요청, 판정 요청, 정리 action 요청, 상태/안내 발행 |
| **object_detection_node** | YOLO seg 인식, 3자세 스캔, camera→base 변환, PCA 각도 계산, 인식 프리뷰 발행 |
| **workspace_judge_node** | 구역(zone) 기반 정상/오배치 판정 |
| **robot_arm_node** | 3자세 스캔 주도(로봇 이동), 정리 action, 실제 pick-and-place |
| **safety_node** | stop/clear, `/emergency_stop`, `/safety_state` 관리 |
| **status_notifier_node** | 상태를 TTS로 음성 안내 |
| **db_node** | 작업 이벤트를 SQLite에 로깅 (구독 전용) |

---

## 주요 topic / service / action

**명령·상태**
- `/task_command`, `/task_command_raw`, `/task_status`, `/user_notice`
- `/safety_command`, `/safety_state`, `/emergency_stop`

**3자세 스캔 (robot_arm ↔ detection)**
- `/start_workspace_scan` (srv `std_srvs/Trigger`) — task_manager → robot_arm, 스캔 시작
- `/scan_pose_transform` (`Float64MultiArray`) — robot_arm → detection, `[index, total, base←camera 4x4]`
- `/scan_capture_done` (`Int32`) — detection → robot_arm, 자세별 캡처 완료 ack
- `/scanned_objects_base` (`String` JSON) — detection → task_manager, 병합된 base 좌표 물체 목록

**판정·정리**
- `/judge_workspace` (srv) — task_manager → judge
- `/organize_objects` (action) — task_manager → robot_arm

**디버깅**
- `/yolo_detection_image` (`sensor_msgs/Image`) — YOLO 인식 결과 프리뷰 (`rqt_image_view`로 확인)

---

## 좌표계 / 인식

- 카메라는 **eye-in-hand**(그리퍼에 장착). 스캔 자세마다 카메라 위치가 달라짐
- detection이 **3개 관측 자세**(중앙/좌/우)에서 감지하고, 각 자세의 `T_gripper2camera.npy` 캘리브레이션으로 **camera → base(mm) 변환**
- 같은 물체는 3자세에서 base 좌표가 같게 나오므로 이름 기준으로 병합
- **물체 각도**: segmentation mask의 PCA로 긴 축을 구하고, base 프레임 각도로 변환(자세 무관) → 파지 시 그리퍼 회전에 사용

## 구역(zone) / 클래스 매핑

작업공간을 **base 좌표계 4개 구역**으로 나눔 (green / yellow / red / blue).

| zone | 클래스 |
|------|--------|
| red | hammer, screwdriver |
| blue | bolt, tape |
| green | green_apple, pineapple |
| yellow | pocari, gatorade |

- 물체 현재 위치가 매핑된 구역 안에 있으면 정상, 아니면 오배치
- 오배치 물체는 해당 구역 **정중앙**에 놓음 (`place_position`)

---

## 실행

**전제 (먼저 실행)**
```bash
# RealSense 카메라 (aligned depth)
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# Doosan 로봇 bringup (dsr01 / m0609)
```

**a4_cobot2 전체**
```bash
cd ~/ws_cobot2_pjt
colcon build --packages-select a4_cobot2
source install/setup.bash
ros2 launch a4_cobot2 a4_cobot2.launch.py
```

**음성 명령**
- "hello rokey" → "작업공간 확인해줘" (스캔+판정)
- "hello rokey" → "정리 시작해줘" (파지·이동)

**인식 화면 확인**
```bash
ros2 run rqt_image_view rqt_image_view /yolo_detection_image
```

---

## 주요 설정 / 튜닝 지점

**robot_arm/robot_motion.py**
- `SCAN_POSES_DEG` — 3개 관측 자세(joint)
- `GRASP_ORIENTATION` — 탑다운 파지 그리퍼 자세 (실측값 필요)
- `APPROACH_Z_OFFSET_MM` — 접근 높이
- `PICK_Z_OFFSET_MM` — 최종 하강 보정(물건 위를 잡으면 키움)
- 파지 각도 회전 방향/오프셋(±90 등)

**workspace/workspace_judge_utils.py**
- `DEFAULT_ZONES` — 4개 구역 base 좌표(mm)
- `CLASS_TO_ZONE` — 클래스 → 구역 매핑

**resource/**
- `yolo_seg_best_v1.pt` — YOLO segmentation 모델
- `class_name_tool.json` — 클래스 이름표 (모델과 일치해야 함)
- `T_gripper2camera.npy` — 카메라 캘리브레이션 (mm)
- `.env` — `OPENAI_API_KEY` (STT/분류)

**TTS**
- `voice/tts.py` (spd-say, 로컬·즉시·기계음)
- `voice/tts2.py` (OpenAI TTS, 자연스러움·느림)
- `voice/tts3.py` (edge-tts, 자연스러움·빠름, `pip install edge-tts`)
- 쓰려는 것에 맞춰 `command_input_node`/`status_notifier_node`의 import 교체

---

## 알려진 한계 / TODO

- **시차(parallax)**: 카메라가 물체를 비스듬히 봐서 윗면 중심이 약간 어긋남 (특히 키 큰 병). → "물체 위에서 재촬영" 또는 mask footprint 중심 사용으로 개선 예정
- **파지 각도**: `RZ_OFFSET`·회전 방향·하강 부호는 실기에서 튜닝 필요
- **GRASP_ORIENTATION**: 실제 안전한 탑다운 집기 자세로 확정 필요
- **한 구역에 2클래스**: 둘 다 오배치면 같은 중앙점에 놓여 겹칠 수 있음 (필요 시 슬롯 분리)
