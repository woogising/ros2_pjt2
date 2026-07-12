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

## 노드 구성 (10개)

| 노드 | 역할 |
|------|------|
| **command_input_node** (voice) | 웨이크워드 "hello rokey" → OpenAI Whisper STT → LLM 명령 분류 → `/task_command` 발행 |
| **task_manager_node** | 전체 흐름 제어. 스캔 시작 요청, 판정 요청, 정리 action 요청, 상태/안내 발행 |
| **object_detection_node** | YOLO seg 인식, 3자세 스캔, camera→base 변환, PCA 각도 계산, 인식 프리뷰 발행 |
| **workspace_judge_node** | 구역(zone) 기반 정상/오배치 판정 |
| **robot_arm_node** | 3자세 스캔 주도(로봇 이동), 정리 action, 실제 pick-and-place |
| **safety_node** | stop/clear, `/emergency_stop`, `/safety_state` 관리 |
| **status_notifier_node** | 상태를 TTS로 음성 안내 |
| **db_node** | 작업 이벤트를 SQLite에 로깅 + 작업 단위로 **실제 `ros2 bag record` 자동 녹화** (rosbag_manager.py, metadata 검증 후 DB 연결) |
| **hmi_interface_node** | PyQt HMI. `/workspace_judgement` 구독으로 스캔 후 실제 물체 현황 테이블 표시(PLACED/MISPLACED), WAKE UP 버튼으로 wakeword 생략 |
| **vlm_report_node** | 재검증 이미지 + 판정 JSON으로 GPT-4o 최종 보고문 생성 (`/generate_final_report`) |

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
- `/workspace_judgement` (`String` JSON, TRANSIENT_LOCAL) — task_manager → HMI, 최신 판정 결과 (no_objects일 때도 빈 결과 발행)
- `/generate_final_report` (srv) — task_manager → vlm_report, 재검증 후 최종 보고문

**디버깅**
- `/yolo_detection_image` (`sensor_msgs/Image`) — YOLO 인식 결과 프리뷰 (`rqt_image_view`로 확인)

---

## 좌표계 / 인식

- 카메라는 **eye-in-hand**(그리퍼에 장착). 스캔 자세마다 카메라 위치가 달라짐
- detection이 **3개 관측 자세**(중앙/좌/우)에서 감지하고, 각 자세의 `T_gripper2camera.npy` 캘리브레이션으로 **camera → base(mm) 변환**
- detector 기본값은 **ensemble**(YOLO-seg + RT-DETR + SAM2.1). YOLO/RT-DETR 모두 프레임 간 track 집계(같은 frame의 동일 클래스 bbox는 다른 track, 최소 2프레임/10% 반복 감지 필터)로 순간 오검출을 제거
- 같은 물체는 이름 기준으로 그룹핑하되, **3자세 클라우드를 합치지 않고 bbox면적×confidence가 가장 높은 자세 하나의 cloud만 사용** (position/angle/width 전부 그 자세 기준 — d813d97 병합에서 vstack 병합 방식에서 변경됨)
- **물체 각도**: segmentation mask의 PCA로 긴 축을 구하고, base 프레임 각도로 변환 → 파지 시 그리퍼 회전에 사용 (robot_motion에서 `rot = object_angle`로 직접 사용, ±90 접기 제거됨)

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
- `yolo_seg_best_v4.pt` — YOLO segmentation 모델 (v2는 d813d97에서 삭제됨)
- `rtdetr_best.pt`, `sam2_1_finetuned.pt`, `sam2.1_hiera_b+.yaml` — ensemble용 RT-DETR/SAM2.1
- `class_name_tool.json` — 클래스 이름표 (모델과 일치해야 함)
- `T_gripper2camera.npy` — 카메라 캘리브레이션 (mm)
- `.env` — `OPENAI_API_KEY` (STT/분류/VLM)

**TTS**
- `voice/tts.py` — 현재 edge-tts 기반(비동기 백그라운드 재생, `pip install edge-tts` + ffplay 필요). RATE +80%, PITCH +50Hz
- (과거의 tts2.py/tts3.py 파일은 현재 존재하지 않음 — tts.py 하나로 통합됨)

**DB / rosbag** (`launch/a4_cobot2.launch.py`의 db_node 파라미터)
- `enable_rosbag: True` — 작업(run) 시작/종료에 맞춰 `ros2 bag record` 자동 실행
- `bag_path: ~/a4_cobot2_ws/a4_cobot2_log/bags`, storage sqlite3, 카메라 원본/depth 토픽 포함 녹화

---

## 알려진 한계 / TODO

- **시차(parallax)**: 카메라가 물체를 비스듬히 봐서 윗면 중심이 약간 어긋남 (특히 키 큰 병). → "물체 위에서 재촬영" 또는 mask footprint 중심 사용으로 개선 예정
- **파지 각도**: `RZ_OFFSET`·회전 방향·하강 부호는 실기에서 튜닝 필요
- **GRASP_ORIENTATION**: 실제 안전한 탑다운 집기 자세로 확정 필요
- **한 구역에 2클래스**: 둘 다 오배치면 같은 중앙점에 놓여 겹칠 수 있음 (필요 시 슬롯 분리)

---
---

# 코드 분석 — 잠재 오류점 & 개선 제안 (2026-07-11)

전체 소스(약 8,500줄, 노드 10개)를 검토한 결과입니다.
**심각도 순서**: ① 안전 → ② 시스템 멈춤(deadlock/crash) → ③ 동작 오류 → ④ 성능/자원 → ⑤ 빌드·설정·문서.

> **갱신 (d813d97 병합 반영)**: 이 분석 직후 팀원 커밋 6090994("angle, hmi 수정 ensemble, ros2 bag 추가")가 병합되었습니다.
> 아래 각 항목에 **[해결됨]** / **[변경됨]** / (표시 없음 = 여전히 유효) 상태를 달았고,
> 병합으로 새로 생긴 이슈는 마지막 **⑥ d813d97 병합 이후 변경·신규 이슈** 섹션에 정리했습니다.

## ① 안전 관련 (로봇이 위험해질 수 있는 것)

### 1. `force_down()` 무한 루프 — `robot_motion.py:222~236`
z축 외력이 6N을 넘을 때까지 `while True`로 대기하는데 **타임아웃도, 최대 하강 거리 제한도 없다.**
- 잡은 물체를 도중에 떨어뜨렸거나, place 지점 아래에 지지면이 없으면 로봇이 힘제어로 계속 내려감.
- 유일한 탈출구가 `_emergency` 플래그(음성 stop → 여러 홉 경유)뿐.
- 추가 버그: emergency 분기에서 `release_force`/`release_compliance_ctrl`을 호출하고 break한 뒤, 루프 밖에서 **같은 해제 함수를 또 호출**(이중 해제 — DSR API에 따라 오류 발생 가능).

**개선**: `시작 z - 현재 z > 한계값` 또는 경과 시간 타임아웃을 루프 탈출 조건에 추가하고, 해제 로직은 `finally` 한 곳으로 모은다.

### 2. stop이 "진짜 정지"가 아님 — `robot_motion.py:97~114`, `robot_arm_node.py` **[해결됨 2026-07-12]**
~~`safe_stop()`은 소프트 플래그만 세워 이미 실행 중인 모션은 끊지 못했다.~~
**적용한 수정**: robot_arm_node에 `/{ROBOT_ID}/motion/move_stop`(`dsr_msgs2/srv/MoveStop`) client를 추가하고, `safe_stop_robot()`에서 플래그 설정 후 `stop_mode=1`(DR_QSTOP, Quick stop)로 `call_async` 호출. DSR 파이썬 래퍼를 거치지 않고 컨트롤러를 직접 호출하므로 movel 블로킹과 스레드 충돌이 없다(이전 프로젝트 cobot1_system에서 검증된 패턴). package.xml에 `dsr_msgs2` exec_depend 추가.
- 남은 유의점: 음성 "멈춰"의 경로 지연(녹음 3초 + STT/GPT 왕복)은 그대로이므로, **진짜 비상정지는 물리 E-stop**이라는 점은 여전히 유효. 실기에서 movel 도중 stop이 실제로 끊기는지 확인 필요.

### 3. 새 작업 시작 시 E-stop 자동 해제 — `task_manager_node.py:275, 619`
`start_workspace_detection()`과 `handle_start_organize()`가 시작하자마자 `SAFETY_COMMAND_CLEAR`를 자동 발행 → safety_node가 `/emergency_stop: False` 발행 → robot_arm이 `clear_stop()`.
즉 **사용자가 "멈춰"라고 한 직후 "정리 시작해줘"라고만 하면 정지 상태가 소리 없이 풀린다.** 정지 원인이 해소됐는지 아무도 확인하지 않는다.

**개선**: E-stop 해제는 명시적 clear 명령(별도 음성/HMI 버튼)으로만 하고, e-stop 활성 중 새 작업 명령은 "정지 상태입니다. 해제 후 다시 시도하세요"로 거부.

### 4. 물체 파지 도중 정지/취소 시 복구 시나리오 없음 — `robot_arm_node.py:125~130`
cancel/E-stop이 오면 남은 movel이 전부 스킵되는데, **그리퍼가 물체를 문 채로 공중에 정지**할 수 있다. 재시작 시 물체를 든 상태인지 아닌지 시스템이 모른다.

## ② 시스템이 멈추거나 죽는 시나리오

### 5. `is_busy` 영구 고착 (워치독 없음) — `task_manager_node.py`
`is_busy=True`가 된 뒤 응답이 영영 안 오면 시스템 전체가 BUSY 응답만 하게 된다. 가능한 구멍:
- `/start_workspace_scan` 서비스 호출 후 robot_arm이 movej에서 hang → future가 완료 안 됨 (스캔 서비스는 응답까지 3자세 전체를 동기 실행하는 구조라 최대 45초+).
- `robot_organize_goal_response_callback`(513행)에서 `future.result()` 예외 미처리 → 콜백이 죽고 `finish_current_task()` 호출 안 됨.

**개선**: 작업 시작 시 타임아웃 타이머(예: 90초)를 걸고, 만료 시 강제 `finish_current_task()` + 사용자 안내. goal response 콜백에 try/except 추가.

### 6. 음성 노드가 예외 한 번에 죽음 — `command_input_node.py:246~267`
`stt.speech2text()`(OpenAI 네트워크 오류), `command_classifier.classify()`(GPT 호출)가 예외를 던지면 main의 `except KeyboardInterrupt` 밖으로 나가 **command_input_node 프로세스가 종료**된다. 데모 중 Wi-Fi 순단 한 번이면 음성 입력 전체가 죽는다. **실전에서 가장 먼저 터질 지점.**

**개선**: `process_voice_command_once()` 안에서 try/except로 감싸고 "일시적인 오류입니다. 다시 시도해주세요" TTS 후 대기 루프로 복귀.

### 7. depth/카메라 없으면 무한 대기 — `detection.py:562~570`
`_wait_for_valid_data()`는 데이터가 올 때까지 무한 spin. RealSense가 안 떠 있으면 노드 초기화(intrinsics 대기)에서 영원히 블록되고, 스캔 중 depth가 끊기면 `/scan_capture_done` ack가 안 나가 robot_arm 15초 타임아웃으로 이어진다.

**개선**: 최대 재시도 횟수/시간을 두고 실패 시 명확한 에러 로그 + 해당 스캔 실패 처리.

### 8. 스캔 ack 인덱스 미확인 레이스 — `robot_arm_node.py:365~367`
`scan_done_callback`은 어떤 인덱스의 ack든 무조건 `event.set()`. 이전 스캔이 타임아웃으로 실패한 뒤 **늦게 도착한 ack**가 다음 스캔의 대기를 조기 해제할 수 있다 → 로봇이 detection 캡처 완료 전에 다음 자세로 이동 → 잘못된 자세에서 촬영된 데이터 사용.

**개선**: `if msg.data == 기다리는 index: event.set()`으로 인덱스를 검사 (`_scan_ack_index` 변수가 이미 있는데 안 쓰고 있음).

### 9. 그리퍼 연결 실패가 조용히 넘어감 — `onrobot.py:29~31`, `robot_motion.py:140`
`ModbusClient.connect()` 반환값을 확인하지 않아 그리퍼 IP가 틀려도 `connect()`가 성공한 것처럼 지나가고, 첫 `grip_open()`에서야 예외/무동작이 발생한다. 또 `pymodbus.client.sync` import는 **pymodbus 2.x 전용**(3.x면 ImportError → robot_arm 전체 불능). requirements에 `pymodbus<3` 고정 필요.

## ③ 동작이 틀릴 수 있는 로직 문제

### 10. 같은 클래스 물체 다중 인스턴스 미지원 — `detection_utils.py:215~239` **[변경됨 — 부분 개선]**
~~기존: 이름 기준으로 클라우드를 vstack 병합 → hammer 2개면 grasp 중심이 두 물체 사이 허공.~~
**d813d97 이후**: `merge_clouds_by_name()`이 클라우드를 합치지 않고 **bbox면적×confidence 최고 자세 하나만 선택**한다. 허공 파지 위험은 사라졌지만, 여전히 **이름 기준 그룹핑이라 클래스당 1개만 처리**된다.
주의할 점: 새 yolo.py/rtdetr.py는 track 집계로 **2D 수준에서는 동일 클래스 2개를 분리**하는데(같은 frame의 bbox 2개 = 다른 track), 3D 병합 단계(`merge_clouds_by_name`)에서 다시 이름으로 뭉개져 **나머지 인스턴스는 조용히 무시**된다. 2D는 2개 감지 → 최종 결과는 1개라 로그만 봐서는 헷갈리기 쉽다.

**개선**: base 좌표 거리 기반 클러스터링(예: 반경 80mm)으로 인스턴스를 분리해 이름+클러스터 단위로 처리하거나, 최소한 "동일 클래스 2개 이상 감지 시 경고 발행".

### 11. zone 좌표가 두 파일에 중복 정의 — `workspace_judge_utils.py:24` vs `grid_allocator.py:38`
`DEFAULT_ZONES`(판정용)와 `ZONES`(그리드 배치용)에 같은 실측 좌표가 복붙되어 있다. **한쪽만 보정하면 "판정상 정상인데 배치 위치는 구역 밖" 같은 미묘한 어긋남**이 생긴다.

**개선**: grid_allocator가 workspace_judge_utils의 zone 정의를 import(또는 공용 config 파일)하도록 단일화.

### 12. 그리드 배치 실패 시 zone 중앙에 '무조건' 놓음 — `workspace_judge_utils.py:319~321`
`place_failed=True`인 물체도 misplaced 목록에 남고 fallback place_position(zone 중앙)으로 로봇이 실제로 옮긴다. **중앙이 이미 점유돼 있어도 충돌 검사 없이 그 위에 놓을 수 있다.**

**개선**: `place_failed=True`는 로봇 실행 목록에서 제외하고 "OO는 놓을 자리가 없어 정리하지 못했습니다"로 사용자에게 안내.

### 13. TTS 안내음이 STT 녹음에 섞임 — `command_input_node.py:247~250` + `tts.py`
tts.py가 edge-tts(비동기, 백그라운드 스레드)로 바뀌면서 `speak()`이 즉시 리턴한다. 합성에 네트워크 왕복이 걸리므로 "동작을 말씀해주세요" **재생이 1초 대기 후 시작되는 3초 녹음과 겹칠 수 있다** → 로봇 목소리가 STT에 입력되어 오인식. (RATE +80%로 빨라졌어도 근본 해결이 아님.)

**개선**: `_synthesize_and_play`가 재생 완료 이벤트(threading.Event)를 set하게 하고, `wait_after_tts`에서 그 이벤트를 timeout과 함께 기다린다.

### 14. STT 녹음 3초 고정 — `stt.py:40`
"잘못 배치된 물건 치워줘" 같은 긴 문장은 잘릴 수 있다. 주석은 5초라는데 실제 3초(문서 불일치). VAD(무음 감지 종료) 또는 4~5초로 조정 권장. 임시 wav 파일도 `delete=False` 후 삭제하지 않아 /tmp에 누적된다.

### 15. `PICK_Z_OFFSET_MM` 이중 정의 — `robot_motion.py:43` vs `robot_motion.py:300~303`
모듈 상수(d813d97에서 33.0→**22.0**으로 변경됨)는 여전히 **죽은 값**이고, 실제로는 함수 안에서 z>60이면 45, 아니면 30으로 하드코딩 분기한다. 병합에서 상수를 22로 튜닝했지만 **함수 내 분기가 우선이라 실제 동작에는 아무 영향이 없다** — 튜닝이 무효가 되고 있는 상태. → 분기 임계값과 오프셋을 모듈 상단 상수로 올리고 함수 내 재정의 삭제.

### 16. place 높이가 pick 높이를 따라감 — `robot_motion.py:294~296`
`place_pose`의 z에 `pick_position['z']`를 사용한다. grid_allocator가 계산한 `place_z(14mm)`는 무시된다. force_down()으로 내려놓기 때문에 현재는 동작하지만, force_down이 실패하는 상황(문제 1)과 결합하면 위험하다.

### 17. `yolo.py PCI_MAP()`에 self 누락 — **[해결됨]**
d813d97 병합에서 yolo.py가 전면 재작성되며 `@staticmethod`로 수정되고 2D 이미지 가드(`len(shape) != 3`이면 그대로 반환)도 추가되었다. `get_best_detection()`에서의 호출도 제거됨.

### 18. YOLO confidence 0.8 컷 — `yolo.py:64` **[해결됨 2026-07-12]**
~~`confidence_threshold=0.80` 기본값이라 0.8 미만 물체는 "감지된 물체가 없습니다"로 흐른다.~~
**적용한 수정**: 기본값을 **0.60**으로 하향. 순간 오검출은 frame_support 반복 감지 필터(최소 2프레임/10%)가 걸러주므로 recall을 올려도 안전하다. 실기에서 오검출이 늘면 0.65~0.7로 되올리며 튜닝.
- 남은 유의점: ensemble의 `rtdetr_only_conf_threshold`는 0.80 그대로(이번 수정 범위 아님) — RT-DETR 단독 보완 후보는 여전히 엄격하다.

### 19. 재검증 무한 순환 가능성 — `task_manager_node.py:547~562`
정리 성공 → 재검증 → 아직 misplaced 남음 → 사용자 "정리 시작" → 또 남음 → ... 반복 횟수 제한이나 "같은 물체가 2번 연속 실패하면 수동 처리 요청" 같은 로직이 없다. 파지 실패가 반복되는 물체에서 무한 루프성 사용자 경험이 된다.

## ④ 성능/자원

### 20. 유휴 상태에서도 0.2초마다 YOLO 추론 — `detection.py:67, 138`
preview 타이머가 시스템이 아무것도 안 할 때도 초당 5회 추론을 돌린다. GPU/CPU를 상시 점유하고, **스캔 콜백과 같은 single-thread executor를 공유하므로 스캔 처리를 지연**시켜 robot_arm의 15초 ack 타임아웃 원인이 될 수 있다.
**개선**: 스캔 중에는 preview 일시정지, 유휴 시 주기를 0.5~1초로 완화(ROS 파라미터화).

### 21. wakeword 모델을 매번 재로드 — `wakeup_word.py:59~61`
`set_stream()`이 호출될 때마다 openWakeWord `Model`을 새로 생성한다. 명령 1회 처리 후 대기 재진입 시마다 tflite 로드가 반복됨 → 생성자에서 1회만 로드.

### 22. 마이크 OSError 시 무한 재시도 — `command_input_node.py:209~211, 280~284`
`wait_for_wakeup()`이 False를 반환하면 main 루프가 즉시 다시 open_stream을 시도한다. 마이크가 없으면 로그 폭주 + CPU 낭비. 재시도 간 sleep/최대 횟수 필요.

## ⑤ 빌드 · 설정 · 문서

### 23. `.env` 없으면 colcon build 실패 — `setup.py:44~46`
`data_files`에 `resource/.env`가 명시돼 있는데 `.gitignore`가 `.env`를 제외하므로 **새로 클론한 팀원은 빌드부터 실패**한다. → `.env.example`을 커밋하고, setup.py에서 `os.path.exists()` 조건부로 포함.

### 24. 경로 하드코딩이 옛 워크스페이스 기준 — `detection.py:194`, `db_node.py:128~129`
scan 이미지 저장 후보와 DB 기본 경로가 `~/a4_cobot2_ws/...`인데 현재 워크스페이스는 `~/ws_cobot2_pjt`. detection은 후보 부모 폴더가 없으면 cwd 기준으로 흘러가 launch 실행 위치에 따라 저장 위치가 바뀐다. 환경변수/파라미터 기본값을 현 워크스페이스 기준으로 정리.

### 25. 문서·주석 불일치 (혼란 유발) **[대부분 해결됨]**
- ~~TTS 섹션 tts2/tts3~~ → 문서 위쪽 갱신 완료 (tts.py = edge-tts 단일).
- `README.md`: 초기 버전(get_3d_position, 테스트 모션) 기준이라 현재 구조(3자세 스캔, 그리드 배치, VLM 보고, rosbag)와 크게 다름. **(여전히 유효)**
- ~~detection.py 기본값 불일치~~ → **[해결됨]** d813d97에서 기본값이 실제로 `model_name="ensemble"`로 변경되어 헤더 주석과 일치. (단, 여전히 코드 상수라 ROS 파라미터화는 개선 여지)
- `workspace_judge_node.py` 헤더: "diablo/..." 복붙 흔적. **(여전히 유효)**
- ~~yolo v2/v4 의문~~ → **[해결됨]** yolo.py가 `yolo_seg_best_v4.pt`를 사용하고 v2 파일은 저장소에서 삭제됨.

### 26. 기타 사소한 것
- `db_node.py` 성공 판정 휴리스틱: `FAILURE_KEYWORDS`에 'remaining', 'busy', 'cancel' 포함 → "정상 종료했지만 물건이 남음"도 실패로 기록되어 통계가 왜곡될 수 있음. **(여전히 유효)**
- ~~bag_records는 메타데이터만 기록~~ → **[해결됨]** d813d97에서 `rosbag_manager.py`가 추가되어 작업 시작/종료에 맞춰 실제 `ros2 bag record` subprocess를 실행하고, SIGINT→SIGTERM→SIGKILL 순 종료 + metadata.yaml/.db3 파일 검증 후에만 DB에 연결한다. 잘 만들어진 구현.
- `grid_allocator.py`의 `__main__` 자가테스트는 좋은 습관 — pytest로 승격해 zone 중복 정의(문제 11) 회귀 테스트로 쓰면 좋음.
- 이 파일(cobot.md)은 `.gitignore`에 있지만 이미 git 추적 중이라(04930ef에 커밋됨) 실제로는 커밋에 포함됨.

## ⑥ d813d97 병합 이후 변경·신규 이슈 (2026-07-11 밤 갱신)

병합된 커밋 6090994("angle, hmi 수정 ensemble, ros2 bag 추가")의 주요 변경과 그에 따른 새 관찰 사항.

### 병합에서 바뀐 것 요약
| 영역 | 변경 |
|---|---|
| yolo.py | 전면 재작성: v4 모델, frame_index 기반 **track 집계**(동일 클래스 다중 인스턴스 2D 분리, 최소 2프레임/10% 반복 감지 필터), PCI_MAP 수정 |
| rtdetr.py | 같은 track 집계 로직 추가 (conf 0.50, 반복 감지 필터) |
| sam2_refiner.py | `last_reason` 실패 원인 추적 추가 (sam2_disabled / no_mask_returned / mask_rejected 등) |
| ensemble_detector.py | rtdetr_only 임계 0.60→0.80, frame_support 메타데이터 보존, 로그 요약화(detailed_debug 플래그) |
| detection.py | **기본 detector가 "ensemble"로 변경**, 좌표 debug 로그 `DETECTION_COORD_DEBUG=False` 봉인, 자세별 로그 요약 |
| detection_utils.py | `merge_clouds_by_name`: 3자세 vstack 병합 → **최고 점수 자세 1개의 cloud만 선택** |
| robot_motion.py | `PICK_Z_OFFSET_MM` 33→22 (여전히 무효, 문제 15), 파지 각도 `rot = object_angle`(±90 접기 제거) |
| task_manager | `/workspace_judgement`(TRANSIENT_LOCAL) 발행 추가, no_objects도 빈 결과 발행 |
| HMI | `/workspace_judgement` 구독 → 실제 스캔 결과로 테이블 갱신(PLACED/MISPLACED, tidy 중복 dedup), 스캔 시작 시 테이블 클리어 |
| DB | `rosbag_manager.py` 신규(실제 bag 녹화 + 검증), db_node 파라미터화(enable_rosbag 등), launch에 파라미터 블록 추가 |

### 신규 이슈 27. 3자세 스캔의 의미 약화 — `detection_utils.py:215~239`
best-pose 단일 선택으로 바뀌면서 **3자세는 이제 "커버리지 보완"이 아니라 "가장 잘 보인 자세 고르기"** 역할만 한다. 윗면 중심(compute_top_center_grasp)이 단일 시점 클라우드 기준이 되어, 어느 자세가 선택되느냐에 따라 parallax 편향 방향이 달라진다(특히 키 큰 병). 문서 상단 "알려진 한계"의 시차 문제와 결합해 재튜닝 필요. 의도된 트레이드오프라면 스캔을 3자세에서 줄이는 것도 검토 가능(스캔 시간 단축).

### 신규 이슈 28. 파지 각도 ±90 접기 제거 — `robot_motion.py:275`
`rot = object_angle - 90 if ... else + 90` → `rot = object_angle`로 변경(주석처리). top_face_angle은 -90~90을 반환하므로 rot 범위가 그대로 -90~90이 된다. 기존 접기는 그리퍼 2지 대칭성을 이용해 손목 회전량을 최소화하는 취지였는데 제거되어 **최대 90° 더 회전**할 수 있다. 실기 튜닝 중의 변경으로 보이므로, 조인트 한계 근처 자세에서 문제가 없는지 확인 필요.

### 신규 이슈 29. rosbag 시작/종료가 db_node 콜백을 블록 — `rosbag_manager.py:170, 220~240`
`start()`는 0.7초 sleep, `stop()`은 flush 0.25초 + 최악 SIGINT 12초 + SIGTERM 3초 + SIGKILL 3초까지 **동기 블록**한다. db_node는 single-thread executor라 그 사이 다른 토픽 이벤트를 처리하지 못한다(구독 큐 depth 10 초과분 유실 가능). 로깅 전용 노드라 시스템 흐름은 안 막지만, 연속 명령 시 다음 run의 명령/상태 기록이 늦거나 빠질 수 있다. → stop을 별도 스레드에서 실행하거나 timeout을 3~5초로 축소 검토.

### 신규 이슈 30. rosbag이 카메라 원본을 매 작업마다 녹화 — 디스크 용량
bag_topics에 `/camera/camera/color/image_raw` + aligned depth(30fps, 비압축)가 포함되어 **작업 1회당 수 GB** 수준으로 커질 수 있다. 정리 작업 하나가 1~2분이면 금방 수십 GB. → compressed image topic 사용, 오래된 bag 자동 삭제(보존 개수/용량 상한), 또는 카메라 토픽 제외 옵션.

### 신규 이슈 31. HMI 테이블의 상태 표시 부정확 — `hmi_interface.py`
- `unknown_rule_objects`(배치 규칙 없는 물체)는 expected_zone이 없어 **MISPLACED(빨강)** 로 표시된다 — 실제로는 "규칙 미정"이므로 UNKNOWN 같은 3번째 상태가 정확.
- tidy(재정렬) 항목은 current==expected라 **PLACED(초록)** 로 표시되는데, 실제로는 로봇이 옮길 대상이다(dedup 의도는 주석에 있으나 사용자에겐 "정리할 게 없어 보이는" 착시).

### 해결된 항목 정리 (병합 덕분)
- 문제 17 (PCI_MAP self 누락) → 해결
- 문제 27번 항목이었던 debug 로그 노이즈 → `DETECTION_COORD_DEBUG` 플래그로 해결
- 문제 25 중 detection 기본값/모델 v2·v4 불일치 → 해결
- 문제 26 중 "bag은 메타데이터만" → 실녹화 구현으로 해결
- 문제 10의 "허공 파지" 형태 → best-pose 선택으로 완화 (다중 인스턴스 무시로 형태 변경)
- HMI가 스캔 결과와 무관한 고정 테이블을 보여주던 것 → 실데이터 연동으로 해결

---

## 우선순위 요약 (d813d97 병합 반영)

| 순위 | 항목 | 파일 | 유형 | 상태 |
|---|---|---|---|---|
| 1 | force_down 무한 루프 + 이중 해제 | robot_motion.py | 안전 | 유효 |
| 2 | stop이 실행 중 모션을 못 끊음 (move_stop 미호출) | robot_arm_node.py | 안전 | **해결됨 07-12** |
| 3 | 새 작업 시 E-stop 자동 해제 | task_manager_node.py | 안전 | 유효 |
| 4 | STT/LLM 예외 → 음성 노드 사망 | command_input_node.py | 크래시 | 유효 |
| 5 | is_busy 영구 고착 (워치독 없음) | task_manager_node.py | 멈춤 | 유효 |
| 6 | 스캔 ack 인덱스 미확인 레이스 | robot_arm_node.py | 동작 오류 | 유효 |
| 7 | zone 좌표 이중 정의 | workspace_judge_utils / grid_allocator | 동작 오류 | 유효 |
| 8 | TTS 안내음이 녹음에 혼입 | command_input_node.py + tts.py | 동작 오류 | 유효 |
| 9 | PICK_Z_OFFSET 상수 튜닝이 무효 (함수 내 하드코딩 우선) | robot_motion.py | 동작 오류 | 유효·병합에서도 반복됨 |
| 10 | 동일 클래스 다중 인스턴스: 2D는 분리, 3D에서 1개만 처리 | detection_utils.py | 동작 오류 | 변경됨(허공 파지→무시) |
| 11 | rosbag stop 최대 ~18초 동기 블록 / 카메라 원본 녹화 용량 | rosbag_manager.py, launch | 운영 | 신규 |
| 12 | .env 빌드 의존 / 경로 하드코딩 / README 갱신 | setup.py 외 | 운영 | 유효 |

**병합으로 해결**: PCI_MAP self 누락, debug 로그 노이즈, detection 기본값·모델 버전 불일치, bag 실녹화 미구현, HMI 고정 테이블.
