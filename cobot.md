# a4_cobot2 — 음성 명령 기반 작업공간 정리 협동로봇

두산 협동로봇(m0609) + RealSense(eye-in-hand) + 딥러닝 비전 + LLM으로, **자연어 음성 명령을 받아 흩어진 물체를 인식하고 → 올바른 구역으로 집어 정리하는** ROS2 시스템.

> "작업공간 확인해줘" 한마디에 **스캔 → 판정 → 음성 안내**가, "정리 시작해줘" 한마디에 **pick-and-place → 재검증 → VLM 최종 보고**가 사람 개입 없이 이어진다.

## 이 프로젝트의 특징 한눈에

단순히 "YOLO로 물체 잡는 로봇"이 아니라, **현실에서 부딪히는 문제를 하나씩 개념으로 풀어낸** 것이 핵심이다. 아래는 그 특징을 압축한 것이고, 자세한 "문제 → 접근 개념 → 구현"은 [핵심 기술 특징](#핵심-기술-특징-구현-개념) 절에 담았다.

| 영역 | 부딪힌 문제 | 우리가 택한 개념 |
|------|-------------|------------------|
| **음성 이해** | "멈춰"·"그만"·"정지해"… 표현이 제각각 | 키워드 매칭 대신 **LLM(GPT-4o) 의도 분류**로 자연어를 고정 명령으로 정규화 |
| **파지점(grasp)** | 옆에서 본 물체는 grasp 점이 **옆면**으로 잡힘 | point cloud에서 **물체 윗면(top surface)의 중심**을 open3d로 추출 |
| **넓은 작업공간** | 카메라 한 프레임에 작업대가 다 안 들어옴 | **eye-in-hand 3자세 스캔** + 자세별 `camera→base` 4×4 변환으로 좌표 통합 |
| **파지 각도** | 길쭉한 물체를 아무 방향으로 잡으면 미끄러짐 | segmentation mask **PCA로 긴 축**을 구해 그리퍼 손목 회전에 반영 |
| **오검출** | 한두 프레임에서 튀는 순간 오검출 | 1초간 다중 프레임 집계 → **반복 감지된 것만 채택**(frame_support 필터) |
| **인식 정확도** | 단일 모델의 한계(누락·거친 mask) | **YOLO-seg + RT-DETR + SAM2.1 앙상블**(후보·검증·mask 정밀화 역할 분담) |
| **놓기 위치** | 오배치 물체를 한 점에 다 놓으면 겹침 | 구역을 **격자로 나눠 빈 슬롯을 찾는 선반형 배치**(grid allocator) |
| **부드러운 안착** | 위치 제어만으로 놓으면 충돌/헛놓기 | **힘 제어(compliance + 외력 감지)** 로 접촉을 느끼며 내려놓음 |
| **진짜 정지** | `movel`은 블로킹이라 소프트 stop으론 못 끊음 | soft 플래그 + **컨트롤러 `move_stop`(QSTOP) 직접 호출** 이중 정지 |
| **흐름 제어** | 비동기 명령이 겹치면 로봇이 두 일을 동시에 | **`is_busy` 직렬화 + `current_task` 세대 검증**으로 지연 응답까지 차단 |
| **결과 설명** | 좌표 JSON만으론 사용자가 이해 못 함 | 재검증 사진+판정을 **VLM(GPT-4o)** 에 넣어 자연어 보고문 생성 |
| **재현·디버깅** | 데모에서 무슨 일이 있었는지 추적 어려움 | 작업 단위로 **`ros2 bag` 자동 녹화 + SQLite 이력** 저장 |
| **견고성** | 부품 하나(VLM/TTS/마이크) 죽으면 전체 정지 | 어디가 없어도 **규칙 기반으로 격하(graceful degradation)** 해 계속 동작 |

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

## 핵심 기술 특징 (구현 개념)

각 기능이 "어떤 문제를 어떤 아이디어로 풀었는지"를 코드 근거와 함께 정리한다. 이 절이 이 프로젝트가 다른 실습 로봇과 다른 지점이다.

### 1. 자연어 음성을 "의도(intent)"로 정규화 — 키워드 매칭이 아니라 LLM 분류

**문제.** "정리해줘", "잘못된 물건 치워줘", "그만", "정지해"처럼 같은 의도를 사람마다 다르게 말한다. `if "정지" in text` 식 키워드 매칭은 표현이 조금만 달라져도 놓치고, 무관한 말("오늘 날씨 어때?")에도 오작동한다.

**접근.** STT 원문을 **GPT-4o에 넣어 사전 정의된 라벨**(`check_workspace` / `start_organize` / `stop` / `shutdown` / `unknown`) 중 하나로 분류한다. few-shot 예시로 출력을 라벨 하나로 강제하고, `temperature=0`으로 결정성을 확보한 뒤, **LLM이 규격 밖 문자열을 뱉으면 코드에서 화이트리스트로 다시 걸러 `unknown` 처리**한다(프롬프트만 믿지 않는다).

**구현.** [command_classifier.py](a4_cobot2/voice/command_classifier.py#L33) — `ChatOpenAI(model="gpt-4o", temperature=0)`, LangChain `PromptTemplate | ChatOpenAI` 체인. 음성 입력 전체 흐름은 **웨이크워드("hello rokey", openWakeWord tflite, 48kHz→16kHz 리샘플 후 2단계 임계) → 3초 녹음 → Whisper STT(`whisper-1`) → GPT-4o 분류 → `/task_command` 발행**이다([command_input_node.py](a4_cobot2/voice/command_input_node.py#L246)). 응답 음성(TTS)은 edge-tts로 백그라운드 비동기 재생해 메인 루프를 막지 않는다([tts.py](a4_cobot2/voice/tts.py#L36)).

### 2. Grasp 포인트 = "물체 윗면의 중심" (open3d point cloud) ★

**문제.** 카메라가 그리퍼에 달린 eye-in-hand 구성이라 물체를 옆에서/비스듬히 본다. bbox 중심 픽셀 하나의 depth만 쓰면 grasp 점이 **물체 옆면**으로 잡혀 탑다운 파지가 어긋난다(특히 키 큰 병).

**접근.** mask 안의 **모든 픽셀을 depth와 결합해 3D point cloud를 만들고**, 그 cloud에서 **윗면(top surface)의 중심점**을 파지 좌표로 쓴다. 윗면은 무거운 평면 세그멘테이션(RANSAC) 대신 **base Z축 기준 상위 슬라이스**로 뽑는다:
1. open3d `remove_statistical_outlier`로 튀는 점(flying pixel) 제거
2. `z_top = np.percentile(z, 95)` — raw max가 아닌 **95 percentile robust max**로 노이즈 방지
3. `z_top − 18mm` 이내의 상위 점들만 "윗면"으로 슬라이스
4. `voxel_down_sample(3mm)`로 픽셀 밀집 편향 제거 후 **평균 → 윗면 중심 (grasp x, y)**, 접근 높이 `grasp_z = z_top`

**구현.** [detection_utils.py `compute_top_center_grasp()`](a4_cobot2/object_detection/detection_utils.py#L243). mask→base(mm) 역투영은 [deproject_mask_to_base()](a4_cobot2/object_detection/detection_utils.py#L178). 라이브러리: **open3d, numpy**. 위치는 윗면 중심을 쓰되 **각도는 물체 전체 발자국(손잡이+머리)의 PCA**로 따로 계산해(긴 축이 더 안정적) 역할을 분리한 것도 포인트다([detection.py](a4_cobot2/object_detection/detection.py#L538)).

### 3. eye-in-hand 3자세 스캔 & camera→base 좌표 통합

**문제.** 카메라 한 대로는 넓은 작업대가 한 프레임에 안 들어오고, 카메라 좌표계 결과만으론 로봇이 파지할 수 없다.

**접근.** 로봇을 **3개 관측 자세(중앙/좌/우)** 로 옮겨가며 스캔한다. 각 자세에서 현재 로봇 pose로 `base←camera` 4×4 변환행렬을 만들어 mask cloud를 **base(mm) 좌표로 통합**한다. 변환은 `base←gripper`(로봇 posx를 ZYZ 오일러로 해석) `@` `gripper←camera`(`T_gripper2camera.npy` 캘리브레이션)로 합성한다.

**구현.** 스캔은 **robot_arm이 주도**한다: 자세 이동 → 변환행렬 계산 → `/scan_pose_transform`(`[index, total, 4×4]`) 발행 → detection의 캡처 완료 ack(`/scan_capture_done`)를 `threading.Event`+타임아웃으로 기다린 뒤 다음 자세로([robot_arm_node.py](a4_cobot2/robot_arm/robot_arm_node.py#L409), [robot_motion.py `get_base_to_camera_matrix()`](a4_cobot2/robot_arm/robot_motion.py#L163)). 3자세 결과는 **vstack으로 합치지 않고**, `bbox면적 × confidence`가 가장 높은 **한 자세의 cloud만 선택**(best-pose)해 position/angle/size를 그 자세 기준으로 쓴다([merge_clouds_by_name()](a4_cobot2/object_detection/detection_utils.py#L217)).

### 4. 파지 각도 = segmentation mask의 PCA 긴 축

**문제.** 길쭉한 물체(드라이버·망치)를 아무 방향으로 잡으면 미끄러지거나 못 잡는다. 그리퍼를 물체 긴 축에 맞춰 돌려야 한다.

**접근.** base cloud를 **XY 평면에 투영**하고 공분산 고유분해(PCA)로 **긴 축(major axis)** 을 구해 `[-90°, 90°]` 각도로 변환, 파지 시 그리퍼 손목 회전(rz)에 반영한다. 같은 축의 짧은 방향을 파지 폭으로 삼아 그리퍼 벌림폭도 물체 크기에 맞춘다.

**구현.** [detection_utils.py `top_face_angle()`](a4_cobot2/object_detection/detection_utils.py#L274) (`np.cov` → `np.linalg.eigh` → `atan2`), 크기는 [footprint_extent()](a4_cobot2/object_detection/detection_utils.py#L300)에서 major/minor 축 투영의 2~98 percentile로 계산. robot_motion이 `object_angle`을 그리퍼 자세 rz에 더하고, `object_width + 여유`만큼만 벌려 인접 물체를 피한다([robot_motion.py](a4_cobot2/robot_arm/robot_motion.py#L262)).

### 5. 오검출 제거 = "여러 프레임에서 반복 감지된 것만" (frame_support 필터)

**문제.** 단일 프레임 추론은 한두 프레임에서만 튀는 순간 오검출을 그대로 통과시킨다.

**접근.** 1초간 여러 프레임을 모아 **IoU 기반으로 같은 track끼리 묶고**, 여러 프레임에서 반복 감지된 track만 남긴다. 필요 감지 수는 `max(2프레임, 전체의 10%)`. 이때 **같은 프레임의 동일 클래스 bbox 2개는 같은 track에 못 들어가게** 막아, 실제로 같은 종류 물체가 여러 개면 서로 다른 track으로 분리 유지한다. 이 필터가 있어서 YOLO confidence 컷을 0.80→**0.60**으로 낮춰 recall을 올려도 안전하다(순간 오검출은 필터가 거른다).

**구현.** [yolo.py `_aggregate_detections()`](a4_cobot2/object_detection/yolo.py#L402), [rtdetr.py `_aggregate_across_frames()`](a4_cobot2/object_detection/rtdetr.py#L244) — 동일 패턴. `frame_support`/`frame_ratio` 메타데이터는 앙상블·로그까지 보존된다.

### 6. 인식 앙상블 — YOLO-seg + RT-DETR + SAM2.1 (역할 분담)

**문제.** 단일 모델은 누락이 있고, YOLO mask 경계는 거칠며, RT-DETR은 mask를 못 만든다.

**접근.** 세 모델을 **역할을 나눠** 결합한다.
- **YOLO-seg**: class/bbox + **mask 후보** 생산(가중치 0.60)
- **RT-DETR**: 프레임 집계된 bbox/class로 **검증 + YOLO 누락 보완**(가중치 0.40, mask 없음)
- **SAM2.1**(파인튜닝): 최종 bbox를 box-prompt로 넣어 **mask 정밀화(refinement)**

같은 class + `IoU ≥ 0.50`이면 confidence 가중 평균으로 병합, YOLO가 놓쳤지만 RT-DETR이 높은 confidence로 반복 감지한 후보는 보완 추가한다. SAM2 mask는 면적·bbox 포함률 게이트를 통과해야 채택되고, 실패하면 YOLO mask로 fallback한다.

**구현.** [ensemble_detector.py](a4_cobot2/object_detection/ensemble_detector.py#L61), [sam2_refiner.py](a4_cobot2/object_detection/sam2_refiner.py#L25). 라이브러리: ultralytics(`YOLO`, `RTDETR`), sam2(`build_sam2`) + torch.

### 7. 구역(zone) 판정 & 겹치지 않는 그리드 배치

**문제.** "이 물체가 제자리에 있나?"를 판단하고, 오배치 물체를 옮길 위치를 정해야 한다. 오배치 물체 여러 개를 구역 중앙 한 점에 다 놓으면 서로 겹친다.

**접근.**
- **판정**: 작업공간을 base 좌표계에서 4개 구역(red/blue/green/yellow)의 **축정렬 박스(AABB)** 로 정의하고, 클래스→구역 매핑과 비교해 `정상 / 오배치 / 규칙없음` 3분류한다([workspace_judge_utils.py](a4_cobot2/workspace/workspace_judge_utils.py#L99)).
- **배치**: 구역을 **20×20 격자**로 나누고, 코너에서 시작해 x축으로 한 줄씩 채우는 **선반형(shelf) 패킹**으로 빈 슬롯을 찾는다. 물체 크기 + 그리퍼 clearance만큼 셀을 점유하고, **이미 놓인 물체(정상+오배치 모두)** 를 먼저 점유 표시해 그 위에 놓지 않는다. 배치가 불가능할 때만 구역 중앙으로 fallback한다([grid_allocator.py](a4_cobot2/workspace/grid_allocator.py#L110)).
- 정상 배치지만 슬롯에서 벗어난 물체를 다시 줄 세우는 **재정렬(tidy)** 도 있고, 실행 순서는 [오배치 이동 먼저 → 재정렬 나중]으로 보장한다.

### 8. 힘 제어로 "느끼며" 내려놓기

**문제.** 위치 제어만으로 놓으면 바닥/물체 높이 오차 때문에 충돌하거나 공중에서 놓는다.

**접근.** 놓기 단계에서 **task-space 순응 제어(compliance)** 를 켜고 Z축으로 하향력(−30N)을 주며 내려가다가, **외력 Z 성분이 임계값(6N)을 넘으면**(접촉) 정지하고 힘·순응 제어를 해제한다.

**구현.** [robot_motion.py `force_down()`](a4_cobot2/robot_arm/robot_motion.py#L204) — DSR `task_compliance_ctrl` / `set_desired_force` / `get_tool_force`. (집기가 아니라 내려놓기에 적용.)

### 9. "진짜 정지" — soft 플래그 + 컨트롤러 QSTOP 직접 호출

**문제.** `movel`/`movej`는 블로킹 호출이라, 소프트 stop 플래그만으로는 **이미 실행 중인 모션을 끊지 못한다**(다음 모션만 막음). DSR 파이썬 래퍼로 정지를 넣으면 블로킹 중인 모션 스레드와 충돌할 수 있다.

**접근.** 정지를 이중화한다. ① `_emergency` **soft 플래그**로 다음 movel/movej 진입 차단 + pick-and-place 단계 사이 조기 return. ② 컨트롤러의 `/{robot}/motion/move_stop`(`dsr_msgs2/MoveStop`)을 `call_async`로 **직접 호출**, `stop_mode=1`(DR_QSTOP, Quick Stop)로 진행 중 모션을 즉시 물리 절단한다. 래퍼를 우회하므로 블로킹/스레드 충돌이 없다(이전 프로젝트에서 검증된 패턴).

**구현.** [robot_arm_node.py `safe_stop_robot()`](a4_cobot2/robot_arm/robot_arm_node.py#L361), soft 가드는 [robot_motion.py `_wrap_motion_guard()`](a4_cobot2/robot_arm/robot_motion.py#L97). 단 음성 "멈춰"의 경로 지연(녹음+STT/GPT 왕복)은 남으므로 **진짜 비상정지는 물리 E-stop**이 원칙.

### 10. 중앙 상태머신 — 직렬화 + 세대 검증 + 재검증 루프

**문제.** 스캔·판정·로봇동작이 모두 비동기(service/action)라 명령이 겹치면 로봇이 두 작업을 동시에 하거나, 늦게 도착한 옛 응답이 새 작업을 오염시킨다.

**접근.** `task_manager_node`가 전체를 단일 스레드 콜백 체인으로 오케스트레이션한다.
- **`is_busy` 직렬화**: 한 번에 하나의 큰 작업만 허용, 겹친 명령은 BUSY로 거절.
- **`current_task` 세대 검증**: 요청 시점의 작업명을 스냅샷으로 잡아, 응답 콜백에서 지금 작업과 다르면 **지연 응답을 폐기**.
- **단일 종료 경로**: 모든 성공/실패 분기가 `finish_current_task()` → `IDLE`로 수렴.
- **service vs action 구분**: 장기 실행(진행률·취소가 필요한 정리)만 action, 나머지는 service. 스캔은 `Trigger 요청 + 결과 토픽 분리` 패턴.
- **재검증 루프**: 정리 성공 → 같은 스캔·판정 파이프라인을 `recheck` 모드로 **1회 재실행** → VLM 최종 보고로 마무리.

**구현.** [task_manager_node.py](a4_cobot2/task_manager/task_manager_node.py#L505). 노드 간 JSON payload 생성/파싱과 topic·상수는 [payload_utils.py](a4_cobot2/task_manager/payload_utils.py)·[task_config.py](a4_cobot2/task_manager/task_config.py)에 **중앙집중**해 필드 오타·불일치를 막는다. 안전은 `safety_node`가 `/safety_command`를 **로봇용 `/emergency_stop`(Bool 즉시신호) + HMI용 `/safety_state`(String 표시)** 로 이원화하고, `stop`(복구 가능)과 `shutdown`(프로세스 종료)을 개념적으로 구분한다.

### 11. VLM 최종 보고 — 판정 우선, 사진은 근거 보조

**문제.** 좌표·판정 JSON만으론 사용자에게 "무엇이 어떻게 됐는지" 자연스럽게 전달하기 어렵다.

**접근.** 재검증 3자세에서 저장한 **실제 사진 + 판정 JSON을 GPT-4o(멀티모달)에 함께 넣어** 듣기 좋은 한국어 보고문을 만든다. 원칙은 **"공식 판단은 판정 JSON이 우선, VLM은 서술 보조"** — 사진이 JSON과 모순되면 단정하지 말고 "추가 확인 필요"로 말하게 하고, 좌표·파일경로 같은 내부 정보는 노출 금지한다. 이미지는 base64 data URL(`detail:low`)로 인코딩해 토큰을 아낀다.

**구현.** [vlm_report_node.py](a4_cobot2/notification/vlm_report_node.py#L152), `/generate_final_report` 서비스. VLM 미가용·이미지 없음·호출 예외의 **3단계 fallback**으로 규칙 기반 보고문을 내보내며, 출처는 `source`(`vlm`/`fallback`/`error`)로 구분한다.

### 12. HMI(PyQt) & 이력/재현 인프라

- **HMI**: `HmiRosBridge`가 **QThread에서 ROS spin, `pyqtSignal`로 GUI 스레드에 전달**해 ROS↔Qt 충돌을 피한다. `/workspace_judgement`(latched QoS) 구독으로 **실제 스캔 결과 테이블**(PLACED/MISPLACED)을 그리고, **WAKE UP 버튼**은 `/voice_start`를 쏴 웨이크워드 없이 음성 입력을 트리거한다([hmi_ros_bridge.py](a4_cobot2/hmi/hmi_ros_bridge.py#L199), [hmi_interface.py](a4_cobot2/hmi/hmi_interface.py#L615)).
- **지연 구독자 동기화**: 판정·안전 상태 토픽에 **`TRANSIENT_LOCAL`(latched) QoS**를 걸어, HMI/DB가 늦게 떠도 마지막 값을 즉시 받게 한다.
- **DB + rosbag**: `db_node`가 **로깅 전용 새 토픽을 만들지 않고** 기존 토픽만 구독해 작업(run) 단위로 SQLite 6개 테이블에 이력을 남기고, `rosbag_manager`가 run마다 실제 `ros2 bag record`를 자동 실행/종료(**SIGINT→SIGTERM→SIGKILL** 단계 종료 + `metadata.yaml`·`.db3` 검증 후에만 DB 연결)한다([db_node.py](a4_cobot2/database/db_node.py), [rosbag_manager.py](a4_cobot2/database/rosbag_manager.py#L204)).

### 13. 견고성 — graceful degradation

부품 하나가 없어도 전체가 죽지 않게 설계했다: VLM 미가용 → **규칙 기반 보고문**, TTS 실패 → **콘솔 출력**, rosbag 실패 → **DB 로깅은 계속**, 상수 import 실패 → **하드코딩 기본값**. 노드 간 통신은 String에 실은 JSON을 컨테이너로 쓰고, 원격 호출은 모두 `wait_for_service/server` 타임아웃 가드 후 전용 UNAVAILABLE 상태를 발행한다.

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

> 소스 코드의 잠재 오류점·개선 제안(안전/멈춤/동작오류/성능/빌드 심각도별 리뷰)은 [code_review.md](code_review.md)로 분리했다.
