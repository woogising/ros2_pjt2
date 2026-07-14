# A4 Cobot2 - 음성 기반 작업공간 정리 보조 로봇
----------------------------------------------------------------------------------------------------
! 외부라이브러리 sam2를 사용하기 때문에 실행 ws 밖에서 아래 내용을 반드시 실행해야 합니다.


git clone https://github.com/facebookresearch/sam2


cd sam2


pip install -e .


----------------------------------------------------------------------------------------------------
> ROS2, RealSense Depth Camera, Doosan M0609 협동로봇을 활용한
> **음성 기반 작업공간 확인 및 정리 보조 시스템**

## 1. 프로젝트 개요

본 프로젝트는 사용자의 음성 명령을 기반으로 작업공간을 확인하고, 카메라로 인식된 물체의 위치를 판단하여 잘못 배치된 물체를 사용자에게 알려주는 협동로봇 시스템이다.

현재 구현 단계에서는 실제 물체를 완전히 집고 옮기는 pick-and-place 동작 대신, `robot_arm_node`가 로봇을 아주 조금만 움직이는 테스트 모션을 수행하고, 사용자가 손으로 물체를 치우는 방식으로 전체 동작 흐름을 검증한다.

추후 팀원이 실제 로봇암 모션 함수를 제공하면 `robot_motion.py`의 테스트 모션 부분을 실제 pick-and-place 함수로 교체하여 완성 동작으로 확장할 수 있다.

---

## 2. 현재 구현 범위

현재까지 구현 및 검증된 기능은 다음과 같다.

| 구분                    | 구현 상태 | 설명                                                    |
| ----------------------- | ----: | ----------------------------------------------------------- |
| RealSense 카메라 입력  |    완료 | RGB, aligned depth, camera info topic 사용            		 |
| 물체 인식             |    완료 | `ObjectDetectionNode`에서 YOLO 기반 물체 인식 및 3D 위치 추정    |
| 작업공간 판단           |    완료 | `workspace_judge_node`에서 정상 배치/오배치 판단               |
| 작업 흐름 제어          |    완료 | `task_manager_node`가 전체 상태 흐름 관리                     |
| 로봇암 action 통신     |    완료 | `task_manager_node` → `robot_arm_node` action 요청/응답 		 |
| 로봇 테스트 모션         |    완료 | 실제 로봇이 소폭 이동하는 테스트 동작 수행                     |
| 사용자 수동 제거         |    완료 | 로봇 테스트 모션 후 사용자가 물체를 손으로 제거                 |
| 재검증               |    완료 | 정리 동작 후 작업공간을 다시 확인                                 |
| 최종 사용자 안내         |    완료 | `/user_notice` topic으로 결과 안내                           |
| 음성 명령 입력          |    완료 | `command_input_node`에서 마이크/STT 기반 명령 입력             |
| 실제 pick-and-place | 추후 구현 | 팀원이 제공할 실제 로봇 모션 함수로 교체 예정                     |

---

## 3. 전체 동작 흐름

0. 로봇 동작 전 대기 상태

1. 사용자가 음성으로 로봇 호출
   예: "로봇아, 작업공간 확인해줘."

2. command_input_node가 음성 명령을 인식
   STT 결과를 내부 명령으로 변환
   /task_command topic 발행

3. task_manager_node가 작업 명령 수신
   check_workspace 명령이면 작업공간 확인 프로세스 시작

4. ObjectDetectionNode가 물체 인식
   RGB + Depth Camera 정보를 이용해 물체 종류와 3D 위치 추정

5. workspace_judge_node가 물체 배치 상태 판단
   정상 배치 물체와 오배치 물체를 구분

6. task_manager_node가 사용자에게 판단 결과 안내
   /user_notice topic 발행

7. 사용자가 정리 시작 명령
   예: "로봇아, 정리 시작해줘."

8. task_manager_node가 robot_arm_node에 action 요청
   /organize_objects action 사용

9. robot_arm_node가 테스트 로봇 모션 수행
   현재는 실제 pick-and-place 대신 소폭 이동 테스트 모션 수행

10. 사용자가 손으로 오배치 물체 제거

11. task_manager_node가 작업공간 재검증

12. 최종 결과를 사용자에게 안내
   정상 완료 또는 오배치 물체 남음 상태 출력


---

## 4. 시스템 구조


사용자 음성
   ↓
command_input_node
   ↓ /task_command
task_manager_node
   ├─ get_3d_position service → ObjectDetectionNode
   ├─ /judge_workspace service → workspace_judge_node
   ├─ /organize_objects action → robot_arm_node
   ├─ /safety_command → safety_node
   └─ /user_notice → 사용자 안내

safety_node
   ↓ /emergency_stop
robot_arm_node
   ↓
robot_motion.py
   ↓
Doosan M0609 Robot

---

## 5. 주요 노드 설명

### 5-1. `command_input_node`

사용자의 마이크 입력을 받아 STT를 수행하고, 인식된 문장을 내부 작업 명령으로 변환한다.

발행 topic:


/task_command


사용 가능한 음성 명령 예시는 다음과 같다.

| 사용자 음성         | 내부 명령             |
| -------------- | ----------------- |
| 로봇아, 작업공간 확인해줘 | `check_workspace` |
| 작업공간 확인해줘      | `check_workspace` |
| 상태 확인해줘        | `check_workspace` |
| 다시 확인해줘        | `check_workspace` |
| 로봇아, 정리 시작해줘   | `start_organize`  |
| 정리해줘           | `start_organize`  |
| 잘못된 물건 치워줘     | `start_organize`  |
| 로봇아, 멈춰        | `stop`            |
| 정지             | `stop`            |
| 그만             | `stop`            |

---

### 5-2. `task_manager_node`

전체 작업 흐름을 제어하는 중심 노드이다.

주요 역할:

- /task_command 수신
- 작업공간 확인 요청
- 오배치 물체 목록 저장
- 로봇 정리 action 요청
- 정리 후 재검증
- 사용자 안내 메시지 발행
- stop 명령 처리


주요 topic:

Subscribe:
- /task_command

Publish:
- /task_status
- /user_notice
- /safety_command


주요 service/action:

Service Client:
- get_3d_position
- /judge_workspace

Action Client:
- /organize_objects


---

### 5-3. `ObjectDetectionNode`

RealSense 카메라의 RGB, aligned depth, camera info를 이용하여 물체를 인식하고 3D 위치를 계산한다.

사용 RealSense topic:

/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info

제공 service:

get_3d_position


현재 인식 대상 예시:

drill
hammer
pliers
screwdriver
wrench


---

### 5-4. `workspace_judge_node`

인식된 물체의 위치를 기준으로 정상 배치 여부를 판단한다.

제공 service:

/judge_workspace


판단 결과:

- all_clear
- misplaced_detected
- misplaced_remaining


---

### 5-5. `robot_arm_node`

`task_manager_node`로부터 `/organize_objects` action 요청을 받아 로봇 동작을 수행한다.

현재 단계에서는 실제 pick-and-place 대신 로봇이 아주 조금만 움직이는 테스트 모션을 수행한다. 이 테스트 모션 후 사용자가 손으로 물체를 치우고, 이후 `task_manager_node`가 작업공간을 다시 확인한다.

사용 topic:

Subscribe:
- /emergency_stop

사용 action:

Action Server:
- /organize_objects

---

### 5-6. `robot_motion.py`

Doosan 로봇 API와 직접 연결되는 모션 모듈이다.

현재 역할:

- DSR_ROBOT2 연결
- DSR 전용 ROS2 node 생성
- 테스트용 소폭 로봇 모션 수행
- stop 요청 시 이후 모션 차단

현재는 테스트 모션만 수행하지만, 추후 실제 pick-and-place 동작이 구현되면 이 파일의 모션 함수를 교체하거나 확장한다.

---

### 5-7. `safety_node`

작업 정지 명령을 받아 emergency stop topic으로 변환한다.

사용 topic:

Subscribe:
- /safety_command

Publish:
- /emergency_stop


주의:

stop     = 로봇 동작 또는 현재 작업 정지
shutdown = 노드, 프로세스, 시스템 종료


본 프로젝트에서는 두 용어를 구분하여 사용한다.

---

## 6. 실행 전 준비

### 6-1. `.env` 파일 설정

마이크/STT 기능을 사용하기 위해 `resource/.env` 파일에 OpenAI API Key를 설정한다.

```text
OPENAI_API_KEY=your_api_key_here
```

`.env` 파일은 숨김 파일이므로 `setup.py`에서 직접 설치 대상으로 추가해야 한다.

`setup.py` 예시:

```python
data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/resource', glob.glob('resource/*')),
    ('share/' + package_name + '/resource', ['resource/.env']),
]
```

---

### 6-2. 빌드

cd ~/a4_cobot2_ws
colcon build --packages-select a4_cobot2
source install/setup.bash


`.env` 설치 확인:

find ~/a4_cobot2_ws/install/a4_cobot2 -name ".env"


정상 설치 예시:

/home/jj/a4_cobot2_ws/install/a4_cobot2/share/a4_cobot2/resource/.env


---

## 7. 실행 방법

### 7-1. RealSense 실행

ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

align_depth.enable:=true` 옵션을 사용해야 RGB 이미지와 depth 이미지가 같은 기준으로 정렬된 aligned depth topic을 사용할 수 있다.

---

### 7-2. 각 노드 실행

각 터미널에서 다음 노드들을 실행한다.

ros2 run a4_cobot2 object_detection_node

ros2 run a4_cobot2 workspace_judge_node

ros2 run a4_cobot2 safety_node

ros2 run a4_cobot2 robot_arm_node

ros2 run a4_cobot2 task_manager_node


ros2 run a4_cobot2 command_input_node


---

### 7-3. 상태 확인용 topic echo

사용자 안내 확인:

ros2 topic echo /user_notice


작업 상태 확인:

ros2 topic echo /task_status


---

## 8. 음성 명령 시나리오

### 8-1. 작업공간 확인

사용자 발화:

로봇아, 작업공간 확인해줘.


예상 흐름:

command_input_node
→ /task_command: check_workspace
→ task_manager_node
→ ObjectDetectionNode
→ workspace_judge_node
→ /user_notice로 결과 안내

예상 안내:

현재 잘못 배치된 물건은 2개입니다.
정리를 원하면 "로봇아, 정리 시작해줘"라고 말해주세요.


---

### 8-2. 정리 시작

사용자 발화:
로봇아, 정리 시작해줘.


예상 흐름:
command_input_node
→ /task_command: start_organize
→ task_manager_node
→ /organize_objects action 요청
→ robot_arm_node
→ robot_motion.py 테스트 모션 수행
→ 사용자가 손으로 물체 제거
→ task_manager_node가 자동 재검증
→ /user_notice로 최종 결과 안내


예상 안내:
로봇 보조 동작을 시작합니다.
로봇이 조금 움직인 뒤 물건을 손으로 치워주세요.

최종 안내 예시:
정리 완료되었습니다. 작업공간이 정상입니다.

또는:
아직 잘못 배치된 물건이 남아 있습니다.

---

### 8-3. 작업 정지

사용자 발화: 로봇아, 멈춰.

또는: 정지.

예상 흐름:
command_input_node
→ /task_command: stop
→ task_manager_node
→ /safety_command
→ safety_node
→ /emergency_stop
→ robot_arm_node
→ robot_motion.py stop 처리

---

## 9. 수동 테스트 명령

음성 입력 없이 topic publish로 직접 테스트할 수도 있다.

작업공간 확인:

```bash
ros2 topic pub --once /task_command std_msgs/msg/String "{data: 'check_workspace'}"
```

정리 시작:

ros2 topic pub --once /task_command std_msgs/msg/String "{data: 'start_organize'}"

작업 정지:
ros2 topic pub --once /task_command std_msgs/msg/String "{data: 'stop'}"

---

## 10. 현재 검증 완료 사항

현재까지 다음 흐름이 정상 동작함을 확인하였다.

1. command_input_node에서 음성 명령 인식
2. /task_command topic 발행
3. task_manager_node가 명령 수신
4. 작업공간 확인 프로세스 실행
5. 오배치 물체 판단 결과 안내
6. 정리 시작 명령 수신
7. task_manager_node가 robot_arm_node에 action 요청
8. robot_arm_node가 robot_motion.py를 통해 실제 로봇 테스트 모션 수행
9. 사용자가 손으로 물체 제거
10. task_manager_node가 작업공간 재검증
11. 최종 결과를 /user_notice로 안내

---

## 11. 향후 개선 사항

추후 구현 또는 개선할 부분은 다음과 같다.

- 실제 pick-and-place 모션 함수 연결
- 물체별 목표 위치 계산 로직 고도화
- 구역 판단 기준 보정
- 음성 명령 LLM 기반 보정
- 사용자 안내 TTS 출력
- launch 파일 통합
- 로봇 stop과 shutdown 동작 분리 고도화
- emergency stop 이후 재시작 시나리오 안정화


---

## 12. 노드 요약표

| 노드명                    | 역할                                   | 사용 topic                                                                                | 사용 service                                    | 사용 action                   | 패키지 의존성                                            | 추가 설치 필요              |
| ---------------------- | ------------------------------------ | --------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------- | -------------------------------------------------- | --------------------- |
| `command_input_node`   | 마이크/STT 입력을 내부 작업 명령으로 변환            | Publish: `/task_command`                                                                | 없음                                            | 없음                          | `rclpy`, `std_msgs`, `voice.stt`, `python-dotenv`  | `OPENAI_API_KEY` 필요   |
| `task_manager_node`    | 작업공간 확인, 정리 시작, stop, 재검증, 사용자 안내 제어 | Subscribe: `/task_command` / Publish: `/task_status`, `/user_notice`, `/safety_command` | Client: `get_3d_position`, `/judge_workspace` | Client: `/organize_objects` | `rclpy`, `std_msgs`, custom interface              | 없음                    |
| `ObjectDetectionNode`  | RGB/Depth 기반 물체 인식 및 3D 위치 추정        | Subscribe: RealSense RGB, aligned depth, camera info                                    | Server: `get_3d_position`                     | 없음                          | `rclpy`, `sensor_msgs`, `cv_bridge`, YOLO 관련 패키지   | RealSense, YOLO 환경 필요 |
| `workspace_judge_node` | 인식된 물체의 위치를 기준으로 정상/오배치 판단           | 없음                                                                                      | Server: `/judge_workspace`                    | 없음                          | `rclpy`, custom interface                          | 없음                    |
| `robot_arm_node`       | 정리 action 요청을 받아 로봇 테스트 모션 수행        | Subscribe: `/emergency_stop`                                                            | 선택: `/{ROBOT_ID}/motion/move_stop`            | Server: `/organize_objects` | `rclpy`, `std_msgs`, custom action, `robot_motion` | Doosan ROS2/DSR 환경 필요 |
| `robot_motion.py`      | Doosan API 연결 및 실제 로봇 테스트 모션 수행      | 없음                                                                                      | 내부적으로 Doosan motion service 사용                | 없음                          | `DR_init`, `DSR_ROBOT2`, `rclpy`                   | Doosan Python API 필요  |
| `safety_node`          | stop 명령을 emergency stop topic으로 변환   | Subscribe: `/safety_command` / Publish: `/emergency_stop`                               | 없음                                            | 없음                          | `rclpy`, `std_msgs`                                | 없음                    |
| `status_notifier_node` | 상태 또는 안내 메시지 출력 담당                   | Subscribe: `/user_notice` 또는 `/task_status`                                             | 없음                                            | 없음                          | `rclpy`, `std_msgs`                                | TTS 사용 시 별도 패키지 필요    |
