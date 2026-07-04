import json
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String

from od_msg.srv import SrvDepthPosition, JudgeWorkspace
from od_msg.action import OrganizeObjects

from task_manager import status_codes as Status
from task_manager.task_config import (
    ACTION_ORGANIZE_OBJECTS,
    ACTION_WAIT_TIMEOUT_SEC,
    SERVICE_GET_3D_POSITION,
    SERVICE_JUDGE_WORKSPACE,
    SERVICE_WAIT_TIMEOUT_SEC,
    TARGET_OBJECTS,
    TOPIC_SAFETY_COMMAND,
    TOPIC_TASK_COMMAND,
    TOPIC_TASK_STATUS,
    TOPIC_USER_NOTICE,
)
from task_manager.payload_utils import (
    is_valid_position,
    is_workspace_detection_task,
    make_detected_object,
    make_organize_goal_json,
    make_workspace_judgement_request_json,
    parse_json_payload,
)
from notification.notice_utils import (
    make_recheck_remaining_notice,
    make_workspace_judgement_notice,
)
from safety.safety_constants import (
    SAFETY_COMMAND_CLEAR,
    SAFETY_COMMAND_STOP,
)


class TaskManagerNode(Node):
    # task_manager_node를 초기화하고 명령 구독자, 상태 발행자, service/action client를 준비하는 함수
    def __init__(self):
        super().__init__('task_manager_node')

        self.task_command_sub = self.create_subscription(
            String,
            TOPIC_TASK_COMMAND,
            self.task_command_callback,
            10,
        )

        self.task_status_pub = self.create_publisher(
            String,
            TOPIC_TASK_STATUS,
            10,
        )

        self.object_position_client = self.create_client(
            SrvDepthPosition,
            SERVICE_GET_3D_POSITION,
        )

        self.judge_workspace_client = self.create_client(
            JudgeWorkspace,
            SERVICE_JUDGE_WORKSPACE,
        )

        self.organize_objects_action_client = ActionClient(
            self,
            OrganizeObjects,
            ACTION_ORGANIZE_OBJECTS,
        )

        self.safety_command_pub = self.create_publisher(
            String,
            TOPIC_SAFETY_COMMAND,
            10,
        )

        self.user_notice_pub = self.create_publisher(
            String,
            TOPIC_USER_NOTICE,
            10,
        )

        self.current_task = None
        self.is_busy = False
        self.target_objects = TARGET_OBJECTS
        self.current_target_index = 0
        self.detected_objects = []
        self.latest_workspace_judgement = None
        self.current_robot_goal_handle = None
        self.stop_requested = False

        self.get_logger().info('TaskManagerNode started.')
        self.publish_status(Status.TASK_MANAGER_READY)

    # 현재 작업이 작업공간 감지 흐름인지 판단하는 함수
    def is_workspace_detection_task(self):
        return is_workspace_detection_task(self.current_task)

    # /task_command 토픽으로 들어온 명령을 확인하고 명령 종류에 따라 처리 함수를 호출하는 함수
    def task_command_callback(self, msg: String):
        command = msg.data.strip()

        self.get_logger().info(f'Received command: {command}')

        if command == Status.COMMAND_CHECK_WORKSPACE:
            self.handle_check_workspace()

        elif command == Status.COMMAND_START_ORGANIZE:
            self.handle_start_organize()

        elif command == Status.COMMAND_STOP:
            self.handle_stop_command()

        elif command == Status.COMMAND_SHUTDOWN:
            self.handle_shutdown_command()

        else:
            self.handle_unknown_command(command)

    # 현재 작업 상태를 /task_status 토픽으로 발행하는 함수
    def publish_status(self, status: str):
        msg = String()
        msg.data = status

        self.task_status_pub.publish(msg)
        self.get_logger().info(f'Published status: {status}')

    # 사용자에게 전달할 자연어 안내 문장을 /user_notice 토픽으로 발행하는 함수
    def publish_user_notice(self, notice: str):
        if notice is None or notice.strip() == '':
            return

        msg = String()
        msg.data = notice

        self.user_notice_pub.publish(msg)
        self.get_logger().info(f'Published /user_notice: {notice}')

    # safety_node에게 stop 또는 clear 명령을 전달하는 함수
    def publish_safety_command(self, command: str):
        msg = String()
        msg.data = command

        self.safety_command_pub.publish(msg)
        self.get_logger().info(f'Published /safety_command: {command}')

    # 작업공간 확인 명령을 받았을 때 ObjectDetectionNode에 물체 위치 요청을 시작하는 함수
    def handle_check_workspace(self):
        if self.is_busy:
            self.get_logger().warn('현재 다른 작업을 처리 중입니다.')
            self.publish_status(Status.BUSY)
            return

        self.is_busy = True
        self.start_workspace_detection(Status.TASK_CHECK_WORKSPACE)

    # 로봇 정리 작업 완료 후 작업공간을 다시 검사하는 함수
    def handle_recheck_workspace(self):
        self.get_logger().info('정리 완료 후 작업공간 재검증을 시작합니다.')

        self.is_busy = True
        self.start_workspace_detection(Status.TASK_RECHECK_WORKSPACE)

    # 작업공간 확인 또는 재검증을 시작하기 위해 공통 상태를 초기화하고 첫 위치 요청을 보내는 함수
    def start_workspace_detection(self, task_name: str):
        self.current_task = task_name
        self.current_target_index = 0
        self.detected_objects = []
        self.stop_requested = False

        if task_name == Status.TASK_CHECK_WORKSPACE:
            self.latest_workspace_judgement = None
            self.publish_status(Status.CHECK_WORKSPACE_REQUESTED)
            self.publish_user_notice('작업공간 확인을 시작합니다.')

        elif task_name == Status.TASK_RECHECK_WORKSPACE:
            self.publish_status(Status.RECHECK_WORKSPACE_REQUESTED)
            self.publish_user_notice('정리 결과를 확인하기 위해 작업공간을 다시 검사합니다.')

        self.publish_safety_command(SAFETY_COMMAND_CLEAR)

        if not self.object_position_client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            self.get_logger().error('get_3d_position 서비스를 찾을 수 없습니다.')
            self.publish_status(Status.OBJECT_DETECTION_SERVICE_UNAVAILABLE)
            self.publish_user_notice('물체 위치 인식 서비스를 찾을 수 없습니다.')
            self.finish_current_task()
            return

        if task_name == Status.TASK_CHECK_WORKSPACE:
            self.publish_status(Status.CHECKING_WORKSPACE)
        else:
            self.publish_status(Status.RECHECKING_WORKSPACE)

        self.request_next_object_position()

    # target_objects 목록에서 다음 물체 이름을 꺼내 ObjectDetectionNode에 3D 위치를 비동기로 요청하는 함수
    def request_next_object_position(self):
        if self.stop_requested:
            self.get_logger().warn('stop 요청으로 object position 요청을 중단합니다.')
            self.publish_status(Status.CHECK_WORKSPACE_STOPPED)
            self.finish_current_task()
            return

        if not self.is_workspace_detection_task():
            self.get_logger().warn('현재 작업이 작업공간 감지 흐름이 아니므로 위치 요청을 중단합니다.')
            return

        if self.current_target_index >= len(self.target_objects):
            self.finish_check_workspace_detection()
            return

        target_name = self.target_objects[self.current_target_index]

        request = SrvDepthPosition.Request()
        request.target = target_name

        self.get_logger().info(f'ObjectDetectionNode에 위치 요청: {target_name}')

        future = self.object_position_client.call_async(request)
        future.add_done_callback(
            lambda future_result, target=target_name: self.object_position_response_callback(
                future_result,
                target,
            )
        )

    # ObjectDetectionNode의 /get_3d_position 응답을 받아 유효한 위치면 저장하고 다음 물체 요청으로 넘어가는 함수
    def object_position_response_callback(self, future, target_name: str):
        if self.stop_requested or not self.is_workspace_detection_task():
            self.get_logger().warn('stop 요청 또는 작업 변경으로 다음 object position 요청을 진행하지 않습니다.')
            return

        try:
            response = future.result()
            position = list(response.depth_position)

            self.get_logger().info(f'{target_name} 위치 응답: {position}')

            if is_valid_position(position):
                detected_object = make_detected_object(target_name, position)
                self.detected_objects.append(detected_object)

                self.get_logger().info(f'감지된 물체 추가: {detected_object}')

            else:
                self.get_logger().warn(f'{target_name} 감지 실패 또는 유효하지 않은 위치입니다.')

        except Exception as e:
            self.get_logger().error(f'{target_name} 위치 요청 중 오류 발생: {e}')

        if self.stop_requested or not self.is_workspace_detection_task():
            self.get_logger().warn('stop 요청으로 다음 object position 요청을 진행하지 않습니다.')
            return

        self.current_target_index += 1
        self.request_next_object_position()

    # 모든 target_objects에 대한 위치 요청이 끝났을 때 workspace_judge_node 판단으로 넘어가는 함수
    def finish_check_workspace_detection(self):
        if len(self.detected_objects) == 0:
            self.get_logger().warn('감지된 물체가 없습니다.')
            self.publish_status(Status.NO_OBJECTS_DETECTED)
            self.publish_user_notice('작업공간에서 감지된 물체가 없습니다.')
            self.finish_current_task()
            return

        self.publish_status(Status.WORKSPACE_DETECTION_FINISHED)
        self.request_workspace_judgement()

    # 감지된 물체 목록을 /judge_workspace 서비스 요청 JSON으로 만들어 workspace_judge_node에 보내는 함수
    def request_workspace_judgement(self):
        if not self.judge_workspace_client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            self.get_logger().error('/judge_workspace 서비스를 찾을 수 없습니다.')
            self.publish_status(Status.JUDGE_WORKSPACE_SERVICE_UNAVAILABLE)
            self.finish_current_task()
            return

        request = JudgeWorkspace.Request()
        request.detected_objects_json = make_workspace_judgement_request_json(
            task_name=self.current_task,
            objects=self.detected_objects,
        )

        self.get_logger().info(f'WorkspaceJudgeNode에 판단 요청: {request.detected_objects_json}')

        self.publish_status(Status.JUDGING_WORKSPACE)

        future = self.judge_workspace_client.call_async(request)
        future.add_done_callback(self.workspace_judgement_response_callback)

    # workspace_judge_node의 /judge_workspace 응답을 받아 판단 결과를 저장하고 다음 상태를 결정하는 함수
    def workspace_judgement_response_callback(self, future):
        task_at_response_time = self.current_task

        try:
            response = future.result()

            if not response.success:
                self.get_logger().error(f'작업공간 판단 실패: {response.message}')
                self.publish_status(Status.WORKSPACE_JUDGEMENT_FAILED)
                self.publish_user_notice('작업공간 판단에 실패했습니다.')
                return

            judgement_payload = parse_json_payload(response.judgement_json)

            self.get_logger().info(f'작업공간 판단 결과: {judgement_payload}')

            if task_at_response_time == Status.TASK_CHECK_WORKSPACE:
                self.handle_initial_workspace_judgement_result(judgement_payload)

            elif task_at_response_time == Status.TASK_RECHECK_WORKSPACE:
                self.handle_recheck_workspace_judgement_result(judgement_payload)

            else:
                self.publish_status(Status.WORKSPACE_JUDGEMENT_UNEXPECTED_TASK)
                self.publish_user_notice('예상하지 못한 작업 상태에서 작업공간 판단 결과를 받았습니다.')

        except json.JSONDecodeError as e:
            self.get_logger().error(f'judgement_json 파싱 실패: {e}')
            self.publish_status(Status.WORKSPACE_JUDGEMENT_JSON_ERROR)
            self.publish_user_notice('작업공간 판단 결과를 해석하지 못했습니다.')

        except Exception as e:
            self.get_logger().error(f'workspace judgement 응답 처리 중 오류 발생: {e}')
            self.publish_status(Status.WORKSPACE_JUDGEMENT_RESPONSE_ERROR)
            self.publish_user_notice('작업공간 판단 응답 처리 중 오류가 발생했습니다.')

        finally:
            self.finish_current_task()

    # 오배치 물체 목록을 robot_arm_node의 /organize_objects action goal로 보내는 함수
    def request_robot_organize(self, misplaced_objects):
        if not self.organize_objects_action_client.wait_for_server(timeout_sec=ACTION_WAIT_TIMEOUT_SEC):
            self.get_logger().error('/organize_objects action server를 찾을 수 없습니다.')
            self.publish_status(Status.ROBOT_ARM_ACTION_UNAVAILABLE)
            self.finish_current_task()
            return

        goal_msg = OrganizeObjects.Goal()
        goal_msg.organize_objects_json = make_organize_goal_json(misplaced_objects)

        self.get_logger().info(f'RobotArmNode에 정리 요청: {goal_msg.organize_objects_json}')
        self.publish_status(Status.REQUESTING_ROBOT_ORGANIZE)

        send_goal_future = self.organize_objects_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.robot_organize_feedback_callback,
        )

        send_goal_future.add_done_callback(self.robot_organize_goal_response_callback)

    # robot_arm_node가 정리 action goal을 수락했는지 확인하는 함수
    def robot_organize_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('RobotArmNode가 정리 goal을 거절했습니다.')
            self.publish_status(Status.ROBOT_ORGANIZE_GOAL_REJECTED)
            self.finish_current_task()
            return

        self.get_logger().info('RobotArmNode가 정리 goal을 수락했습니다.')
        self.publish_status(Status.ROBOT_ORGANIZE_GOAL_ACCEPTED)

        self.current_robot_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.robot_organize_result_callback)

    # robot_arm_node가 보내는 action feedback을 받아 현재 정리 진행 상태를 /task_status로 알리는 함수
    def robot_organize_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback

        self.get_logger().info(
            f'로봇 정리 진행: {feedback.current_index}/{feedback.total_count}, '
            f'object={feedback.current_object}, status={feedback.status}'
        )

        self.publish_status(
            Status.make_robot_organizing_status(
                feedback.current_index,
                feedback.total_count,
            )
        )

    # robot_arm_node의 정리 action result를 받아 성공/실패 상태를 정리하고 필요하면 재검증을 시작하는 함수
    def robot_organize_result_callback(self, future):
        should_finish_task = True

        try:
            action_result = future.result()
            result = action_result.result

            self.get_logger().info(f'Robot organize result: {result.result_json}')

            if result.success:
                self.publish_status(Status.ROBOT_ORGANIZE_FINISHED)
                self.publish_user_notice('로봇 정리 작업이 완료되었습니다. 작업공간을 다시 확인합니다.')

                self.current_robot_goal_handle = None
                should_finish_task = False
                self.handle_recheck_workspace()

            else:
                self.publish_status(Status.ROBOT_ORGANIZE_FAILED)
                self.publish_user_notice(
                    '로봇 정리 작업을 완료하지 못했습니다. 현재는 실제 로봇 제어 함수가 아직 연결되지 않았을 수 있습니다.'
                )

        except Exception as e:
            self.get_logger().error(f'로봇 정리 action result 처리 중 오류 발생: {e}')
            self.publish_status(Status.ROBOT_ORGANIZE_RESULT_ERROR)
            self.publish_user_notice('로봇 정리 결과 처리 중 오류가 발생했습니다.')

        finally:
            self.current_robot_goal_handle = None

            if should_finish_task:
                self.finish_current_task()

    # robot_arm_node에 보낸 organize action cancel 요청 결과를 처리하는 함수
    def robot_organize_cancel_callback(self, future):
        try:
            cancel_response = future.result()

            if len(cancel_response.goals_canceling) > 0:
                self.get_logger().warn('organize action cancel 요청이 수락되었습니다.')
                self.publish_status(Status.ROBOT_ORGANIZE_CANCEL_ACCEPTED)
            else:
                self.get_logger().warn('organize action cancel 요청이 거절되었거나 취소할 goal이 없습니다.')
                self.publish_status(Status.ROBOT_ORGANIZE_CANCEL_REJECTED)

        except Exception as e:
            self.get_logger().error(f'organize action cancel 처리 중 오류 발생: {e}')
            self.publish_status(Status.ROBOT_ORGANIZE_CANCEL_ERROR)

        finally:
            self.current_robot_goal_handle = None
            self.finish_current_task()

    # 현재 작업 상태를 초기화하고 task_manager_node를 idle 상태로 되돌리는 함수
    def finish_current_task(self):
        self.is_busy = False
        self.current_task = None
        self.current_target_index = 0
        self.publish_status(Status.IDLE)

    # 정리 시작 명령을 받았을 때 workspace 판단 결과를 기반으로 robot_arm_node에 정리 action을 요청하는 함수
    def handle_start_organize(self):
        if self.is_busy:
            self.get_logger().warn('현재 다른 작업을 처리 중입니다.')
            self.publish_status(Status.BUSY)
            return

        self.is_busy = True
        self.current_task = Status.TASK_START_ORGANIZE
        self.stop_requested = False

        self.publish_safety_command(SAFETY_COMMAND_CLEAR)

        self.publish_status(Status.START_ORGANIZE_REQUESTED)
        self.get_logger().info('정리 시작 명령을 받았습니다.')

        if self.latest_workspace_judgement is None:
            self.get_logger().warn('저장된 작업공간 판단 결과가 없습니다.')
            self.publish_status(Status.NO_WORKSPACE_JUDGEMENT_AVAILABLE)
            self.publish_user_notice('저장된 작업공간 판단 결과가 없습니다. 먼저 작업공간을 확인해주세요.')
            self.finish_current_task()
            return

        misplaced_objects = self.latest_workspace_judgement.get('misplaced_objects', [])

        if len(misplaced_objects) == 0:
            self.get_logger().info('오배치 물체가 없어 정리 작업이 필요 없습니다.')
            self.publish_status(Status.NOTHING_TO_ORGANIZE)
            self.publish_user_notice('정리할 물체가 없습니다. 모든 물건이 올바른 위치에 있는 것으로 판단됩니다.')
            self.finish_current_task()
            return

        self.get_logger().info(f'정리 대상 물체 목록: {misplaced_objects}')
        self.request_robot_organize(misplaced_objects)

    # stop 명령을 받았을 때 safety_node에 stop을 요청하고 진행 중인 action cancel을 요청하는 함수
    def handle_stop_command(self):
        self.get_logger().warn('stop 명령을 받았습니다.')

        self.stop_requested = True

        self.publish_status(Status.STOP_REQUESTED)
        self.publish_safety_command(SAFETY_COMMAND_STOP)
        self.publish_user_notice('정지 요청을 보냈습니다. 로봇 동작을 중단합니다.')

        if self.current_robot_goal_handle is not None:
            self.get_logger().warn('진행 중인 organize action cancel을 요청합니다.')

            cancel_future = self.current_robot_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.robot_organize_cancel_callback)

            self.publish_status(Status.ROBOT_ORGANIZE_CANCEL_REQUESTED)
            return

        self.finish_current_task()

    # shutdown 명령을 받았을 때 향후 노드 종료 또는 시스템 종료 흐름으로 확장하기 위한 함수
    def handle_shutdown_command(self):
        self.get_logger().warn('shutdown 명령을 받았습니다.')

        self.publish_status(Status.SHUTDOWN_REQUESTED)

    # 알 수 없는 명령을 받았을 때 상태를 발행하고 무시하는 함수
    def handle_unknown_command(self, command: str):
        self.get_logger().warn(f'알 수 없는 명령입니다: {command}')
        self.publish_status(Status.UNKNOWN_COMMAND)

    # 최초 작업공간 확인 결과를 처리하는 함수
    def handle_initial_workspace_judgement_result(self, judgement_payload):
        self.latest_workspace_judgement = judgement_payload

        result = judgement_payload.get('result', 'unknown')

        if result == 'all_clear':
            self.publish_status(Status.WORKSPACE_ALL_CLEAR)

        elif result == 'misplaced_found':
            self.publish_status(Status.WORKSPACE_MISPLACED_FOUND)

        elif result == 'unknown_rule_found':
            self.publish_status(Status.WORKSPACE_UNKNOWN_RULE_FOUND)

        elif result == 'no_objects':
            self.publish_status(Status.NO_OBJECTS_DETECTED)

        else:
            self.publish_status(Status.WORKSPACE_JUDGEMENT_UNKNOWN_RESULT)

        self.publish_status(Status.WORKSPACE_JUDGEMENT_FINISHED)

        notice = make_workspace_judgement_notice(judgement_payload)
        self.publish_user_notice(notice)

    # 정리 후 재검증 결과를 처리하고 최종 완료/미완료 상태를 결정하는 함수
    def handle_recheck_workspace_judgement_result(self, judgement_payload):
        self.latest_workspace_judgement = judgement_payload

        result = judgement_payload.get('result', 'unknown')

        if result == 'all_clear':
            self.publish_status(Status.RECHECK_ALL_CLEAR)
            self.publish_user_notice('정리가 완료되었습니다. 모든 물건이 지정된 구역에 배치되었습니다.')

        elif result == 'misplaced_found':
            self.publish_status(Status.RECHECK_MISPLACED_REMAINING)

            notice = make_recheck_remaining_notice(judgement_payload)
            self.publish_user_notice(notice)

        elif result == 'unknown_rule_found':
            self.publish_status(Status.RECHECK_UNKNOWN_RULE_FOUND)
            self.publish_user_notice('정리 후 재검증을 했지만 일부 물체의 배치 규칙을 찾을 수 없습니다. 확인이 필요합니다.')

        elif result == 'no_objects':
            self.publish_status(Status.RECHECK_NO_OBJECTS_DETECTED)
            self.publish_user_notice('재검증 중 감지된 물체가 없습니다. 카메라 시야 또는 작업공간을 확인해주세요.')

        else:
            self.publish_status(Status.RECHECK_UNKNOWN_RESULT)
            self.publish_user_notice('정리 후 작업공간 상태를 정확히 판단하지 못했습니다. 확인이 필요합니다.')


# ROS2 task_manager_node를 실행하고 콜백을 계속 처리하는 메인 함수
# shutdown은 노드/프로세스 종료 의미이고, stop은 로봇 동작/작업 정지 의미입니다.
def main(args=None):
    rclpy.init(args=args)

    node = TaskManagerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
