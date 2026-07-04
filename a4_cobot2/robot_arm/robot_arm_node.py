import json
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool
from od_msg.action import OrganizeObjects

class RobotArmNode(Node):
    # robot_arm_node를 초기화하고 정리 실행 action server와 emergency_stop 구독자를 준비하는 함수
    def __init__(self):
        super().__init__('robot_arm_node')

        self.callback_group = ReentrantCallbackGroup()
        self.emergency_stop_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # task_manager_node가 정리할 물체 목록을 보내면 로봇팔 정리 작업을 수행하는 action server
        # Goal: organize_objects_json
        # Feedback: current_object, current_index, total_count, status
        # Result: success, result_json, message
        self.organize_action_server = ActionServer(
            self,
            OrganizeObjects,
            '/organize_objects',
            execute_callback=self.execute_organize_objects,
            goal_callback=self.handle_goal,
            cancel_callback=self.handle_cancel,
            callback_group=self.callback_group
        )

        # 나중에 command_input_node 또는 safety_node가 발행할 emergency stop 신호를 받기 위한 구독자
        # 지금은 stop 설계를 열어두기 위한 구조이며, 실제 정지 함수는 로봇 API 연결 후 구현
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_stop_callback,
            self.emergency_stop_qos,
            callback_group=self.callback_group
        )

        self.emergency_stop_requested = False

        self.get_logger().info('RobotArmNode started.')
        self.get_logger().info('/organize_objects action server ready.')

    # 새 정리 작업 goal을 받을지 판단하는 함수
    def handle_goal(self, goal_request):
        self.get_logger().info('Received organize_objects goal.')

        if self.emergency_stop_requested:
            self.get_logger().warn('Emergency stop 상태라 goal을 거절합니다.')
            return GoalResponse.REJECT

        if goal_request.organize_objects_json.strip() == '':
            self.get_logger().warn('organize_objects_json이 비어 있어 goal을 거절합니다.')
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    # action cancel 요청을 받을지 판단하고 로봇 정지를 시도하는 함수
    def handle_cancel(self, goal_handle):
        self.get_logger().warn('Cancel request received.')

        self.safe_stop_robot()

        return CancelResponse.ACCEPT

    # /emergency_stop 토픽을 받아 로봇 정지 요청 플래그를 세우거나 해제하는 함수
    def emergency_stop_callback(self, msg: Bool):
        if msg.data:
            self.emergency_stop_requested = True
            self.get_logger().warn('Emergency stop requested.')

            self.safe_stop_robot()

        else:
            self.emergency_stop_requested = False
            self.get_logger().info('Emergency stop cleared.')

    # task_manager_node가 보낸 정리 대상 목록을 받아 순서대로 정리 작업을 수행하는 action 실행 함수
    def execute_organize_objects(self, goal_handle):
        result = OrganizeObjects.Result()

        try:
            goal_payload = json.loads(goal_handle.request.organize_objects_json)
            objects = goal_payload.get('objects', [])

        except json.JSONDecodeError as e:
            self.get_logger().error(f'organize_objects_json 파싱 실패: {e}')
            result.success = False
            result.result_json = self.make_result_json([], [], 'invalid_goal_json')
            result.message = 'invalid organize_objects_json'
            goal_handle.abort()
            return result

        if len(objects) == 0:
            self.get_logger().warn('정리할 물체가 없습니다.')
            result.success = True
            result.result_json = self.make_result_json([], [], 'nothing_to_organize')
            result.message = 'nothing to organize'
            goal_handle.succeed()
            return result

        total_count = len(objects)
        completed_objects = []
        failed_objects = []

        self.get_logger().info(f'정리 대상 물체 수: {total_count}')

        for index, misplaced_object in enumerate(objects, start=1):
            if goal_handle.is_cancel_requested:
                self.get_logger().warn('정리 action이 cancel되었습니다.')
                result.success = False
                result.result_json = self.make_result_json(
                    completed_objects,
                    failed_objects,
                    'canceled'
                )
                result.message = 'organize action canceled'
                goal_handle.canceled()
                return result

            if self.emergency_stop_requested:
                self.get_logger().warn('Emergency stop으로 정리 action을 중단합니다.')
                result.success = False
                result.result_json = self.make_result_json(
                    completed_objects,
                    failed_objects,
                    'emergency_stop'
                )
                result.message = 'emergency stop requested'
                goal_handle.abort()
                return result

            object_name = misplaced_object.get('name', 'unknown_object')

            self.publish_feedback(
                goal_handle,
                current_object=object_name,
                current_index=index,
                total_count=total_count,
                status='organizing'
            )

            try:
                self.organize_single_object(misplaced_object)
                completed_objects.append(misplaced_object)

                self.publish_feedback(
                    goal_handle,
                    current_object=object_name,
                    current_index=index,
                    total_count=total_count,
                    status='object_done'
                )

            except NotImplementedError as e:
                self.get_logger().error(f'로봇 제어 함수 미구현: {e}')

                failed_objects.append({
                    'object': misplaced_object,
                    'reason': str(e)
                })

                result.success = False
                result.result_json = self.make_result_json(
                    completed_objects,
                    failed_objects,
                    'robot_control_not_implemented'
                )
                result.message = 'robot control function is not implemented'
                goal_handle.abort()
                return result

            except Exception as e:
                self.get_logger().error(f'{object_name} 정리 중 오류 발생: {e}')

                failed_objects.append({
                    'object': misplaced_object,
                    'reason': str(e)
                })

                result.success = False
                result.result_json = self.make_result_json(
                    completed_objects,
                    failed_objects,
                    'organize_failed'
                )
                result.message = f'organize failed: {e}'
                goal_handle.abort()
                return result

        result.success = True
        result.result_json = self.make_result_json(
            completed_objects,
            failed_objects,
            'organize_finished'
        )
        result.message = 'organize finished'

        goal_handle.succeed()
        return result

    # action feedback을 task_manager_node에 전달하는 함수
    def publish_feedback(self, goal_handle, current_object: str, current_index: int, total_count: int, status: str):
        feedback = OrganizeObjects.Feedback()
        feedback.current_object = current_object
        feedback.current_index = current_index
        feedback.total_count = total_count
        feedback.status = status

        goal_handle.publish_feedback(feedback)

        self.get_logger().info(
            f'Feedback: {status} {current_index}/{total_count} - {current_object}'
        )

    # 오배치 물체 하나를 실제로 집어서 정상 위치로 옮기는 함수
    def organize_single_object(self, misplaced_object):
        object_name = misplaced_object.get('name', 'unknown_object')

        self.get_logger().info(f'정리 작업 시작: {object_name}')

        # TODO:
        # 1. misplaced_object["position"]을 robot base 좌표계로 변환
        # 2. 로봇팔을 pick pose로 이동
        # 3. gripper close
        # 4. expected_zone 또는 target pose로 이동
        # 5. gripper open
        # 6. safe pose로 복귀
        #
        # 지금은 실제 로봇 제어 API가 연결되지 않았으므로 성공 처리하지 않습니다.
        raise NotImplementedError(
            'organize_single_object 내부에 실제 로봇 제어 코드를 연결해야 합니다.'
        )

    # action result에 넣을 JSON 문자열을 만드는 함수
    def make_result_json(self, completed_objects, failed_objects, status: str):
        payload = {
            'task': 'organize_objects',
            'status': status,
            'completed_objects': completed_objects,
            'failed_objects': failed_objects,
            'summary': {
                'completed_count': len(completed_objects),
                'failed_count': len(failed_objects),
            }
        }

        return json.dumps(payload, ensure_ascii=False)
    

    # 실제 로봇 API를 이용해 로봇 동작을 안전하게 멈추는 함수
    def safe_stop_robot(self):
        self.get_logger().warn('safe_stop_robot called.')

        # TODO:
        # 실제 로봇 API 연결 후 여기에 정지 코드를 넣습니다.
        #
        # 예시 방향:
        # 1. 현재 motion 정지
        # 2. force control 사용 중이면 release_force 또는 release_compliance_ctrl
        # 3. gripper 동작 중이면 안전 상태 처리
        # 4. 필요하면 safe pose 복귀는 stop이 아니라 별도 recovery에서 수행
        #
        # 주의:
        # stop은 로봇 동작을 즉시 멈추는 의미입니다.
        # shutdown처럼 노드를 종료하면 안 됩니다.
        pass


# ROS2 robot_arm_node를 실행하고 action/service/subscriber callback을 병렬 처리하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = RobotArmNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()