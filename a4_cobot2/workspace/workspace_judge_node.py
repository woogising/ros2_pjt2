import json
import rclpy

from rclpy.node import Node
from od_msg.srv import JudgeWorkspace

from workspace.workspace_judge_utils import get_default_zone_rules, judge_workspace, make_error_payload


class WorkspaceJudgeNode(Node):
    # workspace_judge_node를 초기화하고 /judge_workspace 서비스 서버를 준비하는 함수
    def __init__(self):
        super().__init__('workspace_judge_node')

        self.judge_workspace_srv = self.create_service(
            JudgeWorkspace,
            '/judge_workspace',
            self.handle_judge_workspace
        )

        self.zone_rules = get_default_zone_rules()

        self.get_logger().info('WorkspaceJudgeNode started.')
        self.get_logger().info('/judge_workspace service ready.')

    # task_manager_node의 판단 요청을 받아 JSON을 파싱하고 정상/오배치 판단 결과를 반환하는 함수
    def handle_judge_workspace(self, request, response):
        self.get_logger().info('Received /judge_workspace request.')

        try:
            detected_payload = json.loads(request.detected_objects_json)
            objects = detected_payload.get('objects', [])
            frame = detected_payload.get('frame', 'unknown_frame')

            judgement_payload = judge_workspace(
                objects=objects,
                frame=frame,
                zone_rules=self.zone_rules
            )

            response.success = True
            response.judgement_json = json.dumps(judgement_payload, ensure_ascii=False)
            response.message = 'workspace judgement finished'

            self.get_logger().info(f'Judgement result: {response.judgement_json}')
            return response

        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON 파싱 실패: {e}')

            response.success = False
            response.judgement_json = json.dumps(
                make_error_payload('invalid_detected_objects_json'),
                ensure_ascii=False
            )
            response.message = 'invalid detected_objects_json'
            return response

        except Exception as e:
            self.get_logger().error(f'작업공간 판단 중 오류 발생: {e}')

            response.success = False
            response.judgement_json = json.dumps(
                make_error_payload('workspace_judgement_exception'),
                ensure_ascii=False
            )
            response.message = str(e)
            return response


# ROS2 workspace_judge_node를 실행하고 서비스 요청을 계속 처리하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = WorkspaceJudgeNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
