import rclpy
from rclpy.node import Node
from aruco_manager_interface.srv import ArucoManager as ArucoManagerSrv

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import math
from std_msgs.msg import String

class ArucoManager(Node):
    def __init__(self):
        super().__init__('aruco_manager')
        
        #creating service
        self.service = self.create_service(
            srv_type=ArucoManagerSrv,
            srv_name='aruco_manager_service',
            callback=self.service_callback
        )

        self.aruco_sub = None

        #goto goal
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._goal_handle = None
        self.get_logger().info('Waiting for Nav2 action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Nav2 action server available!')

    
    def service_callback(self, request, response):
        print("Received request: ", request)

        command = request.command.lower()
        x = float(request.x)
        y = float(request.y)
        action = request.action.lower()

        try:
            if command == "goto" and action == 'start':
                self.send_goal(x, y)
                response.success = True
                response.msg = f"Sent navigation goal to x: {x}, y: {y}"
            
            elif command == "aruco" and action == "start":
                if self.aruco_sub is None:
                    self.aruco_sub = self.create_subscription(
                        String,
                        'aruco_detections',
                        self.aruco_callback,
                        10
                    )
                    response.success = True
                    response.msg = "Started ArUco marker detection."
                else:
                    response.success = False
                    response.msg = "something went wrong with aruco detection"

                # pass

            elif action == 'stop':
                self.stop_goal()
                response.success = True
                response.msg = "Sent stop command to cancel the current goal."
            else:
                response.success = False
                response.msg = f"Unknown command: {command} or action: {action}"
        except Exception as e:
            self.get_logger().error(f"Error processing request: {e}")
            response.success = False
            response.msg = f"Error processing request: {e}"

        return response

    def aruco_callback(self, msg):
        print("Received ArUco detection: ", msg.data)
    


    def send_goal(self, x: float, y: float, yaw: float = 0.0):
        goal_msg = NavigateToPose.Goal()
        
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f'Navigating to: x={x}, y={y}, yaw={yaw}')
        
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
        
        return send_goal_future

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return
        
        self.get_logger().info('Goal accepted!')
        self._goal_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        
        if status == 4:  
            self.get_logger().info('Goal reached successfully!')
        elif status == 5:  
            self.get_logger().warn('Goal was canceled')
        elif status == 6: 
            self.get_logger().error('Goal was aborted')
        else:
            self.get_logger().info(f'Navigation finished with status: {status}')
        
        self._goal_handle = None

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose.position
        self.get_logger().info(
            f'Current position: x={current_pose.x:.2f}, y={current_pose.y:.2f}',
            throttle_duration_sec=2.0
        )
    
    def stop_goal(self):
        if self._goal_handle is not None:
            self.get_logger().info('Canceling current goal...')
            self._goal_handle.cancel_goal_async()
        else:
            self.get_logger().info('No active goal to stop.')
    

def main():
    rclpy.init()
    node = ArucoManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()