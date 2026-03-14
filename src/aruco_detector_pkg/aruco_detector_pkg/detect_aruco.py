import cv2
import numpy as np
import cv2.aruco as aruco

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from nav_msgs.msg import Odometry


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        #parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("marker_size", 0.175),
                ("aruco_dict_type", "DICT_4X4_250"),
                ("image_topic_name", "/zed/zed_node/rgb/color/rect/image"),
                ("camera_info_topic", "/zed/zed_node/rgb/color/rect/camera_info"),
                ("odom_topic", "/odom")
            ]
        )

        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.aruco_dict_type = self.get_parameter("aruco_dict_type").get_parameter_value().string_value
        self.image_topic_name = self.get_parameter("image_topic_name").get_parameter_value().string_value
        self.camera_info_topic_name = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.odom_topic_name = self.get_parameter("odom_topic").get_parameter_value().string_value

        #debug
        self.get_logger().info(f'Marker size is set to: {self.marker_size}')
        self.get_logger().info(f'Aruco dict type: {self.aruco_dict_type}')
        self.get_logger().info(f'Image topic name: {self.image_topic_name}')
        self.get_logger().info(f'Camera info topic name: {self.camera_info_topic_name}')
        self.get_logger().info(f'Odometry topic name: {self.odom_topic_name}')

        self.visualization_enabled = True

        #subscriber for camera_matrix and dist_coeffs
        self.camera_info_subscriber = self.create_subscription(
            msg_type=CameraInfo,
            topic=self.camera_info_topic_name,
            callback=self.camera_info_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.get_logger().info(f'Subscribed to {self.camera_info_topic_name}')
        #information from camera_info topic
        self.camera_matrix = None
        self.dist_coeffs = None


        #subscriber for image from zed cam
        self.image_subscriber = self.create_subscription(
            msg_type=Image,
            topic=self.image_topic_name,
            callback=self.image_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.get_logger().info(f'Subscribed to {self.image_topic_name}')

        #subscribe for odometry
        self.odom_subscriber = self.create_subscription(
            msg_type=Odometry,
            topic=self.odom_topic_name,
            callback=self.odom_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.odom_x = 0
        self.odom_y = 0
        self.odom_z = 0

        #required to convert between ROS and OpenCV images
        # self.bridge = CvBridge()

        #result
        self.detections = None

        #publisher
        self.publisher = self.create_publisher(
            msg_type=String, 
            topic="aruco_detections", 
            qos_profile=5
        )
        #publishes at 1Hz
        self.timer = self.create_timer(1, self.publisher_callback)
    
    def odom_callback(self, msg):
        try:
            self.odom_x = msg.pose.pose.position.x
            self.odom_y = msg.pose.pose.position.y
            self.odom_z = msg.pose.pose.position.z
            if self.detections is not None:
                for det in self.detections:
                    det["position"][0] += self.odom_x
                    det["position"][1] += self.odom_y
                    det["position"][2] += self.odom_z
                
            self.get_logger().info(f'Received odometry: x={self.odom_x}, y={self.odom_y}, z={self.odom_z}')
        except Exception as e:
            self.get_logger().error(f'Error processing odometry data: {e}')


    def camera_info_callback(self, msg):
        try:
            self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)
            self.get_logger().info('Camera parameters received.')

            self.destroy_subscription(self.camera_info_subscriber)
        except Exception as e:
            self.get_logger().error(f'Error processing camera info: {e}')
    
    def image_callback(self, msg):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn('Waiting for camera parameters...', throttle_duration_sec=2.0)

            return
        
        try:
            # current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            #bypassign cv_bridge
            #Convert byte stream to numpy array
            im_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
            #Handle encoding (ZED usually sends 'bgra8' or 'bgr8')
            if msg.encoding == 'bgra8':
                current_frame = cv2.cvtColor(im_np, cv2.COLOR_BGRA2BGR)
            elif msg.encoding == 'rgb8':
                current_frame = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
            else:
                current_frame = im_np.copy()
        
            detections, frame = self.detect_aruco_default(
                current_frame,
                self.camera_matrix,
                self.dist_coeffs,
                aruco_dict_type=getattr(cv2.aruco, self.aruco_dict_type),
                marker_length=self.marker_size
            )

            # detections, frame = self.detect_aruco(
            #     current_frame,
            #     self.camera_matrix,
            #     self.dist_coeffs
            #     marker_length=self.marker_size
            # )

            # eucledian distance
            for det in detections:
                pos = det["tvec"]
                distance = np.linalg.norm(pos)
                print(f"Marker ID {det['id']} distance: {distance:.3f} m")

            self.detections = detections #making result available

            if self.visualization_enabled:
                try:
                    cv2.imshow("Aruco Detection", current_frame)
                    cv2.waitKey(1)
                except cv2.error as e:
                    self.visualization_enabled = False
                    self.get_logger().warn(f'OpenCV HighGUI unavailable; disabling visualization: {e}')
        except Exception as e:
            self.get_logger().error(f'Detection failed: {e}')

    def publisher_callback(self):
        if self.detections is not None and len(self.detections) > 0:
            msg = String()
            msg.data = str(self.detections)
            self.publisher.publish(msg)
            self.get_logger().info(f'Published detections: {msg.data}')

    def _create_detector_parameters(self):
        if hasattr(cv2.aruco, 'DetectorParameters'):
            return cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            return cv2.aruco.DetectorParameters_create()
        raise AttributeError(
            "Your OpenCV ArUco module does not provide DetectorParameters APIs. "
            "Install opencv-contrib-python matching your Python/ROS environment."
        )


    def detect_aruco(self, frame, 
                    camera_matrix,
                    dist_coeffs,
                    aruco_dict_type=cv2.aruco.DICT_4X4_250,
                    marker_length=0.2):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = self._create_detector_parameters()

        # 1. Detection (The part that usually works)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        detections = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 2. Manual 3D points for solvePnP (The stable replacement for pose estimation)
            # These are the coordinates of the four corners in the marker's local frame
            obj_points = np.array([
                [-marker_length/2,  marker_length/2, 0],
                [ marker_length/2,  marker_length/2, 0],
                [ marker_length/2, -marker_length/2, 0],
                [-marker_length/2, -marker_length/2, 0]
            ], dtype=np.float32)

            for i in range(len(ids)):
                # 3. Use solvePnP - This is a core OpenCV function and won't Segfault
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    corners[i], 
                    camera_matrix, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    x, y, z = tvec.flatten()
                    detections.append({
                        "id": int(ids[i][0]),
                        "position": [z, -x, -y], #analogous to ros2
                        # "rvec": rvec,
                        # "tvec": tvec
                    })

                    # Draw the axes so you can see it's working
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
        
        return detections, frame


    def detect_aruco_default(self, frame, 
                    camera_matrix,
                    dist_coeffs,
                    aruco_dict_type=cv2.aruco.DICT_4X4_250,
                    marker_length=0.2):
        """This is the original detection method that uses the built-in OpenCV pose estimation.
        It's more concise but can be unstable with certain OpenCV versions. If it doe not work,
        switch to the detect_aruco() method which uses solvePnP directly."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = self._create_detector_parameters()

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        detections = []

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Use OpenCV contrib function to estimate pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                # Draw axis on the marker
                # cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvec, 0.05)
                x, y, z = tvec.flatten()
                detections.append({
                    "id": int(ids[i][0]),
                    "position": [z, -x, -y], #analogous to ros2
                    # "rvec": rvec,
                    # "tvec": tvec
                })

        return detections, frame


def main():
    rclpy.init()
    aruco_detector_node = ArucoDetector()

    try:
        rclpy.spin(aruco_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(aruco_detector_node, 'visualization_enabled') and aruco_detector_node.visualization_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        aruco_detector_node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == '__main__':
    main()