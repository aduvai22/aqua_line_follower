# =======================================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =======================================================

import os
# import cv2
import time
from datetime import datetime
import rclpy
import onnxruntime as ort
import yaml
import socket
from rclpy.node import Node
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ament_index_python.packages import get_package_share_directory
from .perception_util import InferenceOnFrame

class Detector(Node):
    """
    This ROS2 node detects QR codes from front camera and cavelines from down camera.
    The detected lines are published as a Float32MultiArray message.
    The processed segmentation map image is published as an Image message.
    The QR code information is published as an Int32 message.
    """
    def __init__(self):
        super().__init__('detector')

        # Package directory
        package_share = get_package_share_directory('aqua_line_follower')

        # Read config params
        config_file = os.path.join(package_share, 'config', 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.PUBLISH_RATE = config['PUBLISH_RATE']  # 5 Hz
        self.SLEEP_TIME = 1.0 / self.PUBLISH_RATE

        # Create publishers
        self.map_pub = self.create_publisher(Image, '/map', 10)
        self.lines_pub = self.create_publisher(Float32MultiArray, '/detected_lines', 10)

        # CVBridge for converting OpenCV images to ROS Image messages
        self.bridge = CvBridge()


        # Read engine file and prepare bindings
        model_name = 'mobilenet.onnx'
        self.model_path = os.path.join(package_share, 'weights', model_name)
        self.get_logger().info("Loading ONNX model from: " + self.model_path)
        

        # Load ONNX model
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/a15/camera/back/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.map_pub = self.create_publisher(Image, '/map', 10)
        self.lines_pub = self.create_publisher(Float32MultiArray, '/detected_lines', 10)

        self.get_logger().info("Waiting for image")

        # Set processing rate (10 Hz) using a timer
        # self.timer = self.create_timer(1.0 / 10.0, self.timer_callback)

    def image_callback(self, msg):
        """
        Callback  function to keep the node alive.
        """
        try:
            
            # --- Process downward camera for line detection ---

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Check if the frame was read successfully
            if frame is None:
                self.get_logger().warn("Failed to read frame from downward camera.")
            else:
                frame_height, frame_width = frame.shape[:2]
                screen_center_x = frame_width // 2
                screen_center_y = frame_height // 2

                # Detect cavelines on the downward camera frame
                self.get_logger().info("Inferencing on downward frame...")
                lines, line_overlayed_map = InferenceOnFrame(
                    self.session, self.input_name, frame)
                
                # Publish the processed frame
                map_msg = self.bridge.cv2_to_imgmsg(line_overlayed_map, encoding='bgr8')
                self.map_pub.publish(map_msg)

                if lines is not None:
                    # Publish detected line coordinates
                    flat_lines = [float(coord) for line in lines for coord in line]  # flatten the list of lines
                    lines_msg = Float32MultiArray()
                    lines_msg.data = flat_lines
                    self.lines_pub.publish(lines_msg)             

        except Exception as e:
            self.get_logger().error("Exception in callback: {}".format(e))


def main(args=None):
    """
    Main function to run the Detector node.
    """
    # Initialize the ROS2 node
    rclpy.init(args=args)
    detector = Detector()

    try:
        # Spin the node to keep it alive and processing
        rclpy.spin(detector)
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        detector.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        # Clean up and shutdown
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()