import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
import os

class StereoDepth(Node):
    def __init__(self):
        super().__init__('stereo_depth')

        self.declare_parameter('left_topic', '/a15/camera/left/image_raw')
        self.declare_parameter('right_topic', '/a15/camera/right/image_raw')
        self.declare_parameter('depth_topic', '/a15/camera/stereo/depth')
        self.declare_parameter('calib_file', 'config/stereo_camera.yaml')

        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        calib_file = self.get_parameter('calib_file').value

        self.bridge = CvBridge()
        self.load_calibration(calib_file)

        # Precompute rectification maps
        self.init_rectify_maps()

        # Setup stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=9,
            P1=8 * 3 * 9 ** 2,
            P2=32 * 3 * 9 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        # Sync subscribers
        left_sub = message_filters.Subscriber(self, Image, left_topic)
        right_sub = message_filters.Subscriber(self, Image, right_topic)

        ts = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=30, slop=0.5)
        ts.registerCallback(self.image_callback)

        self.depth_pub = self.create_publisher(Image, depth_topic, 10)

        print("init done")

    def load_calibration(self, calib_path):
        package_share_directory = get_package_share_directory('aqua_line_follower')  # Replace with actual package name
        calib_path = os.path.join(package_share_directory, calib_path)

        with open(calib_path, 'r') as file:
            calib = yaml.safe_load(file)

        self.K1 = np.array(calib['left']['camera_matrix']['data']).reshape(3, 3)
        self.D1 = np.array(calib['left']['distortion_coefficients']['data'])
        self.K2 = np.array(calib['right']['camera_matrix']['data']).reshape(3, 3)
        self.D2 = np.array(calib['right']['distortion_coefficients']['data'])
        self.R = np.array(calib['extrinsics']['rotation']).reshape(3, 3)
        self.T = np.array(calib['extrinsics']['translation']).reshape(3, 1)

        self.image_size = (calib['image_width'], calib['image_height'])

        # stereoRectify
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2,
            self.image_size, self.R, self.T, alpha=0
        )

    def init_rectify_maps(self):
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_16SC2)

    def image_callback(self, left_msg, right_msg):
        print("callback called")
        left_raw = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_raw = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        # Rectify images
        left = cv2.remap(left_raw, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right = cv2.remap(right_raw, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity[disparity <= 0.0] = 0.1

        depth_map = cv2.reprojectImageTo3D(disparity, self.Q)[:, :, 2]

        depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')
        depth_msg.header = left_msg.header
        self.depth_pub.publish(depth_msg)
        print("depth image published")

def main(args=None):
    rclpy.init(args=args)
    node = StereoDepth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
