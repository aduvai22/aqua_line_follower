import rclpy
from rclpy.node import Node
from aqua2_navigation.swimmer import SwimmerAPI
from std_msgs.msg import Float32

class SwimmerWrapper(Node):
    def __init__(self):
        self.swimmer = SwimmerAPI()  # rclpy.init() is called here internally
        self.node = self.swimmer  # Since SwimmerAPI is also a rclpy Node
        self.node.get_logger().info("Swimmer Node ready!")
        
        # is_calibrated = self.swimmer.calibrate()
        
        # while not is_calibrated:
        #     is_calibrated = self.swimmer.calibrate()
        for i in range(3):
            self.node.set_mode("swimmode",timeout=60)
            self.node.set_autopilot_mode("depth",timeout=60)
            self.node.zero_local_pose(timeout=60)
            self.node.zero_heading(timeout=60)

        self.node.create_subscription(Float32, '/a15/line_follower/angle', self.swim_callback, 10)


    def swim_callback(self, msg):
        self.node.timed_swim(speed=0.6, depth=0.5, yaw=msg.data, heave=0.0, pitch=0.0, roll=0.0, duration=0.5)    

def main():
    wrapper = SwimmerWrapper()
    try:
        # Spin the node to keep it alive and processing
        rclpy.spin(wrapper.node)
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        wrapper.node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        # Clean up and shutdown
        wrapper.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
