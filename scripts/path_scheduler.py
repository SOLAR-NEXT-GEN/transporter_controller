#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Joy
import tf2_ros

class PathScheduler(Node):
    def __init__(self):
        super().__init__('path_scheduler')
        self.declare_parameter('stop_interval', 4.0)
        self.stop_interval = max(
            0.1,
            self.get_parameter('stop_interval').get_parameter_value().double_value,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.odom_received = False
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.generate_path_client = self.create_client(Trigger, 'generate_straight_path')
        self.pp_client = self.create_client(SetBool, 'pure_pursuit_toggle')
        self.check_timer = self.create_timer(0.5, self.check_and_generate)
        self.track_timer = self.create_timer(0.05, self.track_loop)
        self.state = 'RUNNING'
        self.dist_since_stop = 0.0
        self.last_pos = None
        self.prev_x_btn = 0
        self.path_generated = False
        self.consecutive_ready_count = 0
        self.required_ready_count = 5
        self.log_counter = 0
        self.get_logger().info('Path Scheduler started')

    def odom_callback(self, _msg):
        self.odom_received = True

    def check_tf_ready(self):
        try:
            now = rclpy.time.Time()
            self.tf_buffer.lookup_transform('map', 'base_footprint', now, timeout=rclpy.duration.Duration(seconds=0.1))
            return True
        except Exception:
            return False

    def check_and_generate(self):
        if self.path_generated:
            return
        ready = self.odom_received and self.check_tf_ready() and self.generate_path_client.wait_for_service(timeout_sec=0.5)
        if not ready:
            self.consecutive_ready_count = 0
            return
        self.consecutive_ready_count += 1
        if self.consecutive_ready_count < self.required_ready_count:
            return
        self.generate_path_client.call_async(Trigger.Request()).add_done_callback(self.path_generation_callback)

    def path_generation_callback(self, future):
        try:
            if future.result().success:
                self.path_generated = True
                self.check_timer.cancel()
        except Exception:
            pass

    def track_loop(self):
        if self.state == 'STOPPED':
            self.log_status()
            return
        x, y = self.current_xy()
        if x is None:
            return
        if self.last_pos is None:
            self.last_pos = (x, y)
            return
        dx, dy = x - self.last_pos[0], y - self.last_pos[1]
        self.dist_since_stop += math.hypot(dx, dy)
        self.last_pos = (x, y)
        if self.dist_since_stop >= self.stop_interval:
            self.enter_stopped()
        self.log_status()

    def joy_callback(self, msg: Joy):
        x_btn = msg.buttons[0] if msg.buttons else 0
        if self.state == 'STOPPED' and self.prev_x_btn == 0 and x_btn == 1:
            self.exit_stopped()
        self.prev_x_btn = x_btn

    def current_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            t = tf.transform.translation
            return t.x, t.y
        except Exception:
            return None, None

    def call_pp(self, enable: bool):
        if self.pp_client.wait_for_service(timeout_sec=0.2):
            self.pp_client.call_async(SetBool.Request(data=enable))

    def enter_stopped(self):
        self.state = 'STOPPED'
        self.call_pp(False)
        self.get_logger().info(f'state={self.state}, distance={self.dist_since_stop:.2f} m')

    def exit_stopped(self):
        self.state = 'RUNNING'
        self.dist_since_stop = 0.0
        self.call_pp(True)
        self.get_logger().info(f'state={self.state}, distance={self.dist_since_stop:.2f} m')

    def log_status(self):
        self.log_counter += 1
        if self.log_counter >= 20:
            self.log_counter = 0
            self.get_logger().info(f'state={self.state}, distance={self.dist_since_stop:.2f} m')


def main(args=None):
    rclpy.init(args=args)
    node = PathScheduler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
