#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
import tf2_ros


class PathScheduler(Node):
    def __init__(self):
        super().__init__('path_scheduler')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.odom_received = False
        self.tf_ready = False
        self.path_generated = False
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.generate_path_client = self.create_client(
            Trigger,
            'generate_straight_path'
        )
        
        self.check_timer = self.create_timer(0.5, self.check_and_generate)
        
        self.consecutive_ready_count = 0
        self.required_ready_count = 5
        
        self.get_logger().info('Path Scheduler started')
        
    def odom_callback(self, msg):
        self.odom_received = True
        
    def check_tf_ready(self):
        try:
            now = rclpy.time.Time()
            self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                now,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            self.tf_ready = True
            return True
        except:
            return False
            
    def check_and_generate(self):
        if self.path_generated:
            return
            
        if not self.odom_received:
            self.get_logger().info('Waiting for odometry...', throttle_duration_sec=2.0)
            self.consecutive_ready_count = 0
            return
            
        if not self.check_tf_ready():
            self.get_logger().info('Waiting for TF...', throttle_duration_sec=2.0)
            self.consecutive_ready_count = 0
            return
            
        if not self.generate_path_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info('Waiting for path generator service...', throttle_duration_sec=2.0)
            self.consecutive_ready_count = 0
            return
            
        self.consecutive_ready_count += 1
        
        if self.consecutive_ready_count < self.required_ready_count:
            self.get_logger().info(f'Systems ready check: {self.consecutive_ready_count}/{self.required_ready_count}')
            return
            
        self.get_logger().info('All systems stable! Generating path...')
        
        request = Trigger.Request()
        future = self.generate_path_client.call_async(request)
        future.add_done_callback(self.path_generation_callback)
        
    def path_generation_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.path_generated = True
                self.get_logger().info(f'Path generation successful: {response.message}')
                self.check_timer.cancel()
            else:
                self.get_logger().error(f'Path generation failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PathScheduler()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()