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
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.odom_received = False
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # Existing clients
        self.generate_path_client = self.create_client(Trigger, 'generate_straight_path')
        self.pp_client = self.create_client(SetBool, 'pure_pursuit_toggle')
        
        # New manipulator clients
        self.mani_up_client = self.create_client(Trigger, 'mani_control_up')
        self.mani_down_client = self.create_client(Trigger, 'mani_control_down')
        
        self.check_timer = self.create_timer(0.5, self.check_and_generate)
        self.track_timer = self.create_timer(0.05, self.track_loop)
        
        self.state = 'WAITING_TO_START'
        self.prev_x_btn = 0
        self.prev_stop_btn = 0
        self.path_generated = False
        self.start_button_pressed = False
        self.consecutive_ready_count = 0
        self.required_ready_count = 5
        self.log_counter = 0
        
        # New state tracking for manipulator operations
        self.pending_operation = None  # 'UP' or 'DOWN' or None
        
        self.get_logger().info('Path Scheduler with Manipulator Control started')
        self.get_logger().info('Button[0]: Start/Continue (will move manipulator UP first)')
        self.get_logger().info('Button[2]: Stop (will stop controller then move manipulator DOWN)')

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
        if not self.start_button_pressed:
            return
            
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
                self.get_logger().info('Path generated successfully')
                # Don't start robot movement yet - wait for UP command to complete
                # The UP operation will be triggered by button press
        except Exception as e:
            self.get_logger().error(f'Path generation failed: {str(e)}')

    def track_loop(self):
        self.log_status()

    def joy_callback(self, msg: Joy):
        if not msg.buttons or len(msg.buttons) < 3:
            return
            
        x_btn = msg.buttons[0]
        stop_btn = msg.buttons[2]
        
        # Ignore button presses if manipulator operation is pending
        if self.pending_operation is not None:
            self.get_logger().info(f'Ignoring button press - {self.pending_operation} operation in progress')
            return
        
        if self.prev_x_btn == 0 and x_btn == 1:
            if self.state == 'WAITING_TO_START':
                self.get_logger().info('Start button pressed - initializing path generation')
                self.start_button_pressed = True
                self.state = 'INITIALIZING'
                
            elif self.state == 'STOPPED':
                self.get_logger().info('Continue button pressed - moving manipulator UP first')
                self.state = 'MOVING_UP'
                self.call_mani_up()
            
        elif self.prev_stop_btn == 0 and stop_btn == 1:
            if self.state == 'RUNNING':
                self.get_logger().info('Stop button pressed - stopping controller first')
                self.call_pp(False)
                self.state = 'MOVING_DOWN'
                self.call_mani_down()
            
        self.prev_x_btn = x_btn
        self.prev_stop_btn = stop_btn

    def call_mani_up(self):
        """Call manipulator UP service"""
        if not self.mani_up_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Manipulator UP service not available')
            return
            
        self.pending_operation = 'UP'
        self.get_logger().info('Calling manipulator UP service...')
        future = self.mani_up_client.call_async(Trigger.Request())
        future.add_done_callback(self.mani_up_callback)

    def mani_up_callback(self, future):
        """Handle manipulator UP service response"""
        self.pending_operation = None
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Manipulator UP completed: {response.message}')
                # Now start the robot movement
                if self.state == 'MOVING_UP':
                    self.state = 'RUNNING'
                    self.call_pp(True)
                    self.get_logger().info('Starting robot movement')
            else:
                self.get_logger().error(f'Manipulator UP failed: {response.message}')
                self.state = 'STOPPED'
        except Exception as e:
            self.get_logger().error(f'Manipulator UP service call failed: {str(e)}')
            self.state = 'STOPPED'

    def call_mani_down(self):
        """Call manipulator DOWN service"""
        if not self.mani_down_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Manipulator DOWN service not available')
            return
            
        self.pending_operation = 'DOWN'
        self.get_logger().info('Calling manipulator DOWN service...')
        future = self.mani_down_client.call_async(Trigger.Request())
        future.add_done_callback(self.mani_down_callback)

    def mani_down_callback(self, future):
        """Handle manipulator DOWN service response"""
        self.pending_operation = None
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Manipulator DOWN completed: {response.message}')
                self.state = 'STOPPED'
            else:
                self.get_logger().error(f'Manipulator DOWN failed: {response.message}')
                self.state = 'STOPPED'
        except Exception as e:
            self.get_logger().error(f'Manipulator DOWN service call failed: {str(e)}')
            self.state = 'STOPPED'

    def call_pp(self, enable: bool):
        if self.pp_client.wait_for_service(timeout_sec=0.2):
            self.pp_client.call_async(SetBool.Request(data=enable))

    def log_status(self):
        self.log_counter += 1
        if self.log_counter >= 40:
            self.log_counter = 0
            
            if self.state == 'WAITING_TO_START':
                self.get_logger().info(f'State: {self.state} - Press button[0] to start')
            elif self.state == 'INITIALIZING':
                ready = self.odom_received and self.check_tf_ready() and self.generate_path_client.wait_for_service(timeout_sec=0.1)
                self.get_logger().info(f'State: {self.state} - Ready: {ready}')
            elif self.state == 'MOVING_UP':
                op_status = f" - {self.pending_operation} in progress" if self.pending_operation else ""
                self.get_logger().info(f'State: {self.state}{op_status}')
            elif self.state == 'RUNNING':
                self.get_logger().info(f'State: {self.state} - Press button[2] to stop')
            elif self.state == 'MOVING_DOWN':
                op_status = f" - {self.pending_operation} in progress" if self.pending_operation else ""
                self.get_logger().info(f'State: {self.state}{op_status}')
            elif self.state == 'STOPPED':
                self.get_logger().info(f'State: {self.state} - Press button[0] to continue')


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