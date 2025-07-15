#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Joy
from std_msgs.msg import String, Bool
import tf2_ros

class PathScheduler(Node):
    def __init__(self):
        super().__init__('path_scheduler')
        self.declare_parameter('stop_interval', 4.0)
        self.stop_interval = max(
            0.1,
            self.get_parameter('stop_interval').get_parameter_value().double_value,
        )
        
        # TF and odometry setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.odom_received = False
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.animation_status_sub = self.create_subscription(
            String, '/hinge_animation_status', self.animation_status_callback, 10
        )
        self.animation_complete_sub = self.create_subscription(
            Bool, '/hinge_animation_complete', self.animation_complete_callback, 10
        )
        
        # Service clients
        self.generate_path_client = self.create_client(Trigger, 'generate_straight_path')
        self.pp_client = self.create_client(SetBool, 'pure_pursuit_toggle')
        self.hinges_down_client = self.create_client(SetBool, 'set_hinges_down')
        self.hinges_up_client = self.create_client(SetBool, 'set_hinges_up')
        
        # Timers
        self.check_timer = self.create_timer(0.5, self.check_and_generate)
        self.track_timer = self.create_timer(0.05, self.track_loop)
        
        # State management
        self.state = 'WAITING_TO_START'  # WAITING_TO_START, RUNNING, STOPPING, LOWERING, WAITING_FOR_BUTTON, RAISING
        self.dist_since_stop = 0.0
        self.last_pos = None
        self.prev_x_btn = 0
        self.path_generated = False
        self.start_button_pressed = False
        self.consecutive_ready_count = 0
        self.required_ready_count = 5
        self.log_counter = 0
        
        # Animation tracking
        self.animation_in_progress = False
        self.waiting_for_animation = False
        
        self.get_logger().info('Path Scheduler started with hinge animation integration')
        self.get_logger().info('States: WAITING_TO_START → RUNNING → STOPPING → LOWERING → WAITING_FOR_BUTTON → RAISING → RUNNING')
        self.get_logger().info(f'Stop interval: {self.stop_interval} meters')
        self.get_logger().info('Press button [0] to start path generation and begin movement')

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
        # Don't generate path until start button is pressed
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
                # Start running after path generation
                self.enter_running()
                self.get_logger().info('Path generated successfully, starting robot movement')
        except Exception as e:
            self.get_logger().error(f'Path generation failed: {str(e)}')

    def track_loop(self):
        # Only track distance when actually running
        if self.state != 'RUNNING':
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
            self.enter_stopping_sequence()
        self.log_status()

    def animation_status_callback(self, msg):
        """Handle animation status updates"""
        status = msg.data
        self.get_logger().info(f'Animation status: {status}')
        
        if "ANIMATING" in status:
            self.animation_in_progress = True
        elif "COMPLETE" in status:
            self.animation_in_progress = False
            
            if status == "COMPLETE_DOWN" and self.state == "LOWERING":
                self.enter_waiting_for_button()
            elif status == "COMPLETE_UP" and self.state == "RAISING":
                self.enter_running()

    def animation_complete_callback(self, msg):
        """Handle animation completion signal"""
        if msg.data:
            self.get_logger().info('Animation completed')
            self.waiting_for_animation = False

    def joy_callback(self, msg: Joy):
        x_btn = msg.buttons[0] if msg.buttons else 0
        
        # Handle start button press when waiting to start
        if self.state == 'WAITING_TO_START' and self.prev_x_btn == 0 and x_btn == 1:
            self.get_logger().info('Start button pressed - initializing path generation')
            self.start_button_pressed = True
            self.state = 'INITIALIZING'
            
        # Handle continue button press when waiting for button during loading
        elif self.state == 'WAITING_FOR_BUTTON' and self.prev_x_btn == 0 and x_btn == 1:
            self.get_logger().info('Continue button pressed - starting raise sequence')
            self.enter_raising_sequence()
            
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

    def call_hinges_service(self, service_client, service_name):
        """Call hinge service with error handling"""
        if service_client.wait_for_service(timeout_sec=1.0):
            future = service_client.call_async(SetBool.Request(data=True))
            self.get_logger().info(f'Called {service_name} service')
            return True
        else:
            self.get_logger().error(f'{service_name} service not available')
            return False

    def enter_stopping_sequence(self):
        """Stop robot and start lowering hinges"""
        self.state = 'STOPPING'
        self.call_pp(False)  # Stop path following
        self.get_logger().info(f'STOPPING: Traveled {self.dist_since_stop:.2f}m, stopping robot')
        
        # Start lowering hinges
        self.enter_lowering_sequence()

    def enter_lowering_sequence(self):
        """Start hinge lowering animation"""
        self.state = 'LOWERING'
        self.waiting_for_animation = True
        
        success = self.call_hinges_service(self.hinges_down_client, 'set_hinges_down')
        if success:
            self.get_logger().info('LOWERING: Started hinge down animation for loading')
        else:
            # If service call fails, skip to waiting
            self.get_logger().warn('LOWERING: Service call failed, skipping to waiting state')
            self.enter_waiting_for_button()

    def enter_waiting_for_button(self):
        """Wait for button press to continue"""
        self.state = 'WAITING_FOR_BUTTON'
        self.get_logger().info('WAITING_FOR_BUTTON: Hinges lowered, press button [0] to continue loading')

    def enter_raising_sequence(self):
        """Start hinge raising animation"""
        self.state = 'RAISING'
        self.waiting_for_animation = True
        
        success = self.call_hinges_service(self.hinges_up_client, 'set_hinges_up')
        if success:
            self.get_logger().info('RAISING: Started hinge up animation for transport')
        else:
            # If service call fails, skip to running
            self.get_logger().warn('RAISING: Service call failed, resuming immediately')
            self.enter_running()

    def enter_running(self):
        """Resume robot movement"""
        self.state = 'RUNNING'
        self.dist_since_stop = 0.0
        self.call_pp(True)  # Resume path following
        self.get_logger().info('RUNNING: Hinges raised, resuming robot movement')

    def log_status(self):
        self.log_counter += 1
        if self.log_counter >= 40:  # Log every 2 seconds (40 * 50ms)
            self.log_counter = 0
            
            if self.state == 'WAITING_TO_START':
                self.get_logger().info(f'State: {self.state}, Press button [0] to start path generation')
            elif self.state == 'INITIALIZING':
                ready = self.odom_received and self.check_tf_ready() and self.generate_path_client.wait_for_service(timeout_sec=0.1)
                self.get_logger().info(f'State: {self.state}, Checking readiness... Ready: {ready}')
            elif self.state == 'RUNNING':
                self.get_logger().info(f'State: {self.state}, Distance: {self.dist_since_stop:.2f}m')
            elif self.state == 'LOWERING':
                self.get_logger().info(f'State: {self.state}, Animation: {"In Progress" if self.animation_in_progress else "Waiting"}')
            elif self.state == 'WAITING_FOR_BUTTON':
                self.get_logger().info(f'State: {self.state}, Press button [0] to continue loading')
            elif self.state == 'RAISING':
                self.get_logger().info(f'State: {self.state}, Animation: {"In Progress" if self.animation_in_progress else "Waiting"}')
            else:
                self.get_logger().info(f'State: {self.state}')


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