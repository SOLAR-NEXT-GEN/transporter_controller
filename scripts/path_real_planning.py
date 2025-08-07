#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Joy
import tf2_ros

class PathScheduler(Node):
    def __init__(self):
        super().__init__('path_scheduler')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.odom_received = False
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.state_sub = self.create_subscription(Float64MultiArray, 'hinge_state', self.state_callback, 10)
        
        # Service clients
        self.generate_path_client = self.create_client(Trigger, 'generate_straight_path')
        self.pp_client = self.create_client(SetBool, 'pure_pursuit_toggle')
        self.mani_up_client = self.create_client(Trigger, 'mani_control_up')
        self.mani_down_client = self.create_client(Trigger, 'mani_control_down')
        
        # Timers
        self.check_timer = self.create_timer(0.5, self.check_and_generate)
        self.track_timer = self.create_timer(0.05, self.track_loop)
        
        # State management
        self.state = 'WAITING_TO_START'
        self.prev_x_btn = 0
        self.prev_stop_btn = 0
        self.path_generated = False
        self.start_button_pressed = False
        self.consecutive_ready_count = 0
        self.required_ready_count = 5
        self.log_counter = 0
        self.is_walking = False
        
        # Flex sensor management (from TryManiControl)
        self.current_states = [0, 0]
        self.flex_sensors = [0, 0]
        self.flex_timer = None
        self.flex_triggered = False
        
        self.get_logger().info('Integrated Path Scheduler with Manipulator Control started')
        self.get_logger().info('Button[0]: Start (UP -> walk)')
        self.get_logger().info('Button[2]: Stop (stop -> DOWN)')
        self.get_logger().info('Flex sensor: Auto stop when flex=1 detected (2s delay)')

    def odom_callback(self, _msg):
        self.odom_received = True

    def state_callback(self, msg):
        """Monitor hinge states and flex sensors (from TryManiControl)"""
        if len(msg.data) >= 4:
            old_flex = self.flex_sensors.copy()
            self.current_states = [int(msg.data[0]), int(msg.data[1])]
            self.flex_sensors = [int(msg.data[2]), int(msg.data[3])]
            
            # Check for flex sensor trigger (auto stop with 2-second delay)
            if self.is_walking and (self.flex_sensors[0] == 1 or self.flex_sensors[1] == 1):
                if old_flex != self.flex_sensors and not self.flex_triggered:  # Only trigger on change
                    self.get_logger().info(f'FLEX SENSOR TRIGGERED! FL={self.flex_sensors[0]}, BL={self.flex_sensors[1]}')
                    self.get_logger().info('Walking for 2 more seconds before stopping...')
                    self.flex_triggered = True
                    
                    # Start 2-second timer
                    if self.flex_timer:
                        self.flex_timer.cancel()
                    self.flex_timer = self.create_timer(2.0, self.flex_timer_callback)

    def flex_timer_callback(self):
        """Called after 2-second delay from flex trigger (from TryManiControl)"""
        self.get_logger().info('2-second delay complete - now stopping')
        self.flex_timer.cancel()
        self.flex_timer = None
        self.flex_triggered = False
        self.handle_stop("flex sensor trigger (after 2s delay)")

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
                self.get_logger().info('Path generated successfully - Starting robot sequence')
                
                # Automatically start the robot sequence
                self.handle_start()  # This will do: UP -> enable pure pursuit
                
        except Exception as e:
            self.get_logger().error(f'Path generation failed: {str(e)}')

    def track_loop(self):
        self.log_status()

    def joy_callback(self, msg: Joy):
        """Handle joystick button presses"""
        if not msg.buttons or len(msg.buttons) < 3:
            return
            
        x_btn = msg.buttons[0]
        stop_btn = msg.buttons[2]
        
        # Start button pressed
        if self.prev_x_btn == 0 and x_btn == 1:
            if self.state == 'WAITING_TO_START':
                self.get_logger().info('Start button pressed - initializing path generation')
                self.start_button_pressed = True
                self.state = 'INITIALIZING'
                
            elif self.state == 'READY' and not self.is_walking:
                self.handle_start()
            else:
                self.get_logger().info('Already walking or not ready - ignoring start button')
        
        # Stop button pressed
        elif self.prev_stop_btn == 0 and stop_btn == 1:
            if self.is_walking:
                self.handle_stop("button press")
            else:
                self.get_logger().info('Not walking - ignoring stop button')
        
        self.prev_x_btn = x_btn
        self.prev_stop_btn = stop_btn

    def handle_start(self):
        """Handle start sequence: UP -> walk (from TryManiControl logic)"""
        self.get_logger().info('=== START SEQUENCE ===')
        self.state = 'STARTING'
        
        # Call UP service and wait for success
        if not self.mani_up_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Manipulator UP service not available')
            self.state = 'READY'
            return
            
        self.get_logger().info('Calling UP service...')
        request = Trigger.Request()
        
        try:
            future = self.mani_up_client.call_async(request)
            future.add_done_callback(self.up_service_callback)
                
        except Exception as e:
            self.get_logger().error(f'UP service call failed: {str(e)}')
            self.state = 'READY'

    def handle_stop(self, trigger_source):
        """Handle stop sequence: stop -> DOWN (from TryManiControl logic)"""
        self.get_logger().info(f'=== STOP SEQUENCE (triggered by {trigger_source}) ===')
        self.state = 'STOPPING'
        
        # Stop robot first
        self.get_logger().info('🛑 STOP - Stopping robot movement')
        self.call_pp(False)
        self.is_walking = False
        
        # Call DOWN service
        if not self.mani_down_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Manipulator DOWN service not available')
            self.state = 'READY'
            return
            
        self.get_logger().info('Calling DOWN service...')
        request = Trigger.Request()
        
        try:
            future = self.mani_down_client.call_async(request)
            future.add_done_callback(self.down_service_callback)
                
        except Exception as e:
            self.get_logger().error(f'DOWN service call failed: {str(e)}')
            self.state = 'READY'

    def up_service_callback(self, future):
        """Handle UP service response (from TryManiControl logic)"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'UP SUCCESS: {response.message}')
                # Now start robot movement
                self.get_logger().info('🚶 WALK - Starting robot movement!')
                self.call_pp(True)
                self.is_walking = True
                self.state = 'WALKING'
            else:
                self.get_logger().error(f'UP FAILED: {response.message}')
                self.state = 'READY'
        except Exception as e:
            self.get_logger().error(f'UP service callback failed: {str(e)}')
            self.state = 'READY'

    def down_service_callback(self, future):
        """Handle DOWN service response (from TryManiControl logic)"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'DOWN SUCCESS: {response.message}')
                self.get_logger().info('⬇️  DOWN - Manipulator lowered, ready for next cycle')
                self.state = 'READY'
            else:
                self.get_logger().error(f'DOWN FAILED: {response.message}')
                self.state = 'READY'
        except Exception as e:
            self.get_logger().error(f'DOWN service callback failed: {str(e)}')
            self.state = 'READY'

    def call_pp(self, enable: bool):
        """Enable/disable pure pursuit controller"""
        if self.pp_client.wait_for_service(timeout_sec=0.2):
            self.pp_client.call_async(SetBool.Request(data=enable))

    def log_status(self):
        self.log_counter += 1
        if self.log_counter >= 40:  # Log every 2 seconds (40 * 0.05s)
            self.log_counter = 0
            
            if self.state == 'WAITING_TO_START':
                self.get_logger().info(f'State: {self.state} - Press button[0] to start')
            elif self.state == 'INITIALIZING':
                ready = self.odom_received and self.check_tf_ready() and self.generate_path_client.wait_for_service(timeout_sec=0.1)
                self.get_logger().info(f'State: {self.state} - Ready: {ready}, Count: {self.consecutive_ready_count}/{self.required_ready_count}')
            elif self.state == 'READY':
                self.get_logger().info(f'State: {self.state} - Press button[0] to start walking')
            elif self.state == 'STARTING':
                self.get_logger().info(f'State: {self.state} - Moving manipulator UP...')
            elif self.state == 'WALKING':
                flex_status = f"FL={self.flex_sensors[0]}, BL={self.flex_sensors[1]}"
                self.get_logger().info(f'State: {self.state} - Press button[2] to stop | Flex: {flex_status}')
            elif self.state == 'STOPPING':
                self.get_logger().info(f'State: {self.state} - Moving manipulator DOWN...')


def main(args=None):
    rclpy.init(args=args)
    node = PathScheduler()
    try:
        rclpy.spin(node)    
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Integrated Path Scheduler...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()