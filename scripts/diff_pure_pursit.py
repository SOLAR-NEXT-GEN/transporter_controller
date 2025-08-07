#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
import tf2_ros
import numpy as np
import math
from tf_transformations import euler_from_quaternion


class DifferentialDrivePurePursuitController(Node):
    def __init__(self):
        super().__init__('differential_drive_pure_pursuit_controller')
        
        # Parameters for differential drive
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('max_linear_velocity', 2.0)
        self.declare_parameter('max_angular_velocity', 1.5)
        self.declare_parameter('control_frequency', 100.0)
        self.declare_parameter('goal_tolerance', 0.05)
        
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        control_freq = self.get_parameter('control_frequency').get_parameter_value().double_value
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.current_path = None
        self.path_received = False
        self.active = False
        self.target_index = 0
        self.path_completed = False
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers - Subscribe to path only once
        self.path_sub = None
        self.subscribe_to_path()
        
        # Services
        self.robot_start_srv = self.create_service(
            SetBool, 
            'pure_pursuit_toggle', 
            self.toggle_cb
        )
        
        # Control timer
        self.control_timer = self.create_timer(1.0 / control_freq, self.control_loop)
        
        self.get_logger().info('Differential Drive Pure Pursuit Controller initialized')
        
    def subscribe_to_path(self):
        """Subscribe to path topic"""
        if self.path_sub is None:
            self.path_sub = self.create_subscription(
                Path,
                '/centerline_path',
                self.path_callback,
                10
            )
        
    def path_callback(self, msg):
        """Receive path once and unsubscribe"""
        if not self.path_received:
            self.current_path = msg
            self.path_received = True
            self.target_index = 0
            self.path_completed = False
            self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')
            
            # Unsubscribe after receiving first path
            if self.path_sub is not None:
                self.destroy_subscription(self.path_sub)
                self.path_sub = None
                self.get_logger().info('Unsubscribed from path topic')
        
    def toggle_cb(self, req: SetBool.Request, resp: SetBool.Response):
        """Service to toggle robot active state"""
        self.active = req.data
        if not self.active:
            self.cmd_vel_pub.publish(Twist())
        resp.success = True
        resp.message = 'enabled' if self.active else 'disabled'
        self.get_logger().info(f'Pure Pursuit Path Follower {resp.message}')
        return resp
        
    def get_robot_pose(self):
        """Get current robot pose from tf"""
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                now,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            q = transform.transform.rotation
            euler = euler_from_quaternion([q.x, q.y, q.z, q.w])
            yaw = euler[2]
            
            return x, y, yaw
            
        except Exception as e:
            return None, None, None
    
    def find_lookahead_point(self, robot_x, robot_y):
        """Find target point using lookahead distance"""
        if not self.current_path or len(self.current_path.poses) == 0:
            return None, None
            
        path_points = []
        for pose in self.current_path.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            path_points.append([x, y])
        
        # Start searching from current target index
        for i in range(self.target_index, len(path_points)):
            x, y = path_points[i]
            distance = math.sqrt((x - robot_x)**2 + (y - robot_y)**2)
            
            if distance >= self.lookahead_distance:
                self.target_index = i
                return x, y
        
        # If no point found at lookahead distance, use the last point
        if len(path_points) > 0:
            self.target_index = len(path_points) - 1
            return path_points[-1][0], path_points[-1][1]
            
        return None, None
    
    def is_goal_reached(self, robot_x, robot_y):
        """Check if robot has reached the final goal using index-based method"""
        if not self.current_path or len(self.current_path.poses) == 0:
            return False
        
        # Method 1: Index-based - robot is targeting the last waypoint
        if self.target_index >= len(self.current_path.poses) - 1:
            # Only apply small tolerance when at the very last point to prevent oscillation
            last_point = self.current_path.poses[-1].pose.position
            distance_to_goal = math.sqrt(
                (robot_x - last_point.x)**2 + 
                (robot_y - last_point.y)**2
            )
            # Much smaller tolerance (5cm) only for the final approach
            return distance_to_goal < 0.05
        
        return False
    
    def calculate_control_commands(self, robot_x, robot_y, robot_yaw, target_x, target_y):
        """Calculate linear and angular velocities using Pure Pursuit for differential drive"""
        # Calculate angle to target
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        # Angle from robot heading to target
        alpha = math.atan2(dy, dx) - robot_yaw
        
        # Normalize angle to [-pi, pi]
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        # Pure Pursuit formula for differential drive
        # This uses the kinematically correct relationship: ω = κ * v
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        # Curvature calculation from Pure Pursuit
        curvature = 2 * math.sin(alpha) / self.lookahead_distance
        
        # Linear velocity - reduce speed when turning sharply
        linear_velocity = self.max_linear_vel * (1.0 - abs(alpha) / math.pi)
        linear_velocity = max(0.1, linear_velocity)  # Minimum speed
        
        # Angular velocity using kinematic relationship
        angular_velocity = curvature * linear_velocity
        
        # Clamp to maximum values
        linear_velocity = np.clip(linear_velocity, 0.0, self.max_linear_vel)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_vel, self.max_angular_vel)
        
        return linear_velocity, angular_velocity
    
    def publish_cmd_vel(self, linear_velocity, angular_velocity):
        """Publish velocity commands"""
        cmd_vel = Twist()
        cmd_vel.linear.x = float(linear_velocity)
        cmd_vel.angular.z = float(angular_velocity)
        self.cmd_vel_pub.publish(cmd_vel)
    
    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
    
    def control_loop(self):
        """Main control loop"""
        # Check if robot should be running
        if not self.active:
            self.stop_robot()
            return
            
        # Check if path is available
        if not self.path_received or not self.current_path:
            self.stop_robot()
            return
            
        # Check if path is already completed
        if self.path_completed:
            self.stop_robot()
            return
            
        # Get current robot pose
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        if robot_x is None:
            self.stop_robot()   
            return
        
        # Check if goal is reached
        if self.is_goal_reached(robot_x, robot_y):
            if not self.path_completed:
                self.get_logger().info('Goal reached! Stopping robot.')
                self.path_completed = True
            self.stop_robot()
            return
        
        # Find lookahead point
        target_x, target_y = self.find_lookahead_point(robot_x, robot_y)
        
        if target_x is None:
            self.get_logger().warn('No target point found')
            self.stop_robot()
            return
            
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_control_commands(
            robot_x, robot_y, robot_yaw, target_x, target_y
        )
        
        # Publish commands
        self.publish_cmd_vel(linear_vel, angular_vel)
        
        # Optional: Log progress
        if hasattr(self, 'log_counter'):
            self.log_counter += 1
        else:
            self.log_counter = 0
            
        if self.log_counter % 40 == 0:  # Log every 2 seconds at 20Hz
            distance_to_target = math.sqrt(
                (target_x - robot_x)**2 + (target_y - robot_y)**2
            )
            self.get_logger().info(
                f'Target: ({target_x:.2f}, {target_y:.2f}), '
                f'Distance: {distance_to_target:.2f}m, '
                f'Cmd: ({linear_vel:.2f}, {angular_vel:.2f})'
            )


def main(args=None):
    rclpy.init(args=args)
    
    node = DifferentialDrivePurePursuitController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Pure Pursuit controller...')
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()