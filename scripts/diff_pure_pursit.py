#!/usr/bin/env python3
"""
Pure Pursuit Path Follower Node
Follows path using Pure Pursuit control for differential drive robot
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
import tf2_ros
import numpy as np
import math
from tf_transformations import euler_from_quaternion


class DiffPurePursuitPathFollower(Node):
    def __init__(self):
        super().__init__('diff_pure_pursuit_path_follower')
        
        # Declare parameters
        self.declare_parameter('lookahead_distance', 0.5)  # meters
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        
        # Get parameters
        self.lookahead_dist = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.max_linear_vel = self.get_parameter('max_linear_vel').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_vel').get_parameter_value().double_value
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Current path and target
        self.current_path = None
        self.target_idx = 0
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/centerline_path',
            self.path_callback,
            10
        )
        
        # Control timer (20Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Pure Pursuit Path Follower initialized')
        
    def path_callback(self, msg):
        """Store the received path"""
        self.current_path = msg
        self.target_idx = 0
        
    def get_robot_pose(self):
        """Get robot pose in map frame"""
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
            
            # Get yaw from quaternion
            q = transform.transform.rotation
            euler = euler_from_quaternion([q.x, q.y, q.z, q.w])
            yaw = euler[2]
            
            return x, y, yaw
        except Exception as e:
            return None, None, None
            
    def find_lookahead_point(self, robot_x, robot_y):
        """Find target point on path using lookahead distance"""
        if not self.current_path or len(self.current_path.poses) == 0:
            return None, None, None
            
        # Start from current target index
        for i in range(self.target_idx, len(self.current_path.poses)):
            pose = self.current_path.poses[i]
            target_x = pose.pose.position.x
            target_y = pose.pose.position.y
            
            # Calculate distance to this point
            dist = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            
            # If point is at lookahead distance, use it
            if dist >= self.lookahead_dist:
                self.target_idx = i
                return target_x, target_y, i
                
        # If no point found at lookahead distance, use last point
        if len(self.current_path.poses) > 0:
            last_pose = self.current_path.poses[-1]
            return last_pose.pose.position.x, last_pose.pose.position.y, len(self.current_path.poses) - 1
            
        return None, None, None
        
    def calculate_control(self, robot_x, robot_y, robot_yaw, target_x, target_y):
        """Calculate control commands using Pure Pursuit"""
        # Calculate the lookahead point's position relative to the robot
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        # Calculate the lookahead point's angle
        alpha = math.atan2(dy, dx) - robot_yaw
        
        # Normalize the angle
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        # Calculate the curvature (inverse of turning radius)
        # Formula: curvature = 2 * sin(alpha) / lookahead_distance
        curvature = 2 * math.sin(alpha) / self.lookahead_dist
        
        # Calculate angular velocity using the curvature
        angular_vel = self.max_angular_vel * curvature
        
        # Linear velocity is constant, but can be capped at max speed
        linear_vel = self.max_linear_vel
        
        # Cap angular velocity within the allowed limits
        angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, angular_vel))
        
        return linear_vel, angular_vel
        
    def control_loop(self):
        """Main control loop"""
        # Get robot pose
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        
        if robot_x is None or not self.current_path:
            # Stop robot if no pose or path
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
            
        # Find lookahead point
        target_x, target_y, target_idx = self.find_lookahead_point(robot_x, robot_y)
        
        if target_x is None:
            # No target, stop
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
            
        # Calculate control
        linear_vel, angular_vel = self.calculate_control(
            robot_x, robot_y, robot_yaw, target_x, target_y
        )
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        # Log status every second
        if not hasattr(self, 'log_counter'):
            self.log_counter = 0
        
        self.log_counter += 1
        if self.log_counter >= 20:  # 20 * 0.05 = 1 second
            self.log_counter = 0
            self.get_logger().info(
                f'Following path: target_idx={target_idx}/{len(self.current_path.poses)}, '
                f'distance_to_target={math.sqrt((target_x-robot_x)**2 + (target_y-robot_y)**2):.2f}m, '
                f'cmd_vel=({linear_vel:.2f}, {angular_vel:.2f})'
            )


def main(args=None):
    rclpy.init(args=args)
    
    node = DiffPurePursuitPathFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Pure Pursuit path follower...')
        # Stop robot
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()