#!/usr/bin/env python3
"""
Pure Pursuit Path Follower Node
Follows path using Pure Pursuit control for differential drive robot
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
import tf2_ros
import numpy as np
import math
from tf_transformations import euler_from_quaternion


class DiffPurePursuitPathFollower(Node):
    def __init__(self):
        super().__init__('diff_pure_pursuit_path_follower')
        
        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        
        self.lookahead_dist = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.max_linear_vel = self.get_parameter('max_linear_vel').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_vel').get_parameter_value().double_value
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.current_path = None
        self.target_idx = 0
        self.path_received_time = None
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_service(SetBool, 'pure_pursuit_toggle', self.toggle_cb)

        self.active = True
        
        self.path_sub = self.create_subscription(
            Path,
            '/centerline_path',
            self.path_callback,
            10
        )
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Pure Pursuit Path Follower initialized')

    def toggle_cb(self, req: SetBool.Request, resp: SetBool.Response):
        self.active = req.data
        if not self.active:
            self.cmd_vel_pub.publish(Twist())
        resp.success = True
        resp.message = 'enabled' if self.active else 'disabled'
        print(f'Pure Pursuit Path Follower {resp.message}')
        return resp
    
    def path_callback(self, msg):
        """Store the received path"""
        current_time = self.get_clock().now()
        
        if self.path_received_time is None:
            self.current_path = msg
            self.target_idx = 0
            self.path_received_time = current_time
            self.get_logger().info(f'Received new path with {len(msg.poses)} waypoints')
        else:
            time_diff = (current_time - self.path_received_time).nanoseconds / 1e9
            if time_diff > 5.0:
                self.current_path = msg
                self.target_idx = 0
                self.path_received_time = current_time
                self.get_logger().info(f'Received new path with {len(msg.poses)} waypoints (timeout reset)')
            else:
                self.current_path = msg
        
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
            
        for i in range(self.target_idx, len(self.current_path.poses)):
            pose = self.current_path.poses[i]
            target_x = pose.pose.position.x
            target_y = pose.pose.position.y
            
            dist = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            
            if dist >= self.lookahead_dist:
                self.target_idx = i
                return target_x, target_y, i
                
        if len(self.current_path.poses) > 0:
            last_pose = self.current_path.poses[-1]
            self.target_idx = len(self.current_path.poses) - 1
            return last_pose.pose.position.x, last_pose.pose.position.y, len(self.current_path.poses) - 1
            
        return None, None, None
        
    def calculate_control(self, robot_x, robot_y, robot_yaw, target_x, target_y):
        """Calculate control commands using Pure Pursuit"""
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        alpha = math.atan2(dy, dx) - robot_yaw
        
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        curvature = 2 * math.sin(alpha) / self.lookahead_dist
        
        angular_vel = self.max_angular_vel * curvature
        
        linear_vel = self.max_linear_vel
        
        angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, angular_vel))
        
        return linear_vel, angular_vel
        
    def control_loop(self):
        """Main control loop"""
        if not self.active:
            return
        
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        
        if robot_x is None or not self.current_path:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
            
        target_x, target_y, target_idx = self.find_lookahead_point(robot_x, robot_y)
        
        if target_x is None:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
            
        if len(self.current_path.poses) > 0:
            last_pose = self.current_path.poses[-1]
            dist_to_end = math.sqrt(
                (last_pose.pose.position.x - robot_x)**2 + 
                (last_pose.pose.position.y - robot_y)**2
            )
            
            if dist_to_end < 0.2:
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
                self.get_logger().info('Reached end of path, stopping.')
                return
        
        linear_vel, angular_vel = self.calculate_control(
            robot_x, robot_y, robot_yaw, target_x, target_y
        )
        
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        if not hasattr(self, 'log_counter'):
            self.log_counter = 0
        
        self.log_counter += 1
        # if self.log_counter >= 20:
        #     self.log_counter = 0
        #     if len(self.current_path.poses) > 0:
        #         self.get_logger().info(
        #             f'Following path: target_idx={target_idx}/{len(self.current_path.poses)}, '
        #             f'distance_to_target={math.sqrt((target_x-robot_x)**2 + (target_y-robot_y)**2):.2f}m, '
        #             f'cmd_vel=({linear_vel:.2f}, {angular_vel:.2f})'
        #         )


def main(args=None):
    rclpy.init(args=args)
    
    node = DiffPurePursuitPathFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Pure Pursuit path follower...')
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()