#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import SetBool
import tf2_ros
import numpy as np
import math
from tf_transformations import euler_from_quaternion


class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')
        
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('max_linear_velocity', 2.0)
        self.declare_parameter('max_angular_velocity', 1.5)
        self.declare_parameter('control_frequency', 100.0)
        self.declare_parameter('goal_tolerance', 0.03)
        
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
        self.path_completed = False
        self.status = "IDLE"
        
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/controller_status', 10)
        
        self.path_sub = self.create_subscription(
            Path,
            '/centerline_path',
            self.path_callback,
            10
        )
        
        self.robot_start_srv = self.create_service(
            SetBool, 
            'robot_start', 
            self.toggle_cb
        )
        
        self.control_timer = self.create_timer(1.0 / control_freq, self.control_loop)
        self.status_timer = self.create_timer(0.1, self.publish_status)
        
        self.get_logger().info('Pure Pursuit Controller initialized')
        
    def path_callback(self, msg):
        self.current_path = msg
        self.path_received = True
        self.path_completed = False
        if self.status == "IDLE":
            self.status = "IDLE"
        
        self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')
        if len(msg.poses) > 0:
            final = msg.poses[-1]
            self.get_logger().info(f'Final waypoint: ({final.pose.position.x:.2f},{final.pose.position.y:.2f})')
        
    def toggle_cb(self, req: SetBool.Request, resp: SetBool.Response):
        self.active = req.data
        if not self.active:
            self.cmd_vel_pub.publish(Twist())
            if self.path_received and not self.path_completed:
                self.status = "STOPPED"
            else:
                self.status = "IDLE"
        else:
            if self.path_received and not self.path_completed:
                self.status = "FOLLOWING"
        resp.success = True
        resp.message = 'enabled' if self.active else 'disabled'
        self.get_logger().info(f'Pure Pursuit Path Follower {resp.message}')
        return resp
        
    def publish_status(self):
        msg = String()
        msg.data = self.status
        self.status_pub.publish(msg)
        
    def get_robot_pose(self):
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
    
    def get_closest_point_on_path(self, robot_x, robot_y):
        """Find the closest point on the path to the robot."""
        if not self.current_path or len(self.current_path.poses) == 0:
            return None, -1
        
        min_distance = float('inf')
        closest_index = 0
        
        for i, pose in enumerate(self.current_path.poses):
            px = pose.pose.position.x
            py = pose.pose.position.y
            distance = math.sqrt((px - robot_x)**2 + (py - robot_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index, min_distance
    
    def interpolate_path_point(self, start_pose, end_pose, t):
        """Interpolate between two path points."""
        start_x = start_pose.pose.position.x
        start_y = start_pose.pose.position.y
        end_x = end_pose.pose.position.x
        end_y = end_pose.pose.position.y
        
        x = start_x + t * (end_x - start_x)
        y = start_y + t * (end_y - start_y)
        
        return x, y
    
    def find_lookahead_point(self, robot_x, robot_y):
        """TRUE Pure Pursuit: Find point at exactly lookahead_distance ahead on path."""
        if not self.current_path or len(self.current_path.poses) < 2:
            return None, None
        
        # Find closest point on path
        closest_index, _ = self.get_closest_point_on_path(robot_x, robot_y)
        
        # Search forward from closest point for lookahead intersection
        for i in range(closest_index, len(self.current_path.poses) - 1):
            p1 = self.current_path.poses[i]
            p2 = self.current_path.poses[i + 1]
            
            # Check if circle intersects this segment
            intersections = self.line_circle_intersection(
                (p1.pose.position.x, p1.pose.position.y),
                (p2.pose.position.x, p2.pose.position.y),
                (robot_x, robot_y),
                self.lookahead_distance
            )
            
            # Take the furthest intersection along the path
            for ix, iy, t in intersections:
                if 0 <= t <= 1:  # Valid intersection on this segment
                    return ix, iy
        
        # If no intersection found, target the end of the path
        end_pose = self.current_path.poses[-1]
        return end_pose.pose.position.x, end_pose.pose.position.y
    
    def line_circle_intersection(self, p1, p2, robot_pos, radius):
        """Find intersection points between line segment and circle."""
        x1, y1 = p1
        x2, y2 = p2
        rx, ry = robot_pos
        
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - rx
        fy = y1 - ry
        
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0 or a == 0:
            return []
        
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        intersections = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                ix = x1 + t * dx
                iy = y1 + t * dy
                intersections.append((ix, iy, t))
        
        # Sort by t to get the furthest intersection first
        intersections.sort(key=lambda x: x[2], reverse=True)
        return intersections
    
    def is_path_completed(self, robot_x, robot_y):
        """Check if robot has reached the final goal."""
        if not self.current_path or len(self.current_path.poses) == 0:
            return False
        
        final_pose = self.current_path.poses[-1]
        final_x = final_pose.pose.position.x
        final_y = final_pose.pose.position.y
        distance_to_goal = math.sqrt((final_x - robot_x)**2 + (final_y - robot_y)**2)
        
        return distance_to_goal <= self.goal_tolerance
    
    def calculate_control_commands(self, robot_x, robot_y, robot_yaw, target_x, target_y):
        """Calculate pure pursuit steering commands WITHOUT speed reduction."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        # Calculate the angle to target point
        alpha = math.atan2(dy, dx) - robot_yaw
        
        # Normalize angle
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        # Pure pursuit curvature calculation
        curvature = 2 * math.sin(alpha) / self.lookahead_distance
        
        # CONSTANT SPEED - no reduction based on distance or steering
        linear_velocity = self.max_linear_vel
        
        # Only reduce speed slightly for very sharp turns (optional)
        if abs(alpha) > math.pi / 2:  # Very sharp turn
            linear_velocity *= 0.7
        
        # Angular velocity from curvature
        angular_velocity = curvature * linear_velocity
        
        # Clamp velocities
        linear_velocity = np.clip(linear_velocity, 0.0, self.max_linear_vel)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_vel, self.max_angular_vel)
        
        return linear_velocity, angular_velocity
    
    def publish_cmd_vel(self, linear_velocity, angular_velocity):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(linear_velocity)
        cmd_vel.angular.z = float(angular_velocity)
        self.cmd_vel_pub.publish(cmd_vel)
    
    def stop_robot(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
    
    def control_loop(self):
        if not self.active:
            self.stop_robot()
            return
            
        if not self.path_received or not self.current_path:
            self.stop_robot()
            self.status = "IDLE"
            return
            
        if self.path_completed:
            self.stop_robot()
            return
            
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        if robot_x is None:
            self.stop_robot()   
            return
        
        # Check path completion
        if self.is_path_completed(robot_x, robot_y):
            if not self.path_completed:
                final_pose = self.current_path.poses[-1]
                final_distance = math.sqrt(
                    (final_pose.pose.position.x - robot_x)**2 + 
                    (final_pose.pose.position.y - robot_y)**2
                )
                self.get_logger().info(f'Path completed! Final distance: {final_distance:.3f}m')
                self.path_completed = True
                self.status = "GOAL_REACHED"
            self.stop_robot()
            return
        
        self.status = "FOLLOWING"
        
        # Get lookahead point using pure pursuit
        target_x, target_y = self.find_lookahead_point(robot_x, robot_y)
        
        if target_x is None:
            self.get_logger().warn('No target point found')
            self.stop_robot()
            return
            
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_control_commands(
            robot_x, robot_y, robot_yaw, target_x, target_y
        )
        
        self.publish_cmd_vel(linear_vel, angular_vel)
        
        # Logging (less frequent)
        if hasattr(self, 'log_counter'):
            self.log_counter += 1
        else:
            self.log_counter = 0
            
        if self.log_counter % 50 == 0:  # Every 0.5 seconds
            distance_to_target = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            final_pose = self.current_path.poses[-1]
            distance_to_goal = math.sqrt(
                (final_pose.pose.position.x - robot_x)**2 + 
                (final_pose.pose.position.y - robot_y)**2
            )
            self.get_logger().info(
                f'Robot: ({robot_x:.2f}, {robot_y:.2f}), '
                f'Target: ({target_x:.2f}, {target_y:.2f}), '
                f'Dist to goal: {distance_to_goal:.2f}m, '
                f'Cmd: ({linear_vel:.2f}, {angular_vel:.2f})'
            )


def main(args=None):
    rclpy.init(args=args)
    
    node = PurePursuitController()
    
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