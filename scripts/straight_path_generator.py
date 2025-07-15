#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger
import tf2_ros
import numpy as np
import math
from tf_transformations import euler_from_quaternion


class StraightPathGenerator(Node):
    def __init__(self):
        super().__init__('straight_path_generator')
        
        self.declare_parameter('path_length', 12.0)
        self.declare_parameter('waypoint_spacing', 0.5)
        self.declare_parameter('start_offset', 1.0)
        
        self.path_length = self.get_parameter('path_length').get_parameter_value().double_value
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').get_parameter_value().double_value
        self.start_offset = self.get_parameter('start_offset').get_parameter_value().double_value
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.path_pub = self.create_publisher(Path, '/centerline_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        
        self.generate_service = self.create_service(
            Trigger,
            'generate_straight_path',
            self.generate_path_callback
        )
        
        self.path_generated = False
        self.current_path = None
        
        self.get_logger().info('Straight Path Generator ready')
        
    def get_robot_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            q = transform.transform.rotation
            euler = euler_from_quaternion([q.x, q.y, q.z, q.w])
            yaw = euler[2]
            
            return x, y, yaw
        except Exception as e:
            self.get_logger().error(f'Failed to get robot pose: {e}')
            return None, None, None
            
    def generate_path_callback(self, request, response):
        if self.path_generated:
            response.success = False
            response.message = "Path already generated"
            return response
            
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        
        if robot_x is None:
            response.success = False
            response.message = "Failed to get robot pose"
            return response
            
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        
        num_points = int(self.path_length / self.waypoint_spacing) + 1
        
        start_x = robot_x + self.start_offset * math.cos(robot_yaw)
        start_y = robot_y + self.start_offset * math.sin(robot_yaw)
        
        for i in range(num_points):
            distance = i * self.waypoint_spacing
            
            x = start_x + distance * math.cos(robot_yaw)
            y = start_y + distance * math.sin(robot_yaw)
            
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            pose.pose.orientation.z = math.sin(robot_yaw / 2)
            pose.pose.orientation.w = math.cos(robot_yaw / 2)
            
            path.poses.append(pose)
            
        self.current_path = path
        self.path_generated = True
        
        self.publish_timer = self.create_timer(0.1, self.publish_path)
        
        self.visualize_path()
        
        response.success = True
        response.message = f"Generated path with {len(path.poses)} waypoints"
        self.get_logger().info(response.message)
        
        return response
        
    def publish_path(self):
        if not self.current_path or not self.path_generated:
            return
            
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        if robot_x is None:
            return
            
        # Find closest waypoint to robot
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pose in enumerate(self.current_path.poses):
            dist = math.sqrt(
                (pose.pose.position.x - robot_x)**2 + 
                (pose.pose.position.y - robot_y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Create path starting from closest point
        windowed_path = Path()
        windowed_path.header.frame_id = 'map'
        windowed_path.header.stamp = self.get_clock().now().to_msg()
        
        # Add waypoints from closest point to end
        for i in range(closest_idx, len(self.current_path.poses)):
            windowed_path.poses.append(self.current_path.poses[i])
            
        # Publish windowed path
        self.path_pub.publish(windowed_path)
            
    def visualize_path(self):
        if not self.current_path:
            return
            
        markers = MarkerArray()
        
        line_marker = Marker()
        line_marker.header = self.current_path.header
        line_marker.ns = "path_line"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.1
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        
        for pose in self.current_path.poses:
            line_marker.points.append(pose.pose.position)
            
        markers.markers.append(line_marker)
        
        for i, pose in enumerate(self.current_path.poses):
            if i % 5 == 0:
                sphere = Marker()
                sphere.header = self.current_path.header
                sphere.ns = "waypoints"
                sphere.id = i + 1
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose = pose.pose
                sphere.scale.x = 0.2
                sphere.scale.y = 0.2
                sphere.scale.z = 0.2
                sphere.color.r = 1.0
                sphere.color.g = 0.0
                sphere.color.b = 0.0
                sphere.color.a = 1.0
                markers.markers.append(sphere)
                
        start_marker = Marker()
        start_marker.header = self.current_path.header
        start_marker.ns = "start_end"
        start_marker.id = 1000
        start_marker.type = Marker.ARROW
        start_marker.action = Marker.ADD
        start_marker.pose = self.current_path.poses[0].pose
        start_marker.scale.x = 0.5
        start_marker.scale.y = 0.1
        start_marker.scale.z = 0.1
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        start_marker.color.a = 1.0
        markers.markers.append(start_marker)
        
        end_marker = Marker()
        end_marker.header = self.current_path.header
        end_marker.ns = "start_end"
        end_marker.id = 1001
        end_marker.type = Marker.CUBE
        end_marker.action = Marker.ADD
        end_marker.pose = self.current_path.poses[-1].pose
        end_marker.scale.x = 0.3
        end_marker.scale.y = 0.3
        end_marker.scale.z = 0.3
        end_marker.color.r = 1.0
        end_marker.color.g = 0.0
        end_marker.color.b = 0.0
        end_marker.color.a = 1.0
        markers.markers.append(end_marker)
        
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = StraightPathGenerator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()