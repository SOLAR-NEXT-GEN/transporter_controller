#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import tf_transformations


class RealsenseMoveForwardNode(Node):
    def __init__(self):
        super().__init__("realsense_move_forward")

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.pose_sub = self.create_subscription(
            Odometry,
            "/camera/pose/sample",
            self.pose_callback,
            qos_profile=rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,  # Matches RealSense
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )

        # PID controller parameters
        self.kp_linear = 3.0  # Proportional gain for linear velocity
        self.ki_linear = 0.0  # Integral gain for linear velocity
        self.kd_linear = 0.0  # Derivative gain for linear velocity

        self.kp_angular = 0.0  # Proportional gain for angular velocity
        self.ki_angular = 0.0  # Integral gain for angular velocity
        self.kd_angular = 0.0  # Derivative gain for angular velocity

        # PID controller variables
        self.linear_error_sum = 0.0
        self.angular_error_sum = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0

        # Movement variables
        self.target_distance = 0.50  # Target distance in meters
        self.linear_tolerance = 0.01  # Tolerance in meters
        self.angular_tolerance = 0.01  # Tolerance in radians

        # Initial pose
        self.initial_position = None
        self.current_position = None
        self.current_orientation = None
        self.distance_traveled = 0.0
        self.movement_started = False
        self.movement_completed = False

        # Timer for constant control rate
        self.timer = self.create_timer(0.01, self.control_loop)  # 10 Hz control loop

        self.get_logger().info("Realsense Move Forward Node initialized")

    def pose_callback(self, msg):
        # self.get_logger().info("Pose data received!")
        # Extract position from the odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Store current position and orientation
        self.current_position = np.array([position.x, position.y, position.z])
        self.current_orientation = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        # Store initial position when first message is received
        if not self.movement_started and not self.movement_completed:
            self.initial_position = np.array([position.x, position.y, position.z])
            self.movement_started = True
            self.get_logger().info(
                f"Starting position: [{position.x:.2f}, {position.y:.2f}, {position.z:.2f}]"
            )
            self.get_logger().info(f"Moving forward {self.target_distance} meters...")

    def control_loop(self):
        if (
            not self.movement_started
            or self.movement_completed
            or self.current_position is None
        ):
            return

        # Calculate distance traveled so far
        displacement = self.current_position - self.initial_position
        self.distance_traveled = np.linalg.norm(displacement)

        # Calculate remaining distance
        distance_error = self.target_distance - self.distance_traveled

        # Calculate heading error - we want to go straight (yaw = 0)
        _, _, yaw = tf_transformations.euler_from_quaternion(self.current_orientation)
        angular_error = 0.0 - yaw  # Target yaw - current yaw

        # Check if we've reached the target
        if abs(distance_error) <= self.linear_tolerance:
            self.movement_completed = True
            self.stop_robot()
            self.get_logger().info(
                f"Target reached! Distance traveled: {self.distance_traveled:.2f} meters"
            )
            return

        # Calculate PID for linear velocity
        p_linear = self.kp_linear * distance_error
        self.linear_error_sum += distance_error
        i_linear = self.ki_linear * self.linear_error_sum
        d_linear = self.kd_linear * (distance_error - self.last_linear_error)
        linear_velocity = p_linear + i_linear + d_linear

        # Limit maximum linear velocity to 0.6
        linear_velocity = min(max(linear_velocity, -0.6), 0.6)

        # Calculate PID for angular velocity
        p_angular = self.kp_angular * angular_error
        self.angular_error_sum += angular_error
        i_angular = self.ki_angular * self.angular_error_sum
        d_angular = self.kd_angular * (angular_error - self.last_angular_error)
        angular_velocity = p_angular + i_angular + d_angular

        # Limit maximum angular velocity to 0.3
        angular_velocity = min(max(angular_velocity, -0.3), 0.3)

        # Save errors for derivative term
        self.last_linear_error = distance_error
        self.last_angular_error = angular_error

        # Create and publish velocity command
        if abs(linear_velocity) < 0.05:
            linear_velocity = 0.0

        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.cmd_vel_pub.publish(twist)

        # Log progress
        # if self.distance_traveled % 0.5 < 0.1:  # Log approximately every 0.5 meters
        #     self.get_logger().info(
        #         f"Distance traveled: {self.distance_traveled:.2f}m, Remaining: {distance_error:.2f}m"
        #     )

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Robot stopped")


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseMoveForwardNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by keyboard interrupt")
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()