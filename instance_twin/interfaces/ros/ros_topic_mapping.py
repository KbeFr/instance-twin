"""
Contract topic <-> ROS type, the translation table every ROS robot starts from.
An agent's sensors pick the topics; a topic missing here has no ROS form.
A robot interface only overrides what differs, e.g. TwistStamped on Jazzy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core_msgs.topic_contract import MessageType


@dataclass(frozen=True)
class RosChannel:
    """ROS type carried on a contract topic, plus converter params."""
    ros_type: str
    params: dict = field(default_factory=dict)


ROS_TOPIC_MAPPING: dict[MessageType, RosChannel] = {
    # robot -> twin
    MessageType.ODOM: RosChannel("nav_msgs/msg/Odometry"),
    MessageType.IMU: RosChannel("sensor_msgs/msg/Imu"),
    MessageType.BATTERY: RosChannel("sensor_msgs/msg/BatteryState"),
    MessageType.LIDAR: RosChannel("sensor_msgs/msg/LaserScan"),
    MessageType.ARUCO_DETECTIONS: RosChannel("ros2_aruco_interfaces/msg/ArucoMarkers"),

    # twin -> robot
    MessageType.ACTION: RosChannel("geometry_msgs/msg/Twist"),
}

# Definitions the stock typestores lack, registered on every codec
EXTRA_MSGS: dict[str, str] = {
    "ros2_aruco_interfaces/msg/ArucoMarkers": (
        "std_msgs/Header header\n"
        "int64[] marker_ids\n"
        "geometry_msgs/Pose[] poses\n"
    ),
}
