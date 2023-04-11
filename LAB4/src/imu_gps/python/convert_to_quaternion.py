#!/usr/bin/env python3

import rospy
import math
from imu_gps.srv import convert_to_quaternion, convert_to_quaternionRequest, convert_to_quaternionResponse

def quaternion_conversion(euler_angles):
    roll, pitch, yaw = euler_angles
    yaw_cos = math.cos(yaw * 0.5)
    yaw_sin = math.sin(yaw * 0.5)
    pitch_cos = math.cos(pitch * 0.5)
    pitch_sin = math.sin(pitch * 0.5)
    roll_cos = math.cos(roll * 0.5)
    roll_sin = math.sin(roll * 0.5)
    q_w = (yaw_cos * pitch_cos * roll_cos) + (yaw_sin * pitch_sin * roll_sin)
    q_x = (yaw_cos * pitch_cos * roll_sin) - (yaw_sin * pitch_sin * roll_cos)
    q_y = (yaw_cos * pitch_sin * roll_cos) + (yaw_sin * pitch_cos * roll_sin)
    q_z = (yaw_sin * pitch_cos * roll_cos) - (yaw_cos * pitch_sin * roll_sin)
    return [q_x, q_y, q_z, q_w]

def handle_convert_to_quaternion(req):
    euler_angles = (req.roll, req.pitch, req.yaw)
    q_converted = quaternion_conversion(euler_angles)
    return convert_to_quaternionResponse(q_converted[0], q_converted[1], q_converted[2], q_converted[3])

def my_func():
    rospy.init_node('convert_to_quaternion')
    s = rospy.Service('/convert_to_quaternion', convert_to_quaternion, handle_convert_to_quaternion)
    rospy.spin()

if __name__ == "__main__":
    my_func()
