#!/usr/bin/env python
import numpy as np
import rospy


from imu_driver.srv import convert_to_quaternion, convert_to_quaternionResponse

def euler_to_quaternion(roll, pitch, yaw):
    # Convert Euler angles to quaternion
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return qx,qy,qz,qw

def handle_euler_to_quaternion(req):
    # Extract Euler angles from request
    roll = req.roll
    pitch = req.pitch
    yaw = req.yaw

    # Convert Euler angles to quaternion
    qx,qy,qz,qw = euler_to_quaternion(roll, pitch, yaw)
    temp = [qx,qy,qz,qw]
    # Construct response message
    res = convert_to_quaternionResponse()
    res.result = temp
    return res

def euler_to_quaternion_server():
    rospy.init_node('convert_to_quaternion_server')
    s = rospy.Service('convert_to_quaternion', convert_to_quaternion, handle_euler_to_quaternion)
    rospy.spin()

if __name__ == "__main__":
    euler_to_quaternion_server()