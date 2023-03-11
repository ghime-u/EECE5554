#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import utm
import serial
import sys
import numpy as np
from datetime import datetime
from imu_driver.msg import *
from imu_driver.msg import Vectornav
from imu_driver.srv import *



def conversion_client(roll,pitch,yaw):
    
    try:
       
        convert = rospy.ServiceProxy('convert_to_quaternion', convert_to_quaternion)
        result = convert(roll,pitch,yaw)
        return result
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def driver():
    pub = rospy.Publisher('imu', Vectornav, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(40)
    msg = Vectornav()

    args = rospy.myargv(argv = sys.argv)
    if len(args) != 2:
        print("error")
        sys.exit(1)

    connected_port = args[1]
    serial_port = rospy.get_param('~port',connected_port)
    serial_baud = rospy.get_param('~baudrate',115200)
    rospy.wait_for_service("convert_to_quaternion")

    ser = serial.Serial(serial_port, serial_baud, timeout = 3)
    ser.write(b"$VNWRG,07,40*xx")
    while not rospy.is_shutdown():
        recieve = str(ser.readline().decode())
        

        if "$VNYMR" in str(recieve):
            data = str(recieve).split(",") 

            current_time = rospy.get_rostime()
            # rospy.loginfo("Current time %i %i", current_time.secs, current_time.nsecs)

            yaw = float(data[1])
            pitch = float(data[2])
            roll = float(data[3])
            magX = float(data[4])
            magY = float(data[5])
            magZ = float(data[6])
            accX = float(data[7])
            accY = float(data[8])
            accZ = float(data[9])
            gyroX = float(data[10])
            gyroY = float(data[11])
            gyroZ = float(data[12][0:9])

            res = conversion_client(roll, pitch,yaw)
            result = res.result

            #msg.header.stamp = rospy.Time.from_sec(current_time)
            msg.header.stamp.secs = int(current_time.secs)
            msg.header.stamp.nsecs = int(current_time.nsecs)
            msg.header.frame_id = 'imu1_Frame'
            msg.IMU.orientation.x = result[0]
            msg.IMU.orientation.y = result[1]
            msg.IMU.orientation.z = result[2]
            msg.IMU.orientation.w = result[3]
            msg.IMU.linear_acceleration.x = accX
            msg.IMU.linear_acceleration.y = accY
            msg.IMU.linear_acceleration.z = accZ
            msg.IMU.angular_velocity.x = gyroX
            msg.IMU.angular_velocity.y = gyroY
            msg.IMU.angular_velocity.z = gyroZ
            msg.mag_field.magnetic_field.x = magX
            msg.mag_field.magnetic_field.y = magY
            msg.mag_field.magnetic_field.z = magZ
            msg.VNYMR = recieve
            rospy.loginfo(msg)
            pub.publish(msg)
            #rate.sleep()


if __name__ == '__main__':
    try:
        driver()
    except rospy.ROSInterruptException:
        pass
