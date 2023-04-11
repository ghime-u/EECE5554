#!/usr/bin/env python3

import rospy
import serial
from imu_gps.msg import Vectornav
import sys
from imu_gps.srv import convert_to_quaternion, convert_to_quaternionRequest, convert_to_quaternionResponse
import re

pattern = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"

def imu_isCorrect(values):
    for i in range(1,12):
        if re.match(pattern, values[i]):
            continue
        else:
            return False
    if not re.match(pattern, values[12][0:9]):
        return False
    return True

if __name__ == '__main__':
    args = rospy.myargv(argv = sys.argv)
    if len(args) != 2:
        print(error)
        sys.exit(1)

    serial_port_arg = args[1]

    rospy.init_node('driver')
    port = rospy.get_param('~port',serial_port_arg)
    baud = rospy.get_param('~baudrate',115200)

    rospy.wait_for_service('/convert_to_quaternion')
    convert_to_quaternion = rospy.ServiceProxy('/convert_to_quaternion', convert_to_quaternion)
    
    try:
        port = serial.Serial(port, baud, timeout=3.)# Create conection
        port.write(b"$VNWRG,07,40*xx") # Register write to set output data frequency
    except Exception as e:
        rospy.loginfo("An exception of type ", type(e).__name__, " was raised.")
        sys.exit(1)
    
    rospy.logdebug("Using IMU sensor on port "+serial_port_arg+" at "+str(baud))

    rospy.loginfo("Publishing IMU")

    pub = rospy.Publisher('/imu', Vectornav, queue_size=5) # topic - /imu
    msg = Vectornav()
    
    try:
        while not rospy.is_shutdown():
            line = port.readline()
            #line = port.readline().decode()
            line_str = str(line)
            line_str = line_str.lstrip("b'\\r$")
            line_str = line_str.lstrip("b'$")
            line_str = line_str.rstrip("\\n'")
            if line_str == '':
                rospy.logwarn("IMU: No data")
            else:
                values = line_str.split(',')
                if(values[0] == "VNYMR"):
                    if imu_isCorrect(values):
                        yaw = float(values[1])
                        pitch = float(values[2])
                        roll = float(values[3])
                        magnetoX = float(values[4])
                        magnetoY = float(values[5])
                        magnetoZ = float(values[6])
                        accelX = float(values[7])
                        accelY = float(values[8])
                        accelZ = float(values[9])
                        gyroX = float(values[10])
                        gyroY = float(values[11])
                        gyroZ = float(values[12][0:9])

                        request = convert_to_quaternionRequest() #Request data type
                        request.roll = roll
                        request.pitch = pitch
                        request.yaw = yaw

                        try:
                            response = convert_to_quaternion(request)
                            orientation_x = response.x
                            orientation_y = response.y
                            orientation_z = response.z
                            orientation_w = response.w
                        except Exception as e:
                            rospy.loginfo("An exception of type ", type(e).__name__, " was raised.")
                            rospy.loginfo("Service failed")
                            continue
                        current_time = rospy.get_rostime()
                        time_secs = int(current_time.secs)
                        time_nsecs = int(current_time.nsecs)

                        msg.Header.seq = msg.Header.seq+1
                        msg.Header.stamp.secs = time_secs
                        msg.Header.stamp.nsecs = time_nsecs
                        msg.Header.frame_id = "imu1_Frame"

                        msg.imu.orientation.x = orientation_x
                        msg.imu.orientation.y = orientation_y
                        msg.imu.orientation.z = orientation_z
                        msg.imu.orientation.w = orientation_w

                        msg.imu.linear_acceleration.x = accelX
                        msg.imu.linear_acceleration.y = accelY
                        msg.imu.linear_acceleration.z = accelZ

                        msg.imu.angular_velocity.x = gyroX
                        msg.imu.angular_velocity.y = gyroY
                        msg.imu.angular_velocity.z = gyroZ

                        msg.mag_field.magnetic_field.x = magnetoX
                        msg.mag_field.magnetic_field.y = magnetoY
                        msg.mag_field.magnetic_field.z = magnetoZ

                        msg.yaw = yaw
                        msg.pitch = pitch
                        msg.roll = roll

                        rospy.loginfo(msg)
                        pub.publish(msg)
    except Exception as e:
        rospy.loginfo("An exception of type ", type(e).__name__, " was raised.")
    
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down IMU node... 1")
        port.close()

    except serial.serialutil.SerialException:
        rospy.loginfo("Shutting down IMU node...")
