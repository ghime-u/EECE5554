#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import serial
from std_msgs.msg import Float64
from imu_gps.msg import gps_msg
# Package - gps_driver; msg - gps_msg
import utm
import re
import argparse
import sys

pattern = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"

def convert_utc_to_string(utc):
    utc_string = utc[:2]+ " Hours " + utc[2:4] + " Minutes " + utc[4:] + " Seconds"
    return utc_string

def convert_lat_to_degree(latitude):
    degree = int(latitude[:2]) + (float(latitude[2:])/60)
    return degree

def convert_long_to_degree(longitude):
    degree = int(longitude[:3]) + (float(longitude[3:])/60)
    return degree

def gps_isCorrect(values):
    if re.match(pattern, values[1]) and re.match(pattern, values[2]) and re.match(pattern, values[4]):
        if values[3]=="N" or values[3]=="S":
            if values[5]=="E" or values[5]=="W":
                return True
    return False

if __name__ == '__main__':
    args = rospy.myargv(argv = sys.argv)
    if len(args) != 2:
        print(error)
        sys.exit(1)

    serial_port_arg = args[1]

    rospy.init_node('driver')
    #port = rospy.get_param('~port','/dev/pts/1') # hardcode serial port
    port = rospy.get_param('~port',serial_port_arg)
    baud = rospy.get_param('~baudrate',4800)
    #rate = rospy.Rate(5) # 5hz

    port = serial.Serial(port, baud, timeout=3.)# Create conection
    rospy.logdebug("Using GPS sensor on port "+serial_port_arg+" at "+str(baud))

    rospy.sleep(0.2) # 5Hz sampling rate

    rospy.loginfo("Publishing GPS and UTM")

    pub = rospy.Publisher('/gps', gps_msg, queue_size=5)
    # topic - /gps

    try:
        while not rospy.is_shutdown():
            line = port.readline()
            #print(line)
            # Sample read line - b'$GPGSV,3,2,12,08,05,173,27,07,70,338,16,31,00,000,26,51,37,207,*7A\n'
            line_str = str(line)
            #rospy.loginfo(line)
            #print(line_str)
            line_str = line_str.lstrip("b'\\r$")
            line_str = line_str.lstrip("b'$")
            line_str = line_str.rstrip("\\n'")
            #print(line_str)
            if line_str == '':
                rospy.logwarn("GPS: No data")
            else:
                values = line_str.split(',')
                if(values[0] == "GPGGA"):
                    if gps_isCorrect(values):
                        latitude = convert_lat_to_degree(values[2])
                        latitude_dir = values[3]
                        longitude = convert_long_to_degree(values[4])
                        longitude_dir = values[5]
                        if latitude_dir == 'S':
                            latitude = -latitude
                        if longitude_dir == 'W':
                            longitude = -longitude
                        altitude = (float(values[9]))
                        hdop = (float(values[8]))
                        utc = convert_utc_to_string(values[1])
                        utc_float = float(values[1])
                        utc_hrs = utc_float
                        utc_mnts = utc_float - (utc_hrs * 10000)
                        utc_secs = (utc_float - (utc_hrs * 10000)) - (utc_mnts * 100)
                        utc_secs_t = utc_hrs * 3600 + utc_mnts * 60 + utc_secs
                        utc_nsecs_t = int (utc_secs_t * (10 ** 7))
                        utm_coordinates = utm.from_latlon(latitude, longitude)
                        
                        utm_easting = (utm_coordinates[0])
                        utm_northing = (utm_coordinates[1])
                        zone_num = utm_coordinates[2]
                        zone_letter = utm_coordinates[3]
                        
                        msg = gps_msg()
                        msg.Header.stamp.secs = int (str(utc_secs_t)[:5])
                        msg.Header.stamp.nsecs = int (str(utc_nsecs_t)[:5])
                        msg.Header.frame_id = "GPS1_Frame"
                        msg.Latitude =  latitude
                        msg.Longitude = longitude
                        msg.Altitude = altitude
                        msg.HDOP = hdop
                        msg.UTM_easting = utm_easting
                        msg.UTM_northing = utm_northing
                        msg.UTC = utc
                        msg.Zone = zone_num
                        msg.Letter = zone_letter
                        rospy.loginfo(msg)
                        pub.publish(msg)
            #rate.sleep()

    except Exception as e:
        rospy.loginfo("An exception of type ", type(e).__name__, " was raised.")
    
    except rospy.ROSInterruptException:
        port.close()

    except serial.serialutil.SerialException:
        rospy.loginfo("Shutting down GPS node...")

    #except:
    #    rospy.loginfo("Unknown error...")
