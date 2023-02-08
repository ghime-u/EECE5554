import rospy
import utm
import serial
import sys
from gps_driver.msg import gps_msg

def main():
    publisher = rospy.Publisher('gps', gps_msg, queue_size=5)
    rospy.init_node('gps_talk', anonymous=True)
   
    args = rospy.myargv(argv = sys.argv)
    print(args)
    port = args[1]
    serial_port = rospy.get_param('~port', port)
    serial_baud = rospy.get_param('~baudrate',4800)
    data1 = serial.Serial(serial_port, serial_baud, timeout = 3)
  

    while not rospy.is_shutdown():
        input = str(data1.readline())

        if "$GPGGA" in str(input):
            data2 = str(input).split(',')
            print(data2)

            """ Time Data"""
            utc = float(data2[1])
            hours_utc = utc//10000
            minutes_utc = (utc-(hours_utc*10000))//100
            seconds_utc = (utc - (hours_utc*10000) - (minutes_utc*100))
            utc_sec = (hours_utc*3600 + minutes_utc*60 + seconds_utc)
            utc_nsecs = int((utc_sec * (10**7)) % (10**7))

            """Position Data"""
            Latitude = float(data2[2])
            lat_decimal_degree = int(Latitude/100)
            lat_mm = float(Latitude) - (lat_decimal_degree*100)
            lat_ddmm = float(lat_decimal_degree + lat_mm/60)
            print(f'lat_converted:{lat_ddmm}')
            
            if data2[3]=='S':
                lat_ddmm = lat_ddmm*(-1)

            longitude = float(data2[4])
            long_decimal_degree = int(longitude/100)
            long_mm = float(longitude) - (long_decimal_degree*100)
            long_ddmm = float(long_decimal_degree + long_mm/60)
            print(f'lat_converted:{long_ddmm}')
            
            
            if data2[5]=='W':
                long_ddmm=long_ddmm*(-1)
            hdop = float(data2[8])
            altitude = float(data2[9])

            utm_cord = utm.from_latlon(lat_ddmm, long_ddmm)
            print(f'UTM_East, UTM_north, Zone, Letter: {utm_cord}')
            msg = gps_msg()
            msg.header.stamp.secs = int(utc_sec)
            msg.header.stamp.nsecs = int(utc_nsecs)
            msg.header.frame_id = "GPS1_FRAME"
            msg.HDOP = hdop
            msg.Latitude = lat_ddmm
            msg.Longitude = long_ddmm
            msg.Altitude = altitude
            msg.UTM_easting = utm_cord[0]
            msg.UTM_northing = utm_cord[1]
            msg.Zone = utm_cord[2]
            msg.Letter = utm_cord[3]
            rospy.loginfo(msg)
            publisher.publish(msg)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("Error")
        pass
