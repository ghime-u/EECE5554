import bagpy
import math
import csv
import statistics
from bagpy import bagreader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from geometry_msgs.msg import Quaternion

def quaternion_to_euler(x, y, z, w):
    # Yaw (Z), Pitch (Y), Roll (X)
    # yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z**2 + x**2))
    # pitch = math.asin(2 * (w * x - y * z))
    # roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # return roll, pitch, yaw
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y *y)
    roll = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0, +1.0,t2)
    t2 = np.where(t2<-1.0, -1.0,t2)
    pitch = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y+ z * z)
    yaw = np.degrees(np.arctan2(t3, t4))
    return roll, pitch, yaw



plt.rcParams.update({'font.size': 16})




path = 'data/individual/individual.bag'





bag = bagreader(path)
data = bag.message_by_topic('/imu')
csv_f = pd.read_csv(data)
time = csv_f['Time'].to_numpy()
fig, ax = plt.subplots(3, 1, figsize=(30, 18))
fig.subplots_adjust(hspace=0.4)
ax[0].plot(csv_f['Time'], csv_f['IMU.angular_velocity.x'])
ax[1].plot(csv_f['Time'], csv_f['IMU.angular_velocity.y'])
ax[2].plot(csv_f['Time'], csv_f['IMU.angular_velocity.z'])
ax[0].set_xlabel('Time (Seconds)')
ax[0].set_ylabel('Angular Velocity_X (rad/sec)')
ax[0].set_title('Time vs Angular Velocity_X')
ax[1].set_xlabel('Time (Seconds)')
ax[1].set_ylabel('Angular Velocity_Y (rad/sec)')
ax[1].set_title('Time vs Angular Velocity_Y')
ax[2].set_xlabel('Time (Seconds)')
ax[2].set_ylabel('Angular Velocity_Z (rad/sec)')
ax[2].set_title('Time vs Angular Velocity_Z')
plt.savefig(path.split('.')[0] + 'gyroxyz' + '.png')


fig1, ax1 = plt.subplots(3, 1, figsize=(30, 18))
fig1.subplots_adjust(hspace=0.4)
ax1[0].plot(csv_f['Time'], csv_f['IMU.linear_acceleration.x'])
ax1[1].plot(csv_f['Time'], csv_f['IMU.linear_acceleration.y'])
ax1[2].plot(csv_f['Time'], csv_f['IMU.linear_acceleration.z'])
ax1[0].set_xlabel('Time (Seconds)')
ax1[0].set_ylabel('Linear Acceleration_X (m/s)')
ax1[0].set_title('Time vs Linear Acceleration_X')
ax1[1].set_xlabel('Time (Seconds)')
ax1[1].set_ylabel('Linear Acceleration_Y (m/s)')
ax1[1].set_title('Time vs Linear Acceleration_Y')
ax1[2].set_xlabel('Time (Seconds)')
ax1[2].set_ylabel('Linear Acceleration_Z (m/s)')
ax1[2].set_title('Time vs Linear Acceleration_Z')
plt.savefig(path.split('.')[0] + 'Linearacc' + '.png')


"""orientaion quaternion"""
fig2, ax2 = plt.subplots(3, 1, figsize=(30, 18))
fig2.subplots_adjust(hspace=0.4)
ax2[0].plot(csv_f['Time'], csv_f['IMU.orientation.x'])
ax2[1].plot(csv_f['Time'], csv_f['IMU.orientation.y'])
ax2[2].plot(csv_f['Time'], csv_f['IMU.orientation.z'])
ax2[0].set_xlabel('Time (Seconds)')
ax2[0].set_ylabel('orientation_X (m/s)')
ax2[0].set_title('Time vs orientation_X')
ax2[1].set_xlabel('Time (Seconds)')
ax2[1].set_ylabel('orientation_Y (m/s)')
ax2[1].set_title('Time vs orientation_Y')
ax2[2].set_xlabel('Time (Seconds)')
ax2[2].set_ylabel('Linear orientation_Z (m/s)')
ax2[2].set_title('Time vs orientation_Z')
plt.savefig(path.split('.')[0] + 'orientation' + '.png')

w = csv_f['IMU.orientation.w']
x = csv_f['IMU.orientation.x']
y = csv_f['IMU.orientation.y']
z = csv_f['IMU.orientation.z']
print(w, csv_f['IMU.orientation.w'])

# histogram plot
fig3, ax3 = plt.subplots(2, 1, figsize=(30, 18))
fig3.subplots_adjust(hspace=0.4)
ax3[0].hist(csv_f['IMU.linear_acceleration.x'], bins= 40)
# ax3[1].hist(csv_f['IMU.linear_acceleration.y'], bins= 40)
# ax3[2].hist(csv_f['IMU.linear_acceleration.z'], bins= 40)
ax3[0].set_xlabel('Linear Acceleration_X (m/s))')
ax3[0].set_ylabel('Frequency')
ax3[0].set_title('Linear Acceleration_X (m/s) vs Frequency')
# ax3[1].set_xlabel('Linear Acceleration_Y (m/s\u00b2))')
# ax3[1].set_ylabel('Frequency')
# ax3[1].set_title('Linear Acceleration_Y (m/s\u00b2) vs Frequency')
# ax3[2].set_xlabel('Linear Acceleration_Z (m/s\u00b2)')
# ax3[2].set_ylabel('Frequency')
# ax3[2].set_title('Linear Acceleration_Z (m/s\u00b2) vs Frequency')
plt.savefig(path.split('.')[0] + 'accelxwithnoise' + '.png')

# mean, deviation, median of noise
roll_x, pitch_y, yaw_z = quaternion_to_euler(x,y,z,w)

# noise distribution as minimum would be the default
mean_time = csv_f['Time'] - csv_f['Time'].min()
ang_vel_x = csv_f['IMU.angular_velocity.x'] - csv_f['IMU.angular_velocity.x'].mean()
ang_vel_y = csv_f['IMU.angular_velocity.y'] - csv_f['IMU.angular_velocity.y'].mean()
ang_vel_z = csv_f['IMU.angular_velocity.z'] - csv_f['IMU.angular_velocity.z'].mean()
lin_acc_x = csv_f['IMU.linear_acceleration.x'] - csv_f['IMU.linear_acceleration.x'].mean()
lin_acc_y = csv_f['IMU.linear_acceleration.y'] - csv_f['IMU.linear_acceleration.y'].mean()
lin_acc_z = csv_f['IMU.linear_acceleration.z'] - csv_f['IMU.linear_acceleration.z'].mean()



print('Mean, Median & Standard Deviation of Linear Acceleration:')
print('mean = ',lin_acc_x.mean())
print('median = ',lin_acc_x.median())
print('standard deviation = ', lin_acc_x.std())


fig4, ax4 = plt.subplots(3, 1, figsize=(30, 18))
fig4.subplots_adjust(hspace=0.4)
ax4[0].hist(ang_vel_x, bins= 40)
ax4[1].hist(ang_vel_y, bins= 40)
ax4[2].hist(ang_vel_z, bins= 40)
ax4[0].set_xlabel('Angular Velocity_X (rad/sec)')
ax4[0].set_ylabel('Frequency')
ax4[0].set_title('Angular Velocity_X (rad/sec) vs Frequency')
ax4[1].set_xlabel('Angular Velocity_Y (rad/sec)')
ax4[1].set_ylabel('Frequency')
ax4[1].set_title('Angular Velocity_Y (rad/sec) vs Frequency')
ax4[2].set_xlabel('Angular Velocity_Z (rad/sec)')
ax4[2].set_ylabel('Frequency')
ax4[2].set_title('Angular Velocity_Z (rad/sec) vs Frequency')
plt.savefig(path.split('.')[0] + 'gyroxyzwithoutnoise' + '.png')


fig5, ax5 = plt.subplots(3, 1, figsize=(30, 18))
fig5.subplots_adjust(hspace=0.4)
ax5[0].hist(lin_acc_x, bins= 40)
ax5[1].hist(lin_acc_y, bins= 40)
ax5[2].hist(lin_acc_z, bins= 40)
ax5[0].set_xlabel('Linear Acceleration_X (m/s\u00b2))')
ax5[0].set_ylabel('Frequency')
ax5[0].set_title('Linear Acceleration_X (m/s\u00b2) vs Frequency')
ax5[1].set_xlabel('Linear Acceleration_Y (m/s\u00b2))')
ax5[1].set_ylabel('Frequency')
ax5[1].set_title('Linear Acceleration_Y (m/s\u00b2) vs Frequency')
ax5[2].set_xlabel('Linear Acceleration_Z (m/s\u00b2)')
ax5[2].set_ylabel('Frequency')
ax5[2].set_title('Linear Acceleration_Z (m/s\u00b2) vs Frequency')
plt.savefig(path.split('.')[0] + 'accelwithoutnoise' + '.png')
















plt.show()