import pandas as pd
import bagpy
from bagpy import bagreader
import matplotlib.pyplot as plt
import numpy as np

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




path = "data/group/group.bag"


bag = bagreader(path)
data = bag.message_by_topic('/imu')
data_in_csv = pd.read_csv(data)

# angular  velocity, linear acceleration and orientation

time = data_in_csv['Time'].to_numpy()
accelx = data_in_csv['imu.linear_acceleration.x'].to_numpy()
accely = data_in_csv['imu.linear_acceleration.y'].to_numpy()
accelz = data_in_csv['imu.linear_acceleration.z'].to_numpy()

gyro_x = data_in_csv['imu.angular_velocity.x'].to_numpy()
gyro_y = data_in_csv['imu.angular_velocity.y'].to_numpy()
gyro_z = data_in_csv['imu.angular_velocity.z'].to_numpy()

orienx = data_in_csv['imu.orientation.x'].to_numpy()
orieny = data_in_csv['imu.orientation.y'].to_numpy()
orienz = data_in_csv['imu.orientation.z'].to_numpy()
orienw = data_in_csv['imu.orientation.w'].to_numpy()
roll,pithc,yaw = quaternion_to_euler(orienx,orieny,orienz,orienw)
time = time - time[0]
accelz = accelz - accelz[0]

#print(f"Size of accelx - {accelx.size}")

# Video 1 - Linear accelerating in x axis 0 to 35 seconds

# Create the figure and subplots
fig1, ax1 = plt.subplots(1, 3, figsize=(12, 4))

start_time = 40 #Start time of the video
end_time = 47 #End time of the video

start_index = np.abs(time - start_time).argmin()
idx_end = np.abs(time - end_time).argmin()

# Plot the data on each subplot
ax1[0].plot(time[start_index:idx_end], accelx[start_index:idx_end])
ax1[0].set_title('gyro_x vs Time')
ax1[0].set_xlabel('Time (in secs)', labelpad=12)
ax1[0].set_ylabel('gyro_x (in m/s)')

ax1[1].plot(time[start_index:idx_end], accely[start_index:idx_end])
ax1[1].set_title('gyro_y vs Time')
ax1[1].set_xlabel('Time (in secs)', labelpad=12)
ax1[1].set_ylabel('gyro_y (in m/s)')

ax1[2].plot(time[start_index:idx_end], accelz[start_index:idx_end])
ax1[2].set_title('gyro_z vs Time')
ax1[2].set_xlabel('Time (in secs)', labelpad=12)
ax1[2].set_ylabel('gyro_z (in m/s)')

fig1.suptitle('gyro x,y,z vs. Time - From 0:40 to 0:47 ')
plt.subplots_adjust(wspace=0.5)
fig1.savefig(path.split('.')[0] + 'first_clip' + '.png')
plt.show()

# Video 2 - Linear accelerating in x axis 1:50 - 2:00 
# Create the figure and subplots
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))

start_time = 110 #Start time of the video clip
end_time = 120 #End time of the video clip

start_index = np.abs(time - start_time).argmin()
idx_end = np.abs(time - end_time).argmin()

# Plot the data on each subplot
ax2[0].plot(time[start_index:idx_end], accelx[start_index:idx_end])
ax2[0].set_title('accelx vs time')
ax2[0].set_xlabel('Time (in secs)', labelpad=12)
ax2[0].set_ylabel('accel_x (in m/s^2)')

ax2[1].plot(time[start_index:idx_end], accely[start_index:idx_end])
ax2[1].set_title('accely vs time')
ax2[1].set_xlabel('Time (in secs)', labelpad=12)
ax2[1].set_ylabel('accel_y (in m/s^2)')

ax2[2].plot(time[start_index:idx_end], accelz[start_index:idx_end])
ax2[2].set_title('accelz vs time')
ax2[2].set_xlabel('Time (in secs)', labelpad=12)
ax2[2].set_ylabel('accel_z (in m/s^2)')

fig2.suptitle('Acceleration x,y,z vs. Time - From 1:50 to 2:00')
plt.subplots_adjust(wspace=0.5)
fig2.savefig(path.split('.')[0] + 'second_clip' + '.png')
plt.show()

# Video 3 - Linear accelerating in x axis 2:13 - 2:25 
# Create the figure and subplots
fig3, ax3 = plt.subplots(1, 3, figsize=(12, 4))

start_time = 133 #Start time of the video clip
end_time = 145 #End time of the video clip

start_index = np.abs(time - start_time).argmin()
idx_end = np.abs(time - end_time).argmin()

# Plot the data on each subplot
ax3[0].plot(time[start_index:idx_end], accelx[start_index:idx_end])
ax3[0].set_title('roll vs time')
ax3[0].set_xlabel('Time (in secs)', labelpad=12)
ax3[0].set_ylabel('roll (radian)')

ax3[1].plot(time[start_index:idx_end], accely[start_index:idx_end])
ax3[1].set_title('pitch vs time')
ax3[1].set_xlabel('Time (in secs)', labelpad=12)
ax3[1].set_ylabel('pitch (in radian)')

ax3[2].plot(time[start_index:idx_end], accelz[start_index:idx_end])
ax3[2].set_title('yaw vs time')
ax3[2].set_xlabel('Time (in secs)', labelpad=12)
ax3[2].set_ylabel('yaw (in radian)')

fig3.suptitle('r,p,y vs. Time - From 2:13 to 2:25')
plt.subplots_adjust(wspace=0.5)
fig3.savefig(path.split('.')[0] + 'third_clip' + '.png')
plt.show()

