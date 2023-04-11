import bagpy
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import rosbag
import numpy as np
from scipy.integrate import cumulative_trapezoid, cumtrapz
from scipy.signal import butter, filtfilt
from scipy import integrate
plt.rcParams.update({'font.size': 22})


def tilt_compensation(mx, my, mz, roll, pitch):
    capX = mz * np.sin(roll) - my * np.cos(roll)
    capY = mx * np.cos(pitch) + my * np.sin(pitch) * np.sin(roll) + mz * np.sin(pitch) * np.cos(roll)
    return np.arctan2(capX, capY) * 180 / np.pi

def quaternion_to_euler(x, y, z, w):
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).
    """
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def rotate_vector(x, y, angle):
    """
    Rotate 2D vector (x, y) by specified angle (in radians).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    x_new = x * c - y * s
    y_new = x * s + y * c
    return x_new, y_new



path = "Data/2023-03-25-20-40-18.bag"
data1 = rosbag.Bag(path)
msgs_imu = [msg for topic, msg, t in data1.read_messages(topics=['/imu'])]
msgs_gps = [msg for topic, msg, t in data1.read_messages(topics=['/gps'])]
msgs_circles = msgs_imu[1200:10000]
imu_drive = msgs_imu[10000:100000]
gps_drive = msgs_gps[30:2500]
print(imu_drive[0])



mag = np.array([[msg.mag_field.magnetic_field.x * 1000, msg.mag_field.magnetic_field.y * 1000, msg.mag_field.magnetic_field.z * 1000] for msg in msgs_circles])
x_data = mag[:,0]
y_data = mag[:,1]
z_data = mag[:,2]

x_offset = (x_data.max() + x_data.min()) / 2
y_offset = (y_data.max() + y_data.min()) / 2
z_offset = (z_data.max() + z_data.min()) / 2

x_data = x_data - x_offset
y_data = y_data - y_offset
z_data = z_data - z_offset

# Soft iron correction
avg_x = (x_data.max() - x_data.min())/2
avg_y = (y_data.max() - y_data.min())/2
avg_z = (z_data.max() - z_data.min())/2
avg = (avg_x + avg_y + avg_z)/3


scale_x = avg/ avg_x
scale_y = avg/ avg_y
scale_z = avg/ avg_z

x_data = x_data * scale_x
y_data = y_data * scale_y
z_data = z_data * scale_z

 #Plot the magnetometer data before and after correction
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].plot(mag[:,0], mag[:,1], 'o', label='Raw data')
axs[0].plot(mag[:,0], mag[:,1], 'rx', label=' uncorrected data')
axs[0].set_title("uncorrected data")
axs[0].axis('equal')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].legend()
axs[1].scatter(x_data, y_data, label='Soft iron corrected data')
axs[1].set_title('Soft iron corrected data')
plt.show()






imu_times = np.zeros(len(imu_drive))
for i in range(len(imu_drive)):
    imu_times[i] = imu_drive[i].Header.stamp.secs

imu_times -= np.min(imu_times)
mag_field = np.array([[msg.mag_field.magnetic_field.x * 1000, msg.mag_field.magnetic_field.y * 1000, msg.mag_field.magnetic_field.z * 1000] for msg in imu_drive])
x_org = mag_field[:,0]
y_org = mag_field[:,1]
z_org = mag_field[:,2]

# Compute corrected and scaled magnetic field sensor data
x_data = (x_org - x_offset) * scale_x
y_data = (y_org - y_offset) * scale_y
z_data = (z_org - z_offset) * scale_z

 #Plot the magnetometer data before and after correction
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].plot(mag[:,0], mag[:,1], 'o', label='Raw data')
axs[0].plot(mag_field[:,0], mag_field[:,1], 'rx', label='uncorrected')
axs[0].axis('equal')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].legend()
axs[1].scatter(x_data, y_data, label='corrected')
plt.show()


yaw_data = np.array([msg.yaw for msg in imu_drive])
pitch_data = np.array([msg.pitch for msg in imu_drive])
roll_data = np.array([msg.roll for msg in imu_drive])
yaw_from_mag_raw = np.arctan2(-x_data, y_data)
plt.plot(yaw_data, "b", linewidth=2)
plt.plot(yaw_from_mag_raw, "r", linewidth=2)
plt.title("YAW from IMU vs Raw YAW from Mag")
plt.xlabel("Number of readings")
plt.ylabel("Yaw (radians)")
plt.grid(True)
plt.legend(["YAW from IMU", "Raw YAW from Magnetometer"])
plt.savefig("YAW_IMU_vs_Raw_YAW_Mag.png")


yaw_correct_comp_tilt = tilt_compensation(x_data, y_data, z_data,roll_data,pitch_data)
yaw_original_comp_tilt = tilt_compensation(x_org, y_org, z_org,roll_data,pitch_data)

# plot corrected and uncorrected yaw angles
fig, ax = plt.subplots()
ax.plot(imu_times, yaw_correct_comp_tilt, label='Corrected')
ax.plot(imu_times, yaw_original_comp_tilt, label='Uncorrected', color='red')
ax.set_xlabel("Time (Seconds)")
ax.set_ylabel("Yaw angle (degrees)")
ax.set_title("Corrected and Uncorrected Yaw Angle (Circular Data)")
ax.legend()
ax.grid()
plt.show()

gyro_x = np.array([msg.imu.angular_velocity.x for msg in imu_drive])
gyro_y = np.array([msg.imu.angular_velocity.y for msg in imu_drive])
gyro_z = np.array([msg.imu.angular_velocity.z for msg in imu_drive])

# Integrate gyro data to get yaw angle
yaw_rate = gyro_z  # Assuming gyro_z is the yaw rate data
yaw_angle = (cumulative_trapezoid(yaw_rate, imu_times, initial=0.0)) * (180 / np.pi)


# plot corrected and uncorrected yaw angles
fig, ax = plt.subplots()
ax.plot(imu_times, yaw_correct_comp_tilt, label='magnetometer yaw angle')
ax.plot(imu_times, yaw_angle, label='integrated yaw angle', color='red')
ax.set_xlabel("Time (Seconds)")
ax.set_ylabel("Yaw angle (degrees)")
ax.set_title("Corrected and integrated Yaw Angle")
ax.legend(fontsize = 20)
ax.grid()
plt.show()




# Define the sampling frequency and time step
fs = 100  # Hz
dt = 1/fs  # seconds

# Define the cutoff frequencies for the low-pass and high-pass filters
fc_lp = 1  # Hz
fc_hp = 10  # Hz

# Define the filter order
order = 2



# Define the Butterworth filter coefficients
b_lp, a_lp = butter(order, fc_lp*dt*2, 'low')
b_hp, a_hp = butter(order, fc_hp*dt*2, 'high')

# Filter the magnetometer and gyro yaw measurements
yaw_lp = filtfilt(b_lp, a_lp, yaw_correct_comp_tilt)
yaw_hp = filtfilt(b_hp, a_hp, yaw_angle)

# Apply the complementary filter
alpha = 0.98
yaw_cf = np.zeros_like(yaw_correct_comp_tilt)

# Combine the filtered magnetometer and gyro yaw measurements using a complementary filter
yaw_cf = np.zeros_like(yaw_correct_comp_tilt)
alpha = 0.98
yaw_cf[0] = alpha*yaw_lp[0] + (1-alpha)*yaw_hp[0]
for i in range(1, len(yaw_correct_comp_tilt)):
    yaw_cf[i] = alpha*(yaw_cf[i-1] + yaw_hp[i] - yaw_hp[i-1]) + (1-alpha)*yaw_lp[i]

# Plot the results
t = np.arange(len(yaw_correct_comp_tilt)) / fs


plt.plot(t, yaw_lp, label='Low-pass filtered')
plt.plot(t, yaw_hp, label='High-pass filtered')
plt.plot(t, yaw_cf, label='Complementary filtered')
plt.xlabel('Time (s)')
plt.ylabel('Yaw angle (degrees)')
plt.legend(fontsize = 18)
plt.show()



accel_x = np.array([msg.imu.linear_acceleration.x for msg in imu_drive])
velocity_forward = cumulative_trapezoid(accel_x, x=imu_times, initial=0)
# orientation_x = np.array([msg.imu.orientation.x for msg in imu_drive])
# velocity_forward = velocity * np.cos(orientation_x)

# Define the filter parameters
cutoff_freq = 5 # Hz
fs = 100 # Sampling frequency
print(fs)
order = 4 # Filter order

# Define the Butterworth filter
b, a = butter(order, cutoff_freq/(fs/2), btype='lowpass')

# Apply the filter to the velocity data using the filtfilt function (zero-phase filtering)
filtered_velocity_forward_x = filtfilt(b, a, velocity_forward)

poly_coeffs = np.polyfit(imu_times, filtered_velocity_forward_x, deg=3)
poly = np.poly1d(poly_coeffs)

adjusted_velocity = filtered_velocity_forward_x[20000:80000] - poly(imu_times[20000:80000])
adjusted_velocity[adjusted_velocity < 0] = 0 # Adjust negative velocities to zero

adjusted_velocity1 = filtered_velocity_forward_x - poly(imu_times)
adjusted_velocity1[adjusted_velocity1 < 0] = 0 # Adjust negative velocities to zero

plt.plot(imu_times[20000:80000], filtered_velocity_forward_x[20000:80000], label = "accel velocity")
plt.plot(imu_times[20000:80000], adjusted_velocity , label = "polyfit linear velocity")
plt.xlabel('Time (s)', fontsize = 22)
plt.ylabel('Velocity (m/s)', fontsize = 22)
plt.legend(fontsize = 22)
plt.show()

#gps
UTM_easting = np.array([msg.UTM_easting for msg in gps_drive])
UTM_easting = UTM_easting - UTM_easting.min()
UTM_northing = np.array([msg.UTM_northing for msg in gps_drive])
UTM_northing = UTM_northing - UTM_northing.min()

time_gps = list(range(0, len(gps_drive)*1, 1)) 



# Calculate distance travelled between consecutive GPS measurements
delta_easting = np.diff(UTM_easting)
delta_northing = np.diff(UTM_northing)
distance = np.sqrt(delta_easting**2 + delta_northing**2)

# Calculate time interval between consecutive GPS measurements
delta_time = np.diff(time_gps)

# Calculate velocity as distance divided by time interval
velocity_gps = distance / delta_time

forward_velocity_gps = np.interp(np.arange(0, len(velocity_gps), 1/40), np.arange(0, len(velocity_gps)), velocity_gps)

# Plot the estimated velocity over time
plt.plot(imu_times[20000:80000], forward_velocity_gps[20000:80000], label = "GPS velocity")
plt.plot(imu_times[20000:80000], adjusted_velocity , label = "polyfit linear velocity")
plt.title("Estimated Velocity from GPS", fontsize = 22)
plt.xlabel("Time (s)", fontsize = 22)
plt.ylabel("Velocity (m/s)", fontsize = 22)
plt.legend(fontsize = 22)
plt.show()

#displacement by integrating forward velocity
disp_imu = cumulative_trapezoid(adjusted_velocity,imu_times[20000:80000], initial=0)
disp_gps = cumulative_trapezoid(forward_velocity_gps[20000:80000],imu_times[20000:80000] , initial = 0)

# plot the data
fig, ax = plt.subplots()
ax.plot(imu_times[20000:80000], disp_imu, label='IMU')
ax.plot(imu_times[20000:80000], disp_gps, label='GPS')
ax.legend()
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Displacement (meters)')
ax.set_title('Displacement vs. Time')
ax.legend()
plt.show()


from scipy import integrate




accel_x = np.array([msg.imu.linear_acceleration.x for msg in imu_drive])
ang_z = np.array([msg.imu.angular_velocity.z for msg in imu_drive])
y_obs = np.array([msg.imu.linear_acceleration.y for msg in imu_drive])
y_obs_velocity = cumulative_trapezoid(y_obs, imu_times, initial = 0)
print(len(y_obs))
w = np.array([msg.imu.orientation.w for msg in imu_drive])
x = np.array([msg.imu.orientation.x for msg in imu_drive])
y = np.array([msg.imu.orientation.y for msg in imu_drive])
z = np.array([msg.imu.orientation.z for msg in imu_drive])
t = np.array([msg.Header.stamp.secs for msg in imu_drive])
t = np.concatenate(([0], t))

heading_mag = yaw_data

# Compute Ẋ by integrating the x-component of acceleration
velocity_x = cumulative_trapezoid(accel_x, x=imu_times, initial=0)

# Compute ωẊ by multiplying angular velocity and x-component of velocity
ang_vel_x = ang_z * velocity_x

# Compute estimated y-component of acceleration
y_est = -ang_vel_x
y_est_velocity = cumulative_trapezoid(y_est, imu_times, initial = 0)

# Plot the comparison between measured and estimated y-component of acceleration
plt.plot(imu_times, y_obs, label='Measured')
plt.plot(imu_times, y_est, label='Estimated')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)', fontsize = 24)
plt.title('Comparison between measured and estimated acceleration',fontsize = 24)
plt.legend(fontsize = 24)
plt.show()

#Integrate the x and y components of velocity to estimate the trajectory
ve = np.cumsum(velocity_x * np.cos(heading_mag))
vn = np.cumsum(velocity_x * np.sin(heading_mag))
xn = cumulative_trapezoid(vn, x=imu_times, initial=0) 
xe = cumulative_trapezoid(ve, x=imu_times, initial=0) 


UTM_easting_i = np.interp(np.arange(0, len(UTM_easting), 1/40), np.arange(0, len(UTM_easting)), UTM_easting)
UTM_northing_i = np.interp(np.arange(0, len(UTM_northing), 1/40), np.arange(0, len(UTM_northing)), UTM_northing)


import numpy as np


# Load data
accel_x = np.array([msg.imu.linear_acceleration.x for msg in imu_drive])
ang_z = np.array([msg.imu.angular_velocity.z for msg in imu_drive])
y_obs = np.array([msg.imu.linear_acceleration.y for msg in imu_drive])
w = np.array([msg.imu.orientation.w for msg in imu_drive])
x = np.array([msg.imu.orientation.x for msg in imu_drive])
y = np.array([msg.imu.orientation.y for msg in imu_drive])
z = np.array([msg.imu.orientation.z for msg in imu_drive])
t = np.array([msg.Header.stamp.secs for msg in imu_drive])
t = np.concatenate(([0], t))


# Define initial position and orientation
x_0 = 0
y_0 = 0
theta_0 = yaw_data[0]

# Initialize position and orientation arrays
x = np.zeros(len(imu_drive))
y = np.zeros(len(imu_drive))
theta = np.zeros(len(imu_drive))

dx = cumulative_trapezoid(adjusted_velocity1, imu_times, initial = 0)
dy = cumulative_trapezoid(y_est, imu_times, initial = 0)
# Loop over imu_drive data and compute trajectory using dead reckoning
for i in range(len(imu_drive)-1):
    # Compute time elapsed from previous step
    
    # Compute change in orientation
    dtheta_i = yaw_data[i+1] - yaw_data[i]
    
    # Compute change in x and y coordinates
    dx_i = dx[i]
    dy_i = dy[i]
    
    # Compute new orientation
    theta_i1 = theta[i] + dtheta_i
    
    # Compute new position
    x_i1 = x[i] + dx_i*np.cos(theta_i1)
    y_i1 = y[i] + dy_i*np.sin(theta_i1)
    
    # Store new position and orientation
    x[i+1] = x_i1
    y[i+1] = y_i1
    theta[i+1] = theta_i1
    
# Plot the trajectory
import matplotlib.pyplot as plt
plt.plot(x*(10^-12), -y*(10^-12))
plt.plot(UTM_easting, UTM_northing)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.show()







# try 2
import numpy as np

# Constants
GRAVITY = 9.81  # m/s^2
DT = 0.01
# Extract data from IMU
accel_x = np.array([msg.imu.linear_acceleration.x for msg in imu_drive])
ang_z = np.array([msg.imu.angular_velocity.z for msg in imu_drive])
y_obs = np.array([msg.imu.linear_acceleration.y for msg in imu_drive])
w = np.array([msg.imu.orientation.w for msg in imu_drive])
x = np.array([msg.imu.orientation.x for msg in imu_drive])
y = np.array([msg.imu.orientation.y for msg in imu_drive])
z = np.array([msg.imu.orientation.z for msg in imu_drive])
t = np.array([msg.Header.stamp.secs for msg in imu_drive])
t = np.concatenate(([0], t))
t = t / 1e9 
# Initializations
theta = np.zeros_like(ang_z)  # Heading from gyro measurements
vel_east = np.zeros_like(accel_x)  # Initial eastward velocity
vel_north = np.zeros_like(accel_x)  # Initial northward velocity
pos_east = np.zeros_like(accel_x)  # Initial eastward position
pos_north = np.zeros_like(accel_x)  # Initial northward position

# Iterate over IMU measurements and estimate trajectory using dead reckoning
for i in range(1, len(accel_x)):
    # Compute acceleration in body frame
    accel_bx = accel_x[i] - GRAVITY*np.sin(theta[i-1])
    accel_by = y_obs[i] - GRAVITY*np.cos(theta[i-1])
    
    # Update velocity using trapezoidal rule
    vel_east[i] = vel_east[i-1] + (accel_bx + accel_x[i-1])/2*DT
    vel_north[i] = vel_north[i-1] + (accel_by + y_obs[i-1])/2*DT
    
    # Rotate velocity based on heading from magnetometer
    heading = heading_mag[i] if heading_mag is not None else theta[i-1]  
    vel = np.array([vel_east[i], vel_north[i]])
    R = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
    vel_rotated = np.dot(R, vel)
    vel_east[i], vel_north[i] = vel_rotated[0], vel_rotated[1]
    
    # Update position using trapezoidal rule
    pos_east[i] = pos_east[i-1] + (vel_east[i] + vel_east[i-1])/2*DT
    pos_north[i] = pos_north[i-1] + (vel_north[i] + vel_north[i-1])/2*DT
    
    # Compute orientation using gyro measurements
    dtheta = ang_z[i]*DT
    theta[i] = theta[i-1] + dtheta

# Plot trajectory
plt.plot(pos_east, pos_north)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Dead Reckoning Trajectory')
plt.legend()
plt.show()


import numpy as np


# # Define the x-coordinate of the displacement vector r and the initial x-coordinate of the position vector x
# r_x = 0
# x_c = 0

# # Compute v_x and x_c using the formulas
# v_x = accel_x + 0 * r_x + y * ang_z[-1] - z[-1] * y_obs[-1] - z[-1]**2 * r_x
# x_c = (velocity_forward - v_x)/ang_z

# Define the initial displacement vector r0
r0_x = 0
print(f"time1:{t[2]}")
# Compute the displacement vector r and the center of rotation C
r_x = r0_x + np.cumsum(accel_x * (t[1:] - t[:-1]))
x_c = -r_x[-1] * ang_z[-1] / (z[-1]**2)

# Compute v_x using the formula
v_x = accel_x + 0 * r_x + y * ang_z[-1] - z[-1] * y_obs[-1] - z[-1]**2 * r_x

print(v_x)
print(x_c.mean())




#hard coded
yaw_correction = yaw_data.copy()

yaw_correction[34860:34863] = 3.098
yaw_correction[34863:48533] = yaw_data[34863:48533] + 6.28
yaw_correction[48532:48536] = 3.12

yaw_correction[61189:61290] = 3.12
yaw_correction[61290:61312] = 3.14
yaw_correction[61306:84980] = yaw_data[61306:84980] + 6.24

yaw_correction[84980:85601] = 3.4
yaw_correction[85602:90001] = yaw_correction[85602:90001] + 6.23
heading_mag = yaw_correction
import numpy as np

difference = np.zeros(len(imu_drive))

for i in range(len(imu_drive)-1):
    difference[i] = (accel_x[i+1] - accel_x[i]) * 40

difference[len(imu_drive)-1] = 0


accel_x_corrected = accel_x - difference

forward_velocity_corrected = cumulative_trapezoid(accel_x_corrected, imu_times, initial=0)

for i in range(len(forward_velocity_corrected)):
    if forward_velocity_corrected[i] <= 0:
        forward_velocity_corrected[i] = 0


displace_imu = cumulative_trapezoid(forward_velocity_corrected, imu_times, initial =0)
imu_x = np.zeros((len(displace_imu), 1))
imu_y = np.zeros((len(displace_imu), 1))

for i in range(1, len(displace_imu)):
    imu_x[i, 0] = imu_x[i-1, 0] + np.linalg.norm(displace_imu[i] - displace_imu[i-1]) * np.cos(heading_mag[i-1])
    imu_y[i, 0] = imu_y[i-1, 0] + np.linalg.norm(displace_imu[i] - displace_imu[i-1]) * np.sin(heading_mag[i-1])

imu_x = imu_x - imu_x[0]
imu_y = imu_y - imu_y[0]

imu_x_scaled = imu_x
imu_y_scaled = imu_y

UTM_easting_i = np.interp(np.arange(40), np.arange(len(UTM_easting)), UTM_easting)
UTM_northing_i = np.interp(np.arange(40), np.arange(len(UTM_northing)), UTM_northing)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(UTM_easting_i[29609:59102], "b", linewidth=2)
ax1.plot(imu_x_scaled[29609:59102], "r", linewidth=2)
ax1.set_title("Easting Plot")
ax1.grid(True)

ax2.plot(UTM_northing_i[29609:59102], "b", linewidth=2)
ax2.plot(imu_y_scaled[29609:59102], "r", linewidth=2)
ax2.set_title("Northing Plot")
ax2.grid(True)

plt.show()

#imu_x_scaled(38019:59102)--
fig = plt.figure()

ax = fig.add_subplot(111)

imu_x_man = imu_x_scaled[29500:48500]
imu_y_man = imu_y_scaled[29500:48500]

imu_x_man = imu_x_man - 2700
imu_y_man = imu_y_man - 447
ang = np.pi/2.5
imu_x_man_rot = imu_x_man*np.cos(ang) + imu_y_man*np.sin(ang)
imu_y_man_rot = -imu_x_man*np.sin(ang) + imu_y_man*np.cos(ang)

imu_x_man_rot = imu_x_man_rot - 700
imu_y_man_rot = imu_y_man_rot + 1700

imu_f_x = imu_y_man_rot
imu_f_y = imu_x_man_rot

ang = np.pi/3.5
imu_x_final = imu_f_x*np.cos(ang) + imu_f_y*np.sin(ang)
imu_y_final = -imu_f_x*np.sin(ang) + imu_f_y*np.cos(ang)

imu_y_final = imu_y_final + 1200
ax.plot(UTM_easting_i[29500:59500], UTM_northing_i[29500:59500], "b", linewidth=2)
ax.plot(imu_x_final, imu_y_final, "r", linewidth=2)
ax.legend(["GPS Path","IMU Path"])
ax.set_title("Dead reckoning")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
plt.legend(fontsize = 22)
plt.show()
