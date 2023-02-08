import bagpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
stri = "Data/occluded_stationary.bag"
raw =  bagpy.bagreader(stri)
raw.topic_table
data = raw.message_by_topic('/gps/gps')
print(data)
data_in_csv = pd.read_csv(data)
x = data_in_csv['UTM_easting']
y = data_in_csv['UTM_northing']
data_in_csv['UTM_easting'] = data_in_csv['UTM_easting'] - data_in_csv['UTM_easting'].min()
data_in_csv['UTM_northing'] = data_in_csv['UTM_northing'] - data_in_csv['UTM_northing'].min()
print(data_in_csv[['UTM_easting', 'UTM_northing']])
print(data_in_csv)
plt.rcParams.update({'font.size': 40})
data_in_csv[['UTM_easting','UTM_northing']].plot()
fig, ax = bagpy.create_fig(1)
for axis in ax:
    axis.legend()
    axis.set_xlabel('UTM_easting (meters)', fontsize=40)
    axis.set_ylabel('UTM_northing (meters)', fontsize=40)
ax[0].scatter(x = 'UTM_easting', y = 'UTM_northing', data = data_in_csv, s= 50, label = 'UTM_easting VS UTM_northing')


fig.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + ".png", bbox_inches='tight')
plt.show()

# close figure to proceed
fig1, ax1 = bagpy.create_fig(1)
ax1[0].scatter(x = 'Time', y = 'Altitude', data = data_in_csv, s = 50, label = "Time vs Altitude")
for axis in ax1:
    axis.legend()
    axis.set_xlabel('Time (in seconds)', fontsize=40)
    axis.set_ylabel('Altitude (in meters)', fontsize=40)


fig1.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + "TvsA" + ".png", bbox_inches = 'tight')
plt.show()

#clode figure to proceed

a, b = np.polyfit(x,y,1)
# plt.plot(x, a*x+b) 
# plt.show()
# fig2, ax2 = bagpy.create_fig(1)
# ax2[0].bar(x-a, y-b)
# plt.show()
open_cord = [327742.06,4689336.16] # exact location
occluded_cord = [327965.56,4689485.00] # exact location

# uncomment if occluded is taken
error_easting = [item - occluded_cord[0] for item in x]
error_northing = [item - occluded_cord[1] for item in y]

#uncomment if open is taken
# error_easting = [item - open_cord[0] for item in x]
# error_northing = [item - open_cord[1] for item in y]


fig3,ax3 = plt.subplots(2)
ax3[0].hist(error_easting, bins = 25)
ax3[0].set_title('Easting error(m)')
ax3[0].set_xlabel('Error in Easting(m)')
ax3[0].set_ylabel('Frequency of Error(m)')
ax3[1].hist(error_northing, bins = 25)
ax3[1].set_title('Northing error')
ax3[1].set_xlabel('Error in Northing(m)')
ax3[1].set_ylabel('Frequency of Error(m)')
plt.show()

fig3.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + "error_hist" + ".png", bbox_inches = 'tight')

mean_northing = np.mean(error_northing)
mean_easting = np.mean(error_easting)
median_northing = np.median(error_northing)
median_easting = np.median(error_easting)
std_northing = np.std(error_northing)
std_easting = np.std(error_easting)

print(f"mean_easting:{mean_easting}")
print(f"mean_northing:{mean_northing}")
print(f"median_easting:{median_easting}")
print(f"median_northing:{median_northing}")
print(f"std_northing:{std_easting}")
print(f"std_easting:{std_northing}")