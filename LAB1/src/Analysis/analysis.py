import bagpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
stri = "Data/occluded_stationary.bag"
raw =  bagpy.bagreader(stri)
raw.topic_table
data = raw.message_by_topic('/gps')
print(data)
data_in_csv = pd.read_csv(data)
data_in_csv['UTM_easting'] = data_in_csv['UTM_easting'] - data_in_csv['UTM_easting'].min()
data_in_csv['UTM_northing'] = data_in_csv['UTM_northing'] - data_in_csv['UTM_northing'].min()
print(data_in_csv[['UTM_easting', 'UTM_northing']])
print(data_in_csv)
plt.rcParams.update({'font.size': 40})
data_in_csv[['UTM_easting','UTM_northing']].plot()
fig, ax = bagpy.create_fig(1)
for axis in ax:
    axis.legend()
    axis.set_xlabel('UTM_easting', fontsize=40)
    axis.set_ylabel('UTM_northing', fontsize=40)
ax[0].scatter(x = 'UTM_easting', y = 'UTM_northing', data = data_in_csv, s= 50, label = 'UTM_easting VS UTM_northing')


fig.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + ".png", bbox_inches='tight')
plt.show()

fig1, ax1 = bagpy.create_fig(1)
ax1[0].scatter(x = 'Time', y = 'Altitude', data = data_in_csv, s = 50, label = "Time vs Altitude")
for axis in ax1:
    axis.legend()
    axis.set_xlabel('Time', fontsize=40)
    axis.set_ylabel('Altitude', fontsize=40)


fig1.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + "TvsA" + ".png")
plt.show()
