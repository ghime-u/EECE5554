import bagpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

raw =  bagpy.bagreader("Data/2023-02-06-10-51-47.bag")
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
ax[0].scatter(x = 'UTM_easting', y = 'UTM_northing', data = data_in_csv, s= 50, label = 'UTM_easting VS UTM_northing')
for axis in ax:
    axis.legend()
    axis.set_xlabel('UTM_easting', fontsize=40)
    axis.set_ylabel('UTM_northing', fontsize=40)
plt.show()