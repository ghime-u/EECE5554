import bagpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, math
from sklearn.cluster import KMeans
from sklearn import metrics



#known Values
Altitude = 38
open = [327878,4686848]
occluded = [328071, 4689575]




stri = "Data/open/walking/open_walking.bag"
raw =  bagpy.bagreader(stri)
raw.topic_table
data = raw.message_by_topic('/gps')
data_in_csv = pd.read_csv(data)
alt = np.array(data_in_csv['Altitude'])

x = data_in_csv['UTM_easting']
y = data_in_csv['UTM_northing']

data_in_csv['UTM_easting'] = data_in_csv['UTM_easting'] - data_in_csv['UTM_easting'].min()
data_in_csv['UTM_northing'] = data_in_csv['UTM_northing'] - data_in_csv['UTM_northing'].min()
print(data_in_csv[['UTM_easting', 'UTM_northing']])


data_in_csv[['UTM_easting','UTM_northing']].plot()
plt.xlabel('UTM_easting (meters)')
plt.ylabel('UTM_northing (meters)')
plt.title('UTM_easting vs UTM_northing')
plt.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + ".png", bbox_inches='tight')
plt.show()

fig1, ax1 = bagpy.create_fig(1)
ax1[0].scatter(x = 'UTM_easting', y = 'UTM_northing', data = data_in_csv, s= 50, label = 'UTM_easting VS UTM_northing')
plt.xlabel('UTM_easting (meters)')
plt.ylabel('UTM_northing (meters)')
plt.title('UTM_easting vs UTM_northing')
plt.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + "scatter" + ".png", bbox_inches='tight')
plt.show()

# Altitude Error
error = np.array([Altitude - item for item in alt])
alt_mean_error = np.mean(error)
print(alt_mean_error)


error_easting = [item - open[0] for item in x]  # change to open for open dataset
error_northing = [item - open[1] for item in y]
mean_northing = np.mean(error_northing)
mean_easting = np.mean(error_easting)
median_northing = np.median(error_northing)
median_easting = np.median(error_easting)
std_northing = np.std(error_northing)
std_easting = np.std(error_easting)
print(mean_easting, mean_northing, median_easting, median_northing, std_easting, std_northing)




fig3,ax3 = plt.subplots(2)
ax3[0].hist(error_easting, bins = 25)
ax3[0].set_title('Easting error(m)')
ax3[0].set_xlabel('Error in Easting(m)')
ax3[0].set_ylabel('Frequency of Error')
ax3[1].hist(error_northing, bins = 25)
ax3[1].set_title('Northing error')
ax3[1].set_xlabel('Error in Northing(m)')
ax3[1].set_ylabel('Frequency of Error')
plt.show()

fig3.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + "error_hist" + ".png", bbox_inches = 'tight')



fig4, ax4 = bagpy.create_fig(1)
ax4[0].scatter(x = 'Time', y = 'Altitude', data = data_in_csv, s = 50, label = "Time vs Altitude")
for axis in ax1:
    axis.legend()
    axis.set_xlabel('Time (in seconds)', fontsize=40)
    axis.set_ylabel('Altitude (in meters)', fontsize=40)
plt.show()
fig4.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + "Alt-time" + ".png", bbox_inches = 'tight')


"""comment everything below for stationary data as this is not required for stationary data"""
""" Occluded data co-ordinates for each line comment this and uncomment open co-ordinate for the same"""
# ox1,oy1 = np.array(x[0:60]), np.array(y[0:60])
# a1, b1 = np.polyfit(ox1,oy1,1)

# ox2,oy2 = np.array(x[60:72]), np.array(y[60:72])
# a2, b2 = np.polyfit(ox2,oy2,1)

# ox3, oy3 = np.array(x[72:132]), np.array(y[72:132])
# a3, b3 = np.polyfit(ox3,oy3,1)

# ox4,oy4 = np.array(x[132:145]), np.array(y[132:145])
# a4, b4 = np.polyfit(ox4,oy4,1)


""" Open data co-ordinates for each line, comment this and uncomment occluded co-ordinate for the same"""
ox1,oy1 = np.array(x[0:34]), np.array(y[0:34])

a1, b1 = np.polyfit(ox1,oy1,1)

ox2,oy2 = np.array(x[34:76]), np.array(y[34:76])
a2, b2 = np.polyfit(ox2,oy2,1)

ox3, oy3 = np.array(x[76:98]), np.array(y[76:98])
a3, b3 = np.polyfit(ox3,oy3,1)

ox4,oy4 = np.array(x[98:134]), np.array(y[98:134])
a4, b4 = np.polyfit(ox4,oy4,1)


fig2,ax2 = bagpy.create_fig(1)
ax2[0].scatter(x,y)
ax2[0].plot(ox1, a1*ox1+b1,c='black')
# errorl1 = math.sqrt(metrics.mean_squared_error(np.array([ox1, a1*ox1+b1]),np.array([ox1,oy1])))
errorl1 = math.sqrt(metrics.mean_squared_error(np.array([a1*ox1 + b1]),np.array([oy1])))
print(errorl1)

ax2[0].plot(ox2, a2*ox2+b2,c='black')
errorl2 = math.sqrt(metrics.mean_squared_error(np.array([a2*ox2 + b2]),np.array([oy2])))

ax2[0].plot(ox3, a3*ox3+b3, c='black')
errorl3 = math.sqrt(metrics.mean_squared_error(np.array([a3*ox3 + b3]),np.array([oy3])))

ax2[0].plot(ox4, a4*ox4+b4, c='black')
errorl4 = math.sqrt(metrics.mean_squared_error(np.array([a4*ox4 + b4]),np.array([oy4])))

ax2[0].set_title('UTM easting vs UTM northing with line of best fit')
ax2[0].set_xlabel('UTM_easting(m)')
ax2[0].set_ylabel('UTM northing(m)')
plt.show()
fig2.savefig("/root/catkin_ws/src/" + stri.split('.')[0] + "line_of_bestvsutm" + ".png", bbox_inches = 'tight')
totalrms_error = (errorl1 + errorl2 + errorl3 + errorl4)/4
print(totalrms_error)









"""Code for kmeans clustering for line of best fit future update"""
# arr = np.vstack((x, y)).T
# kmeans = KMeans(n_clusters=4, random_state=0, max_iter= 2000).fit(arr)
# labels = kmeans.labels_
# filtered_label0 = arr[labels == 0]
# f0x,f0y = np.split(filtered_label0,2,axis=1)
# a0,b0 = np.polyfit(f0x.flatten(),f0y.flatten(),1)

# filtered_label1 = arr[labels == 1]
# f1x,f1y = np.split(filtered_label1,2,axis=1)
# a1,b1 = np.polyfit(f1x.flatten(),f1y.flatten(),1)

# filtered_label2 = arr[labels == 2]
# f2x,f2y = np.split(filtered_label2,2,axis=1)
# print(f2x.shape)
# a2,b2 = np.polyfit(f2x.flatten(),f2y.flatten(),1)

# filtered_label3 = arr[labels == 3]
# f3x,f3y = np.split(filtered_label3,2,axis=1)
# print(f3x.shape)
# a3,b3 = np.polyfit(f3x.flatten(),f3y.flatten(),1)

# fig3,ax3 = bagpy.create_fig(1)

# line = ax3[0].scatter(x, y, c=labels)
# line0 = ax3[0].scatter(f0x, a0*f0x + b0, c = 'black')
# line1 = ax3[0].scatter(f1x, a1*f1x + b1, c = 'black')
# line2 = ax3[0].scatter(f2x, a2*f2x + b2, c = 'black')
# line3 = ax3[0].scatter(f3x, a3*f3x + b3, c = 'black')
# for axis in ax3:
#     axis.legend([line,line0,line1,line2,line3],['scatter','line0','line1','line2','line3'])
#     axis.set_title("Clustered Rectangle Scatterplot using Kmeans")
#     axis.set_xlabel("UTM Easting") 
#     axis.set_ylabel("UTM Northing")
# plt.show()




# Open data co-ordinates for each line

