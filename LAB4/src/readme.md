## Overview
This repository contains code and documentation for the EECE5554 Lab 4, focusing on Inertial Measurement Unit (IMU), Global Positioning System (GPS), and Magnetometer-based navigation. The lab covers various aspects, including sensor calibration, filter design, velocity estimation, dead reckoning, and trajectory simulation.

## Calibration
The magnetometer is susceptible to distortions caused by nearby objects. Calibration involves hard iron correction (offset correction) and soft iron correction (scale factor correction). The provided figures show the impact of calibration on magnetic field measurements.

## Complementary Filter
A complementary filter is implemented using low-pass filtered magnetometer data and high-pass filtered gyro data. The filter coefficients are determined using the Butterworth filter design method with specified cutoff frequencies.

## Yaw Estimation
The lab explores different yaw estimation methods, including magnetometer-based estimates (adjusted yaw and raw yaw) and gyro-based estimates. The combination of these estimates with specific weighting provides a more robust yaw for navigation.

## Forward Velocity Estimate
A low-pass filter is applied to the forward velocity data to remove noise. A third-order polynomial fit is used to remove long-term drift or bias, resulting in adjusted velocity. The combination of accelerometer-based and GPS-based velocity estimates is discussed for improved accuracy.

## Dead Reckoning
The lab covers dead reckoning using IMU data to estimate acceleration in the y-direction. Factors contributing to discrepancies are discussed, and figures illustrate dead reckoning with and without adjustments.

## Estimating Xc
The formula for estimating the x-coordinate (xc) is provided, considering measured acceleration, angular velocity, and displacement. A sample calculation for xc is included.

## Trajectory Simulation
Instructions for simulating a vehicle's trajectory are provided. The process involves initializing position and orientation variables, calculating changes in position and orientation at each time step, and updating the trajectory for plotting.

## Limitations and Error Analysis
The readme discusses the limitations of dead reckoning, including error accumulation over time and sensitivity to external factors. The estimated heading and velocity errors of the VectorNav VN-100 IMU are considered for navigation duration without a position fix.
