path = "C:\Users\utkar\downloads\extracted_data\LocationC.bag";
bag = rosbag(path);
sel = select(bag,'Topic','/vectornav');
msgs = readMessages(sel,'DataFormat','struct');

% Initialize empty arrays to store gyro x,y,z values
g_xyz = zeros(length(msgs), 3);

for i = 1:length(msgs)
    values = split(msgs{i}.Data, ',');
    if length(values) == 13
        % Convert strings to numbers
        gyrox = str2double(values{11});
        gyroy = str2double(values{12});
        gyroz_1 = split(values{13}, '*');
        gyroz = str2double(gyroz_1{1});
        g_xyz(i,:) = [gyrox, gyroy, gyroz];
    else
        % If message is invalid, replace with default values
        g_xyz(i,:) = [0, 0, 0];
    end
end

% Replace any NaN or invalid values with default value
g_xyz(isnan(g_xyz) | ~isfinite(g_xyz)) = 0;


Fs = 40;
t0 = 1/Fs;


%{
the commented code gives output without slopes for each axis
theta_gyrox = cumsum(g_xyz(:,1), 1)*t0;
max_num_m = 100;
L = size(theta_gyrox, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), max_num_m).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.


[avarFromFunc, tauFromFunc] = allanvar(g_xyz(:,1), m, Fs);
adevFromFunc = sqrt(avarFromFunc);

figure
loglog(tauFromFunc, adevFromFunc);
title('Allan Deviations for x gyro')
xlabel('\tau')
ylabel('\sigma(\tau)')
grid on
axis equal

% gyro in y
Fs = 40;
t0 = 1/Fs;
theta_gyroy = cumsum(g_xyz(:,2), 1)*t0;
max_num_m = 100;
L = size(theta_gyroy, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), max_num_m).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.


[avarFromFunc, tauFromFunc] = allanvar(g_xyz(:,2), m, Fs);
adevFromFunc = sqrt(avarFromFunc);

figure
loglog(tauFromFunc, adevFromFunc);
title('Allan Deviations for y gyro')
xlabel('\tau')
ylabel('\sigma(\tau)')
grid on
axis equal


% gyro in z
Fs = 40;
t0 = 1/Fs;
theta_gyroz = cumsum(g_xyz(:,3), 1)*t0;
max_num_m = 100;
L = size(theta_gyroz, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), max_num_m).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.


[avarFromFunc, tauFromFunc] = allanvar(g_xyz(:,3), m, Fs);
adevFromFunc = sqrt(avarFromFunc);

figure
loglog(tauFromFunc, adevFromFunc);
title('Allan Deviations for z gyro')
xlabel('\tau')
ylabel('\sigma(\tau)')
grid on
axis equal 
%}

%% %Gyro Time-Series Plot
figure('Name',"Angular Velocity Time Series", 'NumberTitle', 'off')
plot(linspace(1,length(msgs)/40,length(msgs)), g_xyz(:,1),'-', 'color','r')
hold on
plot(linspace(1,length(msgs)/40,length(msgs)), g_xyz(:,2),'-', 'color','g')
hold on
plot(linspace(1,length(msgs)/40,length(msgs)), g_xyz(:,3),'-', 'color','b')
grid on

xlabel('Time (in secs)')
ylabel('Angular Velocity (rad/s)')
legend ('G_X','G_Y','G_Z')
title('Angular Velocity Time Series')
hold off
saveas(gcf, 'g_xyz_vs_Time_LocationC.png')

theta_gyrox = cumsum(g_xyz(:,1), 1)*t0;
max_num_m = 100;
L = size(theta_gyrox, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), max_num_m).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.
tau = m*t0;
[avarFromFunc, tauFromFunc] = allanvar(g_xyz(:,1), m, Fs);
adevFromFunc = sqrt(avarFromFunc);

%  index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tauFromFunc);
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N_gyrox = 10^logN;

%  index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tauFromFunc);
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K_gyrox = 10^logK;

%  index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tauFromFunc); %tau = tauFromFunc
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the bias instability coefficient from the line.
sb = sqrt(2*log(2)/pi);
log_B = b - log10(sb);
B_gyrox = 10^log_B;
% Plot
tau_N = 1;
lineN = N_gyrox ./ sqrt(tau);
% Plot
tauK = 3;
lineK = K_gyrox .* sqrt(tau/3);
% Plot
tau_B = tau(i);
lineB = B_gyrox * sb * ones(size(tau));
% Plot results
figure('Name',"Allan Deviation of Gyro X (With Noise Parameters)", 'NumberTitle', 'off')
loglog(tauFromFunc, adevFromFunc, tauFromFunc, lineN, '--', tau_N, N_gyrox, 'o', tauFromFunc, lineK, '--', tauK, K_gyrox, 'o', tau, lineB, '--', tau_B, sb*B_gyrox, 'o');
legend('\sigma (rad/s)', '\sigma_N ((rad/s)sqrt{Hz})', '','\sigma_K ((rad/s)sqrt{Hz})','','\sigma_B (rad/s)')
title('Allan Deviation with Noise Parameters of Gyro X')
xlabel('\tau')
ylabel('\sigma(\tau)')
text(tau_N, N_gyrox, 'N')
text(tauK, K_gyrox, 'K')
text(tau_B, sb*B_gyrox, '0.664B')
grid on
axis equal
saveas(gcf, 'LocationC_GyroX_Allan.png')

%%
% Allan variance - gyro y
Fs = 40; 
t0 = 1/Fs;
theta = cumsum(g_xyz(:,2), 1)*t0;
max_num_m = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), max_num_m).';
m = ceil(m); 
m = unique(m); 
tau = m*t0;
[avarFromFunc, tauFromFunc] = allanvar(g_xyz(:,2), m, Fs);
adevFromFunc = sqrt(avarFromFunc);

%  index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tauFromFunc);
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N_gyroy = 10^logN;
%  index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tauFromFunc);
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K_gyroy = 10^logK;
%  index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tauFromFunc); %tau = tauFromFunc
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the bias instability coefficient from the line.
sb = sqrt(2*log(2)/pi);
log_B = b - log10(sb);
B_gyroy = 10^log_B;
% Plot
tau_N = 1;
lineN = N_gyroy ./ sqrt(tau);
% Plot 
tauK = 3;
lineK = K_gyroy .* sqrt(tau/3);
% Plot 
tau_B = tau(i);
lineB = B_gyroy * sb * ones(size(tau));
% Plot results
figure('Name',"Allan Deviation of Gyro Y (with Noise Parameters)", 'NumberTitle', 'off')
loglog(tauFromFunc, adevFromFunc, tauFromFunc, lineN, '--', tau_N, N_gyroy, 'o', tauFromFunc, lineK, '--', tauK, K_gyroy, 'o', tau, lineB, '--', tau_B, sb*B_gyroy, 'o');
legend('\sigma (rad/s)', '\sigma_N ((rad/s)sqrt{Hz})', '','\sigma_K ((rad/s)sqrt{Hz})','','\sigma_B (rad/s)')
title('Allan Deviation of Gyro Y with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
text(tau_N, N_gyroy, 'N')
text(tauK, K_gyroy, 'K')
text(tau_B, sb*B_gyroy, '0.664B')
grid on
axis equal
saveas(gcf, 'LocationC_GyroY_Allan.png')
%%
% Allan variance - gyro z
Fs = 40;
t0 = 1/Fs;
theta = cumsum(g_xyz(:,3), 1)*t0;

max_num_m = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), max_num_m).';
m = ceil(m); 
m = unique(m); 
tau = m*t0;
g_xyz(402372,3) = double(0.0);
g_xyz(402371,3) = double(0.0);

[avarFromFunc, tauFromFunc] = allanvar(g_xyz(:,3), m, Fs);
adevFromFunc = sqrt(avarFromFunc);
% index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tauFromFunc);
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N_gyroz = 10^logN;
% index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tauFromFunc);
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K_gyroz = 10^logK;
% index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tauFromFunc); %tau = tauFromFunc
logadev = log10(adevFromFunc);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));
%  y-intercept of the line.
b = logadev(i) - slope*logtau(i);
% Determine the bias instability coefficient from the line.
sb = sqrt(2*log(2)/pi);
log_B = b - log10(sb);
B_gz = 10^log_B;
% Plot
tau_N = 1;
lineN = N_gyroz ./ sqrt(tau);
% Plot
tauK = 3;
lineK = K_gyroz .* sqrt(tau/3);
% Plot
tau_B = tau(i);
lineB = B_gz * sb * ones(size(tau));

% Plot results
figure('Name',"Allan Deviation of Gyro Z (with Noise Parameters)", 'NumberTitle', 'off')
loglog(tauFromFunc, adevFromFunc, tauFromFunc, lineN, '--', tau_N, N_gyroz, 'o', tauFromFunc, lineK, '--', tauK, K_gyroz, 'o', tau, lineB, '--', tau_B, sb*B_gz, 'o');
legend('\sigma (rad/s)', '\sigma_N ((rad/s)sqrt{Hz})', '','\sigma_K ((rad/s)sqrt{Hz})','','\sigma_B (rad/s)')
title('Allan Deviation Gyro Z with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
text(tau_N, N_gyroz, 'N')
text(tauK, K_gyroz, 'K')
text(tau_B, sb*B_gz, '0.664B')
grid on
axis equal
saveas(gcf, 'LocationC_GyroZ_Allan.png')

