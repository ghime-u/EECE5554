
% Camera Calibration using Checkerboard Images

% Load checkerboard images from a folder
imageFolder = 'calib_resize';
images = dir(fullfile(imageFolder, '*.jpg')); % Change extension as necessary
numImages = length(images);
imageFileNames = {images.name};
imageLoc = fullfile(imageFolder,imageFileNames);
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageLoc);
original_image = imread(imageLoc{1});
[mrows, ncols, ~] = size(original_image);

squareSize = 30;
worldPoints = generateCheckerboardPoints(boardSize, squareSize);
[cameraparams , imagesUsed, estimateError] = estimateCameraParameters(imagePoints, worldPoints,'EstimateSkew',false, 'EstimateTangentialDistortion',false,'NumRadialDistortionCoefficients',2,'WorldUnits','millimeters','InitialIntrinsicMatrix',[],'InitialRadialDistortion',[], 'ImageSize', [mrows,ncols]);

% Define output folder for undistorted images
outputFolder = 'undistorted';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Loop through each image and undistort it
for i = 1:numImages
    % Read image
    image = imread(imageLoc{i});
    
    % Undistort image
    undistortedImage = undistortImage(image, cameraparams);
    
    % Save undistorted image to output folder
    [~, imageName, imageExt] = fileparts(imageFileNames{i});
    undistortedImageName = [imageName, '_undistorted', imageExt];
    undistortedImageLoc = fullfile(outputFolder, undistortedImageName);
    imwrite(undistortedImage, undistortedImageLoc);
end
% view reprojection error
h1 = figure, showReprojectionErrors(cameraparams);
% visualize pattern
h2 = figure, showExtrinsics(cameraparams,'CameraCentric');

% Display parameter estimate error
displayErrors(estimateError, cameraparams)
reprojectionError = cameraparams.ReprojectionErrors;
disp(mean(reprojectionError))






% Mosiac Panorama

%%% : Note - Change ImageFolder to LSC or Ruggles50 or Ruggles15 or
%%% Cinder_wall to get the output you desire, Also change the outputFolder
%%% below in undistorted to save the undistorted image in another folder.
%%% e.g.LSC undistorted images will have output folder as LSC/undistorted.
%%% Please change the output folder everytime before you run or undistorted
%%% images will get saved in normal folders and give wrong output
%%% change points to 2000 for cinder-wall

imageFolder = 'Ruggles15';
images = dir(fullfile(imageFolder, '*.jpg')); % Change extension as necessary
numImages = length(images);
imageFileNames = {images.name};
imageLoc = fullfile(imageFolder,imageFileNames);

% Define output folder for undistorted images
outputFolder = 'Ruggles15/undistorted';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Loop through each image and undistort it
for i = 1:numImages
    % Read image
    image = imread(imageLoc{i});
    
    % Undistort image
    undistortedImage = undistortImage(image, cameraparams);
    
    % Save undistorted image to output folder
    [~, imageName, imageExt] = fileparts(imageFileNames{i});
    undistortedImageName = [imageName, '_undistorted', imageExt];
    undistortedImageLoc = fullfile(outputFolder, undistortedImageName);
    imwrite(undistortedImage, undistortedImageLoc);
end


%%%% Note: change building dir to whichever murial you desire, you can also
%%%% use LSC/undistorted to get murial of undistortion, can repeat same for
%%%% all
buildingDir = fullfile('Ruggles15');
buildingScene = imageDatastore(buildingDir);

% Display images to be stitched
montage(buildingScene.Files)
% Read the first image
I = readimage(buildingScene,1);

img_gray = im2gray(I);

[y,x,m] = harris(img_gray,1000,'tile',[2 2],'disp');


[features, points] = extractFeatures(img_gray,[x,y]);
numImages = numel(buildingScene.Files);
tforms(numImages) = projtform2d;

% Initialize variable to hold image sizes.

imageSize = zeros(numImages,2);



% Iterate over remaining image pairs
for n = 2:numImages
    % Read I(n) and convert to grayscale
    prev_point = points;
    prev_feats = features;
    
    I = readimage(buildingScene,n);
    img_gray = im2gray(I);

    imageSize(n,:) = size(img_gray);
    [y,x,m] = harris(img_gray,1000,'tile',[2 2],'disp');

    [features, points] = extractFeatures(img_gray,[x,y]);
    indexes = matchFeatures(features, prev_feats, 'Unique', true);
    matched_Points = points(indexes(:,1), :);
    matched_Points_Prev = prev_point(indexes(:,2), :);        
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estgeotform2d(matched_Points, matched_Points_Prev,...
        'projective', 'Confidence',99.9, 'MaxNumTrials', 2000);
    
  
    tforms(n).A = tforms(n-1).A * tforms(n).A; 
end

% Compute the output limits for each transformation.
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end
avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).A = Tinv.A * tforms(i).A;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    
    I = readimage(buildingScene, i);   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)

