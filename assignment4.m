%% Sebastian Puerta-Hincapie / 23200457
% Computer Vision - Assignment #4
clear; clc; format short; load stereoPointPairs;
Img1 = imread('image1.bmp');
Img2 = imread('image2.bmp');

C1 = rgb2gray(Img1);
C2 = rgb2gray(Img2);

disp("Success...");

%% Question 4 - part 1  (Fundamental Matrix) - 1st attempt
clc;
% 3x3 matrix (estimated fundamental matrix)
% The RANSAC method requires that the input points are already putatively
% matched. We can, for example, use the matchFeatures function for this. 
% Using the RANSAC algorithm eliminates any outliers which may still be 
% contained within putatively matched points.
% disp("Random Sample Concensus Method");
% fRANSAC = estimateFundamentalMatrix(matchedPoints1,...
%     matchedPoints2,'Method','RANSAC',...
%     'NumTrials',2000,'DistanceThreshold',1e-4);
% 
% % using least median of squares method to find inliers
% % flMeds = 3x3, inliers = 18x1 logical array
% % disp("Least median of Squares");
% [fLMedS, inliers] = estimateFundamentalMatrix(matchedPoints1,...
%     matchedPoints2,'NumTrials',2000);
% 
% figure;
% showMatchedFeatures(C1,C2,matchedPoints1,matchedPoints2,...
%     'montage','PlotOptions',{'ro','go','y--'});
% title('Putative point matches');
% 
% % showing the inlier points
% figure;
% showMatchedFeatures(C1, C2, matchedPoints1(inliers,:),...
%     matchedPoints2(inliers,:),'montage','PlotOptions',{'ro','go','y--'});
% title('Point matches after outliers were removed');
% 
% % using normalized eight point algorithm to compute the fundamental matrix
% % Compute the fundamental matrix for input points which do not contain 
% % any outliers.
% inlierPts1 = matchedPoints1(knownInliers,:);
% inlierPts2 = matchedPoints2(knownInliers,:);
% fNorm8Point = estimateFundamentalMatrix(inlierPts1,inlierPts2,...
%     'Method','Norm8Point');
% disp("Fundamental Matrix");
% disp(fNorm8Point);

disp("Success...");
%% Question 4 - Part 2  (Featured based Matching)
clc;
% Match Features
% Finding the Corners
points1 = detectHarrisFeatures(C1);
points2 = detectHarrisFeatures(C2);

% Extracting the neighborhood Features
[features1,valid_points1] = extractFeatures(C1,points1);
[features2,valid_points2] = extractFeatures(C2,points2);

% Match the features
indexPairs = matchFeatures(features1,features2);

% Retrieve the locations of the corresponding points for each image
matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

% Visualize the corresponding points. You can see the effect of translation
% between the two images despite several erroneous matches
figure; showMatchedFeatures(C1,C2,matchedPoints1,matchedPoints2);
title("Matching Features");

% Show Match Features
indexPairs = matchFeatures(features1, features2) ;
matchedPoints1 = valid_points1(indexPairs(1:20, 1));
matchedPoints2 = valid_points2(indexPairs(1:20, 2));

figure; ax = axes;
showMatchedFeatures(C1,C2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
title(ax, 'Candidate point matches');
legend(ax, 'Matched points 1','Matched points 2');

disp("Success...");

%% Part 1 (Fundamental Matrix) - Better Version
clc;
% Concatinating Images
[ROWS, COLS, CHANNELS] = size(Img1);
display = [Img1 Img2];
image(display), title('Point Matches');

% (Control points are the correspondences (matches) used in computing the
% fundamental matrix)
% Total Number of control points
controlPts = 4; % 12

% (Test points are those used to check the accuracy of the computation)
% Total Number of test points
testPts = 4;  % 4

% Saving points matches
load pl.mat pl;
load pr.mat pr;

% interface for picking up both the control points and 
% the test points
K = 1;
hold;

while(K <= controlPts+testPts)
    % size of the rectangle to indicate point locations
    dR = 50;
    dC = 50;

    % pick up a point in the left image and display it with a rectangle....
    % if you loaded the point matches, comment the point picking up (3 lines)%%%
    [X, Y] = ginput(1);
    Cl = X(1); 
    Rl = Y(1);
    pl(K,:) = [Cl Rl 1];

    % and draw it 
    Cl= pl(K,1);  
    Rl=pl(K,2); 
    rectangle('Curvature', [0 0], 'Position', [Cl Rl dC dR]);

    % and then pick up the correspondence in the right image
    % if you loaded the point matches, comment the point picking up (three lines)%%%
    [X, Y] = ginput(1);
    Cr = X(1); Rr = Y(1);
    pr(K,:) = [Cr-COLS Rr 1];

    % draw it
    Cr=pr(K,1)+COLS; 
    Rr=pr(K,2);
    rectangle('Curvature', [0 0], 'Position', [Cr Rr dC dR]);
    plot(Cr+COLS,Rr,'r*');
    drawnow;
    
    % increment
    K = K+1;
end
disp("STEP #1 - SUCCESS")
save pr.mat pr;
save pl.mat pl;
save pr.txt pr -ASCII;
save pl.txt pl -ASCII;

% EIGHT POINTS ALGORITHM
% using matlab function
% M = 16;
% matchedPoints1 = rand(M,2);
% matchedPoints2 = rand(M,2);
% F = estimateFundamentalMatrix(matchedPoints1,matchedPoints2);
% [flMeds, inliers] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2);

% Generate the A matrix
[a, b] = size(pl);
A = [];
for i=1:a
    x1 = pl(i,1);
    y1 = pl(i,2);
    x2 = pr(i,1);
    y2 = pr(i,2);
    A(i,:) = [x1*x2 y1*x2 x2 x1*y2 y1*y2 y2 x1 y1 1];
end

% Singular Value Decomposition (SVD) of A
[Ua, Da, Va] = svd(A);

% Estimate of F
f = Va(:,9);
F = [f(1) f(2) f(3); f(4) f(5) f(6); f(7) f(8) f(9)];
tmp = F;

[Uf, Df, Vf] = svd(F);
Df2 = Df;
Df2(3,3) = 0;
F = Uf*Df2*(Vf)';
disp("STEP #2 - SUCCESS")       % check for debugging

% Draw the epipolar lines for both the controls points and the test
% points, one by one; the current one (* in left and line in right) is in
% red and the previous ones turn into blue
% I suppose that your Fundamental matrix is F, a 3x3 matrix
d = [];
for K=1:1:controlPts+testPts
    an = F*pl(K,:)';
    x = 0:COLS; 
    y = -(an(1)*x+an(3))/an(2);
    x = x+COLS;
    plot(pl(K,1),pl(K,2),'r*');
    line(x,y,'Color', 'r');
    [X, Y] = ginput(1); %% the location doesn't matter, press mouse to continue...
    plot(pl(K,1),pl(K,2),'b*');
    line(x,y,'Color', 'b');
end 
disp("STEP #3 - SUCCESS");

for K = 1:1:controlPts+testPts
    % Student work (1c): Check the accuracy of the result by measuring the 
    % distance between the estimated epipolar lines and image points not 
    % used by the matrix estimation.
    [X1, Y1] = ginput(1);
    [X2, Y2] = ginput(1);
    d(K) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
    disp("distance...")
end

% Save the corresponding points for later use... see discussions above
save pr.mat pr;
save pl.mat pl;
save F.txt F -ASCII;

% Student work (1d): Find epipoles using the EPIPOLES_LOCATION algorithm page. 157
% --------------------------------------------------------------------
[isIn1, eR] = isEpipoleInImage(F,size(Img2));
[isIn2, eL] = isEpipoleInImage(F',size(Img1));

% Plotting the epipoles
plot(eR(1,1),eR(1,2),'y*');
plot(eL(1,1),eL(1,2),'y*');

save eR.txt eR -ASCII; 
save eL.txt eR -ASCII; 
disp("STEP #4 - SUCCESS");
