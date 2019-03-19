%% Gaussian Filter
clear; clc; format short;
Img = imread('image1.jpg');
% Img = imread('image2.jpg');
% Img = imread('image3.jpg');
% Img = imread('image4.jpg');
% Img = imread('image5.jpg');
% Img = imread('image6.jpg');
% Img = imread('image7.jpg');
% Img = imread('image8.jpg');
% Img = imread('image9.jpg');
% Img = imread('image10.jpg');
% Img = imread('image11.jpg');
% Img = imread('image12.jpg');

figure;
imshow(Img);

hsize = 31;
sigma = 5;
h = fspecial('gaussian',hsize,sigma);
figure;
surf(h);

figure;
imagesc(h);

outim = imfilter(Img,h);
figure;
imshow(outim);

%% Remove Noise with a Gaussian Filter
clear; clc;
Img1 = imread('image1.jpg');

% Adding noise
noise_sigma = 25;
noise = randn(size(Img1)) .* noise_sigma;
noisy_img = Img1 + noise;

% Creating a Gaussian Filter
hsize = 31;
sigma = 5;
h = fspecial('gaussian',hsize,sigma);
surf(h);
imagesc(h);
outim = imfilter(Img1,h);

display = [Img1, noisy_img, outim];
figure;
imshow(display); title('Removing Noise with a Gaussian Filter');

%% Applying a median filter
clear; clc;
Img1 = imread('image1.jpg');
Cin = rgb2gray(Img1);

% Add salt & pepper noise
noisy_img = imnoise(Cin,'salt & pepper',0.02);

% Apply a median filter
median_filtered = medfilt2(noisy_img);

% display image
result = [noisy_img,median_filtered];
imshow(result);

%% Gradient
clear; clc;
Img1 = imread('image1.jpg');
Cin = rgb2gray(Img1);

% Compute x,y gradients
[gx gy] = imgradientxy(Cin,'sobel');

% Obtain gradient magnitude and direction
[gmax gdir] = imgradient(gx,gy);
imshow((gdir + 180.0) / 360.0);

% Find pixels with desired gradient direciton
my_grad = select_gdir(gmag,gdir,1,30,60);
imshow(my_grad);


