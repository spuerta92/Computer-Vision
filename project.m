%% Sebastian Puerta Hincapie / 23200457
% Zebra Crossing Detection 
% Computer Vision, Fall 2018, Prof. Zhigang Zhu
clear; clc;

% Trials
% Cin = imread('image1.jpg');
% Cin = imread('image2.jpg');
% Cin = imread('image3.jpg');
% Cin = imread('image4.jpg');
Cin = imread('image5.jpg');
% Img = imread('image6.jpg');
% Img = imread('image7.jpg');
% Cin = imread('image8.jpg');
% Img = imread('image9.jpg');
% Img = imread('image10.jpg');
% Img = imread('image11.jpg');
% Img = imread('image12.jpg');

scale = [400,400];
Cin = imresize(Cin,scale);

% Image parameters
[ROWS COLUMNS CHANNELS] = size(Cin);

% Image 3-band (RGB) scale
Cred = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cgreen = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cblue = uint8(zeros(ROWS,COLUMNS,CHANNELS));
for i = 1:CHANNELS
   Cred(:,:,i) = Cin(:,:,1);
   Cgreen(:,:,i) = Cin(:,:,2);
   Cblue(:,:,i) = Cin(:,:,3);
end

% B&W image
Cout = rgb2gray(Cin);

% Edge detection
BW = edge(Cout,'canny');
figure;
imshow(BW);

% Hough Transform
[H,theta,rho] = hough(BW);
figure;
imshow(imadjust(rescale(H)),[],'XData',theta,'YData',rho,...
    'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal
hold on 
colormap(gca,cool)

% Peaks
P = houghpeaks(H,50,'threshold',ceil(0.6*max(H(:))),'NHoodSize',[5 5]);

% Superimposing image
x = theta(P(:,2));
y = rho(P(:,1));
figure;
plot(x,y,'s','color','black');

% Find lines in the image using the houghlines function
lines = houghlines(BW,theta,rho,P,'FillGap',50,'MinLength',100);

% Displaying original image with the lines superimposed
figure, imshow(BW), hold on
max_len = 0;
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
    
    % Plot beginning and ends of lines
    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
    
    % Determine the endpoints of the longest line segment
    len = norm(lines(k).point1 - lines(k).point2);
    if(len > max_len)
        max_len = len;
        xy_long = xy;
    end
end

% Highlight the longest line segment
% plot(xy_long(:,1),xy_long(:,2),'LineWidth',2','Color','red');

%% Trial 2
clear; clc;

% Trials
% Cin = imread('image1.jpg');
% Cin = imread('image2.jpg');
% Cin = imread('image3.jpg');
% Cin = imread('image4.jpg');
Cin = imread('image5.jpg');
% Cin = imread('image6.jpg');
% Cin = imread('image7.jpg');
% Cin = imread('image8.jpg');
% Cin = imread('image9.jpg');
% Cin = imread('image10.jpg');
% Img = imread('image11.jpg');
% Img = imread('image12.jpg');

scale = [400,400];
Cin = imresize(Cin,scale);
figure, imshow(Cin);

% Image parameters
[ROWS COLUMNS CHANNELS] = size(Cin);

% Image 3-band (RGB) scale
Cred = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cgreen = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cblue = uint8(zeros(ROWS,COLUMNS,CHANNELS));
for i = 1:CHANNELS
   Cred(:,:,i) = Cin(:,:,1);
   Cgreen(:,:,i) = Cin(:,:,2);
   Cblue(:,:,i) = Cin(:,:,3);
end

rotI  = imrotate(Cin,0,'crop');

% B&W image
grays = rgb2gray(rotI);
grays2 = grays;

% zooming on the zebra crossing
for i = 1:ROWS
    for j = 1:COLUMNS
        if(grays2(i,j) < 190)
            grays2(i,j) = 0;
        else 
            grays2(i,j) = 255;
        end
    end
end
figure, imshow(grays2);

edges = edge(grays2, 'canny');
figure, imshow(edges);

[accum theta rho] = hough(edges);
figure, imagesc(accum, 'XData', theta, 'YData', rho), title('Hough accumulator');

% ============ 1st Attempt
% peaks = houghpeaks(accum, 100);
% hold on; plot(theta(peaks(:,2)), rho(peaks(:,1)), 'rs'); hold off;
% 
% size(peaks)
% 
% line_segs = houghlines(edges, theta, rho, peaks); 
% line_segs
% 
% figure, imshow(grays), title('Line segments');
% hold on;
% for k = 1:length(line_segs)
%     endpoints = [line_segs(k).point1; line_segs(k).point2];
%     plot(endpoints(:,1), endpoints(:,2), 'LineWidth', 2, 'Color', 'green');
% end
% hold off;

% ============== 2nd Attempt
peaks = houghpeaks(accum, 100, 'Threshold', ceil(0.40*max(accum(:))), 'NHoodSize', [5 5]);
size(peaks)

% figure, imagesc(theta,rho,accum), title('Hough accumulator');
hold on; plot(theta(peaks(:,2)), rho(peaks(:,1)), 'rs'); hold off;

line_segs = houghlines(edges, theta, rho, peaks, 'FillGap', 100, 'Minlength', 5);

figure, imshow(grays), title('Line Segments');
hold on;
for k = 1:length(line_segs)
    endpoints = [line_segs(k).point1; line_segs(k).point2];
    plot(endpoints(:,1), endpoints(:,2), 'LineWidth', 2, 'Color', 'green');
end
hold off;

%% Trial 3 <---- CURRENT MOST ACCURATE SOLUTION 
clear; clc;

% Trials
% Cin = imread('image1.jpg');
% Cin = imread('image2.jpg');
Cin = imread('image3.jpg');     % <--- final output (.8, .8999)
% Cin = imread('image4.jpg');
% Cin = imread('image5.jpg');
% Cin = imread('image5.jpeg');
% Cin = imread('image6.jpg');
% Cin = imread('image7.jpg');
% Cin = imread('image8.jpg'); 
% Cin = imread('image9.jpg');  
% Cin = imread('image10.jpg'); 
% Cin = imread('image11.jpg');
% Cin = imread('image12.jpg');

scale = [400,400];
Cin = imresize(Cin,scale);

% Image parameters
[ROWS COLUMNS CHANNELS] = size(Cin);

% Image 3-band (RGB) scale
Cred = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cgreen = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cblue = uint8(zeros(ROWS,COLUMNS,CHANNELS));
for i = 1:CHANNELS
   Cred(:,:,i) = Cin(:,:,1);
   Cgreen(:,:,i) = Cin(:,:,2);
   Cblue(:,:,i) = Cin(:,:,3);
end

rotI  = imrotate(Cin,0,'crop');
figure, imshow(rotI);

% B&W image
grays = rgb2gray(rotI);

% Gaussian filter
hsize = 31;
sigma = 5;
h = fspecial('gaussian',hsize,sigma*0.5);
I = imfilter(grays,h);
figure, imshow(I);

%Median filter
I = medfilt2(I);
figure, imshow(I);
grays2 = I;

% grays2 = grays;
% zooming on the zebra crossing
% for i = 1:ROWS
%     for j = 1:COLUMNS
%         if(grays2(i,j) < 150)
%             grays2(i,j) = 0;
%         else 
%             grays2(i,j) = 255;
%         end
%     end
% end
% figure, imshow(grays2);

edges = edge(grays2, 'sobel');
figure, imshow(edges);
BW1=edge(edges,'sobel',(graythresh(grays2)*0.3),'vertical');
BW2=edge(edges,'sobel',(graythresh(grays2)*0.3),'horizontal');

[accum theta rho] = hough(BW1);
[accum2 theta2 rho2] = hough(BW2);

% figure, imagesc(accum, 'XData', theta, 'YData', rho), title('Hough accumulator');
peaks = houghpeaks(accum, 100, 'Threshold', ceil(0.8*max(accum(:))), 'NHoodSize', [5 5]);
peaks2 = houghpeaks(accum2, 100, 'Threshold', ceil(1*max(accum(:))), 'NHoodSize', [5 5]);
size(peaks)
size(peaks2)

% figure, imagesc(theta,rho,accum), title('Hough accumulator');
hold on; 
plot(theta(peaks(:,2)), rho(peaks(:,1)), 'rs'); 
plot(theta(peaks2(:,2)), rho(peaks2(:,1)),'rs');
hold off;

line_segs = houghlines(edges, theta, rho, peaks, 'FillGap', 100, 'Minlength', 300); % 310
line_segs2 = houghlines(BW2, theta2, rho2, peaks2, 'FillGap', 150, 'Minlength',10); % 10

figure, imshow(grays), title('Line Segments');
hold on;
for k = 1:length(line_segs2)
    endpoints2 = [line_segs2(k).point1; line_segs2(k).point2];
    plot(endpoints2(:,1), endpoints2(:,2), 'LineWidth', 2, 'Color', 'green');
end
for k = 1:length(line_segs)
    endpoints = [line_segs(k).point1; line_segs(k).point2];
    plot(endpoints(:,1), endpoints(:,2), 'LineWidth', 2, 'Color', 'yellow');
    % Plot beginnings and ends of lines
    plot(endpoints(1,1),endpoints(1,2),'x','LineWidth',2,'Color','red');
    plot(endpoints(2,1),endpoints(2,2),'x','LineWidth',2,'Color','blue');
end

% taking out parallel lines based on the most frequent same slopes
% this will give parallel lines, the zebra crossing itself is equidistant 
% from each other with the same slope.
% Extracting the vertical lines, by looking at the normal of the parallel
% lines - take two lines that have the same vanishing point and the least 
% frequent slope 

% adding corners
% corners = detectHarrisFeatures(grays2);
% [features,valid_corners] = extractFeatures(grays2,corners);
% plot(valid_corners);
hold off;

% Considering localization


%% Trial 4
clear; clc;

% Trials
% Cin = imread('image1.jpg');
% Cin = imread('image2.jpg');
% Cin = imread('image3.jpg');
% Cin = imread('image4.jpg');
% Cin = imread('image5.jpg');
% Cin = imread('image6.jpg');
% Cin = imread('image7.jpg');
% Cin = imread('image8.jpg');
Cin = imread('image9.jpg');
% Cin = imread('image10.jpg');
% Img = imread('image11.jpg'
% Img = imread('image12.jp);g');

scale = [400,400];
Cin = imresize(Cin,scale);
%figure, imshow(Cin);

% Image parameters
[ROWS COLUMNS CHANNELS] = size(Cin);

% Image 3-band (RGB) scale
Cred = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cgreen = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cblue = uint8(zeros(ROWS,COLUMNS,CHANNELS));
for i = 1:CHANNELS
   Cred(:,:,i) = Cin(:,:,1);
   Cgreen(:,:,i) = Cin(:,:,2);
   Cblue(:,:,i) = Cin(:,:,3);
end

rotI  = imrotate(Cin,90,'crop');

%Convert the image to Grayscale
I=rgb2gray(rotI);
figure(1),imshow(I);title('Grayscale Image');
x= I;

%Generate Histogram
imhist(x);

%Perform Median Filtering
x = medfilt2(x);

%Convert Grayscale image to Binary
threshold=98;%assign a threshold value
x(x<threshold)=0;
x(x>=threshold)=1; 
x=logical(x);%convert image to binary compared to threshold

%Perform Median Filtering
im2=medfilt2(x);
% figure(1),imshow(im2);title('filtered image');

%Edge Detection
a=edge(im2,'sobel');
% imshow(a); title('Edge Detection');

%Horizontal Edge Detection
BW=edge(im2,'sobel',(graythresh(I)*0.3),'vertical');
% imshow(BW);

%Hough Transform
[H,theta,rho] = hough(BW);
% figure, imshow(imadjust(mat2gray(H)),[],'XData',theta,'YData',rho,...
% 'InitialMagnification','fit');
% xlabel('\theta (degrees)'), ylabel('\rho');
% axis on, axis normal, hold on;
% colormap(hot)

% Finding the Hough peaks (number of peaks is set to 10)
P = houghpeaks(H,50,'threshold',ceil(0.3*max(H(:))));
x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','black');

%Fill the gaps of Edges and set the Minimum length of a line
lines = houghlines(BW,theta,rho,P,'FillGap',120,'MinLength',100);
figure, imshow(rotI), hold on
max_len = 0;
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','red');
    % Plot beginnings and ends of lines
    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','green');
end

%% Trial 5 - Localizing Zebra Crossing
clear; clc;

% Trials
% Cin = imread('image1.jpg');
% Cin = imread('image2.jpg');
% Cin = imread('image3.jpg');
% Cin = imread('image4.jpg');
% Cin = imread('image5.jpg');
Cin = imread('image6.jpg');
% Cin = imread('image7.jpg');
% Cin = imread('image8.jpg');
% Cin = imread('image9.jpg');
% Cin = imread('image10.jpg');
% Img = imread('image11.jpg'
% Img = imread('image12.jp);g');

scale = [400,400];
Cin = imresize(Cin,scale);
%figure, imshow(Cin);

% Image parameters
[ROWS COLUMNS CHANNELS] = size(Cin);

% Image 3-band (RGB) scale
Cred = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cgreen = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cblue = uint8(zeros(ROWS,COLUMNS,CHANNELS));
for i = 1:CHANNELS
   Cred(:,:,i) = Cin(:,:,1);
   Cgreen(:,:,i) = Cin(:,:,2);
   Cblue(:,:,i) = Cin(:,:,3);
end

rotI  = imrotate(Cin,90,'crop');

%Convert the image to Grayscale
I=rgb2gray(rotI);

% Gaussian filter
hsize = 31;
sigma = 5;
h = fspecial('gaussian',hsize,sigma*1.5);
I = imfilter(I,h);

%Median filter
I = medfilt2(I);

x = edge(I,'canny');
corners = detectHarrisFeatures(x);
[features,valid_corners] = extractFeatures(x,corners);
figure; imshow(x); hold on;
plot(valid_corners);

