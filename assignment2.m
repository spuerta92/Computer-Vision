% By: Sebastian Puerta Hincapie - 23200457
% Computer Vision - Assignment #2

% Initializing
clear; clc; format short;

% Uploading Images (multiple for testing purposes)
inputImg1 = 'IDPicture.bmp';
inputImg2 = 'IDPicture2.bmp';
inputImg3 = 'marbles.bmp';
inputImg4 = 'fruits.bmp';
inputImg5 = 'tiger.bmp';
inputImg6 = 'grayscalebutterfly.bmp';

Cin = imread(inputImg5);
Cin2 = imread(inputImg2);

[ROWS,COLUMNS,CHANNELS] = size(Cin);
[ROWS2,COLUMNS2,CHANNELS2] = size(Cin2);

% Image 3-band (RGB) scale
Cred = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cgreen = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cblue = uint8(zeros(ROWS,COLUMNS,CHANNELS));
Cred2 = uint8(zeros(ROWS2,COLUMNS2,CHANNELS2));
Cgreen2 = uint8(zeros(ROWS2,COLUMNS2,CHANNELS2));
Cblue2 = uint8(zeros(ROWS2,COLUMNS2,CHANNELS2));
for i = 1:CHANNELS
   Cred(:,:,i) = Cin(:,:,1);
   Cgreen(:,:,i) = Cin(:,:,2);
   Cblue(:,:,i) = Cin(:,:,3);
   Cred2(:,:,i) = Cin2(:,:,1);
   Cgreen2(:,:,i) = Cin2(:,:,2);
   Cblue2(:,:,i) = Cin2(:,:,3);
end

% Conc. Images (for comparison)
displayAll = [Cred Cgreen Cblue];
displayAll2 = [Cred2 Cgreen2 Cblue2];
figure(1); 
subplot(2,1,1); imshow(displayAll); title("RGB Bands - Tiger");
subplot(2,1,2); imshow(displayAll2); title("RGB Bands - Prof.Zhu");

% Intensity Image (using formula)
IntImg = uint8(0.299*Cred+0.587*Cgreen+0.114*Cblue);
IntImg2 = uint8(0.299*Cred2+0.587*Cgreen2+0.114*Cblue2);

figure(2); 
subplot(2,2,1); image(Cin); title("Original - Tiger");
subplot(2,2,2); image(IntImg); title("Grayscale - Tiger");
subplot(2,2,3); image(Cin2); title("Original - Prof. Zhu");
subplot(2,2,4); image(IntImg2); title("Grayscale - Prof. Zhu");

% Equivalently in order to change to grayscale - rgb2gray can be used for a
% 2D output

%% Part 1 - Histogram, Enhancement, Equalization, Thresholding
clc;
% ============================ Histograms
bit_counts = zeros(1,255,1);
bit_counts2 = zeros(1,255,1);
bit_counts3 = zeros(1,255,1);
% (Pixel Count)
for i = 1:ROWS
    for j = 1:COLUMNS
        for k = 1:255
            if(double(IntImg(i,j,1)) == k)
                bit_counts(1,k,1) = bit_counts(1,k,1) + 1;
            end
        end
    end  
end

bit_values = transpose(1:255);
count1 = transpose(bit_counts(:,:,1));

figure(3); 
subplot(1,3,1); bar(count1); title("Histogram");
%imhist(IntImg)

% (Probability Interpretation)
total_sum = sum(count1);
probability = double(count1 ./ total_sum);
subplot(1,3,2); bar(probability); title("Probability Histogram");

% (Cumulative Density Function)
cumulative = zeros(255,1);
cumulative(1,1) = count1(1,1);
cdf = zeros(255,1);
cdf(1,1) = probability(1,1);
for i = 2:255
    cumulative(i,1) = cumulative(i-1,1) + count1(i,1);
    cdf(i,1) = cdf(i-1,1) + probability(1,1);
end
subplot(1,3,3); bar(cumulative); title("Cumulative Histogram");

% =============================== Enhancements
% Using matlab functions
% IntImg1 = IntImg;
% IntImg1x = imadjust(IntImg1);
% IntImg1y = adapthisteq(IntImg1);
% figure(7);
% montage({IntImg1x,IntImg1y},'Size',[1 2]);

% Using Linear Scaling
g = 50;
IntImgX = double(IntImg) + g; 
IntImgY = double(IntImg) - g;
figure(4);
subplot(1,2,1); imshow(uint8(IntImgX)); title("Linear Scale");
subplot(1,2,2); imshow(uint8(IntImgY)); title("Linear scale");

IntImg1 = IntImg;
Imax = 75;
Imin = 25;
K = 255;
for k = 1:CHANNELS
    for i = 1:ROWS
        for j = 1:COLUMNS
            IntImg1(i,j,k) = ((K / (Imax - Imin))*IntImg1(i,j,k)) - ((K / (Imax - Imin))*(Imin));
        end
    end
end
figure(5);
subplot(1,2,1); image(IntImg1); title("Linear Scaling");

% ============================== Thresholding
t = multithresh(IntImg);
IntImg2 = IntImg;
Imax = max(max(IntImg2(:,:,1)));
Imin = min(min(IntImg2(:,:,1)));
for k = 1:CHANNELS
    for i = 1:ROWS
        for j = 1:COLUMNS
            if(IntImg2(i,j,k) <= t)
                IntImg2(i,j,k) = Imin;
            elseif(IntImg2(i,j,k) > t)
                IntImg2(i,j,k) = Imax;
            end
        end
    end
end
subplot(1,2,2); image(IntImg2); title("Thresholding");

% ============================= Equalization
% using matlab functions
% figure(8);
% image(imnoise(IntImg));
% figure(9);
% imhist(histeq(IntImg));
% figure(10);
% image(histeq(IntImg));

IntImg3 = IntImg;
ah = uint8(zeros(ROWS,COLUMNS,CHANNELS));
n = ROWS * COLUMNS;
f = zeros(256,1);
pdf = zeros(256,1);
cdf = zeros(256,1);
cum = zeros(256,1);
out = zeros(256,1);
for k = 1:CHANNELS
    for i = 1:ROWS
        for j = 1:COLUMNS
            value = IntImg3(i,j,k);
            f(value + 1) = f(value + 1) + 1;
            pdf(value + 1) = f(value+1) / n;
        end
    end
end

total_sum = 0; L = 255;
for i = 1:size(pdf)
    total_sum = total_sum + f(i);
    cum(i) = total_sum;
    cdf(i) = cum(i) / n;
    out(i) = round(cdf(i) * L);
end
for k = 1:CHANNELS
    for i = 1:ROWS
        for j = 1:COLUMNS
            ah(i,j,k) = out(IntImg3(i,j) + 1);
        end
    end
end
figure(6);
subplot(1,2,1); image(ah); title("Equalization");
subplot(1,2,2); imhist(ah); ylim([0 3000]); xlim([1 255]); title("Eq. Histogram");
    
disp("Success - End of Program");

%% Part 2 - (1x2 + Sobel Operator)
clc;
% === First Attempt ===
% (1 x 2 operator)
% Cout = IntImg;
% t = multithresh(Cout);
% Cmax = max(max(max(Cout)));
% Cmin = min(min(min(Cout)));
% 
% for k = 1:CHANNELS
%     for i = 1:ROWS
%         for j = 1:COLUMNS-1
%            if(i == 1 || j == 1 || i == ROWS || j == COLUMNS)
%                Cout(i,j,k) = Cmax;
%            else
%                 gx = double((-1) * Cout(i,j,k) + (1) * Cout(i+1,j));
%                 gy = double((-1) * Cout(i,j,k) + (1) * Cout(i,j+1));
%                 G = sqrt((gx .^ 2) + (gy .^ 2));
%                 if(G > t)
%                     Cout(i,j,k) = Cmax;
%                 else
%                     Cout(i,j,k) = Cmin;
%                 end
%            end
%         end
%     end
% end
% figure(6); image(Cout); title("1x2 Operator");

% Sobel Operator
% Cout2 = IntImg;
% for k = 1:CHANNELS
%     for i = 1:ROWS - 2
%         for j = 1:COLUMNS - 2
%            % Vertical
%            gy = (-1) * Cout2(i,j+2,k) + (-2) * Cout2(i+1,j+2,k) + (-1) * Cout2(i+2,j+2,k) + ...
%                 (1) * Cout2(i,j,k) + (2) * Cout2(i+1,j,k) + (1) * Cout2(i+2,j,k);
%            gy = double(gy);
%            
%            % Horizontal
%            gx = (-1) * Cout2(i+2,j,k) + (-2) * Cout2(i+2,j+1,k) + (-1) * Cout2(i+2,j+2,k) + ...
%                 (1) * Cout2(i,j,k) + (2) * Cout2(i,j+1,k) + (1) * Cout2(i,j+2,k);
%            gx = double(gx);
%            
%            % Combined
%            % magnitude
%            Cout2(i,j,k) = sqrt((gx.^2)+(gy.^2));
%            % direction
%            %Gdir = atan(gx/gy);
%            % result (Polar to Cartersian) + Normalization
%            %Cout2(i,j,k) = (1/4) * (Gmag * (sin(Gdir) + cos(Gdir)));
%         end
%     end
% end
% figure(7); image(Cout2); title("Sobel Operator");
% figure(8); image(Cout2 - Cout); title("Sobel Operator - 1x2 Operator");

% ======= Using matlab functions ========
% 2x1 Operator
% Cout2 = Cout;
% A = [-1 1];
% for i = 1:CHANNELS
%     Cout2 = conv2(Cout(:,:,i),A,'same');
% end
% figure(9), image(Cout2), title('2x1 Operator');

% Sobel Operator 
% Csobel = IntImg;
% for i = 1:CHANNELS
%     Csobel(:,:,i) = edge(IntImg(:,:,i),'Sobel');
% end
% figure(9), image(Csobel), title('Sobel Operator');

% Prewitt Operator
% Cprewitt = edge(Cout,'Prewitt');
% figure(9), image(Cprewitt), title('Prewitt Operator');

% Canny Operator
% Ccanny = edge(Cout,'Canny');
% figure(9), image(Ccanny), title('Canny Operator');

% Roberts Operator
% Croberts = edge(Cout,'Roberts');
% figure(9), image(CRoberts), title('Roberts Operator');

% Zerocross Operator
% Czerocross = edge(Cout,'zerocross');
% figure(9), image(Czerocross), title('zerocross Operator');

% Log Operator
% log = edge(Cout,'log');
% figure(9), image(Clog), title('log operator');

% ==== (1 x 2 operator) Final Attempt: After blood sweat and tears === 
Cout = rgb2gray(Cin);
horizontal = [-1 1];
vertical = transpose(horizontal);
Ch = zeros(ROWS,COLUMNS);
Cv = zeros(ROWS,COLUMNS);
Ccomb = zeros(ROWS,COLUMNS);

for i = 1:ROWS - 1
    for j = 1:COLUMNS - 1
        Cv(i,j) = sum(double(Cout(i,j:j+1)).*horizontal);
        Ch(i,j) = sum(double(Cout(i:i+1,j)).*vertical);
        Cout(i,j) = sqrt(Cv(i,j).^2 + Ch(i,j).^2);
    end
end

% Prof. Zhu
Cout3 = rgb2gray(Cin2);
Ch3 = zeros(ROWS2,COLUMNS2);
Cv3 = zeros(ROWS2,COLUMNS2);
Ccomb3 = zeros(ROWS2,COLUMNS2);

for i = 1:ROWS2 - 1
    for j = 1:COLUMNS2 - 1
        Cv3(i,j) = sum(double(Cout3(i,j:j+1)).*horizontal);
        Ch3(i,j) = sum(double(Cout3(i:i+1,j)).*vertical);
        Cout3(i,j) = sqrt(Cv3(i,j).^2 + Ch3(i,j).^2);
    end
end

% (Sobel Operator) - Final Attempt
Cout2 = rgb2gray(Cin);
Cv2 = zeros(ROWS,COLUMNS);
Ch2 = zeros(ROWS,COLUMNS);
horizontal2 = [-1 -2 -1; 0 0 0; 1 2 1];
vertical2 = transpose(horizontal2);

for i = 1:ROWS - 2
    for j = 1:COLUMNS - 2
        Cv2(i,j) = sum(sum(double(Cout2(i:i+2,j:j+2)) .* vertical2));
        Ch2(i,j) = sum(sum(double(Cout2(i:i+2,j:j+2)) .* horizontal2));
        Cout2(i,j) = 0.25* sqrt(Cv2(i,j).^2 + Ch2(i,j).^2);
    end
end

% Prof. Zhu
Cout4 = rgb2gray(Cin2);
Cv4 = zeros(ROWS2,COLUMNS2);
Ch4 = zeros(ROWS2,COLUMNS2);

for i = 1:ROWS2 - 2
    for j = 1:COLUMNS2 - 2
        Cv4(i,j) = sum(sum(double(Cout4(i:i+2,j:j+2)) .* vertical2));
        Ch4(i,j) = sum(sum(double(Cout4(i:i+2,j:j+2)) .* horizontal2));
        Cout4(i,j) = 0.25 * sqrt(Cv4(i,j).^2 + Ch4(i,j).^2);
    end
end

% Output - 1x2 Gradient Maps
figure(7);
subplot(1,3,1); imshow(Ch); title('1x2 Horizontal Gradient');
subplot(1,3,2); imshow(Cv); title('1x2 Vertical Gradient');
subplot(1,3,3); imshow(abs(Cout)); title('1x2 Combined Gradient');

% Output - Sobel Gradient Maps
figure(8);
subplot(1,3,1); imshow(Ch2); title('Sobel Horizontal Gradient');
subplot(1,3,2); imshow(Cv2); title('Sobel Vertical Gradient');
subplot(1,3,3); imshow(abs(Cout2)); title('Sobel Combined Gradient');

% Comparing
figure(9);
subplot(1,2,1); imshow(abs(Cout)); title('1x2 Operator Gradient');
subplot(1,2,2); imshow(abs(Cout2)); title('Sobel Operator Gradient');

figure(10); 
subplot(1,2,1); imshow(abs(Cout2) - abs(Cout)); title('Sobel - (1x2) (Tiger)');
subplot(1,2,2); imshow(abs(Cout4) - abs(Cout3)); title('Sobel - (1x2) (Prof. Zhu)');

figure(21); 
imshow(imsubtract(Cout2,Cout));

%% Part 3 - Edge Maps
clc;
% (1x2 Gradient) Edge Map
t = multithresh(Cout);  % 49
edgemap1 = max(Cout, t);
edgemap1(edgemap1 == round(t)) = 0;
edgemap1 = uint8(edgemap1);

% Sobel Gradient Edge Map
t2 = multithresh(Cout2);    % 39
edgemap2 = max(Cout2, t2);
edgemap2(edgemap2 == round(t2)) = 0;
edgemap2 = uint8(edgemap2);

% Using Local Adaptive Thresholding
edgemap1_x = uint8(~edgemap1);
edgemap2_y = uint8(~edgemap2);
T1 = adaptthresh(edgemap1_x,.75);
T2 = adaptthresh(edgemap2_y,.75);
BW1 = imbinarize(edgemap1_x,T1);
BW2 = imbinarize(edgemap2_y,T2);

% Outputs
figure(11);
% histograms used to determine a threshold value
subplot(1,2,1); imhist(Cout); title('1x2');
subplot(1,2,2); imhist(Cout2); title('Sobel');

figure(12); 
subplot(2,2,1); imshow(~edgemap1); title("1x2 Edge Map");
subplot(2,2,2); imshow(~edgemap2); title("Sobel Edge Map");
subplot(2,2,3); imshow(BW1); title("1x2 Edge Map - Local Adaptive Thresholding");
subplot(2,2,4); imshow(BW2); title("Sobel Edge Map - Local Adaptive Thresholding");

% Sketch
Cin3 = imread(inputImg1);
Cin3 = rgb2gray(Cin3);
Cin3 = edge(Cin3,'sobel',0.05);
figure(13); imshow(Cin3); title("Sketch of Prof. Zhu");


%% Part 4
clc;
% Cout = rgb2gray(Cin);
% 
% % 1 x 2
% tic
% horizontal = [-1 1];
% vertical = transpose(horizontal);
% Ch = zeros(ROWS,COLUMNS);
% Cv = zeros(ROWS,COLUMNS);
% Ccomb = zeros(ROWS,COLUMNS);
% 
% for i = 1:ROWS - 1
%     for j = 1:COLUMNS - 1
%         Cv(i,j) = sum(double(Cout(i,j:j+1)).*horizontal);
%         Ch(i,j) = sum(double(Cout(i:i+1,j)).*vertical);
%         Cout(i,j) = sqrt(Cv(i,j).^2 + Ch(i,j).^2);
%     end
% end
% toc
% 
% % 3 x 3
% tic
% Cv2 = zeros(ROWS,COLUMNS);
% Ch2 = zeros(ROWS,COLUMNS);
% horizontal2 = [-1 -2 -1; 0 0 0; 1 2 1];
% vertical2 = transpose(horizontal2);
% 
% for i = 1:ROWS - 2
%     for j = 1:COLUMNS - 2
%         Cv2(i,j) = sum(sum(double(Cout2(i:i+2,j:j+2)) .* vertical2));
%         Ch2(i,j) = sum(sum(double(Cout2(i:i+2,j:j+2)) .* horizontal2));
%         Cout2(i,j) = 0.25* sqrt(Cv2(i,j).^2 + Ch2(i,j).^2);
%     end
% end
% toc

% 5 x 5

% 7 x 7

% 1 x 2
N = 2000;
disp("1x2");
tic
for i = 1:N
    filtered1x2Image = medfilt2(Cout, [1, 2]);
end
toc

% 3 x 3
disp("3x3");
tic
for i = 1:N
    filtered3x3Image = medfilt2(Cout, [3, 3]);
end
toc

% 5 x 5
disp("5x5");
tic
for i = 1:N
    filtered5x5Image = medfilt2(Cout, [5, 5]);
end
toc

% 7 x 7
disp("7x7");
tic
for i = 1:N
    filtered7x7Image = medfilt2(Cout, [7, 7]);
end
toc
figure(20);
montage({filtered1x2Image,filtered3x3Image,filtered5x5Image,filtered7x7Image},'size',[1 4]);

% imnoise: gaussian, salt&pepper, speckle, poisson, localvar

%% Part 5
clc;
% Sobel Bands - Edge Maps
% [Rsobel,Rthresh] = edge(Cred(:,:,1),'sobel');
% [Gsobel,Gthresh] = edge(Cgreen(:,:,1),'sobel');
% [Bsobel,Bthresh] = edge(Cblue(:,:,1), 'sobel');
% [Rsobel2,Rthresh2] = edge(Cred2(:,:,1),'sobel');
% [Gsobel2,Gthresh2] = edge(Cgreen2(:,:,1),'sobel');
% [Bsobel2,Bthresh2] = edge(Cblue2(:,:,1), 'sobel');

% Tiger RGB Edge Map
Rsobel = edge_map(Cred(:,:,1),50);
Gsobel = edge_map(Cgreen(:,:,1),50);
Bsobel = edge_map(Cblue(:,:,1),50);

% Prof. RGB Edge Map
Rsobel2 = edge_map(Cred2(:,:,1),15);
Gsobel2 = edge_map(Cgreen2(:,:,1),15);
Bsobel2 = edge_map(Cblue2(:,:,1),15);

Combined1(:,:,1) = Rsobel;
Combined1(:,:,2) = Gsobel;
Combined1(:,:,3) = Bsobel;
Combined2(:,:,1) = Rsobel2;
Combined2(:,:,2) = Gsobel2;
Combined2(:,:,3) = Bsobel2;

% outputs
figure(13);
subplot(2,4,1); imshow(Cin); title('Original');
subplot(2,4,2); imshow(Cred); title('Red Band');
subplot(2,4,3); imshow(Cgreen); title('Green Band');
subplot(2,4,4); imshow(Cblue); title('Blue Band');
subplot(2,4,5); imshow(Rsobel); title('Red Sobel');
subplot(2,4,6); imshow(Gsobel); title('Green Sobel');
subplot(2,4,7); imshow(Bsobel); title('Blue Sobel');

figure(14);
subplot(2,4,1); imshow(Cin2); title('Original');
subplot(2,4,2); imshow(Cred2); title('Red Band');
subplot(2,4,3); imshow(Cgreen2); title('Green Band');
subplot(2,4,4); imshow(Cblue2); title('Blue Band');
subplot(2,4,5); imshow(Rsobel2); title('Red Sobel');
subplot(2,4,6); imshow(Gsobel2); title('Green Sobel');
subplot(2,4,7); imshow(Bsobel2); title('Blue Sobel');

figure(15);
subplot(1,2,1); image(~Combined2); title('Color Edge Detector');
subplot(1,2,2); image(~Combined1); title('Color Edge Detector');



