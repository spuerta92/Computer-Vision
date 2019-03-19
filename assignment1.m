%% Assignment #1
% Sebastian Puerta Hincapie - 23200457
clear; clc; format short;

% ---------------- Step 1 ------------------------
% Read in an image, get information
% type help imread for more information

InputImage = 'IDPicture.bmp'; 
%OutputImage1 = 'IDPicture_bw.bmp';

C1 = imread(InputImage);
[ROWS COLS CHANNELS] = size(C1);

% Constant variables
klevels = [2 4 16 32 64];
disp = [];

%% ---------------- Step 2 ------------------------
% If you want to display the three separate bands
% with the color image in one window, here is 
% what you need to do
% Basically you generate three "color" images
% using the three bands respectively
% and then use [] operator to concatenate the four images
% the orignal color, R band, G band and B band

% First, generate a blank image. Using "uinit8" will 
% give you an image of 8 bits for each pixel in each channel
% Since the Matlab will generate everything as double by default

% Note to self: CHANNELS = 3
clc;

% Note how to put the Red band of the color image C1 into 
% each band of the three-band grayscale image CR1
CR1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CR1(:,:,band) = (C1(:,:,1));
end

% Do the same thing for G
CG1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CG1(:,:,band) = (C1(:,:,2));
end

% and for B
CB1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CB1(:,:,band) = (C1(:,:,3));
end

% Whenever you use figure, you generate a new figure window 
No1 = figure(1);  % Figure No. 1

%This is what I mean by concatenation
disimg = [C1, CR1; CG1, CB1]; 

% Then "image" will do the display for you!
image(disimg);
title('Original + 3 B/W')

%% ---------------- Step 3 ------------------------
% Now we can calculate its intensity image from 
% the color image. Don't forget to use "uint8" to 
% convert the double results to unsigned 8-bit integers
clc;

figure(2); 
% creates false image?
I1 = uint8(round(sum(C1,3)/3));
I2 = I1;
subplot(1,3,1);
image(I1);
title('False Color');

% simple average
I1 = uint8(round((CR1 + CG1 + CB1) / 3));
I3 = I1;
subplot(1,3,2);
image(I1);
title('Simple Average');

% USING the NTSC standard for luminance
I1 = uint8(0.299*CR1+0.587*CG1+0.114*CB1);
% I1 = 0.299*CR1+0.587*CG1+0.114*CB1;
subplot(1,3,3);
image(I1);
title('Intensity using NTSC standard');

% If you just stop your program here, you will see a 
% false color image since the system needs a colormap to 
% display a 8-bit image  correctly. 
% The above display uses a default color map
% which is not correct. It is beautiful, though

%% ---------------- Step 4 ------------------------
% So we need to generate a color map for the grayscale
% I think Matlab should have a function to do this,
% but I am going to do it myself anyway.

% Colormap is a 256 entry table, each index has three entries 
% indicating the three color components of the index
clc;

MAP =zeros(256, 3);

% For a gray scale C[i] = (i, i, i)
% But Matlab use color value from 0 to 1 
% so I scale 0-255 into 0-1 (and note 
% that I do not use "unit8" for MAP

for i = 1 : 256,  % a comma means pause 
    for band = 1:CHANNELS,
        MAP(i,band) = (i-1)/255;
    end 
end

figure(4);
% Test #1
subplot(1,3,1);
image(I2);
title('False Image using colormap');

% Test #2
subplot(1,3,2);
image(I3);
title('Simple Average using colormap');

% Test #3
subplot(1,3,3);
image(I1);
title('NTSC using colormap');

% in order for colormap to work, it needs to go at the end
% colormap winter;
% colormap summer;
colormap(MAP);

figure(5);

% ================= Function Attempts ====================
% for i = 1:length(klevels)
%     K = klevels(i);
%     % Alt #1
%     Q1 = uint8(mat2gray(I1)*K-1);
%     % Alt #2
%     Q2 = uint8(floor(I1 ./ (256/K)));
%     disp = horzcat(disp,Q1);
%      nfig = nfig + i;
%     subplot(4,2,i)
%     imshow(Q2);
%     title(str + K)
% end
% figure(7);
% image(disp);
% title('K levels combined')
% ========================================================

% Implementation using imquantize
% nthresh = [2 4 8 16];
% str = "K = ";
% for i = 1:length(nthresh)
%     % Alt #3
%     K = nthresh(i);
%     thresh = multithresh(I1,K);
%     Q3 = imquantize(I1,thresh,[0 thresh]);
%     subplot(2,3,i);
%     imshow(Q3);
%     title(str + K);
% end

% Implementation using conditions
% arr_col = [];
% arr_intensity = [];
% for i = 1:length(klevels)
%     K = klevels(i);
%     col = 256 / K;    % columns
%     intensity = 256 / (K - 1);
%     p = 0; q = 0;
%     for j = 1:K
%         arr_intensity(j) = p + intensity;
%         p = intensity;
%         arr_col(j) = q + col;
%         q = col;
%     end
%     % first element
%     I1(:,1:arr_col(1),:) = uint8(arr_intensity(1));
%     for h = 2:K-1
%         if(arr_col(h) >= arr_col(1) && arr_col(h) <= arr_col(h+1))
%             I1(:,arr_col(h):arr_col(h+1),:) =  uint8(arr_intensity(h));
%         end
%     end
%     % last element
%     I1(:,arr_col(K-1):arr_col(K),:) = uint8(arr_intensity(K));
%     subplot(4,2,i);
%     imshow(I1);
% end

% Alternative Implementation
m = double(I1) / 255;  % color factor
str = "K = ";
for i = 1:length(klevels)
    K = klevels(i);
    Cr1 = uint8(m * K);
    Cr1 = double(Cr1) / K;
    subplot(2,2,i);
    imshow(Cr1);
    title(str + K);
end


%% ---------------- Step 5 ------------------------
% Use imwrite save any image
% check out image formats supported by Matlab
% by typing "help imwrite
clc;
% imwrite(I1, OutputImage1, 'BMP');
figure(6);

% ====== Function attempts ==================
% klevels = [2 4 8 16 32 64 128 256];
% disp = [];
% str = "Original 3-Band Image(C1) K = ";
% figure(8)
% for i = 1:length(klevels)
%     K = klevels(i);
%     % Alt #1
%     Q1 = uint8(mat2gray(C1)*K-1);
%     % Alt #2
%     Q2 = uint8(floor(C1 ./ (256/K)));
%     % Alt #3
%     thresh = multithresh(C1,2);
%     Q3 = imquantize(C1,thresh,[0 thresh]);
%     disp = horzcat(disp,Q1);
% %     nfig = nfig + i;
%     subplot(4,2,i)
%     imshow(Q1);
%     title(str + K)
% end
% figure(9);
% image(disp);
% title('K levels combined')
% =============================================

% Alternative Implementation
m = double(C1) / 255;  % color factor
str = "K = ";
for i = 1:length(klevels)
    K = klevels(i);
    Cr1 = uint8(m * K);
    Cr1 = double(Cr1) / K;
    subplot(1,2,i);
    imshow(Cr1);
    title(str + K);
end

%% ---------------- Step 6 ------------------------
clc;
% C = ceil(rand(1,1)*256);
% x = C1 + 1;
% Ln1 = C + log(x);
% result = uint8(Ln1);

% Log implementation 
figure(7);
i = 1;
for C = linspace(47,52,6)
    Cx = double(C1);
    result = uint8(C * log(Cx + 1));
    subplot(3,2,i);
    imshow(result);
    i = i + 1;
end

figure(8)
C = 47;
Cx = double(C1);
result = uint8(C * log(Cx + 1));
image(result);
title('Original 3-Band Image Log')

% Sebastian Puerta Hincapie - 23200457



