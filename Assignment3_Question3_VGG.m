% preparation: build the classifiers
classnames = {'0','1','2','3','4','5','6','7'};
model = importKerasNetwork('./notebooks/vgg_fine_tuned.h5','Classnames',classnames); %load the tuned VGG from Python

%%
imageName = 'WSI.jp2';
%wsi=imread(imageName,'ReductionLevel',3);   % for wsi at level 3
wsi=imread(imageName,'ReductionLevel',1); %for patch at level 1
%%
%wsi = wsi(1:24900,1:9900,:);  %crop on level 3
wsi = wsi(80000:94999,11000:25999,:); %crop on level 1
%%
imshow(wsi)  %display the image (optional)
%%
% compute the binary mask (background/no background)

f=@stdfilt; %build a function handle, use local standard deviation filter

[mask, features] = Segmentation(wsi,f); %run the function Segmentation (see below)

% pruning out regions that are smaller than a threshold
mask_prune = imopen(mask, strel('disk',25));         %morphologically open the image, structuring element: disc with radius 25
mask_prune2 = imclose(mask_prune, strel('disk',25)); %morphologically close the image, structuring element: disc with radius 25

B = bwboundaries(mask_prune2/255,'noholes');
disp('Done stdfilt')
%save('./tissue_mask_std.mat','mask_prune2')         % save mask level 3     
save('./tissue_mask_std_level1.mat','mask_prune2'); %save mask level 1
%%

%Pruned=load('./tissue_mask_std.mat');   %load mask on level 3 (if necessary)
%Pruned = Pruned.mask_prune2(1:24900,1:9900);
Pruned = load ('./tissue_mask_std_level1.mat'); %load mask on level 1 (if necessary)
Pruned = Pruned.mask_prune2;

%%

imshow(Pruned)  %show the mask 

%%
% Prepare classification 
S = size(wsi);                
num_patches_h = S(1)/150;  %calculate number of blocks in height
num_patches_w = S(2)/150;  % calculate number of blocks in width
num_patches = num_patches_w * num_patches_h;  % get total number of blocks
Mask = zeros(S(1),S(2),3);                    % intitialise output array
colors = [255 0 0; 255 128 0; 255 0 255; 0 255 0; 0 255 255; 0 0 255; 127 0 255; 255 255 255];  %initialise colors
%%

% classification using deep learning 
s = [1,1];                                    % set starting point

for i = 1:num_patches_h              %loop through patches vertically
    s(2) = 1;                   
    for j = 1:num_patches_w-1        %loop through patches horizontally
       A = Pruned(s(1):s(1)+149,s(2):s(2)+149);  %get the corresponding patch in the binary mask
       if sum(A) > 0                             % check if the patch contains tissue
       patch = wsi(s(1):s(1)+149,s(2):s(2)+149,1:3);  %get the patch on the slide
       pred = classify(model,patch);                  % classify
       pred=cellstr(pred);                           % transform from categorical to cell
       pred = str2double(pred);                      % transform from cell to double
       c = colors(pred+1,:);                           %get the color
       disp(c);
       Mask(s(1):s(1)+149,s(2):s(2)+149,1) = c(1);   %set the color
       Mask(s(1):s(1)+149,s(2):s(2)+149,2) = c(2);
       Mask(s(1):s(1)+149,s(2):s(2)+149,3) = c(3);
       end
       s(2) = s(2) + 150;               %move to the next block
       disp(i);disp(j);                % display progress
       disp(num_patches_h);disp(num_patches_w);
       
    end
    s(1)= s(1)+150;   %move to the next column
end


%%
imshow(Mask) %show

%%
%save('./classification_VGG_matlab.mat','Mask') %save the mask
save('./classification_VGG_matlab_level1.mat','Mask') %save the mask

%%
function [mask, features] = Segmentation(im,f) %takes the image and a function handle

    % Convert image into grayscale image
    grayImage = rgb2gray(im);
    
    neighbourhoood = ones(15,15); % specify local neighbourhood. Can't be too big or too small.
    features = f(grayImage,neighbourhoood);
    % f is a function handle to the filter (entrpoyfilter), compute the local
    %entropy of the grayscale image
    
    features = features/max(max(features)); %normalize
    threshold = graythresh(features);   %use otsu thresholding to set the threshold
 
    
    % Generate mask using threshold values
    mask = features;
    mask(mask>threshold) = 255;  %% if local entropy greater than threshold, put white (tissue)
    mask(mask<=threshold) = 0;   % if local entropy smaller than threshold, put black (no tissue)
    
end
