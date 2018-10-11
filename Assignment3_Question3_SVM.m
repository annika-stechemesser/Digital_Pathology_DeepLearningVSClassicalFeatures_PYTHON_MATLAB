%classify vectors using the support vector machine (see python notebook)

imageName = 'WSI.jp2';
wsi=imread(imageName,'ReductionLevel',3);   % for wsi at level 3
%wsi=imread(imageName,'ReductionLevel',1); %for patch at level 1
%%
wsi = wsi(1:24900,1:9900,:);  %crop on level 3
%wsi = wsi(80000:94999,11000:25999,:); %crop on level 1

%%
Pruned=load('./tissue_mask_std.mat');   %load mask on level 3 (if necessary)
Pruned = Pruned.mask_prune2(1:24900,1:9900);
%Pruned = load ('./tissue_mask_std_level1.mat'); %load mask on level 1 (if necessary)
%Pruned = Pruned.mask_prune2;

%%
% Prepare classification 
S = size(wsi);                
num_patches_h = S(1)/150;  %calculate number of blocks in height
num_patches_w = S(2)/150;  % calculate number of blocks in width
num_patches = num_patches_w * num_patches_h;  % get total number of blocks
Mask = zeros(S(1),S(2),3);                    % intitialise output array
colors = [255 0 0; 255 128 0; 255 0 255; 0 255 0; 0 255 255; 0 0 255; 127 0 255; 255 255 255];  %initialise colors

%%
tissue_patch=0;
s = [1,1]; 
for i = 1:num_patches_h
    s(2) = 1;
    for j = 1:num_patches_w-1
        A = Pruned(s(1):s(1)+149,s(2):s(2)+149);
       if sum(A) > 0
           tissue_patch = tissue_patch+1;
       end
       s(2) = s(2) + 150;
    end
    s(1)= s(1)+150;
end

%%
cnst.featureType = {'best5'}; %'best2','best3','best4','best5','all6'}; % one or more of {'histogram_lower','histogram_higher','gabor','perceptual','f-lbp','glcmRotInv','best2','best3'};  
cnst.gaborArray = gabor(2:2:12,0:30:150); % create gabor filter bank, requires Matlab R2015b
%%

F = zeros(tissue_patch,74);
s = [1,1];
counter = 1;
for i = 1:num_patches_h
    s(2) = 1;
    for j = 1:num_patches_w-1
        A = Pruned(s(1):s(1)+149,s(2):s(2)+149);
       if sum(A) > 0
       patch = wsi(s(1):s(1)+149,s(2):s(2)+149,:);
       %extract features from the patch
       for currFeat = cnst.featureType % iterate through feature types
    
            currFeat = char(currFeat);         % convert name of feature set to char
            cnst.numFeatures = getNumFeat(currFeat); % request number of features
            F(counter,:) = computeFeatureVector(patch, currFeat, cnst.gaborArray); %get the features
            counter = counter+1;
       end
       end
       s(2) = s(2) + 150;
       disp(i);disp(j);
       disp(num_patches_h);disp(num_patches_w);
    end
    s(1)= s(1)+150;
end

%%
 %csvwrite('features_level1.csv',F) %on level 1
 csvwrite('features_level3.csv',F) %on level 3
 
 %compute the classes in Python, see notebook "Assignment3_question2"

 %%
classes_level1=csvread('classes_level1.csv'); %on level 1
classes_level3=csvread('classes_level3.csv'); %on level 3
%%
counter = 1;
s = [1,1]; 
for i = 1:num_patches_h
    s(2) = 1;
    for j = 1:num_patches_w-1
        A = Pruned(s(1):s(1)+149,s(2):s(2)+149);
       if sum(A) > 0
        pred = classes_level1(counter); %on level 1
        %pred = classes_level3(counter); %on level 3
       %disp(pred);
       c = colors(pred,:);
       disp(c);
       disp(pred);
       Mask(s(1):s(1)+149,s(2):s(2)+149,1) = c(1);
       Mask(s(1):s(1)+149,s(2):s(2)+149,2) = c(2);
       Mask(s(1):s(1)+149,s(2):s(2)+149,3) = c(3);
       counter = counter+1;
       end
       s(2) = s(2) + 150;
       disp(i);disp(j);
       disp(num_patches_h);disp(num_patches_w);
       
    end
    s(1)= s(1)+150;
end

%%
imshow(Mask)
