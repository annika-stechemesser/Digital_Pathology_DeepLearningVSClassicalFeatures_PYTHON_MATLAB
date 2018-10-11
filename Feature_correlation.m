F1=csvread('features_level1.csv');
F2=csvread('features_level3.csv');
%%
R = corrcoef(F1);
S = corrcoef(F2);

%%

colormap('spring')
imagesc(R)
title('Feature correlation, level 1')
colorbar

%%

colormap('spring')
imagesc(S)
title('Feature correlation, level 3')
colorbar