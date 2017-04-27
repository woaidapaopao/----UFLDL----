close all

%%================================================================
%% Step 0: Load data
%  We have provided the code to load data from pcaData.txt into x.
%  x is a 2 * 45 matrix, where the kth column x(:,k) corresponds to
%  the kth data point.Here we provide the code to load natural image data into x.
%  You do not need to change the code below.

x = load('pcaData.txt','-ascii');
figure(1);
scatter(x(1, :), x(2, :));
title('Raw data');


%%================================================================
%% Step 1a: Implement PCA to obtain U 
%  Implement PCA to obtain the rotation matrix U, which is the eigenbasis
%  sigma. 

% -------------------- YOUR CODE HERE -------------------- 
u = zeros(size(x, 1)); % You need to compute this
[n m] = size(x);
j = mean(x,2);
z = x-repmat(j,1,m);%预处理，均值为0
sigma = (1.0/m)*z*z';
[u s v] = svd(sigma);


% -------------------------------------------------------- 
hold on
plot([0 u(1,1)], [0 u(2,1)]);%画第一条线
plot([0 u(1,2)], [0 u(2,2)]);%第二条线
scatter(z(1, :), z(2, :));
hold off

%%================================================================
%% Step 1b: Compute xRot, the projection on to the eigenbasis
%  Now, compute xRot by projecting the data on to the basis defined
%  by U. Visualize the points by performing a scatter plot.

% -------------------- YOUR CODE HERE -------------------- 
xRot = zeros(size(z)); % You need to compute this
xRot = u'*z;%旋转后的数据


% -------------------------------------------------------- 

% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure(2);
scatter(xRot(1, :), xRot(2, :));
title('xRot');


%%================================================================
%% Step 2: Reduce the number of dimensions from 2 to 1. 
%  Compute xRot again (this time projecting to 1 dimension).
%  Then, compute xHat by projecting the xRot back onto the original axes 
%  to see the effect of dimension reduction

% -------------------- YOUR CODE HERE -------------------- 
k = 1; % Use k = 1 and project the data onto the first eigenbasis
xHat = zeros(size(x)); % You need to compute this

%方法1:把旋转后数据中k维以后的数据另为0
% xRot(2,:) = 0;
% % figure(3);可以画出来看看
% % scatter(xRat(1, :), xRat(2, :));
% % title('xRat_del');
% xHat = u* xRot;
%方法2,投影时确定所需的投影数量
xHat = u*([u(:,1),zeros(n,1)]'*z);
% -------------------------------------------------------- 
figure(3);
scatter(xHat(1, :), xHat(2, :));
title('xHat');


%%================================================================
%% Step 3: PCA Whitening
%  Complute xPCAWhite and plot the results.

epsilon = 1e-5;
% -------------------- YOUR CODE HERE -------------------- 
xPCAWhite = zeros(size(x)); % You need to compute this
%进行PCA白化，其中白化要求输入特征相关性低并且所有特征的方差相同。
%PCA已经没满足特征间不相关，只要让特征有相同的方差即可。
%其中协方差公式为Cov(X,Y)=E(XY)-E(X)E(Y)，现在只要知道自协方差（也就是方差）即可，这时有一个结论就是这个协方差矩阵的对角线恰好是特征值s，非对角线元素为0

xPCAWhite = diag(1./sqrt(diag(s)+epsilon))*xRot;%把特征向量s变为对角阵加上一定的偏置否则可能会导致上溢，再取出对角元素（列向量）
%或写为(1./sqrt(diag(S) + epsilon)) * u' * z;

% -------------------------------------------------------- 
figure(4);
scatter(xPCAWhite(1, :), xPCAWhite(2, :));
title('xPCAWhite');

%%================================================================
%% Step 3: ZCA Whitening
%  Complute xZCAWhite and plot the results.
% ZCA白化是先利用PCA进行数据旋转但是并不降维，然后对PCA进行白化并将白化后的数据旋转回原始坐标，就得到了ZCA白化
% -------------------- YOUR CODE HERE -------------------- 
xZCAWhite = zeros(size(x)); % You need to compute this
xZCAWhite = u*xPCAWhite;%或者写为u*(1./sqrt(diag(S) + epsilon)) * u' * z;

% -------------------------------------------------------- 
figure(5);
scatter(xZCAWhite(1, :), xZCAWhite(2, :));
title('xZCAWhite');

%% Congratulations! When you have reached this point, you are done!
%  You can now move onto the next PCA exercise. :)
