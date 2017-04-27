%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28;
numLabels  = 5;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('train-images.idx3-ubyte');
mnistLabels = loadMNISTLabels('train-labels.idx1-ubyte');

% Set Unlabeled Set (All Images)

% Simulate a Labeled and Unlabeled set
%find用于查找数据并返回非0元素位置（索引）
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);

%numel用于求元素总数
numTrain = round(numel(labeledSet)/2);%一半的有标签数据，分别进行训练和测试
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

unlabeledData = mnistData(:, unlabeledSet);

trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));
%得到的结果
% # examples in unlabeled set: 29404
% # examples in supervised training set: 15298
% 
% # examples in supervised testing set: 15298

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 
%第二步用无标签数据训练稀疏编码器
%  Randomly initialize the parameters
%初始化参数
theta = initializeParameters(hiddenSize, inputSize);

%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

%这里是直接调用minFunc进行最小化目标函数
opttheta = theta; 

addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
      inputSize, hiddenSize, ...
      lambda, sparsityParam, ...
      beta, unlabeledData), ...
      theta, options);

%% -----------------------------------------------------
                          
% Visualize weights
%这里对W1权重进行可视化，也就是对学到的特征进行可视化
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.
%  利用自编码器学到的网络提取训练和测试特征
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier
%用上面得到的特征训练softmax分类器
softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels
lambda = 1e-4;
inputSize = hiddenSize;
options.maxIter = 100;
numClasses = 5; %1-5共5类
%下面这个方法可能更通用，首先找出非重复元素，接着统计其数量
%numClasses = numel(unique(trainLabels));
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, trainFeatures, trainLabels, options);

%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

[pred] = softmaxPredict(softmaxModel, testFeatures);


%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
% 
