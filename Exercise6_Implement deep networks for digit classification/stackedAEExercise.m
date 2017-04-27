%% CS294A/CS294W Stacked Autoencoder Exercise
%  栈式自编码器
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size隐层1
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
trainData = loadMNISTImages('train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.
%  初始化第一层

%  Randomly initialize the parameters
%  随机初始化参数
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
      inputSize, hiddenSizeL1, ...
      lambda, sparsityParam, ...
      beta, trainData), ...
      sae1Theta, options);
 
% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.
%计算第一隐层的激活度参数（也就是第一层得到的特征）
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

%用上一层的输出激活度作为这一层的输入
[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
      hiddenSizeL2, hiddenSizeL1, ...
      lambda, sparsityParam, ...
      beta, sae1Features), ...
      sae2Theta, options);

% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.
%计算第二隐层的输出激活度（作为softmax的输入特征）
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);%注意这里

% 训练softmax
lambda = 1e-4;
inputSize2 = hiddenSizeL2;%输入数据维度
numClasses = 10;%输入类别数
options.maxIter = 100;

softmaxModel = softmaxTrain(inputSize2, numClasses, lambda, sae2Features, trainLabels, options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);%这个得到了分类器的参数
% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model
% 对得到的模型进行微调，这里是最复杂的部分
% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
%每层的参数都写入一个cell中
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...%w1维度为（隐层1*输入特征）
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...%w1维度为（隐层2*隐层1）
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
%stack2params是为了方便使用miFunc写的：
%1、stackparams主要是将‘栈’转换为扁化的的参数，
% ----比如原来的 w1 = [1,2,3]  b1 = [1,2]  w2 = [3,4]  b2 = [7,8]
% ----               [3,4,5]                   [5,6]  
% ----则现在stackparams为[1,3,2,4,3,5,1,2,3,5,4,6,7,8]'
%2、netconfig为网络的一些参数：netconfig.inputsize为输入特征数，netconfig.layersizes为每个隐层的节点数量
[stackparams, netconfig] = stack2params(stack);
%将分类器参数和栈式自编码的参数放在一起
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%

[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL2,...
                         numClasses, netconfig,lambda, trainData, trainLabels),...
                        stackedAETheta,options);%训练出第三层网络的参数



% -------------------------------------------------------------------------



%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
