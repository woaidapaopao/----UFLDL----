function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 类别数
% inputSize - the size N of the input vector输入数据的特征数
% lambda - weight decay parameter惩罚系数
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the
% input data这里label是一个行向量
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);%输入数据量

%这里的目的是产生一个稀疏矩阵并把其编程全矩阵.具体为产生一个lables*numCases大小的矩阵，
%它的第labels(i)行第i列的元素值为1，其余全为0，其中i为1到numCases
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);%类别数*data数量的矩阵，这个和groundTruth的尺寸是相同的

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%首先为了防止exp(theta*x)的值太大造成overflow，解决方法就是给所有的theta*x减去
%一个常数值（不改变结果，这个在课件里讲到了），这个值可以用xi在所有类别中theta*xi的最大值
%max(A,[],1)%每一列的最值，得到行向量等价于max(A)
%max(A,[],2)%每一行的最值，得到列向量，等价于max(A(:,:))
%max(max(A))等价于max(A(:))
%下面让theta*data减去其最大值（行向量），用bsxfun函数会比repmat快一点
%目的是防止溢出
M = bsxfun(@minus,theta*data,max(theta*data, [], 1));

M = exp(M);
P = bsxfun(@rdivide,M,sum(M));%sum(M)也是个行向量
%计算代价函数
%方法1
cost = (-1/numCases)*(groundTruth(:)'*log(P(:))) + lambda/2 * sum(theta(:).^2);
%方法2
%cost = (-1/numCases).*sum(sum(groundTruth'*log(p))) + lambda/2 *sum(sum(theta.^2))
%接着计算梯度
thetagrad = (-1/numCases)*(groundTruth - P)*data' + lambda * theta;




% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

