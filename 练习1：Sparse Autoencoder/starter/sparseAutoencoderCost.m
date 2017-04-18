function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 输入尺寸
% hiddenSize: the number of hidden units (probably 25) 隐单元数
% lambda: weight decay parameter权重惩罚
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks
%                           like a lower-case "p").期望的稀疏参数
% beta: weight of sparsity penalty term稀疏项惩罚因子
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be
% a vector). 参数向量，为了后面进行lbfgs
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);%每列对应一个隐单元
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);%一个列向量
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
%1、前向传播计算活跃度
[n,m] = size(data);%m为样本数，n为特征数

z2 = W1*data + repmat(b1,1,m);%b为列向量，向量变矩阵相加
a2 = sigmoid(z2);
z3 = W2*a2 + repmat(b2,1,m);%b为列向量，向量变矩阵相加
a3 = sigmoid(z3);
%2、计算误差
Jcost = (0.5/m)*sum(sum((a3-data).^2));%均方误差
%计算权值惩罚
Jweight = (0.5*lambda)*(sum((sum(W1.^2))+ sum(sum(W2.^2))));
%计算隐藏单元平均活跃度
roh = (1/m) .* sum(a2,2);%按行求和得到每个隐藏单元的平均活跃度
Jsparse = sum(sparsityParam.*log(sparsityParam./roh) + ...
    (1 - sparsityParam).*log((1 - sparsityParam)./(1 - roh)));
%总的损失函数
cost = Jcost+lambda*Jweight+beta*Jsparse;%这个主要是为了表达出损失函数的构成，但并不是直接对这一损失函数求导求梯度，而是用了一些技巧

%3、反向传播求误差
delta3 = -(data - a3).*sigmoidInv(z3);%delta3为n*m，n = visibleSize
%计算新的后向传播第二层残差
sparsity_penalty = beta*(-sparsityParam./roh+(1-sparsityParam)./(1-roh));
delta2 = (W2'*delta3 + repmat(sparsity_penalty,1,m)).*sigmoidInv(z2);%delta2为hiddenSize*m，技巧1、利用自编码算法与稀疏性一节中
                                                                     %所述求相对熵导数的实现技巧对delta2进行更新，这个需要记录保存前面的roh（平均活跃度）

%4、反向计算更新权重
W1grad = (delta2*data')./m + lambda.*W1;%W1grad尺寸为hiddenSize*visibleSize，技巧2、加入惩罚系数这个与线性回归类似
W2grad = (delta3*a2')./m + lambda.*W2;
b1grad = (sum(delta2,2))./m;
b2grad = (sum(delta3,2))./m;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end

