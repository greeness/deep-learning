function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% theta is a k x n matrix:
% - where k is the number of classes, in this case, 10 classes
% - and n is the number of features, in our non-debug version n = 28*28 =
% 784
% data is a n x m matrix where m is the number of training examples
% - So the j-th row of theta is the parameters for class j;
% - the i-th column of data is the features of the i-th training exmaple
M = theta * data;

M = bsxfun(@minus, M, max(M, [], 1));

expM = exp(M);

% normalized class probabilities
h = expM ./ repmat(sum(expM, 1), numClasses, 1);

cost = -sum(sum(groundTruth .* log(h))) / numCases + 0.5 * lambda * sum(sum(theta.^2));
thetagrad = (data * (groundTruth-h)')' ./ (-numCases) + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

