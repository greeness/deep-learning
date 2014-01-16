function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix

numCases = size(data, 2);
pred = zeros(1, numCases);

numClasses = size(theta, 1);


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

M = theta * data;

M = bsxfun(@minus, M, max(M, [], 1));

expM = exp(M);

% normalized class probabilities
h = expM ./ repmat(sum(expM, 1), numClasses, 1);

[predValue, pred] = max(h);
% ---------------------------------------------------------------------

end

