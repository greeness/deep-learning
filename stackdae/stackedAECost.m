function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) 
%       is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));

for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful

% number of examples
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%


% Perform a feedforward pass, computing the activations for layers L1, L2

%x = data;

%nl = numel(stack);
%for d = 1:nl
%    stack{d}.z = stack{d}.w * x + repmat(stack{d}.b, 1, m);
%    stack{d}.a = sigmoid(stack{d}.z);
%    x = stack{d}.a;
%end

stack{1}.a = sigmoid(stack{1}.w * data + repmat(stack{1}.b, 1, m));
stack{2}.a = sigmoid(stack{2}.w * stack{1}.a + repmat(stack{2}.b, 1, m));

% Perfrom a feedforward pass for the output layer
M = softmaxTheta * stack{2}.a;

M = bsxfun(@minus, M, max(M, [], 1));

expM = exp(M);

% normalized class probabilities
h = expM ./ repmat(sum(expM, 1), numClasses, 1);

cost = -sum(sum(groundTruth .* log(h))) / m + ...
       0.5 * lambda * sum(sum(softmaxTheta.^2));

softmaxThetaGrad = -(stack{2}.a * (groundTruth-h)')' ./m + lambda * softmaxTheta;

softmaxDelta = - softmaxTheta' * (groundTruth - h) .* (stack{2}.a .* (1 - stack{2}.a));

L2Delta = (stack{2}.w' * softmaxDelta) .* (stack{1}.a .* (1 - stack{1}.a));

% Note should not include the decay term here. Because in the fine tuning,
% our current cost function only have lambda in the `softmaxTheta` term.
% Thus when taking derivative w.r.t W, the lambda never appears in the 
% result.
stackgrad{2}.w = softmaxDelta * stack{1}.a' ./ m; % + lambda * stack{2}.w;
stackgrad{2}.b = mean(softmaxDelta, 2);

stackgrad{1}.w = L2Delta * data' ./ m; %+ lambda * stack{1}.w;
stackgrad{1}.b = mean(L2Delta, 2);
% -------------------------------------------------------------------------
%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
