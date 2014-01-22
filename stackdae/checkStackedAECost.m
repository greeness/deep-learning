function [] = checkStackedAECost()

% Check the gradients for the stacked autoencoder
%
% In general, we recommend that the creation of such files for checking
% gradients when you write new cost functions.
%

%% Setup random data / small model
inputSize = 4;
hiddenSize = 5;
lambda = 0.01;
data   = randn(inputSize, 5);
labels = [ 1 2 1 2 1 ];
numClasses = 2;

stack = cell(2,1);
stack{1}.w = 0.1 * randn(3, inputSize);
stack{1}.b = zeros(3, 1);
stack{2}.w = 0.1 * randn(hiddenSize, 3);
stack{2}.b = zeros(hiddenSize, 1);
softmaxTheta = 0.005 * randn(hiddenSize * numClasses, 1);

[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ softmaxTheta ; stackparams ];


[cost, grad] = stackedAECost(stackedAETheta, inputSize, hiddenSize, ...
                             numClasses, netconfig, ...
                             lambda, data, labels);

% Check that the numerical and analytic gradients are the same
numgrad = computeNumericalGradient( @(x) stackedAECost(x, inputSize, ...
                                        hiddenSize, numClasses, netconfig, ...
                                        lambda, data, labels), ...
                                        stackedAETheta);

                                    
% the first 10 is for softmaxTheta
%numgrad = numgrad(1:10);
%grad = grad(1:10);
% the subsequent parameters are [w1; b1; w2; b2]
% w1 is 3 x 4
% b1 is 3 x 1
% w2 is 5 x 3
% b2 is 5 x 1

% w1
%numgrad = numgrad(11:22);
%grad = grad(11:22);

% b1 is correct
%numgrad = numgrad(23:25);
%grad = grad(23:25);

%numgrad = numgrad(26:40);
%grad = grad(26:40);

% b2 is correct
%numgrad = numgrad(41:45);
%grad = grad(41:45);

%size(numgrad), size(grad)
% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
disp('Norm between numerical and analytical gradient (should be less than 1e-9)');
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 
            
            
