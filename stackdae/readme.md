# Building Deep Networks for Classification

Reading:

http://deeplearning.stanford.edu/wiki/index.php/Self-Taught_Learning_to_Deep_Networks

Ex:

http://deeplearning.stanford.edu/wiki/index.php/Exercise:_Implement_deep_networks_for_digit_classification

![](http://deeplearning.stanford.edu/wiki/images/thumb/5/5c/Stacked_Combined.png/500px-Stacked_Combined.png)

Note:

Adding regularization terms to both output layer and hidden layer in the fine tuning step, we get an accurary of 98.30% (compared with 97.81% when only adding regularization term to the output layer, and 91.97% when we don't have fine tuning at all).

```matlab
cost = -sum(sum(groundTruth .* log(h))) / m + ...
       0.5 * lambda * sum(sum(softmaxTheta.^2)) + ...
       0.5 * lambda * (sum(sum(stack{1}.w.^2)) + sum(sum(stack{2}.w.^2)));
```
