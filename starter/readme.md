My attempt for the [autoencoder exercise](http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder).

### Play with the dataset first (ipython nodebook and octave)

http://nbviewer.ipython.org/gist/greeness/8346181


### Check parameter initialization
http://nbviewer.ipython.org/gist/greeness/8363819

### Forward computing, Cost Function and Derivatives
http://nbviewer.ipython.org/gist/greeness/8366171

### Gradient Checking
http://nbviewer.ipython.org/gist/greeness/8367471


=====

### Mnist trainer
The same code with a bit change on the trainer enable us to do the feature learning for mnist dataset.
* Added .m file: train_mnist.m
* See original exercise [here](http://deeplearning.stanford.edu/wiki/index.php/Exercise:Vectorization)


=====

## Auto Encoder core algorithm

### Notations
* `m`: number of training examples
* `n`: feature dimension
* `x`: input features
* `W1, W2`: weights at layer 1, layer 2
* `b1, b2`: bias at layer 1, layer 2

### Forward computation
* input: `a1 = x`
* hidden layer:
  * `z2 = W1 * a1 + repmat(b1, 1, m)`
  * `a2 = sigmoid(z2)`
* output layer:
  * `z3 = W2 * a2 + repmat(b2, 1, m)`
  * `a3 = sigmoid(z3)`
  
### Back Propagation
* error terms
  * `delta3 = -(a1 - a3) .* (a3 .* (1 - a3))`
  * `delta2 = (W2' * delta3) .* (a2 .* (1 - a2))`
* gradient
  * `W2grad = delta3 * a2' ./m + lambda * W2`
  * `W1grad = delta2 * a1' ./m + lambda * W1`
  * `b2grad = mean(delta3, 2)`
  * `b1grad = mean(delta2, 2)`
