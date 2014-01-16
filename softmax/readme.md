Readings before you try the exercise:
http://deeplearning.stanford.edu/wiki/index.php/Softmax_Regression

Exercise for softmax regression.
http://deeplearning.stanford.edu/wiki/index.php/Exercise:Softmax_Regression

You might want to create the dataset in a folder named "mnist" under this directory.
In the minist folder, you need to download the four dataset files from [MNIST](http://yann.lecun.com/exdb/mnist/):
* train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
* train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
* t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
* t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

Note:

We add the path to `minFunc` in `softmaxExercise.m`:

```matlab
addpath('/Users/greeness/dataset/ufldl/minFunc');```

You might want to modify the line if your `minFunc` is located elsewhere.
