# Udacity-Deep-Learning-with-PyTorch
This repo contains reference examples on how to build deep learnig models using [PyTorch](https://pytorch.org/). PyTorch is an open-source Python framework form the [Facebook AI Research](https://ai.facebook.com/) team used for develpoing deep neural networks.
## Dependencies
These notebooks require PyTorch v0.4 or newer, and torchvision. The easiest way to install PyTorch and torchvision locally is by following [the instructions on the PyTorch site](https://pytorch.org/get-started/locally/). Choose the stable version, your appropriate OS and Python versions, and how you'd like to install it. You'll also need to install numpy and jupyter notebooks, the newest versions of these should work fine.
## [Implementing Gradient Descent](Introduction_to_Neural_Networks/gradient-descent/GradientDescent.ipynb)
Implementing functions that build the gradient decsent algorithm, namely:
- `sigmoid`: The sigmoid activation function.
- `output_formula`: The formula for the prediction.
- `error_formula`: The formula for the error at a point.
- `update_weight`: The function that updates the parameters with one gradient descent step.
- `train`: Iterate the gradient descent algorithm through all the data, for a number of epochs.

## [Analyzing Student Data](Introduction_to_Neural_Networks/student-admissions/StudentAdmissions.ipynb)
Predict student admissions to graduate school at UCLA based on *GRE Scores*, *GPA Scores*, and *Class Rank*. Implementing some of the steps in the training of the neural network, namely:
- One-hot encoding the data
- Scaling the data
- Writing the backpropagation step
- Gradient decent algorithm for training the neural network

## Introduction to PyTorch
The following notebooks cover the basic introduction to PyTorch.
### [Part 1 - Tensors in PyTorch](Introduction_to_PyTorch/Part&#32;1&#32;-&#32;Tensors&#32;in&#32;PyTorch.ipynb)
Tensors are the main data structure of PyTorch. In this notebook, we'll see how to create tensors, how to do simple operations, and how tensors interact with NumPy.
### [Part 2 - Neural Networks in PyTorch](Introduction_to_PyTorch/Part&#32;2&#32;-&#32;Neural&#32;Networks&#32;in&#32;PyTorch.ipynb)
PyTorch has a nice module `nn` that provides a nice way to efficiently build large neural networks. In this notebook, we'll see how to build a neural network with 784 inputs, 256 hidden units, 10 output units and a softmax output. In this example we are building a neural network to identify text in an image. We'll use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset which consists of greyscale handwritten digits. Each image is 28$\times$28 pixels. Here's an example how the data looks:
<img src = Introduction_to_PyTorch/assets/mnist.png>
### [Part 3 - Training Neural Networks](Introduction_to_PyTorch/Part&#32;3&#32;-&#32;Training&#32;Neural&#32;Networks.ipynb)
In this notebook, we'll learn about a module called **autograd** that PyTorch uses to calculate gradients for training neural networks. Next we'll use PyTorch to build a network and run data forward through it. Then we'll define a loss and an optimization method to train the neural network on a dataset of handwritten digits.
### [Part 4 - Fashion-MNIST](Introduction_to_PyTorch/Part&#32;4&#32;-&#32;Fashion-MNIST.ipynb)
This notebook is a practice example for Part 1 - 3. We'll be building and training a neural network for [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Fashion-MNIST is a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. Here's an example of how the data looks:
<img src = Introduction_to_PyTorch/assets/fashion-mnist-sprite.png>
### [Part 5 - Inference and Validation](Introduction_to_PyTorch/Part&#32;5&#32;-&#32;Inference&#32;and&#32;Validation.ipynb)
In this notebook, we measure the performance of the trained model on the validation data. During training, we avoid overfitting through regularization such as **dropout**. We'll see how to perform inference and validation on Fashion-MNIST data in PyTorch.
### [Part 6 - Saving and Loading Models](Introduction_to_PyTorch/Part&#32;6&#32;-&#32;Saving&#32;and&#32;Loading&#32;Models.ipynb)
Neural network models in PyTorch can be saved by using `torch.save` and loaded by using `torch.load` functions. In this notebook, we'll see an example on how to save and load a model in PyTorch.
### [Part 7 - Loading Image Data](Introduction_to_PyTorch/Part&#32;7&#32;-&#32;Loading&#32;Image&#32;Data.ipynb)
In this notebook, we'll be using a [dataset of cat and dog photos](https://www.kaggle.com/c/dogs-vs-cats) available from Kaggle. Once the dataset is loaded we can perform transformations such as `transforms.Resize()` or `transforms.CenterCrop()` or `transforms.RandomResizedCrop()`, etc so that all the images be the same size for training. Now, this image folder can be passed through `DataLoader` so that it can return batches of images and the corresponding labels.
