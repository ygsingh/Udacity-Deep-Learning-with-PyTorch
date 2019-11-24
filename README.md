# Udacity-Deep-Learning-with-PyTorch
This repo contains reference examples on how to build deep learnig models using [PyTorch](https://pytorch.org/). PyTorch is an open-source Python framework form the [Facebook AI Research](https://ai.facebook.com/) team used for develpoing deep neural networks.
## Dependencies
These notebooks require PyTorch v0.4 or newer, and torchvision. The easiest way to install PyTorch and torchvision locally is by following [the instructions on the PyTorch site](https://pytorch.org/get-started/locally/). Choose the stable version, your appropriate OS and Python versions, and how you'd like to install it. You'll also need to install numpy and jupyter notebooks, the newest versions of these should work fine.
## Implementing Gradient Descent
Implementing functions that build the gradient decsent algorithm, namely:
- `sigmoid`: The sigmoid activation function.
- `output_formula`: The formula for the prediction.
- `error_formula`: The formula for the error at a point.
- `update_weight`: The function that updates the parameters with one gradient descent step.
- `train`: Iterate the gradient descent algorithm through all the data, for a number of epochs.

## [Analyzing Student Data](https://github.com/ygsingh/Udacity-Deep-Learning-with-PyTorch/blob/master/Introduction%20to%20Neural%20Networks/student-admissions/StudentAdmissions.ipynb)
Predict student admissions to graduate school at UCLA based on *GRE Scores*, *GPA Scores*, and *Class Rank*. Implementing some of the steps in the training of the neural network, namely:
- One-hot encoding the data
- Scaling the data
- Writing the backpropagation step
- Gradient decent algorithm for training the neural network

## Introduction to PyTorch
The following notebooks cover the basic introduction to PyTorch.
### Part 1 - Tensors in PyTorch
Tensors ae the main data structure of PyTorch. In this notebook, we'll see how to create tensors, how to do simple operations, and how tensors intereact with NumPy.
