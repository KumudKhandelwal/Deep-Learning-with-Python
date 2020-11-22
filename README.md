# Deep-Learning-with-Python
Algorithms, concepts, and logics based on PyTorch and Tensorflow


A summary of which lab covers what topic:

**DSTI_DL_Lab1**
---------
1. Introduction to tensor and its implementation
2. Conversion between numpy and tensors
3. Operations we can perform using tensors
4. 3D tensors
5. Broadcasting semantics
6. Computational graphs
7. An example of Linear Regression using PyTorch

**DSTI_DL_Lab2**
-----
Autograd: Automatic differentiation 
  1. Visualizing the computational graph
  2. Computing the gradient on need
  3. Another example of tensor
  4. Gradient accumulation

Playing with a neural network in pytorch: non-linear regression
  1. Non-linear regression with a neural network
  2. Neural network with full implementation (gradient computation and estimate update)
  3. Neural network with automatic gradient computation (but still the explicit estimate update)
  4. Neural network with automatic gradient computation and automatic estimate update

**DSTI_DL_Lab3**
-----
Training a classifier
How to use Dataset and DataLoader
  1. Data Loading
  2. Data Loader
Training an image classifier
  1. Load and normalizing the CIFAR10 training and test datasets using torchvision 
  2. Define a Convolutional Neural Network
  3. Define a loss function  
  4. Train the network on the training data
  5. Test the network on the test data
 
**DSTI_DL_Lab4**
-----
Neural Network on GPU (MNIST Classifier problem)
How to use Dataset and DataLoader
  1. Loading the data
  2. Making the train and test dataset iterable
  3. Defining a model class to build Neural Network
  4. Instantiation of the model class, the loss function (here, crossentropy), and the optimizer function to use (SDG, Adam, RMSprop, Adadelta, etc.)
  5. Training of the model for accuracy
