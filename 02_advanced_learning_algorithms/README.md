<h1>02 - Advanced Learning Algorithms</h1>

https://www.coursera.org/learn/advanced-learning-algorithms/home/info

Contents
- [Week 1: Neural networks](#week-1-neural-networks)
  - [Neural Networks Intuition](#neural-networks-intuition)
  - [From Logistic Regression to Neural Networks](#from-logistic-regression-to-neural-networks)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Neural Network notation](#neural-network-notation)
  - [Hand digit recognition example](#hand-digit-recognition-example)
  - [TensorFlow implementation of a neural network](#tensorflow-implementation-of-a-neural-network)
  - [Week 1: Labs](#week-1-labs)
- [Week 2: Neural network training](#week-2-neural-network-training)
  - [Neural network training](#neural-network-training)
  - [Activation functions](#activation-functions)
  - [Choosing activation functions](#choosing-activation-functions)
  - [Multiclass classification](#multiclass-classification)
  - [Softmax function](#softmax-function)
  - [Week 2: Labs](#week-2-labs)
- [Week 3: Advice for applying machine learning](#week-3-advice-for-applying-machine-learning)
  - [Notes](#notes)
  - [Week 3: Labs](#week-3-labs)
- [Week 4: Decision trees](#week-4-decision-trees)
  - [Notes](#notes-1)
  - [Week 4: Labs](#week-4-labs)

<hr>

## Week 1: Neural networks
>This week, you'll learn about neural networks and how to use them for classification tasks. You'll use the TensorFlow framework to build a neural network with just a few lines of code. Then, dive deeper by learning how to code up your own neural network in Python, "from scratch". Optionally, you can learn more about how neural network computations are implemented efficiently using parallel processing (vectorization).

__Learning Objectives__
* Get familiar with the diagram and components of a neural network
* Understand the concept of a "layer" in a neural network
* Understand how neural networks learn new features.
* Understand how activations are calculated at each layer.
* Learn how a neural network can perform classification on an image.
* Use a framework, TensorFlow, to build a neural network for classification of an image.
* Learn how data goes into and out of a neural network layer in TensorFlow
* Build a neural network in regular Python code (from scratch) to make predictions.
* (Optional): Learn how neural networks use parallel processing (vectorization) to make computations faster.

### Neural Networks Intuition
* `Neural Networks` are a type of machine learning model inspired by the structure and function of the human brain. They consist of layers of interconnected nodes, called neurons, that process and transmit information.  
  * Each neuron receives input from other neurons, applies a mathematical function to the input, and produces an output that is passed on to other neurons in the network. 
  * Neural networks can learn complex patterns in data and are commonly used for tasks such as image recognition, natural language processing, and speech recognition. 
  * The architecture of a neural network can vary depending on the specific task and the amount of data available, but they typically consist of an input layer, one or more hidden layers, and an output layer.


* Bioligical neuron vs simplified artificial neural network
<img width="1594" alt="Image" src="https://github.com/user-attachments/assets/c1dc2678-3a20-4109-8880-26fae3ea86f1" />
&nbsp;

### From Logistic Regression to Neural Networks

* Logistic regression can be seen as a simple neural network with no hidden layers and a sigmoid activation function. By adding hidden layers and using different activation functions, we can create more complex neural networks that can learn more complex patterns in the data.  

* Only 1 feature  
<img width="1562" alt="Image" src="https://github.com/user-attachments/assets/f857dd53-a352-403d-af89-57cb166b68f6" />
&nbsp;

* Layers in a neural network
  * __Input layer__: the layer that receives the input data.
  * __Hidden layers__: the layers that perform computations and learn features from the input data.
  * __Output layer__: the layer that produces the final output of the neural network, such as a prediction or classification.   
* Example with 3 layers (input, hidden, output) and 4 features. 
<img width="1550" alt="Image" src="https://github.com/user-attachments/assets/33586680-8e40-4683-a048-61b205200b0e" />
&nbsp;

### Neural Network Architecture
* The architecture of a neural network refers to the number of layers and the number of neurons in each layer. The architecture can be designed based on the complexity of the problem and the amount of data available. A common architecture for image classification tasks is a __convolutional neural network__ (CNN), which consists of convolutional layers, pooling layers, and fully connected layers. The choice of architecture can have a significant impact on the performance of the neural network.  
<img width="1554" alt="Image" src="https://github.com/user-attachments/assets/560363ae-887d-4be1-8f58-3a61ae65040a" />
&nbsp;

### Neural Network notation
* $a^{[l]}$: activation of layer $l$
* $w^{[l]}$: weights of layer $l$
* $b^{[l]}$: bias of layer $l$
* $z^{[l]}$: linear transformation of layer $l$, calculated as $z^{[l]} = w^{[l]} a^{[l-1]} + b^{[l]}$
* $f^{[l]}$: activation function of layer $l$, applied to $z^{[l]}$ to get $a^{[l]} = f^{[l]}(z^{[l]})$  
* Example of a 4-layer neural network with input layer, hidden layer, and output layer:
<img width="1980" alt="Image" src="https://github.com/user-attachments/assets/226441a7-e685-484e-92c5-25390e360df5" />

### Hand digit recognition example
* The hand digit recognition example is a common application of neural networks, where the goal is to classify images of handwritten digits (0-9) into their respective categories. The input to the neural network is a 28x28 pixel image of a handwritten digit, which is flattened into a 784-dimensional vector. The neural network consists of an input layer with 784 neurons, one or more hidden layers with a certain number of neurons, and an output layer with 10 neurons (one for each digit). The output of the neural network is a probability distribution over the 10 possible classes, and the predicted class is the one with the highest probability. This example is often used as a benchmark for evaluating the performance of different neural network architectures and training algorithms.
* __Forward propagation__: the process of calculating the activations of each layer in the neural network, starting from the input layer and moving forward through the hidden layers to the output layer. This involves applying the weights and biases of each layer to the input data and applying the activation function to produce the output of each layer. The final output of the neural network is obtained after forward propagation through all layers.
* Example of a neural network architecture for hand digit recognition:
<img width="1964" height="982" alt="Image" src="https://github.com/user-attachments/assets/f24b484e-1664-4351-a4d7-3d780fd28275" />

### TensorFlow implementation of a neural network
* TensorFlow is a popular open-source machine learning framework that provides a high-level API for building and training neural networks. With TensorFlow, you can easily define the architecture of your neural network, specify the loss function and optimization algorithm, and train your model on a dataset. TensorFlow also provides tools for visualizing the training process and evaluating the performance of your model. By using TensorFlow, you can quickly build and experiment with different neural network architectures to find the best model for your specific task.
* Example of a simple neural network implementation in TensorFlow for hand digit recognition:
```python
import tensorflow as tf
from tensorflow import keras  
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
```
<img width="1942" alt="Image" src="https://github.com/user-attachments/assets/cd98ee0a-2c01-4e95-b33a-2bfd55ba7c9b" />


### Week 1: Labs
* Lab 01: [Neurons and Layers, introduction to TensorFlow and Keras](01_week/C2_W1_Lab01_Neurons_and_Layers.ipynb)
* Lab 02: [Simple Neural Network with TensorFlow, coffee roasting example](01_week/C2_W1_Lab02_CoffeeRoasting_TF.ipynb)
* Lab 03: [Simple Neural Network with Numpy, coffee roasting example](01_week/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb)

## Week 2: Neural network training
>This week, you'll learn how to train your model in `TensorFlow`, and also learn about other important `activation functions` (besides the sigmoid function), and where to use each type in a neural network. You'll also learn how to go beyond binary classification to `multiclass classification` (3 or more categories). Multiclass classification will introduce you to a new activation function and a new loss function. Optionally, you can also learn about the difference between multiclass classification and multi-label classification. You'll learn about the `Adam optimizer`, and why it's an improvement upon regular gradient descent for neural network training. Finally, you will get a brief introduction to other layer types besides the one you've seen thus far.

__Learning Objectives__
* Train a neural network on data using TensorFlow
* Understand the difference between various activation functions (sigmoid, ReLU, and linear)
* Understand which activation functions to use for which type of layer
* Understand why we need non-linear activation functions
* Understand multiclass classification
* Calculate the softmax activation for implementing multiclass classification
* Use the categorical cross entropy loss function for multiclass classification
* Use the recommended method for implementing multiclass classification in code
* (Optional): Explain the difference between multi-label and multiclass classification

### Neural network training
* Model training steps:
  1. Create the model
  2. Loss and cost functions
  3. Gradient Descent

<img width="2392" alt="Image" src="https://github.com/user-attachments/assets/e48f7e50-2a33-422c-aa9a-89e475877bc6" />

### Activation functions

* Linear Activation Function
* Sigmoid
* **ReLU: Rectified Linear Unit** $a = \max(0,z)$
<img width="1000" src="https://github.com/user-attachments/assets/243a44bc-2707-4b56-ba0a-3dc8e1bf58d0" />

* Non-linear activation functions are necessary in neural networks because they allow the network to learn and model complex relationships between the input and output data. Without non-linear activation functions, a neural network would only be able to learn linear relationships, which would limit its ability to solve complex problems. Non-linear activation functions introduce non-linearity into the network, enabling it to capture intricate patterns and make more accurate predictions. 
  
* Common non-linear activation functions include __ReLU__ (Rectified Linear Unit), __sigmoid__, and __tanh__.

<img width="2392" alt="Image" src="https://github.com/user-attachments/assets/2b9be96d-2c80-4cfa-a1eb-1ef0060daf5b" />

### Choosing activation functions
* The choice of activation function depends on the specific problem and the architecture of the neural network. 
* For __hidden layers__, __ReLU__ is often a good choice because it is computationally efficient and helps to mitigate the vanishing gradient problem. 
* For __output layers__, the choice of activation function depends on the type of problem being solved. 
  * For binary classification problems, sigmoid is commonly used because it outputs a probability between 0 and 1. 
  * For regression problems, a linear activation function may be appropriate. 
  * For multiclass classification problems, softmax is often used because it outputs a probability distribution over multiple classes.
  * Ultimately, the choice of activation function should be guided by experimentation and evaluation of the model's performance on the specific task at hand.  

<img width="2368" alt="Image" src="https://github.com/user-attachments/assets/fb83ec92-ee23-4c1f-9972-c259a6c555e2" />

* __Why non-linear activation functions are necessary__

  * If we only use linear activation functions, the output of each layer would be a linear combination of the inputs, and the entire neural network would essentially be a linear model. This would limit the network's ability to __learn complex patterns__ and relationships in the data. 

  * Non-linear activation functions allow the network to capture non-linear relationships, enabling it to model more complex functions and make more accurate predictions. 

  * Without non-linear activation functions, a neural network would not be able to learn and __represent the intricate patterns__ that are often present in real-world data, making it less effective for tasks such as image recognition, natural language processing, and other complex problems. 


### Multiclass classification
* Multiclass classification is a type of classification problem where there are more than two classes or categories to predict.
* In multiclass classification, the goal is to assign an input to one of several possible classes. This is in contrast to binary classification, where there are only two classes (e.g., yes/no, true/false). 
* Multiclass classification can be implemented using various algorithms, such as logistic regression, decision trees, support vector machines, and neural networks. 
* The choice of algorithm depends on the specific problem, the amount of data available, and the desired level of interpretability. 
* Common applications of multiclass classification include image recognition, natural language processing, and medical diagnosis.

<img width="2342" alt="Image" src="https://github.com/user-attachments/assets/02d99375-b9c2-4d1b-9da4-bbe5af1ee5aa" />

### Softmax function
* The softmax function is a mathematical function that converts a vector of real numbers into a probability distribution.
* It is commonly used in the output layer of a neural network for multiclass classification problems. 
* The softmax function takes a vector of raw scores (logits) as input and applies the following transformation to produce a probability distribution over the classes:  

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
``` 

where $z_i$ is the raw score for class $i$, and $K$ is the total number of classes. The softmax function ensures that the output values are between 0 and 1, and that they sum to 1, making them interpretable as probabilities. The class with the highest probability is typically chosen as the predicted class for a given input.  

<img width="2380" alt="Image" src="https://github.com/user-attachments/assets/7de9e06b-586a-4641-87bb-9812943f4219" />

* Softmax and cross-entropy loss for multiclass classification
  * The softmax function is often used in conjunction with the cross-entropy loss function for training neural networks on multiclass classification problems. 
  * The cross-entropy loss function measures the difference between the predicted probability distribution (output of the softmax function) and the true distribution (the one-hot encoded labels). The loss is calculated as follows:  
```math
J(w,b) = -\frac{1}{m} \sum_{i=0}^{m-1} \sum_{j=1}^{K} y_j^{(i)} \log(\text{softmax}(z_j^{(i)}))
```
where $y_j^{(i)}$ is the true label for class $j$ for
the $i$-th training example, and $\text{softmax}(z_j^{(i)})$ is the predicted probability for class $j$ for the $i$-th training example. The goal of training is to minimize this loss function, which encourages the model to output probabilities that are close to the true labels.

<img width="2368" alt="Image" src="https://github.com/user-attachments/assets/a69a47ac-7de4-428f-ba64-033fd09c84b2" />
&nbsp;

<img width="2408" alt="Image" src="https://github.com/user-attachments/assets/b1162260-c707-40ad-b3fd-e6cb7dc34f76" />


* Choosing Activation
  * __Output Layer__: depends on the response we are waiting.
    * `sigmoid`: for binary classification, $y=1/0$
    * `linear`: for regression, $y = +/-$
    * `ReLU`: for regression, $y>=0$ 
  * __Hidden Layers__: ReLU

* Softmax for multiple classification.

* Multilab classification.

* Adam Algorithm: Adaptive Moment estimation
  
* Convolutional layer and Convonutional Neural Network


### Week 2: Labs
* Lab 01: [ReLU activation](02_week/C2_W2_lab01_Relu.ipynb)
* Lab 02: [Softmax function](02_week/C2_W2_lab02_SoftMax.ipynb)
* Lab 03: [Neural Network for multi-class classification](02_week/C2_W2_lab03_Multiclass_TF.ipynb)

## Week 3: Advice for applying machine learning
>This week you'll learn best practices for training and evaluating your learning algorithms to improve performance. This will cover a wide range of useful advice about the machine learning lifecycle, tuning your model, and also improving your training data 

__Learning Objectives__
* Evaluate and then modify your learning algorithm or data to improve your model's performance
* Evaluate your learning algorithm using cross validation and test datasets.
* Diagnose bias and variance in your learning algorithm
* Use regularization to adjust bias and variance in your learning algorithm
* Identify a baseline level of performance for your learning algorithm
* Understand how bias and variance apply to neural networks
* Learn about the iterative loop of Machine Learning Development that's used to update and improve a machine learning model
* Learn to use error analysis to identify the types of errors that a learning algorithm is making
* Learn how to add more training data to improve your model, including data augmentation and data synthesis
* Use transfer learning to improve your model's performance.
* Learn to include fairness and ethics in your machine learning model development
* Measure precision and recall to work with skewed (imbalanced) datasets

### Notes
* Training data set
* Bias
* Variance

### Week 3: Labs
* Lab 01: [model evaluation and selection](03_week/C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb)
* Lab 02: [diagnosing bias and variance](03_week/C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb)

## Week 4: Decision trees
>This week, you'll learn about a practical and very commonly used learning algorithm the decision tree. You'll also learn about variations of the decision tree, including random forests and boosted trees (XGBoost).

__Learning Objectives__
* See what a decision tree looks like and how it can be used to make predictions
* Learn how a decision tree learns from training data
* Learn the "impurity" metric "entropy" and how it's used when building a decision tree
* Learn how to use multiple trees, "tree ensembles" such as random forests and boosted trees
* Learn when to use decision trees or neural networks

### Notes
* __Entropy__ as measure of impurity.  
$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$
<img width="400" alt="Image" src="https://github.com/user-attachments/assets/8b3677aa-b31e-4bf1-b9c5-c96443269cb0" />  
  


&nbsp;
* __Information Gain__ or reduction of entropy, use to choose a feature to split
  
$\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right)$

<img width="2610" alt="Image" src="https://github.com/user-attachments/assets/376f662b-1c8c-4d06-b42e-930b7646cb83" />

&nbsp;
* __One Hot Encoding__  
solution when a feature can take more than two possible values. One Hot because only one feature is selected.  

<img width="2500" alt="Image" src="https://github.com/user-attachments/assets/1a703253-6786-46a0-9319-d32c2651d583" />


&nbsp;
  
* __Continue Value features__
<img width="1244" height="547" alt="Image" src="https://github.com/user-attachments/assets/eda27b66-63b2-44ce-97bf-ac11e5ab53f2" />

&nbsp;

* __Tree Ensemble__  
using branch of decision trees instead only one, the final decision is taken by mayority of each tree.  
<img width="2500" alt="Image" src="https://github.com/user-attachments/assets/a2998729-de96-4888-82fc-b4f09a4cef31" />
&nbsp;


* __Sampling with replacement__  
the idea is to build a new training set, similar from the original.


* __Random Forest Algorithm__   
algorithm to build tree ensemble.
Idea: when choosing a feature to use split, if $n$ features are available, pick a `random` subset of features $(k < n)$ and allow the algorithm to only choose from that subset of features. Usually $k = \sqrt{n}$.

* __XGBoost__   (eXtreme Gradient Boosting)
Boosted trees, where each tree is trained to correct the errors of the previous tree. The final decision is taken by weighted mayority of each tree.


### Week 4: Labs
* Lab 01: [Decision Trees](04_week/C2_W4_Lab_01_Decision_Trees.ipynb)
In this notebook you will visualize how a decision tree is split using information gain.
* Lab 02: [Tree Ensemble](04_week/C2_W4_Lab_02_Tree_Ensemble.ipynb)
