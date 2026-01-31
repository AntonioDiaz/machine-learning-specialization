<h1> Notes for Coursera: Machine Learning Specialization </h1>

Contents
- [Links](#links)
- [01 - Supervised Machine Learning: Regression and Classification](#01---supervised-machine-learning-regression-and-classification)
  - [Week 1: Introduction to Machine Learning](#week-1-introduction-to-machine-learning)
    - [Notes](#notes)
    - [Labs](#labs)
  - [Week 2: Regression with multiple input variables](#week-2-regression-with-multiple-input-variables)
    - [Notes](#notes-1)
    - [Labs](#labs-1)
  - [Week 3: Classification](#week-3-classification)
    - [Notes](#notes-2)
    - [Labs](#labs-2)
- [02 - Advanced Learning Algorithms](#02---advanced-learning-algorithms)
  - [Week 1: Neural networks](#week-1-neural-networks)
    - [Notes](#notes-3)
    - [Labs](#labs-3)
  - [Week 2: Neural network training](#week-2-neural-network-training)
    - [Notes](#notes-4)
    - [Labs](#labs-4)
  - [Week 3: Advice for applying machine learning](#week-3-advice-for-applying-machine-learning)
    - [Notes](#notes-5)
    - [Labs](#labs-5)
  - [Week 4: Decision trees](#week-4-decision-trees)
    - [Notes](#notes-6)
    - [Labs](#labs-6)
- [03 - Unsupervised Learning, Recommenders, Reinforcement Learning](#03---unsupervised-learning-recommenders-reinforcement-learning)
<hr>

## Links
* Coursera  
https://www.coursera.org/specializations/machine-learning-introduction

* Forum  
https://community.deeplearning.ai/c/course-q-a/generative-ai-for-software-development/478

* DeepLearningAI  
https://www.deeplearning.ai/courses/machine-learning-specialization/

* Cool repositories  
  https://github.com/pmulard/machine-learning-specialization-andrew-ng



## 01 - Supervised Machine Learning: Regression and Classification

https://www.coursera.org/learn/machine-learning/home/info

### Week 1: Introduction to Machine Learning
* Define machine learning
* Define supervised learning
* Define unsupervised learning
* Write and run Python code in Jupyter Notebooks
* Define a regression model
* Implement and visualize a cost function
* Implement gradient descent
* Optimize a regression model using gradient descent


#### Notes
* `Linear Regression Model`  
  
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

* `Squared Error Cost function`  
  
$$J(w,b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

* `Gradient Descent`: optimizing w and b  
  
$$\begin{aligned} 
&\text{repeat until convergence: } \lbrace \\
&\quad w = w - \alpha \frac{\partial J(w,b)}{\partial w} \\
&\quad b = b - \alpha \frac{\partial J(w,b)}{\partial b} \\
&\rbrace
\end{aligned}$$

$$\begin{aligned} \text{where:} \newline
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \\
\end{aligned}$$

* `Learning rate`   __$\alpha$__  

* `Derivative term for` __$\omega$__  

$$\frac{\partial J(w,b)}{\partial w}$$  

#### Labs
* [Lab 01](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab01_Python_Jupyter_Soln.ipynb): Jupyter notebook introduction. 
* [Lab 02](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab02_Model_Representation_Soln.ipynb): Linear regression for one variable. 
* [Lab 03](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab03_Cost_function_Soln.ipynb): Cost function for linear regression with one variable. 
* [Lab 04](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab04_Gradient_Descent_Soln.ipynb): Gradient Descent. 
  
### Week 2: Regression with multiple input variables
* Use vectorization to implement multiple linear regression
* Use feature scaling, feature engineering, and polynomial regression to * improve model training
* Implement linear regression in code

#### Notes 
todo

#### Labs
* Lab 01: Python, NumPy and Vectorization
* Lab 02: Multiple Variable Linear Regression
* Lab 03: Feature scaling and Learning Rate (Multi-variable)
* Lab 04: Feature Engineering and Polynomial Regression
* Lab 05: Linear Regression using Scikit-Learn, Gradient Descent
* Lab 06: Linear Regression using Scikit-Learn, Linear Regression using a close form solution

### Week 3: Classification
* Use logistic regression for binary classification
* Implement logistic regression for binary classification
* Address overfitting using regularization, to improve model performance

#### Notes 
* `sigmoid function` todo
* todo

#### Labs
* Lab 01: classification
* Lab 02: Logistic Regression
* Lab 03: Logistic Regression and Decision Boundary
* Lab 04: Logistic Regression and Logistic Loss
* Lab 05: Cost Function for Logistic Regression
* Lab 06: Gradient Descent for Logisic Regression
* Lab 07: Logistic Regression using Scikit-Learn
* Lab 08: Overfitting
* Lab 09: Regularized Cost and Gradient

## 02 - Advanced Learning Algorithms

https://www.coursera.org/learn/advanced-learning-algorithms/home/info

### Week 1: Neural networks

This week, you'll learn about neural networks and how to use them for classification tasks. You'll use the TensorFlow framework to build a neural network with just a few lines of code. Then, dive deeper by learning how to code up your own neural network in Python, "from scratch". Optionally, you can learn more about how neural network computations are implemented efficiently using parallel processing (vectorization).

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

#### Notes
todo

#### Labs
* [Lab 01](02_advanced_learning_algorithms/01_week/C2_W1_Lab01_Neurons_and_Layers.ipynb): Neurons and Layers, introduction to Tensorflow and Keras.
* [Lab 02](02_advanced_learning_algorithms/01_week/C2_W1_Lab02_CoffeeRoasting_TF.ipynb): Simple Neural Network with Tensorflow, coffee roasting example.
* [Lab 03](02_advanced_learning_algorithms/01_week/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb): Simple Neural Network with Numpy, coffee roasting example.

### Week 2: Neural network training

This week, you'll learn how to train your model in `TensorFlow`, and also learn about other important `activation functions` (besides the sigmoid function), and where to use each type in a neural network. You'll also learn how to go beyond binary classification to `multiclass classification` (3 or more categories). Multiclass classification will introduce you to a new activation function and a new loss function. Optionally, you can also learn about the difference between multiclass classification and multi-label classification. You'll learn about the `Adam optimizer`, and why it's an improvement upon regular gradient descent for neural network training. Finally, you will get a brief introduction to other layer types besides the one you've seen thus far.

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

#### Notes
* Model training steps
  1. Create the model
  2. Loss and cont functions
  3. Gradient Descent

* Activation functions
  * Linear Activation Function
  * Sigmoid
  * **ReLU: Rectified Linear Unit** $a = \max(0,z)$
<img width="1000" src="https://github.com/user-attachments/assets/243a44bc-2707-4b56-ba0a-3dc8e1bf58d0" />

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


#### Labs
* [Lab 01](02_advanced_learning_algorithms/02_week/C2_W2_lab01_Relu.ipynb): ReLU activation
* [Lab 02](02_advanced_learning_algorithms/02_week/C2_W2_lab02_SoftMax.ipynb): Softmax function
* [Lab 03](02_advanced_learning_algorithms/02_week/C2_W2_lab03_Multiclass_TF.ipynb): Neural Network for multi-class classification

### Week 3: Advice for applying machine learning

This week you'll learn best practices for training and evaluating your learning algorithms to improve performance. This will cover a wide range of useful advice about the machine learning lifecycle, tuning your model, and also improving your training data 

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

#### Notes
todo

#### Labs
* Lab 01: [model evaluation and selection](02_advanced_learning_algorithms/03_week/C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb)
* Lab 02: d[iagnosing bias and variance](02_advanced_learning_algorithms/03_week/C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb)


### Week 4: Decision trees

#### Notes
todo

#### Labs
todo


## 03 - Unsupervised Learning, Recommenders, Reinforcement Learning

https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction