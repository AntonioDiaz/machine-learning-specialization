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
>Welcome to the Machine Learning Specialization! You're joining millions of others who have taken either this or the original course, which led to the founding of Coursera, and has helped millions of other learners, like you, take a look at the exciting world of machine learning!

__Learning Objectives__
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

>This week, you'll extend linear regression to handle multiple input features. You'll also learn some methods for improving your model's training and performance, such as vectorization, feature scaling, feature engineering and polynomial regression. At the end of the week, you'll get to practice implementing linear regression in code.

__Learning Objectives__
* Use vectorization to implement multiple linear regression
* Use feature scaling, feature engineering, and polynomial regression to * improve model training
* Implement linear regression in code

#### Notes 
todo

#### Labs
* [Lab 01](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb): Python, NumPy and Vectorization
* [Lab 02](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab02_Multiple_Variable_Soln.ipynb): Multiple Variable Linear Regression
* [Lab 03](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb): Feature scaling and Learning Rate (Multi-variable)
* [Lab 04](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb): Feature Engineering and Polynomial Regression
* [Lab 05](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab05_Sklearn_GD_Soln.ipynb): Linear Regression using Scikit-Learn, Gradient Descent
* [Lab 06](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb): Linear Regression using Scikit-Learn, Linear Regression using a close form solution

### Week 3: Classification
>This week, you'll learn the other type of supervised learning, classification. You'll learn how to predict categories using the logistic regression model. You'll learn about the problem of overfitting, and how to handle this problem with a method...

__Learning Objectives__
* Use logistic regression for binary classification
* Implement logistic regression for binary classification
* Address overfitting using regularization, to improve model performance

#### Notes 
* `sigmoid function` todo
* todo

#### Labs
* [Lab 01](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab01_Classification_Soln.ipynb): classification
* [Lab 02](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab02_Sigmoid_function_Soln.ipynb): Logistic Regression
* [Lab 03](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb): Logistic Regression and Decision Boundary
* [Lab 04](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab04_LogisticLoss_Soln.ipynb): Logistic Regression and Logistic Loss
* [Lab 05](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab05_Cost_Function_Soln.ipynb): Cost Function for Logistic Regression
* [Lab 06](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab06_Gradient_Descent_Soln.ipynb): Gradient Descent for Logisic Regression
* [Lab 07](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab07_Scikit_Learn_Soln.ipynb): Logistic Regression using Scikit-Learn
* [Lab 08](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab08_Overfitting_Soln.ipynb): Overfitting
* [Lab 09](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab09_Regularization_Soln.ipynb): Regularized Cost and Gradient

## 02 - Advanced Learning Algorithms

https://www.coursera.org/learn/advanced-learning-algorithms/home/info

### Week 1: Neural networks
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

#### Notes
todo

#### Labs
* [Lab 01](02_advanced_learning_algorithms/01_week/C2_W1_Lab01_Neurons_and_Layers.ipynb): Neurons and Layers, introduction to Tensorflow and Keras.
* [Lab 02](02_advanced_learning_algorithms/01_week/C2_W1_Lab02_CoffeeRoasting_TF.ipynb): Simple Neural Network with Tensorflow, coffee roasting example.
* [Lab 03](02_advanced_learning_algorithms/01_week/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb): Simple Neural Network with Numpy, coffee roasting example.

### Week 2: Neural network training
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

#### Notes
* Training data set
* Bias
* Variance

#### Labs
* Lab 01: [model evaluation and selection](02_advanced_learning_algorithms/03_week/C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb)
* Lab 02: [diagnosing bias and variance](02_advanced_learning_algorithms/03_week/C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb)


### Week 4: Decision trees
>This week, you'll learn about a practical and very commonly used learning algorithm the decision tree. You'll also learn about variations of the decision tree, including random forests and boosted trees (XGBoost).

__Learning Objectives__
* See what a decision tree looks like and how it can be used to make predictions
* Learn how a decision tree learns from training data
* Learn the "impurity" metric "entropy" and how it's used when building a decision tree
* Learn how to use multiple trees, "tree ensembles" such as random forests and boosted trees
* Learn when to use decision trees or neural networks

#### Notes
* __Entropy__ as measure of impurity. 
  
<img width="400" alt="Image" src="https://github.com/user-attachments/assets/8b3677aa-b31e-4bf1-b9c5-c96443269cb0" />

$$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$   

* __Information Gain__ or reduction of entropy, use to choose a feature to split
  
$$\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right)$$

<img width="2610" alt="Image" src="https://github.com/user-attachments/assets/376f662b-1c8c-4d06-b42e-930b7646cb83" />

* __One Hot Encoding__: solution when a feature can take more than two possible values. One Hot because only one feature is selected.  

<img width="2500" alt="Image" src="https://github.com/user-attachments/assets/1a703253-6786-46a0-9319-d32c2651d583" />

* Continue Value features

#### Labs
* [Lab 01](02_advanced_learning_algorithms/04_week/C2_W4_Lab_01_Decision_Trees.ipynb): Decision Trees
* [Lab 02](02_advanced_learning_algorithms/04_week/C2_W4_Lab_02_Tree_Ensemble.ipynb): Treen Ensemble


## 03 - Unsupervised Learning, Recommenders, Reinforcement Learning

https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction