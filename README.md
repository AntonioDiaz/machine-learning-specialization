<h1> Notes for Coursera: Machine Learning Specialization </h1>

Contents
- [Links](#links)
- [01 - Supervised Machine Learning: Regression and Classification](#01---supervised-machine-learning-regression-and-classification)
  - [Week 1: Introduction to Machine Learning](#week-1-introduction-to-machine-learning)
    - [Notes](#notes)
  - [Week 2: Regression with multiple input variables](#week-2-regression-with-multiple-input-variables)
  - [Week 3: Classification](#week-3-classification)
- [02 - Advanced Learning Algorithms](#02---advanced-learning-algorithms)
  - [Week 1: Neural networks](#week-1-neural-networks)
  - [Week 2: Neural network training](#week-2-neural-network-training)
  - [Week 3: Advide for applying machine learning](#week-3-advide-for-applying-machine-learning)
  - [Week 4: Decision trees](#week-4-decision-trees)
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

* `Cost function`
$$J(w,b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

* `Gradient Descent`
$$
\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}
$$
Where:
$$
\begin{aligned}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \\
\end{aligned}
$$


### Week 2: Regression with multiple input variables
* Use vectorization to implement multiple linear regression
* Use feature scaling, feature engineering, and polynomial regression to * improve model training
* Implement linear regression in code

### Week 3: Classification
* Use logistic regression for binary classification
* Implement logistic regression for binary classification
* Address overfitting using regularization, to improve model performance

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


### Week 3: Advide for applying machine learning

### Week 4: Decision trees

## 03 - Unsupervised Learning, Recommenders, Reinforcement Learning

https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction