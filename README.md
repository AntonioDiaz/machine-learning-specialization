<h1> Notes for Coursera: Machine Learning Specialization </h1>

Contents
- [Links](#links)
- [01 - Supervised Machine Learning: Regression and Classification](#01---supervised-machine-learning-regression-and-classification)
  - [Week 1: Introduction to Machine Learning](#week-1-introduction-to-machine-learning)
    - [Definitions](#definitions)
    - [Intro Supervised Learning](#intro-supervised-learning)
    - [Intro to Unsupervised Learning](#intro-to-unsupervised-learning)
    - [Linear regression](#linear-regression)
    - [Squared Error Cost function](#squared-error-cost-function)
    - [Gradient Descent](#gradient-descent)
    - [Labs](#labs)
  - [Week 2: Regression with multiple input variables](#week-2-regression-with-multiple-input-variables)
    - [Notes](#notes)
    - [Labs](#labs-1)
  - [Week 3: Classification](#week-3-classification)
    - [Notes](#notes-1)
    - [Labs](#labs-2)
- [02 - Advanced Learning Algorithms](#02---advanced-learning-algorithms)
  - [Week 1: Neural networks](#week-1-neural-networks)
    - [Neural Networks](#neural-networks)
    - [Labs](#labs-3)
  - [Week 2: Neural network training](#week-2-neural-network-training)
    - [Notes](#notes-2)
    - [Labs](#labs-4)
  - [Week 3: Advice for applying machine learning](#week-3-advice-for-applying-machine-learning)
    - [Notes](#notes-3)
    - [Labs](#labs-5)
  - [Week 4: Decision trees](#week-4-decision-trees)
    - [Notes](#notes-4)
    - [Labs](#labs-6)
- [03 - Unsupervised Learning, Recommenders, Reinforcement Learning](#03---unsupervised-learning-recommenders-reinforcement-learning)
  - [Week 1: Unsupervised learning](#week-1-unsupervised-learning)
    - [Notes](#notes-5)
    - [K-means clustering](#k-means-clustering)
    - [Anomaly detection](#anomaly-detection)
    - [Labs](#labs-7)
  - [Week 2: Recommender systems](#week-2-recommender-systems)
    - [Colaborative filtering recommender systems](#colaborative-filtering-recommender-systems)
    - [Mean normalization for collaborative filtering](#mean-normalization-for-collaborative-filtering)
    - [TensorFlow implementation of collaborative filtering](#tensorflow-implementation-of-collaborative-filtering)
    - [Content-based filtering](#content-based-filtering)
    - [Labs](#labs-8)
  - [Week 3 Reinforcement Learning](#week-3-reinforcement-learning)
    - [Reinforcement Learning introduction](#reinforcement-learning-introduction)
    - [State-action value function](#state-action-value-function)
    - [Bellman Equation](#bellman-equation)
    - [Deep Reinforcement learning](#deep-reinforcement-learning)
    - [Labs](#labs-9)


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

#### Definitions
* __Machine Learning__ is the science of getting computers to act without being explicitly programmed. It is a subset of artificial intelligence (AI) that focuses on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention. Machine learning algorithms use statistical techniques to enable machines to improve their performance on a specific task over time as they are exposed to more data.
    

#### Intro Supervised Learning
* __Supervised Learning__ is a type of machine learning where the algorithm is trained on a __labeled dataset__, which means that each training example is paired with an output label. The goal of supervised learning is to learn a mapping from inputs to outputs, so that the model can make predictions on new, unseen data. Examples of supervised learning tasks include regression (predicting continuous values) and classification (predicting discrete categories).
* There are two main types of supervised learning tasks:
  * __Regression__: predicting a continuous value (e.g., price of a house)
  * __Classification__: predicting a discrete category (e.g., whether an email is spam or not)
  
<img width="1392" alt="Image" src="https://github.com/user-attachments/assets/33748e53-666b-4e32-b342-54d7556d211b" />
&nbsp;

<img width="1586" alt="Image" src="https://github.com/user-attachments/assets/fbbb6308-9640-44df-b0ce-23fdbd54b584" />
&nbsp;

#### Intro to Unsupervised Learning
* __Unsupervised Learning__ is a type of machine learning where the algorithm is trained on an __unlabeled dataset__, which means that the training examples do not have output labels. The goal of unsupervised learning is to find hidden __patterns__ or structures in the data. Examples of unsupervised learning tasks include clustering (grouping similar data points together) and dimensionality reduction (reducing the number of features in the data while preserving important information).
* There are three main types of unsupervised learning tasks:
  * __Clustering__: grouping data points into clusters of similar examples.
  * __Anomaly detection__: identifying data points that are significantly different from the majority of the data.
  * __Dimensionality reduction__: reducing the number of features in the data while preserving important information.

<img width="1486" alt="Image" src="https://github.com/user-attachments/assets/71d49132-0f73-4baa-b796-fe52a44410b3" />  
&nbsp;

#### Linear regression 
* `Linear Regression Model`
  
$f_{w,b}(x^{(i)}) = wx^{(i)} + b$

<img width="1990" alt="Image" src="https://github.com/user-attachments/assets/91d229ca-6a96-4e74-8865-fa4b0214c129" />
&nbsp;

#### Squared Error Cost function 
 
$J(w,b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $

<img width="2001" alt="Image" src="https://github.com/user-attachments/assets/287900f1-fa39-4080-9725-e5c111ab741e" />
&nbsp;


#### Gradient Descent
* `Gradient Descent` is an optimization algorithm used to minimize the cost function $J(w,b)$ by iteratively updating the parameters $w$ and $b$ in the direction of the negative gradient of the cost function with respect to those parameters. The learning rate $\alpha$ determines the size of the steps taken towards the minimum of the cost function. The algorithm continues until convergence, which occurs when the parameters no longer change significantly or when a predetermined number of iterations is reached.
  
<img width="1972" alt="Image" src="https://github.com/user-attachments/assets/43719b95-e84f-47c2-b166-c57369b67e2b" />
&nbsp;

<img width="1992" alt="Image" src="https://github.com/user-attachments/assets/ffcc1e19-42ef-4690-9842-556a0d78d4a2" />
&nbsp;

```math
\begin{aligned}
\text{repeat until convergence: } \{ \\
 w &= w - \alpha \frac{\partial J(w,b)}{\partial w} \\
 b &= b - \alpha \frac{\partial J(w,b)}{\partial b} \\
\}
\end{aligned}
```

```math
\begin{aligned}
\text{where:} \\
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right) x^{(i)} \\
\frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
\end{aligned}
```

* `Learning rate`   __$\alpha$__  

* `Derivative term for` __$\omega$__  

$\frac{\partial J(w,b)}{\partial w}$  


#### Labs
* Lab 01: [Python Jupyter Notebook introduction](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab01_Python_Jupyter_Soln.ipynb)
* Lab 02: [Linear regression for one variable](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab02_Model_Representation_Soln.ipynb)
* Lab 03: [Cost function for linear regression with one variable](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab03_Cost_function_Soln.ipynb)
* Lab 04: [Gradient Descent](01_supervised_ml_regression_and_classification/01_week/C1_W1_Lab04_Gradient_Descent_Soln.ipynb)

### Week 2: Regression with multiple input variables

>This week, you'll extend linear regression to handle multiple input features. You'll also learn some methods for improving your model's training and performance, such as vectorization, feature scaling, feature engineering and polynomial regression. At the end of the week, you'll get to practice implementing linear regression in code.

__Learning Objectives__
* Use vectorization to implement multiple linear regression
* Use feature scaling, feature engineering, and polynomial regression to * improve model training
* Implement linear regression in code

#### Notes 
todo

#### Labs
* Lab 01: [Python, NumPy and Vectorization](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb)
* Lab 02: [Multiple Variable Linear Regression](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab02_Multiple_Variable_Soln.ipynb)
* Lab 03: [Feature scaling and Learning Rate (Multi-variable)](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb)
* Lab 04: [Feature Engineering and Polynomial Regression](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb)
* Lab 05: [Linear Regression using Scikit-Learn, Gradient Descent](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab05_Sklearn_GD_Soln.ipynb)
* Lab 06: [Linear Regression using Scikit-Learn, close form solution](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb)

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
* Lab 01: [Classification](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab01_Classification_Soln.ipynb)
* Lab 02: [Logistic Regression](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab02_Sigmoid_function_Soln.ipynb)
* Lab 03: [Logistic Regression and Decision Boundary](01_supervised_ml_regression_and_classification/02_week/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb)
* Lab 04: [Logistic Regression and Logistic Loss](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab04_LogisticLoss_Soln.ipynb)
* Lab 05: [Cost Function for Logistic Regression](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab05_Cost_Function_Soln.ipynb)
* Lab 06: [Gradient Descent for Logistic Regression](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab06_Gradient_Descent_Soln.ipynb)
* Lab 07: [Logistic Regression using Scikit-Learn](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab07_Scikit_Learn_Soln.ipynb)
* Lab 08: [Overfitting](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab08_Overfitting_Soln.ipynb)
* Lab 09: [Regularized Cost and Gradient](01_supervised_ml_regression_and_classification/03_week/C1_W3_Lab09_Regularization_Soln.ipynb)

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

#### Neural Networks
* Bioligical neuron vs simplified artificial neural network
<img width="1594" alt="Image" src="https://github.com/user-attachments/assets/c1dc2678-3a20-4109-8880-26fae3ea86f1" />
&nbsp;

* From Logistic Regression to Neural Networks
  * Logistic regression can be seen as a simple neural network with no hidden layers and a sigmoid activation function. By adding hidden layers and using different activation functions, we can create more complex neural networks that can learn more complex patterns in the data.  
  * Only 1 feature
<img width="1562" alt="Image" src="https://github.com/user-attachments/assets/f857dd53-a352-403d-af89-57cb166b68f6" />
&nbsp;

* Layers in a neural network
  * Input layer: the layer that receives the input data.
  * Hidden layers: the layers that perform computations and learn features from the input data.
  * Output layer: the layer that produces the final output of the neural network, such as a prediction or classification.   
<img width="1550" alt="Image" src="https://github.com/user-attachments/assets/33586680-8e40-4683-a048-61b205200b0e" />
&nbsp;

* Neural Network Architecture
  * The architecture of a neural network refers to the number of layers and the number of neurons in each layer. The architecture can be designed based on the complexity of the problem and the amount of data available. A common architecture for image classification tasks is a convolutional neural network (CNN), which consists of convolutional layers, pooling layers, and fully connected layers. The choice of architecture can have a significant impact on the performance of the neural network.  
<img width="1554" alt="Image" src="https://github.com/user-attachments/assets/560363ae-887d-4be1-8f58-3a61ae65040a" />
&nbsp;

#### Labs
* Lab 01: [Neurons and Layers, introduction to TensorFlow and Keras](02_advanced_learning_algorithms/01_week/C2_W1_Lab01_Neurons_and_Layers.ipynb)
* Lab 02: [Simple Neural Network with TensorFlow, coffee roasting example](02_advanced_learning_algorithms/01_week/C2_W1_Lab02_CoffeeRoasting_TF.ipynb)
* Lab 03: [Simple Neural Network with Numpy, coffee roasting example](02_advanced_learning_algorithms/01_week/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb)

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
* Lab 01: [ReLU activation](02_advanced_learning_algorithms/02_week/C2_W2_lab01_Relu.ipynb)
* Lab 02: [Softmax function](02_advanced_learning_algorithms/02_week/C2_W2_lab02_SoftMax.ipynb)
* Lab 03: [Neural Network for multi-class classification](02_advanced_learning_algorithms/02_week/C2_W2_lab03_Multiclass_TF.ipynb)

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


#### Labs
* Lab 01: [Decision Trees](02_advanced_learning_algorithms/04_week/C2_W4_Lab_01_Decision_Trees.ipynb)
In this notebook you will visualize how a decision tree is split using information gain.

* Lab 02: [Tree Ensemble](02_advanced_learning_algorithms/04_week/C2_W4_Lab_02_Tree_Ensemble.ipynb)


## 03 - Unsupervised Learning, Recommenders, Reinforcement Learning

https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/home/welcome


### Week 1: Unsupervised learning
> This week, you will learn two key unsupervised learning algorithms: clustering and anomaly detection

__Learning Objectives__
* Implement the k-means clustering algorithm
* Implement the k-means optimization objective
* Initialize the k-means algorithm
* Choose the number of clusters for the k-means algorithm
* Implement an anomaly detection system
* Decide when to use supervised learning vs. anomaly detection
* Implement the centroid update function in k-means
* Implement the function that finds the closest centroids to each point in k-means

#### Notes
* __Unsupervised learning__ learning from data that is not labeled.
  
* __Clustering__ grouping data points into clusters of similar examples.
  
#### K-means clustering  
  
Algorithm for clustering data points into K clusters. The algorithm iteratively assigns each data point to the closest cluster centroid and then updates the centroids based on the mean of the assigned points.  

K-means clustering algorithm:  
  * Randomly initialize K centroids.
  * Then, repeat the following steps until convergence:
    * Step 1: Assigng each point to its closest centroid to form K clusters.
    * Step 2: Recommpute the centroids.  
<img width="2370" alt="Image" src="https://github.com/user-attachments/assets/2891c95d-9e63-42cd-b919-2635ef7b32c2" />
&nbsp;

<img width="2320" alt="Image" src="https://github.com/user-attachments/assets/8b21adf1-1603-4cbd-98da-03946260caae" />
&nbsp;

* __K_means algorithm__
<img width="2338" alt="Image" src="https://github.com/user-attachments/assets/67f4f7c0-967e-45e2-b67f-bb5d96b8d8b5" />
&nbsp;

* __K-means optimization objective__
  * The K-means algorithm is trying to minimize the following cost function, also called `distortion`:
$J(c^{(1)}, \ldots, c^{(m)}, \mu_1, \ldots, \mu_K) = \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$
<img width="2316" alt="Image" src="https://github.com/user-attachments/assets/c48efe10-2eea-4cb7-bde7-8ad645a65462" />
&nbsp;

* __Initialization of K-means__
  * Random initialization: randomly select K data points as initial centroids.

* __Choosing the number of clusters K__
  * Elbow method: plot the cost function J as a function of K and look for an "elbow" in the graph where the cost starts to decrease more slowly.

#### Anomaly detection  
Identifying data points that are significantly different from the majority of the data. This can be useful for tasks such as fraud detection, network security, and quality control.

* __Density estimation__  
A common approach to anomaly detection is to estimate the probability density function of the data and then flag data points that have a low probability as anomalies.

* __Gaussian distribution__
<img width="2354" alt="Image" src="https://github.com/user-attachments/assets/909f8ae5-9707-4a04-a682-b0487bfd8a41" />
&nbsp;

* __Anomaly detection algorithm with one feature__
  * Estimate the parameters $\mu$ and $\sigma^2$ of the Gaussian distribution using the training data.
  * For a new data point $x$, compute the probability density function $p(x)$ using the estimated parameters.
  * Flag $x$ as an anomaly if $p(x) < \epsilon$, where $\epsilon$ is a threshold that you can choose based on your desired false positive rate.  
<img width="2344" alt="Image" src="https://github.com/user-attachments/assets/1b8e293c-eefc-4060-845b-8789328e20cf" />
&nbsp;

* __Anomaly detection algorithm with multiple features__
  * Estimate the parameters $\mu$ and $\sigma^2$ of the multivariate Gaussian distribution using the training data. 
  * For a new data point $x$, compute the probability density function $p(x)$ using the estimated parameters.
  * Flag $x$ as an anomaly if $p(x) < \epsilon$, where $\epsilon$ is a threshold that you can choose based on your desired false positive rate.
<img width="2318" alt="Image" src="https://github.com/user-attachments/assets/c62f2aa9-7850-4af3-be2e-48fc9ee5a643" />
&nbsp;

* __Developing and evaluating an anomaly detection system__
  * Split your data into a training set, a cross-validation set, and a test set.
  * Use the training set to estimate the parameters of the Gaussian distribution.
  * Use the cross-validation set to select the threshold $\epsilon$ that gives you the desired false positive rate.
  * Use the test set to evaluate the performance of your anomaly detection system.
  * Example of dataset for anomaly detection of aircraft engine failure: there are 2 situations:
    * 20 anomalies out of 10_000 data points
    * 2 anomalies out of 10_000 data points.   
<img width="1985" alt="Image" src="https://github.com/user-attachments/assets/fa15aba9-5168-4f82-86f4-efd24d44b8d3" />

* __Anomaly detection vs supervised learning__
Anomaly detection is used when you have very few examples of the anomaly (positive class) and many examples of the normal data (negative class). In contrast, supervised learning is used when you have a balanced dataset with enough examples of both classes. 
<img width="1916" alt="Image" src="https://github.com/user-attachments/assets/d4968fc7-55d4-4282-9bb5-2a803b1a1f38" />
&nbsp;

* __Choosing Features for anomaly detection__
  * The choice of features is crucial for the performance of an anomaly detection system. You should choose features that are relevant to the problem and that can help distinguish between normal and anomalous data points. For example, in the case of aircraft engine failure, you might choose features such as temperature, pressure, and vibration.


#### Labs
* Lab 01: [K-means clustering](03_unsupervised_learning/01_week/C3_W1_KMeans_Assignment.ipynb)
* Lab 02: [Anomaly detection](03_unsupervised_learning/01_week/C3_W1_Anomaly_Detection.ipynb)

### Week 2: Recommender systems

__Learning Objectives__
* Implement __collaborative filtering__ recommender systems in TensorFlow.
* Implement deep learning __content based filtering__ using a neural network in TensorFlow.
* Understand ethical considerations in building recommender systems.

#### Colaborative filtering recommender systems
* __Collaborative filtering__  
  * Is a method of making recommendations based on the preferences of similar users.  
  * The idea is to find users who have similar preferences and then recommend items that those similar users have liked.  
<img width="1970" alt="Image" src="https://github.com/user-attachments/assets/edbd5367-6987-4782-ac1d-75004af73ce8" />
&nbsp;

* __Cost function for collaborative filtering__  
To learn parameters w and b for collaborative filtering, we can use the following cost function
<img width="1994" alt="Image" src="https://github.com/user-attachments/assets/39fc439e-d423-49e2-8abb-94e8ec4eeb9c" />
&nbsp;

* Function to learn parameters w and b for collaborative filtering
<img width="1992" alt="Image" src="https://github.com/user-attachments/assets/88e5be2b-5d03-40f2-8940-95ca658313d1" />
&nbsp;

* Function to learn features x for collaborative filtering, where x represents the features of the items (e.g., movies) that users interact with. In collaborative filtering, we want to learn both the parameters w and b for the users, as well as the features x for the items. 
* The cost function for learning features x can be defined as follows  
$$J(x) = \frac{1}{2m} \sum_{i=1}^m \sum_{j=1}^n (f_{w,b}(x^{(i)}) - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^n ||x^{(j)}||^2$$
where m is the number of users, n is the number of items, $f_{w,b}(x^{(i)})$ is the predicted rating for user i and item j, $y^{(i,j)}$ is the actual rating given by user i for movie j, and $\lambda$ is a regularization parameter to prevent overfitting. The first term in the cost function measures the difference between the predicted ratings and the actual ratings, while the second term adds a regularization penalty to prevent overfitting by encouraging smaller feature values.  

<img width="2392" alt="Image" src="https://github.com/user-attachments/assets/12d2c7f0-0535-4464-90c4-7404a0725a9e" />
&nbsp;

* Function to learn both parameters w and b (users on the example), and features x (movies) for collaborative filtering  
<img width="2006" alt="Image" src="https://github.com/user-attachments/assets/20ed170c-3ce2-4054-affa-0c68fc708425" />
&nbsp;

* __Gradient descent__ for collaborative filtering  
<img width="1458" alt="Image" src="https://github.com/user-attachments/assets/127e838e-3688-44f7-94ee-7c3b93ace7d3" />
&nbsp;

* __Binary labels__: favs, likes and clicks   
  * Previously, we have been working with ratings as labels, which are continuous values. However, in many cases, we only have binary labels, such as whether a user liked an item or not. In this case, we can use a different cost function that is more appropriate for binary labels.

* __Cost function for binary labels__   

  $$J(w,b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]$$  

  * Where $f_{w,b}(x^{(i)})$ is the predicted probability that user $i$ will like item $j$, and $y^{(i)}$ is the actual label (1 if the user liked the item, 0 otherwise). This cost function is known as the __binary cross-entropy loss__.  
  
  * $f_{w,b}(x^{(i)})$ can be calculated using the sigmoid function, which maps the output of the linear model to a value between 0 and 1, representing the predicted probability of the user liking the item.
  $$f_{w,b}(x^{(i)}) = \sigma(w^T x^{(i)} + b) = \frac{1}{1 + e^{-(w^T x^{(i)} + b)}}$$  

<img width="1576" alt="Image" src="https://github.com/user-attachments/assets/a6ebcd20-9cb5-480a-987a-00ceef2fb92c" />

#### Mean normalization for collaborative filtering
* __Mean normalization__  
To improve the performance of collaborative filtering, we can apply mean normalization to the ratings. This involves subtracting the mean rating for each item from the ratings before training the model. This helps to account for differences in user preferences and can lead to better recommendations.  
* Normalization by rows (users) or by columns (items), when new customer is added uses normalization by rows (example below) when added new movie use normalization by columns.  
<img width="1976" alt="Image" src="https://github.com/user-attachments/assets/419a6f25-ded4-4305-a782-e76ae12b1b09" />
&nbsp;

#### TensorFlow implementation of collaborative filtering
* In TensorFlow, we can implement collaborative filtering using the Keras API. We can define a model that takes the user and item features as input and outputs the predicted rating. We can then compile the model with the appropriate loss function (e.g., binary cross-entropy for binary labels) and optimizer (e.g., Adam), and train the model on the training data. After training, we can use the model to make predictions for new user-item pairs and generate recommendations based on those predictions.

* Gradient decent reminder from previous weeks:  
<img width="1982" alt="Image" src="https://github.com/user-attachments/assets/805af192-f9bd-4262-b296-ee1c983ae229" />
&nbsp;

* __Auto Diff__ in TensorFlow: automatically compute derivative for gradient descent.  
<img width="1986" alt="Image" src="https://github.com/user-attachments/assets/2623b0a5-a71e-466d-8f48-1c98a9183efc" />

#### Content-based filtering  
* __Collaborative filtering:__ makes recommendations based on the preferences of similar users.
* __Content-based filtering:__ makes recommendations based on the features of the items themselves.  
<img width="1942" alt="Image" src="https://github.com/user-attachments/assets/4a7e517e-0437-4e13-94e6-bea8754c841d" />  
&nbsp;


* Content-based neural network architecture
<img width="2000" alt="Image" src="https://github.com/user-attachments/assets/77ad8c58-6de1-4fb6-8cf3-dee12a9808e1" />
&nbsp;

* Content-based filtering cost function
<img width="1916" alt="Image" src="https://github.com/user-attachments/assets/2a323365-604c-4b0f-8de9-e4d224a82c34" />
&nbsp;

* Finding simmilar items using content-based filtering
<img width="1892" alt="Image" src="https://github.com/user-attachments/assets/50944ae0-96b3-4bf8-8c42-a17a9e396c44" />
&nbsp;

* Recommendations for a large catalogue
  

#### Labs
* Lab 01: [Collaborative Filtering Recommender Systems](03_unsupervised_learning/02_week/C3_W2_colaborative_filtering/C3_W2_Collaborative_RecSys_Assignment.ipynb)
* Lab 02: [Content-Based Filtering Recommender Systems](03_unsupervised_learning/02_week/C2_W2_content_based_filtering/C3_W2_RecSysNN_Assignment.ipynb)

### Week 3 Reinforcement Learning
>This week, you will learn about reinforcement learning, and build a deep Q-learning neural network in order to land a virtual lunar lander on Mars!
__Learning Objectives__
* Understand key terms such as return, state, action, and policy as it applies to reinforcement learning
* Understand the Bellman equations
* Understand the state-action value function
* Understand continuous state spaces
* Build a deep Q-learning network

#### Reinforcement Learning introduction

<img width="1879" alt="Image" src="https://github.com/user-attachments/assets/807323d4-2d9c-48c2-b3e1-bd0a32199e6f" />
&nbsp;

<img width="1896" alt="Image" src="https://github.com/user-attachments/assets/d8dd8ae3-a27a-4a67-8712-8b348f256647" />
&nbsp;

#### State-action value function
* State action value function: $Q(s,a)$ represents the expected return (cumulative future reward) of taking action a in state s and following a certain policy thereafter. The goal of reinforcement learning is to learn an optimal policy that maximizes the expected return, which can be achieved by learning the optimal state-action value function $Q^*(s,a)$.  
  
<img width="1976" alt="Image" src="https://github.com/user-attachments/assets/3bc07fcf-59ac-467e-919e-df7c29f590e7" />

#### Bellman Equation
* The Bellman equation is a fundamental equation in reinforcement learning that describes the relationship between the value of a state and the values of its successor states. It can be expressed as follows:
$$Q(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s',a') | s, a]$$
where $Q(s,a)$ is the state-action value function, $r$ is the reward received after taking action $a$ in state $s$, $\gamma$ is the discount factor that determines the importance of future rewards, and $s'$ is the next state resulting from taking action $a$ in state $s$. The Bellman equation states that the value of taking action $a$ in state $s$ is equal to the expected reward plus the discounted value of the best action in the next state $s'$. This equation is used to derive the optimal policy and to update the state-action value function during the learning process.
<img width="1706" alt="Image" src="https://github.com/user-attachments/assets/1b923961-55a9-4619-8288-d9ca2588de60" />

#### Deep Reinforcement learning
* Actions and states
  * There are 8 states:
    * x position
    * y position
    * x velocity: $\dot{x}$
    * y velocity: $\dot{y}$
    * angle: $\theta$
    * angular velocity: $\dot{\theta}$
    * left leg contact (1 or 0)
    * right leg contact (1 or 0)
  * There are 4 actions:
    * do nothing
    * fire left orientation engine
    * fire main engine
    * fire right orientation engine. 
  
<img width="1898" alt="Image" src="https://github.com/user-attachments/assets/3b68fa60-923e-4573-9ccf-2564e4156b7c" />
&nbsp;

* Rewards
<img width="1942" alt="Image" src="https://github.com/user-attachments/assets/a6d52a4d-f26d-4c14-9821-cf86e940c5e1" />
&nbsp;

* Policy and Discount factor, the objective of reinforcement learning is to learn a policy that maximizes the expected return, which is the cumulative future reward discounted by a factor $\gamma$ that determines the importance of future rewards compared to immediate rewards.  
$\gamma = 0.985$
&nbsp;

* Deep Reinforcement Learning 
  
<img width="2370" alt="Image" src="https://github.com/user-attachments/assets/64d01fb9-e71f-44b9-a236-db6e8e3014ea" />
&nbsp;

* Building trainning data for deep reinforcement learning.
  
<img width="2272" alt="Image" src="https://github.com/user-attachments/assets/45c1a8d2-2536-4a5a-80c9-87fc348c0277" />
&nbsp;

* Deep Q-learning algorithm

<img width="2350" alt="Image" src="https://github.com/user-attachments/assets/76768634-eb3c-4ac3-88ff-1fc40f9baa67" />
&nbsp;

* Improved neural network architecture, instead of carry 4 inferences from every single state is more efficient ​to train a single neural network to ​output all four of these values simultaneously
<img width="1594" alt="Image" src="https://github.com/user-attachments/assets/abf2d880-29fe-4827-9a6f-73297fd9d653" />
&nbsp;

* Algorightm refinment: ε-greedy policy
<img width="1606" alt="Image" src="https://github.com/user-attachments/assets/9ad2ca3d-69dd-42fa-bf25-2c202ecd823e" />
&nbsp;

#### Labs
* Lab 01: [State Action function](03_unsupervised_learning/03_week/01_lab_reinforcement_learning/state_action_value_function_example.ipynb)
* Lab 02: [Lunar Lander with Deep Q-learning](03_unsupervised_learning/03_week/02_lab_lunar_lander/C3_W3_A1_Assignment.ipynb)