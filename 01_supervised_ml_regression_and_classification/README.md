<h1>01 - Supervised Machine Learning: Regression and Classification</h1>

https://www.coursera.org/learn/machine-learning/home/info

Contents
- [Week 1: Introduction to Machine Learning](#week-1-introduction-to-machine-learning)
  - [Definitions](#definitions)
  - [Intro Supervised Learning](#intro-supervised-learning)
  - [Intro to Unsupervised Learning](#intro-to-unsupervised-learning)
  - [Linear regression](#linear-regression)
  - [Squared Error Cost function](#squared-error-cost-function)
  - [Gradient Descent](#gradient-descent)
  - [Week 1: Labs](#week-1-labs)
- [Week 2: Regression with multiple input variables](#week-2-regression-with-multiple-input-variables)
  - [Multiple variable linear regression](#multiple-variable-linear-regression)
  - [Vectorization](#vectorization)
  - [Gradient Descent with multiple variables](#gradient-descent-with-multiple-variables)
  - [Week 2: Labs](#week-2-labs)
- [Week 3: Classification](#week-3-classification)
  - [Classification with logistic regression](#classification-with-logistic-regression)
  - [Decision boundary](#decision-boundary)
  - [Cost function for logistic regression](#cost-function-for-logistic-regression)
  - [Overfitting and regularization](#overfitting-and-regularization)
  - [Regularization to address overfitting](#regularization-to-address-overfitting)
  - [Week 3: Labs](#week-3-labs)

<hr>

## Week 1: Introduction to Machine Learning
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

### Definitions

* __Machine Learning__ is the science of getting computers to act without being explicitly programmed. It is a subset of artificial intelligence (AI) that focuses on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention.   

* Machine learning algorithms use statistical techniques to enable machines to improve their performance on a specific task over time as they are exposed to more data.
    

### Intro Supervised Learning

* __Supervised Learning__ is a type of machine learning where the algorithm is trained on a __labeled dataset__, which means that each training example is paired with an output label.  
* The goal of supervised learning is to learn a mapping from inputs to outputs, so that the model can make predictions on new, unseen data. Examples of supervised learning tasks include regression (predicting continuous values) and classification (predicting discrete categories).
  
* There are two main types of supervised learning tasks:
  * __Regression__: predicting a continuous value (e.g., price of a house)
  * __Classification__: predicting a discrete category (e.g., whether an email is spam or not)
  
<img width="1392" alt="Image" src="https://github.com/user-attachments/assets/33748e53-666b-4e32-b342-54d7556d211b" />
&nbsp;

<img width="1586" alt="Image" src="https://github.com/user-attachments/assets/fbbb6308-9640-44df-b0ce-23fdbd54b584" />
&nbsp;

### Intro to Unsupervised Learning

* __Unsupervised Learning__ is a type of machine learning where the algorithm is trained on an __unlabeled dataset__, which means that the training examples do not have output labels.  
* The goal of unsupervised learning is to find hidden __patterns__ or structures in the data. 
* Examples of unsupervised learning tasks include clustering (grouping similar data points together) and dimensionality reduction (reducing the number of features in the data while preserving important information).
* There are three main types of unsupervised learning tasks:
  * __Clustering__: grouping data points into clusters of similar examples.
  * __Anomaly detection__: identifying data points that are significantly different from the majority of the data.
  * __Dimensionality reduction__: reducing the number of features in the data while preserving important information.

<img width="1486" alt="Image" src="https://github.com/user-attachments/assets/71d49132-0f73-4baa-b796-fe52a44410b3" />  
&nbsp;

### Linear regression 

* `Linear Regression Model` is a simple machine learning algorithm used for regression tasks, which aims to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.
  
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

<img width="1990" alt="Image" src="https://github.com/user-attachments/assets/91d229ca-6a96-4e74-8865-fa4b0214c129" />
&nbsp;

### Squared Error Cost function 
* The `squared error cost function` is a commonly used cost function for regression problems, which measures the average squared difference between the predicted values and the actual target values.
 
$$J(w,b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

<img width="2001" alt="Image" src="https://github.com/user-attachments/assets/287900f1-fa39-4080-9725-e5c111ab741e" />
&nbsp;


### Gradient Descent
* `Gradient Descent` is an optimization algorithm used to minimize the `cost function` $J(w,b)$ by iteratively updating the parameters $w$ and $b$ in the direction of the negative gradient of the cost function with respect to those parameters. 
* The `learning rate` $\alpha$ determines the size of the steps taken towards the minimum of the cost function. 
* The algorithm continues until convergence, which occurs when the parameters no longer change significantly or when a predetermined number of iterations is reached.
  
<img width="1972" alt="Image" src="https://github.com/user-attachments/assets/43719b95-e84f-47c2-b166-c57369b67e2b" />
&nbsp;

<img width="1992" alt="Image" src="https://github.com/user-attachments/assets/ffcc1e19-42ef-4690-9842-556a0d78d4a2" />
&nbsp;

* `Gradient Descent Algorithm`
```math
\text{repeat until convergence: \{} \\
\quad w := w - \alpha \frac{\partial J(w,b)}{\partial w} \\
\quad b := b - \alpha \frac{\partial J(w,b)}{\partial b} \\
\text{\}}
```

Where the partial derivatives of the cost function with respect to $w$ and $b$ are given by:
```math
\begin{aligned}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right) x^{(i)} \\
\frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
\end{aligned}
```
 
* `Derivative term for` __$\omega$__  

$$\frac{\partial J(w,b)}{\partial w}$$  


### Week 1: Labs
* Lab 01: [Python Jupyter Notebook introduction](01_week/C1_W1_Lab01_Python_Jupyter_Soln.ipynb)
* Lab 02: [Linear regression for one variable](01_week/C1_W1_Lab02_Model_Representation_Soln.ipynb)
* Lab 03: [Cost function for linear regression with one variable](01_week/C1_W1_Lab03_Cost_function_Soln.ipynb)
* Lab 04: [Gradient Descent](01_week/C1_W1_Lab04_Gradient_Descent_Soln.ipynb)

## Week 2: Regression with multiple input variables

>This week, you'll extend linear regression to handle multiple input features. You'll also learn some methods for improving your model's training and performance, such as vectorization, feature scaling, feature engineering and polynomial regression. At the end of the week, you'll get to practice implementing linear regression in code.

__Learning Objectives__
* Use vectorization to implement multiple linear regression
* Use feature scaling, feature engineering, and polynomial regression to * improve model training
* Implement linear regression in code

### Multiple variable linear regression

* `Multiple variable linear regression` is an extension of linear regression that allows for multiple input features. 
* The model can be represented as follows:  
  
$$f_{w,b}(x) = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b$$

* Example: predicting the price of a house based on its size, number of bedrooms, and location. 
  * The input features would be the size, number of bedrooms, and location, and the output would be the price of the house. 
  * The model would learn the weights for each feature to make accurate predictions.  

<img width="1982" alt="Image" src="https://github.com/user-attachments/assets/d94ad870-56b7-4500-9f57-6bddb46e1d66" />

### Vectorization

* `Vectorization` is a technique used to optimize the performance of machine learning algorithms by performing operations on entire arrays or matrices of data, rather than using explicit loops.  
* This allows for faster computations and can significantly reduce the time it takes to train a model.  
* In the context of linear regression, vectorization can be used to compute the cost function and gradients more efficiently, which can speed up the training process.  
  
<img width="1994" alt="Image" src="https://github.com/user-attachments/assets/1a98e322-9a5a-4681-8461-6cebaa57050a" />
&nbsp;

<img width="1984" alt="Image" src="https://github.com/user-attachments/assets/1146320b-87d6-4f72-98ca-b55ea649274e" />

### Gradient Descent with multiple variables
* The gradient descent algorithm can be extended to handle multiple input features by updating the weights for each feature simultaneously. 

* The update rule for each weight $w_j$ is as follows:
```math 
\begin{aligned}
w_j &= w_j - \alpha \frac{\partial J(w,b)}{\partial w_j} \\
\text{where:} \\
\frac{\partial J(w,b)}{\partial w_j}  &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right) x_j^{(i)}
\end{aligned}
```
* The update rule for the bias term $b$ is as follows:
```math
\begin{aligned}
b &= b - \alpha \frac{\partial J(w,b)}{\partial b} \\
\text{where:} \\
\frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
\end{aligned}
```

<img width="1994" alt="Image" src="https://github.com/user-attachments/assets/0c3e98c3-a320-4812-95ce-630079392ff6" />

### Week 2: Labs
* Lab 01: [Python, NumPy and Vectorization](02_week/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb)
* Lab 02: [Multiple Variable Linear Regression](02_week/C1_W2_Lab02_Multiple_Variable_Soln.ipynb)
* Lab 03: [Feature scaling and Learning Rate (Multi-variable)](02_week/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb)
* Lab 04: [Feature Engineering and Polynomial Regression](02_week/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb)
* Lab 05: [Linear Regression using Scikit-Learn, Gradient Descent](02_week/C1_W2_Lab05_Sklearn_GD_Soln.ipynb)
* Lab 06: [Linear Regression using Scikit-Learn, close form solution](02_week/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb)

## Week 3: Classification

>This week, you'll learn the other type of supervised learning, classification. You'll learn how to predict categories using the logistic regression model. You'll learn about the problem of overfitting, and how to handle this problem with a method...

__Learning Objectives__
* Use logistic regression for binary classification
* Implement logistic regression for binary classification
* Address overfitting using regularization, to improve model performance

### Classification with logistic regression
* `Logistic regression` is a type of classification algorithm that is used to predict the probability of a binary outcome (e.g., yes/no, true/false, 1/0). 
* The model can be represented as follows:
  
$$f_{w,b}(x) = \sigma(w^T x + b)$$
  
* where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the __sigmoid function__, which maps any real-valued number into the (0, 1) interval. 
* The output of the logistic regression model can be interpreted as the probability of the positive class (e.g., yes, true, 1). 
* To make a binary prediction, we can use a threshold (e.g., 0.5) to classify the output as either the positive class or the negative class (e.g., no, false, 0). 
* Logistic regression is commonly used in applications such as spam detection, medical diagnosis, and customer churn prediction.  
&nbsp;
<img width="1984" alt="Image" src="https://github.com/user-attachments/assets/8a7edf81-3ab1-4c36-8005-cb48d5a399ab" />
&nbsp;

### Decision boundary

* The decision boundary is the line (or hyperplane) that separates the positive class from the negative class in the feature space.  
* In logistic regression, the decision boundary is defined by the equation $w^T x + b = 0$.  
* Data points on one side of the decision boundary are classified as the positive class, while data points on the other side are classified as the negative class.  
* The position and shape of the decision boundary can be influenced by the weights and bias of the logistic regression model, as well as by the features used in the model.

<img width="1974" alt="Image" src="https://github.com/user-attachments/assets/db9a0e4d-106c-43d0-9664-9d1ef5b83327" />

* Non-linear decision boundary
  * By adding polynomial features to the logistic regression model, we can create a non-linear decision boundary that can capture more complex relationships between the features and the target variable. 
  * This allows the model to fit the data better and make more accurate predictions, especially when the relationship between the features and the target variable is not linear.

<img width="1972" alt="Image" src="https://github.com/user-attachments/assets/27ff21c5-bf58-4b69-93b6-5ecd73305b51" />

### Cost function for logistic regression
* The cost function for linear regression is not suitable for logistic regression because it is not convex and can lead to multiple local minima, making it difficult to optimize. 
* Instead, logistic regression uses a different cost function called the `logistic loss` or `cross-entropy loss`, which is convex and has a single global minimum, making it easier to optimize using gradient descent. 
* The logistic loss function is defined as follows:  
 
$$J(w,b) = -\frac{1}{m} \sum_{i=0}^{m-1} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]$$  

<img width="1978" alt="Image" src="https://github.com/user-attachments/assets/a2108749-888b-4bb7-86ca-4cf99698e441" />
&nbsp;

<img width="1982" alt="Image" src="https://github.com/user-attachments/assets/86ed10a4-5735-49f1-acc1-1f51c6460af6" />

### Overfitting and regularization
* `Overfitting` occurs when a machine learning model learns the training data too well, including the noise and outliers, which can lead to poor performance on new, unseen data. 
* This happens when the model is too complex relative to the amount of training data available.  
* To address overfitting, we can use a technique called `regularization`, which adds a penalty term to the cost function to discourage the model from fitting the noise in the training data. Regularization can help improve the generalization of the model and prevent it from overfitting.

<img width="1952" alt="Image" src="https://github.com/user-attachments/assets/3897d115-92f2-4b08-9d35-fe9f824ba556" />
&nbsp;

<img width="1922" alt="Image" src="https://github.com/user-attachments/assets/35d8ce58-82bb-43e6-9a61-d2021b9dcce9" />
&nbsp;

### Regularization to address overfitting

* Regularization adds a penalty term to the cost function that discourages the model from fitting the noise in the training data. 
* This can help improve the __generalization__ of the model and prevent it from overfitting. 
* The most common types of regularization are L1 regularization (Lasso) and L2 regularization (Ridge). 
  * __L1 regularization__ adds a penalty term proportional to the absolute value of the weights
  * __L2 regularization__ adds a penalty term proportional to the square of the weights. 
* By adjusting the strength of the regularization, we can find a balance between fitting the training data well and generalizing to new data.

<img width="1876" alt="Image" src="https://github.com/user-attachments/assets/892ba313-e646-4dac-8831-bb90e5c99af1" />
&nbsp;

* Cost function with regularization for linear regression  

$$J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$
&nbsp;

* Gradient descent with regularization for linear regression
```math
\begin{aligned}
w_j &= w_j - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right) \\
b &= b - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \right)
\end{aligned}
```
&nbsp;

<img width="1968" alt="Image" src="https://github.com/user-attachments/assets/1376b901-89b6-48a0-89e1-8841f990b3c9" />

* Cost function with regularization for logistic regression  
  
$J(w,b) = -\frac{1}{m} \sum_{i=0}^{m-1} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$
&nbsp;

* Gradient descent with regularization for logistic regression
```math
\begin{aligned}
w_j &= w_j - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right) \\
b &= b - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \right)
\end{aligned}
```

<img width="2019" height="990" alt="Image" src="https://github.com/user-attachments/assets/a1c81036-2c82-4231-ad23-b8ef53297878" />
&nbsp;

<img width="2002" alt="Image" src="https://github.com/user-attachments/assets/5c505719-e9cb-436c-96da-9758a91670b2" />


### Week 3: Labs 
* Lab 01: [Classification](03_week/C1_W3_Lab01_Classification_Soln.ipynb)
* Lab 02: [Logistic Regression](03_week/C1_W3_Lab02_Sigmoid_function_Soln.ipynb)
* Lab 03: [Logistic Regression and Decision Boundary](02_week/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb)
* Lab 04: [Logistic Regression and Logistic Loss](03_week/C1_W3_Lab04_LogisticLoss_Soln.ipynb)
* Lab 05: [Cost Function for Logistic Regression](03_week/C1_W3_Lab05_Cost_Function_Soln.ipynb)
* Lab 06: [Gradient Descent for Logistic Regression](03_week/C1_W3_Lab06_Gradient_Descent_Soln.ipynb)
* Lab 07: [Logistic Regression using Scikit-Learn](03_week/C1_W3_Lab07_Scikit_Learn_Soln.ipynb)
* Lab 08: [Overfitting](03_week/C1_W3_Lab08_Overfitting_Soln.ipynb)
* Lab 09: [Regularized Cost and Gradient](03_week/C1_W3_Lab09_Regularization_Soln.ipynb)
