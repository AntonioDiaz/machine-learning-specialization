# 01 — Supervised Machine Learning: Regression and Classification

[![Course](https://img.shields.io/badge/Coursera-Course%201%20of%203-0056D2?logo=coursera&logoColor=white)](https://www.coursera.org/learn/machine-learning/home/info)
![Weeks](https://img.shields.io/badge/weeks-3-informational)
![Labs](https://img.shields.io/badge/labs-19-success)
![Stack](https://img.shields.io/badge/NumPy%20%7C%20scikit--learn%20%7C%20matplotlib-013243?logo=numpy&logoColor=white)

> [!NOTE]
> Notes and lab solutions for **Course 1** of the DeepLearning.AI / Stanford Machine Learning Specialization.
> Course home: <https://www.coursera.org/learn/machine-learning/home/info>

## Contents

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

---

## Week 1: Introduction to Machine Learning

> [!NOTE]
> Welcome to the Machine Learning Specialization! You're joining millions of others who have taken either this or the original course, which led to the founding of Coursera, and has helped millions of other learners, like you, take a look at the exciting world of machine learning!

<details>
<summary><b>Learning Objectives</b></summary>

* Define machine learning
* Define supervised learning
* Define unsupervised learning
* Write and run Python code in Jupyter Notebooks
* Define a regression model
* Implement and visualize a cost function
* Implement gradient descent
* Optimize a regression model using gradient descent

</details>

### Definitions

* __Machine Learning__ is the science of getting computers to act without being explicitly programmed. It is a subset of artificial intelligence (AI) that focuses on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention.

* Machine learning algorithms use statistical techniques to enable machines to improve their performance on a specific task over time as they are exposed to more data.

```mermaid
flowchart TD
    ML[Machine Learning]
    ML --> SUP[Supervised Learning<br/><i>labeled data</i>]
    ML --> UNS[Unsupervised Learning<br/><i>unlabeled data</i>]
    SUP --> REG["Regression<br/>predict a continuous value<br/>(house price)"]
    SUP --> CLS["Classification<br/>predict a discrete category<br/>(spam / not spam)"]
    UNS --> CLU[Clustering]
    UNS --> ANO[Anomaly detection]
    UNS --> DIM[Dimensionality reduction]
```

### Intro Supervised Learning

* __Supervised Learning__ is a type of machine learning where the algorithm is trained on a __labeled dataset__, which means that each training example is paired with an output label.
* The goal of supervised learning is to learn a mapping from inputs to outputs, so that the model can make predictions on new, unseen data. Examples of supervised learning tasks include regression (predicting continuous values) and classification (predicting discrete categories).

* There are two main types of supervised learning tasks:
  * __Regression__: predicting a continuous value (e.g., price of a house)
  * __Classification__: predicting a discrete category (e.g., whether an email is spam or not)

<p align="center">
  <img width="700" alt="Supervised learning: input x mapped to output label y, with example applications"
       src="https://github.com/user-attachments/assets/33748e53-666b-4e32-b342-54d7556d211b">
</p>

<p align="center">
  <img width="700" alt="Regression predicts continuous values, classification predicts discrete categories"
       src="https://github.com/user-attachments/assets/fbbb6308-9640-44df-b0ce-23fdbd54b584">
</p>

### Intro to Unsupervised Learning

* __Unsupervised Learning__ is a type of machine learning where the algorithm is trained on an __unlabeled dataset__, which means that the training examples do not have output labels.
* The goal of unsupervised learning is to find hidden __patterns__ or structures in the data.
* Examples of unsupervised learning tasks include clustering (grouping similar data points together) and dimensionality reduction (reducing the number of features in the data while preserving important information).
* There are three main types of unsupervised learning tasks:
  * __Clustering__: grouping data points into clusters of similar examples.
  * __Anomaly detection__: identifying data points that are significantly different from the majority of the data.
  * __Dimensionality reduction__: reducing the number of features in the data while preserving important information.

<p align="center">
  <img width="700" alt="Unsupervised learning finds structure in unlabeled data: clustering, anomaly detection, dimensionality reduction"
       src="https://github.com/user-attachments/assets/71d49132-0f73-4baa-b796-fe52a44410b3">
</p>

### Linear regression

* `Linear Regression Model` is a simple machine learning algorithm used for regression tasks, which aims to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.

$$f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

<p align="center">
  <img width="700" alt="Linear regression fitting a straight line through house size vs price training data"
       src="https://github.com/user-attachments/assets/91d229ca-6a96-4e74-8865-fa4b0214c129">
</p>

### Squared Error Cost function

* The `squared error cost function` is a commonly used cost function for regression problems, which measures the average squared difference between the predicted values and the actual target values.

$$J(w,b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

<p align="center">
  <img width="700" alt="Squared error cost function J(w,b) plotted as a convex bowl-shaped surface"
       src="https://github.com/user-attachments/assets/287900f1-fa39-4080-9725-e5c111ab741e">
</p>

### Gradient Descent

* `Gradient Descent` is an optimization algorithm used to minimize the `cost function` $J(w,b)$ by iteratively updating the parameters $w$ and $b$ in the direction of the negative gradient of the cost function with respect to those parameters.
* The `learning rate` $\alpha$ determines the size of the steps taken towards the minimum of the cost function.
* The algorithm continues until convergence, which occurs when the parameters no longer change significantly or when a predetermined number of iterations is reached.

```mermaid
flowchart TD
    A["Initialize w, b"] --> B["Compute cost J(w,b)"]
    B --> C["Compute gradients ∂J/∂w, ∂J/∂b"]
    C --> D["Update simultaneously:<br/>w := w − α ∂J/∂w<br/>b := b − α ∂J/∂b"]
    D --> E{Converged?}
    E -- No --> B
    E -- Yes --> F([Minimum reached])
```

> [!IMPORTANT]
> The updates to $w$ and $b$ must be **simultaneous** — compute both gradients from the *old* parameter values before assigning either one.

<p align="center">
  <img width="700" alt="Gradient descent taking steps downhill on the cost surface toward the minimum"
       src="https://github.com/user-attachments/assets/43719b95-e84f-47c2-b166-c57369b67e2b">
</p>

<p align="center">
  <img width="700" alt="Effect of the learning rate alpha: too small converges slowly, too large overshoots and diverges"
       src="https://github.com/user-attachments/assets/ffcc1e19-42ef-4690-9842-556a0d78d4a2">
</p>

* `Gradient Descent Algorithm`

```math
\begin{aligned}
& \text{repeat until convergence} \; \lbrace \\
& \qquad w := w - \alpha \frac{\partial J(w,b)}{\partial w} \\
& \qquad b := b - \alpha \frac{\partial J(w,b)}{\partial b} \\
& \rbrace
\end{aligned}
```

* Where the partial derivatives of the cost function with respect to $w$ and $b$ are given by:

```math
\begin{aligned}
\frac{\partial J(w,b)}{\partial w} &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right) x^{(i)} \\[8pt]
\frac{\partial J(w,b)}{\partial b} &= \frac{1}{m} \sum_{i = 0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
\end{aligned}
```

* __Simultaneous update in code__ — the sequential version feeds the already-updated $w$ into the gradient for $b$, so it descends a slightly different surface than the one you meant to:

| ✅ Correct — simultaneous | ❌ Wrong — sequential |
|---|---|
| `tmp_w = w - alpha * dj_dw` | `w = w - alpha * dj_dw` |
| `tmp_b = b - alpha * dj_db` | `b = b - alpha * dj_db`&nbsp;← uses the **new** `w` |
| `w = tmp_w` | |
| `b = tmp_b` | |

### Week 1: Labs

| # | Lab | What you build |
|:--:|-----|----------------|
| 01 | [Python Jupyter Notebook introduction](01_week/C1_W1_Lab01_Python_Jupyter_Soln.ipynb) | Get comfortable with markdown and code cells |
| 02 | [Linear regression for one variable](01_week/C1_W1_Lab02_Model_Representation_Soln.ipynb) | Represent $f_{w,b}(x) = wx + b$ and plot it against the data |
| 03 | [Cost function for linear regression](01_week/C1_W1_Lab03_Cost_function_Soln.ipynb) | Implement and visualize $J(w,b)$ as a contour and surface plot |
| 04 | [Gradient Descent](01_week/C1_W1_Lab04_Gradient_Descent_Soln.ipynb) | Code the update rule and watch the parameters converge |

---

## Week 2: Regression with multiple input variables

> [!NOTE]
> This week, you'll extend linear regression to handle multiple input features. You'll also learn some methods for improving your model's training and performance, such as vectorization, feature scaling, feature engineering and polynomial regression. At the end of the week, you'll get to practice implementing linear regression in code.

<details>
<summary><b>Learning Objectives</b></summary>

* Use vectorization to implement multiple linear regression
* Use feature scaling, feature engineering, and polynomial regression to improve model training
* Implement linear regression in code

</details>

### Multiple variable linear regression

* `Multiple variable linear regression` is an extension of linear regression that allows for multiple input features.
* The model can be represented as follows:

$$f_{w,b}(x) = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b$$

* Example: predicting the price of a house based on its size, number of bedrooms, and location.
  * The input features would be the size, number of bedrooms, and location, and the output would be the price of the house.
  * The model would learn the weights for each feature to make accurate predictions.

<p align="center">
  <img width="700" alt="Multiple linear regression: house price predicted from size, bedrooms, floors and age"
       src="https://github.com/user-attachments/assets/d94ad870-56b7-4500-9f57-6bddb46e1d66">
</p>

### Vectorization

* `Vectorization` is a technique used to optimize the performance of machine learning algorithms by performing operations on entire arrays or matrices of data, rather than using explicit loops.
* This allows for faster computations and can significantly reduce the time it takes to train a model.
* In the context of linear regression, vectorization can be used to compute the cost function and gradients more efficiently, which can speed up the training process.

> [!TIP]
> `np.dot(w, x) + b` is not just shorter than a `for` loop — NumPy runs the multiplications in parallel on specialized hardware, so the gap widens dramatically as the number of features grows.

<p align="center">
  <img width="700" alt="Vectorized NumPy dot product compared with an explicit for-loop implementation"
       src="https://github.com/user-attachments/assets/1a98e322-9a5a-4681-8461-6cebaa57050a">
</p>

<p align="center">
  <img width="700" alt="Vectorization performing all feature multiplications in parallel in a single step"
       src="https://github.com/user-attachments/assets/1146320b-87d6-4f72-98ca-b55ea649274e">
</p>

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

<p align="center">
  <img width="700" alt="Gradient descent update rules for multiple variable linear regression"
       src="https://github.com/user-attachments/assets/0c3e98c3-a320-4812-95ce-630079392ff6">
</p>

### Week 2: Labs

| # | Lab | What you build |
|:--:|-----|----------------|
| 01 | [Python, NumPy and Vectorization](02_week/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb) | Vector/matrix operations and a timing comparison against loops |
| 02 | [Multiple Variable Linear Regression](02_week/C1_W2_Lab02_Multiple_Variable_Soln.ipynb) | Extend the model and gradient descent to $n$ features |
| 03 | [Feature scaling and Learning Rate](02_week/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb) | Z-score normalization and how $\alpha$ affects convergence |
| 04 | [Feature Engineering and Polynomial Regression](02_week/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb) | Fit curves by engineering $x^2$, $x^3$ features |
| 05 | [Linear Regression using Scikit-Learn, Gradient Descent](02_week/C1_W2_Lab05_Sklearn_GD_Soln.ipynb) | The same model with `SGDRegressor` |
| 06 | [Linear Regression using Scikit-Learn, closed form](02_week/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb) | The normal equation via `LinearRegression` |

---

## Week 3: Classification

> [!NOTE]
> This week, you'll learn the other type of supervised learning, classification. You'll learn how to predict categories using the logistic regression model. You'll learn about the problem of overfitting, and how to handle this problem with a method called regularization.

<details>
<summary><b>Learning Objectives</b></summary>

* Use logistic regression for binary classification
* Implement logistic regression for binary classification
* Address overfitting using regularization, to improve model performance

</details>

### Classification with logistic regression

* `Logistic regression` is a type of classification algorithm that is used to predict the probability of a binary outcome (e.g., yes/no, true/false, 1/0).
* The model can be represented as follows:

$$f_{w,b}(x) = \sigma(w^T x + b)$$

* where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the __sigmoid function__, which maps any real-valued number into the (0, 1) interval.
* The output of the logistic regression model can be interpreted as the probability of the positive class (e.g., yes, true, 1).
* To make a binary prediction, we can use a threshold (e.g., 0.5) to classify the output as either the positive class or the negative class (e.g., no, false, 0).
* Logistic regression is commonly used in applications such as spam detection, medical diagnosis, and customer churn prediction.

<p align="center">
  <img width="700" alt="Sigmoid function squashing any real value into the interval between 0 and 1"
       src="https://github.com/user-attachments/assets/8a7edf81-3ab1-4c36-8005-cb48d5a399ab">
</p>

### Decision boundary

* The decision boundary is the line (or hyperplane) that separates the positive class from the negative class in the feature space.
* In logistic regression, the decision boundary is defined by the equation $w^T x + b = 0$.
* Data points on one side of the decision boundary are classified as the positive class, while data points on the other side are classified as the negative class.
* The position and shape of the decision boundary can be influenced by the weights and bias of the logistic regression model, as well as by the features used in the model.

<p align="center">
  <img width="700" alt="Linear decision boundary separating two classes in a two-feature plot"
       src="https://github.com/user-attachments/assets/db9a0e4d-106c-43d0-9664-9d1ef5b83327">
</p>

* Non-linear decision boundary
  * By adding polynomial features to the logistic regression model, we can create a non-linear decision boundary that can capture more complex relationships between the features and the target variable.
  * This allows the model to fit the data better and make more accurate predictions, especially when the relationship between the features and the target variable is not linear.

<p align="center">
  <img width="700" alt="Circular non-linear decision boundary produced by adding polynomial features"
       src="https://github.com/user-attachments/assets/27ff21c5-bf58-4b69-93b6-5ecd73305b51">
</p>

### Cost function for logistic regression

> [!WARNING]
> The squared error cost function is **not** suitable for logistic regression — it produces a non-convex surface full of local minima that gradient descent can get stuck in.

* Instead, logistic regression uses a different cost function called the `logistic loss` or `cross-entropy loss`, which is convex and has a single global minimum, making it easier to optimize using gradient descent.
* The logistic loss function is defined as follows:

$$J(w,b) = -\frac{1}{m} \sum_{i=0}^{m-1} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]$$

<p align="center">
  <img width="700" alt="Logistic loss curves for y=1 and y=0, penalizing confident wrong predictions heavily"
       src="https://github.com/user-attachments/assets/a2108749-888b-4bb7-86ca-4cf99698e441">
</p>

<p align="center">
  <img width="700" alt="Convex logistic cost surface compared with the non-convex squared error surface"
       src="https://github.com/user-attachments/assets/86ed10a4-5735-49f1-acc1-1f51c6460af6">
</p>

### Overfitting and regularization

* `Overfitting` occurs when a machine learning model learns the training data too well, including the noise and outliers, which can lead to poor performance on new, unseen data.
* This happens when the model is too complex relative to the amount of training data available.
* To address overfitting, we can use a technique called `regularization`, which adds a penalty term to the cost function to discourage the model from fitting the noise in the training data. Regularization can help improve the generalization of the model and prevent it from overfitting.

```mermaid
flowchart LR
    U["Underfit<br/><b>high bias</b><br/>too simple"]
    J["Just right<br/><b>generalizes</b>"]
    O["Overfit<br/><b>high variance</b><br/>too complex"]
    U --- J --- O
    O -.->|collect more data| J
    O -.->|use fewer features| J
    O -.->|regularization: increase λ| J
```

<p align="center">
  <img width="700" alt="Underfitting, good fit and overfitting shown on regression curves"
       src="https://github.com/user-attachments/assets/3897d115-92f2-4b08-9d35-fe9f824ba556">
</p>

<p align="center">
  <img width="700" alt="Underfitting and overfitting shown on classification decision boundaries"
       src="https://github.com/user-attachments/assets/35d8ce58-82bb-43e6-9a61-d2021b9dcce9">
</p>

### Regularization to address overfitting

* Regularization adds a penalty term to the cost function that discourages the model from fitting the noise in the training data.
* This can help improve the __generalization__ of the model and prevent it from overfitting.
* The most common types of regularization are L1 regularization (Lasso) and L2 regularization (Ridge).
  * __L1 regularization__ adds a penalty term proportional to the absolute value of the weights
  * __L2 regularization__ adds a penalty term proportional to the square of the weights.
* By adjusting the strength of the regularization, we can find a balance between fitting the training data well and generalizing to new data.

<p align="center">
  <img width="700" alt="Regularization term shrinking the weights to produce a smoother fit"
       src="https://github.com/user-attachments/assets/892ba313-e646-4dac-8831-bb90e5c99af1">
</p>

> [!TIP]
> By convention the bias term $b$ is **not** regularized — the penalty sum runs over $j = 1 \ldots n$ only.

* Cost function with regularization for linear regression

$$J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

* Gradient descent with regularization for linear regression

```math
\begin{aligned}
w_j &= w_j - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right) \\
b &= b - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \right)
\end{aligned}
```

<p align="center">
  <img width="700" alt="Regularized gradient descent update rule for linear regression"
       src="https://github.com/user-attachments/assets/1376b901-89b6-48a0-89e1-8841f990b3c9">
</p>

* Cost function with regularization for logistic regression

$$J(w,b) = -\frac{1}{m} \sum_{i=0}^{m-1} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

* Gradient descent with regularization for logistic regression

```math
\begin{aligned}
w_j &= w_j - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right) \\
b &= b - \alpha \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \right)
\end{aligned}
```

<p align="center">
  <img width="700" alt="Regularized cost function for logistic regression"
       src="https://github.com/user-attachments/assets/a1c81036-2c82-4231-ad23-b8ef53297878">
</p>

<p align="center">
  <img width="700" alt="Regularized gradient descent update rule for logistic regression"
       src="https://github.com/user-attachments/assets/5c505719-e9cb-436c-96da-9758a91670b2">
</p>

### Week 3: Labs

| # | Lab | What you build |
|:--:|-----|----------------|
| 01 | [Classification](03_week/C1_W3_Lab01_Classification_Soln.ipynb) | Why linear regression fails on categorical targets |
| 02 | [Sigmoid function](03_week/C1_W3_Lab02_Sigmoid_function_Soln.ipynb) | Plot $\sigma(z)$ and read it as a probability |
| 03 | [Decision Boundary](03_week/C1_W3_Lab03_Decision_Boundary_Soln.ipynb) | Visualize the boundary $w^T x + b = 0$ |
| 04 | [Logistic Loss](03_week/C1_W3_Lab04_LogisticLoss_Soln.ipynb) | See why squared error is non-convex here |
| 05 | [Cost Function for Logistic Regression](03_week/C1_W3_Lab05_Cost_Function_Soln.ipynb) | Implement the cross-entropy cost |
| 06 | [Gradient Descent for Logistic Regression](03_week/C1_W3_Lab06_Gradient_Descent_Soln.ipynb) | Train the classifier and animate the boundary |
| 07 | [Logistic Regression using Scikit-Learn](03_week/C1_W3_Lab07_Scikit_Learn_Soln.ipynb) | The same model in a few lines with `LogisticRegression` |
| 08 | [Overfitting](03_week/C1_W3_Lab08_Overfitting_Soln.ipynb) | Interactive demo of bias vs. variance |
| 09 | [Regularized Cost and Gradient](03_week/C1_W3_Lab09_Regularization_Soln.ipynb) | Add the L2 penalty to both cost and gradient |

---

<div align="center">

[🏠 **Home**](../README.md) · [Course 02 — Advanced Learning Algorithms ➡️](../02_advanced_learning_algorithms/README.md)

</div>
