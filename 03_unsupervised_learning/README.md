# 03 — Unsupervised Learning, Recommenders, Reinforcement Learning

[![Course](https://img.shields.io/badge/Coursera-Course%203%20of%203-0056D2?logo=coursera&logoColor=white)](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/home/welcome)
![Weeks](https://img.shields.io/badge/weeks-3-informational)
![Labs](https://img.shields.io/badge/labs-6-success)
![Stack](https://img.shields.io/badge/TensorFlow%20%7C%20Gym%20%7C%20NumPy-FF6F00?logo=tensorflow&logoColor=white)

> [!NOTE]
> Notes and lab solutions for **Course 3** of the DeepLearning.AI / Stanford Machine Learning Specialization.
> Course home: <https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/home/welcome>

## Contents

- [Week 1: Unsupervised learning](#week-1-unsupervised-learning)
  - [Notes](#notes)
  - [K-means clustering](#k-means-clustering)
  - [Anomaly detection](#anomaly-detection)
  - [Labs](#labs)
- [Week 2: Recommender systems](#week-2-recommender-systems)
  - [Colaborative filtering recommender systems](#colaborative-filtering-recommender-systems)
    - [Collaborative filtering](#collaborative-filtering)
    - [Cost function for CF](#cost-function-for-cf)
    - [Gradient descent for CF](#gradient-descent-for-cf)
    - [Binary labels for CF](#binary-labels-for-cf)
    - [Mean normalization for CF](#mean-normalization-for-cf)
    - [TensorFlow implementation of CF](#tensorflow-implementation-of-cf)
  - [Content-based filtering](#content-based-filtering)
  - [Labs](#labs-1)
- [Week 3 Reinforcement Learning](#week-3-reinforcement-learning)
  - [Reinforcement Learning introduction](#reinforcement-learning-introduction)
  - [State-action value function](#state-action-value-function)
  - [Bellman Equation](#bellman-equation)
  - [Deep Reinforcement learning](#deep-reinforcement-learning)
  - [Labs](#labs-2)

---

## Week 1: Unsupervised learning

> [!NOTE]
> This week, you will learn two key unsupervised learning algorithms: clustering and anomaly detection.

<details>
<summary><b>Learning Objectives</b></summary>

* Implement the k-means clustering algorithm
* Implement the k-means optimization objective
* Initialize the k-means algorithm
* Choose the number of clusters for the k-means algorithm
* Implement an anomaly detection system
* Decide when to use supervised learning vs. anomaly detection
* Implement the centroid update function in k-means
* Implement the function that finds the closest centroids to each point in k-means

</details>

### Notes

* __Unsupervised learning__ — learning from data that is not labeled.

* __Clustering__ — grouping data points into clusters of similar examples.

### K-means clustering

<p align="center">
  <img width="700" alt="K-means clustering overview partitioning data points into groups"
       src="https://github.com/user-attachments/assets/b8ec847c-7e1f-4a7b-bdd1-baa725015e27">
</p>

* K-means is the most popular clustering algorithm.
* It is an iterative algorithm that tries to partition the dataset into K clusters, where each cluster is represented by its __centroid__ (the mean of the points in the cluster).
* The algorithm iteratively assigns each data point to the closest cluster __centroid__ and then updates the centroids based on the mean of the assigned points.

```mermaid
flowchart TD
    A["Randomly initialize<br/>K centroids μ₁ … μ_K"] --> B["<b>Step 1 — Assign</b><br/>each point to its<br/>closest centroid"]
    B --> C["<b>Step 2 — Update</b><br/>move each centroid to the<br/>mean of its assigned points"]
    C --> D{Assignments changed?}
    D -- Yes --> B
    D -- No --> E([Converged])
```

* **Step 1:** Assign each point to its closest centroid to form K clusters.

<p align="center">
  <img width="700" alt="K-means assignment step colouring each point by its nearest centroid"
       src="https://github.com/user-attachments/assets/2891c95d-9e63-42cd-b919-2635ef7b32c2">
</p>

* **Step 2:** Recompute the centroids.

<p align="center">
  <img width="700" alt="K-means update step moving each centroid to the mean of its cluster"
       src="https://github.com/user-attachments/assets/8b21adf1-1603-4cbd-98da-03946260caae">
</p>

* __K-means algorithm__
  * Edge case: if a centroid has no points assigned to it, we can choose a random data point as the new centroid or we can remove that centroid from the algorithm.
  * Convergence: the K-means algorithm is guaranteed to converge to a local minimum, but it may not converge to the global minimum.

> [!TIP]
> Because K-means only finds a *local* minimum, run it 50–1000 times with different random initializations and keep the run with the lowest distortion $J$.

<p align="center">
  <img width="700" alt="Full K-means algorithm pseudocode with assignment and centroid update loops"
       src="https://github.com/user-attachments/assets/67f4f7c0-967e-45e2-b67f-bb5d96b8d8b5">
</p>

* __K-means optimization objective__
  * The K-means algorithm is trying to minimize the following cost function, also called `distortion`:

$$J(c^{(1)}, \ldots, c^{(m)}, \mu_1, \ldots, \mu_K) = \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$$

  * Where:
    * $m$ is the number of training examples.
    * $x^{(i)}$ is the i-th training example.
    * $c^{(i)}$ is the index of the cluster to which the i-th training example is assigned.
    * $\mu_k$ is the centroid of the k-th cluster.
    * $||x^{(i)} - \mu_{c^{(i)}}||^2$ is the squared distance between the i-th training example and the centroid of the cluster to which it is assigned.

<p align="center">
  <img width="700" alt="Distortion cost function decreasing monotonically over K-means iterations"
       src="https://github.com/user-attachments/assets/c48efe10-2eea-4cb7-bde7-8ad645a65462">
</p>

* __Initialization of K-means__
  * Random initialization: randomly select K data points as initial centroids.

* __Choosing the number of clusters K__
  * Elbow method: plot the cost function J as a function of K and look for an "elbow" in the graph where the cost starts to decrease more slowly.

### Anomaly detection

Identifying data points that are significantly different from the majority of the data. This can be useful for tasks such as fraud detection, network security, and quality control.

<p align="center">
  <img width="700" alt="Anomaly detection overview flagging points that fall outside the normal distribution"
       src="https://github.com/user-attachments/assets/52dc6564-51e3-4ef7-a321-84baf41c546e">
</p>

* __Density estimation__

A common approach to anomaly detection is to estimate the probability density function of the data and then flag data points that have a low probability as anomalies.

* __Gaussian distribution__

<p align="center">
  <img width="700" alt="Gaussian bell curve parameterized by mean mu and variance sigma squared"
       src="https://github.com/user-attachments/assets/909f8ae5-9707-4a04-a682-b0487bfd8a41">
</p>

* __Anomaly detection algorithm with one feature__
  * Estimate the parameters $\mu$ and $\sigma^2$ of the Gaussian distribution using the training data.
  * For a new data point $x$, compute the probability density function $p(x)$ using the estimated parameters.
  * Flag $x$ as an anomaly if $p(x) < \epsilon$, where $\epsilon$ is a threshold that you can choose based on your desired false positive rate.

<p align="center">
  <img width="700" alt="Single-feature anomaly detection flagging points where p(x) falls below epsilon"
       src="https://github.com/user-attachments/assets/1b8e293c-eefc-4060-845b-8789328e20cf">
</p>

* __Anomaly detection algorithm with multiple features__
  * Estimate the parameters $\mu$ and $\sigma^2$ of the multivariate Gaussian distribution using the training data.
  * For a new data point $x$, compute the probability density function $p(x)$ using the estimated parameters.
  * Flag $x$ as an anomaly if $p(x) < \epsilon$, where $\epsilon$ is a threshold that you can choose based on your desired false positive rate.

<p align="center">
  <img width="700" alt="Multi-feature anomaly detection multiplying per-feature Gaussian probabilities"
       src="https://github.com/user-attachments/assets/c62f2aa9-7850-4af3-be2e-48fc9ee5a643">
</p>

* __Developing and evaluating an anomaly detection system__
  * Split your data into 3 sets: __training set__, __cross-validation set__, and __test set__.
  * Use the training set to estimate the parameters of the Gaussian distribution.
  * Use the cross-validation set to select the threshold $\epsilon$ that gives you the desired false positive rate.
  * Use the test set to evaluate the performance of your anomaly detection system.
  * Example of dataset for anomaly detection of aircraft engine failure — there are 2 situations:
    * 20 anomalies out of 10 000 data points
    * 2 anomalies out of 10 000 data points

<p align="center">
  <img width="700" alt="Aircraft engine dataset split into train, cross-validation and test sets"
       src="https://github.com/user-attachments/assets/fa15aba9-5168-4f82-86f4-efd24d44b8d3">
</p>

* __Anomaly detection vs supervised learning__

| | Anomaly detection | Supervised learning |
|---|---|---|
| **Positive examples** | Very few (0–20) | Many |
| **Negative examples** | Many | Many |
| **Anomaly types** | Many different, future ones may look unlike any seen so far | Enough examples for the algorithm to learn what positives look like |
| **Typical use** | Fraud detection, manufacturing defects, monitoring machines | Spam, weather prediction, disease classification |

<p align="center">
  <img width="700" alt="Side-by-side comparison of when to use anomaly detection versus supervised learning"
       src="https://github.com/user-attachments/assets/d4968fc7-55d4-4282-9bb5-2a803b1a1f38">
</p>

* __Choosing Features for anomaly detection__
  * The choice of features is crucial for the performance of an anomaly detection system. You should choose features that are relevant to the problem and that can help distinguish between normal and anomalous data points.
  * For example, in the case of aircraft engine failure, you might choose features such as temperature, pressure, and vibration.

### Labs

| # | Lab | What you build |
|:--:|-----|----------------|
| 01 | [K-means clustering](01_week/C3_W1_KMeans_Assignment.ipynb) | Implement assignment/update steps, then compress an image |
| 02 | [Anomaly detection](01_week/C3_W1_Anomaly_Detection.ipynb) | Fit Gaussians to server metrics and tune $\epsilon$ |

---

## Week 2: Recommender systems

> [!NOTE]
> This week you'll build two families of recommender system — collaborative filtering, which learns from the ratings themselves, and content-based filtering, which learns from features of the users and items.

<details>
<summary><b>Learning Objectives</b></summary>

* Implement __collaborative filtering__ recommender systems in TensorFlow.
* Implement deep learning __content based filtering__ using a neural network in TensorFlow.
* Understand ethical considerations in building recommender systems.

</details>

<p align="center">
  <img width="700" alt="Recommender systems overview with a user-item rating matrix"
       src="https://github.com/user-attachments/assets/760dd57b-1830-4aa6-974e-5280429c267f">
</p>

### Colaborative filtering recommender systems

#### Collaborative filtering

* Is a method of making recommendations based on the preferences of similar users.

* The idea is to find users who have similar preferences and then recommend items that those similar users have liked.

<p align="center">
  <img width="700" alt="Movie rating matrix with missing entries to be predicted per user"
       src="https://github.com/user-attachments/assets/edbd5367-6987-4782-ac1d-75004af73ce8">
</p>

#### Cost function for CF

To learn parameters w and b for collaborative filtering, we can use the following cost function.

<p align="center">
  <img width="700" alt="Cost function to learn user parameters w and b from known ratings"
       src="https://github.com/user-attachments/assets/39fc439e-d423-49e2-8abb-94e8ec4eeb9c">
</p>

* Function to learn features x for collaborative filtering, where x represents the features of the items (e.g., movies) that users interact with. In collaborative filtering, we want to learn both the parameters w and b for the users, as well as the features x for the items.
* The cost function for learning features x can be defined as follows, where:
  * $m$ is the number of users.
  * $n$ is the number of items (e.g., movies).
  * $f_{w,b}(x^{(i)})$ is the predicted rating for user i and item j
  * $y^{(i,j)}$ is the actual rating given by user i for movie j.
  * $\lambda$ is a regularization parameter to prevent overfitting.
  * The first term in the cost function measures the difference between the predicted ratings and the actual ratings, while the second term adds a regularization penalty to prevent overfitting by encouraging smaller feature values.

$$J(x) = \frac{1}{2m} \sum_{i=1}^m \sum_{j=1}^n (f_{w,b}(x^{(i)}) - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^n ||x^{(j)}||^2$$

<p align="center">
  <img width="700" alt="Cost function to learn item feature vectors x given user parameters"
       src="https://github.com/user-attachments/assets/88e5be2b-5d03-40f2-8940-95ca658313d1">
</p>

* If we do not have features for the items (e.g., movies), we can learn them from the data using a similar cost function.
* In this case, we would learn the features x for the items, while keeping the parameters w and b fixed. The cost function for learning features x can be defined as follows:

$$J(x) = \frac{1}{2m} \sum_{i=1}^m \sum_{j=1}^n (f_{w,b}(x^{(i)}) - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^n ||x^{(j)}||^2$$

* Where:
  * $m$ is the number of users.
  * $n$ is the number of items (e.g., movies).
  * $f_{w,b}(x^{(i)})$ is the predicted rating for user i and item j
  * $y^{(i,j)}$ is the actual rating given by user i for movie j.
  * $\lambda$ is a regularization parameter to prevent overfitting.
  * The first term in the cost function measures the difference between the predicted ratings and the actual ratings, while the second term adds a regularization penalty to prevent overfitting by encouraging smaller feature values.

<p align="center">
  <img width="700" alt="Learning item features from ratings when no item features are given"
       src="https://github.com/user-attachments/assets/12d2c7f0-0535-4464-90c4-7404a0725a9e">
</p>

* Function to learn both parameters w and b (users on the example), and features x (movies) for __collaborative filtering__.

> [!IMPORTANT]
> **Why is collaborative filtering called "collaborative"?**
> Because it learns the user parameters $w, b$ **and** the item features $x$ *simultaneously* — many users collaborating by rating the same items is what makes both halves recoverable.

* The cost function for learning both parameters w and b, and features x can be defined as follows:

$$J(w,b,x) = \frac{1}{2m} \sum_{i=1}^m \sum_{j=1}^n (f_{w,b}(x^{(i)}) - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i=1}^m (||w^{(i)}||^2 + b^{(i)2}) + \frac{\lambda}{2} \sum_{j=1}^n ||x^{(j)}||^2$$

Where:
* $m$ is the number of users.
* $n$ is the number of items (e.g., movies).
* $f_{w,b}(x^{(i)})$ is the predicted rating for user i and item j.
* $y^{(i,j)}$ is the actual rating given by user i for movie j.
* $\lambda$ is a regularization parameter to prevent overfitting.
* The first term in the cost function measures the difference between the predicted ratings and the actual ratings, while the second and third terms add regularization penalties to prevent overfitting by encouraging smaller parameter values for both the users and the items.

<p align="center">
  <img width="700" alt="Combined collaborative filtering cost function over w, b and x"
       src="https://github.com/user-attachments/assets/20ed170c-3ce2-4054-affa-0c68fc708425">
</p>

#### Gradient descent for CF

* To learn the parameters w and b for the users, and the features x for the items (e.g., movies) simultaneously, we can use gradient descent to minimize the cost function J(w,b,x).
* The update rules for the parameters w and b, and the features x can be derived from the cost function as follows:

$$w^{(i)} := w^{(i)} - \alpha \frac{\partial J(w,b,x)}{\partial w^{(i)}}$$
$$b^{(i)} := b^{(i)} - \alpha \frac{\partial J(w,b,x)}{\partial b^{(i)}}$$
$$x^{(j)} := x^{(j)} - \alpha \frac{\partial J(w,b,x)}{\partial x^{(j)}}$$

* Where:
  * $\alpha$ is the learning rate.
  * The gradients can be computed using the cost function J(w,b,x) and the predicted ratings $f_{w,b}(x^{(i)})$.
  * The update rules will iteratively adjust the parameters w and b for the users, and the features x for the items until convergence, where the cost function J(w,b,x) is minimized.

<p align="center">
  <img width="700" alt="Gradient descent updating user parameters and item features together"
       src="https://github.com/user-attachments/assets/127e838e-3688-44f7-94ee-7c3b93ace7d3">
</p>

#### Binary labels for CF

* __Binary labels__ can be used in collaborative filtering when we only have information about whether a user liked an item or not, rather than the actual rating, e.g., 1 for liked and 0 for not liked.
* Previously, we have been working with ratings as labels, which are continuous values. However, in many cases, we only have binary labels, such as whether a user liked an item or not. In this case, we can use a different cost function that is more appropriate for binary labels.

* Cost function for binary labels

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]$$

  * Where $f_{w,b}(x^{(i)})$ is the predicted probability that user $i$ will like item $j$, and $y^{(i)}$ is the actual label (1 if the user liked the item, 0 otherwise).
  * This cost function is known as the __binary cross-entropy loss__.
  * $f_{w,b}(x^{(i)})$ can be calculated using the __sigmoid function__, which maps the output of the linear model to a value between 0 and 1, representing the predicted probability of the user liking the item:

$$f_{w,b}(x^{(i)}) = \sigma(w^T x^{(i)} + b) = \frac{1}{1 + e^{-(w^T x^{(i)} + b)}}$$

<p align="center">
  <img width="700" alt="Binary label examples such as clicked, favourited or purchased instead of star ratings"
       src="https://github.com/user-attachments/assets/a6ebcd20-9cb5-480a-987a-00ceef2fb92c">
</p>

#### Mean normalization for CF

* To improve the performance of collaborative filtering, we can apply __mean normalization__ to the ratings. This involves subtracting the mean rating for each item from the ratings before training the model. This helps to account for differences in user preferences and can lead to better recommendations.

* Normalization by rows (users) or by columns (items): when a new customer is added use normalization by rows (example below); when a new movie is added use normalization by columns.

<p align="center">
  <img width="700" alt="Mean normalization of the rating matrix so a new user gets average predictions"
       src="https://github.com/user-attachments/assets/419a6f25-ded4-4305-a782-e76ae12b1b09">
</p>

#### TensorFlow implementation of CF

* In TensorFlow, we can implement collaborative filtering using the __Keras__ API.
* We can define a model that takes the user and item features as input and outputs the predicted rating.
* We can then compile the model with the appropriate loss function (e.g., binary cross-entropy for binary labels) and optimizer (e.g., Adam), and train the model on the training data.
* After training, we can use the model to make predictions for new user-item pairs and generate recommendations based on those predictions.

* Gradient descent reminder from previous weeks:

<p align="center">
  <img width="700" alt="Gradient descent recap before introducing automatic differentiation"
       src="https://github.com/user-attachments/assets/805af192-f9bd-4262-b296-ee1c983ae229">
</p>

* __Auto Diff__ in TensorFlow: automatically compute derivatives for gradient descent.

<p align="center">
  <img width="700" alt="TensorFlow GradientTape computing derivatives automatically inside the training loop"
       src="https://github.com/user-attachments/assets/2623b0a5-a71e-466d-8f48-1c98a9183efc">
</p>

* Finding related items — item $k$ is similar to item $i$ when $||x^{(k)} - x^{(i)}||^2$ is small.

<p align="center">
  <img width="700" alt="Finding related items by smallest squared distance between learned feature vectors"
       src="https://github.com/user-attachments/assets/9a07d7ec-927a-4871-a91f-c251843a7476">
</p>

> [!WARNING]
> **Limitations of collaborative filtering**
> * **Cold start problem** — when a new user or item is added to the system, there is no historical data to make recommendations from.
> * **User side information** — collaborative filtering does not take into account features of the users, such as demographics or preferences, which can be useful for making recommendations.

### Content-based filtering

* Finding similar items based on the features of the items themselves, rather than the preferences of similar users.

* Collaborative filtering vs content-based filtering:
  * __Collaborative filtering:__ makes recommendations based on the preferences of similar users.
  * __Content-based filtering:__ makes recommendations based on the features of the items themselves.

<p align="center">
  <img width="700" alt="Collaborative filtering compared with content-based filtering inputs"
       src="https://github.com/user-attachments/assets/4a7e517e-0437-4e13-94e6-bea8754c841d">
</p>

* Content-based neural network architecture — two towers, one for the user and one for the item, whose outputs are combined with a dot product.

```mermaid
flowchart LR
    XU["User features<br/>x_u<br/><i>age, gender, ratings…</i>"] --> NU["User network"] --> VU["v_u"]
    XM["Item features<br/>x_m<br/><i>year, genre, avg rating…</i>"] --> NM["Item network"] --> VM["v_m"]
    VU --> DOT(("v_u · v_m"))
    VM --> DOT
    DOT --> P["Predicted rating"]
```

<p align="center">
  <img width="700" alt="Two-tower content-based neural network with user and item networks joined by a dot product"
       src="https://github.com/user-attachments/assets/77ad8c58-6de1-4fb6-8cf3-dee12a9808e1">
</p>

* Content-based filtering cost function

<p align="center">
  <img width="700" alt="Cost function training both user and item networks on observed ratings"
       src="https://github.com/user-attachments/assets/2a323365-604c-4b0f-8de9-e4d224a82c34">
</p>

* Finding similar items using content-based filtering

<p align="center">
  <img width="700" alt="Finding similar items by distance between item network output vectors"
       src="https://github.com/user-attachments/assets/50944ae0-96b3-4bf8-8c42-a17a9e396c44">
</p>

* Recommending from a large catalogue — done in two steps:
  * __Retrieval__: generate a large list of plausible candidates (e.g., similar items to the user's recent picks, top items in their favourite genres).
  * __Ranking__: score that shorter list with the learned model and display the top results.

### Labs

| # | Lab | What you build |
|:--:|-----|----------------|
| 01 | [Collaborative Filtering Recommender Systems](02_week/C3_W2_colaborative_filtering/C3_W2_Collaborative_RecSys_Assignment.ipynb) | Movie recommender learning $w$, $b$ and $x$ with `GradientTape` |
| 02 | [Content-Based Filtering Recommender Systems](02_week/C2_W2_content_based_filtering/C3_W2_RecSysNN_Assignment.ipynb) | Two-tower neural network on the MovieLens dataset |

---

## Week 3 Reinforcement Learning

> [!NOTE]
> This week, you will learn about reinforcement learning, and build a deep Q-learning neural network in order to land a virtual lunar lander on Mars!

<details>
<summary><b>Learning Objectives</b></summary>

* Understand key terms such as return, state, action, and policy as it applies to reinforcement learning
* Understand the Bellman equations
* Understand the state-action value function
* Understand continuous state spaces
* Build a deep Q-learning network

</details>

### Reinforcement Learning introduction

```mermaid
flowchart LR
    A["Agent"] -- "action a" --> E["Environment"]
    E -- "state s′" --> A
    E -- "reward r" --> A
```

<p align="center">
  <img width="700" alt="Reinforcement learning setup with an agent taking actions in an environment for rewards"
       src="https://github.com/user-attachments/assets/807323d4-2d9c-48c2-b3e1-bd0a32199e6f">
</p>

<p align="center">
  <img width="700" alt="Mars rover example showing states, actions, rewards and terminal states"
       src="https://github.com/user-attachments/assets/d8dd8ae3-a27a-4a67-8712-8b348f256647">
</p>

### State-action value function

* State action value function: $Q(s,a)$ represents the expected return (cumulative future reward) of taking action a in state s and following a certain policy thereafter. The goal of reinforcement learning is to learn an optimal policy that maximizes the expected return, which can be achieved by learning the optimal state-action value function $Q^*(s,a)$.

<p align="center">
  <img width="700" alt="State-action value function Q(s,a) tabulated for each state and action"
       src="https://github.com/user-attachments/assets/3bc07fcf-59ac-467e-919e-df7c29f590e7">
</p>

### Bellman Equation

* The Bellman equation is a fundamental equation in reinforcement learning that describes the relationship between the value of a state and the values of its successor states. It can be expressed as follows:

$$Q(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s',a') | s, a]$$

where $Q(s,a)$ is the state-action value function, $r$ is the reward received after taking action $a$ in state $s$, $\gamma$ is the discount factor that determines the importance of future rewards, and $s'$ is the next state resulting from taking action $a$ in state $s$. The Bellman equation states that the value of taking action $a$ in state $s$ is equal to the expected reward plus the discounted value of the best action in the next state $s'$. This equation is used to derive the optimal policy and to update the state-action value function during the learning process.

> [!TIP]
> Read it as two pieces: **the reward you get right now** ($r$) plus **the discounted best you can do from wherever you land next** ($\gamma \max_{a'} Q(s',a')$).

<p align="center">
  <img width="700" alt="Bellman equation broken into immediate reward and discounted future return"
       src="https://github.com/user-attachments/assets/1b923961-55a9-4619-8288-d9ca2588de60">
</p>

### Deep Reinforcement learning

* Actions and states for the lunar lander:

| 8 states | 4 actions |
|----------|-----------|
| $x$ position | do nothing |
| $y$ position | fire left orientation engine |
| $x$ velocity $\dot{x}$ | fire main engine |
| $y$ velocity $\dot{y}$ | fire right orientation engine |
| angle $\theta$ | |
| angular velocity $\dot{\theta}$ | |
| left leg contact (1 or 0) | |
| right leg contact (1 or 0) | |

<p align="center">
  <img width="700" alt="Lunar lander state vector and the four available engine actions"
       src="https://github.com/user-attachments/assets/3b68fa60-923e-4573-9ccf-2564e4156b7c">
</p>

* Rewards

<p align="center">
  <img width="700" alt="Lunar lander reward function for landing, crashing and fuel usage"
       src="https://github.com/user-attachments/assets/a6d52a4d-f26d-4c14-9821-cf86e940c5e1">
</p>

* Policy and discount factor: the objective of reinforcement learning is to learn a policy that maximizes the expected return, which is the cumulative future reward discounted by a factor $\gamma$ that determines the importance of future rewards compared to immediate rewards — here $\gamma = 0.985$.

* Deep Reinforcement Learning

<p align="center">
  <img width="700" alt="Neural network approximating Q(s,a) in place of a lookup table"
       src="https://github.com/user-attachments/assets/64d01fb9-e71f-44b9-a236-db6e8e3014ea">
</p>

* Building training data for deep reinforcement learning.

<p align="center">
  <img width="700" alt="Turning stored experience tuples into supervised training examples x and y"
       src="https://github.com/user-attachments/assets/45c1a8d2-2536-4a5a-80c9-87fc348c0277">
</p>

* Deep Q-learning algorithm

```mermaid
flowchart TD
    A["Initialize Q-network<br/>with random weights"] --> B["Take actions in the lunar lander<br/>store (s, a, R(s), s′) in replay buffer"]
    B --> C["Sample a mini-batch<br/>of stored tuples"]
    C --> D["Build training set:<br/>x = (s, a)<br/>y = R(s) + γ max Q(s′, a′)"]
    D --> E["Train Q_new so that<br/>Q_new(s,a) ≈ y"]
    E --> F["Set Q := Q_new"]
    F --> B
```

<p align="center">
  <img width="700" alt="Deep Q-learning algorithm with experience replay buffer and target updates"
       src="https://github.com/user-attachments/assets/76768634-eb3c-4ac3-88ff-1fc40f9baa67">
</p>

* Improved neural network architecture: instead of running 4 separate inferences for every state, it is more efficient to train a single neural network that outputs all four Q-values simultaneously.

<p align="center">
  <img width="700" alt="Single network outputting all four Q-values at once instead of one per inference"
       src="https://github.com/user-attachments/assets/abf2d880-29fe-4827-9a6f-73297fd9d653">
</p>

* Algorithm refinement: ε-greedy policy — with probability $1-\epsilon$ pick the action that maximizes $Q(s,a)$ (*exploitation*), and with probability $\epsilon$ pick a random action (*exploration*).

<p align="center">
  <img width="700" alt="Epsilon-greedy policy balancing exploitation of the best known action with random exploration"
       src="https://github.com/user-attachments/assets/9ad2ca3d-69dd-42fa-bf25-2c202ecd823e">
</p>

### Labs

| # | Lab | What you build |
|:--:|-----|----------------|
| 01 | [State Action value function](03_week/01_lab_reinforcement_learning/state_action_value_function_example.ipynb) | Interactive $Q(s,a)$ explorer on the Mars rover |
| 02 | [Lunar Lander with Deep Q-learning](03_week/02_lab_lunar_lander/C3_W3_A1_Assignment.ipynb) | Train a DQN with experience replay to land the craft |

---

<div align="center">

[⬅️ Course 02 — Advanced Learning Algorithms](../02_advanced_learning_algorithms/README.md) · [🏠 **Home**](../README.md)

</div>
