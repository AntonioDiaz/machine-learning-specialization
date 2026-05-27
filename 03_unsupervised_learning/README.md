<h1>03 - Unsupervised Learning, Recommenders, Reinforcement Learning</h1>

https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/home/welcome

Contents
- [Week 1: Unsupervised learning](#week-1-unsupervised-learning)
  - [Notes](#notes)
  - [K-means clustering](#k-means-clustering)
  - [Anomaly detection](#anomaly-detection)
  - [Labs](#labs)
- [Week 2: Recommender systems](#week-2-recommender-systems)
  - [Colaborative filtering recommender systems](#colaborative-filtering-recommender-systems)
  - [Mean normalization for collaborative filtering](#mean-normalization-for-collaborative-filtering)
  - [TensorFlow implementation of collaborative filtering](#tensorflow-implementation-of-collaborative-filtering)
  - [Content-based filtering](#content-based-filtering)
  - [Labs](#labs-1)
- [Week 3 Reinforcement Learning](#week-3-reinforcement-learning)
  - [Reinforcement Learning introduction](#reinforcement-learning-introduction)
  - [State-action value function](#state-action-value-function)
  - [Bellman Equation](#bellman-equation)
  - [Deep Reinforcement learning](#deep-reinforcement-learning)
  - [Labs](#labs-2)

<hr>

## Week 1: Unsupervised learning
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

### Notes
* __Unsupervised learning__ learning from data that is not labeled.
  
* __Clustering__ grouping data points into clusters of similar examples.
  
### K-means clustering   

<img width="2752" alt="Image" src="https://github.com/user-attachments/assets/b8ec847c-7e1f-4a7b-bdd1-baa725015e27" />

* K-means is the most popular clustering algorithm.
* It is an iterative algorithm that tries to partition the dataset into K clusters, where each cluster is represented by its __centroid__ (the mean of the points in the cluster).
* The algorithm iteratively assigns each data point to the closest cluster __centroid__ and then updates the centroids based on the mean of the assigned points.  
* K-means clustering algorithm:  
  * (1) Randomly initialize K centroids.
  * (2) Then, repeat the following 2 steps until convergence:
    * Step 1: Assigng each point to its closest centroid to form K clusters.
    * Step 2: Recommpute the centroids.  
&nbsp;

* Step 1: Assigng each point to its closest centroid to form K clusters.
<img width="2370" alt="Image" src="https://github.com/user-attachments/assets/2891c95d-9e63-42cd-b919-2635ef7b32c2" />
&nbsp;

* Step 2: Recomputing the centroids.
<img width="2320" alt="Image" src="https://github.com/user-attachments/assets/8b21adf1-1603-4cbd-98da-03946260caae" />
&nbsp;

* __K-means algorithm__ 
  * Edge case: if a centroid has no points assigned to it, we can choose a random data point as the new centroid or we can remove that centroid from the algorithm.
  * Convergence: the K-means algorithm is guaranteed to converge to a local minimum, but it may not converge to the global minimum. Therefore, it is common to run the algorithm multiple times with different random initializations and choose the solution with the lowest cost function value.
<img width="2338" alt="Image" src="https://github.com/user-attachments/assets/67f4f7c0-967e-45e2-b67f-bb5d96b8d8b5" />
&nbsp;

* __K-means optimization objective__
  * The K-means algorithm is trying to minimize the following cost function, also called `distortion`:     
$J(c^{(1)}, \ldots, c^{(m)}, \mu_1, \ldots, \mu_K) = \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$
  
  * Where:
    * $m$ is the number of training examples.
    * $x^{(i)}$ is the i-th training example.
    * $c^{(i)}$ is the index of the cluster to which the i-th training example is assigned.
    * $\mu_k$ is the centroid of the k-th cluster.
    * $||x^{(i)} - \mu_{c^{(i)}}||^2$ is the squared distance between the i-th training example and the centroid of the cluster to which it is assigned.  


<img width="2316" alt="Image" src="https://github.com/user-attachments/assets/c48efe10-2eea-4cb7-bde7-8ad645a65462" />
&nbsp;

* __Initialization of K-means__
  * Random initialization: randomly select K data points as initial centroids.

* __Choosing the number of clusters K__
  * Elbow method: plot the cost function J as a function of K and look for an "elbow" in the graph where the cost starts to decrease more slowly.

### Anomaly detection  
Identifying data points that are significantly different from the majority of the data. This can be useful for tasks such as fraud detection, network security, and quality control.

<img width="2752" alt="Image" src="https://github.com/user-attachments/assets/52dc6564-51e3-4ef7-a321-84baf41c546e" />

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
  * Split your data into 3 sets: __training set__, __cross-validation set__, and __test set__.
  * Use the training set to estimate the parameters of the Gaussian distribution.
  * Use the cross-validation set to select the threshold $\epsilon$ that gives you the desired false positive rate.
  * Use the test set to evaluate the performance of your anomaly detection system.
  * Example of dataset for anomaly detection of aircraft engine failure: there are 2 situations:
    * 20 anomalies out of 10_000 data points
    * 2 anomalies out of 10_000 data points.   
<img width="1985" alt="Image" src="https://github.com/user-attachments/assets/fa15aba9-5168-4f82-86f4-efd24d44b8d3" />
&nbsp;

* __Anomaly detection vs supervised learning__  
  * Anomaly detection is used when you have very few examples of the anomaly (positive class) and many examples of the normal data (negative class). 
  * In contrast, supervised learning is used when you have a balanced dataset with enough examples of both classes. 

<img width="1916" alt="Image" src="https://github.com/user-attachments/assets/d4968fc7-55d4-4282-9bb5-2a803b1a1f38" />
&nbsp;

* __Choosing Features for anomaly detection__
  * The choice of features is crucial for the performance of an anomaly detection system. You should choose features that are relevant to the problem and that can help distinguish between normal and anomalous data points. 
  * For example, in the case of aircraft engine failure, you might choose features such as temperature, pressure, and vibration.


### Labs
* Lab 01: [K-means clustering](01_week/C3_W1_KMeans_Assignment.ipynb)
* Lab 02: [Anomaly detection](01_week/C3_W1_Anomaly_Detection.ipynb)

## Week 2: Recommender systems

__Learning Objectives__
* Implement __collaborative filtering__ recommender systems in TensorFlow.
* Implement deep learning __content based filtering__ using a neural network in TensorFlow.
* Understand ethical considerations in building recommender systems.

### Colaborative filtering recommender systems
* __Collaborative filtering__  
  * Is a method of making recommendations based on the preferences of similar users.  
 
  * The idea is to find users who have similar preferences and then recommend items that those similar users have liked.  
<img width="1970" alt="Image" src="https://github.com/user-attachments/assets/edbd5367-6987-4782-ac1d-75004af73ce8" />
&nbsp;

* __Cost function for collaborative filtering__  

To learn parameters w and b for collaborative filtering, we can use the following cost function.

<img width="1994" alt="Image" src="https://github.com/user-attachments/assets/39fc439e-d423-49e2-8abb-94e8ec4eeb9c" />
&nbsp;

* Function to learn features x for collaborative filtering, where x represents the features of the items (e.g., movies) that users interact with. In collaborative filtering, we want to learn both the parameters w and b for the users, as well as the features x for the items. 
  * The cost function for learning features x can be defined as follows, where: 
    * $m$ is the number of users. 
    * $n$ is the number of items (e.g., movies). 
    * $f_{w,b}(x^{(i)})$ is the predicted rating for user i and item j 
    * $y^{(i,j)}$ is the actual rating given by user i for movie j.
    * $\lambda$ is a regularization parameter to prevent overfitting. 
    * The first term in the cost function measures the difference between the predicted ratings and the actual ratings, while the second term adds a regularization penalty to prevent overfitting by encouraging smaller feature values.  

$$J(x) = \frac{1}{2m} \sum_{i=1}^m \sum_{j=1}^n (f_{w,b}(x^{(i)}) - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^n ||x^{(j)}||^2$$

<img width="1992" alt="Image" src="https://github.com/user-attachments/assets/88e5be2b-5d03-40f2-8940-95ca658313d1" />
&nbsp;

  
* If we do not have features for the items (e.g., movies), we can learn them from the data using a similar cost function. In this case, we would learn the features x for the items, while keeping the parameters w and b fixed. The cost function for learning features x can be defined as follows: 
$$J(x) = \frac{1}{2m} \sum_{i=1}^m \sum_{j=1}^n (f_{w,b}(x^{(i)}) - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^n ||x^{(j)}||^2$$
* Where:
  * $m$ is the number of users. 
  * $n$ is the number of items (e.g., movies). 
  * $f_{w,b}(x^{(i)})$ is the predicted rating for user i and item j 
  * $y^{(i,j)}$ is the actual rating given by user i for movie j.
  * $\lambda$ is a regularization parameter to prevent overfitting. 
  * The first term in the cost function measures the difference between the predicted ratings and the actual ratings, while the second term adds a regularization penalty to prevent overfitting by encouraging smaller feature values. 
  

<img width="2392" alt="Image" src="https://github.com/user-attachments/assets/12d2c7f0-0535-4464-90c4-7404a0725a9e" />
&nbsp;

* Function to learn both parameters w and b (users on the example), and features x (movies) for __collaborative filtering__.  

Why is collaborative filtering called "collaborative"? Because it learns both the parameters w and b for the users, and the features x for the items (e.g., movies) simultaneously. The cost function for learning both parameters w and b, and features x can be defined as follows:  

$$J(w,b,x) = \frac{1}{2m} \sum_{i=1}^m \sum_{j=1}^n (f_{w,b}(x^{(i)}) - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i=1}^m (||w^{(i)}||^2 + b^{(i)2}) + \frac{\lambda}{2} \sum_{j=1}^n ||x^{(j)}||^2$$
Where:
* $m$ is the number of users.
* $n$ is the number of items (e.g., movies).
* $f_{w,b}(x^{(i)})$ is the predicted rating for user i and item j.
* $y^{(i,j)}$ is the actual rating given by user i for movie j.
* $\lambda$ is a regularization parameter to prevent overfitting.
* The first term in the cost function measures the difference between the predicted ratings and the actual ratings, while the second and third terms add regularization penalties to prevent overfitting by encouraging smaller parameter values for both the users and the items.  

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
&nbsp;
<img width="1576" alt="Image" src="https://github.com/user-attachments/assets/a6ebcd20-9cb5-480a-987a-00ceef2fb92c" />

### Mean normalization for collaborative filtering
* __Mean normalization__  
To improve the performance of collaborative filtering, we can apply mean normalization to the ratings. This involves subtracting the mean rating for each item from the ratings before training the model. This helps to account for differences in user preferences and can lead to better recommendations.  
* Normalization by rows (users) or by columns (items), when new customer is added uses normalization by rows (example below) when added new movie use normalization by columns.  
<img width="1976" alt="Image" src="https://github.com/user-attachments/assets/419a6f25-ded4-4305-a782-e76ae12b1b09" />
&nbsp;

### TensorFlow implementation of collaborative filtering
* In TensorFlow, we can implement collaborative filtering using the Keras API. We can define a model that takes the user and item features as input and outputs the predicted rating. We can then compile the model with the appropriate loss function (e.g., binary cross-entropy for binary labels) and optimizer (e.g., Adam), and train the model on the training data. After training, we can use the model to make predictions for new user-item pairs and generate recommendations based on those predictions.

* Gradient decent reminder from previous weeks:  
<img width="1982" alt="Image" src="https://github.com/user-attachments/assets/805af192-f9bd-4262-b296-ee1c983ae229" />
&nbsp;

* __Auto Diff__ in TensorFlow: automatically compute derivative for gradient descent.  
<img width="1986" alt="Image" src="https://github.com/user-attachments/assets/2623b0a5-a71e-466d-8f48-1c98a9183efc" />

### Content-based filtering  
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
  

### Labs
* Lab 01: [Collaborative Filtering Recommender Systems](02_week/C3_W2_colaborative_filtering/C3_W2_Collaborative_RecSys_Assignment.ipynb)
* Lab 02: [Content-Based Filtering Recommender Systems](02_week/C2_W2_content_based_filtering/C3_W2_RecSysNN_Assignment.ipynb)

## Week 3 Reinforcement Learning
>This week, you will learn about reinforcement learning, and build a deep Q-learning neural network in order to land a virtual lunar lander on Mars!

__Learning Objectives__
* Understand key terms such as return, state, action, and policy as it applies to reinforcement learning
* Understand the Bellman equations
* Understand the state-action value function
* Understand continuous state spaces
* Build a deep Q-learning network

### Reinforcement Learning introduction

<img width="1879" alt="Image" src="https://github.com/user-attachments/assets/807323d4-2d9c-48c2-b3e1-bd0a32199e6f" />
&nbsp;

<img width="1896" alt="Image" src="https://github.com/user-attachments/assets/d8dd8ae3-a27a-4a67-8712-8b348f256647" />
&nbsp;

### State-action value function
* State action value function: $Q(s,a)$ represents the expected return (cumulative future reward) of taking action a in state s and following a certain policy thereafter. The goal of reinforcement learning is to learn an optimal policy that maximizes the expected return, which can be achieved by learning the optimal state-action value function $Q^*(s,a)$.  
  
<img width="1976" alt="Image" src="https://github.com/user-attachments/assets/3bc07fcf-59ac-467e-919e-df7c29f590e7" />

### Bellman Equation
* The Bellman equation is a fundamental equation in reinforcement learning that describes the relationship between the value of a state and the values of its successor states. It can be expressed as follows:
$$Q(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s',a') | s, a]$$
where $Q(s,a)$ is the state-action value function, $r$ is the reward received after taking action $a$ in state $s$, $\gamma$ is the discount factor that determines the importance of future rewards, and $s'$ is the next state resulting from taking action $a$ in state $s$. The Bellman equation states that the value of taking action $a$ in state $s$ is equal to the expected reward plus the discounted value of the best action in the next state $s'$. This equation is used to derive the optimal policy and to update the state-action value function during the learning process.
<img width="1706" alt="Image" src="https://github.com/user-attachments/assets/1b923961-55a9-4619-8288-d9ca2588de60" />

### Deep Reinforcement learning
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

### Labs
* Lab 01: [State Action function](03_week/01_lab_reinforcement_learning/state_action_value_function_example.ipynb)
* Lab 02: [Lunar Lander with Deep Q-learning](03_week/02_lab_lunar_lander/C3_W3_A1_Assignment.ipynb)
