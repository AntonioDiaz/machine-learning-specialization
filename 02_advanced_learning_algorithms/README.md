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
  - [Evaluating a model](#evaluating-a-model)
  - [Week 3: Labs](#week-3-labs)
- [Week 4: Decision trees](#week-4-decision-trees)
  - [Decision Tree Model](#decision-tree-model)
  - [Decision tree learning](#decision-tree-learning)
    - [Entropy or measure of impurity](#entropy-or-measure-of-impurity)
    - [Information Gain](#information-gain)
    - [Decision tree learining algorithm](#decision-tree-learining-algorithm)
    - [One Hot Encoding](#one-hot-encoding)
    - [Continue Value features](#continue-value-features)
  - [Tree Ensemble](#tree-ensemble)
    - [Sampling with replacement](#sampling-with-replacement)
    - [Random Forest Algorithm](#random-forest-algorithm)
    - [XGBoost Algorithm](#xgboost-algorithm)
  - [When to use decision trees or neural networks](#when-to-use-decision-trees-or-neural-networks)
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

### Evaluating a model
* In order to tell if your model is doing well, specially for applications where you have more than 1 or 2 features, you need to evaluate your model using a test set. 
* The __test set__ is a separate dataset that is not used during the training of the model, and it allows you to evaluate how well your model generalizes to new, unseen data. 
* By evaluating your model on a test set, you can get an estimate of its performance and identify any issues such as overfitting or underfitting. 
* This is an important step in the machine learning process, as it helps you to ensure that your model is not just memorizing the training data but is able to make accurate predictions on new data.


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

### Decision Tree Model
* A decision tree is a __type of machine learning model__ that is used for classification and regression tasks. 
* It is a tree-like structure where each internal node represents a decision based on a __feature__, each branch represents the outcome of the decision, and each leaf node represents a class label or a regression value. 
* The decision tree learns from the __training data__ by recursively splitting the data based on the feature that provides the best separation of the classes or values. 
* The goal of the decision tree is to create a model that can make accurate predictions on new, unseen data by following the path from the root node to a leaf node based on the features of the input data.

* __Cat example__   
Having this dataset with 3 features (ear shape, face shape and wiskers length) and 2 classes (cat or not cat), we can build a decision tree to classify new examples of cats based on these features. The decision tree would learn from the training data by finding the best feature to split the data at each node, and it would continue to split the data until it reaches a leaf node that represents a class label (cat or not cat).
  
<img width="1558" alt="Image" src="https://github.com/user-attachments/assets/99b99962-2f68-47a2-b3bd-32278e483a54" />
&nbsp;

* __Decision tree example__ 
 
<img width="1582" alt="Image" src="https://github.com/user-attachments/assets/b81de9fa-a7fc-4ad7-9e5e-cf4c029483cb" />

### Decision tree learning

* There are 2 decisions to make when building a decision tree.  

* Decision 1: Which feature to use for splitting the data at each node.  
<img width="1528" alt="Image" src="https://github.com/user-attachments/assets/cd6024b1-7bf1-4e0e-ab2e-1be53332ff1c" />
&nbsp;

* Decision 2: When to stop splitting the data and create a leaf node, 4 criteria to stop splitting the data:
  * When all examples in the node belong to the __same class__.
  * When spliting a node will result in the tree exceeding a __maximum depth__ (e.g., more than 10 levels).
  * When the __improvement in the impurity__ are below a threshold (e.g., less than 0.01).
  * When the __number of examples__ in the node are below a threshold (e.g., less than 5 examples).



#### Entropy or measure of impurity
* The decision tree learning algorithm uses a measure of impurity called __entropy__ to determine which feature to use for splitting the data at each node. 
* The feature that provides the greatest reduction in entropy (or the greatest information gain) is chosen as the feature to split the data.
* __Entropy__ is a measure of impurity or disorder in a dataset. It is calculated using the formula:  
$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$
&nbsp;
<img width="1574" height="774" alt="Image" src="https://github.com/user-attachments/assets/bd10af10-8023-4634-a446-10dd17389600" />
&nbsp;

#### Information Gain
* Information gain is a measure of the reduction in entropy that results from splitting a dataset based on a particular feature. 
* It is calculated as the __difference__ between the entropy of the __parent node__ and the weighted average of the entropies of the __child nodes__.  
* The feature that provides the highest information gain is chosen as the feature to split the data at each node in the decision tree.  
* The formula for information gain is:
  
$$\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right)$$

* Cats dataset example:  
<img width="2610" alt="Image" src="https://github.com/user-attachments/assets/376f662b-1c8c-4d06-b42e-930b7646cb83" />


#### Decision tree learining algorithm
1. Start with all examples at the root node
2. Calculate information gain for all possible features, and pick the one with
the highest information gain
3. Split dataset according to selected feature, and create left and right
branches of the tree
4. Keep repeating splitting process until stopping criteria is met:
   1. When a node is 100% one class
   2. When splitting a node will result in the tree exceeding a maximum
      depth
   3. When information gain from additional splits is less than threshold
   4. When the number of examples in a node is below a threshold


* Recursive spliting process to build the decision tree:  
<img width="1300" alt="Image" src="https://github.com/user-attachments/assets/0612767d-ec25-49f1-9df2-e6cf876e1799" />
&nbsp;

#### One Hot Encoding  
solution when a feature can take more than two possible values. One Hot because only one feature is selected.  

<img width="1578" height="402" alt="Image" src="https://github.com/user-attachments/assets/35489ceb-ffae-4a54-aa32-2c8f44dc9efd" />
&nbsp;
<img width="2500" alt="Image" src="https://github.com/user-attachments/assets/1a703253-6786-46a0-9319-d32c2651d583" />

&nbsp;
  
#### Continue Value features
* When a feature can take on a continuous range of values, we can use a threshold to split the data. 
* For example, if we have a feature "weight" that can take on any value, we can choose a threshold (e.g., weight <= 9 lbs) to split the data into two branches.
* The decision tree learning algorithm will evaluate different thresholds for the continuous feature and choose the one that provides the best information gain for splitting the data. 
* This allows the decision tree to handle both categorical and continuous features effectively.  
  
<img width="1244" height="547" alt="Image" src="https://github.com/user-attachments/assets/eda27b66-63b2-44ce-97bf-ac11e5ab53f2" />

&nbsp;

### Tree Ensemble  
* Decision can be very sensitive to small changes in the training data, which can lead to overfitting.
  
* The idea is to use __multiple trees__ to make a prediction, instead of just one tree.
* The prediction of the ensemble is typically made by aggregating the predictions of the individual trees, such as by taking a majority vote for classification tasks or averaging for regression tasks.

<img width="2500" alt="Image" src="https://github.com/user-attachments/assets/a2998729-de96-4888-82fc-b4f09a4cef31" />
&nbsp;


#### Sampling with replacement
* The idea is to create multiple different training datasets by sampling from the original training dataset with replacement.
* Each example in the training dataset is __randomly selected__ to be included in the sample, and it can be __selected multiple__ times (replacement). This means that some examples may be included in the sample more than once, while others may not be included at all.
* This technique is often used in ensemble learning methods, such as __random forests__, to create multiple different training datasets for each individual model in the ensemble. By sampling with replacement, we can create diverse training datasets that help to reduce overfitting and improve the generalization of the ensemble model.
* For example, if we have a training dataset with 100 examples, we can create a sample of 100 examples by randomly selecting examples from the original dataset with replacement. This means that some examples may be selected multiple times, while others may not be selected at all. This process is repeated multiple times to create different training datasets for each model in the ensemble.
* Sampling with replacement is a powerful technique that allows us to create diverse training datasets and improve the performance of ensemble models, such as random forests, by reducing overfitting and increasing generalization.
* Example with tokens, of 4 colours:  
<img width="1334" height="712" alt="Image" src="https://github.com/user-attachments/assets/f370312f-d3b5-4fcc-b5d3-7edb00181535" />



#### Random Forest Algorithm   
* Random Forest (also know as Bagged Decision Trees) is an ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of predictions. 
* It works by creating a "forest" of decision trees, where each tree is trained on a __random subset__ of the __training data__ and a random subset of the __features__. 
* The final prediction is made by aggregating the predictions from all the individual trees, typically through __majority voting__ for classification tasks or averaging for regression tasks. 
* Random Forest helps to __reduce overfitting__ and increase generalization by introducing randomness into the training process, making it less likely for the model to memorize the training data and more likely to capture the underlying patterns in the data.
  
<img width="1024" height="559" alt="Image" src="https://github.com/user-attachments/assets/e9878ee6-7d78-490c-b8ab-a7252d48b21c" />
&nbsp;

#### XGBoost Algorithm
* Boosted Trees Intuition: boosting is an ensemble learning method that combines multiple weak learners (e.g., decision trees) to create a strong learner. The idea of __deliberate practice__ is behind boosting, where the model focuses on learning from the examples that are __misclassified__ by previous models in the ensemble.
 
<img width="1932" height="762" alt="Image" src="https://github.com/user-attachments/assets/910361c5-6baa-448c-a7d1-e333ce26f911" />
&nbsp;

* __XGBoost__   (eXtreme Gradient Boosting)
  * XGBoost is an optimized implementation of the gradient boosting algorithm that is designed to be efficient and scalable. 
  * It uses a combination of decision trees and gradient descent to iteratively improve the model's performance. 
  * XGBoost is known for its speed and performance, making it a popular choice for machine learning competitions and real-world applications. 
  * It can handle both regression and classification tasks and is particularly effective for large datasets with high dimensionality.
  * XGBoost includes features such as __regularization__, __parallel processing__, and handling of missing values, which contribute to its effectiveness and efficiency in training models.
  * Using XGBoost  
<img width="1948" height="660" alt="Image" src="https://github.com/user-attachments/assets/5428276a-bbea-4b9f-8731-02cd97fdd25b" />


### When to use decision trees or neural networks
* Decision Trees and Tree ensembles:
  * Works well with structured data (tabular data) where the features have clear relationships and decision boundaries.
  * Not recomended for unstructured data such as images, audio, or text, where neural networks tend to perform better.
  * Use when you need interpretable models and have structured data with clear decision boundaries.
* Neural Networks:
  * Works well with both structured and unstructured data such as images, audio, and text, where they can learn complex patterns and representations.
  * May be slower to train and less interpretable than decision trees, but they can achieve higher accuracy on complex tasks with unstructured data. 
  * Works well with transfer learning, where a pre-trained neural network can be fine-tuned on a new task with limited data.
  * When building a system of multiple models working together, it might be easier to string together multiple neural networks.

* In summary, the choice between decision trees and neural networks depends on the type of data you have, the complexity of the relationships in the data, and whether interpretability is a requirement for your model. Decision trees are often preferred for structured data with clear decision boundaries, while neural networks are more suitable for unstructured data and complex patterns.

<img width="2752" height="1536" alt="Image" src="https://github.com/user-attachments/assets/8adb8bf1-6523-4908-b63f-71864c3b969d" />
&nbsp;

### Week 4: Labs
* Lab 01: [Decision Trees](04_week/C2_W4_Lab_01_Decision_Trees.ipynb)
In this notebook you will visualize how a decision tree is split using information gain.
* Lab 02: [Tree Ensemble](04_week/C2_W4_Lab_02_Tree_Ensemble.ipynb)
