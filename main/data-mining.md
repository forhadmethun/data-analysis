# Intro to ML
## Logistic Regression
## Multilayer Perceptron
## Convolution Neural Networks

Numerical analysis
AI

# Data Mining
 - mathematical methods (logistic, regression, neural network, segmentation classification, clustering)
 - extracting information   

## What is data
 - collection of objects defined by attributes(properties, behavior)
 - Attribute classification
   - discrete
   - continuous

## 1.1 Introduction
    - collect, clean, process, analyze, gain useful info
    - huge data is generated regular
      - www
      - atm
      - ui
      - sensors
    - data maybe unstructured -> need to convert to structured 
    - similar to mineral mining
    - challenging as different problems are different types
    - DM applications are connected to 4 super problem
      - pattern mining
      - clustering
      - classification
      - outline detection
## 1.2 Data Mining Process
    - Data Collection
    - Data Processing
      - feature extraction
      - data cleaning
      - feature selection and transformation
    - analytical processing & algo
## 1.3 Data Types
    - Nondependency oriented/ Multidimensional Data
        - eg. age, weight
      - Quantitative multidimensional(continuous)
      - categorical(discreet)
      - Binary & set data
      - Text data
    - Dependency Oriented
        - eg. social network
      - types
        - implicit
        - explicit
      - time series data
      - discrete sequence & strings
      - spatial data
      - network & graph data
## 1.4 Major Building blocks
    - Association Pattern Mining
      - defined context of sparse binary databases
    - Data clustering
      - data partition to sets so that each cluster becomes similar
    - Outline detection 
      - determine rows of data matrix that are very different from the remanining
    - Data classification
      - from training data create training model can be used to predict class label of record
## 1.5 Scalability Issue
## 1.6 Some Application Scenario
    - Store product placement
    - customer recommendation
    - medical diagnosis
    - web log anomalies

## Data Preparation
### Steps
    - Feature extraction and portability
      - domains
        - sensor data
        - image data
        - web logs
        - network traffic
        - document data
    - Data cleaning
    - Data reduction, selection & transformation

### Data Type Portability
    - Converting data between various data types
    - types
      - numeric to categorical
      - category to numeric
      - text to numeric
      - time series to discrete sequence
      - time series to numeric
      - discrete to numeric
      - spacial to numeric
      - graph to numeric
      - any type to graph

### Data Cleaning
    - handling missing entities
    - handling incorrect and inconsistent entities
    - scaling and normalization

### Data Reduction and Transformation
    - sampling
      - for static data
      - reservior sample
    - feature selection
      - unsupervised feature
        - removal of noisy and redundant attribute
      - supervised
        - relevant to data classfication, predict
    - data reduction with axis rotation
    - Data reduction with type transformation


## Similarity and Distances
### 3.1 Introduction
Given two objects O1 & O2, determine value of Similarity Sim(O1, O2) (or distance Dist(O1,O2)) between those objects.   
In similarity func, larger value implies greater similarity
In distance func, smaller value implies greater simmilarity

In Spatial domain distance function is convenient where other domains like text, similiary function is more convenient.

most cases expressed in closed form, Eucledean distance
in some domains, defined algorithmically 


### 3.2 Distance functions for Multidimensional Data
#### Quantitative Data
Lp-norm
```
# - > Minkowski Distance
Dist(x,y) = (E|x-y|^p)^(1/p)
p = 1, Manhattan
p = 2, Euclidean

# for high dimension these functions may not work very well
```
 - Impact of domain-specific relevance
   - weighted formula of dist
 - impact of high dimensionality
   - while p increased
 - impact of locally irrelevant features
   - many irrelevant features
 - Impact of different Lp-Norms
   - p -> infinity
 - Match based Similarity Computation
   - to select locally relevant features for distance computation, simiple approach - cumulative evidence of matching
   - Pselect
 - Impact of Data Distribution
   - Mahalanobis distance 
     - similar to euclidean distance except that it normalizes the data on the basis of the interattribute correlation
 - Non linear distribution ISOMAP
   - dimensionality reduction and embedding method
 - Impact of local data distribution
   - shared nearest-neighbor 
   - generic methods
 - computational complexity
   - carefully choose Lp-norm and match-mased tehchniques are the fastest methods for the large-scale applications




SM -> Simple matching
SMC = # of matches / num of attributes
    = (m11 + m00)/(m01 + m10 + m11 + m00)
-> counts both presence and absence equally. used for objects with symmetric binary attributes. 
-> can be used to find students who answered similar in a test

JC -> Jaccard coefficients
J = (# of 11 matches) / (# of not-both-zero attribute values) 
  = (m11) / (m01 + m10 + m11)
-> handle objects with asymmetric binary attributes

Cosine Similarity
Extended Jaccard Coefficent(Tanimoto)
Correlation
    - linear relationship between objects [-1, 1]

#### Categorical Data
 - no order exists among discrete data so binarization approach can be helpful and similiarity function are more convenient
 - In the context of categorical data, the aggregate statistical properties of the data set should be used in computing similarity.
 - Inverse occurrence frequency

#### Mixed Quantitative and Categorical Data
 - generalizeing the approach to mixed data by adding the weights of the numeric and quantitative components
 - lambda param as the imporantace of the categorical and numerical attributes
 - needs normalization and standard deviation

### 3.3 Text Similarity Measures
 - L2 of a larger document will be always bigger for a longer document (though includes more common word) than a shorter document
 - cosine measure computes the angle which is insensitve to the absolute length
 - statistical measure is used to improve similarity
 - inverse document frequency idi
 - dampling function
 - cosine measure
 - jaccarad coefficient

#### Binary and Set Data
    - binary multidimensional data are a representation of set-based data, where 1 indicates the presence of an element in a set. 
    - 

### 3.4 Temporal Similarity
    - single contextual attribute representing time and one or more attribute varying along a particular time period
#### Time-Series Similarity Measure
##### Impact of Behavioral Attribute Normalization
##### L- Norm
##### Dynamic Time Wrapping Distance
##### Window Based Methods
#### Discrete Sequence Similarity Measures
##### Edit Distance
##### LCS
### 3.5 Graph Similarity Measures
#### Similarity Between two nodes in graph
#### Similarity Between two graphs 
### 3.6 Supervised Similarity Functions
 - previous methods don't require any understanding of user intentions
 - explicit feedback can be used to incorporate the info



#### Mixed Attribute



## Clustering
 - grouping / groups of similar objects
 - partitioning data into subclasses based on similarity
 - eg.
   - search engine
   - social network
   - genetics
   - marketing
 - types
   - 1. Centroid
     - k-mean
   - 2. connectivity
   - 3. distribution
   - 4. density based

### 6.1 Introduction
 - def:
   - given a set of data points, partition them into groups containing very similar data points
 - some applications
   - data summarization
   - customer segmentation
     - product recommendation of certain group of interest
   - social network analysis
   - relation to data mining problem
     - often used as a preprocessing step in many classification and outlier detection models
 - feature selection
   - there are many models for cluster analysis but many features may be noisy or uninformative for the analysis so these noise need to be removedd in the early stage of the analysis, such thing is called feature selection. 

### 6.2 Feature Selection for Clustering
    - goal is to remove noisy attribute
#### Filter Models
    - a score is associated with each feature with the use of a similarity based criterion. 
    - specific criteria is used to evaulate the imapct of specific features, or subset of features. Commonly used criteria - 
##### Term Strength
    - suitable for sparse domains such as text datawhere it's more meaningful to talk about presence or absence of nonzero values on the words. Similarity functions are more convenient. 
    - defined as the function of similar document pairs

##### Predictive Atribute dependance
    - motivation of this measure is that correlated features will result in better clusters than uncorrelated data
    - if data is numeric then regression algo is used else classification algo used
##### Entropy
    - highly clustered data reflects some of its clustering characteristics on the underlying distance attributes
    - A uniform distribution with poor clustering behavior has high entropy, whereas clustered data has lower entropy.
##### Hopkins Stat
    - used to measure the clustering tendency of a data set, althogh can be used on a particular subset of attributes

#### Wrapper Model 
    - a clustering algorithm is used to evaulate the quality of a subset of features
### Representative Based Algorithm
    - simplest of all clustering algo as they rely on intuitive notions of distance/similar to cluster data points

#### K-means algo
    - the sum of squqres of the eucledean distances of data points to their closest representatives is used to quantify the objective function of the clustering
#### Kernel k-means algo
    - k-mean can be extended to discuver arbitrary shape with use of kernel trick
#### k-medians algo
    - mahnattan distance is used as the objective function of choice
#### k-medoids algo
    - similar as k-representative algorithm bt representatives are always selected from the database D
### Hierarchical Clustering Algo
    - data with distances
    - many of them are, density-orgraph-based methods, 
#### Bottom-up agglomerative
    - individual data points are successively agglomerated into higher-level clusters
    - dendogram
      - relationship between clusters are shown by this
    - differnt ways of computing distances between two groups of objects are as follows
      - Best(single) linkage
      - Worst(cimplete) linkage
      - group-avg
      - closest centroid
      - variance-based
      - world's most popular
#### Top-down divisive
    - successively partition the data points into a tree-like structure
### 6.5 Probabilistic Model-Based Algo
### 6.6 Grid-Based & Density-Based Algo
### 6.7 Graph-Based Algo
### 6.8 Non-negative matrix Factorization
### 6.9 Cluster Validation




## Classification
    - predicting something based on given input
    - systematic approach to build model from input
    - Mathematical techniques
      - decision tree
        - flow chart like tree structure
        - each node testing of an attribute value
        - branch represents outcome of test
        - leaves represents decision
      - linear tree
      - neural network
      - statistics
    - Algorithms
      - decision tree
      - rule based induction
      - neural network
      - bayesian network
      - genetic algorithm
    - eg.
      - fraud detection
    - phases
      - learning phase
        - training data analyzed by model thorugh algorithm
      - classification phase

already partitioned into groups, that are referred to as categories or classes. The learning of these categories is typically achieved with a model. This model is used to estimate the group identifiers (or class labels) of one or more previously unseen data examples with unknown labels
### Feature Selection
### Decision Tree
  classification methodologies where classification process is modeled with the use of a set of hierarchical decisions on the feature variables, arranged in a tree-liked structure. classical algo: c4, ID3, CART
#### Split Criteria
  goal is to maximize the separation of the different classes among the children nodes. 
  The design of split criteria based on
    - binary attribute
    - categorical attribute
    - numeric attribute
  Quantification of best split from set of choices are based on the principles
    - Error rate
    - Gini index
    - Entropy

#### Stopping Criteria and Prunning
Unfortunately, there is no way of knowing the correct point at which to stop the growth of the tree. Therefore, a natural strategy is to prune overfitting portions of the decision tree and convert internal nodes to leaf nodes.Many different criteria are available to decide whether a node should be pruned, one strategy is minimum description length principle(MDL). 
#### Practical Issues
The main problem is that the amount of training data required to properly
approximate a complex boundary with a treelike model is very larg
### Rule-Based Classifiers
it uses: if condition then conclusion
Rule set follow one or more of the following properties:
  - mutually exclusive rules
  - exhaustive rules

In case of non mutually exclusive  conflicts can be resolved 
  - rule ordering
  - unordered rules

#### Rule Generation from Decision Trees
Rule can be extracted from the different paths in a decision tree. 
#### Sequencial Covering Algorithms
used for creating ordered rule lists
##### Learn-One-Rule
#### Rule Prunning
#### Associative Classifiers
### Probabilistic Classifiers
construct a model that quantifies the relationship between the feature variables and the target(class) variable as probability. 
Two most popular models are
  - Bayes classifier
  - Logistic regression

#### Naive Bayes Classifier
##### The ranking Model for Classification
##### Discussion of Naive Assumption
#### Logistic Regression
##### Training a Logistic Regression Classifier
### Support Vector Machines
  defined for binary classification of numeric data.as with all linear model SVMs use separating hyperplanes as the decision boundary between the two classes. 
#### SVM for Linearly Seperable Data
##### Solving the Lagrangian Dual
#### SVM with Soft Margin for Nonseparable Data
##### Comparison with Other Linear Models
#### Nonlinear Support Vector Machines
#### The Kernel Trick
  - Gaussian basis Kernel
  - Polynomial Kernel
  - Sigmoid kernel
##### Other Applications of Kernel Methods
### Neural Networks
are a model of simulation of the human nervous system. cells referred as nurons. contact point called synapses. learning performed in living organisms by changing the strength of synaptic connections between nurons. 
#### Single Layer Neural Network: The Perceptron
#### Multi Layer Neural Networks
#### Comparing Various Linear Models
### Instance Based Learning
in this learning, training is delayed until the last step of classification, are also referred lazy learners
#### Nearet Neighbor Classfiers
##### Neares Neighbors with Linear Discriminant Analysis
### Classfier Evaluation
#### Methodological Issues


# Basics
## Data Mining
### Unsupervised learning
    - i build algo and computer, look for pattern in data. I'll not label and won't tell computer that what kind of data it is, goal of the algo is to find the pattern 
    - data are not labelled
    - usage
      - pattern in data*
      - clusters*
    - given list of input and learning alrightm  will make cluster/group of the data
### Supervised data
    - i know something about data, i label, so that computer can classify/ predict
    - usage
      - providing label ahead of time
      - how to label new data
      - training -> test
      - 80% train -> 20% test
    - classification + regression*
      - the line seperate the classes is regression
    - given list of (input, output) and a learning algorithm that buidls a model and new input given to that model gives an output
    - types
      - classification
        - discrete 
          - eg. weather, symptom to desies
      - regression
        - continuous
          - eg. feature to price of house
    - Phases
      - Training Phase: Trainign Set -> Learning Algorithm -> Model
      - Testing Phase:  X -> Model -> Predicted, Y
    - Or, Phases
      - Input -> Feature Extraction -> Features -> Machine Learning Algorithm -> classifier Model
      - Input -> Feature Extraction -> Features -> Classifier Model -> Label

### Features
    - set of quantifiable properties
    - types
      - categorical. eg. A, B, blood type
      - ordinal. eg. large, small
      - integer-valued, eg. num of word in text
      - real-valued, eg. height

### Semisupervised learning
    - we have combination of labelled and unlabelled data
    - from labelled data -> supervised learning
    - from labelled & unlabelled data -> semi-supervised learning
### Reinforcement learning
 - each step take decision based on some penalty



Cross validation is always necessary*

Clustering is an optimization problem

Hierarchical Clustering
    - n cluster initially then matches common one and reduce to one and thus becomes n-1 and continue
    - also calleda aggelometer hierarchical clustering

Linkage Metrics
    - Single-linkage
      - shortest distance
    - complete-linkage
      - grest distance
