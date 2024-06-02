# DataScience_TermProject
Medical Data Preprocessing and Model Training

# Description of Models
1. **RandomForestClassifier**
- Description: A Random Forest classifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees.
- Key Features:
  - Handles both classification and regression tasks.
  - Reduces overfitting by averaging multiple decision trees.
  - Robust to noisy data and can handle large datasets with higher dimensionality.
- Hyperparameters Tuned:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of each tree.
  - `min_samples_split`: Minimum number of samples required to split an internal node.

2. **DecisionTreeClassifier**
- Description: A Decision Tree classifier is a non-parametric supervised learning method used for classification and regression. It splits the data into subsets based on the value of input features, creating a tree structure.
- Key Features:
  - Easy to interpret and visualize.
  - Can handle both numerical and categorical data.
  - Prone to overfitting, but this can be controlled through pruning and setting appropriate hyperparameters.
- Hyperparameters Tuned:
  - `max_depth`: Maximum depth of the tree.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `criterion`: Function to measure the quality of a split (e.g., 'gini' or 'entropy').

3. **LogisticRegression**
- Description: Logistic Regression is a linear model for binary classification that estimates the probability that an instance belongs to a particular class using a logistic function.
- Key Features:
  - Simple and easy to implement.
  - Good for linearly separable data.
  - Can be extended to multi-class classification using techniques like one-vs-rest.
- Hyperparameters Tuned:
  - `penalty`: Regularization technique (e.g., 'l1', 'l2', 'elasticnet', 'none').
  - `C`: Inverse of regularization strength; smaller values specify stronger regularization.
  - `solver`: Algorithm to use in the optimization problem (e.g., 'liblinear', 'lbfgs', 'saga').

4. **SVC (Support Vector Classifier)**
- Description: Support Vector Classifier (SVC) is a supervised learning model used for classification tasks. It finds the hyperplane that best separates the data into different classes.
- Key Features
  - Effective in high-dimensional spaces.
  - Uses a subset of training points in the decision function, making it memory efficient.
  - Supports different kernel functions (linear, polynomial, RBF) for non-linear classification.
- Hyperparameters Tuned:
  - `C`: Regularization parameter; trade-off between correct classification of training examples and maximization of the decision function's margin.
  - `kernel`: Kernel type to be used in the algorithm (e.g., 'linear', 'poly', 'rbf').
  
5. **KNeighborsClassifier**
- Description: K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that classifies a data point based on how its neighbors are classified.
- Key Features:
  - Non-parametric and lazy learning algorithm.
  - Simple and easy to implement.
  - Sensitive to the local structure of the data.
- Hyperparameters Tuned:
  - `n_neighbors`: Number of neighbors to use.
  - `weights`: Weight function used in prediction (e.g., 'uniform', 'distance').

# Library used(Dependencies)
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib
