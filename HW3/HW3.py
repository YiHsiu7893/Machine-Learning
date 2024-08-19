# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    length = y.size

    # probability of 0 and 1
    p0 = np.count_nonzero(y==0)/length
    p1 = np.count_nonzero(y==1)/length

    # formula of gini impurity
    g = 1 - (p0**2 + p1**2)

    return g

# This function computes the entropy of a label array.
def entropy(y):
    length = y.size

    # probability of 0 and 1
    p0 = np.count_nonzero(y==0)/length
    p1 = np.count_nonzero(y==1)/length

    if p0==0 or p1==0:
        en = 0
    else:
        # formula of entropy
        en = -(p0*np.log2(p0) + p1*np.log2(p1))

    return en

# Define tree node class
class Node:
    def __init__(self, prediction=None, feature=None, threshold=None, left=None, right=None):
        # prediction for leaf nodes
        self.prediction = prediction
        # feature, threshold, left and right for internal nodes
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        # Initialize feature_importance
        self.feature_importance = np.zeros(6)
        # Call GenerateTree() to build the decision tree
        self.root = self.GenerateTree(X, y, 0)

    # This function recursively build a decision tree
    def GenerateTree(self, X, y, depth):
        if depth==self.max_depth or np.unique(y).size==1:
            # Create a leaf node
            prediction = np.argmax(np.bincount(y))
            return Node(prediction=prediction)
        
        # Find the best split
        # And count the number of times each feature is used to split the data
        feature, threshold = self.SplitAttribute(X, y)
        self.feature_importance[feature] += 1

        # Split the data
        leftBranch = X[:, feature] <= threshold
        rightBranch = X[:, feature] > threshold

        # Recursively build the left and right subtrees
        leftChild = self.GenerateTree(X[leftBranch], y[leftBranch], depth+1)
        rightChild = self.GenerateTree(X[rightBranch], y[rightBranch], depth+1)

        # Create an internal node
        return Node(feature=feature, threshold=threshold, left=leftChild, right=rightChild)
    
    # This function returns feature and threshold for the best splitting tree node
    def SplitAttribute(self, X, y):
        Num, features = X.shape
        MinEnt = float('inf')
        bestf = None
        bestth = None

        # Find the best feature for doing split
        for feature in range(features):
            thresholds = np.unique(X[:, feature])
            # Find the best threshold of each feature for doing split
            for threshold in thresholds:
                leftBranch = X[:, feature] <= threshold
                rightBranch = X[:, feature] > threshold

                leftNum = np.sum(leftBranch)
                rightNum = np.sum(rightBranch)

                # Skip if all samples are on the same side
                if leftNum==0 or rightNum==0:
                    continue

                # Compute entropy after split
                e = (leftNum/Num) * self.impurity(y[leftBranch]) + (rightNum/Num) * self.impurity(y[rightBranch])

                if e<MinEnt:
                    MinEnt = e
                    bestf = feature
                    bestth = threshold

        return bestf, bestth
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = []

        for x in X:
            cur = self.root

            # Traverse the tree until a leaf node is reached
            while cur.left is not None and cur.right is not None:
                if x[cur.feature] <= cur.threshold:
                    cur = cur.left
                else:
                    cur = cur.right

            predictions.append(cur.prediction)

        return np.array(predictions)
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        plt.barh(columns, self.feature_importance, color='mediumblue')
        plt.title('Feature Importance')
        plt.tight_layout()

        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.weakClassifiers = []
        self.alphas = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        # Initialize weight which are uniform distributed
        weights = np.full(y.size, 1/y.size)

        # Project the range of y from {0, 1} to {-1, 1}
        y_proj = np.copy(y)
        y_proj[y_proj==0] = -1

        for _ in range(self.n_estimators):
            # Choose samples based on their weights
            chosen_indices = np.random.choice(X.shape[0], X.shape[0], p=weights)
            X_samples = X[chosen_indices]
            y_samples = y[chosen_indices]

            # Create a decision tree as the weak classifier
            tree = DecisionTree(criterion=self.criterion, max_depth=1)
            tree.fit(X_samples, y_samples)
            self.weakClassifiers.append(tree)

            # Make predictions and project the range to {-1, 1}
            predictions = tree.predict(X)
            predictions[predictions==0] = -1
            
            # Compute error and alpha
            error = np.sum(weights * (predictions!=y_proj))
            alpha = 1/2 * np.log((1-error) / (error+1e-10))
            self.alphas.append(alpha)

            # Update weight distribution
            weights *= np.exp(-alpha * y_proj * predictions)
            weights /= np.sum(weights)
        
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i in range(self.n_estimators):
            pred_tmp = self.weakClassifiers[i].predict(X)
            # Project the temporary predictions to {-1, 1}
            pred_tmp[pred_tmp==0] = -1
            y_pred += self.alphas[i] * pred_tmp
        
        predictions = np.sign(y_pred)
        # Project the predictions from {-1, 1} back to {0, 1}
        predictions[predictions==-1] = 0

        return predictions
        
    
# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=12)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

## Plot the feature importance
    #columns = ["age", "sex", "cp", "fbs", "thalach", "thal"]
    #tree = DecisionTree(criterion='gini', max_depth=15)
    #tree.fit(X_train, y_train)
    #tree.plot_feature_importance_img(columns)
