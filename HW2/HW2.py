# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((ones_column, X), axis=1)    # Add a column of ones (bias term) to X
        y = y.reshape(y.shape[0], 1)

        theta = np.zeros((X.shape[1], 1))               # Initialize weights with zeros

        for i in range(self.iteration):
            prediction = self.sigmoid(np.matmul(X, theta))
            gradient = np.matmul(X.T, (prediction-y)) / y.shape[0]  # Calculate gradient
            theta -= self.learning_rate*gradient

        self.weights = theta.flatten()[1:]
        self.intercept = theta.flatten()[0]
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        prediction = self.sigmoid(np.matmul(X, self.weights) + self.intercept)
        return [1 if p>=0.5 else 0 for p in prediction]              # Round prediction values to 0 or 1
                 
    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        # Compute m0 and m1
        self.m0 = np.mean(X[y==0], axis=0)
        self.m1 = np.mean(X[y==1], axis=0)

        # Compute sb
        self.sb = np.outer(self.m0-self.m1, self.m0-self.m1)

        # Compute sw
        sw_class0 = sw_class1 = [[0, 0], [0, 0]]
        for i in range(len(y)):
            if y[i]==0:
                sw_class1 += np.outer(X[i]-self.m0, X[i]-self.m0)
            else:
                sw_class1 += np.outer(X[i]-self.m1, X[i]-self.m1)
        self.sw = sw_class0 + sw_class1

        # Compute w
        self.w = np.matmul(np.linalg.inv(self.sw), (self.m0-self.m1))
        self.w /= np.linalg.norm(self.w)
        
        # Compute slope
        self.slope = (self.m0[1]-self.m1[1]) / (self.m0[0]-self.m1[0])

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        # projected result of the testing data
        projected_data = np.matmul(X, self.w)          
        # projected means of the training data
        projected_m0 = np.matmul(self.m0, self.w)  
        projected_m1 = np.matmul(self.m1, self.w)

        # Determine the result y by comparing the distance with projected m0 and m1
        dist0 = np.abs(projected_data-projected_m0)
        dist1 = np.abs(projected_data-projected_m1)

        results = []
        for i in range(len(projected_data)):
            # closer to m0
            if dist0[i]<=dist1[i]:
                results.append(0)
            # closer to m1
            else:
                results.append(1)

        return np.array(results)
        
    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        # Compute slope (w) and intercept (b) of the projection line
        w = self.slope
        b = self.m0[1] - self.slope*self.m0[0]

        # Plot the projection line
        x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y = w*x+b
        plt.plot(x, y)
        
        # prediction of the testing set
        prediction = self.predict(X)
        
        c0_x, c0_y = X[prediction==0].T
        c1_x, c1_y = X[prediction==1].T
        p0_x = (w*c0_y + c0_x - w*b) / (w**2 + 1)
        p0_y = (w**2*c0_y + w*c0_x + b) / (w**2 + 1)
        p1_x = (w*c1_y + c1_x - w*b) / (w**2 + 1)
        p1_y = (w**2*c1_y + w*c1_x + b) / (w**2 + 1)

        # Plot the testing set and the prediction
        plt.scatter(c0_x, c0_y, c='r', marker='o')
        plt.scatter(c1_x, c1_y, c='b', marker='o')
        plt.scatter(p0_x, p0_y, c='r', marker='o')
        plt.scatter(p1_x, p1_y, c='b', marker='o')
        plt.plot([c0_x, p0_x], [c0_y, p0_y], c='r', linewidth=0.1)
        plt.plot([c1_x, p1_x], [c1_y, p1_y], c='b', linewidth=0.1)
        
        # Set the title with slope and intercept
        plt.title(f'Projection Line: w={w:f}, b={b:f}')
        
        plt.show()
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.0001, iteration=100000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

    #FLD.plot_projection(X_test)