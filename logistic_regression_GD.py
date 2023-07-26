import numpy as np
from abc import ABC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils.logger import logger

# NOTE: logistic regression only works on binary classification. Multinomal logistic regression (softmax) works on multi-class classification
class LogisticRegressionGD(ABC):
    """
    logisitic regression model class
    """

    def __init__(self, eta: np.float64, epochs: np.int32, random_state: np.int32, threshold_accuracy: np.int32):
        """
        
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.threshold_accuracy = threshold_accuracy

    def lifecycle(self, X: np.ndarray, y: np.ndarray):
        """
        
        """

        # call preprocess
        X_train, X_test, y_train, y_test = self.preprocess(X, y)

        # call fit
        self.fit(X_train, y_train)

        # call predict
        y_pred = self.predict(X_test, y_test)

        # calculate final accuracy
        self.accuracy_ = accuracy_score(y_test, y_pred)

        logger.info(f"final model accuracy: {self.accuracy_}")

    def preprocess(self, X: np.ndarray, y: np.ndarray):
        """
        
        """

        # split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

        # normalize X
        sc = StandardScaler()
        sc.fit(X_train)
        sc.transform(X_train)
        sc.transform(X_test)

        # TODO: use OneHotEncoder from sklearn
        # convert categorical labels to integers
        y_train = np.unique(y, return_inverse=True)
        y_test = np.unique(y, return_inverse=True)

        return X_train, X_test, y_train, y_test

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        
        """

        # initialize bias to 0
        self.b_ = 0

        # initialize weights to small random numbers
        rgen = np.random.RandomState(seed=self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.1, size=X_train.shape[1])

        epoch_ctr = 0
        while epoch_ctr <= self.epochs:

            # calculate weighed inputs
            z = np.dot(self.w_, X_train) + self.b_

            # calculate sigmoid activations
            sigmoid_activations = self.calculate_sigmoid_activations(z)

            # calculate step activations
            step_activations = self.calculate_step_activations(sigmoid_activations)

            # calulcate accuracy
            accuracy = accuracy_score(y_train, step_activations)

            # check if converged
            if accuracy == 1.0:
                logger.info(f"model converged at epoch {epoch_ctr}")
                break

            # check if threshold accuracy reached
            if accuracy >= self.threshold_accuracy:
                logger.info(f"model reached/passed threshold accuracy at epoch {epoch_ctr} with accuracy {accuracy}")
                break
            
            # TODO: SGD instead of batch GD
            # calculate delta w and b
            delta_w, delta_b = self.calculate_deltas()

            # apply delta w and b to w and b
            self.w_ += delta_w
            self.b_ += delta_b

            # increment epoch counter
            epoch_ctr += 1

    def predict(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        
        """

        # calcualte weighted inputs
        z = np.dot(X_test, self.w_) + self.b_

        # calculate sigmoid activations
        sigmoid_activations = self.calculate_sigmoid_activations(z)

        # calculate step activations
        step_activations = self.calculate_step_activations(sigmoid_activations)

        # return predictions
        return step_activations

    def calculate_deltas(self, ground_truth: np.ndarray, sigmoid_activations: np.ndarray, X: np.ndarray):
        """
        
        """

        delta_w = np.float64((2/ground_truth.size) * self.eta * np.dot(np.subtract(ground_truth, sigmoid_activations), X))

        delta_b = np.float64(np.sum((2/ground_truth.size) * (self.eta * np.subtract(ground_truth, sigmoid_activations))))

        return delta_w, delta_b


    def calculate_sigmoid_activations(self, z: np.ndarray):
        """
        
        """

        return 1.0 / (1.0 + np.exp(-z)) 
    
    def calculate_step_activations(self, sigmoid_activations: np.ndarray):
        """
        
        """

        for i in range(sigmoid_activations.size):
            if sigmoid_activations[i] >= 0.5:
                sigmoid_activations[i] = 1
            else:
                sigmoid_activations[i] = 0

if __name__ == "__main__":
    y = np.array([0,0,0])