import numpy as np

from abc import ABC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from utils.logger import logger

# NOTE: logistic regression only works on binary classification. Multinomal logistic regression (softmax) works on multi-class classification
class LogisticRegressionGD(ABC):
    """
    logisitic regression model class

    Attributes
    ----------
    eta: learning rate
    epochs: number of training iterations over entire dataset
    random_state: seed in a random number generator allowing for reproducible results
    threshold_accuracy: minimum threshold required before model completes training
    w_: weight matrix
    b_: bias vector
    accuracy_: final model accuracy
    """

    def __init__(self, eta: np.float64, epochs: np.int32, random_state: np.int32, threshold_accuracy: np.float64):
        """
        constructor for logistic regression model class

        Parameters
        ----------
        eta: learning rate
        epochs: number of training iterations over entire dataset
        random_state: seed in a random number generator allowing for reproducible results
        threshold_accuracy: minimum threshold required before model completes training
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.threshold_accuracy = threshold_accuracy

    def lifecycle(self, X: np.ndarray, y: np.ndarray):
        """
        method that executes the lifecycle of the logistic regression model clas

        Parameters
        ----------
        X: input feautures matrix
        y: labels vector
        """

        # call preprocess
        X_train, X_test, y_train, y_test = self.preprocess(X, y)

        # call fit
        self.fit(X_train, y_train)

        # call predict
        y_pred = self.predict(X_test)

        # calculate final accuracy
        self.accuracy_ = accuracy_score(y_test, y_pred)

        logger.info(f"final model accuracy: {self.accuracy_}")

    def preprocess(self, X: np.ndarray, y: np.ndarray):
        """
        method that preprocesses data

        Parameters
        ----------
        X: input feautres matrix 
        y: labels vector

        Returns
        -------
        X_train: preprocessed train input features matrix
        X_test: preprocessed test input features matrix
        y_train: preprocessed train labels vector
        y_test: preprocessed test labels vector
        """

        # split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

        # normalize X
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        # convert categorical labels to integers
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        return X_train, X_test, y_train, y_test

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        method that fits the model (trains the model by adjusting weights and bias)

        Parameters
        ----------
        X_train: train input feautres matrix
        y_train: train labels vector
        """

        # initialize bias to 0
        self.b_ = 0

        # initialize weights to small random numbers
        rgen = np.random.RandomState(seed=self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.1, size=X_train.shape[1])

        epoch_ctr = 0
        while epoch_ctr <= self.epochs:

            # calculate weighed inputs
            z = np.dot(self.w_, np.transpose(X_train)) + self.b_

            # calculate sigmoid activations
            sigmoid_activations = self.calculate_sigmoid_activations(z)

            # calculate step activations
            step_activations = self.calculate_step_activations(sigmoid_activations)

            # calulcate accuracy
            accuracy = accuracy_score(y_train, step_activations)
            logger.info(f"accuracy at epoch {epoch_ctr}: {accuracy}")

            # check if converged
            if accuracy == 1.0:
                logger.info(f"model converged at epoch {epoch_ctr}")
                break

            # check if threshold accuracy reached
            if accuracy >= self.threshold_accuracy:
                logger.info(f"model reached/passed threshold accuracy at epoch {epoch_ctr} with accuracy {accuracy}")
                break
            
            # TODO: SGD instead of batch GD?
            # calculate delta w and b
            delta_w, delta_b = self.calculate_deltas(y_train, sigmoid_activations, X_train)

            # apply delta w and b to w and b
            self.w_ += delta_w
            self.b_ += delta_b

            # increment epoch counter
            epoch_ctr += 1

    def predict(self, X_test: np.ndarray):
        """
        method that predicts the labels given input features

        Parameters
        ----------
        X_test: test input features matrix

        Returns
        -------
        y_pred: predicted labels vector
        """

        # calcualte weighted inputs
        z = np.dot(X_test, self.w_) + self.b_

        # calculate sigmoid activations
        sigmoid_activations = self.calculate_sigmoid_activations(z)

        # calculate step activations
        y_pred = self.calculate_step_activations(sigmoid_activations)

        # return predictions
        return y_pred

    def calculate_deltas(self, ground_truth: np.ndarray, sigmoid_activations: np.ndarray, X: np.ndarray):
        """
        method that calculates detlas for weights and bias adjustments

        Parameters
        ----------
        ground_truth: ground turth labels vector
        sigmoid_activations: sigmoid activations matrix
        X: input features matrix

        Returns
        -------
        delta_w: change in weights matrix
        delta_b: change in bias vector
        """

        delta_w = np.float64((2/ground_truth.size) * self.eta * np.dot(np.subtract(ground_truth, sigmoid_activations), X))

        delta_b = np.float64(np.sum((2/ground_truth.size) * (self.eta * np.subtract(ground_truth, sigmoid_activations))))

        return delta_w, delta_b


    def calculate_sigmoid_activations(self, z: np.ndarray):
        """
        method that calculates sigmoid activations

        Parameters
        ----------
        z: weighted inputs matrix

        Returns
        -------
        sigmoid_activations: sigmoid activations matrix
        """

        return 1.0 / (1.0 + np.exp(-z)) 
    
    def calculate_step_activations(self, sigmoid_activations: np.ndarray):
        """
        method that calculates step activations

        Parameters
        ----------
        sigmoid_activations: sigmoid activations matrix

        Returns
        -------
        step_activations: step activations matrix
        """

        step_activations = np.ndarray(sigmoid_activations.shape)

        for i in range(sigmoid_activations.size):
            if sigmoid_activations[i] >= 0.5:
                step_activations[i] = 1
            else:
                step_activations[i] = 0

        return step_activations
