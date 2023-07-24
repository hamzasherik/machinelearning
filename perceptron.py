import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from utils.logger import logger


class Perceptron(ABC):
    """
    perceptron class
    """

    def __init__(self, eta: float=0.01, epochs: int=50, random_state: int=1, threshold_accuracy: float=0.8) -> None:
        """
        
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.threshold_accuracy = threshold_accuracy

    def lifecycle(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        method that executes the lifecycle of the perceptron class

        Parameters
        ----------
        X: input features matrix
        y: ground truth labels vector
        Returns
        -------
        bool: true if lifecycle method executed successfully, false otherwise
        """

        X_preprocessed, y_preprocessed = self.preprocess(X, y)

        self.train(X_preprocessed, y_preprocessed)

        # TODO: create test and trian sets and pass in test data instead of train data to calculate accuracy of model
        self.accuracy_ = self.test(X_preprocessed, y_preprocessed)

        return True

    def preprocess(self, X: np.ndarray, y: np.ndarray):
        """
        method that preprocesses input data

        Parameters
        ----------
        X: input features matrix
        y: ground truth labels vector
        Returns
        -------
        X_preprocessed: preprocessed input feautures
        y_preprocessed: preprocessed input labels
        """

        # convert labels column from str dtype to float dtype
        y_preprocessed = np.unique(y, return_inverse=True)[1]
 
        # TODO: preprocess X
        X_preprocessed = X

        return X_preprocessed, y_preprocessed

    # TODO: return bool?
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        method that trains the model (adjusts weights and bias) 

        Parameters
        ----------
        X: input features matrix
        y: ground truth labels vector
        """

        # initialize bias to zero
        self.b_ = 0

        # initialize weights to random small numbers (or zero)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(scale=0.01, size=X.shape[1])

        # train until number of epochs is reached, or model converged, or after threhsold accuracy reached
        epoch_ctr = 0
        while epoch_ctr <= self.epochs:
            # calculate weighted inputs (z)
            z = np.dot(self.w_, np.transpose(X)) + self.b_

            # calculate activations
            activations = self.calculate_activations(z)

            # calculate current accuracy
            accuracy = np.count_nonzero(np.equal(activations, y)) / np.size(y)
            logger.info(f"accuracy at epoch {epoch_ctr}: {accuracy}")

            # break if model converged
            if accuracy == 1.0:
                logger.info(f"model converged at epoch {epoch_ctr}")
                break

            # break if accuracy reaches threshold accuracy
            if accuracy >= self.threshold_accuracy:
                logger.info(f"model converged at epoch {epoch_ctr} with accuracy {accuracy}")
                break

            # calculate delta_w and delta_b
            delta_w, delta_b = self.calculate_deltas(y, activations, X)

            # apply delta_w and delta_b to w and b respectively
            self.w_ += delta_w
            self.b_ += delta_b

            #increment epoch ctr
            epoch_ctr += 1

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        method that tests trained model against test set

        Parameters
        ----------
        X_test: test set input matrix
        y_test: ground truth labels vector
        Returns
        -------
        accuracy: accuracy on test set
        """

        # calculate weighted input (z)
        z = np.dot(self.w_, np.transpose(X_test)) + self.b_

        # calculate activations
        activations = self.calculate_activations(z)

        # calculate accuracy
        accuracy = np.count_nonzero(np.equal(activations, y_test)) / np.size(y_test)

        return accuracy

    def calculate_activations(self, z: np.ndarray) -> np.ndarray:
        """
        method that calculates the activations for each input

        Parameters
        ----------
        z: weighted input matrix

        Returns
        -------
        activations: activations matrix
        """

        activations = np.ndarray(z.shape)

        for i, weighted_input in enumerate(z):
            if weighted_input >= 0:
                activations[i] = 1
            else:
                activations[i] = 0

        return activations
    
    def calculate_deltas(self, ground_truth: np.ndarray, predicted: np.ndarray, X: np.ndarray):
        """
        method that calculates delta_w and delta_b

        Parameters
        ----------
        ground_truth: ground truth labels
        predicted: predicted labels 
        X: input data (samples)

        Returns
        -------
        delta_w:
        delta_b:
        """

        # calculate delta_w 
        delta_w = np.float64(np.dot(self.eta * np.subtract(ground_truth, predicted), X))

        # calculate delta_b
        delta_b = np.sum(self.eta * np.subtract(ground_truth, predicted))

        return delta_w, delta_b
    
    # TODO: finish this method. Will need to save accuracy at each epoch to use this method
    # TODO: could rename this method and have it graph everything we want, not just conergence over time.
    #def visualize_convergence(self, accuracy_arr: list[float]) -> None:
        """
        method that outputs a graph to visualize model convergence over time

        Parameters
        ----------
        accuracy_arr: array housing accuracy at each epoch
        """

        # plot convergence over time (x-axis: epoch number, y-axis: accuracy)
