import numpy as np
import torch as t
from torch.nn.functional import normalize
from abc import ABC
from utils.logger import logger

# TODO: make sure all data types are consistent/make sense
class AdalineSGD(ABC):
    """
    adaline (adaptive linear nueron) SGD (stochastic gradient descent) class
    """

    # TODO: need the ability to disable the threshold accuracy feature. We could just set it to 1.0
    # TODO: add self.errors_ attribute to be used to graph convergence over number of epochs.
    # TODO: add shuffle attribute (bool) to shuffle data for each epoch
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

        logger.info(f"final model accuracy: {self.accuracy_}")

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

        X = X.astype(np.float64)

        # convert labels column from str dtype to float dtype
        y_preprocessed = np.unique(y, return_inverse=True)[1].astype(np.float64)

        # TODO: add ability to configure preprocessing steps to take (or learn what steps to take!)

        # NOTE: normalization of features DOES NOT stop bias if one feature is always larger than the rest!
        # normalize input features
        # X = X.astype(np.float64)
        # X_tensor = t.tensor(X, dtype=t.float64)
        # X_preprocessed = normalize(X_tensor, dim=1).cpu().detach().numpy()

        # NOTE: why does standardization work so well on IRIS while normalization doesn't? 
        # standardize input features
        X_tensor = t.tensor(X, dtype=t.float64)
        X_preprocessed = ((X_tensor - X_tensor.mean(dim=0)) / X_tensor.std(dim=0)).cpu().detach().numpy()

        return X_preprocessed, y_preprocessed
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        method that trains the model (adjusts weights and bias) 

        Parameters
        ----------
        X: input features matrix
        y: ground truth labels vector
        """

        # NOTE: bias will need to be initialized as np array with float64 dtype when using more than one node and/or more than one layer
        # initialize bias to zero
        self.b_ = 0

        # initialize weights to random small numbers (or zero)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(scale=0.01, size=X.shape[1]).astype(np.float64)

        # train until number of epochs is reached, or model converged, or after threhsold accuracy reached
        epoch_ctr = 0
        while epoch_ctr <= self.epochs:
            # calculate weighted inputs (z)
            z = np.dot(self.w_, np.transpose(X)) + self.b_
            
            # calculate linear activations
            linear_activations = z

            # calculate step activations
            step_activations = self.calculate_step_activations(linear_activations)

            # calculate current accuracy
            accuracy = np.count_nonzero(np.equal(step_activations, y)) / np.size(y)
            logger.info(f"accuracy at epoch {epoch_ctr}: {accuracy}")

            # break if model converged
            if accuracy == 1.0:
                logger.info(f"model converged at epoch {epoch_ctr}")
                break

            # break if accuracy reaches threshold accuracy
            if accuracy >= self.threshold_accuracy:
                logger.info(f"model reached/passed threshold accuracy at epoch {epoch_ctr} with accuracy {accuracy}")
                break

            # calculate delta_w and delta_b
            for i in range(y.size):
                delta_w, delta_b = self.calculate_delta(y[i], linear_activations[i], X[i])

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
        linear_activations = z
        step_activations = self.calculate_step_activations(linear_activations)

        # calculate accuracy
        accuracy = np.count_nonzero(np.equal(step_activations, y_test)) / np.size(y_test)

        return accuracy
    
    def calculate_step_activations(self, linear_activations: np.ndarray) -> np.ndarray:
        """
        method that calculates the step activation after the linear activation to determine model accuracy

        Parameters
        ----------
        linear_activations: linear activations matrix
        
        Returns
        -------
        step_activations: step activations matrix
        """

        step_activations = np.ndarray(linear_activations.shape)

        for i, linear_activation in enumerate(linear_activations):
            if linear_activation >= 0:
                step_activations[i] = 1
            else:
                step_activations[i] = 0

        return step_activations

    def calculate_delta(self, ground_truth: np.float64, linear_activation: np.float64, X: np.ndarray):
        """
        method that calculates delta_w and delta_b for a particular sample 

        Parameters
        ----------
        ground_truth: ground truth label for a particular sample
        linear_activations: linear activation for a particular sample
        X: input data for a particular sample

        Returns
        -------
        delta_w:
        delta_b:
        """

        # NOTE: is there a way to adjust each weight differently depending on how much each weight contributes to an accuracte prediction?
        delta_w = np.float64((self.eta * (np.subtract(ground_truth, linear_activation))) * X)

        delta_b = np.float64(self.eta * np.subtract(ground_truth, linear_activation))

        return delta_w, delta_b
