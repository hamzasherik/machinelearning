import numpy as np
from abc import ABC


class Perceptron(ABC):
    """
    perceptron class
    """

    def __init__(self, eta: float=0.01, epochs: int=50, random_state: int=1) -> None:
        """
        
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def lifecycle(self):
        """
        method that executes the lifecycle of the perceptron class
        """

        X_preprocessed, y_preprocessed = self.preprocess()

        self.train(X_preprocessed, y_preprocessed)

        #test_set_accuracy = self.test(X_test, y_test)

    # add preprocess data method to handle missing values and normalize features (similar scale for all features)
    def preprocess(self, X: np.ndarray, y: np.ndarray):
        """
        method that preprocesses input data

        Parameters
        ----------
        X:
        y:
        Returns
        -------
        X_preprocessed:
        y_preprocessed:
        """

        # convert labels column from str dtype to float dtype
        y_preprocessed = np.unique(y, return_inverse=True)[1]
 
        # TODO: preprocess X
        X_preprocessed = X

        return X_preprocessed, y_preprocessed

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        method that trains the model (adjusts weights and bias) 

        Parameters
        ----------
        X: training set input matrix
        y: ground truth labels vector
        """

        # initialize bias to zero
        self.b_ = 0

        # initialize weights to random small numbers (or zero)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(scale=0.01, size=X.shape[1])

        # TODO: train until number of epochs is reached OR convergence OR threshold accuracy reached
        # train until number of epochs is reached
        epoch_ctr = 0
        while epoch_ctr < self.epochs:
            # calculate weighted input (z)
            z = np.dot(self.w_, np.transpose(X)) + self.b_

            # calculate activation
            activations = self.calculate_activations(z)

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
        method that calculates the activation for each input

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
        delta_w = np.dot(self.eta * np.subtract(ground_truth, predicted), X)

        # calculate delta_b
        delta_b = np.sum(self.eta * np.subtract(ground_truth, predicted))

        return delta_w, delta_b


        

if __name__ == "__main__":
    labels = np.array(['hi', 'hi', 'bye'])
    print(np.unique(labels, return_inverse=True)[1])

    # create map 