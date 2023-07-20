import numpy as np
from abc import ABC


class perceptron(ABC):
    """
    perceptron class
    """

    def __init__(self, eta: float=0.01, epochs: int=50, random_state: int=1) -> None:
        """
        
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    # add preprocess data method to handle missing values and normalize features (similar scale for all features)

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
        for i in range(self.epochs):
            # calculate weighted input (z)
            z = np.dot(self.w_, np.transpose(X)) + self.b_

            # calculate activation
            activations = self.calculate_activations(z)

            # calculate delta_w and delta_b
            delta_w, delta_b = self.calculate_deltas(y, activations, X)

            # apply delta_w and delta_b to w and b respectively
            self.w_ += delta_w
            self.b_ += delta_b

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        method that tests trained model against test set

        Parameters
        ----------
        X_test: test set input matrix
        y_test: ground truth labels vector
        Returns
        -------
        float: accuracy on test set
        """

        # calculate weighted input (z)
        z = np.dot(self.w_, np.transpose(X_test)) + self.b_

        # calculate activations
        activations = self.calculate_activations(z)

        # calculate accuracy
        accuracy = np.count_nonzero(np.equal(activations, y_test)) / np.size(y)

        return accuracy

    def calculate_activations(self, z: np.ndarray) -> np.ndarray:
        """
        method that calculates the activation for each input

        Parameters
        ----------
        z: weighted input matrix

        Returns
        -------
        np.ndarray: activations matrix
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
        np.ndarray, np.ndarray: delta w and delta b matrices
        """

        # calculate delta_w 
        delta_w = np.dot(self.eta * np.subtract(ground_truth, predicted), X)

        # calculate delta_b
        delta_b = np.sum(self.eta * np.subtract(ground_truth, predicted))

        return delta_w, delta_b


        

if __name__ == "__main__":
    ground_truth = np.array([1,0,0])
    predicted = np.array([1,1,0])
    X = np.array([[1,3,4,5], [1,1,2,6], [4,5,1,4]]) 
    eta = 1

    rgen = np.random.RandomState(1)
    w_ = rgen.normal(scale=0.01, size=X.shape[1])

    print(w_ + np.dot(eta * np.subtract(ground_truth,predicted), X))

    print(np.dot(eta * np.subtract(ground_truth,predicted), X))
    print(np.sum(eta * np.subtract(ground_truth, predicted)))

    activations = np.array([1,1,0,0,1,1])
    y = np.array([1,1,0,1,1,0])

    test = np.equal(activations, y)
    print(np.count_nonzero(test))

    print(np.size(test))