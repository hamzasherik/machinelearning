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

        # calculate weighted input (z)
        z = np.dot(self.w_, np.transpose(X)) + self.b_

        # calculate activation
        activations = self.calculate_activations(z)

        # NOTE: can split into 2 methods
        # calculate delta_w and delta_b
        delta_w, delta_b = self.calculate_deltas(y, activations, X)

        # apply delta w and delta b to w and b respectively

        # rerun until prediction converges (linear decision boundary where model accuracy is 100%) or after threshold accuracy is reached or after a number of epochs

        # return models (weights and bias)
        #self.w_ =
        #self.b_ = 

    def test(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        method that tests trained model against test set

        Parameters
        ----------
        X: test set input matrix
        y: ground truth labels vector
        """

        # calculate weighted input (z)

        # calculate activation (unit step function)

        # determine if prediction == ground truth label and increment correct counter by 1

        # compare correct counter to length of y (correct counter / len(y) * 100) and return result

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

        """

        # calculate delta_w 
        delta_w = np.dot(self.eta * np.subtract(ground_truth, predicted), X)

        # calculate delta_b
        # NOTE: this returns an array of biases, 1 bias for each sample. Is that correct or should bias be a single integer element.
        delta_b = self.eta * np.subtract(ground_truth, predicted)

        return delta_w, delta_b


        

if __name__ == "__main__":
    ground_truth = np.array([1,0,0])
    predicted = np.array([1,1,0])
    X = np.array([[1,3,4,5], [1,1,2,6], [4,5,1,4]]) 
    eta = 1

    print(np.dot(eta * np.subtract(ground_truth,predicted), X))
    print(eta * np.subtract(ground_truth, predicted))