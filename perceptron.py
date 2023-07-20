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

        # TODO: confirm for-in changes sample in z. Confirm for-in iterates through each element in a numpy array. Move activation check to a separate method.
        # calculate activation
        for sample in z:
            if sample >= 0:
                sample = 1
            else:
                sample = 0

        # TODO: confirm len() can be used on numpy array.
        # TODO: z shouldn't be in calculation of delta_w and delta_b below
        # calculate delta w and delta b
        for label in range(len(z)):
            delta_w = self.eta * (z[label] - y[label]) * X[label]

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

if __name__ == "__main__":
    arr = np.array([1,2,3,4])
    print(arr + 2)