import torch

from util.utils import HyperParameters
from util.Module import Module
from util.SyntheticRegressionData import SyntheticRegressionData
from util.Trainer import Trainer


class StochasticGradientDescent(HyperParameters):
    def __init__(self, params, learning_rate):
        self.save_hyperparameters()

    # For each param, calculate the error from the gradient and the learning rate
    # and then subtract that from it to minimize loss.
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    # Zero out all the parameter gradients
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class LinearRegression(Module):
    def __init__(self, num_inputs, learning_rate, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        # Initialize weights by drawing from a random distribution with mean 0 and standard deviation
        # of 0.01.
        self.weights = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)

        # Initialize biases to be 0.
        self.biases = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        # Multiply the design matrix by the weights and add the constant biases
        return torch.matmul(X, self.weights) + self.biases

    def loss(self, y_hat, y):
        # The squared error loss function
        l = (y_hat - y) ** 2 / 2

        # mean() returns the average of all the elements of the input tensor.
        return l.mean()

    # Returns an instance of the StochasticGradientDescent class
    def configure_optimizers(self):
        return StochasticGradientDescent(
            [self.weights, self.biases], self.learning_rate
        )


if __name__ == "__main__":
    model = LinearRegression(2, learning_rate=0.03)

    # Create some ground truth values with the SyntheticRegressionData class
    data = SyntheticRegressionData(weights=torch.tensor([2, -3.4]), biases=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)
