import torch
import numpy as np

from torch import nn
from util.Module import Module
from util.SyntheticRegressionData import SyntheticRegressionData
from util.Trainer import Trainer


class LinearRegression(Module):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        # Specify a lazy linear regression model with one output feature
        self.net = nn.LazyLinear(1)

        # Fill the network with default weights drawn from a normal distribution
        self.net.weight.data.normal_(0, 0.01)

        # Zero-initialize the weights
        self.net.bias.data.fill_(0)

    # invoke the __call__ method of the predefined layers to compute the outputs
    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        # MSELoss is the mean squared error function.
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        # Use SGD to optimize the parameters
        return torch.optim.SGD(self.parameters(), self.learning_rate)

    def get_weights_biases(self):
        return (self.net.weight.data, self.net.bias.data)


if __name__ == "__main__":
    model = LinearRegression(learning_rate=0.03)
    data = SyntheticRegressionData(weights=torch.tensor([2, -3.4]), biases=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)
    weights, biases = model.get_weights_biases()

    print(f"weights: {weights}, biases: {biases}")
    print(f"weights loss: {data.weights - weights.reshape(data.weights.shape)}")
    print(f"bias loss: {data.biases - biases}")
