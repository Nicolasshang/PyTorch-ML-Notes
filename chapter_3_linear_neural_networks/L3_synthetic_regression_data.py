import random
import torch

from util.DataModule import DataModule


class SyntheticRegressionData(DataModule):
    def __init__(
        self, weights, biases, noise=0.01, num_train=1000, num_val=1000, batch_size=32
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # num_train is the amount of data to read if training, and num_val is the amount to
        # read if not.
        n = num_train + num_val
        self.X = torch.randn(n, len(weights))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, weights.reshape((-1, 1))) + biases + noise

    # Takes a batch size, a matrix of features, and a vector of labels, and generates minibatches
    # of size batch_size.
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


if __name__ == "__main__":
    data = SyntheticRegressionData(weights=torch.tensor([2, -3.4]), biases=4.2)

    # Each feature is a vector while each label is a scalar.
    print("features:", data.X[0], "\nlabel:", data.y[0])

    # iter() returns an iterator from the object, next() returns its next element.
    X, y = next(iter(data.train_dataloader()))
    print("X shape:", X.shape, "\ny shape:", y.shape)

    # X is a 32-long tensor of vectors, y is a 32-long tensor of scalars.
    print("X:", X, "\ny:", y)

    # Since the torch DataLoader supports the built-in __len__ attribute, we can query the
    # batch size.
    print("batch size:", len(data.train_dataloader()))
