import torch

from torch import nn
from util.DataModule import DataModule
from util.Trainer import Trainer
from L5_concise_implementation import LinearRegression


class Data(DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)

    # Calculate the l2 penalty, which is the sum of the weight tensor norm squared divided by 2.
    def l2_penalty(w):
        return torch.sum(w**2) / 2


class WeightDecay(LinearRegression):
    def __init__(self, weight_decay, learning_rate):
        super().__init__(learning_rate)
        self.save_hyperparameters()
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.SGD(
            [
                {
                    "params": self.net.weight,
                    "weight_decay": self.weight_decay,
                },
                {"params": self.net.bias},
            ],
            lr=self.learning_rate,
        )


if __name__ == "__main__":
    data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
    trainer = Trainer(max_epochs=10)

    # Train the actual model
    model = WeightDecay(weight_decay=3, learning_rate=0.01)
    trainer.fit(model, data)

    print("l2 norm of w:", float(torch.sum(model.get_weights_biases()[0] ** 2) / 2))
