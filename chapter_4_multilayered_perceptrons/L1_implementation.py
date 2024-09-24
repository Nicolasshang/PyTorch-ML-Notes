import os
import random
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # Flatten() converts 3D image representations into 1D format
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    # Forward pass
    def forward(self, x):
        return self.layers(x)


def train(mlp, trainloader, loss_function, optimizer, epochs):
    # A total of 5 passes over the entire dataset
    for epoch in range(epochs):
        print(f"starting epoch {epoch + 1}")

        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad()

            outputs = mlp(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()

            optimizer.step()

            # Print statistics
            if i % 500 == 499:
                print(f"Loss after mini-batch {i + 1}: {loss.item()}")


def visualize(trainloader):
    labels_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    train_features, train_labels = next(iter(trainloader))
    sample_idx = random.randint(0, 9)

    img = train_features[sample_idx].squeeze()
    label = train_labels[sample_idx]

    plt.title(labels_map[label.item()])
    plt.imshow(img.T)
    plt.show()


if __name__ == "__main__":
    # Set the seed for generating a random number
    torch.manual_seed(49)

    # Prepare the CIFAR dataset
    # os.getcwd gets the current working directory
    dataset = CIFAR10(
        root="C:/Users/23shaang/Desktop/pytorch-ML-notes/data",
        transform=transforms.ToTensor(),
    )

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=1
    )

    mlp = MLP()

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    train(mlp, trainloader, loss_function, optimizer, epochs=5)

    print("training has finished.")
