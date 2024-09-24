import torch

from util.plot import plot


def vanishing_gradients():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x))

    # Here, the gradient completely vanishes both when the inputs are large/small.
    plot(
        x.detach().numpy(),
        [y.detach().numpy(), x.grad.numpy()],
        legend=["sigmoid", "gradient"],
        figsize=[4.5, 2.5],
        savepath="C:/Users/23shaang/Desktop/pytorch-ML-notes/image/vanishing_gradients_plot.png",
    )


def exploding_gradients():
    M = torch.normal(0, 1, size=(4, 4))

    # If we are handing just one matrix, the model has a change to converge.
    print("just one matrix:\n", M)

    # Multiplying the matrix by itself 100 times. The numerical values explode. If this happens
    # at the initialization of a deep network, there is no chance that it converges.
    for i in range(100):
        M = M @ torch.normal(0, 1, size=(4, 4))
    print("after multiplying 100 matrices:\n", M)


if __name__ == "__main__":
    vanishing_gradients()
    exploding_gradients()
