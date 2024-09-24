import torch


def autograd() -> None:
    # requires_grad=True means that the autograd should record the operations on the current tensor.
    x = torch.arange(4.0, requires_grad=True)

    # Calculate the function of x and return the result to y
    y = 2 * torch.dot(x, x)

    # Take the gradient of y with respect to x by calling the backward() method
    y.backward()

    print(x.grad)
    print(
        x.grad == 4 * x
    )  # The gradient is 4x, so this returns tensor([True, True, True, True])

    x.grad.zero_()  # Fills the gradient tensor with zeroes (doesn't change x itself)
    y = x.sum().backward()
    print(x.grad, "\n")  # Should print tensor([1., 1., 1., 1.])


def detachment() -> None:
    x = torch.arange(4.0, requires_grad=True)
    y = x * x

    # Create a variable whose provenence (how it was created) on the computation graph is wiped out.
    # Since u has no ancestors, the gradients will not flow from u to x.
    # This is called a detachment because it is not connected with the rest of the graph.
    u = y.detach()
    z = u * x

    # The gradient of z with respect to x
    z.sum().backward()

    # Returns u, not the derivative of z.
    print(x.grad)
    print(x.grad == u, "\n")


def gradient_control_flow() -> None:
    # Define a function f(a) that is lienar and has piecewise defined scale.
    def f(a: torch.tensor) -> torch.tensor:
        b = a * 2
        while b.norm() < 1000:
            b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
        return c

    # Define a tensor a filled with random numbers from a normal distribution
    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    d.backward()

    # Since f(a) is linear and scales with a, f(a) / a must be a vactor of constant entries
    # moreover, f(a) / a must match the gradient of f(a) with respect to a.
    print(a.grad == d / a)  # Returns tensor([True])


def main():
    autograd()
    detachment()
    gradient_control_flow()


if __name__ == "__main__":
    main()
