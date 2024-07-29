import torch


def Tensors() -> None:
    # arange() demonstration
    x = torch.arange(12, dtype=torch.float32)
    print(x)

    # reshape() demonstration
    print(x.shape)

    X = x.reshape(3, 4)
    print(X)

    Y = x.reshape(-1, 4)  # Also works (first parameter is inferred)
    print(Y)

    # other functions demonstration
    print(torch.zeros((2, 3, 4)), torch.ones((2, 3, 4)), torch.randn(3, 4), sep="\n")


def Indexing() -> None:
    x = torch.randn(3, 4)
    x[:2, :] = 12  # sets first two rows to 12
    print(x)


def Operations() -> None:
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])

    # lifting mathamatical operations to identically shaped vectors
    print(x + y, x - y, x * y, x / y, x**y, sep="\n")

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

    # Concatenating the two tensors row-wise (notice how the rows/columns are twice as long as the individual tensors in the two cases respectively)
    print(torch.cat((x, y), dim=0), torch.cat((x, y), dim=1), sep="\n")

    # returns a tensor of booleans
    print(x == y)


def Broadcasting() -> None:
    x = torch.arange(3).reshape(3, 1)
    y = torch.arange(2).reshape(1, 2)

    # implicitly broadcasts into a 3*2 matrix
    print(x + y)


def SavingMemory() -> None:
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])

    z = torch.zeros_like(y)
    print("memory address of z: ", id(z))
    z[:] = x + y  # z[:] is slice notation for fetching everything
    print("memory address of z: ", id(z))  # should be the same


def Conversion() -> None:
    x = torch.tensor([1.0, 2, 4, 8])

    a = x.numpy()
    b = torch.from_numpy(a)

    # a should be a numpy.ndarray, b should be torch.Tensor
    print(type(a), type(b))

    a = torch.tensor([3.5])

    # converts the size-1 tensor into a scalar
    print(a.item())


if __name__ == "__main__":
    Tensors()
    Indexing()
    Operations()
    Broadcasting()
    SavingMemory()
    Conversion()
