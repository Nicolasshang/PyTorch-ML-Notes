import torch
import numpy as np
from util.utils import add_to_class, HyperParameters, ProgressBoard


def decorator_demo() -> None:
    class A:
        def __init__(self) -> None:
            self.b = 1

    @add_to_class(A)
    def do(self) -> None:
        print("the value of attribute b is: ", self.b)

    # Above code is equivalent to:

    # def do(self) -> None:
    #     print("the value of attribute b is: ", self.b)
    # setattr(A, "do", do)

    a = A()
    a.do()  # Should print 1


def hyperparameters_demo() -> None:
    class B(HyperParameters):
        def __init__(self, a, b, c) -> None:
            self.save_hyperparameters(ignore=["c"])
            print("self.a =", self.a, "self.b =", self.b)
            print("There is no self.c =", not hasattr(self, "c"))  # returns false

    b = B(a=1, b=2, c=3)


def progressboard_demo() -> None:
    board = ProgressBoard("x")

    for x in np.arange(0, 10, 0.1):
        board.draw(x, np.sin(x), "sin", every_n=2)
        board.draw(x, np.cos(x), "cos", every_n=10)


if __name__ == "__main__":
    decorator_demo()
    hyperparameters_demo()
    progressboard_demo()
