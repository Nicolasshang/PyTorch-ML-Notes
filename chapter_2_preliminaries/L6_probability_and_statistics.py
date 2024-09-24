import random
import torch
import matplotlib.pyplot as plt
from torch.distributions.multinomial import Multinomial
from L4_calculus import set_figsize


def coin_toss() -> None:
    num_tosses = 100

    # random.random() returns a probability between 0 and 1, so only choosing the values > 0.5
    # is effectively a 1/2 probability.
    heads = sum([random.random() > 0.5 for toss in range(num_tosses)])
    tails = num_tosses - heads

    print("heads, tails: ", [heads, tails])

    # The Multinomial() function takes in the number of draws and the associated probabilities,
    # and returns a vector detailing the number of occurences of each outcome.
    fair_probability = torch.tensor([0.5, 0.5])

    print(Multinomial(100, fair_probability).sample())

    # Dividing by the number of tosses gives us the frequency of each outcome.
    print(Multinomial(100, fair_probability).sample() / 100, "\n")


def law_of_large_numbers() -> None:
    # The law of large numbers states that as stochastic events repeat, our estimates will converge
    # to the true underlying probabilities.

    fair_probability = torch.tensor([0.5, 0.5])

    # The (10000,) specifies that the output tensor should be 10,000 elements long.
    counts = Multinomial(1, fair_probability).sample((10000,))

    # Find the total number of head or tails at each point in time
    cumulative_counts = counts.cumsum(dim=0)

    # Normalize the cumulative counts between 1 and 0 and convert to numpy array
    estimates = cumulative_counts / cumulative_counts.sum(dim=1, keepdims=True)
    estimates = estimates.numpy()

    set_figsize([4.5, 3.5])
    plt.plot(estimates[:, 0], label="Probability of heads")
    plt.plot(estimates[:, 1], label="Probability of tails")
    plt.axhline(y=0.5, color="black", linestyle="dashed")
    plt.xlabel("Samples")
    plt.ylabel("Estimated probability")
    plt.legend()

    plt.savefig("image/coin_flip_plot.png")


def main() -> None:
    coin_toss()
    law_of_large_numbers()


if __name__ == "__main__":
    main()
