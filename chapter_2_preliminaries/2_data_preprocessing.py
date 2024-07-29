import torch
import os
import pandas as pd


def ReadData() -> None:
    os.makedirs(os.path.join("..", "data"), exist_ok=True)
    data_file = os.path.join("..", "data", "house_tiny.csv")
    with open(data_file, "w") as f:
        f.write(
            """NumRooms,RoofType,Price
            NA,NA,127500
            2,NA,106000
            4,Slate,178100
            NA,NA,140000"""
        )

    # read the data with pandas
    data = pd.read_csv(data_file)
    print(data, "\n")

    # the common heuristic of adding a separate category for NaN values
    inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = pd.get_dummies(inputs, dummy_na=True, dtype=f)
    print(inputs, "\n")

    # the common heuristic of averaging the entire column for NaN values
    inputs = inputs.fillna(inputs.mean())
    print(inputs, "\n")

    # converting both data soruces to tensors
    x = torch.tensor(inputs.to_numpy(dtype=float))
    y = torch.tensor(targets.to_numpy(dtype=float))
    print(x, y, sep="\n")


if __name__ == "__main__":
    ReadData()
