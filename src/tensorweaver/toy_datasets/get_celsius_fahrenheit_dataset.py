import pathlib
import pandas as pd

CURRENT_DIR = pathlib.Path(__file__).resolve().parent


def get_celsius_fahrenheit_dataset():
    data_df = pd.read_csv(CURRENT_DIR / "celsius_fahrenheit_mapping_with_noise.csv")

    x = data_df["Celsius"].values.reshape(-1, 1)
    y = data_df["Fahrenheit"].values.reshape(-1, 1)

    return x, y
