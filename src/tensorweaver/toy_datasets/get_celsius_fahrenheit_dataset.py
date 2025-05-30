import pathlib
import numpy as np
import csv

CURRENT_DIR = pathlib.Path(__file__).resolve().parent


def get_celsius_fahrenheit_dataset():
    csv_path = CURRENT_DIR / "celsius_fahrenheit_mapping_with_noise.csv"

    celsius_data = []
    fahrenheit_data = []

    with open(csv_path, "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            celsius_data.append(float(row["Celsius"]))
            fahrenheit_data.append(float(row["Fahrenheit"]))

    x = np.array(celsius_data, dtype=np.float32).reshape(-1, 1)
    y = np.array(fahrenheit_data, dtype=np.float32).reshape(-1, 1)

    return x, y
