import numpy as np 
import pandas as pd


def generate_data(n_samples = 5000):
    np.random.seed(42)
    X = np.random.rand(n_samples, 5)
    noise = np.random.normal(0, 0.1, size=n_samples)
    y = (
        3 * X[:, 0]
        + 2 * X[:, 1]
        - 1.5 * X[:, 2]
        + noise
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(target=y)

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/raw/train.csv", index=False)
