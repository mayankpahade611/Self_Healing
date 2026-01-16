import numpy as np
import pandas as pd

def generate_live(n_samples=1000, drift=False):
    np.random.seed(7)
    X = np.random.rand(n_samples, 5)

    if drift:
        X[:, 0] = X[:, 0] * 3 + 2   # intentional drift

    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

if __name__ == "__main__":
    normal = generate_live(drift=False)
    drifted = generate_live(drift=True)

    normal.to_csv("data/raw/live_normal.csv", index=False)
    drifted.to_csv("data/raw/live_drifted.csv", index=False)
