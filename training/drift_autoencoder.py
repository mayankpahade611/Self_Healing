import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class DriftAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 5)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder():
    df = pd.read_csv("data/raw/train.csv")
    X = torch.tensor(df.drop("target", axis=1).values, dtype=torch.float32)

    model = DriftAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(40):
        optimizer.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "training/drift_autoencoder.pt")
    return model

def reconstruction_error(model, df):
    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X)
    return ((X - recon) ** 2).mean(dim=1).numpy()

if __name__ == "__main__":
    model = train_autoencoder()

    normal = pd.read_csv("data/raw/live_normal.csv")
    drifted = pd.read_csv("data/raw/live_drifted.csv")

    err_normal = reconstruction_error(model, normal).mean()
    err_drifted = reconstruction_error(model, drifted).mean()

    print("Reconstruction Error (Normal):", err_normal)
    print("Reconstruction Error (Drifted):", err_drifted)

    threshold = err_normal * 2
    print("Drift detected?" , err_drifted > threshold)
