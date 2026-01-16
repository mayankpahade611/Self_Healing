import torch
import pandas as pd
import numpy as np
from training.drift_autoencoder import DriftAutoencoder

AUTOENCODER_PATH = "training/drift_autoencoder.pt"

class DriftChecker:
    def __init__(self, threshold):
        self.model = DriftAutoencoder()
        self.model.load_state_dict(torch.load(AUTOENCODER_PATH))
        self.model.eval()
        self.threshold = threshold

    def check(self, features: np.ndarray) -> bool:
        x = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(x)
        error = ((x - recon) ** 2).mean().item()
        return error > self.threshold
