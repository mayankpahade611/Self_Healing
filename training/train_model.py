import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.pytorch


class MLP(nn.Module):
    def __init__(self):
        super().__init__()   
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)
    

def main():
    df = pd.read_csv("data/raw/train.csv")
    X = df.drop("target", axis=1).values
    y = df["target"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)


    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    mlflow.set_experiment("self-healing-ml")

    with mlflow.start_run():
        for epoch in range(30):
            optimizer.zero_grad()
            preds = model(X_train)
            loss = loss_fn(preds, y_train)
            loss.backward()
            optimizer.step()

        val_preds = model(X_val).detach().numpy()
        mse = mean_squared_error(y_val.numpy(), val_preds)

        mlflow.log_param("epochs", 30)
        mlflow.log_metric("mse", mse)

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="self_healing_regressor"
        )

        print("Validation MSE:", mse)

if __name__ == "__main__":
    main()
