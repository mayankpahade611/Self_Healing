import subprocess
import sys

def retrain():
    print("Retraining triggered due to drift...")
    subprocess.run(
        [sys.executable, "training/train_model.py"],
        check=True
    )

if __name__ == "__main__":
    retrain()
