from pipelines.retrain_pipeline import retrain
from pipelines.evaluate_model import evaluate_and_promote

def heal(drift_detected: bool):
    if drift_detected:
        retrain()
        evaluate_and_promote()

if __name__ == "__main__":
    heal(drift_detected=True)
