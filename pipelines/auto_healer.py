import time
import requests
from pipelines.healer import heal

DRIFT_THRESHOLD = 1  

def check_drift():
    # In real systems, Prometheus is queried.
    # Here we simulate the signal.
    return True  # drift detected

if __name__ == "__main__":
    while True:
        if check_drift():
            print("⚠️ Drift signal received")
            heal(drift_detected=True)
        time.sleep(300)  # check every 5 minutes
