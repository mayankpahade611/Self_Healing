import mlflow

MODEL_NAME = "self_healing_regressor"

def get_latest_two_versions():
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    versions = sorted(versions, key=lambda v: int(v.version))
    return versions[-2], versions[-1]

def get_mse_for_version(version):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(version.run_id)
    return run.data.metrics.get("mse")

def evaluate_and_promote():
    client = mlflow.tracking.MlflowClient()

    old, new = get_latest_two_versions()

    old_mse = get_mse_for_version(old)
    new_mse = get_mse_for_version(new)

    print(f"Old model MSE: {old_mse}")
    print(f"New model MSE: {new_mse}")

    if new_mse is None or old_mse is None:
        print("⚠️ Metrics missing — skipping promotion")
        return

    if new_mse < old_mse:
        print("New model better — promoting to Production")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new.version,
            stage="Production"
        )
    else:
        print("New model worse — keeping existing model")

if __name__ == "__main__":
    evaluate_and_promote()
