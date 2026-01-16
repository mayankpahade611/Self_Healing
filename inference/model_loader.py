import mlflow.pyfunc

MODEL_NAME = "self_healing_regressor"
MODEL_STAGE = "None"  # using latest version locally

def load_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
