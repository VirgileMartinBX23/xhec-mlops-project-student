import os
from typing import Optional

import numpy as np
from predicting import predict
from prefect import flow
from preprocessing import process_data
from sklearn.linear_model import LinearRegression
from training import evaluate_model, train_model
from utils import load_pickle, save_pickle


@flow(name="Train model")
def model_training_flow(trainset_path: str, save_model_path: str):
    """Prefect flow to manage model training."""
    X_train, X_test, y_train, y_test = process_data(trainset_path)
    model = train_model(X_train, y_train)
    y_pred = predict(X_test, model)
    rmse = evaluate_model(y_test, y_pred)

    if save_model_path is not None:
        save_pickle(os.path.join(save_model_path, "model.pkl"), model)

    return {"model": model, "rmse": rmse}


@flow(name="Batch predict", retries=1, retry_delay_seconds=30)
def batch_predict_workflow(
    input_filepath: str,
    model: Optional[LinearRegression] = None,
    artifacts_filepath: Optional[str] = None,
) -> np.ndarray:
    """Make predictions on a new dataset"""
    if model is None:
        model = load_pickle(os.path.join(artifacts_filepath, "model.pkl"))

    X = process_data(filepath=input_filepath, with_target=False)
    y_pred = predict(X, model)

    return y_pred


if __name__ == "__main__":
    # Run the flow directly if executed as a script
    model_training_flow(
        "abalone/abalone.data", "src/web_service/local_objects/model.pkl"
    )
