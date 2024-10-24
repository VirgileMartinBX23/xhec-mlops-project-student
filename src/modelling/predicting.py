import pickle

import numpy as np
import scipy.sparse
from prefect import task
from sklearn.linear_model import LinearRegression


def load_model(filepath="model.pkl"):
    """Load a trained model from a pickle file."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


@task(name="Make predictions")
def predict(X: scipy.sparse.csr_matrix, model: LinearRegression) -> np.ndarray:
    """Make predictions using the trained model."""
    predictions = model.predict(X)
    return predictions
