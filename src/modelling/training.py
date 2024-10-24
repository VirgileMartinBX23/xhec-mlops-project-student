import numpy as np
import scipy.sparse
from prefect import task
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task(name="Train model")
def train_model(X_train: scipy.sparse.csr_matrix, y_train: np.ndarray):
    """Train a Linear Regression model on the given training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


@task(name="Evaluate model")
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for two arrays"""
    return mean_squared_error(y_true, y_pred, squared=False)
