import pickle

import pandas as pd
from preprocessing import prepare_features


def load_model(filepath="model.pkl"):
    """Load a trained model from a pickle file."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def make_predictions(model, data: pd.DataFrame):
    """Make predictions using the trained model."""
    X, _ = prepare_features(data)
    predictions = model.predict(X)
    return predictions
