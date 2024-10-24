import argparse
import os
from pathlib import Path

from preprocessing import load_data, prepare_features
from training import train_model
from utils import pickle_object


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = load_data(trainset_path)

    # Preprocess data
    X, y = prepare_features(df)

    # Train model
    model = train_model(X, y)

    # Pickle model
    output_dir = os.path.join(
        os.path.dirname(__file__), "../web_service/local_objects/"
    )
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")

    pickle_object(model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
