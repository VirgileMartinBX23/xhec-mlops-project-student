import pandas as pd
from prefect import flow
from sklearn.model_selection import train_test_split


def load_data(trainset_path: str) -> pd.DataFrame:
    """Load the dataset from the given path."""
    column_names = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings",
    ]

    abalone_df = pd.read_csv(trainset_path, names=column_names)
    return abalone_df


@flow(name="Preprocess data")
def process_data(filepath: str, with_target: bool = True):
    """
    Load data from a parquet file
    Compute target (duration column) and apply threshold filters (optional)
    Turn features to sparce matrix
    :return The sparce matrix, the target' values and the
    dictvectorizer object if needed.
    """
    df = load_data(filepath)
    if with_target:
        df_encoded = pd.get_dummies(df, columns=["Sex"], drop_first=True)
        X = df_encoded.drop(columns=["Rings"])
        y = df_encoded["Rings"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    else:
        X = pd.get_dummies(df, columns=["Sex"], drop_first=True)
        return X
