import pandas as pd


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


def prepare_features(df: pd.DataFrame):
    """One-hot encode 'Sex' column and split data into features and target."""
    df_encoded = pd.get_dummies(df, columns=["Sex"], drop_first=True)
    X = df_encoded.drop(columns=["Rings"])
    y = df_encoded["Rings"]
    return X, y
