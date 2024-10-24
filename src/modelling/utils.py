import pickle


def pickle_object(obj, filename: str):
    """Pickle any object (e.g., model or encoder) and save it to the specified file."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filename}")
