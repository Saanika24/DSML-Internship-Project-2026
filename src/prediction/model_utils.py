
from pathlib import Path
import joblib


def save_model(model, file_path):
    """
    Save a trained model or preprocessing pipeline.

    Parameters:
        model: Trained model or pipeline.
        file_path: Destination path.

    Returns:
        Path of the saved model.
    """

    if model is None:
        raise ValueError("Cannot save an empty model.")

    file_path = Path(file_path)

    if not file_path.suffix:
        file_path = file_path.with_suffix(".joblib")

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, file_path)

    print(f"Model saved successfully: {file_path}")

    return file_path


def load_model(file_path):
    """
    Load a previously saved model.

    Parameters:
        file_path: Location of the saved model.

    Returns:
        Loaded model.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {file_path}"
        )

    model = joblib.load(file_path)

    print(f"Model loaded successfully: {file_path}")

    return model
