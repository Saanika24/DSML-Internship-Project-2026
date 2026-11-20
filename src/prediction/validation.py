def validate_required_columns(data, required_columns):
    """
    Check whether the dataset contains
    all required columns.

    Parameters:
        data: Input dataset
        required_columns: Expected column names

    Returns:
        True if all required columns exist.

    Raises:
        ValueError if columns are missing.
    """

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True