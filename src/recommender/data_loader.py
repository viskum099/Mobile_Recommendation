
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "name", "rating", "price", "camera", "display_type",
    "display_size", "battery", "storage", "ram", "weight",
    "processor",
]


def load_products(file_path: str | Path) -> pd.DataFrame:
    """Load the Excel file and validate its required columns."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Product file not found: {file_path}"
        )

    df = pd.read_excel(file_path)
    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["name"] = df["name"].astype("string").str.strip()

    if df["name"].isna().any() or df["name"].eq("").any():
        raise ValueError("Some products have missing names.")

    return df
