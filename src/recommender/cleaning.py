
import re

import pandas as pd


def _number(value):
    """Extract a number from a numeric value or a simple unit string."""
    if pd.isna(value):
        return None

    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize product fields while preserving review flags."""
    result = df.copy()

    # Prices with a currency symbol need manual verification.
    raw_price = result["price"].astype("string").str.strip()
    result["price_needs_review"] = raw_price.str.contains(
        r"[$€£]", regex=True, na=False
    )

    # Numeric prices are interpreted as INR, per the project assumption.
    result["price"] = raw_price.str.replace(
        r"[₹,\s]", "", regex=True
    ).str.replace(r"[$€£]", "", regex=True)
    result["price"] = pd.to_numeric(result["price"], errors="coerce")

    # Convert numeric fields into consistent numeric types.
    for column in ["rating", "camera", "display_size",
                   "battery", "ram", "weight"]:
        result[column] = result[column].map(_number)

    # Convert storage to GB: 1 TB = 1024 GB.
    def storage_to_gb(value):
        if pd.isna(value):
            return None

        text = str(value).strip().upper()
        number = _number(text)

        if number is None:
            return None
        if "TB" in text:
            return number * 1024
        return number

    result["storage_gb"] = result["storage"].map(storage_to_gb)

    # Treat explicit unknown markers as missing information.
    result["processor"] = result["processor"].astype("string").str.strip()
    unknown_processor = result["processor"].str.lower().isin(
        ["not mentioned", "unknown", "n/a", ""]
    )
    result.loc[unknown_processor, "processor"] = pd.NA

    result["display_type"] = (
        result["display_type"].astype("string").str.strip()
    )

    # Zero camera values are suspicious; do not treat them as real specs.
    result["camera_needs_review"] = result["camera"].eq(0)
    result.loc[result["camera_needs_review"], "camera"] = pd.NA

    # Validate plausible ranges without silently deleting records.
    result["rating_needs_review"] = ~result["rating"].between(0, 5)
    result["price_needs_review"] |= (
        result["price"].isna() | result["price"].le(0)
    )

    return result
