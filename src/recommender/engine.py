
import pandas as pd

DEFAULT_WEIGHTS = {
    "rating": 0.25,
    "camera": 0.20,
    "battery": 0.15,
    "ram": 0.15,
    "storage": 0.10,
    "price": 0.15,
}


def recommend_products(
    products: pd.DataFrame,
    budget: float,
    min_rating: float = 0,
    min_ram_gb: float = 0,
    min_storage_gb: float = 0,
    preferred_display: str | None = None,
    processor_query: str | None = None,
    limit: int = 5,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Filter eligible phones and rank them with explainable scores."""

    if budget <= 0:
        raise ValueError("Budget must be greater than zero.")

    if not 0 <= min_rating <= 5:
        raise ValueError("Minimum rating must be between 0 and 5.")

    if min_ram_gb < 0 or min_storage_gb < 0:
        raise ValueError("RAM and storage requirements cannot be negative.")

    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    # Use the existing default weights unless the user supplies custom ones.
    active_weights = (
        DEFAULT_WEIGHTS.copy() if weights is None else weights.copy()
    )

    if set(active_weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(
            "Weights must include rating, camera, battery, ram, storage, and price."
        )

    if any(
        not isinstance(value, (int, float))
        or not pd.notna(value)
        or value < 0
        for value in active_weights.values()
    ):
        raise ValueError("Weights must be non-negative numbers.")

    weight_total = sum(active_weights.values())

    if weight_total <= 0:
        raise ValueError("At least one weight must be greater than zero.")

    # Normalize weights so their total is 1, even when slider values
    # add up to something other than 100.
    active_weights = {
        feature: value / weight_total
        for feature, value in active_weights.items()
    }

    df = products.copy()

    # Exclude invalid or unverified prices and phones over budget.
    eligible = df.loc[
        df["price"].notna()
        & ~df["price_needs_review"].fillna(True)
        & (df["price"] > 0)
        & (df["price"] <= budget)
    ].copy()

    if min_rating > 0:
        eligible = eligible.loc[
            eligible["rating"].notna()
            & (eligible["rating"] >= min_rating)
        ]

    if min_ram_gb > 0:
        eligible = eligible.loc[
            eligible["ram"].notna()
            & (eligible["ram"] >= min_ram_gb)
        ]

    if min_storage_gb > 0:
        eligible = eligible.loc[
            eligible["storage_gb"].notna()
            & (eligible["storage_gb"] >= min_storage_gb)
        ]

    if preferred_display:
        eligible = eligible.loc[
            eligible["display_type"].str.contains(
                preferred_display,
                case=False,
                na=False,
                regex=False,
            )
        ]

    if processor_query and processor_query.strip():
        eligible = eligible.loc[
            eligible["processor"]
            .astype("string")
            .str.contains(
                processor_query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    eligible = eligible.copy()

    # Return an empty result with the expected columns if no phones match.
    
    # Convert each feature into a score between 0 and 1.
    # These are the existing heuristic scales, retained to preserve ranking.
    scores = pd.DataFrame(index=eligible.index)

    scores["rating"] = (
        pd.to_numeric(eligible["rating"], errors="coerce") / 5
    ).clip(0, 1)

    scores["camera"] = (
        pd.to_numeric(eligible["camera"], errors="coerce") / 200
    ).clip(0, 1)

    scores["battery"] = (
        pd.to_numeric(eligible["battery"], errors="coerce") / 7000
    ).clip(0, 1)

    scores["ram"] = (
        pd.to_numeric(eligible["ram"], errors="coerce") / 24
    ).clip(0, 1)

    scores["storage"] = (
        pd.to_numeric(eligible["storage_gb"], errors="coerce") / 2048
    ).clip(0, 1)

    scores["price"] = (
        1 - eligible["price"] / budget
    ).clip(0, 1)

    # Keep the previous rule: missing feature scores receive a neutral 0.5.
    scores = scores.fillna(0.5)

    # Save the normalized feature scores and their weighted contributions.
    for feature in DEFAULT_WEIGHTS:
        eligible[f"{feature}_normalized"] = scores[feature]
        eligible[f"{feature}_contribution"] = (
            scores[feature] * active_weights[feature] * 100
        )

    # The total score is the sum of the six weighted contributions.
    contribution_columns = [
        f"{feature}_contribution" for feature in DEFAULT_WEIGHTS
    ]

    eligible["match_score"] = eligible[contribution_columns].sum(axis=1)

    def explain(row):
        reasons = [
            f"Price: ${row['price']:,.2f}",
            (
                f"Rating: {row['rating']:.1f}/5"
                if pd.notna(row["rating"])
                else "Rating unavailable"
            ),
        ]

        if pd.notna(row["camera"]):
            reasons.append(f"Camera: {row['camera']:g} MP")
        else:
            reasons.append("Camera specification unavailable")

        if pd.notna(row["battery"]):
            reasons.append(f"Battery: {row['battery']:g} mAh")

        if pd.notna(row["ram"]):
            reasons.append(f"RAM: {row['ram']:g} GB")

        if pd.notna(row["storage_gb"]):
            reasons.append(f"Storage: {row['storage_gb']:g} GB")

        if pd.isna(row["processor"]):
            reasons.append("Processor information unavailable")

        return " | ".join(reasons)

    eligible["recommendation_reason"] = eligible.apply(explain, axis=1)

    # Define the specification fields used to measure data completeness.
    quality_columns = [
        column
        for column in [
            "rating",
            "camera",
            "battery",
            "ram",
            "storage_gb",
            "weight",
            "processor",
            "display_type",
            "display_size",
        ]
        if column in eligible.columns
    ]

    # Count how many specification fields contain actual values.
    available_count = eligible[quality_columns].notna().sum(axis=1)

    # Calculate the percentage of available specification fields.
    if quality_columns:
        eligible["data_completeness"] = (
            available_count / len(quality_columns) * 100
        ).round(1)

        eligible["missing_specifications"] = (
            len(quality_columns) - available_count
        )
    else:
        # Handle datasets that contain none of the expected fields.
        eligible["data_completeness"] = 0.0
        eligible["missing_specifications"] = 0

    return (
        eligible.sort_values(
            ["match_score", "rating"],
            ascending=[False, False],
            na_position="last",
        )
        .head(limit)
        .reset_index(drop=True)
    )
