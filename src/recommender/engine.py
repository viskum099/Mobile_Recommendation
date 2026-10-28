
import pandas as pd


def recommend_products(
    products: pd.DataFrame,
    budget: float,
    min_rating: float = 0,
    min_ram_gb: float = 0,
    min_storage_gb: float = 0,
    preferred_display: str | None = None,
    limit: int = 5,
) -> pd.DataFrame:
    """Filter by hard constraints and rank remaining products."""

    if budget <= 0:
        raise ValueError("Budget must be greater than zero.")
    if not 0 <= min_rating <= 5:
        raise ValueError("Minimum rating must be between 0 and 5.")
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    df = products.copy()

    # Hard constraints: never recommend above budget.
    eligible = df.loc[
        df["price"].notna()
        & ~df["price_needs_review"]
        & (df["price"] <= budget)
        & (df["rating"] >= min_rating)
        & (df["ram"] >= min_ram_gb)
        & (df["storage_gb"] >= min_storage_gb)
    ].copy()

    if preferred_display:
        eligible = eligible.loc[
            eligible["display_type"].str.contains(
                preferred_display, case=False, na=False, regex=False
            )
        ].copy()

    if eligible.empty:
        return eligible.assign(match_score=pd.Series(dtype=float))

    # Normalize each numeric feature to a 0–1 scale.
    def normalize(series, higher_is_better=True):
        values = pd.to_numeric(series, errors="coerce")
        low, high = values.min(), values.max()

        if pd.isna(low) or pd.isna(high) or low == high:
            return pd.Series(0.5, index=series.index)

        score = (values - low) / (high - low)
        return score if higher_is_better else 1 - score

    # Transparent weighted score. Missing specs receive a neutral score.
    eligible["score_rating"] = normalize(eligible["rating"])
    eligible["score_camera"] = normalize(eligible["camera"]).fillna(0.5)
    eligible["score_battery"] = normalize(eligible["battery"]).fillna(0.5)
    eligible["score_ram"] = normalize(eligible["ram"])
    eligible["score_storage"] = normalize(eligible["storage_gb"])
    eligible["score_price"] = normalize(eligible["price"], False)

    weights = {
        "score_rating": 0.25,
        "score_camera": 0.20,
        "score_battery": 0.15,
        "score_ram": 0.15,
        "score_storage": 0.10,
        "score_price": 0.15,
    }

    eligible["match_score"] = sum(
        eligible[column] * weight
        for column, weight in weights.items()
    ) * 100

    # Include a short reason for the ranking.
    eligible["recommendation_reason"] = eligible.apply(
        lambda row: (
            f"Rating {row['rating']}/5; "
            f"₹{row['price']:,.0f}; "
            f"{row['ram']:g} GB RAM; "
            f"{row['storage_gb']:g} GB storage."
        ),
        axis=1,
    )

    return eligible.sort_values(
        ["match_score", "rating"],
        ascending=False,
    ).head(limit).reset_index(drop=True)
