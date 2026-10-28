
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender.data_loader import load_products
from src.recommender.cleaning import clean_products
from src.recommender.engine import recommend_products


DATA_PATH = Path("data/products.xlsx")

st.set_page_config(
    page_title="PJ1 | Mobile Recommendation System",
    page_icon="📱",
    layout="wide",
)


@st.cache_data
def get_products(file_path: str, modified_time: float):
    """Load and clean the dataset, caching it until the file changes."""
    raw = load_products(file_path)
    return clean_products(raw)


def main():
    st.title("📱 Mobile Recommendation System")
    st.write(
        "Find phones that match your budget and preferences."
    )

    st.caption(
        "Dataset specifications and prices have not been "
        "independently verified. Recommendations are indicative."
    )

    if not DATA_PATH.is_file():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.info("Place your Excel file at data/products.xlsx.")
        st.stop()

    try:
        products = get_products(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime,
        )
    except Exception as exc:
        st.error(f"Could not load the product dataset: {exc}")
        st.stop()

    # Explain data quality before the user starts filtering.
    review_count = int(products["price_needs_review"].sum())

    with st.expander("About the dataset and data quality"):
        st.write(f"Product records loaded: {len(products)}")
        st.write(f"Prices requiring review: {review_count}")
        st.write(
            "Products with unverified currency are excluded "
            "from recommendations."
        )
        st.write(
            "Missing specifications are not proof of poor quality."
        )

    st.sidebar.header("Your preferences")

    budget = st.sidebar.number_input(
        "Maximum budget ($)",
        min_value=1,
        max_value=100000,
        value=1000,
        step=50,
    )

    min_rating = st.sidebar.slider(
        "Minimum customer rating",
        min_value=0.0,
        max_value=5.0,
        value=3.5,
        step=0.5,
    )

    min_ram = st.sidebar.selectbox(
        "Minimum RAM (GB)",
        options=[0, 4, 6, 8, 12, 16],
        index=1,
    )

    min_storage = st.sidebar.selectbox(
        "Minimum storage (GB)",
        options=[0, 64, 128, 256, 512, 1024],
        index=2,
    )

    display_types = sorted(
        products["display_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )
    processor_query = st.sidebar.text_input(
    "Processor keyword (optional)",
    placeholder="e.g. Snapdragon",
    )

    display_choice = st.sidebar.selectbox(
        "Preferred display type",
        options=["Any"] + display_types,
    )

    result_limit = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=5,
    )

    preferred_display = (
        None if display_choice == "Any" else display_choice
    )
    st.sidebar.subheader("Personalize your recommendations")
    st.sidebar.caption(
        "Give higher values to features that matter more to you."
    )

    rating_weight = st.sidebar.slider(
        "Customer rating importance",
        0, 100, 25,
    )

    camera_weight = st.sidebar.slider(
        "Camera importance",
        0, 100, 20,
    )

    battery_weight = st.sidebar.slider(
        "Battery importance",
        0, 100, 15,
    )

    ram_weight = st.sidebar.slider(
        "Performance (RAM) importance",
        0, 100, 15,
    )

    storage_weight = st.sidebar.slider(
        "Storage importance",
        0, 100, 10,
    )

    price_weight = st.sidebar.slider(
        "Affordability importance",
        0, 100, 15,
    )

    custom_weights = {
        "rating": rating_weight,
        "camera": camera_weight,
        "battery": battery_weight,
        "ram": ram_weight,
        "storage": storage_weight,
        "price": price_weight,
    }

    if sum(custom_weights.values()) == 0:
        st.sidebar.warning(
            "Choose at least one feature to prioritize."
        )
    

    try:
        recommendations = recommend_products(
            products,
            budget=float(budget),
            min_rating=float(min_rating),
            min_ram_gb=float(min_ram),
            min_storage_gb=float(min_storage),
            preferred_display=preferred_display,
            processor_query=processor_query or None,
            limit=result_limit,
            weights=custom_weights,
        )
    except Exception as exc:
        st.error(f"Recommendation failed: {exc}")
        st.stop()

    st.subheader("Your recommendations")

    if recommendations.empty:
        st.warning(
            "No products match all your selected preferences. "
            "Try increasing your budget or relaxing a filter."
        )
        st.stop()
    
    if sum(custom_weights.values()) == 0:
        st.warning("Select at least one feature to rank products.")
        st.stop()
    
    st.success(
        f"Found {len(recommendations)} matching product(s)."
    )

    # Show each recommendation as a separate card.
    for rank, (_, phone) in enumerate(
        recommendations.iterrows(), start=1
    ):
        with st.container(border=True):
            st.markdown(f"### {rank}. {phone['name']}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Price", f"${phone['price']:,.0f}")
            col2.metric(
                "Customer rating",
                (
                    f"{phone['rating']:.1f}/5"
                    if pd.notna(phone["rating"])
                    else "Not available"
                ),
            )
            col3.metric(
                "Match score",
                f"{phone['match_score']:.1f}/100",
            )

            st.write(
                f"**Display:** {phone['display_type']}"
            )
            
            st.write(
                f"**RAM:** {phone['ram']} GB"
                if pd.notna(phone["ram"])
                else "**RAM:** Not available"
            )
            st.write(
                f"**Storage:** {phone['storage_gb']:g} GB"
                if pd.notna(phone["storage_gb"])
                else "**Storage:** Not available"
            )
            st.write(
                f"**Battery:** {phone['battery']} mAh"
                if pd.notna(phone["battery"])
                else "**Battery:** Not available"
            )
            st.write(
                f"**Processor:** {phone['processor']}"
                if pd.notna(phone["processor"])
                else "**Processor:** Not mentioned in dataset"
            )

            st.caption(phone["recommendation_reason"])

    st.subheader("Compare results")

    comparison_columns = [
        "name",
        "price",
        "rating",
        "camera",
        "display_type",
        "battery",
        "ram",
        "storage_gb",
        "weight",
        "processor",
        "match_score",
    ]

    available_columns = [
        column for column in comparison_columns
        if column in recommendations.columns
    ]

    st.dataframe(
        recommendations[available_columns],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
