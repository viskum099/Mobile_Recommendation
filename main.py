
from pathlib import Path

from src.recommender.data_loader import load_products
from src.recommender.cleaning import clean_products
from src.recommender.engine import recommend_products


def my_app():
    data_path = Path("data/products.xlsx")

    raw_products = load_products(data_path)
    products = clean_products(raw_products)

    print(f"Loaded products: {len(products)}")
    print(f"Products needing price review: "
          f"{products['price_needs_review'].sum()}")
    print(f"Products with unknown processor: "
          f"{products['processor'].isna().sum()}")
    print(f"Products needing camera review: "
          f"{products['camera_needs_review'].sum()}")

    recommendations = recommend_products(
        products,
        budget=25000,
        min_rating=3.5,
        min_ram_gb=4,
        min_storage_gb=128,
        limit=5,
    )

    print("\nTop recommendations:")
    if recommendations.empty:
        print("No products match these preferences.")
    else:
        print(
            recommendations[
                ["name", "price", "rating", "ram",
                 "storage_gb", "match_score", "recommendation_reason"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    my_app()
