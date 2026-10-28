
import pandas as pd
import pytest

from src.recommender.engine import recommend_products


@pytest.fixture
def products():
    return pd.DataFrame([
        {
            "name": "Phone A",
            "price": 18000,
            "price_needs_review": False,
            "rating": 4.5,
            "camera": 50,
            "battery": 5000,
            "ram": 8,
            "storage_gb": 128,
            "display_type": "AMOLED",
        },
        {
            "name": "Phone B",
            "price": 30000,
            "price_needs_review": False,
            "rating": 4.8,
            "camera": 100,
            "battery": 5000,
            "ram": 12,
            "storage_gb": 256,
            "display_type": "AMOLED",
        },
        {
            "name": "Phone C",
            "price": 20000,
            "price_needs_review": True,
            "rating": 4.7,
            "camera": 80,
            "battery": 5000,
            "ram": 8,
            "storage_gb": 128,
            "display_type": "AMOLED",
        },
    ])


def test_budget_and_price_review_filter(products):
    result = recommend_products(products, budget=25000)
    assert result["name"].tolist() == ["Phone A"]


def test_invalid_budget(products):
    with pytest.raises(ValueError):
        recommend_products(products, budget=0)


def test_display_filter(products):
    result = recommend_products(
        products,
        budget=25000,
        preferred_display="AMOLED",
    )
    assert (result["price"] <= 25000).all()
    assert not result["price_needs_review"].any()
