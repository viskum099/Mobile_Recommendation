
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
            "processor": "Snapdragon 7 Gen 3",
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
            "processor": "Snapdragon 8 Gen 2",
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


def test_processor_search(products):
    products["processor"] = [
        "Snapdragon 7 Gen 3",
        "Exynos 2400",
        "Snapdragon 8 Gen 3",
    ]

    result = recommend_products(
        products,
        budget=25000,
        processor_query="snapdragon",
    )

    assert not result.empty
    assert result["processor"].str.contains(
        "snapdragon",
        case=False,
        na=False,
    ).all()


def test_missing_processor_does_not_match_search(products):
    products["processor"] = [
        None,
        "Exynos 2400",
        "Snapdragon 8 Gen 3",
    ]

    result = recommend_products(
        products,
        budget=25000,
        processor_query="snapdragon",
    )

    assert result["processor"].str.contains(
        "snapdragon",
        case=False,
        na=False,
    ).all()


def test_custom_weights_are_supported(products):
    result = recommend_products(
        products,
        budget=25000,
        weights={
            "rating": 0,
            "camera": 1,
            "battery": 0,
            "ram": 0,
            "storage": 0,
            "price": 0,
        },
    )

    assert not result.empty
    assert result["price"].le(25000).all()
    assert result["match_score"].between(0, 100).all()


def test_all_zero_weights_are_rejected(products):
    with pytest.raises(ValueError):
        recommend_products(
            products,
            budget=25000,
            weights={
                "rating": 0,
                "camera": 0,
                "battery": 0,
                "ram": 0,
                "storage": 0,
                "price": 0,
            },
        )

def make_test_products():
    """Create a small, predictable dataset for scoring tests."""
    return pd.DataFrame(
        [
            {
                "name": "Test Phone A",
                "price": 500.0,
                "price_needs_review": False,
                "rating": 4.0,
                "camera": 100.0,
                "battery": 5000.0,
                "ram": 8.0,
                "storage_gb": 256.0,
                "display_type": "AMOLED",
                "processor": "Test Processor",
            }
        ]
    )


def test_feature_contributions_add_up_to_match_score():
    products = make_test_products()

    result = recommend_products(products, budget=1000)

    contribution_columns = [
        "rating_contribution",
        "camera_contribution",
        "battery_contribution",
        "ram_contribution",
        "storage_contribution",
        "price_contribution",
    ]

    contribution_total = result.loc[0, contribution_columns].sum()

    assert result.loc[0, "match_score"] == pytest.approx(
        contribution_total
    )


def test_custom_weights_change_contributions():
    products = make_test_products()

    rating_focused = recommend_products(
        products,
        budget=1000,
        weights={
            "rating": 100,
            "camera": 0,
            "battery": 0,
            "ram": 0,
            "storage": 0,
            "price": 0,
        },
    )

    camera_focused = recommend_products(
        products,
        budget=1000,
        weights={
            "rating": 0,
            "camera": 100,
            "battery": 0,
            "ram": 0,
            "storage": 0,
            "price": 0,
        },
    )

    assert rating_focused.loc[0, "rating_contribution"] == pytest.approx(80)
    assert camera_focused.loc[0, "camera_contribution"] == pytest.approx(50)


def test_no_matching_products_returns_empty_result():
    products = make_test_products()

    result = recommend_products(products, budget=100)

    assert result.empty
    assert "match_score" in result.columns
    assert "rating_contribution" in result.columns

