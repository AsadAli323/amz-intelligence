import numpy as np
import pandas as pd

from shelfsense.cleaning import assign_category, clean
from shelfsense.train import demand_tier


def _raw(**over):
    row = {
        "Title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "Rating": "4.6 out of 5 stars", "Number_of_reviews": "1,234",
        "bought_in_last_month": "6K+ bought in past month",
        "Current/discounted_price": "1,299.50", "Price_on_variant": "basic variant price: nan",
        "Listed_price": "$1,500.00", "is_best_seller": "Best Seller", "is_sponsored": "Sponsored",
        "is_couponed": "Save 10%  with coupon", "buy_box_availability": "Add to cart",
        "delivery_details": "", "sustainability_badges": "Small Business", "image_url": "x",
        "product_url": "/Sony/dp/B0ABCDEFGH/ref=sr_1_1", "collected_at": "21-08-2025 11:14",
    }
    row.update(over)
    return pd.DataFrame([row])


def test_parsing():
    d = clean(_raw()).iloc[0]
    assert d.rating == 4.6 and d.reviews == 1234
    assert d.units_sold_month == 6000
    assert d.price == 1299.5 and d.list_price == 1500.0
    assert 13 < d.discount_pct < 14
    assert d.is_best_seller == 1 and d.is_sponsored == 1 and d.has_coupon == 1
    assert d.asin == "B0ABCDEFGH"
    assert d.category == "Headphones & Audio"


def test_non_sales_text_becomes_missing():
    d = clean(_raw(bought_in_last_month="No featured offers available")).iloc[0]
    assert np.isnan(d.units_sold_month)


def test_dedup_by_asin():
    raw = pd.concat([_raw(), _raw(Rating=None)], ignore_index=True)
    assert len(clean(raw)) == 1


def test_categories():
    assert assign_category("Asurion 2 Year Protection Plan") == "Protection Plans & Services"
    assert assign_category("Dell Laptop 15 inch") == "Laptops & Computers"
    assert assign_category("zzz widget") == "Other Electronics"


def test_demand_tier():
    t = demand_tier(pd.Series([50, 100, 200, 999, 1000, 20000]))
    assert list(t) == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_new_categories():
    assert assign_category("Wilton Cake Decorating Kit with Piping Tips") == "Kitchen & Baking"
    assert assign_category("iPhone 15 Pro Max Case") == "Phones & Accessories"


def test_resolver():
    from shelfsense.resolve import parse_input
    r = parse_input("https://www.amazon.com/Apple-AirPods-Pro-2/dp/B0D1XD1ZV3/ref=sr_1_4?dib=x")
    assert r.kind == "link" and r.asin == "B0D1XD1ZV3" and r.title_hint == "Apple AirPods Pro 2"
    assert parse_input("b0d1xd1zv3").asin == "B0D1XD1ZV3"
    assert parse_input("Headphones").kind == "name"
    assert parse_input("cake decorating kit").title_hint == "cake decorating kit"
