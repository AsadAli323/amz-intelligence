"""Turn the raw scraped listing export into a tidy, analysis-ready table."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Ordered rules: the first keyword group that matches the title wins.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Protection Plans & Services", ["protection plan", "asurion", "warranty", "squaretrade",
                                     "accidental damage", "service plan", "subscription"]),
    ("Kitchen & Baking", ["cake", "baking", "bakeware", "cupcake", "fondant", "frosting", "icing",
                          "piping", "pastry", "cookie cutter", "kitchen", "cookware", "utensil",
                          "spatula", "cutting board", "mixing bowl", "air fryer", "blender",
                          "coffee maker", "dinnerware"]),
    ("Toys, Games & Hobbies", [" toy", "lego", "puzzle", "board game", "doll ", "action figure",
                               "diy kit", "model kit", "plush"]),
    ("Home & Garden", ["garden", "lawn", "mattress", "pillow", "bedding", "curtain",
                       "candle", "vacuum cleaner"]),
    ("Health & Beauty", ["shampoo", "skincare", "moisturizer", "lipstick", "vitamin",
                         "supplement", "toothbrush", "razor", "hair dryer", "massager", "beard"]),
    ("Tools, Auto & Outdoor", ["tool kit", "wrench", "screwdriver", "car wash", "tire",
                               "motor oil", "tent ", "fishing", "car seat"]),
    ("Fashion & Accessories", ["shirt", "jeans", "shoes", "sneaker", "jacket", "dress ", "handbag",
                               "sunglasses"]),
    ("Renewed / Refurbished", ["renewed", "refurbished", "pre-owned", "open box"]),
    ("Laptops & Computers", ["laptop", "notebook pc", "chromebook", "macbook", "desktop", "mini pc",
                             "all-in-one pc", "gaming pc", "workstation", "thinkpad"]),
    ("Tablets & E-readers", ["tablet", "ipad", "kindle", "e-reader", "fire hd"]),
    ("Phones & Accessories", ["smartphone", "iphone", "galaxy", "unlocked phone", "cell phone",
                              "phone case", "screen protector", "magsafe", "phone holder",
                              "phone mount", "popsocket"]),
    ("Headphones & Audio", ["headphone", "earbud", "earphone", "airpods", "speaker", "soundbar",
                            "microphone", "headset", "bluetooth audio", "subwoofer", "amplifier",
                            "turntable", "receiver"]),
    ("Cameras & Drones", ["camera", "drone", "gopro", "dashcam", "dash cam", "lens", "tripod",
                          "camcorder", "webcam", "security cam", "doorbell"]),
    ("TV & Home Theater", ["tv ", " tv", "television", "projector", "streaming", "roku",
                           "fire stick", "chromecast", "hdmi"]),
    ("Gaming", ["gaming", "playstation", "xbox", "nintendo", "switch", "controller", "ps5",
                "ps4", "joystick", "gamepad", "steam deck"]),
    ("Wearables & Smart Home", ["smartwatch", "smart watch", "fitness tracker", "watch band",
                                "apple watch", "smart plug", "smart light", "smart home",
                                "alexa", "echo dot", "thermostat", "smart bulb", "led strip"]),
    ("Chargers, Cables & Power", ["charger", "cable", "power bank", "adapter", "charging",
                                  "battery", "usb-c", "usb c", "lightning", "power strip",
                                  "surge protector", "extension cord", "wireless charger"]),
    ("Computer Peripherals", ["keyboard", "mouse", "monitor", "printer", "scanner", "router",
                              "ssd", "hard drive", "flash drive", "memory card", "sd card",
                              "usb hub", "docking", "graphics card", "gpu", "ram", "motherboard",
                              "wifi", "wi-fi", "ethernet", "toner", "ink"]),
    ("Cases, Bags & Cleaning", ["case", "cover", "sleeve", "bag", "backpack", "stand",
                                "cleaning", "wipes", "cleaner", "skin"]),
]
DEFAULT_CATEGORY = "Other Electronics"
# Categories outside what the models were trained on (they were learned from electronics listings).
NON_ELECTRONIC = {"Kitchen & Baking", "Toys, Games & Hobbies", "Home & Garden", "Health & Beauty",
                  "Tools, Auto & Outdoor", "Fashion & Accessories", "Protection Plans & Services"}

_ASIN_RE = re.compile(r"(?:/dp/|%2Fdp%2F)([A-Z0-9]{10})")
_UNITS_RE = re.compile(r"^\s*([\d.]+)\s*([KkMm]?)\+?\s*bought", re.I)
_MONEY_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


_COMPILED = [
    (name, re.compile("|".join(r"(?<![a-z0-9])" + re.escape(k.strip()) for k in kws)))
    for name, kws in CATEGORY_RULES
]


def assign_category(title: str) -> str:
    """First rule whose keyword starts a word in the title wins (so 'mop' does not match 'Mophie')."""
    t = str(title).lower()
    for name, rx in _COMPILED:
        if rx.search(t):
            return name
    return DEFAULT_CATEGORY


def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False), errors="coerce")


def _parse_units(s: pd.Series) -> pd.Series:
    mult = {"": 1, "k": 1_000, "m": 1_000_000}

    def one(v):
        if not isinstance(v, str):
            return np.nan
        m = _UNITS_RE.match(v)
        if not m:
            return np.nan
        return float(m.group(1)) * mult[m.group(2).lower()]

    return s.map(one)


def _first_money(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype("string").str.extract(_MONEY_RE, expand=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw listing table (one row per product)."""
    df = pd.DataFrame(index=raw.index)
    df["title"] = raw["Title"].astype("string").str.strip()
    url = raw["product_url"].astype("string")
    df["asin"] = url.str.extract(_ASIN_RE, expand=False)

    df["rating"] = pd.to_numeric(
        raw["Rating"].astype("string").str.extract(r"([\d.]+)", expand=False), errors="coerce"
    ).where(lambda s: s.between(1, 5))
    df["reviews"] = _to_number(raw["Number_of_reviews"])
    df["units_sold_month"] = _parse_units(raw["bought_in_last_month"])

    current = _to_number(raw["Current/discounted_price"])
    variant = _first_money(raw["Price_on_variant"])
    list_price = _first_money(raw["Listed_price"])
    df["price"] = current.fillna(variant)
    df["list_price"] = list_price
    valid = df["list_price"].notna() & df["price"].notna() & (df["list_price"] > df["price"])
    df["discount_pct"] = np.where(valid, (df["list_price"] - df["price"]) / df["list_price"] * 100, 0.0)

    badge = raw["is_best_seller"].astype("string")
    df["is_best_seller"] = (badge == "Best Seller").astype(int)
    df["is_amazons_choice"] = badge.str.contains("Amazon's", na=False).astype(int)
    df["is_limited_deal"] = badge.str.contains("Limited time|Ends in", na=False).astype(int)
    df["is_sponsored"] = (raw["is_sponsored"].astype("string") == "Sponsored").astype(int)
    coupon = raw["is_couponed"].astype("string")
    df["has_coupon"] = (coupon.str.contains("coupon", case=False, na=False)
                        & ~coupon.str.contains("No Coupon", na=False)).astype(int)
    sus = raw["sustainability_badges"].astype("string")
    df["has_sustainability_badge"] = sus.notna().astype(int)
    df["is_small_business"] = sus.str.contains("Small Business", na=False).astype(int)

    df["title_len"] = df["title"].str.len()
    df["title_words"] = df["title"].str.split().str.len()
    df["is_renewed"] = df["title"].str.contains("renewed|refurbished", case=False, na=False).astype(int)
    brand = df["title"].str.extract(r"^([A-Za-z0-9][\w&'.\-]*)", expand=False)
    canon = brand.dropna().groupby(brand.dropna().str.lower()).agg(lambda x: x.mode().iat[0])
    df["brand"] = brand.str.lower().map(canon)
    df["category"] = df["title"].map(assign_category)
    df["collected_at"] = pd.to_datetime(raw["collected_at"], format="%d-%m-%Y %H:%M", errors="coerce")

    # --- row hygiene -------------------------------------------------------
    df = df[df["title"].notna() & (df["title"].str.len() > 5)]
    df = df[df["price"].isna() | ((df["price"] > 0.5) & (df["price"] < 20_000))]
    key = df["asin"].fillna(df["title"])
    df = (df.assign(_k=key, _n=df.notna().sum(axis=1))
            .sort_values("_n", ascending=False)
            .drop_duplicates("_k")
            .drop(columns=["_k", "_n"])
            .sort_index())
    return df.reset_index(drop=True)


def load_clean(path: str) -> pd.DataFrame:
    return clean(pd.read_csv(path))
