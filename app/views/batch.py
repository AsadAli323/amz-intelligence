import numpy as np
import pandas as pd
import streamlit as st

from common import add_features, assign_category, html, load_models, load_sample, to_tidy

demand_model, price_model = load_models()
html("""<div class="ss-hero"><h1>Batch Scoring</h1>
<p>Upload a CSV of products. Each row gets a demand forecast, a model price estimate and a price check.</p></div>""")

st.markdown("**Required columns:** `title`, `price`, `rating`, `reviews`. Optional: `category`, `discount_pct`, "
            "`is_sponsored`, `is_best_seller`, `has_coupon`, `is_renewed`.")
template = load_sample().head(100)[["title", "price", "rating", "reviews"]]
st.download_button("Download a template CSV", template.to_csv(index=False), "template.csv")

up = st.file_uploader("Upload a CSV file", type="csv")
if not up:
    st.markdown('<div class="ss-note">No file yet. Download the template, add your products, and upload it here.</div>',
                unsafe_allow_html=True)
    st.stop()

df = to_tidy(pd.read_csv(up))
missing = [c for c in ["title", "price", "rating", "reviews"] if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}. Add them and upload the file again.")
    st.stop()
for col in ["discount_pct", "is_sponsored", "is_best_seller", "is_amazons_choice", "is_limited_deal",
            "has_coupon", "has_sustainability_badge", "is_renewed"]:
    if col not in df:
        df[col] = 0
df["title"] = df["title"].astype(str)
if "category" not in df:
    df["category"] = df["title"].map(assign_category)
df["title_words"] = df["title"].str.split().str.len()
df["title_len"] = df["title"].str.len()
df["price"] = pd.to_numeric(df["price"], errors="coerce")
X = add_features(df.assign(price=df["price"].fillna(df["price"].median() if df["price"].notna().any() else 50)))
out = df.copy()
out["predicted_demand_tier"] = demand_model.predict(X)
out["model_price_estimate"] = np.expm1(price_model.predict(X)).round(2)
out["price_vs_estimate_pct"] = ((out["price"] / out["model_price_estimate"] - 1) * 100).round(1)

m1, m2, m3 = st.columns(3)
m1.metric("Rows scored", f"{len(out):,}")
m2.metric("Predicted High demand", f"{(out['predicted_demand_tier'] == 'High').mean():.0%}")
m3.metric("Priced above estimate by 40%+", f"{(out['price_vs_estimate_pct'] > 40).mean():.0%}")
st.dataframe(out[["title", "category", "price", "model_price_estimate", "price_vs_estimate_pct",
                  "predicted_demand_tier"]], width="stretch", hide_index=True)
st.download_button("Download scored CSV", out.to_csv(index=False), "scored_products.csv", type="primary")
