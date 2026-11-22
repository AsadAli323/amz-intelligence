import pandas as pd
import streamlit as st

from common import html, load_metrics

m = load_metrics()
d, p = m["demand"], m["price"]
html("""<div class="ss-hero"><h1>About the models</h1>
<p>What the app predicts, how well it works and where it falls short.</p></div>""")

a, b = st.columns(2, gap="large")
with a:
    st.subheader("Sales forecast")
    st.write(f"Sorts a product into Low, Medium or High monthly sales, using {d['rows']:,} products. "
             f"The best model ({d['best_model']}) is right about {d['results'][d['best_model']]['accuracy']:.0%} of the time. "
             f"Always guessing the most common tier is right {d['majority_baseline_accuracy']:.0%} of the time.")
    st.dataframe(pd.DataFrame(d["results"]).T.rename_axis("Model"), width="stretch")
with b:
    st.subheader("Price estimate")
    st.write(f"Estimates price from the product name, category, rating and reviews, using {p['rows']:,} products. "
             f"The best model ({p['best_model']}) is off by ${p['results'][p['best_model']]['mae_usd']:.0f} on average. "
             f"Guessing the median price for everything is off by ${p['median_baseline_mae_usd']:.0f}.")
    st.dataframe(pd.DataFrame(p["results"]).T.rename_axis("Model"), width="stretch")

st.subheader("Good to know")
st.markdown("""
- The models learned from electronics listings. Results for kitchen, toys, beauty and other categories are rough guides.
- Sales are reported in rounded groups (50+, 100+, 1K+), so the forecast is approximate.
- All data is from one snapshot in August 2025. Seasonality and price changes over time are not captured.
- Categories are assigned by keywords in the title, and some products land in the wrong one. You can change the category in the Product Analyzer.
- Review count reflects past sales, so it is a strong but partly circular signal.
- The app cannot read live Amazon pages. Links and ASINs are matched to the bundled dataset, and the product name is read from the link when possible.
""")
st.caption("Data: Amazon Product Sales Data, 42k items (2025), by srisyra02 on Kaggle. See the repository for the license note.")
