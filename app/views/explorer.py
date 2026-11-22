import plotly.express as px
import streamlit as st

from common import INK, LABEL, html, load_sample, money, plot_layout

df = load_sample()
html("""<div class="ss-hero"><h1>Market Explorer</h1>
<p>See how categories, prices and ratings relate to sales across the listing data.</p></div>""")

f1, f2, f3 = st.columns([1.4, 1.4, 1.2])
cats = f1.multiselect("Category", sorted(df["category"].unique()), placeholder="All categories")
brands = f2.multiselect("Brand", df["brand"].value_counts().head(40).index.tolist(), placeholder="All brands")
lo, hi = float(df["price"].min()), float(df["price"].quantile(0.99))
price = f3.slider("Price range (USD)", lo, hi, (lo, hi))

f = df[df["price"].between(*price) | df["price"].isna()]
if cats:
    f = f[f["category"].isin(cats)]
if brands:
    f = f[f["brand"].isin(brands)]
if f.empty:
    st.markdown('<div class="ss-note">No products match these filters. Widen the price range or clear a filter.</div>',
                unsafe_allow_html=True)
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Products", f"{len(f):,}")
k2.metric("Average rating", f"{f['rating'].mean():.2f}")
k3.metric("Median price", money(f["price"].median()))
k4.metric("Share sponsored", f"{f['is_sponsored'].mean():.0%}")

t1, t2, t3 = st.tabs(["Overview", "What drives sales", "Data"])
with t1:
    l, r = st.columns(2)
    cc = f["category"].value_counts().reset_index()
    l.plotly_chart(plot_layout(px.bar(cc, x="count", y="category", orientation="h", title="Products per category")
                               .update_traces(marker_color=INK)
                               .update_layout(yaxis={"categoryorder": "total ascending", "title": ""},
                                              xaxis_title="Products"), 460), width="stretch")
    r.plotly_chart(plot_layout(px.box(f.dropna(subset=["price"]), x="category", y="price", log_y=True,
                                      title="Price by category (log scale)", color_discrete_sequence=[INK])
                               .update_layout(xaxis_tickangle=-40, xaxis_title="", yaxis_title="Price (USD)"), 460),
                   width="stretch")
with t2:
    s = f.dropna(subset=["units_sold_month", "reviews"])
    l, r = st.columns(2)
    l.plotly_chart(plot_layout(px.scatter(s, x="reviews", y="units_sold_month", color="category", log_x=True, log_y=True,
                                          hover_name="title", opacity=.65, title="Reviews vs monthly units sold")
                               .update_layout(xaxis_title="Reviews", yaxis_title="Units sold per month",
                                              legend_title="")), width="stretch")
    s2 = s.assign(group=s["is_best_seller"].map({0: "No badge", 1: "Best Seller"}))
    g = s2.groupby("group")["units_sold_month"].median().reset_index()
    r.plotly_chart(plot_layout(px.bar(g, x="group", y="units_sold_month", title="Median units sold per month by badge")
                               .update_traces(marker_color=[INK, LABEL][: len(g)])
                               .update_layout(xaxis_title="", yaxis_title="Units per month")), width="stretch")
    st.plotly_chart(plot_layout(px.box(s.assign(rating_band=s["rating"].round(1)), x="rating_band", y="units_sold_month",
                                       log_y=True, title="Units sold by rating", color_discrete_sequence=[INK])
                                .update_layout(xaxis_title="Rating", yaxis_title="Units per month")), width="stretch")
with t3:
    cols = ["title", "brand", "category", "price", "rating", "reviews", "units_sold_month", "discount_pct", "asin"]
    st.dataframe(f[cols], width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", f[cols].to_csv(index=False), "filtered_products.csv")
