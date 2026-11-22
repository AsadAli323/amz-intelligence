import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import (BAD, CATEGORIES, GOOD, INK, LABEL, MUTED, NON_ELECTRONIC, TIERS, add_features,
                    assign_category, esc, html, load_models, load_sample, money)
from shelfsense.resolve import parse_input

df = load_sample()
demand_model, price_model = load_models()

# --------------------------------------------------------------------------- hero
html("""
<div class="ss-hero">
<h1>Check any product before you list it.</h1>
<p>Paste an Amazon link, an ASIN or just a product name. You get a fair-price estimate, a sales forecast
and the closest competing products.</p>
</div>
""")


def _set(q):
    st.session_state["q"] = q


top = df.dropna(subset=["asin", "reviews"]).sort_values("reviews").iloc[-1]
examples = {
    "Earbuds": "wireless earbuds with charging case",
    "Cake decorating kit": "cake decorating kit with piping tips and turntable",
    "Try a product link": f"https://www.amazon.com/dp/{top.asin}",
    "Try an ASIN": str(df.dropna(subset=['asin']).sort_values('reviews').iloc[-30].asin),
}
q = st.text_input("Product link, ASIN or name", key="q", label_visibility="collapsed",
                  placeholder="Paste a link, an ASIN such as B0D1XD1ZV3, or type a product name")
cols = st.columns([1, 1.5, 1.5, 1.2, 2])
for c, (label, val) in zip(cols, examples.items()):
    c.button(label, on_click=_set, args=(val,), key=f"ex_{label}", use_container_width=True)

# --------------------------------------------------------------------------- empty state
if not q.strip():
    st.write("")
    a, b, c = st.columns(3)
    for col, h, t in [
        (a, "Is the price fair?", "A model trained on thousands of listings estimates the market price and "
                                   "shows where your price sits among competitors."),
        (b, "Will it sell?", "Get a Low, Medium or High monthly sales forecast with the probability of each."),
        (c, "Who are the rivals?", "See the most similar products, with their prices, ratings and sales."),
    ]:
        col.markdown(f'<div class="ss-step"><h3>{h}</h3><p>{t}</p></div>', unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------------------- resolve the input
info = parse_input(q)
row = None
if info.asin:
    hit = df[df["asin"] == info.asin]
    row = hit.iloc[0] if len(hit) else None

title = row["title"] if row is not None else info.title_hint
if not title:
    st.markdown('<div class="ss-note">That ASIN is not in the demo data, so I cannot read its title. '
                'Paste the product name below and the analysis will run on it.</div>', unsafe_allow_html=True)
    title = st.text_input("Product name", key="manual_title", placeholder="e.g. Anker 737 Power Bank 24000mAh")
    if not title:
        st.stop()

ctx = info.asin or title
guess_cat = row["category"] if row is not None else assign_category(title)

# --------------------------------------------------------------------------- listing details
with st.expander("Listing details (edit to see how the result changes)", expanded=row is None and info.kind != "name"):
    d1, d2, d3, d4 = st.columns(4)
    price_in = d1.number_input("Listed price (USD)", min_value=0.0, value=float(row["price"]) if row is not None and pd.notna(row["price"]) else None,
                               placeholder="optional", key=f"price_{ctx}")
    st.session_state.setdefault(f"rating_{ctx}", float(row["rating"]) if row is not None and pd.notna(row["rating"]) else 4.5)
    st.session_state.setdefault(f"reviews_{ctx}", int(row["reviews"]) if row is not None and pd.notna(row["reviews"]) else 0)
    rating = d2.number_input("Rating", min_value=1.0, max_value=5.0, step=0.1, key=f"rating_{ctx}")
    reviews = d3.number_input("Number of reviews", min_value=0, max_value=5_000_000, step=1, key=f"reviews_{ctx}",
                              help="Use 0 for a brand-new product.")
    category = d4.selectbox("Category", CATEGORIES, index=CATEGORIES.index(guess_cat), key=f"cat_{ctx}")
    e1, e2, e3, e4 = st.columns(4)
    discount = e1.slider("Discount (%)", 0, 80, int(row["discount_pct"]) if row is not None else 0, key=f"disc_{ctx}")
    best = e2.checkbox("Best Seller badge", value=bool(row["is_best_seller"]) if row is not None else False, key=f"bs_{ctx}")
    coupon = e3.checkbox("Has coupon", value=bool(row["has_coupon"]) if row is not None else False, key=f"cp_{ctx}")
    sponsored = e4.checkbox("Sponsored ad", value=bool(row["is_sponsored"]) if row is not None else False, key=f"sp_{ctx}")

# --------------------------------------------------------------------------- similarity index
uniq = df.drop_duplicates("title").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_index(titles: tuple):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
    return vec, vec.fit_transform(titles)


vec, mat = build_index(tuple(uniq["title"]))
sims = cosine_similarity(vec.transform([title]), mat).ravel()

# --------------------------------------------------------------------------- model predictions
base = {"title": title, "rating": rating, "reviews": reviews, "category": category,
        "is_renewed": int("renewed" in title.lower() or "refurbished" in title.lower()),
        "title_words": len(title.split()), "title_len": len(title), "is_sponsored": int(sponsored)}
est_price = float(np.expm1(price_model.predict(add_features(pd.DataFrame([base])))[0]))
used_price = price_in if price_in else est_price
dem_row = {**base, "price": used_price, "discount_pct": discount, "is_best_seller": int(best),
           "is_amazons_choice": 0, "is_limited_deal": 0, "has_coupon": int(coupon), "has_sustainability_badge": 0}
probs = demand_model.predict_proba(add_features(pd.DataFrame([dem_row])))[0]
tier_probs = dict(zip(demand_model.classes_, probs))
top_tier = max(tier_probs, key=tier_probs.get)

cat_peers = df[df["category"] == category]
enough_peers = len(cat_peers) >= 15
peers = cat_peers if enough_peers else df
n_peers = len(peers)
out_of_domain = category in NON_ELECTRONIC


def pct_below(series, value):
    s = series.dropna()
    return float((s < value).mean() * 100) if len(s) else 50.0


# --------------------------------------------------------------------------- product header
asin_chip = f'<span class="ss-chip">ASIN {esc(info.asin)}</span>' if info.asin else ""
found_chip = '<span class="ss-chip hl">Found in dataset</span>' if row is not None else ""
amazon = (f'<a href="https://www.amazon.com/dp/{info.asin}" target="_blank" rel="noopener">Open on Amazon</a>'
          if info.asin else "")
html(f"""
<div class="ss-product">
<span class="ss-chip hl">{esc(category)}</span>{asin_chip}{found_chip}
<h2>{esc(title)}</h2>
<div class="ss-muted">{amazon}</div>
</div>
""")

# --------------------------------------------------------------------------- borrow details from the closest match
best_i = int(sims.argmax())
near = np.where(sims >= sims[best_i] * 0.9)[0]          # near-ties: prefer the best-known listing
best_i = int(near[np.argmax(uniq["reviews"].fillna(0).to_numpy()[near])])
bm = uniq.iloc[best_i]
can_borrow = (row is None and sims[best_i] >= 0.35 and not out_of_domain and reviews == 0
              and pd.notna(bm["reviews"]) and pd.notna(bm["rating"]))


def _borrow(key_ctx, rt, rv):
    st.session_state[f"rating_{key_ctx}"] = float(rt)
    st.session_state[f"reviews_{key_ctx}"] = int(rv)


# --------------------------------------------------------------------------- out-of-domain notes
notes = []
if category in NON_ELECTRONIC:
    notes.append(f"<b>{esc(category)}</b> is outside what these models learned. They were trained on "
                 f"electronics listings, so the price estimate is switched off and the sales forecast is only a rough "
                 f"guide. The demo data holds just {len(cat_peers)} products in this category.")
if sims.max() < 0.25:
    notes.append("No close matches exist in the demo data for this product, so the estimate leans on general "
                 "patterns. Similar-product results below may be loosely related.")
if reviews == 0:
    notes.append("With no reviews, demand forecasts lean low. New listings usually start slowly.")
for n in notes:
    st.markdown(f'<div class="ss-note">{n}</div>', unsafe_allow_html=True)
if can_borrow:
    st.button(f"Use the rating and reviews of the closest match ({int(bm['reviews']):,} reviews, {bm['rating']:.1f} stars)",
              on_click=_borrow, args=(ctx, bm["rating"], bm["reviews"]), key=f"borrow_{ctx}")

# --------------------------------------------------------------------------- three panels
c1, c2, c3 = st.columns([1.15, 1.05, 1.2], gap="large")

with c1:
    if out_of_domain:
        med = cat_peers["price"].median() if len(cat_peers) >= 5 else None
        sub = (f"Typical price for {esc(category)} in the data: {money(med)}." if med is not None
               else "There are too few comparable products in the data to suggest a price.")
        html(f"""
<div class="ss-tag">
<div class="k">Estimated market price</div>
<div class="ss-big" style="font-size:2.1rem">Not available</div>
<div class="ss-muted" style="margin-top:.5rem">The price model only covers electronics. {sub}</div>
<div class="ss-muted" style="margin-top:.8rem">Your listed price: {money(price_in) if price_in else 'not entered'}.</div>
</div>
""")
    elif price_in:
        ratio = price_in / est_price
        if ratio < 0.7:
            verdict, color = "Priced below the market estimate", GOOD
        elif ratio > 1.4:
            verdict, color = "Priced above the market estimate", BAD
        else:
            verdict, color = "In line with the market estimate", INK
        delta = (ratio - 1) * 100
        sub = f"Listed at {money(price_in)}, {abs(delta):.0f}% {'above' if delta > 0 else 'below'} the estimate."
        badge = f'<div class="ss-verdict" style="background:{color}">{verdict}</div>'
    else:
        sub, badge = "Add a listed price in the details above to compare.", ""
    if not out_of_domain:
        html(f"""
<div class="ss-tag">
<div class="k">Estimated market price</div>
<div class="ss-big">{money(est_price)}</div>
<div class="ss-muted" style="margin-top:.5rem">{sub}</div>
{badge}
<div class="ss-muted" style="margin-top:.8rem">Typical error is about $80, so read this as a range, not an exact figure.</div>
</div>
""")

with c2:
    order = ["Low", "Medium", "High"]
    shades = {"Low": "#D5DEE2", "Medium": "#9FB5BF", "High": LABEL}
    segs = "".join(f'<div style="width:{tier_probs[t]*100:.1f}%;background:{shades[t]}">'
                   f'{tier_probs[t]*100:.0f}%</div>' if tier_probs[t] > 0.07 else
                   f'<div style="width:{tier_probs[t]*100:.1f}%;background:{shades[t]}"></div>' for t in order)
    html(f"""
<div class="ss-panel">
<h3>Sales forecast</h3>
<div class="ss-big" style="font-size:2.4rem">{top_tier} demand</div>
<div class="ss-muted">{TIERS[top_tier]}. The model is {tier_probs[top_tier]*100:.0f}% confident.</div>
<div class="ss-stack">{segs}</div>
<div class="ss-legend"><span>Low</span><span>Medium</span><span>High</span></div>
</div>
""")

with c3:
    rows_html = ""
    metrics = [] if not enough_peers else [("Price", used_price, peers["price"], money(used_price)),
               ("Rating", rating, peers["rating"], f"{rating:.1f} stars"),
               ("Reviews", reviews, peers["reviews"], f"{reviews:,}")]
    for name, val, series, shown in metrics:
        p = pct_below(series, val)
        rows_html += (f'<div class="ss-rule"><div class="row"><span>{name}: <b>{shown}</b></span>'
                      f'<span class="ss-muted">higher than {p:.0f}%</span></div>'
                      f'<div class="ss-track"><div class="ss-pin" style="left:{min(max(p,2),98):.0f}%"></div></div></div>')
    if not enough_peers:
        rows_html = (f'<div class="ss-muted">Only {len(cat_peers)} {esc(category)} products are in the data, '
                     'which is too few for a fair comparison.</div>')
    html(f"""
<div class="ss-panel">
<h3>{'Where it sits among ' + format(n_peers, ',') + ' ' + esc(category) + ' products' if enough_peers else 'Where it sits'}</h3>
{rows_html}
</div>
""")

# --------------------------------------------------------------------------- similar products
st.write("")
st.subheader("Closest competing products")
same_cat = st.toggle("Only show the same category", value=enough_peers, key=f"same_{ctx}")
cand = uniq.assign(sim=sims)
if info.asin:
    cand = cand[cand["asin"] != info.asin]
if same_cat:
    cand = cand[cand["category"] == category]
cand = cand[cand["sim"] >= 0.12].sort_values("sim", ascending=False).head(6)

if cand.empty or (out_of_domain and len(cat_peers) < 5):
    st.markdown('<div class="ss-note">The demo data has no close matches for this product. Try a more descriptive '
                'name, or turn off the category filter.</div>', unsafe_allow_html=True)
else:
    items = ""
    for _, r in cand.iterrows():
        link = f'<a href="https://www.amazon.com/dp/{r.asin}" target="_blank" rel="noopener">{esc(r.title[:110])}</a>' \
            if pd.notna(r.asin) else esc(r.title[:110])
        sold = f"{int(r.units_sold_month):,} sold per month" if pd.notna(r.units_sold_month) else "sales not reported"
        rt = f"{r.rating:.1f} stars" if pd.notna(r.rating) else "no rating"
        rv = f"{int(r.reviews):,} reviews" if pd.notna(r.reviews) else "no reviews"
        items += (f'<div class="ss-peer"><div><div class="t">{link}</div>'
                  f'<div class="m">{esc(str(r.brand))}, {rt}, {rv}, {sold}</div></div>'
                  f'<div style="text-align:right"><div class="ss-big" style="font-size:1.5rem">{money(r.price)}</div>'
                  f'<div class="ss-match" title="Similarity"><div style="width:{r.sim*100:.0f}%"></div></div>'
                  f'<div class="ss-muted" style="font-size:.8rem">{r.sim*100:.0f}% match</div></div></div>')
    html(f'<div class="ss-panel">{items}</div>')

st.caption("Estimates come from statistical models trained on listing data collected in August 2025. "
           "They show likely patterns, not guarantees.")
