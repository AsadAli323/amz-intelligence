"""Create the README figures from the cleaned sample or a full cleaned table."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shelfsense.cleaning import load_clean  # noqa: E402
from shelfsense.train import demand_tier  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else None
df = load_clean(src) if src else pd.read_csv(ROOT / "data/sample/products_sample.csv")
out = ROOT / "reports/figures"
out.mkdir(exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
C = "#1d3557"; A = "#FF9900"

fig, ax = plt.subplots(figsize=(8, 5))
vc = df["category"].value_counts().sort_values()
ax.barh(vc.index, vc.values, color=C); ax.set_title("Products per category"); ax.set_xlabel("Products")
fig.tight_layout(); fig.savefig(out / "categories.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(np.log10(df["price"].dropna()), bins=40, color=C)
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["$1", "$10", "$100", "$1,000"])
ax.set_title("Price distribution (log scale)"); ax.set_ylabel("Products")
fig.tight_layout(); fig.savefig(out / "price_distribution.png", dpi=150); plt.close(fig)

d = df.dropna(subset=["units_sold_month"]).copy()
d["tier"] = demand_tier(d["units_sold_month"])
ct = pd.crosstab(d["category"], d["tier"], normalize="index")[["Low", "Medium", "High"]].sort_values("High")
fig, ax = plt.subplots(figsize=(8, 5))
ct.plot(kind="barh", stacked=True, ax=ax, color=["#a8b5c7", C, A])
ax.set_title("Demand tier mix by category"); ax.set_xlabel("Share of products"); ax.set_ylabel("")
ax.legend(title="Tier", frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(out / "demand_by_category.png", dpi=150); plt.close(fig)

s = d.dropna(subset=["reviews"])
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(s["reviews"], s["units_sold_month"], s=8, alpha=.35, color=C)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Reviews"); ax.set_ylabel("Units sold / month"); ax.set_title("Reviews vs monthly sales")
fig.tight_layout(); fig.savefig(out / "reviews_vs_sales.png", dpi=150); plt.close(fig)
print("saved to", out)
