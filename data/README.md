# Data

This project uses a public dataset of ~42,000 scraped online electronics listings
(collected August 2025), published on Kaggle by **srisyra02**:
<https://www.kaggle.com/datasets/srisyra02/amazon-product-sales-data-42k-items-2025>

Please check the dataset page for its license and terms before reuse.

## What is in this repo
- `data/sample/products_sample.csv` is a 3,000-row random sample of the **cleaned** table
  (product URLs and image links removed). It powers the demo app.
- The full raw file is **not** committed. To reproduce the results, download it from the link
  above and place it at `data/raw/amazon_sales_data_uncleaned.csv`.

## Raw columns
`Title, Rating, Number_of_reviews, bought_in_last_month, Current/discounted_price, Price_on_variant,
Listed_price, is_best_seller, is_sponsored, is_couponed, buy_box_availability, delivery_details,
sustainability_badges, image_url, product_url, collected_at`

## Cleaning summary (`src/shelfsense/cleaning.py`)
| Problem in raw data | Fix |
|---|---|
| `"4.6 out of 5 stars"`, `"35,882"` stored as text | parsed to numbers |
| `"6K+ bought in past month"` | parsed to `units_sold_month` (6000); other text becomes missing |
| Prices with `$`, commas, `"No Discount"`, `"basic variant price: ..."` | parsed to `price`, `list_price`, `discount_pct` |
| Badge/coupon text in several columns | converted to 0/1 flags |
| Same product repeated across scrape runs (42k rows, ~8.8k products) | de-duplicated by product ID (ASIN) |
| No category column | rule-based category from title keywords |
