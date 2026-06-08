<div align="center">

# 🏙️ NYC Airbnb Price Prediction Pipeline

**End-to-end data engineering pipeline on 12M+ rows of NYC Airbnb data**  
*From raw messy CSVs → production-ready ML model in 7 notebooks*

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-red?style=flat-square)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🎯 What This Project Does

Takes three raw, messy Airbnb CSV files and builds a complete ML pipeline to predict nightly listing prices across New York City — cleaning 90 raw columns down to 45 engineered features, training three models, tuning hyperparameters, and explaining every prediction with SHAP — achieving **76.8% R² with $188 average prediction error** on unseen data.

---

## 📊 Results — Current Best

| Model | Val R² | Test R² | Avg Dollar Error |
|-------|--------|---------|-----------------|
| Ridge Baseline | 0.629 | — | $1,170 |
| Random Forest | 0.720 | — | $247 |
| XGBoost (default) | 0.764 | 0.763 | $192 |
| **XGBoost (tuned) ✓** | **0.771** | **0.768** | **$188** |

> Trained on 16,416 listings · Validated on 2,052 · Tested on 2,053  
> Stratified 80/10/10 split by NYC borough · 45 training features

---

## 📈 Model Iteration History

This project went through four distinct versions. Each one is documented here honestly — including a leakage mistake that was caught and fixed, and a full hyperparameter tuning journey.

---

### Version 1 — Leaked (inflated, incorrect)

| Model | Val R² | Test R² | Avg Dollar Error |
|-------|--------|---------|-----------------|
| Ridge Baseline | 0.913 | — | $1,014 |
| Random Forest | 0.988 | — | $148 |
| XGBoost | 0.988 | 0.991 | $114 |

**Why these scores were wrong:** `price_per_person` and `price_per_bedroom` were engineered from the target variable `price` and incorrectly kept in X. The model was mathematically reconstructing the answer — `price = price_per_bedroom × bedrooms` — rather than learning genuine price patterns from listing characteristics. R² of 0.991 is a red flag for any real-world regression problem.

---

### Version 2 — Leakage Fixed (honest baseline)

| Model | Val R² | Test R² | Avg Dollar Error |
|-------|--------|---------|-----------------|
| Ridge Baseline | 0.613 | — | $246 |
| Random Forest | 0.718 | — | $246 |
| **XGBoost** | **0.769** | **0.761** | **$189** |

**What changed:** Removed `price_per_person` and `price_per_bedroom` from X before splitting. The model now learns from genuine listing characteristics — size, location, room type, amenities, host experience — not from derived price ratios.

Val/test gap is only 0.008 — model generalizes cleanly with no overfitting.

---

### Version 3 — 6 New Engineered Features

| Model | Val R² | Test R² | Avg Dollar Error |
|-------|--------|---------|-----------------|
| Ridge Baseline | 0.629 | — | $246 |
| Random Forest | 0.720 | — | $247 |
| **XGBoost** | **0.764** | **0.763** | **$192** |

**What was added:**

| Feature | Description | Reasoning |
|---------|-------------|-----------|
| `listing_age_years` | Days since first review ÷ 365 | Established listings command higher prices |
| `is_recently_active` | Reviewed in last 6 months (binary) | Freshness signals active, bookable listing |
| `reviews_per_year` | Review count ÷ listing age | Review velocity — popular listings get reviewed more |
| `bathrooms_per_guest` | Bathrooms ÷ accommodates | Normalized comfort ratio |
| `bedrooms_per_guest` | Bedrooms ÷ accommodates | Normalized space ratio |
| `amenity_density` | Amenity count ÷ accommodates | Amenities per person, not just total count |

**What happened:** Val R² moved from 0.769 → 0.764 (slight decrease) while Test R² moved from 0.761 → 0.763 (slight increase). The new features marginally hurt validation but marginally helped generalization — suggesting they add genuine signal but the model needed better hyperparameter tuning to extract full value from them.

---

### Version 4 — Hyperparameter Tuning ✅ (current best)

| Model | Val R² | Test R² | Avg Dollar Error |
|-------|--------|---------|-----------------|
| Ridge Baseline | 0.629 | — | $1,170 |
| Random Forest | 0.720 | — | $247 |
| XGBoost (default) | 0.764 | 0.763 | $192 |
| **XGBoost (tuned) ✓** | **0.771** | **0.768** | **$188** |

**Tuning strategy — 4 steps:**

| Step | Method | Purpose | Result |
|------|--------|---------|--------|
| 1 | Early Stopping (2000 estimators) | Find optimal tree count automatically | best_n = 627 → Val R² 0.7664 |
| 2 | RandomizedSearchCV (50 combos, 3-fold) | Explore wide hyperparameter space | Best region found, Val R² 0.7657 |
| 3 | GridSearchCV (72 combos, 3-fold) | Zoom into best region precisely | Best params confirmed, Val R² 0.7689 |
| 4 | Best params + Early Stopping | Re-tune tree count for final params | **722 trees, Val R² 0.7711** |

**Final hyperparameters:**

```
max_depth        = 8       # deeper trees than default (6) — more expressive
learning_rate    = 0.04    # slower than default (0.05) — learns more carefully
subsample        = 0.8     # 80% of rows per tree — reduces overfitting
colsample_bytree = 0.8     # 80% of features per tree — reduces overfitting
min_child_weight = 1       # minimum samples in leaf
gamma            = 0       # no minimum gain threshold needed
reg_alpha        = 0.01    # light L1 regularization
reg_lambda       = 1       # standard L2 regularization
n_estimators     = 722     # found by early stopping on final params
```

**What improved:**

| Metric | Before Tuning | After Tuning | Δ |
|--------|--------------|--------------|---|
| Val R² | 0.7641 | **0.7711** | +0.0070 |
| Test R² | 0.7627 | **0.7680** | +0.0053 |
| Dollar Error | $191.51 | **$188.06** | **−$3.45** |

Val and Test R² gap is only 0.003 — excellent generalization, no overfitting.

---

## ⚠️ Leakage Identified and Fixed

During self-review, I identified **target leakage** in Version 1.

`price_per_person` and `price_per_bedroom` were derived directly from the target variable `price` and incorrectly included in X — allowing the model to mathematically reconstruct the answer:

```
price_per_bedroom = price / bedrooms
→ price = price_per_bedroom × bedrooms   ← model just does this in reverse
```

**Fix:** Removed both columns from X in `05_ml_pipeline.ipynb` before splitting. They remain in the processed dataset for reference but are excluded from model training. Scores dropped from 0.991 → 0.761 — a more honest and useful number.

Identifying and fixing target leakage independently is one of the most important skills in real-world ML — it's one of the most common and dangerous mistakes in production pipelines.

---

## 🔍 Model Explainability — SHAP Values

Feature importance tells you which features matter globally. SHAP goes further — it explains **why the model made each individual prediction**, assigning a + or − contribution to every feature for every listing.

### Summary Plot — What drives price across all 2,052 listings?

![SHAP Summary](plots/shap_summary.png)

Each dot is one listing. Pink = high feature value, blue = low. Position on the x-axis = impact on the predicted price (log scale).

Key findings:
- **`neighbourhood_cleansed`** — the single most impactful feature overall, surpassing even `accommodates`. High-value neighbourhoods push price up by 0.5–1.0 log points; low-value ones pull it down just as sharply. **Location dominates everything.**
- **`accommodates`** — large capacity (pink) strongly pushes price up. The most extreme high-price outliers are all high-capacity listings, spread far to the right.
- **`bedrooms`** — consistent positive impact. More bedrooms = higher price across the board.
- **`reviews_per_year`** — an engineered feature in the **top 5**. High review velocity signals a popular, in-demand listing. SHAP confirms this feature adds genuine predictive value.
- **`longitude`** — east/west position within NYC matters independently of neighbourhood. Western Manhattan vs eastern Queens is priced differently even within the same borough.
- **`bathrooms_per_guest`** — engineered comfort ratio in the **top 15**, validating that normalized ratios carry more signal than raw counts alone.

---

### Waterfall Plot — Why did the model predict ~$2,950 for this listing?

![SHAP Waterfall](plots/shap_waterfall.png)

Starting from the average listing price, SHAP shows exactly how each feature pushed the prediction up or down:

| Feature | Value | SHAP Impact |
|---------|-------|-------------|
| `accommodates` | 7 guests | **+1.03** |
| `bathrooms` | 6.5 | **+0.49** |
| `bedrooms` | 6 | **+0.20** |
| `neighbourhood_cleansed` | premium area | **+0.19** |
| `total_availability_score` | 0.339 | **+0.16** |
| `longitude` | −73.993 (Manhattan) | **+0.16** |
| `latitude` | 40.762 | **+0.10** |
| `estimated_occupancy_l365d` | 0 | **+0.09** |
| 88 other features | — | **+0.19** |
| **Base (avg listing)** | — | **5.138** (~$170/night) |
| **Final prediction** | — | **7.984** (~$2,950/night) |

The model correctly identified this as a large, luxury Manhattan property with 7 guests, 6.5 bathrooms, and 6 bedrooms — and built the price up logically, feature by feature, from the $170 average.

---

## 🗺️ Feature Importance

Top 15 features by XGBoost's built-in importance score (tuned model):

![Feature Importance](plots/feature_importance_tuned.png)

Key findings:
- `accommodates` — dominant signal by raw importance score
- `bedrooms` — second strongest predictor
- `neighbourhood_cleansed` — location (target encoded across 200+ neighbourhoods)
- `room_type` — entire homes command a significant price premium
- `bathrooms_per_guest` — engineered feature in top 10 ✅
- Note: SHAP (above) reveals `neighbourhood_cleansed` has *greater overall impact* than `accommodates` when measured by actual prediction shift — a nuance the built-in importance score misses.

---

## 📈 Model Performance

![Actual vs Predicted](plots/actual_vs_predicted_tuned.png)

Points cluster tightly around the perfect prediction line especially in the $50–$250 range. Spread increases at higher prices — expected since luxury listings ($400+) are rarer and harder to generalize from.

---

## 🗂️ Project Structure

```
nyc-airbnb-price-predictor/
│
├── 📓 notebooks/
│   ├── 01_cleaning_listing_data.ipynb   # clean 90-column listings file
│   ├── 02_cleaning_reviews_data.ipynb   # aggregate 700k review rows
│   ├── 03_cleaning_calendar_data.ipynb  # aggregate 12M calendar rows
│   ├── 04_feature_engineering.ipynb     # merge, engineer 45 features, outlier capping
│   ├── 05_ml_pipeline.ipynb             # train/val/test split (leakage fix here)
│   ├── 06_ml_encoding.ipynb             # encoding, scaling, train + tune models (Colab)
│   └── 07_plots.ipynb                   # all visualizations
│
├── 📊 plots/
│   ├── 01_price_distribution.png        # before/after log transform
│   ├── 02_price_by_borough.png          # price by NYC borough
│   ├── 03_price_by_room_type.png        # entire home vs private room
│   ├── 04_price_map.png                 # geographic price heatmap
│   ├── 05_correlation_heatmap.png       # feature correlations
│   ├── actual_vs_predicted_tuned.png    # model performance scatter
│   ├── feature_importance_tuned.png     # XGBoost top 15 features
│   ├── shap_summary.png                 # SHAP beeswarm — all features, all listings
│   └── shap_waterfall.png              # SHAP waterfall — single listing breakdown
│
├── 🤖 models/
│   └── xgboost_tuned_final.pkl          # tuned XGBoost model (best)
│
├── 📁 data/
│   ├── raw/                             # original CSVs (not in repo)
│   ├── processed/                       # cleaned intermediate files
│   └── output/                          # final train/val/test splits
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔍 Pipeline Overview

```
Raw Data                    Cleaning                  Features
─────────                   ────────                  ────────
listings.csv.gz   ──►  drop 47 useless cols  ──►  host_experience_level
reviews.csv.gz    ──►  parse $1,200 → 1200   ──►  total_availability_score
calendar.csv.gz   ──►  fill nulls smartly    ──►  amenity binary flags (12)
                         fix t/f booleans         review_score_avg
35,036 listings   ──►  parse amenity JSON    ──►  listing_age_years
90 raw columns         IQR outlier capping        reviews_per_year
                                                  neighbourhood target enc.
                                                        │
                                                        ▼
                                              20,521 clean rows
                                              45 training features
                                                        │
                                              ┌─────────┼─────────┐
                                           Train      Val       Test
                                          16,416     2,052     2,053
                                                        │
                                    RandomizedSearchCV + GridSearchCV
                                       Early Stopping (722 trees)
                                                        │
                                          XGBoost Regressor (tuned)
                                                        │
                                           SHAP Explainability
                                                        │
                                              R² = 0.768
                                           $188 avg error
```

---

## 🛠️ Key Challenges Solved

**1. Price column entirely NULL in latest scrape**  
Inside Airbnb removed the static price column from recent datasets. Solved by extracting `price_per_night` from the `price_quote_raw` JSON blob in the new scrape format.

**2. 12M row calendar file**  
Aggregated per listing using `groupby().agg()` — reducing 12M rows to one row per listing with `availability_rate` and `total_days`.

**3. Amenities stored as JSON-like strings**  
`'["Wifi", "Kitchen", "TV"]'` parsed with `json.loads()` + loop to create 12 binary feature columns for high-signal amenities.

**4. neighbourhood_cleansed had 200+ unique values**  
One-hot encoding would create 200+ columns. Used **target encoding** — replaced each neighbourhood with its mean `log_price` computed from the train set only, with global mean fallback for unseen neighbourhoods.

**5. Target leakage identified and fixed**  
`price_per_person` and `price_per_bedroom` derived from target variable. Identified during self-review, removed before splitting. Scores corrected from 0.991 → 0.761.

**6. 99999 sentinel values in date features**  
`days_since_first_review` used 99999 for listings with no reviews. Dividing by 365 naively gave 273-year-old listings. Fixed with conditional logic before engineering `listing_age_years` and `reviews_per_year`.

**7. Train/val/test leakage prevention**  
All encoders and scalers fit exclusively on X_train. Target encoding computed from y_train only.

**8. GridSearchCV timeout on free Colab**  
GridSearchCV with too many combinations ran for over an hour on Colab's free tier. Solved by switching to `tree_method='hist'` and narrowing the grid to 72 combinations — reducing runtime from 60+ min to under 15 min with no loss in search quality.

**9. Feature importance vs SHAP disagreement**  
XGBoost's built-in importance ranked `accommodates` #1. SHAP revealed `neighbourhood_cleansed` actually has greater total prediction impact across all listings. Built-in importance counts how often a feature is used for splits; SHAP measures actual prediction shift — a meaningfully different and more reliable signal.

---

## 📦 Data

This project uses the [Inside Airbnb](http://insideairbnb.com/get-the-data/) dataset for **New York City** (April 2026 scrape).

To reproduce locally:

1. Go to http://insideairbnb.com/get-the-data/
2. Scroll to **New York City**
3. Download:
   - `listings.csv.gz` — Detailed listings data (~50MB)
   - `reviews.csv.gz` — Detailed review data
   - `calendar.csv.gz` — Daily availability data (~300MB)
4. Place all three in `data/raw/`

> Raw data files are not included in this repo due to file size.

---

## 🚀 How to Run

```bash
# 1. clone the repo
git clone https://github.com/master-zero1/nyc-airbnb-price-predictor.git
cd nyc-airbnb-price-predictor

# 2. install dependencies
pip install -r requirements.txt

# 3. download data (see above) and place in data/raw/

# 4. run notebooks in order (locally)
# 01 → 02 → 03 → 04 → 05

# 5. run 06_ml_encoding.ipynb in Google Colab for model training
# upload data/output/ CSVs to Colab session
```

---

## 🔮 Use the Trained Model

```python
import joblib
import numpy as np

model = joblib.load('models/xgboost_tuned_final.pkl')

# model outputs log_price — convert back to dollars
prediction = model.predict(X_encoded)
price_dollars = np.expm1(prediction)
print(f"Predicted price: ${price_dollars[0]:.2f}/night")
```

---

## 📚 Tech Stack

| Tool | Purpose |
|------|---------|
| Pandas | Data loading, cleaning, transformation |
| NumPy | Numerical operations, log transforms |
| Scikit-learn | Encoding, scaling, Ridge, Random Forest, CV search |
| XGBoost | Final production model |
| SHAP | Model explainability — per-prediction feature attribution |
| Matplotlib | All visualizations |
| Joblib | Model serialization |

---

## 📖 What I Learned

- **Target leakage** is subtle and devastating — engineered features derived from the target variable inflate scores from 0.761 to 0.991. Always trace every feature back to its source before training.
- **Honest scores matter more than impressive scores** — a 0.768 R² you understand is worth more than a 0.991 you can't explain.
- **Early stopping + grid search together** outperforms either alone — RandomizedSearchCV finds the right hyperparameter region, GridSearchCV zooms in, and early stopping re-optimizes tree count for the final param set.
- **SHAP reveals what feature importance hides** — built-in XGBoost importance ranked `accommodates` #1, but SHAP showed `neighbourhood_cleansed` has greater total prediction impact. The two metrics measure fundamentally different things.
- **IQR capping** preserves data while limiting damage from extreme values.
- **Flag before filling** — null values in review scores carry information (no reviews ≠ bad reviews).
- **Target encoding** for high-cardinality categoricals avoids column explosion.
- **Fit on train only** — the single most important rule in ML preprocessing.
- **log transform** on skewed targets dramatically improves model learning.
- **99999 sentinel values** in engineered features cause silent bugs — always inspect fill values before deriving new features from them.
- **Free-tier compute constraints are real** — GridSearchCV must be designed with runtime in mind, not just search quality.

---

## 🔭 What's Next

- [x] Feature engineering (6 new features)
- [x] Hyperparameter tuning (RandomizedSearchCV + GridSearchCV + Early Stopping)
- [x] SHAP values for model explainability
- [ ] Build a Streamlit web app for live price predictions
- [ ] NLP sentiment analysis on review text (VADER — fast, rule-based)
- [ ] External data: NYC subway proximity using lat/lon

---

## 🤖 A Note on AI-Assisted Development

The Streamlit frontend (`app.py`) was built with Claude (Anthropic) as a 
coding assistant. My focus for this project was the ML pipeline — data 
engineering, feature engineering, model tuning, and explainability — not 
web development.

I made all design and feature decisions for the app (SHAP per-prediction
breakdown, borough comparison, what-if simulator), understood every 
component, and debugged issues like Streamlit's session state resets 
independently.

Using AI tooling to move fast on out-of-scope work while staying focused on 
the core problem is — in my view — the correct engineering decision.
The ML pipeline, feature engineering, hyperparameter tuning, and SHAP 
analysis are entirely my own work.

---

<div align="center">

**Built by [master-zero1](https://github.com/master-zero1)**  
**LIVE DEMO -> https://master-zero1-nyc-airbnb-price-predictor.streamlit.app/**
*NYC Airbnb Data · April 2026 Scrape · XGBoost (tuned) · R² = 0.768 · $188 avg error*

</div>
