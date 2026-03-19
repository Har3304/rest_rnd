"""
=======================================================================
  INDIAN RESTAURANT ORDER FORECASTING
  Synthetic Dataset Generation + Predictive Model (Monthly Orders)
=======================================================================
  Features:
  - 5-year daily synthetic dataset (2019-01-01 to 2023-12-31)
  - Indian festival calendar, seasonality, weather, promotions
  - Random Forest + Gradient Boosting ensemble
  - Monthly aggregation & next-month prediction
  - Feature importance plot + forecast visualization
=======================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

def generate_indian_restaurant_data(start="2019-01-01", end="2023-12-31"):
    dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({"date": dates})

    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day"]         = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["week_of_year"]= df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["date"].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # ── Indian Major Festivals (approximate fixed/lunar dates per year) ──
    festivals = {
        # Diwali (Oct/Nov)
        "2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12",
        # Holi (Mar)
        "2019-03-21", "2020-03-10", "2021-03-29", "2022-03-18", "2023-03-08",
        # Eid ul-Fitr
        "2019-06-05", "2020-05-25", "2021-05-13", "2022-05-03", "2023-04-22",
        # Navratri start
        "2019-10-07", "2020-10-17", "2021-10-07", "2022-09-26", "2023-10-15",
        # Durga Puja / Dussehra
        "2019-10-08", "2020-10-25", "2021-10-15", "2022-10-05", "2023-10-24",
        # Christmas
        "2019-12-25", "2020-12-25", "2021-12-25", "2022-12-25", "2023-12-25",
        # New Year's Eve
        "2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31",
        # Independence Day
        "2019-08-15", "2020-08-15", "2021-08-15", "2022-08-15", "2023-08-15",
        # Republic Day
        "2019-01-26", "2020-01-26", "2021-01-26", "2022-01-26", "2023-01-26",
        # Ganesh Chaturthi
        "2019-09-02", "2020-08-22", "2021-09-10", "2022-08-31", "2023-09-19",
    }
    festival_dates = pd.to_datetime(list(festivals))
    df["is_festival_day"] = df["date"].isin(festival_dates).astype(int)

    # Window: ±2 days around festival also get a boost
    festival_window = set()
    for fd in festival_dates:
        for delta in range(-2, 3):
            festival_window.add(fd + pd.Timedelta(days=delta))
    df["festival_window"] = df["date"].isin(festival_window).astype(int)

    # ── Indian Public Holidays ──
    holidays = pd.to_datetime([
        "2019-01-26","2019-03-04","2019-04-14","2019-04-19","2019-05-18",
        "2019-08-15","2019-10-02","2019-10-08","2019-10-27","2019-11-12",
        "2019-12-25",
        "2020-01-26","2020-03-10","2020-04-02","2020-04-10","2020-05-07",
        "2020-08-15","2020-10-02","2020-11-14","2020-12-25",
        "2021-01-26","2021-03-29","2021-04-02","2021-04-14","2021-05-26",
        "2021-08-15","2021-10-02","2021-10-15","2021-11-04","2021-12-25",
        "2022-01-26","2022-03-18","2022-04-14","2022-05-03","2022-08-15",
        "2022-09-26","2022-10-02","2022-10-05","2022-10-24","2022-12-25",
        "2023-01-26","2023-03-08","2023-04-07","2023-04-14","2023-04-22",
        "2023-08-15","2023-09-19","2023-10-02","2023-10-24","2023-12-25",
    ])
    df["is_public_holiday"] = df["date"].isin(holidays).astype(int)

    # ── Season (India: Summer/Monsoon/Winter) ──
    def get_season(m):
        if m in [3, 4, 5]:   return "Summer"
        if m in [6, 7, 8, 9]: return "Monsoon"
        if m in [10, 11]:    return "Post-Monsoon"
        return "Winter"  # Dec, Jan, Feb

    df["season"] = df["month"].apply(get_season)
    season_map = {"Summer": 0, "Monsoon": 1, "Post-Monsoon": 2, "Winter": 3}
    df["season_code"] = df["season"].map(season_map)

    # Season effect on orders
    season_effect = {"Summer": 0.85, "Monsoon": 1.15, "Post-Monsoon": 1.10, "Winter": 1.05}
    df["season_multiplier"] = df["season"].map(season_effect)

    # ── Weather (synthetic: temp, rainfall, humidity) ──
    month_temp = {1:18, 2:21, 3:27, 4:33, 5:36, 6:34, 7:30, 8:29, 9:29, 10:27, 11:22, 12:18}
    month_rain = {1:5, 2:8, 3:10, 4:15, 5:20, 6:120, 7:200, 8:180, 9:100, 10:40, 11:15, 12:8}
    month_humidity = {1:55, 2:50, 3:45, 4:40, 5:45, 6:70, 7:85, 8:85, 9:80, 10:70, 11:60, 12:58}

    df["temperature_c"]   = df["month"].map(month_temp) + np.random.normal(0, 2, len(df))
    df["rainfall_mm"]     = (df["month"].map(month_rain) * 
                             np.random.exponential(1, len(df))).clip(0, 400)
    df["humidity_pct"]    = (df["month"].map(month_humidity) + 
                             np.random.normal(0, 5, len(df))).clip(20, 100)

    # Heavy rain reduces walk-in orders
    df["heavy_rain"] = (df["rainfall_mm"] > 30).astype(int)

    # ── Promotions & Discounts ──
    # Random promo days (~10% of days) — weekend promos more likely
    promo_prob = np.where(df["is_weekend"] == 1, 0.20, 0.07)
    df["is_promotion_day"]  = (np.random.random(len(df)) < promo_prob).astype(int)
    df["discount_pct"]      = np.where(df["is_promotion_day"] == 1,
                                       np.random.uniform(10, 30, len(df)), 0).round(1)

    # ── Online Delivery Availability (grows over time) ──
    # Assumed: joined Swiggy/Zomato from mid-2019
    df["online_delivery_active"] = (df["date"] >= "2019-07-01").astype(int)
    # Delivery platform growth factor (gradual increase)
    days_since_delivery = (df["date"] - pd.Timestamp("2019-07-01")).dt.days.clip(lower=0)
    df["delivery_growth_factor"] = (1 + days_since_delivery / 1500).clip(upper=1.6)

    # ── Staff Count (affects capacity) ──
    base_staff = 8
    df["staff_on_duty"] = (base_staff + 
                           np.random.randint(-2, 3, len(df)) +
                           df["is_weekend"] * 2 +
                           df["is_festival_day"] * 3).clip(lower=4)

    # ── Competitor Restaurant Opening ──
    # A competitor opened nearby in March 2021
    df["competitor_nearby"] = (df["date"] >= "2021-03-01").astype(int)

    # ── COVID Lockdown Impact ──
    df["lockdown_level"] = 0
    # Full lockdown: Apr–Jun 2020
    df.loc[(df["date"] >= "2020-04-01") & (df["date"] <= "2020-06-30"), "lockdown_level"] = 3
    # Partial: Jul–Oct 2020
    df.loc[(df["date"] >= "2020-07-01") & (df["date"] <= "2020-10-31"), "lockdown_level"] = 2
    # Minor restrictions: Nov 2020 – Mar 2021
    df.loc[(df["date"] >= "2020-11-01") & (df["date"] <= "2021-03-31"), "lockdown_level"] = 1

    # ── Average Order Value (INR) ──
    base_aov = 350
    df["avg_order_value_inr"] = (base_aov +
                                  np.random.normal(0, 40, len(df)) +
                                  df["is_weekend"] * 50 +
                                  df["is_festival_day"] * 80 +
                                  df["discount_pct"] * (-2) +
                                  df["year"].apply(lambda y: (y - 2019) * 20)  # inflation
                                 ).clip(lower=200)

    # ── Menu Items Count (restaurant grew its menu) ──
    df["menu_item_count"] = (40 + (df["year"] - 2019) * 5 + 
                              np.random.randint(-2, 3, len(df))).clip(lower=35)

    # ── Google Rating (slow drift with noise) ──
    df["google_rating"] = (4.1 + np.random.normal(0, 0.15, len(df))).clip(1.0, 5.0).round(1)

    # ──────────────────────────────────────────────────
    # SYNTHETIC ORDER COUNT GENERATION
    # Realistic base + multiplicative effects
    # ──────────────────────────────────────────────────
    base_orders = 120  # baseline weekday orders

    # Day-of-week effect
    dow_effect = {0: 0.85, 1: 0.82, 2: 0.88, 3: 0.90, 4: 1.00, 5: 1.35, 6: 1.25}
    df["dow_effect"] = df["day_of_week"].map(dow_effect)

    # Monthly seasonality (Indian dining patterns)
    month_effect = {1: 0.95, 2: 0.90, 3: 1.00, 4: 0.95, 5: 0.88,
                    6: 1.05, 7: 1.10, 8: 1.08, 9: 1.05, 10: 1.15, 11: 1.20, 12: 1.25}
    df["month_effect"] = df["month"].map(month_effect)

    # Year-over-year growth (5% per year before COVID)
    df["yoy_growth"] = df["year"].apply(lambda y: 1 + (y - 2019) * 0.05)

    orders = (
        base_orders
        * df["dow_effect"]
        * df["month_effect"]
        * df["season_multiplier"]
        * df["yoy_growth"]
        * df["delivery_growth_factor"]
        + df["is_festival_day"] * 60
        + df["festival_window"] * 25
        + df["is_public_holiday"] * 30
        + df["is_promotion_day"] * 40
        - df["heavy_rain"] * 20
        - df["lockdown_level"].map({0: 0, 1: 30, 2: 70, 3: 100})
        - df["competitor_nearby"] * 8
        + df["staff_on_duty"] * 1.5
        + np.random.normal(0, 12, len(df))  # daily random noise
    ).clip(lower=0)

    df["total_orders"] = orders.round().astype(int)

    # ── Lag Features (rolling history) ──
    df = df.sort_values("date").reset_index(drop=True)
    df["orders_lag_1"]  = df["total_orders"].shift(1)
    df["orders_lag_7"]  = df["total_orders"].shift(7)
    df["orders_lag_30"] = df["total_orders"].shift(30)
    df["orders_roll_7_mean"]  = df["total_orders"].shift(1).rolling(7).mean()
    df["orders_roll_30_mean"] = df["total_orders"].shift(1).rolling(30).mean()
    df["orders_roll_7_std"]   = df["total_orders"].shift(1).rolling(7).std()

    return df


print("=" * 65)
print("  STEP 1: Generating Synthetic Dataset...")
print("=" * 65)
df = generate_indian_restaurant_data()
print(f"  ✓ Dataset shape  : {df.shape}")
print(f"  ✓ Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  ✓ Total days     : {len(df)}")
print(f"  ✓ Avg daily orders: {df['total_orders'].mean():.1f}")
print(f"  ✓ Min / Max orders: {df['total_orders'].min()} / {df['total_orders'].max()}")


# ─────────────────────────────────────────────
# 2. MONTHLY AGGREGATION
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 2: Aggregating to Monthly Level...")
print("=" * 65)

monthly = df.groupby(["year", "month"]).agg(
    total_orders          = ("total_orders",         "sum"),
    avg_daily_orders      = ("total_orders",         "mean"),
    festival_days         = ("is_festival_day",      "sum"),
    holiday_days          = ("is_public_holiday",    "sum"),
    promotion_days        = ("is_promotion_day",     "sum"),
    weekend_days          = ("is_weekend",           "sum"),
    avg_temperature       = ("temperature_c",        "mean"),
    total_rainfall        = ("rainfall_mm",          "sum"),
    avg_humidity          = ("humidity_pct",         "mean"),
    heavy_rain_days       = ("heavy_rain",           "sum"),
    avg_order_value       = ("avg_order_value_inr",  "mean"),
    avg_staff             = ("staff_on_duty",        "mean"),
    avg_google_rating     = ("google_rating",        "mean"),
    menu_item_count       = ("menu_item_count",      "last"),
    lockdown_level        = ("lockdown_level",       "mean"),
    online_delivery_active= ("online_delivery_active","max"),
    competitor_nearby     = ("competitor_nearby",    "max"),
    delivery_growth       = ("delivery_growth_factor","mean"),
    season_code           = ("season_code",          "first"),
    quarter               = ("quarter",              "first"),
).reset_index()

monthly["date"] = pd.to_datetime(monthly[["year","month"]].assign(day=1))
monthly = monthly.sort_values("date").reset_index(drop=True)

# Monthly lag features
monthly["orders_lag_1m"]  = monthly["total_orders"].shift(1)
monthly["orders_lag_2m"]  = monthly["total_orders"].shift(2)
monthly["orders_lag_3m"]  = monthly["total_orders"].shift(3)
monthly["orders_lag_12m"] = monthly["total_orders"].shift(12)
monthly["orders_roll3_mean"] = monthly["total_orders"].shift(1).rolling(3).mean()
monthly["orders_roll6_mean"] = monthly["total_orders"].shift(1).rolling(6).mean()
monthly["pct_change_1m"] = monthly["total_orders"].pct_change(1)

monthly = monthly.dropna().reset_index(drop=True)

print(f"  ✓ Monthly records: {len(monthly)}")
print(f"  ✓ Columns        : {len(monthly.columns)}")
print(f"\n  Monthly Order Summary:")
print(monthly[["year","month","total_orders"]].to_string(index=False))


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & MODEL TRAINING
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 3: Feature Engineering & Model Training...")
print("=" * 65)

FEATURE_COLS = [
    # Temporal
    "month", "quarter", "season_code",
    # Lag & rolling
    "orders_lag_1m", "orders_lag_2m", "orders_lag_3m",
    "orders_lag_12m", "orders_roll3_mean", "orders_roll6_mean",
    # Festival / events
    "festival_days", "holiday_days", "promotion_days", "weekend_days",
    # Weather
    "avg_temperature", "total_rainfall", "avg_humidity", "heavy_rain_days",
    # Business
    "avg_order_value", "avg_staff", "avg_google_rating", "menu_item_count",
    # External
    "lockdown_level", "online_delivery_active", "competitor_nearby",
    "delivery_growth",
]

TARGET = "total_orders"

X = monthly[FEATURE_COLS]
y = monthly[TARGET]

# Time-based train/test split: last 6 months = test
TRAIN_END = len(monthly) - 6
X_train, X_test = X.iloc[:TRAIN_END], X.iloc[TRAIN_END:]
y_train, y_test = y.iloc[:TRAIN_END], y.iloc[TRAIN_END:]

print(f"  ✓ Train samples  : {len(X_train)}")
print(f"  ✓ Test  samples  : {len(X_test)}")

# ── Models ──
models = {
    "Random Forest":        RandomForestRegressor(n_estimators=300, max_depth=8,
                                                   min_samples_leaf=2, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                       learning_rate=0.05, random_state=42),
    "Ridge Regression":     Ridge(alpha=10.0),
}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

results = {}
for name, model in models.items():
    if name == "Ridge Regression":
        model.fit(X_train_scaled, y_train)
        train_preds = model.predict(X_train_scaled)
        preds       = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        preds       = model.predict(X_test)

    mae       = mean_absolute_error(y_test, preds)
    rmse      = np.sqrt(mean_squared_error(y_test, preds))
    r2        = r2_score(y_test, preds)
    mape      = np.mean(np.abs((y_test - preds) / y_test)) * 100
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2  = r2_score(y_train, train_preds)

    results[name] = {
        "model": model, "preds": preds,
        "train_preds": train_preds,
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
        "train_MAE": train_mae, "train_R2": train_r2,
    }
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.0f} orders/month")
    print(f"    RMSE : {rmse:.0f}")
    print(f"    R²   : {r2:.4f}")
    print(f"    MAPE : {mape:.2f}%")

# Best model by MAPE
best_name = min(results, key=lambda n: results[n]["MAPE"])
best_model = results[best_name]["model"]
print(f"\n  ★ Best model: {best_name} (MAPE: {results[best_name]['MAPE']:.2f}%)")


# ─────────────────────────────────────────────
# 4. NEXT-MONTH PREDICTION
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 4: Predicting Next Month (Jan 2024)...")
print("=" * 65)

last_row = monthly.iloc[-1]

next_month_features = {
    "month":                  1,      # January 2024
    "quarter":                1,
    "season_code":            3,      # Winter
    "orders_lag_1m":          last_row["total_orders"],
    "orders_lag_2m":          monthly.iloc[-2]["total_orders"],
    "orders_lag_3m":          monthly.iloc[-3]["total_orders"],
    "orders_lag_12m":         monthly[monthly["month"] == 1]["total_orders"].iloc[-1],
    "orders_roll3_mean":      monthly["total_orders"].iloc[-3:].mean(),
    "orders_roll6_mean":      monthly["total_orders"].iloc[-6:].mean(),
    "festival_days":          1,      # Republic Day
    "holiday_days":           1,
    "promotion_days":         4,
    "weekend_days":           8,      # Jan 2024 has 4 Sat + 4 Sun
    "avg_temperature":        18.0,
    "total_rainfall":         6.0,
    "avg_humidity":           55.0,
    "heavy_rain_days":        0,
    "avg_order_value":        440.0,
    "avg_staff":              10.0,
    "avg_google_rating":      4.2,
    "menu_item_count":        61,
    "lockdown_level":         0.0,
    "online_delivery_active": 1,
    "competitor_nearby":      1,
    "delivery_growth":        1.58,
}

next_X = pd.DataFrame([next_month_features])[FEATURE_COLS]

predictions_all = {}
for name, res in results.items():
    m = res["model"]
    if name == "Ridge Regression":
        pred = m.predict(scaler.transform(next_X))[0]
    else:
        pred = m.predict(next_X)[0]
    predictions_all[name] = round(pred)
    print(f"  {name:25s}: {round(pred):,} orders")

ensemble_pred = round(np.mean(list(predictions_all.values())))
print(f"\n  {'Ensemble (avg)':25s}: {ensemble_pred:,} orders")
print(f"\n  ★ FINAL FORECAST for Jan 2024: {ensemble_pred:,} orders")
print(f"     ≈ {ensemble_pred/31:.0f} orders/day  |  "
      f"Estimated Revenue: ₹{ensemble_pred * 440:,.0f}")


# ─────────────────────────────────────────────
# 5. SAVE DATASET
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 5: Saving Datasets...")
print("=" * 65)

df.to_csv("restaurant_daily_data.csv", index=False)
monthly.to_csv("restaurant_monthly_data.csv", index=False)
print("  ✓ Daily dataset   → restaurant_daily_data.csv")
print("  ✓ Monthly dataset → restaurant_monthly_data.csv")


# ─────────────────────────────────────────────
# 6. GENERATE HTML DASHBOARD
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 6: Generating HTML Dashboard...")
print("=" * 65)

import json

# ── Prepare data for charts ──
test_date_labels  = monthly["date"].iloc[TRAIN_END:].dt.strftime("%b %Y").tolist()
all_date_labels   = monthly["date"].dt.strftime("%b %Y").tolist()
all_actuals       = monthly["total_orders"].tolist()
all_roll6         = monthly["orders_roll6_mean"].fillna(0).round().astype(int).tolist()
test_actuals      = y_test.tolist()

# Build full-series predictions (train + test) per model
def full_preds(name):
    r = results[name]
    return [round(float(v)) for v in list(r["train_preds"]) + list(r["preds"])]

rf_full    = full_preds("Random Forest")
gb_full    = full_preds("Gradient Boosting")
ridge_full = full_preds("Ridge Regression")

rf_test    = [round(float(v)) for v in results["Random Forest"]["preds"]]
gb_test    = [round(float(v)) for v in results["Gradient Boosting"]["preds"]]
ridge_test = [round(float(v)) for v in results["Ridge Regression"]["preds"]]

# Residuals
rf_resid    = [int(a) - p for a, p in zip(test_actuals, rf_test)]
gb_resid    = [int(a) - p for a, p in zip(test_actuals, gb_test)]
ridge_resid = [int(a) - p for a, p in zip(test_actuals, ridge_test)]

# Feature importances (RF, top 15)
fi = pd.Series(
    results["Random Forest"]["model"].feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False).head(15)
fi_labels = fi.index.tolist()
fi_values = [round(float(v), 4) for v in fi.values]

# Seasonal box data
season_box = {}
snames = {0: "Summer", 1: "Monsoon", 2: "Post-Monsoon", 3: "Winter"}
for code, label in snames.items():
    vals = monthly[monthly["season_code"] == code]["total_orders"].tolist()
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    season_box[label] = {
        "min":    round(float(max(min(vals), q1 - 1.5 * iqr))),
        "q1":     round(float(q1)),
        "median": round(float(med)),
        "q3":     round(float(q3)),
        "max":    round(float(min(max(vals), q3 + 1.5 * iqr))),
    }

# Day-of-week averages
dow_avg   = df.groupby("day_of_week")["total_orders"].mean().round(1).tolist()
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Jan 2024 forecast bar
forecast_names  = list(predictions_all.keys()) + ["Ensemble"]
forecast_values = list(predictions_all.values()) + [ensemble_pred]

# Model metric summary rows
model_order = ["Random Forest", "Gradient Boosting", "Ridge Regression"]
metrics_js  = json.dumps([{
    "name":      n,
    "mape":      round(results[n]["MAPE"], 2),
    "mae":       round(results[n]["MAE"]),
    "rmse":      round(results[n]["RMSE"]),
    "r2":        round(results[n]["R2"], 4),
    "train_mae": round(results[n]["train_MAE"]),
    "train_r2":  round(results[n]["train_R2"], 4),
} for n in model_order])

# Overfitting chart data
train_r2_vals = [round(results[n]["train_R2"], 4) for n in model_order]
test_r2_vals  = [round(results[n]["R2"], 4)       for n in model_order]

# ── Build HTML ──
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Restaurant Order Forecast — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0b0e14;--surface:#131720;--surface2:#1a1f2e;
  --border:rgba(255,255,255,0.07);--text:#e8eaf0;--muted:#6b7280;
  --blue:#4f8ef7;--green:#34d399;--amber:#fbbf24;
  --rose:#fb7185;--purple:#a78bfa;--cyan:#22d3ee;
  --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif;
}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6;padding:2rem}}
.header{{display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:2rem;border-bottom:1px solid var(--border);padding-bottom:1.25rem}}
.header h1{{font-family:var(--mono);font-size:1.05rem;font-weight:700;letter-spacing:.04em}}
.header p{{font-size:11px;color:var(--muted);margin-top:4px;font-family:var(--mono)}}
.badge{{background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.3);color:var(--blue);font-size:11px;font-family:var(--mono);padding:4px 10px;border-radius:4px}}
.tabs{{display:flex;gap:6px;margin-bottom:1.25rem;flex-wrap:wrap}}
.tab{{background:transparent;border:1px solid var(--border);color:var(--muted);font-size:12px;font-family:var(--mono);padding:6px 16px;border-radius:4px;cursor:pointer;transition:all .15s}}
.tab:hover{{border-color:rgba(255,255,255,.2);color:var(--text)}}
.tab.active{{background:rgba(79,142,247,.12);border-color:var(--blue);color:var(--blue)}}
.metrics-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1rem}}
.mc{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.25rem}}
.mc-label{{font-size:11px;font-family:var(--mono);color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px}}
.mc-value{{font-size:1.7rem;font-family:var(--mono);font-weight:700;line-height:1}}
.mc-sub{{font-size:11px;color:var(--muted);margin-top:4px}}
.good{{color:var(--green)}}.warn{{color:var(--amber)}}.bad{{color:var(--rose)}}
.alert{{border-radius:6px;padding:10px 14px;font-size:12px;font-family:var(--mono);margin-bottom:1.75rem;border-left:3px solid}}
.alert-good{{background:rgba(52,211,153,.08);border-color:var(--green);color:var(--green)}}
.alert-warn{{background:rgba(251,191,36,.08);border-color:var(--amber);color:var(--amber)}}
.alert-bad{{background:rgba(251,113,133,.08);border-color:var(--rose);color:var(--rose)}}
.sec{{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:2rem 0 .75rem;display:flex;align-items:center;gap:8px}}
.sec::after{{content:'';flex:1;height:1px;background:var(--border)}}
.chart-wrap{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.25rem;margin-bottom:1.25rem}}
.chart-title{{font-size:12px;font-family:var(--mono);color:var(--muted);margin-bottom:1rem}}
.cc{{position:relative;width:100%}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}}
.li{{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);font-family:var(--mono)}}
.ld{{width:18px;height:3px;border-radius:2px;display:inline-block}}
table{{width:100%;border-collapse:collapse}}
.tw{{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:1.25rem}}
thead th{{background:var(--surface2);font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:10px 14px;text-align:left;border-bottom:1px solid var(--border)}}
thead th:not(:first-child){{text-align:right}}
tbody td{{padding:8px 14px;font-size:13px;border-bottom:1px solid var(--border);font-family:var(--mono)}}
tbody td:not(:first-child){{text-align:right}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover{{background:var(--surface2)}}
tfoot td{{padding:9px 14px;font-size:12px;font-family:var(--mono);font-weight:700;background:var(--surface2);border-top:1px solid var(--border);text-align:right}}
tfoot td:first-child{{text-align:left;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;font-size:11px}}
.eg{{color:var(--green)}}.ew{{color:var(--amber)}}.eb{{color:var(--rose)}}
.model-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:2rem}}
.mcard{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.25rem;position:relative}}
.mcard.best{{border-color:var(--green);background:rgba(52,211,153,.05)}}
.mcard-name{{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--text);margin-bottom:10px}}
.mrow{{display:flex;justify-content:space-between;font-size:12px;font-family:var(--mono);margin:3px 0}}
.mrow span:first-child{{color:var(--muted)}}
.best-badge{{position:absolute;top:10px;right:10px;background:rgba(52,211,153,.15);color:var(--green);font-size:10px;font-family:var(--mono);padding:2px 8px;border-radius:3px;border:1px solid rgba(52,211,153,.3)}}
.footer{{margin-top:2.5rem;border-top:1px solid var(--border);padding-top:1rem;font-size:11px;font-family:var(--mono);color:var(--muted);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
@media(max-width:700px){{.metrics-grid{{grid-template-columns:repeat(2,1fr)}}.model-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>INDIAN RESTAURANT ORDER FORECAST — ACCURACY DASHBOARD</h1>
    <p>Train: Jan 2020 – Jun 2023 (42 months) &nbsp;·&nbsp; Test: Jul – Dec 2023 (6 months) &nbsp;·&nbsp; Forecast: Jan 2024</p>
  </div>
  <span class="badge">SYNTHETIC DATASET 2019–2023</span>
</div>

<!-- ── TABS ── -->
<div class="tabs">
  <button class="tab active" onclick="switchModel('ridge')" id="tab-ridge">Ridge Regression</button>
  <button class="tab" onclick="switchModel('gb')"    id="tab-gb">Gradient Boosting</button>
  <button class="tab" onclick="switchModel('rf')"    id="tab-rf">Random Forest</button>
</div>

<!-- RIDGE panel -->
<div id="panel-ridge">
  <div class="metrics-grid">
    <div class="mc"><div class="mc-label">MAPE — test</div><div class="mc-value good">{results["Ridge Regression"]["MAPE"]:.2f}%</div><div class="mc-sub">Mean abs % error</div></div>
    <div class="mc"><div class="mc-label">MAE — test</div><div class="mc-value">{round(results["Ridge Regression"]["MAE"]):,}</div><div class="mc-sub">orders / month</div></div>
    <div class="mc"><div class="mc-label">RMSE — test</div><div class="mc-value">{round(results["Ridge Regression"]["RMSE"]):,}</div><div class="mc-sub">orders / month</div></div>
    <div class="mc"><div class="mc-label">R² — test</div><div class="mc-value warn">{results["Ridge Regression"]["R2"]:.4f}</div><div class="mc-sub">variance explained</div></div>
  </div>
  <div class="alert alert-good">Train MAE = {round(results["Ridge Regression"]["train_MAE"]):,} &nbsp;·&nbsp; Train R² = {results["Ridge Regression"]["train_R2"]:.4f} — minimal gap. Best generalisation of all three models.</div>
</div>

<!-- GB panel -->
<div id="panel-gb" style="display:none">
  <div class="metrics-grid">
    <div class="mc"><div class="mc-label">MAPE — test</div><div class="mc-value warn">{results["Gradient Boosting"]["MAPE"]:.2f}%</div><div class="mc-sub">Mean abs % error</div></div>
    <div class="mc"><div class="mc-label">MAE — test</div><div class="mc-value">{round(results["Gradient Boosting"]["MAE"]):,}</div><div class="mc-sub">orders / month</div></div>
    <div class="mc"><div class="mc-label">RMSE — test</div><div class="mc-value">{round(results["Gradient Boosting"]["RMSE"]):,}</div><div class="mc-sub">orders / month</div></div>
    <div class="mc"><div class="mc-label">R² — test</div><div class="mc-value warn">{results["Gradient Boosting"]["R2"]:.4f}</div><div class="mc-sub">variance explained</div></div>
  </div>
  <div class="alert alert-bad">Train MAE = {round(results["Gradient Boosting"]["train_MAE"]):,} &nbsp;·&nbsp; Train R² = {results["Gradient Boosting"]["train_R2"]:.4f} — model memorised training data (overfitting). Test R² = {results["Gradient Boosting"]["R2"]:.2f}.</div>
</div>

<!-- RF panel -->
<div id="panel-rf" style="display:none">
  <div class="metrics-grid">
    <div class="mc"><div class="mc-label">MAPE — test</div><div class="mc-value bad">{results["Random Forest"]["MAPE"]:.2f}%</div><div class="mc-sub">Mean abs % error</div></div>
    <div class="mc"><div class="mc-label">MAE — test</div><div class="mc-value bad">{round(results["Random Forest"]["MAE"]):,}</div><div class="mc-sub">orders / month</div></div>
    <div class="mc"><div class="mc-label">RMSE — test</div><div class="mc-value bad">{round(results["Random Forest"]["RMSE"]):,}</div><div class="mc-sub">orders / month</div></div>
    <div class="mc"><div class="mc-label">R² — test</div><div class="mc-value bad">{results["Random Forest"]["R2"]:.4f}</div><div class="mc-sub">variance explained</div></div>
  </div>
  <div class="alert alert-bad">Train R² = {results["Random Forest"]["train_R2"]:.4f} but Test R² = {results["Random Forest"]["R2"]:.4f} — severe overfitting. Performs below a naive mean predictor on test data.</div>
</div>

<!-- ── CHART 1: Full history + predictions ── -->
<div class="sec">Full history — actual vs predicted (all months)</div>
<div class="chart-wrap">
  <div class="chart-title">Monthly orders Jan 2020 – Dec 2023 · train region shaded · test region highlighted · Jan 2024 forecast marked</div>
  <div class="cc" style="height:300px"><canvas id="fullChart"></canvas></div>
  <div class="legend">
    <span class="li"><span class="ld" style="background:var(--blue)"></span>Actual</span>
    <span class="li"><span class="ld" style="background:var(--green);opacity:.6"></span>Ridge (all)</span>
    <span class="li"><span class="ld" style="background:var(--amber);opacity:.6"></span>Gradient Boosting (all)</span>
    <span class="li"><span class="ld" style="background:var(--rose);opacity:.6"></span>Random Forest (all)</span>
    <span class="li"><span class="ld" style="background:var(--purple)"></span>Ensemble forecast Jan 2024</span>
  </div>
</div>

<!-- ── CHART 2: Actual vs Predicted test window ── -->
<div class="sec">Actual vs predicted — test window (Jul–Dec 2023)</div>
<div class="chart-wrap">
  <div class="chart-title">Close-up of the 6-month held-out test set</div>
  <div class="cc" style="height:260px"><canvas id="avpChart"></canvas></div>
  <div class="legend">
    <span class="li"><span class="ld" style="background:var(--blue)"></span>Actual</span>
    <span class="li"><span class="ld" style="background:var(--green)"></span>Ridge</span>
    <span class="li"><span class="ld" style="background:var(--amber)"></span>Gradient Boosting</span>
    <span class="li"><span class="ld" style="background:var(--rose)"></span>Random Forest</span>
  </div>
</div>

<!-- ── CHART 3: Residuals ── -->
<div class="sec">Residual errors — test set (actual − predicted)</div>
<div class="chart-wrap">
  <div class="chart-title">Positive = under-predicted &nbsp;·&nbsp; Negative = over-predicted</div>
  <div class="cc" style="height:200px"><canvas id="residChart"></canvas></div>
  <div class="legend">
    <span class="li"><span class="ld" style="background:var(--green)"></span>Ridge</span>
    <span class="li"><span class="ld" style="background:var(--amber)"></span>GB</span>
    <span class="li"><span class="ld" style="background:var(--rose)"></span>RF</span>
  </div>
</div>

<!-- ── CHARTS 4+5 side by side ── -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
  <div>
    <div class="sec">Train vs test R² — overfitting check</div>
    <div class="chart-wrap" style="margin-bottom:0">
      <div class="chart-title">Large gap = overfitting</div>
      <div class="cc" style="height:200px"><canvas id="overChart"></canvas></div>
      <div class="legend">
        <span class="li"><span class="ld" style="background:var(--blue)"></span>Train R²</span>
        <span class="li"><span class="ld" style="background:var(--rose)"></span>Test R²</span>
      </div>
    </div>
  </div>
  <div>
    <div class="sec">Jan 2024 forecast — all models</div>
    <div class="chart-wrap" style="margin-bottom:0">
      <div class="chart-title">Ensemble avg = {ensemble_pred:,} orders</div>
      <div class="cc" style="height:200px"><canvas id="forecastChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── CHARTS 6+7 side by side ── -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
  <div>
    <div class="sec">Feature importance (Random Forest, top 15)</div>
    <div class="chart-wrap" style="margin-bottom:0">
      <div class="chart-title">Higher = more influential on predictions</div>
      <div class="cc" style="height:380px"><canvas id="fiChart"></canvas></div>
    </div>
  </div>
  <div>
    <div class="sec">Order distribution by season</div>
    <div class="chart-wrap" style="margin-bottom:0">
      <div class="chart-title">Monsoon & Winter drive highest order volumes</div>
      <div class="cc" style="height:220px"><canvas id="seasonChart"></canvas></div>
      <div class="sec" style="margin-top:1rem">Avg daily orders by day of week</div>
      <div class="chart-title">Weekends outperform weekdays significantly</div>
      <div class="cc" style="height:140px"><canvas id="dowChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── TABLE: Month-by-month ── -->
<div class="sec" style="margin-top:1.5rem">Month-by-month breakdown — test set</div>
<div class="tw">
  <table>
    <thead>
      <tr>
        <th>Month</th>
        <th>Actual</th>
        <th>Ridge</th><th>Ridge err</th>
        <th>GB</th><th>GB err</th>
        <th>RF</th><th>RF err</th>
      </tr>
    </thead>
    <tbody id="compTable"></tbody>
    <tfoot id="compFoot"></tfoot>
  </table>
</div>

<!-- ── MODEL SUMMARY CARDS ── -->
<div class="sec">Model summary</div>
<div class="model-grid">
  <div class="mcard best">
    <div class="best-badge">BEST</div>
    <div class="mcard-name">Ridge Regression</div>
    <div class="mrow"><span>MAPE</span><span class="good">{results["Ridge Regression"]["MAPE"]:.2f}%</span></div>
    <div class="mrow"><span>MAE</span><span>{round(results["Ridge Regression"]["MAE"]):,} orders</span></div>
    <div class="mrow"><span>RMSE</span><span>{round(results["Ridge Regression"]["RMSE"]):,}</span></div>
    <div class="mrow"><span>Test R²</span><span class="warn">{results["Ridge Regression"]["R2"]:.4f}</span></div>
    <div class="mrow"><span>Train R²</span><span>{results["Ridge Regression"]["train_R2"]:.4f}</span></div>
    <div class="mrow"><span>Overfit gap</span><span class="good">{abs(results["Ridge Regression"]["train_R2"] - results["Ridge Regression"]["R2"]):.2f}</span></div>
  </div>
  <div class="mcard">
    <div class="mcard-name">Gradient Boosting</div>
    <div class="mrow"><span>MAPE</span><span class="warn">{results["Gradient Boosting"]["MAPE"]:.2f}%</span></div>
    <div class="mrow"><span>MAE</span><span>{round(results["Gradient Boosting"]["MAE"]):,} orders</span></div>
    <div class="mrow"><span>RMSE</span><span>{round(results["Gradient Boosting"]["RMSE"]):,}</span></div>
    <div class="mrow"><span>Test R²</span><span class="warn">{results["Gradient Boosting"]["R2"]:.4f}</span></div>
    <div class="mrow"><span>Train R²</span><span>{results["Gradient Boosting"]["train_R2"]:.4f}</span></div>
    <div class="mrow"><span>Overfit gap</span><span class="bad">{abs(results["Gradient Boosting"]["train_R2"] - results["Gradient Boosting"]["R2"]):.2f}</span></div>
  </div>
  <div class="mcard">
    <div class="mcard-name">Random Forest</div>
    <div class="mrow"><span>MAPE</span><span class="bad">{results["Random Forest"]["MAPE"]:.2f}%</span></div>
    <div class="mrow"><span>MAE</span><span class="bad">{round(results["Random Forest"]["MAE"]):,} orders</span></div>
    <div class="mrow"><span>RMSE</span><span class="bad">{round(results["Random Forest"]["RMSE"]):,}</span></div>
    <div class="mrow"><span>Test R²</span><span class="bad">{results["Random Forest"]["R2"]:.4f}</span></div>
    <div class="mrow"><span>Train R²</span><span>{results["Random Forest"]["train_R2"]:.4f}</span></div>
    <div class="mrow"><span>Overfit gap</span><span class="bad">{abs(results["Random Forest"]["train_R2"] - results["Random Forest"]["R2"]):.2f}</span></div>
  </div>
</div>

<div class="footer">
  <span>Generated by restaurant_forecast.py &nbsp;·&nbsp; Synthetic dataset 2019–2023</span>
  <span>Train: 42 months &nbsp;·&nbsp; Test: 6 months (Jul–Dec 2023) &nbsp;·&nbsp; Best model: {best_name} (MAPE {results[best_name]["MAPE"]:.2f}%)</span>
</div>

<script>
// ── Data ──
const allDates   = {json.dumps(all_date_labels)};
const allActuals = {json.dumps(all_actuals)};
const trainEnd   = {TRAIN_END};
const testDates  = {json.dumps(test_date_labels)};
const testActuals= {json.dumps([int(v) for v in test_actuals])};
const rfFull     = {json.dumps(rf_full)};
const gbFull     = {json.dumps(gb_full)};
const ridgeFull  = {json.dumps(ridge_full)};
const rfTest     = {json.dumps(rf_test)};
const gbTest     = {json.dumps(gb_test)};
const ridgeTest  = {json.dumps(ridge_test)};
const rfResid    = {json.dumps(rf_resid)};
const gbResid    = {json.dumps(gb_resid)};
const ridgeResid = {json.dumps(ridge_resid)};
const fiLabels   = {json.dumps(fi_labels)};
const fiValues   = {json.dumps(fi_values)};
const seasonData = {json.dumps(season_box)};
const dowAvg     = {json.dumps(dow_avg)};
const dowLabels  = {json.dumps(dow_labels)};
const forecastNames  = {json.dumps(forecast_names)};
const forecastValues = {json.dumps([int(v) for v in forecast_values])};
const trainR2    = {json.dumps(train_r2_vals)};
const testR2     = {json.dumps(test_r2_vals)};
const modelNames = ['Random Forest','Gradient Boosting','Ridge Regression'];
const ensemblePred = {ensemble_pred};
const metrics    = {metrics_js};

const gc = {{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}}}};
const gx = {{color:'rgba(255,255,255,0.06)'}};
const gt = (cb) => ({{color:'#6b7280',callback:cb}});

// ── Chart 1: Full history ──
new Chart('fullChart', {{
  type:'line',
  data:{{
    labels:allDates,
    datasets:[
      {{label:'Actual',data:allActuals,borderColor:'#4f8ef7',backgroundColor:'rgba(79,142,247,0.08)',borderWidth:2.5,pointRadius:2,tension:.3,fill:true,order:1}},
      {{label:'Ridge',data:ridgeFull,borderColor:'rgba(52,211,153,.55)',borderWidth:1.5,pointRadius:0,tension:.3,borderDash:[4,3],order:2}},
      {{label:'GB',data:gbFull,borderColor:'rgba(251,191,36,.55)',borderWidth:1.5,pointRadius:0,tension:.3,borderDash:[4,3],order:3}},
      {{label:'RF',data:rfFull,borderColor:'rgba(251,113,133,.55)',borderWidth:1.5,pointRadius:0,tension:.3,borderDash:[4,3],order:4}},
    ]
  }},
  options:{{...gc,
    plugins:{{...gc.plugins,
      annotation:{{annotations:{{
        trainEnd:{{type:'line',xMin:trainEnd-.5,xMax:trainEnd-.5,borderColor:'rgba(167,139,250,.5)',borderWidth:1.5,borderDash:[4,3]}},
        forecast:{{type:'point',xValue:allDates.length,yValue:ensemblePred,radius:8,backgroundColor:'#a78bfa',borderColor:'#a78bfa'}},
      }}}},
    }},
    scales:{{
      x:{{ticks:{{...gt(),autoSkip:true,maxTicksLimit:16,maxRotation:0}},grid:gx}},
      y:{{ticks:{{...gt(v=>v.toLocaleString())}},grid:gx}},
    }}
  }}
}});

// ── Chart 2: Test window ──
new Chart('avpChart', {{
  type:'line',
  data:{{
    labels:testDates,
    datasets:[
      {{label:'Actual',data:testActuals,borderColor:'#4f8ef7',backgroundColor:'rgba(79,142,247,0.1)',borderWidth:2.5,pointRadius:5,tension:.3,fill:true}},
      {{label:'Ridge',data:ridgeTest,borderColor:'#34d399',borderWidth:2,pointRadius:4,borderDash:[5,3],tension:.3}},
      {{label:'GB',data:gbTest,borderColor:'#fbbf24',borderWidth:2,pointRadius:4,borderDash:[5,3],tension:.3}},
      {{label:'RF',data:rfTest,borderColor:'#fb7185',borderWidth:2,pointRadius:4,borderDash:[3,4],tension:.3}},
    ]
  }},
  options:{{...gc,
    scales:{{
      x:{{ticks:{{...gt(),autoSkip:false,maxRotation:0}},grid:gx}},
      y:{{min:7500,ticks:{{...gt(v=>v.toLocaleString())}},grid:gx}},
    }}
  }}
}});

// ── Chart 3: Residuals ──
new Chart('residChart', {{
  type:'bar',
  data:{{
    labels:testDates,
    datasets:[
      {{label:'Ridge',data:ridgeResid,backgroundColor:'#34d399',borderRadius:3}},
      {{label:'GB',data:gbResid,backgroundColor:'#fbbf24',borderRadius:3}},
      {{label:'RF',data:rfResid,backgroundColor:'#fb7185',borderRadius:3}},
    ]
  }},
  options:{{...gc,scales:{{x:{{ticks:{{...gt(),autoSkip:false,maxRotation:0}},grid:gx}},y:{{ticks:gt(),grid:gx}}}}}}
}});

// ── Chart 4: Overfitting ──
new Chart('overChart', {{
  type:'bar',
  data:{{
    labels:['RF','GB','Ridge'],
    datasets:[
      {{label:'Train R²',data:trainR2,backgroundColor:'#4f8ef7',borderRadius:3}},
      {{label:'Test R²', data:testR2, backgroundColor:'#fb7185',borderRadius:3}},
    ]
  }},
  options:{{...gc,scales:{{x:{{ticks:gt(),grid:{{display:false}}}},y:{{min:-0.25,max:1.1,ticks:gt(),grid:gx}}}}}}
}});

// ── Chart 5: Forecast ──
new Chart('forecastChart', {{
  type:'bar',
  data:{{
    labels:['RF','GB','Ridge','Ensemble'],
    datasets:[{{
      data:forecastValues,
      backgroundColor:['#fb7185','#fbbf24','#34d399','#a78bfa'],
      borderRadius:4,
    }}]
  }},
  options:{{...gc,
    plugins:{{...gc.plugins,datalabels:{{display:false}},tooltip:{{callbacks:{{label:ctx=>ctx.parsed.y.toLocaleString()+' orders'}}}}}},
    scales:{{
      x:{{ticks:gt(),grid:{{display:false}}}},
      y:{{ticks:{{...gt(v=>v.toLocaleString()),maxTicksLimit:5}},grid:gx}},
    }}
  }}
}});

// ── Chart 6: Feature importance ──
new Chart('fiChart', {{
  type:'bar',
  data:{{
    labels:fiLabels,
    datasets:[{{data:fiValues,backgroundColor:fiValues.map(v=>v>0.15?'#fbbf24':'#4f8ef7'),borderRadius:3}}]
  }},
  options:{{...gc,
    indexAxis:'y',
    scales:{{
      x:{{ticks:gt(),grid:gx}},
      y:{{ticks:{{...gt(),font:{{size:11}}}},grid:{{display:false}}}},
    }}
  }}
}});

// ── Chart 7: Season box-ish (grouped bar) ──
const seasons = Object.keys(seasonData);
new Chart('seasonChart', {{
  type:'bar',
  data:{{
    labels:seasons,
    datasets:[
      {{label:'Q1 (25th pct)',data:seasons.map(s=>seasonData[s].q1),backgroundColor:'rgba(79,142,247,.3)',borderRadius:3}},
      {{label:'Median',data:seasons.map(s=>seasonData[s].median),backgroundColor:'#4f8ef7',borderRadius:3}},
      {{label:'Q3 (75th pct)',data:seasons.map(s=>seasonData[s].q3),backgroundColor:'rgba(79,142,247,.6)',borderRadius:3}},
    ]
  }},
  options:{{...gc,
    scales:{{
      x:{{ticks:gt(),grid:{{display:false}}}},
      y:{{ticks:{{...gt(v=>v.toLocaleString()),maxTicksLimit:5}},grid:gx}},
    }}
  }}
}});

// ── Chart 8: Day of week ──
new Chart('dowChart', {{
  type:'bar',
  data:{{
    labels:dowLabels,
    datasets:[{{
      data:dowAvg,
      backgroundColor:dowAvg.map((_,i)=>i>=5?'#34d399':'rgba(79,142,247,.6)'),
      borderRadius:3,
    }}]
  }},
  options:{{...gc,
    scales:{{
      x:{{ticks:gt(),grid:{{display:false}}}},
      y:{{ticks:{{...gt(),maxTicksLimit:4}},grid:gx}},
    }}
  }}
}});

// ── Comparison table ──
const tbody = document.getElementById('compTable');
let rmae=0, gmae=0, fmae=0;
testDates.forEach((d,i) => {{
  const re = testActuals[i]-ridgeTest[i], ge=testActuals[i]-gbTest[i], fe=testActuals[i]-rfTest[i];
  rmae+=Math.abs(re); gmae+=Math.abs(ge); fmae+=Math.abs(fe);
  const ec = e => Math.abs(e)<300?'eg':Math.abs(e)<600?'ew':'eb';
  const fmt = e => (e>0?'+':'')+e.toLocaleString();
  tbody.innerHTML += `<tr>
    <td>${{d}}</td>
    <td>${{testActuals[i].toLocaleString()}}</td>
    <td>${{ridgeTest[i].toLocaleString()}}</td><td class="${{ec(re)}}">${{fmt(re)}}</td>
    <td>${{gbTest[i].toLocaleString()}}</td><td class="${{ec(ge)}}">${{fmt(ge)}}</td>
    <td>${{rfTest[i].toLocaleString()}}</td><td class="${{ec(fe)}}">${{fmt(fe)}}</td>
  </tr>`;
}});
const n = testDates.length;
document.getElementById('compFoot').innerHTML = `
  <tr><td>MAE</td><td></td><td></td><td class="eg">${{Math.round(rmae/n).toLocaleString()}}</td><td></td><td class="ew">${{Math.round(gmae/n).toLocaleString()}}</td><td></td><td class="eb">${{Math.round(fmae/n).toLocaleString()}}</td></tr>
  <tr><td>MAPE</td><td></td><td></td><td class="eg">{results["Ridge Regression"]["MAPE"]:.2f}%</td><td></td><td class="ew">{results["Gradient Boosting"]["MAPE"]:.2f}%</td><td></td><td class="eb">{results["Random Forest"]["MAPE"]:.2f}%</td></tr>
`;

// ── Tab switching ──
function switchModel(m) {{
  ['ridge','gb','rf'].forEach(id => {{
    document.getElementById('panel-'+id).style.display = id===m?'':'none';
    document.getElementById('tab-'+id).classList.toggle('active', id===m);
  }});
}}
</script>
</body>
</html>"""

with open("restaurant_forecast_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)
print("  ✓ Dashboard saved → restaurant_forecast_dashboard.html")


print("\n" + "=" * 65)
print("  ALL DONE!")
print("=" * 65)
print(f"""
  Files generated:
    1. restaurant_daily_data.csv         – 5-year daily records ({len(df):,} rows)
    2. restaurant_monthly_data.csv       – Monthly aggregated ({len(monthly)} rows)
    3. restaurant_forecast_dashboard.html – Interactive HTML dashboard

  Forecast Summary:
    ► Jan 2024 predicted orders: {ensemble_pred:,}
    ► Best model: {best_name} (MAPE: {results[best_name]['MAPE']:.2f}%)
    ► Estimated Revenue: ₹{ensemble_pred * 440:,.0f}
""")
