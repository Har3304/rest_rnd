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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")

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
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    results[name] = {"model": model, "preds": preds, "MAE": mae,
                     "RMSE": rmse, "R2": r2, "MAPE": mape}
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
# 6. VISUALIZATIONS  (saved to one figure file)
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 6: Generating Visualizations...")
print("=" * 65)

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor("#0f1117")
DARK  = "#0f1117"
PANEL = "#1a1d27"
GOLD  = "#f5a623"
CYAN  = "#4ecdc4"
RED   = "#e74c3c"
GREEN = "#2ecc71"
BLUE  = "#3498db"
GRAY  = "#7f8c8d"

def ax_style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3040")
    if title:
        ax.set_title(title, color=GOLD, fontsize=12, fontweight="bold", pad=10)

# ── 6a: Monthly Orders Timeline ──
ax1 = fig.add_subplot(4, 2, (1, 2))
ax_style(ax1, "Monthly Total Orders — 5-Year History (2019–2023)")
ax1.fill_between(monthly["date"], monthly["total_orders"],
                 alpha=0.25, color=CYAN)
ax1.plot(monthly["date"], monthly["total_orders"],
         color=CYAN, linewidth=2, label="Actual Monthly Orders")
ax1.plot(monthly["date"], monthly["orders_roll6_mean"],
         color=GOLD, linewidth=1.5, linestyle="--", label="6-Month Rolling Avg")
# Mark Jan 2024 forecast
ax1.axvline(pd.Timestamp("2024-01-01"), color=RED, linestyle=":", linewidth=1.5)
ax1.scatter([pd.Timestamp("2024-01-01")], [ensemble_pred],
            color=RED, zorder=5, s=120, label=f"Jan 2024 Forecast: {ensemble_pred:,}")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.set_ylabel("Orders / Month", color="white")
ax1.legend(framealpha=0.2, labelcolor="white", fontsize=9)

# ── 6b: Actual vs Predicted (Test Set) ──
ax2 = fig.add_subplot(4, 2, 3)
ax_style(ax2, "Test Set: Actual vs Predicted Orders")
test_dates = monthly["date"].iloc[TRAIN_END:]
ax2.plot(test_dates, y_test.values, color=CYAN, marker="o", ms=6, label="Actual")
for name, color in [("Random Forest", GOLD), ("Gradient Boosting", GREEN), ("Ridge Regression", BLUE)]:
    ax2.plot(test_dates, results[name]["preds"], color=color,
             marker="s", ms=5, linestyle="--", alpha=0.8, label=name)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.set_ylabel("Orders / Month", color="white")
ax2.legend(framealpha=0.2, labelcolor="white", fontsize=8)

# ── 6c: Feature Importance (RF) ──
ax3 = fig.add_subplot(4, 2, 4)
ax_style(ax3, "Feature Importance (Random Forest)")
importances = pd.Series(results["Random Forest"]["model"].feature_importances_,
                         index=FEATURE_COLS).sort_values(ascending=True).tail(15)
colors = [GOLD if v > importances.quantile(0.75) else CYAN for v in importances.values]
bars = ax3.barh(importances.index, importances.values, color=colors, edgecolor="none")
ax3.set_xlabel("Importance", color="white")
for bar, val in zip(bars, importances.values):
    ax3.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", color="white", fontsize=7)

# ── 6d: Monthly Orders by Season (Box) ──
ax4 = fig.add_subplot(4, 2, 5)
ax_style(ax4, "Order Distribution by Season")
season_names = {0: "Summer", 1: "Monsoon", 2: "Post-\nMonsoon", 3: "Winter"}
season_data = [monthly[monthly["season_code"] == s]["total_orders"].values
               for s in range(4)]
bp = ax4.boxplot(season_data, patch_artist=True, labels=[season_names[s] for s in range(4)])
colors_box = [RED, BLUE, GREEN, GOLD]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for el in bp["whiskers"] + bp["caps"] + bp["medians"] + bp["fliers"]:
    el.set_color("white")
ax4.set_ylabel("Monthly Orders", color="white")

# ── 6e: Day-of-Week Effect ──
ax5 = fig.add_subplot(4, 2, 6)
ax_style(ax5, "Avg Daily Orders by Day of Week")
dow_avg = df.groupby("day_of_week")["total_orders"].mean()
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
bar_colors = [GREEN if d >= 5 else CYAN for d in range(7)]
ax5.bar(dow_labels, dow_avg.values, color=bar_colors, edgecolor="#1a1d27", linewidth=0.5)
ax5.set_ylabel("Avg Daily Orders", color="white")
for i, v in enumerate(dow_avg.values):
    ax5.text(i, v + 1, f"{v:.0f}", ha="center", color="white", fontsize=8)

# ── 6f: Model Metrics Comparison ──
ax6 = fig.add_subplot(4, 2, 7)
ax_style(ax6, "Model Performance Comparison (Test Set)")
metric_df = pd.DataFrame({
    name: {"MAE": r["MAE"], "RMSE": r["RMSE"], "MAPE%": r["MAPE"]}
    for name, r in results.items()
}).T
x = np.arange(len(metric_df))
width = 0.25
ax6.bar(x - width, metric_df["MAE"],   width, color=GOLD,  label="MAE",   alpha=0.85)
ax6.bar(x,         metric_df["RMSE"],  width, color=CYAN,  label="RMSE",  alpha=0.85)
ax6.bar(x + width, metric_df["MAPE%"], width, color=GREEN, label="MAPE%", alpha=0.85)
ax6.set_xticks(x)
ax6.set_xticklabels(["RF", "GBM", "Ridge"], color="white")
ax6.legend(framealpha=0.2, labelcolor="white", fontsize=9)
ax6.set_ylabel("Error Metric", color="white")

# ── 6g: Jan 2024 Forecast bar ──
ax7 = fig.add_subplot(4, 2, 8)
ax_style(ax7, "Jan 2024 Forecast — All Models + Ensemble")
pred_names  = list(predictions_all.keys()) + ["Ensemble"]
pred_values = list(predictions_all.values()) + [ensemble_pred]
bar_colors2 = [GOLD, GREEN, BLUE, RED]
bars7 = ax7.bar(pred_names, pred_values, color=bar_colors2, edgecolor="#1a1d27",
                linewidth=0.5, alpha=0.85)
ax7.set_ylabel("Forecasted Orders", color="white")
ax7.set_xticklabels(["RF", "GBM", "Ridge", "Ensemble"], color="white")
for bar, val in zip(bars7, pred_values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}", ha="center", color="white", fontsize=10, fontweight="bold")
ax7.set_ylim(0, max(pred_values) * 1.15)

# Title
fig.suptitle("🍽  Indian Restaurant Order Forecasting Dashboard",
             fontsize=18, fontweight="bold", color=GOLD, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("restaurant_forecast_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  ✓ Dashboard saved → restaurant_forecast_dashboard.png")


print("\n" + "=" * 65)
print("  ALL DONE!")
print("=" * 65)
print(f"""
  Files generated:
    1. restaurant_daily_data.csv    – 5-year daily records ({len(df):,} rows)
    2. restaurant_monthly_data.csv  – Monthly aggregated ({len(monthly)} rows)
    3. restaurant_forecast_dashboard.png – Visual dashboard

  Forecast Summary:
    ► Jan 2024 predicted orders: {ensemble_pred:,}
    ► Best model: {best_name} (MAPE: {results[best_name]['MAPE']:.2f}%)
""")
