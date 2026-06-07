#2. create features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import sys

def count_null(df):
    print("\n" + "=" * 55)
    print("1. Count Null")
    print("=" * 55)

# check null values
    null_count=df.isnull().sum()
    print(f"null_count: {null_count}")

#convert columns to datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

# Zero-coordinate entries
    zero_coords = ((df["pickup_longitude"] == 0) | (df["pickup_latitude"] == 0) | (df["dropoff_longitude"] == 0) | (df["dropoff_latitude"] == 0)).sum()
    print(f"Zero-coordinate rows : {zero_coords:,}")

# Negative / zero duration
    bad_dur = (df["trip_duration"] <= 0).sum()
    print(f"Zero/negative duration: {bad_dur:,}")


# dropoff before pickup
    bad_time = (df["dropoff_datetime"] < df["pickup_datetime"]).sum()
    print(f"Dropoff before pickup : {bad_time:,}")


# NYC bounding box
NYC_LON = (-74.05, -73.75)
NYC_LAT = (40.63,  40.85)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("2. OUTLIER DETECTION")
    print("=" * 55)
 
    # ── Duration IQR filter ────────────────────────────────────────
    Q1, Q3 = df["trip_duration"].quantile([0.25, 0.75])
    IQR     = Q3 - Q1
    dur_lo, dur_hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    dur_hi  = min(dur_hi, 24 * 3600)   # hard cap: 24 hours
    dur_lo  = max(dur_lo, 60)          # hard floor: 60 seconds
 
    n_dur = ((df["trip_duration"] < dur_lo) | (df["trip_duration"] > dur_hi)).sum()
    print(f"Duration outliers (IQR): {n_dur:,}  [{dur_lo:.0f}s – {dur_hi:.0f}s]")


    # ── Coordinate bounds ──────────────────────────────────────────
    out_coords = (
        ~df["pickup_longitude"].between(*NYC_LON) |
        ~df["pickup_latitude"].between(*NYC_LAT)  |
        ~df["dropoff_longitude"].between(*NYC_LON)|
        ~df["dropoff_latitude"].between(*NYC_LAT)
    )
    print(f"Coords outside NYC     : {out_coords.sum():,}")

    mask = (df["trip_duration"].between(dur_lo, dur_hi) &
        ~out_coords)
    df_clean = df[mask].copy()
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("3. FEATURE ENGINEERING")
    print("=" * 55)
 
    # ── Datetime features ──────────────────────────────────────────
    df["pickup_hour"]      = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek   # 0=Mon
    df["pickup_month"]     = df["pickup_datetime"].dt.month
    df["pickup_weekofyear"]= df["pickup_datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"]       = (df["pickup_dayofweek"] >= 5).astype(int)
    df["is_night"]         = (df["pickup_hour"].between(22, 23) |
                               df["pickup_hour"].between(0, 5)).astype(int)
    df["is_rush_hour"]     = (
        df["pickup_hour"].between(7, 9) | df["pickup_hour"].between(16, 18)
    ).astype(int) * (1 - df["is_weekend"]) 

    # ── Target transform ───────────────────────────────────────────
    df["log_trip_duration"] = np.log1p(df["trip_duration"])

    df = df.drop(
    columns=[
        'id',
        'pickup_datetime',
        'dropoff_datetime',
    ])

    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({
    'N': 0,
    'Y': 1
})  

    new_cols = [
        "pickup_hour", "pickup_dayofweek", "pickup_month", "pickup_weekofyear",
        "is_weekend", "is_night", "is_rush_hour", "log_trip_duration"]
    
    print("New columns:", ", ".join(new_cols))



    return df

def run_eda(df: pd.DataFrame, save_prefix: str = "eda") :
    print("\n" + "=" * 55)
    print("5. EXPLORATORY DATA ANALYSIS")
    print("=" * 55)
 
    # ── 5a. Target distribution ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(df["trip_duration"], bins=100, color="#4c72b0", edgecolor="none")
    axes[0].set_title("Trip duration (raw seconds)")
    axes[0].set_xlabel("Seconds")
 
    axes[1].hist(df["log_trip_duration"], bins=100, color="#55a868", edgecolor="none")
    axes[1].set_title("log1p(trip_duration)  — target")
    axes[1].set_xlabel("log(seconds + 1)")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_01_target_distribution.png")
    plt.close(fig)
    print("Saved: target distribution")
 
    # ── 5b. Passenger count distribution ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    vc = df["passenger_count"].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="#c44e52", edgecolor="none")
    ax.set_title("Passenger count distribution")
    ax.set_xlabel("Passengers")
    ax.set_ylabel("Trips")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_02_passenger_count.png")
    plt.close(fig)
    print("Saved: passenger count")
 
    # ── 5c. Hourly & weekly patterns ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
 
    hourly = df.groupby("pickup_hour")["trip_duration"].median()  #trip durations often contain extreme values.
    axes[0].plot(hourly.index, hourly.values, marker="o", color="#4c72b0")
    axes[0].set_title("Median trip duration by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Seconds")
    axes[0].axvspan(7, 10, alpha=0.12, color="red", label="AM rush")
    axes[0].axvspan(16, 19, alpha=0.12, color="orange", label="PM rush")
    axes[0].legend(fontsize=8)
 
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df.groupby("pickup_dayofweek")["trip_duration"].median()
    axes[1].bar(day_labels, weekly.values, color="#8172b2", edgecolor="none")
    axes[1].set_title("Median trip duration by day of week")
    axes[1].set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_03_time_patterns.png")
    plt.close(fig)
    print("Saved: time patterns")
 
 
    # ── 5d. Correlation heatmap ────────────────────────────────────
    heat_cols = [
        "log_trip_duration", "passenger_count", "pickup_hour",
        "pickup_dayofweek", "is_weekend", "is_rush_hour"
    ]
    corr = df[heat_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_05_correlation_heatmap.png")
    plt.close(fig)
    print("Saved: correlation heatmap")
 
    # ── 5e. Geo scatter — pickup hotspots ─────────────────────────
    sample_geo = df.sample(min(10000, len(df)), random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (lon_col, lat_col, title) in zip(axes, [
        ("pickup_longitude",  "pickup_latitude",  "Pickup locations"),
        ("dropoff_longitude", "dropoff_latitude", "Dropoff locations"),
    ]):
        sc = ax.scatter(
            sample_geo[lon_col], sample_geo[lat_col],
            c=sample_geo["log_trip_duration"], cmap="YlOrRd",
            alpha=0.25, s=2
        )
        plt.colorbar(sc, ax=ax, label="log(duration)")
        ax.set_xlim(*NYC_LON)
        ax.set_ylim(*NYC_LAT)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_06_geo_scatter.png")
    plt.close(fig)
    print("Saved: geo scatter")
 
    # ── 5f. Vendor comparison ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df.groupby("vendor_id")["trip_duration"].median().plot(
        kind="bar", ax=axes[0], color=["#4c72b0", "#55a868"], edgecolor="none"
    )
    axes[0].set_title("Median trip duration by vendor")
    axes[0].set_xlabel("Vendor ID")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=0)
 
    df.groupby("vendor_id").size().plot(
        kind="bar", ax=axes[1], color=["#4c72b0", "#55a868"], edgecolor="none"
    )
    axes[1].set_title("Trip count by vendor")
    axes[1].set_xlabel("Vendor ID")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_07_vendor_comparison.png")
    plt.close(fig)
    print("Saved: vendor comparison")
 
    print("\nAll EDA plots saved.")


def save_data(train,test,output_path):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    train.to_csv(output_path + '/train_csv',index=False)
    test.to_csv(output_path + '/test_csv',index=False)

def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent.parent


    input_file=sys.argv[1]
    data_path=home_dir.as_posix() + input_file
    output_path=home_dir.as_posix() +'/data/preprocessed'
    
    train_data=pd.read_csv(data_path + '/train_csv')
    test_data=pd.read_csv(data_path + '/test_csv')

    count_null(train_data)
    count_null(test_data)

    train_final=detect_outliers(train_data)
    test_final=detect_outliers(test_data)

    train_final = engineer_features(train_final)
    test_final = engineer_features(test_final)

    run_eda(train_final,save_prefix="taxi_analysis")
    run_eda(test_final,save_prefix="taxi_analysis")

    save_data(train_final,test_final,output_path)

if __name__ == "__main__":
    main()
