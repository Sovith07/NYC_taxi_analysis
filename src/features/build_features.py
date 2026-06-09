#2. create features
import numpy as np
import pandas as pd
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

    save_data(train_final,test_final,output_path)

if __name__ == "__main__":
    main()
