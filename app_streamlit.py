# main.py
import streamlit as st
from joblib import load

# Load the pre-trained RandomForest model
model_path = "models/model.joblib"
model = load(model_path)


def predict(features):
     
    prediction = model.predict([features])[0].item()
    return prediction

def main():
    st.title("Machine Learning Model Prediction")

    st.subheader("Trip Details")

    vendor_id = st.selectbox(
    "Vendor ID",
    options=[1, 2]
)

    passenger_count = st.number_input(
    "Passenger Count",
    min_value=1,
    max_value=10,
    value=1
)

    pickup_longitude = st.number_input(
    "Pickup Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-73.985
)

    pickup_latitude = st.number_input(
    "Pickup Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=40.758
)

    dropoff_longitude = st.number_input(
    "Dropoff Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-73.985
)

    dropoff_latitude = st.number_input(
    "Dropoff Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=40.758
)

    store_and_fwd_flag = st.selectbox(
    "Store and Forward Flag",
    options=[0, 1]
)

    pickup_hour = st.slider(
    "Pickup Hour",
    min_value=0,
    max_value=23,
    value=12
)

    pickup_dayofweek = st.slider(
    "Pickup Day of Week",
    min_value=0,
    max_value=6,
    value=0
)

    pickup_month = st.slider(
    "Pickup Month",
    min_value=1,
    max_value=12,
    value=1
)

    pickup_weekofyear = st.slider(
    "Pickup Week of Year",
    min_value=1,
    max_value=53,
    value=1
)

    is_weekend = st.selectbox(
    "Is Weekend",
    options=[0, 1]
)

    is_night = st.selectbox(
    "Is Night",
    options=[0, 1]
)

    is_rush_hour = st.selectbox(
    "Is Rush Hour",
    options=[0, 1]
)
    if st.button("predict"):
        features=[
    vendor_id,
    passenger_count,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    store_and_fwd_flag,
    pickup_hour,
    pickup_dayofweek,
    pickup_month,
    pickup_weekofyear,
    is_weekend,
    is_night,
    is_rush_hour
]    
    result=predict(features)
    st.success(f"The result is : {result}")

if __name__ == "__main__":
    main()
