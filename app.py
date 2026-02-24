import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("flight_rf.pkl", "rb"))

st.title("Flight Price Prediction")

# Departure
dep_time = st.datetime_input("Departure Date & Time")

# Arrival
arr_time = st.datetime_input("Arrival Date & Time")

# Source
source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai"])

# Destination
destination = st.selectbox("Destination",
                           ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"])

# Stops
stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])

# Airline
airline = st.selectbox("Airline",
                       ["Jet Airways", "IndiGo", "Air India", "Multiple carriers",
                        "SpiceJet", "Vistara", "GoAir",
                        "Multiple carriers Premium economy",
                        "Jet Airways Business", "Vistara Premium economy", "Trujet"])

if st.button("Predict"):

    Journey_day = dep_time.day
    Journey_month = dep_time.month
    Dep_hour = dep_time.hour
    Dep_min = dep_time.minute

    Arrival_hour = arr_time.hour
    Arrival_min = arr_time.minute

    dur_hour = abs(Arrival_hour - Dep_hour)
    dur_min = abs(Arrival_min - Dep_min)

    # Airline Encoding
    airlines = ["Jet Airways","IndiGo","Air India","Multiple carriers",
                "SpiceJet","Vistara","GoAir",
                "Multiple carriers Premium economy",
                "Jet Airways Business","Vistara Premium economy","Trujet"]

    airline_encoded = [1 if airline == a else 0 for a in airlines]

    # Source Encoding
    sources = ["Delhi","Kolkata","Mumbai","Chennai"]
    source_encoded = [1 if source == s else 0 for s in sources]

    # Destination Encoding
    destinations = ["Cochin","Delhi","Hyderabad","Kolkata","New Delhi"]
    destination_encoded = [1 if destination == d else 0 for d in destinations]

    prediction = model.predict([[
        stops,
        Journey_day,
        Journey_month,
        Dep_hour,
        Dep_min,
        Arrival_hour,
        Arrival_min,
        dur_hour,
        dur_min,
        airline_encoded[2],  # Air India
        airline_encoded[6],  # GoAir
        airline_encoded[1],  # IndiGo
        airline_encoded[0],  # Jet Airways
        airline_encoded[8],  # Jet Airways Business
        airline_encoded[3],  # Multiple carriers
        airline_encoded[7],  # Multiple carriers Premium economy
        airline_encoded[4],  # SpiceJet
        airline_encoded[10], # Trujet
        airline_encoded[5],  # Vistara
        airline_encoded[9],  # Vistara Premium economy
        source_encoded[3],   # Chennai
        source_encoded[0],   # Delhi
        source_encoded[1],   # Kolkata
        source_encoded[2],   # Mumbai
        destination_encoded[0], # Cochin
        destination_encoded[1], # Delhi
        destination_encoded[2], # Hyderabad
        destination_encoded[3], # Kolkata
        destination_encoded[4]  # New Delhi
    ]])

    st.success(f"Estimated Flight Price: ₹ {round(prediction[0],2)}")

st.markdown(
    """
    <hr style="margin-top:50px;">
    <div style="text-align: center; color: grey; font-size: 14px;">
        © 2026 Aditya Kumar Jha
    </div>
    """,
    unsafe_allow_html=True
)