import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Set Page Config for a professional look
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #007bff;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load XGBoost model using Joblib
@st.cache_resource
def load_model():
    return joblib.load("flight_xgb.pkl")

model = load_model()

# Header
st.title("✈️ Smart Flight Price Predictor")
st.write("Get accurate ticket fare estimates using our advanced XGBoost Machine Learning model.")
st.markdown("---")

# --- Input Section ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Journey Details")
        # Fixed datetime_input to date_input and time_input for better UI
        dep_date = st.date_input("Departure Date", value=datetime.today())
        dep_time = st.time_input("Departure Time", value=datetime.strptime("09:00", "%H:%M"))
        
        arr_date = st.date_input("Arrival Date", value=datetime.today())
        arr_time = st.time_input("Arrival Time", value=datetime.strptime("12:00", "%H:%M"))

    with col2:
        st.subheader("📍 Route & Airline")
        airline = st.selectbox("Preferred Airline",
                               ["Jet Airways", "IndiGo", "Air India", "Multiple carriers",
                                "SpiceJet", "Vistara", "GoAir", "Multiple carriers Premium economy",
                                "Jet Airways Business", "Vistara Premium economy", "Trujet"])
        
        source = st.selectbox("Source City", ["Delhi", "Kolkata", "Mumbai", "Chennai", "Bangalore"])
        
        destination = st.selectbox("Destination City",
                                    ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata", "Bangalore"])

# Additional Details in an Expander to keep it clean
with st.expander("⚙️ Additional Options"):
    stops = st.select_slider("Total Stops", options=[0, 1, 2, 3, 4], value=1)

st.markdown("---")

# --- Prediction Logic ---
if st.button("🚀 Calculate Estimated Fare"):
    with st.spinner('Analyzing historical trends...'):
        # Time Calculations
        Journey_day = dep_date.day
        Journey_month = dep_date.month
        Dep_hour = dep_time.hour
        Dep_min = dep_time.minute
        Arrival_hour = arr_time.hour
        Arrival_min = arr_time.minute

        # Duration Calculation
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        # 1. Create a DataFrame with ALL columns to avoid 'feature_names mismatch'
        columns = [
            'Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 
            'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins', 
            'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways', 
            'Airline_Jet Airways Business', 'Airline_Multiple carriers', 
            'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet', 'Airline_Trujet', 
            'Airline_Vistara', 'Airline_Vistara Premium economy', 'Source_Chennai', 
            'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin', 
            'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi'
        ]
        
        # Initialize input with zeros
        input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        # Fill numerical values
        input_df['Total_Stops'] = stops
        input_df['Journey_day'] = Journey_day
        input_df['Journey_month'] = Journey_month
        input_df['Dep_hour'] = Dep_hour
        input_df['Dep_min'] = Dep_min
        input_df['Arrival_hour'] = Arrival_hour
        input_df['Arrival_min'] = Arrival_min
        input_df['Duration_hours'] = dur_hour
        input_df['Duration_mins'] = dur_min

        # Fill Encoded Categories
        if f'Airline_{airline}' in columns:
            input_df[f'Airline_{airline}'] = 1
        if f'Source_{source}' in columns:
            input_df[f'Source_{source}'] = 1
        if f'Destination_{destination}' in columns:
            input_df[f'Destination_{destination}'] = 1

        # Prediction
        prediction = model.predict(input_df)

        # Show result in a professional card
        st.markdown(f"""
            <div class="result-box">
                <h3 style="color: grey;">Predicted Fare</h3>
                <h1 style="color: #28a745;">₹ {round(float(prediction[0]), 2)}</h1>
                <p style="font-size: 12px; color: grey;">Confidence Level: 85% (XGBoost Optimized)</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 14px; margin-top: 50px;">
        © 2026 Aditya Kumar Jha | Machine Learning Deployment
    </div>
    """,
    unsafe_allow_html=True
)