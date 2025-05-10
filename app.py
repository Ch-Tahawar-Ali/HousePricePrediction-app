import streamlit as st
import pandas as pd
import numpy as np
# other imports...

# THIS MUST BE FIRST Streamlit command
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Now you can use other Streamlit commands
st.title("Welcome to the House Price Predictor App")
# rest of your code...



# THEN LOAD MODEL/DATA ▼▼▼
@st.cache_resource
def load_model():
    return load("model_pipeline.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

# THEN THE REST OF YOUR APP CODE ▼▼▼
st.title("California House Price Prediction 🏠")
# ... rest of your code ...t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# Add this at the top after imports
def add_derived_features(data):
    """MUST BE THE EXACT SAME FUNCTION AS IN YOUR TRAINING CODE"""
    data = data.copy()
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data

# Then keep the rest of your app.py code

# Load model and data
@st.cache_resource
def load_model():
    return load("model_pipeline.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

model = load_model()
data = load_data()




# Sidebar inputs
st.sidebar.header("Enter Property Details")
with st.sidebar.form("input_form"):
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    age = st.number_input("House Median Age", min_value=1, value=41)
    rooms = st.number_input("Total Rooms", min_value=1, value=880)
    bedrooms = st.number_input("Total Bedrooms", min_value=1, value=129)
    population = st.number_input("Population", min_value=1, value=322)
    households = st.number_input("Households", min_value=1, value=126)
    income = st.number_input("Median Income", min_value=0.0, value=8.3252)
    ocean = st.selectbox("Ocean Proximity", options=[
        'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])
    
    submitted = st.form_submit_button("Predict Price")

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [age],
            'total_rooms': [rooms],
            'total_bedrooms': [bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [income],
            'ocean_proximity': [ocean]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction Result")
        st.metric("Estimated House Value", f"${prediction:,.2f}")
        
        # Show input summary
        st.subheader("Input Summary")
        st.write(input_data)

with col2:
    st.subheader("Data Distribution")
    plot_type = st.selectbox("Choose a feature to visualize", 
                           options=['median_income', 'housing_median_age', 'total_rooms'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[plot_type], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {plot_type.replace('_', ' ').title()}")
    st.pyplot(fig)

# Show raw data option
if st.checkbox("Show Raw Data"):
    st.subheader("Housing Data")
    st.dataframe(data)