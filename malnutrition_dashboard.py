import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("log_reg_model.pkl")

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Malnutrition Risk Predictor", layout="centered")

st.title("🧠 Predict Malnutrition Risk")
st.write("Enter socioeconomic indicators to predict whether a region is at high risk of malnutrition.")

# User inputs (only features used in training)
gdp_input = st.number_input("GDP per Capita (USD)", min_value=0.0, value=1000.0, step=100.0)
food_index_input = st.number_input("Avg Food Price Index", min_value=0.0, value=100.0, step=10.0)

if st.button("Predict Risk"):
    # Create input DataFrame using correct feature names
    input_df = pd.DataFrame([{
        'GDP_per_capita': gdp_input,
        'Avg_Food_Price_Index': food_index_input
    }])

    try:
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.error("⚠️ High Risk of Malnutrition")
        else:
            st.success("✅ Low Risk of Malnutrition")
    except Exception as e:
        st.exception(f"Prediction error: {e}")
