
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Malnutrition Risk Predictor", layout="centered")

st.title("🧠 Predict Malnutrition Risk")
st.markdown("Enter socioeconomic indicators to predict whether a region is at high risk of malnutrition.")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_log_reg_model.pkl")

model = load_model()

# Input Form
with st.form("malnutrition_form"):
    gdp = st.number_input("GDP per Capita (USD)", min_value=0.0, step=100.0, value=1000.0)
    food_price = st.number_input("Avg Food Price Index", min_value=0.0, step=10.0, value=100.0)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_df = pd.DataFrame([[gdp, food_price]], columns=["GDP_per_capita", "Avg_Food_Price_Index"])

    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("🔍 Prediction Result")
        if pred == 1:
            st.error(f"⚠️ High Risk of Malnutrition (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Risk of Malnutrition (Probability: {prob:.2f})")

        # Risk band charts
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.axvspan(0, 2000, color="red", alpha=0.3, label="High Risk Zone")
        ax.axvspan(2000, 10000, color="green", alpha=0.3, label="Low Risk Zone")
        ax.axvline(gdp, color="blue", lw=3, label="Your Input")
        ax.set_xlim(0, 10000)
        ax.set_title("GDP per Capita Risk Zones")
        ax.get_yaxis().set_visible(False)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 1))
        ax2.axvspan(0, 150, color="green", alpha=0.3, label="Low Risk Zone")
        ax2.axvspan(150, 400, color="red", alpha=0.3, label="High Risk Zone")
        ax2.axvline(food_price, color="blue", lw=3, label="Your Input")
        ax2.set_xlim(0, 400)
        ax2.set_title("Food Price Index Risk Zones")
        ax2.get_yaxis().set_visible(False)
        ax2.legend(loc="upper right")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
