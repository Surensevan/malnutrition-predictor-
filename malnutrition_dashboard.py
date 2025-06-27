import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load model
@st.cache_data

def load_model():
    return joblib.load("log_reg_model.pkl")

model = load_model()

# Dummy country data (replace with actual data)
data = pd.DataFrame({
    "Country": ["Fiji", "India", "Nigeria", "Brazil", "USA"],
    "GDP_per_capita": [1995.72, 2100.34, 1800.67, 8900.12, 62000.45],
    "Food_Price_Index": [54.94, 85.23, 92.45, 65.34, 50.23]
})

st.set_page_config(page_title="Malnutrition Risk Predictor", layout="centered")
st.title("\U0001F9E0 Predict Malnutrition Risk")
st.markdown("Select a country to assess malnutrition risk based on socioeconomic indicators.")

selected_country = st.selectbox("Select Country", data["Country"].unique())

# Extract country data
gdp = data[data["Country"] == selected_country]["GDP_per_capita"].values[0]
fpi = data[data["Country"] == selected_country]["Food_Price_Index"].values[0]

# Display indicators in cards
col1, col2, col3 = st.columns(3)
col1.metric("🌍 Country", selected_country)
col2.metric("💰 GDP per Capita (USD)", f"${gdp:,.2f}")
col3.metric("🍽️ Food Price Index", f"{fpi:.2f}")

# Predict button
if st.button("\U0001F50D Predict Risk"):
    input_df = pd.DataFrame([[gdp, fpi]], columns=["GDP_per_capita", "Avg_Food_Price_Index"])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)

    # Display prediction result
    st.subheader("\U0001F50D Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Malnutrition")
    else:
        st.success("✅ Low Risk of Malnutrition")

    confidence_percent = round(prob[0][1]*100, 2)
    st.progress(confidence_percent / 100)
    st.caption(f"Model Confidence for High Risk: **{confidence_percent}%**")

    # Display gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_percent,
        title={'text': "Malnutrition Risk Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if prediction == 1 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ]
        }
    ))
    st.plotly_chart(fig)
