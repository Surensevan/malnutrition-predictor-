
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === Load model and data ===
@st.cache_data
def load_model():
    return joblib.load("log_reg_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("final_data.csv")

model = load_model()
df = load_data()

# === UI ===
st.set_page_config(page_title="Malnutrition Risk Predictor", layout="centered")
st.title("🧠 Predict Malnutrition Risk")
st.write("Select a country to assess malnutrition risk based on socioeconomic indicators.")

# === Country selection ===
country = st.selectbox("Select Country", sorted(df['Country'].unique()))

# === Autofill data ===
country_data = df[df['Country'] == country].iloc[0]
gdp = country_data['GDP_per_capita']
food_index = country_data['Avg_Food_Price_Index']

st.markdown("#### Socioeconomic Indicators")
st.metric("GDP per Capita (USD)", f"{gdp:,.2f}")
st.metric("Average Food Price Index", f"{food_index:.2f}")

# === Predict button ===
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame([[gdp, food_index]], columns=['GDP_per_capita', 'Avg_Food_Price_Index'])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # probability of class 1 (High risk)

    risk_label = "🔴 High Risk" if pred == 1 else "🟢 Low Risk"
    st.subheader(f"### Prediction: {risk_label}")
    st.write(f"**Model confidence (High Risk)**: {prob:.2%}")

    # === Plot: Simple risk gauge ===
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([""], [prob], color="red" if pred == 1 else "green")
    ax.set_xlim(0, 1)
    ax.set_title("Risk Probability (High)")
    ax.set_yticks([])
    st.pyplot(fig)

# === Optional: show raw data ===
with st.expander("📊 View Country Data"):
    st.dataframe(df[df['Country'] == country])
