
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Malnutrition Risk Dashboard", layout="centered")

# Load dataset and model
@st.cache_data
def load_data():
    df = pd.read_csv("final_data.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("log_reg_model.pkl")

df = load_data()
model = load_model()

# Title
st.title("🌍 Malnutrition Risk Dashboard")
st.markdown("Supports **SDG 2: Zero Hunger** by analyzing child malnutrition risk based on economic indicators.")

# Section 1: EDA Chart
st.header("📈 Trends by Country")

country = st.selectbox("Select a Country", sorted(df["Country"].unique()))
filtered = df[df["Country"] == country]

st.subheader(f"Stunting (%) and Food Price Index in {country}")
st.line_chart(filtered.set_index("Year")[["Stunting (%)", "Avg_Food_Price_Index"]])

# Section 2: Prediction
st.header("🧠 Predict Malnutrition Risk")

with st.form("prediction_form"):
    year = st.selectbox("Year", sorted(df["Year"].unique()))
    gdp = st.number_input("GDP per Capita (USD)", min_value=0.0, value=1000.0)
    food_price = st.number_input("Avg Food Price Index", min_value=0.0, value=100.0)
    submit = st.form_submit_button("Predict Risk")

if submit:
    input_df = pd.DataFrame([[year, gdp, food_price]], columns=["Year", "GDP_per_Capita", "Avg_Food_Price_Index"])
    pred = model.predict(input_df)[0]
    label = "High Risk" if pred == 1 else "Low Risk"
    st.success(f"🔎 Predicted Child Malnutrition Risk: **{label}**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Data from UNICEF, FAOSTAT, World Bank")
