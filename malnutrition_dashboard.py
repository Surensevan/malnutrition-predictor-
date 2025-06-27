import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model
@st.cache_resource
def load_model():
    return joblib.load("log_reg_model.pkl")

model = load_model()

# Load final dataset
@st.cache_data
def load_data():
    return pd.read_csv("final_data.csv")

data = load_data()

st.set_page_config(page_title="Malnutrition Risk Dashboard", layout="centered")
st.title("🧠 Predict Malnutrition Risk")
st.write("Select a country to assess malnutrition risk based on socioeconomic indicators.")

# Dropdown
country = st.selectbox("Select Country", sorted(data['Country'].unique()))

# Get selected country features
country_data = data[data['Country'] == country].iloc[0]
X_input = pd.DataFrame({
    'GDP_per_capita': [country_data['GDP_per_capita']],
    'Avg_Food_Price_Index': [country_data['Avg_Food_Price_Index']]
})

# Show features
col1, col2, col3 = st.columns(3)
col1.metric("🌍 Country", country)
col2.metric("💰 GDP per Capita (USD)", f"${country_data['GDP_per_capita']:,.2f}")
col3.metric("🍽️ Food Price Index", f"{country_data['Avg_Food_Price_Index']:.2f}")

# Predict button
if st.button("🔍 Predict Risk"):
    prob = model.predict_proba(X_input)[0][1]  # Probability for class 1 (High Risk)
    pred = model.predict(X_input)[0]
    
    st.subheader("🔎 Prediction Result")
    if pred == 1:
        st.error("❌ High Risk of Malnutrition")
    else:
        st.success("✅ Low Risk of Malnutrition")

    st.markdown(f"**Model Confidence for High Risk:** {prob:.1%}")

    # Plot gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Malnutrition Risk Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "gray"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "tomato"},
            ],
        }))
    st.plotly_chart(fig, use_container_width=True)

# Show top countries by risk probability
st.markdown("---")
st.subheader("🌍 Top Countries with Predicted High Malnutrition Risk")

# Predict for all countries
features = data[['GDP_per_capita', 'Avg_Food_Price_Index']]
data['Risk_Probability'] = model.predict_proba(features)[:, 1]

# Sort and show top 10
top_risk = data[['Country', 'GDP_per_capita', 'Avg_Food_Price_Index', 'Risk_Probability']] \
    .sort_values(by='Risk_Probability', ascending=False).head(10)

st.dataframe(top_risk.reset_index(drop=True), use_container_width=True)
