import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("log_reg_model.pkl")

model = load_model()

# Load the dataset (with country info and indicators)
@st.cache_data
def load_data():
    return pd.read_csv("final_data.csv")

data = load_data()

# Page layout
st.set_page_config(page_title="Malnutrition Risk Predictor", layout="wide")
st.title("🧠 Predict Malnutrition Risk")
st.markdown("Select a country to assess malnutrition risk based on socioeconomic indicators.")

# Sidebar - Country selection
country_list = data['Country'].dropna().unique()
selected_country = st.selectbox("Select Country", sorted(country_list))

# Display country-specific indicators
country_data = data[data['Country'] == selected_country].iloc[0]
gdp = country_data['GDP_per_capita']
food_index = country_data['Avg_Food_Price_Index']

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric(label="🌐 Country", value=selected_country)
with col2:
    st.metric(label="💰 GDP per Capita (USD)", value=f"${gdp:,.2f}")
with col3:
    st.metric(label="🍽️ Food Price Index", value=f"{food_index:.2f}")

# Predict Button
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame({
        'GDP_per_capita': [gdp],
        'Avg_Food_Price_Index': [food_index]
    })
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # Probability of class 1 (High Risk)

    st.subheader("🔎 Prediction Result")
    if pred == 1:
        st.success("🚨 High Risk of Malnutrition")
    else:
        st.info("✅ Low Risk of Malnutrition")

    st.markdown(f"**Model Confidence for High Risk:** {prob*100:.1f}%")

    # Gauge Chart using Plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob*100,
        title = {'text': "Malnutrition Risk Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"},
            ]
        }))
    st.plotly_chart(fig, use_container_width=True)

# ============ Bulk Prediction and Table =============

# Clean the data by removing rows with missing predictors
predict_df = data[['Country', 'GDP_per_capita', 'Avg_Food_Price_Index']].dropna()

if not predict_df.empty:
    X_pred = predict_df[['GDP_per_capita', 'Avg_Food_Price_Index']]
    predict_df['Risk_Probability'] = model.predict_proba(X_pred)[:, 1]
    top_risk_countries = predict_df.sort_values(by='Risk_Probability', ascending=False).head(10)

    st.markdown("---")
    st.subheader("🌍 Top Countries with Predicted High Malnutrition Risk")
    st.dataframe(top_risk_countries.reset_index(drop=True))
else:
    st.warning("⚠️ Not enough valid data for global prediction table.")
