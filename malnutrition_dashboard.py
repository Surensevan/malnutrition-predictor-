import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("log_reg_model.pkl")

model = load_model()

# Load the dataset (with country info and indicators)
@st.cache_data
def load_data():
    raw = pd.read_csv("final_data.csv")
    clean = raw.groupby('Country', as_index=False).agg({
        'GDP_per_capita': 'mean',
        'Avg_Food_Price_Index': 'mean'
    }).dropna()
    return raw, clean

raw_data, data = load_data()

# Fit scaler based on available data
scaler = StandardScaler()
scaler.fit(data[['GDP_per_capita', 'Avg_Food_Price_Index']])

# Page layout
st.set_page_config(page_title="Malnutrition Risk Predictor", layout="wide")
st.title("🧠 Predict Malnutrition Risk")

# ============ Choropleth Map of Stunting Rates =============
st.subheader("🗺️ Global Stunting Rates (2022)")

# Prepare choropleth data
stunting_map_df = raw_data[raw_data['Year'] == 2022].dropna(subset=['Stunting (%)'])

choropleth_fig = px.choropleth(
    stunting_map_df,
    locations="ISO_Code",
    color="Stunting (%)",
    hover_name="Country",
    color_continuous_scale="Reds",
    projection="natural earth",
    title="Global Stunting Rates (2022)"
)
st.plotly_chart(choropleth_fig, use_container_width=True)

# Sidebar - Country selection
st.markdown("---")
st.markdown("Select a country to assess malnutrition risk based on socioeconomic indicators.")

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
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  # Probability of class 1 (High Risk)

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

# Prepare prediction input
predict_df = data[['Country', 'GDP_per_capita', 'Avg_Food_Price_Index']].copy()
X_pred = predict_df[['GDP_per_capita', 'Avg_Food_Price_Index']]
X_scaled = scaler.transform(X_pred)
predict_df['Risk_Probability'] = model.predict_proba(X_scaled)[:, 1]

# Show only countries with risk probability > 30%
high_risk_df = predict_df[predict_df['Risk_Probability'] > 0.3]

st.markdown("---")
st.subheader("🌍 Countries with Elevated Malnutrition Risk (>30%)")
st.dataframe(high_risk_df.sort_values(by='Risk_Probability', ascending=False).reset_index(drop=True))

# Add a bar chart for full risk probability distribution
st.markdown("---")
st.subheader("📊 Malnutrition Risk Probability by Country")
bar_fig = px.bar(
    predict_df.sort_values(by='Risk_Probability', ascending=False),
    x='Risk_Probability',
    y='Country',
    orientation='h',
    color='Risk_Probability',
    color_continuous_scale='reds',
    labels={'Risk_Probability': 'Risk Probability'},
    title="Malnutrition Risk Score by Country"
)
bar_fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(bar_fig, use_container_width=True)
