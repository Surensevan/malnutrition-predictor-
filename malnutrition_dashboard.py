import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    return joblib.load("best_log_reg_model.pkl")

model = load_model()

# App title
st.title("🧠 Predict Malnutrition Risk")
st.markdown("Enter socioeconomic indicators to predict whether a region is at high risk of malnutrition.")

# Input fields
gdp_input = st.number_input("GDP per Capita (USD)", min_value=0.0, value=1000.0, step=100.0)
food_index_input = st.number_input("Avg Food Price Index", min_value=0.0, value=100.0, step=1.0)

# Predict button
if st.button("Predict Risk"):
    input_df = pd.DataFrame({
        'GDP_per_capita': [gdp_input],
        'Avg_Food_Price_Index': [food_index_input]
    })

    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # Probability of class 1 (high risk)

    # Show prediction
    if pred == 1:
        st.error("⚠️ Predicted: **High Risk of Malnutrition**")
    else:
        st.success("✅ Predicted: **Low Risk of Malnutrition**")

    # Show probability
    st.markdown(f"**📊 Probability of High Risk:** `{proba:.2%}`")

    # Risk Chart for GDP per Capita
    st.subheader("📉 Risk Indicator Chart: GDP per Capita")
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axvspan(0, 2000, color='red', alpha=0.3, label='Likely High Risk')
    ax.axvspan(2000, 10000, color='green', alpha=0.3, label='Likely Low Risk')
    ax.axvline(gdp_input, color='blue', linewidth=2, label='Your Input')
    ax.set_xlabel('GDP per Capita (USD)')
    ax.set_yticks([])
    ax.set_title('GDP Risk Zone')
    ax.legend()
    st.pyplot(fig)

    # Risk Chart for Food Price Index
    st.subheader("📉 Risk Indicator Chart: Avg Food Price Index")
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    ax2.axvspan(0, 115, color='green', alpha=0.3, label='Likely Low Risk')
    ax2.axvspan(115, 200, color='red', alpha=0.3, label='Likely High Risk')
    ax2.axvline(food_index_input, color='blue', linewidth=2, label='Your Input')
    ax2.set_xlabel('Food Price Index')
    ax2.set_yticks([])
    ax2.set_title('Food Price Risk Zone')
    ax2.legend()
    st.pyplot(fig2)
