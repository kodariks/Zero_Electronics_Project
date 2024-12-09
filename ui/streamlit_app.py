import streamlit as st
import requests
import pandas as pd
import altair as alt

# Set Streamlit page configuration
st.set_page_config(page_title="Zero Electronics Project", layout="wide")

# App title and description
st.title("Zero Electronics Project: Predictive Insights")
st.write("""
This application predicts the likelihood of a customer buying a phone using multiple models (Logistic Regression, Random Forest, Naive Bayes, and K-Means). 
It also provides actionable insights and model performance metrics to help make informed decisions.
""")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Income ($)", 20000, 120000, 50000)
visits = st.sidebar.slider("Visits", 1, 30, 10)
price = st.sidebar.slider("Price ($)", 1000, 20000, 10000)
battery_life = st.sidebar.slider("Battery Life (hours)", 5, 30, 15)
discount_sensitivity = st.sidebar.slider("Discount Sensitivity", 0.1, 1.0, 0.5)
gender_male = st.sidebar.selectbox("Gender (Male=1, Female=0)", [1, 0])
loyalty_score = st.sidebar.slider("Loyalty Score", 0.0, 2.0, 1.0)
promo_response = st.sidebar.selectbox("Promo Response (Yes=1, No=0)", [1, 0])
purchase_frequency = st.sidebar.slider("Purchase Frequency", 1, 10, 5)

# Prepare input data
input_data = {
    "Age": age,
    "Income": income,
    "Visits": visits,
    "Price": price,
    "Battery_Life": battery_life,
    "Discount_Sensitivity": discount_sensitivity,
    "Gender_Male": gender_male,
    "Loyalty_Score": loyalty_score,
    "Promo_Response": promo_response,
    "Purchase_Frequency": purchase_frequency
}

# Display input data
st.write("### Input Data Summary")
st.dataframe(pd.DataFrame([input_data]))

# API Endpoints
API_URL_PREDICTION = "http://127.0.0.1:5000/predict_purchase"
API_URL_SCORES = "http://127.0.0.1:5000/get_model_scores"

# Predictions Section
if st.button("Get Prediction"):
    response = requests.post(API_URL_PREDICTION, json=input_data)
    if response.status_code == 200:
        result = response.json()

        # Display predictions
        st.write("## Predictions")
        st.success(f"**Logistic Regression:** {'Likely to Buy' if result['logistic_prediction'] == 1 else 'Unlikely to Buy'}")
        st.info(f"**Random Forest:** {'Likely to Buy' if result['random_forest_prediction'] == 1 else 'Unlikely to Buy'}")
        st.warning(f"**Naive Bayes Sentiment Analysis:** {'Positive Sentiment' if result['naive_bayes_prediction'] == 1 else 'Negative Sentiment'}")
        st.write(f"**K-Means Cluster Assigned:** {result['kmeans_cluster']}")

        # Actionable Insights Section
        st.write("## Actionable Insights")
        insights = []
        if result['logistic_prediction'] == 1:
            insights.append("Logistic Regression indicates the customer is highly likely to buy the phone.")
        else:
            insights.append("Logistic Regression suggests the customer is unlikely to buy the phone.")

        if result['random_forest_prediction'] == 1:
            insights.append("Random Forest supports the likelihood of purchase.")
        else:
            insights.append("Random Forest predicts that the customer is less likely to buy the phone.")

        if result['kmeans_cluster'] in [1, 2]:  # Assuming clusters 1 and 2 are high-purchasing groups
            insights.append("K-Means clustering places the customer in a high-purchasing segment.")
        else:
            insights.append("K-Means clustering suggests the customer is in a low-purchasing segment.")

        for insight in insights:
            st.write(f"- {insight}")

        # Feature Importance Section
        st.write("## Feature Importance (Random Forest)")
        if result.get("random_forest_importance"):
            rf_importance = pd.DataFrame(result["random_forest_importance"])
            chart = alt.Chart(rf_importance).mark_bar().encode(
                x="Importance",
                y=alt.Y("Feature", sort="-x"),
                color=alt.Color("Feature", legend=None)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Feature importance data is not available.")
    else:
        st.error("Error fetching predictions. Please check the server.")

# Model Performance Metrics Section
st.write("### Model Performance Metrics")
try:
    response = requests.get(API_URL_SCORES)
    if response.status_code == 200:
        # Transform API response into a DataFrame
        metrics = response.json()
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index()

        # Dynamically rename columns based on available metrics
        column_names = ["Model"] + list(metrics_df.columns[1:])
        metrics_df.columns = column_names

        # Display as a DataFrame
        st.dataframe(metrics_df)

        # Visualization of model metrics
        st.write("### Model Scores Visualization")
        chart = alt.Chart(metrics_df.melt(id_vars=["Model"], var_name="Metric Type", value_name="Score")).mark_bar().encode(
            x="Metric Type:O",
            y="Score:Q",
            color="Metric Type:N",
            column="Model:N"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No model performance metrics available.")
except Exception as e:
    st.error(f"Error fetching model scores: {e}")

# Footer Section
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: #333;
        }
    </style>
    <div class="footer">
        Made with ❤️ by Zero Electronics aka Sai Koushik Kodari
    </div>
    """,
    unsafe_allow_html=True,
)

