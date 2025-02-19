import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load pre-trained model, scaler, and training columns
# -----------------------------
model = pickle.load(open("model2.pkl", "rb"))
scaler = pickle.load(open("scaler2.pkl", "rb"))
training_columns = pickle.load(open("training_columns2.pkl", "rb"))

st.title("Ship Operational Cost Predictor & Explanation")
st.write("Enter voyage details to predict the operational cost, receive dynamic recommendations, and view a model explanation.")

# -----------------------------
# Input Section
# -----------------------------
# Dropdown options (adjust according to your dataset)
ship_types = ["Container Ship", "Bulk Carrier", "Fish Carrier", "Tanker"]
route_types = ["Long-haul", "Short-haul", "Transoceanic", "Coastal"]

selected_ship_type = st.selectbox("Select Ship Type", ship_types)
selected_route_type = st.selectbox("Select Route Type", route_types)

# Numeric inputs for continuous variables (with typical range values from your summary)
speed = st.number_input("Speed Over Ground (knots)", min_value=10.0, max_value=25.0, value=17.7)
distance = st.number_input("Distance Traveled (nm)", min_value=50.0, max_value=2000.0, value=1036.4)
cargo_weight = st.number_input("Cargo Weight (tons)", min_value=50.0, max_value=2000.0, value=1032.6)

# -----------------------------
# Prediction and Dynamic Recommendation Section
# -----------------------------
if st.button("Predict Operational Cost"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        "Ship_Type": [selected_ship_type],
        "Route_Type": [selected_route_type],
        "Speed_Over_Ground_knots": [speed],
        "Distance_Traveled_nm": [distance],
        "Cargo_Weight_tons": [cargo_weight]
    })
    
    # One-hot encode categorical features exactly as during training
    input_encoded = pd.get_dummies(input_data, columns=["Ship_Type", "Route_Type"], drop_first=True)
    
    # Reindex to match training columns (fill missing columns with zeros)
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
    
    # Scale features using the pre-fitted scaler
    input_scaled = scaler.transform(input_encoded)
    
    # Predict operational cost
    predicted_cost = model.predict(input_scaled)[0]
    st.success(f"Predicted Operational Cost (USD): {predicted_cost:,.2f}")
    
    # -----------------------------
    # Dynamic Recommendations
    # -----------------------------
    st.subheader("Dynamic Recommendations")
    
    # These average values are based on the summary of your dataset (adjust as needed)
    avg_cargo_weight = 1032.57
    avg_distance = 1036.4
    avg_speed = 17.6
    avg_cost = 255143.34
    
    recommendations = []
    
    if cargo_weight < avg_cargo_weight:
        recommendations.append("Your cargo weight is below average. Increasing cargo load might help reduce cost per unit.")
    else:
        recommendations.append("Your cargo weight is optimal for cost efficiency.")
    
    if distance > avg_distance:
        recommendations.append("The distance traveled is above average. Consider optimizing your route to reduce fuel consumption.")
    else:
        recommendations.append("The distance traveled is within an efficient range.")
    
    if speed < avg_speed:
        recommendations.append("Speed is slightly below average. Increasing speed could reduce turnaround time, if fuel efficiency is maintained.")
    else:
        recommendations.append("Your vessel's speed is optimal.")
    
    if predicted_cost > avg_cost:
        recommendations.append("Predicted operational cost is higher than average. Review your operational practices and maintenance schedules to lower costs.")
    else:
        recommendations.append("Operational cost is within a favorable range.")
    
    for rec in recommendations:
        st.write("- " + rec)
    
    # -----------------------------
    # Model Explainability with SHAP
    # -----------------------------
    st.subheader("Model Explanation")
    
    # Use SHAP TreeExplainer (suitable for tree-based models like RandomForestRegressor)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    # Generate a waterfall plot for local explanation
    st.write("The following plot explains how each feature contributed to the predicted cost:")
    
    # Generate the waterfall plot using SHAP's internal function
    # Note: For regression, shap_values is a 2D array; we use the first (and only) instance's SHAP values.
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[0], feature_names=input_encoded.columns, max_display=10)
    st.pyplot(plt.gcf())
