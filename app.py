import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load the pre-trained model, scaler, and training columns
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
training_columns = pickle.load(open("training_columns.pkl", "rb"))

# -----------------------------
# UI: Title and Description
# -----------------------------
st.title("Ship Operational Cost Predictor")
st.write("Enter the details of your voyage to predict the operational cost and get recommendations on how to increase revenue and decrease costs.")

# -----------------------------
# UI: Input Section
# -----------------------------
# Example options for dropdown menus (adjust these to match your dataset)
ship_types = ["Container Ship", "Bulk Carrier", "Fish Carrier", "Tanker"]
route_types = ["Long-haul", "Short-haul", "Transoceanic", "Coastal"]

selected_ship_type = st.selectbox("Select Ship Type", ship_types)
selected_route_type = st.selectbox("Select Route Type", route_types)

# Numeric inputs for continuous variables
speed = st.number_input("Speed Over Ground (knots)", min_value=10.0, max_value=25.0, value=17.7)
distance = st.number_input("Distance Traveled (nm)", min_value=50.0, max_value=2000.0, value=1036.4)
cargo_weight = st.number_input("Cargo Weight (tons)", min_value=50.0, max_value=2000.0, value=1032.6)

# -----------------------------
# Prediction and Recommendation Section
# -----------------------------
if st.button("Predict Operational Cost"):
    # Build a DataFrame from the user input
    input_data = pd.DataFrame({
        "Ship_Type": [selected_ship_type],
        "Route_Type": [selected_route_type],
        "Speed_Over_Ground_knots": [speed],
        "Distance_Traveled_nm": [distance],
        "Cargo_Weight_tons": [cargo_weight]
    })
    
    # One-hot encode categorical features exactly as in training
    input_encoded = pd.get_dummies(input_data, columns=["Ship_Type", "Route_Type"], drop_first=True)
    
    # Reindex to ensure we have the same columns as used during training, filling missing ones with zeros
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
    
    # Scale the features using the pre-fitted scaler
    input_scaled = scaler.transform(input_encoded)
    
    # Predict the operational cost
    predicted_cost = model.predict(input_scaled)
    st.success(f"Predicted Operational Cost (USD): {predicted_cost[0]:,.2f}")
    
    # Provide recommendations
    st.subheader("Recommendations to Increase Revenue & Decrease Operational Cost")
    st.write("• **Optimize Route Planning:** Consider shorter or more fuel-efficient routes to lower fuel costs.")
    st.write("• **Maximize Cargo Load Efficiency:** Increase average load percentage to improve revenue per voyage.")
    st.write("• **Enhance Maintenance Practices:** Regular and proactive maintenance can reduce unexpected downtime.")
    st.write("• **Invest in Fuel-Efficient Technologies:** Modern engines and retrofits can help cut operational costs.")

