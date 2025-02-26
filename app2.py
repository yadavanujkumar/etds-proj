# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import shap
# from sklearn.linear_model import LinearRegression

# # Load pre-trained model, scaler, and training columns
# model = pickle.load(open("model2.pkl", "rb"))
# scaler = pickle.load(open("scaler2.pkl", "rb"))
# training_columns = pickle.load(open("training_columns2.pkl", "rb"))

# st.title("Ship Operational Cost Predictor & Explanation")
# st.write("Enter voyage details to predict the operational cost, receive dynamic recommendations, and view a model explanation.")

# # Input Section
# ship_types = ["Container Ship", "Bulk Carrier", "Fish Carrier", "Tanker"]
# route_types = ["Long-haul", "Short-haul", "Transoceanic", "Coastal"]

# selected_ship_type = st.selectbox("Select Ship Type", ship_types)
# selected_route_type = st.selectbox("Select Route Type", route_types)

# # Numeric inputs for continuous variables
# speed = st.number_input("Speed Over Ground (knots)", min_value=10.0, max_value=25.0, value=17.7)
# distance = st.number_input("Distance Traveled (nm)", min_value=50.0, max_value=2000.0, value=1036.4)
# cargo_weight = st.number_input("Cargo Weight (tons)", min_value=50.0, max_value=2000.0, value=1032.6)

# # Prediction and Dynamic Recommendation Section
# if st.button("Predict Operational Cost"):
#     # Create a DataFrame from user input
#     input_data = pd.DataFrame({
#         "Ship_Type": [selected_ship_type],
#         "Route_Type": [selected_route_type],
#         "Speed_Over_Ground_knots": [speed],
#         "Distance_Traveled_nm": [distance],
#         "Cargo_Weight_tons": [cargo_weight]
#     })
    
#     # One-hot encode categorical features
#     input_encoded = pd.get_dummies(input_data, columns=["Ship_Type", "Route_Type"], drop_first=True)
    
#     # Reindex to match training columns (fill missing columns with zeros)
#     input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
    
#     # Scale features using the pre-fitted scaler
#     input_scaled = scaler.transform(input_encoded)
    
#     # Predict operational cost
#     predicted_cost = model.predict(input_scaled)[0]
#     st.success(f"Predicted Operational Cost (USD): {predicted_cost:,.2f}")
    
#     # Dynamic Recommendations
#     st.subheader("Dynamic Recommendations")
    
#     avg_cargo_weight = 1032.57
#     avg_distance = 1036.4
#     avg_speed = 17.6
#     avg_cost = 255143.34
    
#     recommendations = []
    
#     if cargo_weight < avg_cargo_weight:
#         recommendations.append("Your cargo weight is below average. Increasing cargo load might help reduce cost per unit.")
#     else:
#         recommendations.append("Your cargo weight is optimal for cost efficiency.")
    
#     if distance > avg_distance:
#         recommendations.append("The distance traveled is above average. Consider optimizing your route to reduce fuel consumption.")
#     else:
#         recommendations.append("The distance traveled is within an efficient range.")
    
#     if speed < avg_speed:
#         recommendations.append("Speed is slightly below average. Increasing speed could reduce turnaround time, if fuel efficiency is maintained.")
#     else:
#         recommendations.append("Your vessel's speed is optimal.")
    
#     if predicted_cost > avg_cost:
#         recommendations.append("Predicted operational cost is higher than average. Review your operational practices and maintenance schedules to lower costs.")
#     else:
#         recommendations.append("Operational cost is within a favorable range.")
    
#     for rec in recommendations:
#         st.write("- " + rec)
#     # Model Explainability
# # Model Explainability
# st.subheader("Model Explanation")

# if isinstance(model, LinearRegression):
#     feature_importance = model.coef_
#     importance_df = pd.DataFrame({
#         "Feature": input_encoded.columns,
#         "Importance": feature_importance
#     })
#     importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
#     st.write("The following table shows the features' importance (higher values indicate a greater impact on the predicted cost):")
#     st.dataframe(importance_df)
    
#     # Plot feature importance
#     plt.figure(figsize=(10, 6))
#     plt.barh(importance_df['Feature'], importance_df['Importance'])
#     plt.xlabel('Importance')
#     plt.title('Feature Importance - Linear Model')
#     st.pyplot(plt.gcf())

# elif hasattr(model, 'feature_importances_'):  # For tree-based models
#     st.write("For tree-based models, we use SHAP for a more detailed explanation.")
    
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(input_scaled)
    
#     # The following ensures that the shap_values are wrapped as an Explanation object
#     if isinstance(shap_values, list):
#         shap_values = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[0], data=input_scaled)
#     else:
#         shap_values = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=input_scaled)

#     st.write("The following plot explains how each feature contributed to the predicted cost:")
#     shap.plots.waterfall(shap_values)  # For a single prediction
#     st.pyplot(plt.gcf())
    
# else:
#     st.write("The model does not support explainability via SHAP or feature importance directly.")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.preprocessing import MinMaxScaler

# # Load all models, scalers, PCA, and training columns
# model2 = pickle.load(open("model2.pkl", "rb"))
# scaler2 = pickle.load(open("scaler2.pkl", "rb"))
# training_columns2 = pickle.load(open("training_columns2.pkl", "rb"))

# model5 = pickle.load(open("model5.pkl", "rb"))
# pca5 = pickle.load(open("pca5.pkl", "rb"))

# model6 = pickle.load(open("model6.pkl", "rb"))
# scaler6 = pickle.load(open("scaler6.pkl", "rb"))
# training_columns6 = pickle.load(open("training_columns6.pkl", "rb"))

# # App Title
# st.title("Ship Operational Cost , Revenue ,Turn Around Time Predictor & Explanation")
# st.write("Enter voyage details to predict the operational cost, turnaround time, and receive dynamic recommendations.")

# # Input Section
# ship_types = ["Container Ship", "Bulk Carrier", "Fish Carrier", "Tanker"]
# route_types = ["Long-haul", "Short-haul", "Transoceanic", "Coastal"]

# selected_ship_type = st.selectbox("Select Ship Type", ship_types)
# selected_route_type = st.selectbox("Select Route Type", route_types)

# # Numeric inputs for continuous variables
# speed = st.number_input("Speed Over Ground (knots)", min_value=10.0, max_value=25.0, value=17.7)
# distance = st.number_input("Distance Traveled (nm)", min_value=50.0, max_value=2000.0, value=1036.4)
# cargo_weight = st.number_input("Cargo Weight (tons)", min_value=50.0, max_value=2000.0, value=1032.6)

# # Input for additional models
# engine_type = st.selectbox("Select Engine Type", ["Type A", "Type B", "Type C"])
# weather_condition = st.selectbox("Select Weather Condition", ["Clear", "Storm", "Rain"])
# weekly_voyage_count = st.number_input("Weekly Voyage Count", min_value=1, max_value=10, value=3)

# # Model Selection
# model_option = st.selectbox("Select Model for Prediction", ["Operational Cost Prediction", "Revenue per Voyage Prediction", "Turnaround Time Prediction"])

# # Prediction and Dynamic Recommendation Section
# if st.button("Predict"):
#     # Data Preparation for Model 1 (Operational Cost Prediction)
#     input_data2 = pd.DataFrame({
#         "Ship_Type": [selected_ship_type],
#         "Route_Type": [selected_route_type],
#         "Speed_Over_Ground_knots": [speed],
#         "Distance_Traveled_nm": [distance],
#         "Cargo_Weight_tons": [cargo_weight]
#     })
    
#     input_encoded2 = pd.get_dummies(input_data2, columns=["Ship_Type", "Route_Type"], drop_first=True)
#     input_encoded2 = input_encoded2.reindex(columns=training_columns2, fill_value=0)
#     input_scaled2 = scaler2.transform(input_encoded2)
    
#     if model_option == "Operational Cost Prediction":
#         # Predict Operational Cost
#         predicted_cost = model2.predict(input_scaled2)[0]
#         st.success(f"Predicted Operational Cost (USD): {predicted_cost:,.2f}")
        
#         # Dynamic Recommendations
#         avg_cargo_weight = 1032.57
#         avg_distance = 1036.4
#         avg_speed = 17.6
#         avg_cost = 255143.34
#         recommendations = []

#         if cargo_weight < avg_cargo_weight:
#             recommendations.append("Your cargo weight is below average. Increasing cargo load might help reduce cost per unit.")
#         else:
#             recommendations.append("Your cargo weight is optimal for cost efficiency.")
        
#         if distance > avg_distance:
#             recommendations.append("The distance traveled is above average. Consider optimizing your route to reduce fuel consumption.")
#         else:
#             recommendations.append("The distance traveled is within an efficient range.")
        
#         if speed < avg_speed:
#             recommendations.append("Speed is slightly below average. Increasing speed could reduce turnaround time, if fuel efficiency is maintained.")
#         else:
#             recommendations.append("Your vessel's speed is optimal.")
        
#         if predicted_cost > avg_cost:
#             recommendations.append("Predicted operational cost is higher than average. Review your operational practices and maintenance schedules to lower costs.")
#         else:
#             recommendations.append("Operational cost is within a favorable range.")
        
#         for rec in recommendations:
#             st.write("- " + rec)

#     # Model Explainability (For Linear Regression)
#     if isinstance(model2, LinearRegression):
#         feature_importance = model2.coef_
#         importance_df = pd.DataFrame({
#             "Feature": input_encoded2.columns,
#             "Importance": feature_importance
#         })
#         importance_df = importance_df.sort_values(by="Importance", ascending=False)
        
#         st.write("The following table shows the features' importance (higher values indicate a greater impact on the predicted cost):")
#         st.dataframe(importance_df)
        
#         # Plot feature importance
#         plt.figure(figsize=(10, 6))
#         plt.barh(importance_df['Feature'], importance_df['Importance'])
#         plt.xlabel('Importance')
#         plt.title('Feature Importance - Linear Model')
#         st.pyplot(plt.gcf())
    
#     elif hasattr(model2, 'feature_importances_'):  # For tree-based models
#         st.write("For tree-based models, we use SHAP for a more detailed explanation.")
        
#         explainer = shap.TreeExplainer(model2)
#         shap_values = explainer(input_scaled2)
        
#         # Ensuring correct reference to the input scaled for SHAP
#         if isinstance(shap_values, list):
#             shap_values = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[0], data=input_scaled2)
#         else:
#             shap_values = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=input_scaled2)

#         st.write("The following plot explains how each feature contributed to the predicted cost:")
#         shap.plots.waterfall(shap_values[0])  # Plot the first prediction
#         shap.summary_plot(shap_values, input_encoded2, feature_names=training_columns2)
#         st.pyplot(plt.gcf())

# Load all models and preprocessing objects
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import MinMaxScaler

# Load all models and components
with open("model2.pkl", "rb") as f:
    cost_model = pickle.load(f)
with open("scaler2.pkl", "rb") as f:
    cost_scaler = pickle.load(f)
with open("training_columns2.pkl", "rb") as f:
    cost_columns = pickle.load(f)

with open("model5.pkl", "rb") as f:
    revenue_model = pickle.load(f)
with open("pca5.pkl", "rb") as f:
    revenue_pca = pickle.load(f)
with open("training_columns5.pkl", "rb") as f:  # New file needed for PCA alignment
    revenue_columns = pickle.load(f)

with open("model6.pkl", "rb") as f:
    turnaround_model = pickle.load(f)
with open("scaler6.pkl", "rb") as f:
    turnaround_scaler = pickle.load(f)
with open("training_columns6.pkl", "rb") as f:
    turnaround_columns = pickle.load(f)

# App Interface
st.title("Maritime Operations Analyzer")
st.write("Predict key operational metrics with AI-powered explanations")

# Input Section
col1, col2 = st.columns(2)
with col1:
    ship_type = st.selectbox("Ship Type", ["Container Ship", "Bulk Carrier", "Tanker", "Fish Carrier"])
    route_type = st.selectbox("Route Type", ["Long-haul", "Short-haul", "Transoceanic", "Coastal"])
    engine_type = st.selectbox("Engine Type", ["HFO", "Diesel", "Steam Turbine"])
    weather = st.selectbox("Weather Condition", ["Moderate", "Rough", "Calm"])

with col2:
    speed = st.number_input("Speed (knots)", 10.0, 30.0, 17.7)
    distance = st.number_input("Distance (nm)", 50.0, 5000.0, 1036.4)
    cargo = st.number_input("Cargo Weight (tons)", 100.0, 5000.0, 1032.6)
    engine_power = st.number_input("Engine Power (kW)", 5000.0, 50000.0, 25000.0)

# Additional inputs
st.subheader("Additional Operational Parameters")
col3, col4 = st.columns(2)
with col3:
    draft = st.number_input("Draft (meters)", 5.0, 25.0, 12.5)
    efficiency = st.number_input("Efficiency (nm/kWh)", 0.1, 5.0, 2.5)
with col4:
    weekly_voyages = st.number_input("Weekly Voyages", 1, 10, 3)
    maintenance = st.selectbox("Maintenance Status", ["Good", "Fair", "Critical"])
    seasonal_score = st.number_input("Seasonal Impact Score", 0.0, 1.0, 0.5)

model_choice = st.selectbox("Prediction Type", ["Operational Cost", "Revenue", "Turnaround Time"])

def prepare_features():
    """Create feature dataframe with full feature alignment"""
    base_features = {
        "Ship_Type": ship_type,
        "Route_Type": route_type,
        "Engine_Type": engine_type,
        "Maintenance_Status": maintenance,
        "Speed_Over_Ground_knots": speed,
        "Engine_Power_kW": engine_power,
        "Distance_Traveled_nm": distance,
        "Draft_meters": draft,
        "Weather_Condition": weather,
        "Cargo_Weight_tons": cargo,
        "Efficiency_nm_per_kWh": efficiency,
        "Seasonal_Impact_Score": seasonal_score,
        "Weekly_Voyage_Count": weekly_voyages,
        "Turnaround_Time_hours": 0,
        "Average_Load_Percentage": (cargo/5000)*100
    }
    
    # Create DataFrame with proper encoding
    df = pd.DataFrame([base_features])
    df = pd.get_dummies(df, columns=["Ship_Type", "Route_Type", "Engine_Type", 
                                   "Weather_Condition", "Maintenance_Status"])
    
    return df

if st.button("Analyze"):
    try:
        input_df = prepare_features()
        
        if model_choice == "Operational Cost":
            # Cost Prediction
            processed = input_df.reindex(columns=cost_columns, fill_value=0)
            scaled = cost_scaler.transform(processed)
            prediction = cost_model.predict(scaled)[0]
            
            st.success(f"Predicted Operational Cost: ${prediction:,.2f}")
            
            # SHAP explanation
            explainer = shap.TreeExplainer(cost_model)
            shap_values = explainer.shap_values(scaled)
            plt.figure(figsize=(10,5))
            shap.summary_plot(shap_values, processed, feature_names=cost_columns, plot_type="bar")
            st.pyplot(plt.gcf())

        elif model_choice == "Revenue":
            # Revenue Prediction
            processed = input_df.reindex(columns=revenue_columns, fill_value=0)
            scaled = MinMaxScaler().fit_transform(processed)
            pca_data = revenue_pca.transform(scaled)
            prediction = revenue_model.predict(pca_data)[0]
            
            st.success(f"Predicted Revenue: ${prediction:,.2f}")
            
            # PCA visualization
            plt.figure(figsize=(10,5))
            plt.bar(range(revenue_pca.n_components_), revenue_pca.explained_variance_ratio_)
            plt.title("Key Revenue Drivers (PCA Components)")
            st.pyplot(plt.gcf())

        elif model_choice == "Turnaround Time":
            # Turnaround Prediction
            processed = input_df.reindex(columns=turnaround_columns, fill_value=0)
            scaled = turnaround_scaler.transform(processed)
            prediction = turnaround_model.predict(scaled)[0]
            
            st.success(f"Predicted Turnaround Time: {prediction:.1f} hours")
            
            # Feature importance
            plt.figure(figsize=(10,5))
            plt.barh(processed.columns, turnaround_model.coef_)
            plt.title("Turnaround Time Influencers")
            st.pyplot(plt.gcf())

        # Recommendations
        st.subheader("Optimization Recommendations")
        if speed > 20:
            st.warning("⚠️ Reduce speed by 1-2 knots for better fuel efficiency")
        if cargo < 2000:
            st.info("ℹ️ Increase cargo load for better cost efficiency")
        if efficiency < 2.0:
            st.info("🔧 Consider engine maintenance or route optimization")

    except Exception as e:
        st.error(f"Processing error: {str(e)}")

st.markdown("---")
st.caption("Maritime Analytics Engine v2.1 | Feature Alignment Fix")