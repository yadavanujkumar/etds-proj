import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load cost model components
with open("model2.pkl", "rb") as f:
    cost_model = pickle.load(f)
with open("scaler2.pkl", "rb") as f:
    cost_scaler = pickle.load(f)
with open("training_columns2.pkl", "rb") as f:
    cost_columns = pickle.load(f)

# App Interface
st.title("Maritime Operations Cost Analyzer")
st.write("Predict operational costs with AI-powered explanations")

# Input Section
col1, col2 = st.columns(2)
with col1:
    ship_type = st.selectbox("Ship Type", ["Container Ship", "Bulk Carrier", "Tanker", "Fish Carrier"])
    route_type = st.selectbox("Route Type", ["Long-haul", "Short-haul", "Transoceanic", "Coastal"])
    # engine_type = st.selectbox("Engine Type", ["HFO", "Diesel", "Steam Turbine"])
    # weather = st.selectbox("Weather Condition", ["Moderate", "Rough", "Calm"])

with col2:
    speed = st.number_input("Speed (knots)", 10.0, 30.0, 17.7)
    distance = st.number_input("Distance (nm)", 50.0, 5000.0, 1036.4)
    cargo = st.number_input("Cargo Weight (tons)", 100.0, 5000.0, 1032.6)
    # engine_power = st.number_input("Engine Power (kW)", 5000.0, 50000.0, 25000.0)

# # Additional inputs
# st.subheader("Additional Operational Parameters")
# col3, col4 = st.columns(2)
# with col3:
#     draft = st.number_input("Draft (meters)", 5.0, 25.0, 12.5)
#     efficiency = st.number_input("Efficiency (nm/kWh)", 0.1, 5.0, 2.5)
# with col4:
#     weekly_voyages = st.number_input("Weekly Voyages", 1, 10, 3)
#     maintenance = st.selectbox("Maintenance Status", ["Good", "Fair", "Critical"])
#     seasonal_score = st.number_input("Seasonal Impact Score", 0.0, 1.0, 0.5)

def prepare_features():
    """Create feature dataframe with full feature alignment"""
    base_features = {
        "Ship_Type": ship_type,
        "Route_Type": route_type,
        # "Engine_Type": engine_type,
        # "Maintenance_Status": maintenance,
        "Speed_Over_Ground_knots": speed,
        # "Engine_Power_kW": engine_power,
        "Distance_Traveled_nm": distance,
        # "Draft_meters": draft,
        # "Weather_Condition": weather,
        "Cargo_Weight_tons": cargo,
        # "Efficiency_nm_per_kWh": efficiency,
        # "Seasonal_Impact_Score": seasonal_score,
        # "Weekly_Voyage_Count": weekly_voyages,
        "Average_Load_Percentage": (cargo/5000)*100
    }
    
    # Create DataFrame with proper encoding
    df = pd.DataFrame([base_features])
    df = pd.get_dummies(df, columns=["Ship_Type", "Route_Type" ])
                                #    "Weather_Condition", "Maintenance_Status""Engine_Type"
    
    return df

# ... (keep all previous imports and data loading parts unchanged)

if st.button("Analyze Operational Costs"):
    try:
        input_df = prepare_features()
        
        # Cost Prediction
        processed = input_df.reindex(columns=cost_columns, fill_value=0)
        scaled = cost_scaler.transform(processed)
        prediction = cost_model.predict(scaled)[0]
        
        st.success(f"Predicted Operational Cost: ${prediction:,.2f}")
        
        # SHAP explanation with improved formatting
        st.subheader("Cost Drivers Analysis")
        
        explainer = shap.TreeExplainer(cost_model)
        shap_values = explainer.shap_values(processed)
        
        # Create a clean matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Customize the feature names for readability
        feature_names = [name.replace('_', ' ').title() 
                        for name in processed.columns]
        
        # Generate SHAP plot with explanations
        shap.summary_plot(shap_values, processed, 
                         feature_names=feature_names,
                         plot_type="bar", 
                         show=False,
                         max_display=10,
                         color='#1f77b4')  # Use consistent color
        
        # Customize plot appearance
        plt.title("Main Cost Influencers", fontweight='bold', pad=20)
        plt.xlabel("Impact on Operational Cost (USD)", fontsize=10)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#888888')
        plt.gca().spines['bottom'].set_color('#888888')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.clf()

        # Recommendations in expanders
        with st.expander("💡 Optimization Recommendations", expanded=True):
            if speed > 20:
                st.warning("**Speed Reduction**\n\n⚠️ Reducing speed by 1-2 knots could save 5-7% in fuel costs")
                
            if cargo < 2000:
                st.info("**Cargo Utilization**\n\nℹ️ Increasing load to 2500+ tons would improve cost efficiency by ~15%")
                
            st.markdown("---")
            st.markdown("#### Always consider:")
            st.markdown("- Weather routing optimization\n- Regular engine maintenance\n- Fuel type selection strategies")

    except Exception as e:
        st.error(f"Processing error: {str(e)}")

st.markdown("---")
st.caption("Maritime Cost Analytics Engine v2.1")