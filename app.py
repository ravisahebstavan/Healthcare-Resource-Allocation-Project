# app.py - Healthcare Resource Allocation Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Healthcare Resource Allocation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Healthcare Resource Allocation System</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Predictive Analytics for Hospital Operations")
st.markdown("---")

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module:",
    ["📊 Dashboard Overview", 
     "🔄 Readmission Prediction", 
     "💰 Cost Prediction",
     "📈 Patient Flow Forecast",
     "🦟 Dengue Outbreak Alert"]
)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    try:
        readmit_model = joblib.load(MODELS_DIR / "readmission_model_final.pkl")
        cost_model = joblib.load(MODELS_DIR / "cost_prediction_model.pkl")
        flow_model = joblib.load(MODELS_DIR / "patient_flow_prophet_model.pkl")
        dengue_model = joblib.load(MODELS_DIR / "dengue_forecast_model.pkl")
        return readmit_model, cost_model, flow_model, dengue_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

readmit_model, cost_model, flow_model, dengue_model = load_models()

# -------------------------------
# Load CSV data
# -------------------------------
@st.cache_data
def load_csv(file_name):
    try:
        return pd.read_csv(DATA_DIR / file_name)
    except Exception as e:
        st.warning(f"File not found or error loading {file_name}: {e}")
        return pd.DataFrame()

integrated_df = load_csv("integrated_predictions.csv")
forecast_df = load_csv("7day_forecast.csv")
allocation_df = load_csv("bed_allocation_recommendations.csv")
dengue_df = load_csv("dengue_singapore.csv")
alert_df = load_csv("dengue_alerts.csv")

# ========================================
# PAGE 1: DASHBOARD OVERVIEW
# ========================================
if page == "📊 Dashboard Overview":
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Readmission Model", "AUC: 0.6857", delta="Target: 0.75", delta_color="inverse")
    with col2:
        st.metric("Cost Prediction", "R²: 0.8982", delta="Excellent", delta_color="normal")
    with col3:
        st.metric("Patient Flow", "MAPE: 4.14%", delta="Target: <15%", delta_color="normal")
    with col4:
        st.metric("Dengue Forecast", "MAPE: 17.34%", delta="Target: <15%", delta_color="inverse")
    
    st.markdown("---")
    
    st.subheader("📋 System Capabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔄 Readmission Prediction**
        - Predict 30-day readmission risk
        - Identify high-risk patients
        - Enable early intervention
        
        **💰 Cost Prediction**
        - Forecast healthcare costs
        - Budget planning support
        - Resource optimization
        """)
    with col2:
        st.markdown("""
        **📈 Patient Flow Forecasting**
        - Predict daily admissions
        - Optimize bed allocation
        - Capacity planning
        
        **🦟 Dengue Outbreak Alert**
        - Early warning system
        - Weather-based prediction
        - Singapore-specific model
        """)
    
    st.markdown("---")
    
    if not integrated_df.empty:
        st.subheader("📊 Population Health Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_risk = integrated_df['readmit_risk'].mean() * 100
            st.metric("Average Readmission Risk", f"{avg_risk:.1f}%")
        with col2:
            avg_cost = integrated_df['predicted_cost'].mean()
            st.metric("Average Predicted Cost", f"${avg_cost:,.0f}")
        with col3:
            high_priority = (integrated_df['priority_score'] >= integrated_df['priority_score'].quantile(0.90)).sum()
            st.metric("High-Priority Patients", f"{high_priority:,}")
    else:
        st.info("Integrated data not available. Run all analysis notebooks first.")

# ========================================
# PAGE 2: READMISSION PREDICTION
# ========================================
elif page == "🔄 Readmission Prediction":
    st.header("Hospital Readmission Risk Calculator")
    st.markdown("Enter patient information to predict 30-day readmission risk.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 5)
        num_procedures = st.slider("Number of Procedures", 0, 6, 2)
        num_medications = st.slider("Number of Medications", 1, 80, 15)
    with col2:
        num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 45)
        is_emergency = st.checkbox("Emergency Admission")
    
    if st.button("Calculate Readmission Risk", type="primary"):
        st.info("⚠️ Simplified demo: full model requires 52 features")
        risk_score = (
            (time_in_hospital / 14) * 0.2 +
            (num_procedures / 6) * 0.15 +
            (num_medications / 80) * 0.15 +
            (num_diagnoses / 16) * 0.2 +
            (num_lab_procedures / 132) * 0.15 +
            (0.15 if is_emergency else 0)
        ) * 100
        risk_score = min(max(risk_score, 5), 95)
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Readmission Risk", f"{risk_score:.1f}%")
        with col2:
            if risk_score < 25:
                risk_level, color = "LOW", "🟢"
            elif risk_score < 50:
                risk_level, color = "MEDIUM", "🟡"
            else:
                risk_level, color = "HIGH", "🔴"
            st.metric("Risk Level", f"{color} {risk_level}")
        with col3:
            estimated_cost = 40000 + (risk_score * 300)
            st.metric("Estimated Cost", f"${estimated_cost:,.0f}")
        
        st.markdown("---")
        st.subheader("💡 Recommendations")
        if risk_score >= 50:
            st.error("""
            **High Risk Patient - Immediate Action Required:**
            - Schedule follow-up within 7 days
            - Assign care coordinator
            - Review medication compliance
            - Consider home health services
            """)
        elif risk_score >= 25:
            st.warning("""
            **Medium Risk Patient - Monitor Closely:**
            - Schedule follow-up within 14 days
            - Phone check-in at 7 days
            - Ensure clear discharge instructions
            """)
        else:
            st.success("""
            **Low Risk Patient - Standard Care:**
            - Standard follow-up within 30 days
            - Provide discharge education
            """)

# ========================================
# PAGE 3: COST PREDICTION
# ========================================
elif page == "💰 Cost Prediction":
    st.header("Healthcare Cost Estimator")
    st.markdown("Estimate total healthcare costs based on patient characteristics.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Expected Hospital Days", 1, 30, 7)
        procedures = st.slider("Expected Procedures", 0, 10, 3)
        lab_tests = st.slider("Expected Lab Tests", 0, 150, 50)
    with col2:
        medications = st.slider("Expected Medications", 0, 100, 20)
        icu_days = st.slider("ICU Days", 0, 10, 0)
        surgery = st.checkbox("Surgery Required")
    
    if st.button("Estimate Cost", type="primary"):
        base_cost = 5000
        daily_cost = days * 2500
        procedure_cost = procedures * 3000
        lab_cost = lab_tests * 150
        med_cost = medications * 300
        icu_cost = icu_days * 8000
        surgery_cost = 25000 if surgery else 0
        total_cost = base_cost + daily_cost + procedure_cost + lab_cost + med_cost + icu_cost + surgery_cost
        
        st.markdown("---")
        st.subheader("💵 Cost Breakdown")
        col1, col2 = st.columns([2, 1])
        with col1:
            breakdown = {
                'Base Admission': base_cost,
                'Hospital Stay': daily_cost,
                'Procedures': procedure_cost,
                'Lab Tests': lab_cost,
                'Medications': med_cost,
                'ICU': icu_cost,
                'Surgery': surgery_cost
            }
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Set3(range(len(breakdown)))
            ax.pie(breakdown.values(), labels=breakdown.keys(), autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Cost Breakdown', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        with col2:
            st.metric("Total Estimated Cost", f"${total_cost:,.2f}")
            st.markdown("---")
            st.markdown("**Cost Components:**")
            for item, cost in breakdown.items():
                if cost > 0:
                    st.write(f"• {item}: ${cost:,.0f}")

# ========================================
# PAGE 4: PATIENT FLOW FORECAST
# ========================================
elif page == "📈 Patient Flow Forecast":
    st.header("Patient Flow & Bed Allocation")
    st.markdown("View predicted hospital admissions and optimize bed allocation.")
    st.markdown("---")
    
    if not forecast_df.empty and not allocation_df.empty:
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        st.subheader("📅 7-Day Admission Forecast")
        col1, col2 = st.columns([2,1])
        with col1:
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(forecast_df['ds'], forecast_df['yhat'], marker='o', linewidth=2, color='blue')
            ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.3, color='blue')
            ax.set_xlabel('Date')
            ax.set_ylabel('Predicted Admissions')
            ax.set_title('7-Day Admission Forecast', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        with col2:
            st.metric("Average Daily Admissions", f"{forecast_df['yhat'].mean():.0f}")
            st.metric("Peak Day", forecast_df.loc[forecast_df['yhat'].idxmax(), 'ds'].strftime('%A'))
            st.metric("Lowest Day", forecast_df.loc[forecast_df['yhat'].idxmin(), 'ds'].strftime('%A'))
        
        st.markdown("---")
        st.subheader("🛏️ Bed Allocation Recommendations")
        st.dataframe(
            allocation_df.style.format({
                'Expected_Demand': '{:.1f}',
                'Utilization_Rate': '{:.1f}%'
            }).background_gradient(subset=['Utilization_Rate'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        fig, ax = plt.subplots(figsize=(12,6))
        x = np.arange(len(allocation_df))
        width = 0.35
        ax.bar(x - width/2, allocation_df['Base_Allocation'], width, label='Base Allocation', color='lightblue')
        ax.bar(x + width/2, allocation_df['Optimized_Allocation'], width, label='Optimized Allocation', color='orange')
        ax.set_ylabel('Number of Beds')
        ax.set_title('Base vs Optimized Bed Allocation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(allocation_df['Department'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Forecast or allocation data not available.")

# ========================================
# PAGE 5: DENGUE OUTBREAK ALERT
# ========================================
elif page == "🦟 Dengue Outbreak Alert":
    st.header("Dengue Outbreak Early Warning System")
    st.markdown("Monitor dengue cases in Singapore with weather-based early alerts.")
    st.markdown("---")
    
    if not dengue_df.empty:
        dengue_df['date'] = pd.to_datetime(dengue_df['date'])
        latest_cases = dengue_df['dengue_cases'].iloc[-1]
        latest_date = dengue_df['date'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Weekly Cases", f"{latest_cases}")
        with col2:
            if latest_cases >= 150:
                status = "🔴 OUTBREAK"
            elif latest_cases >= 100:
                status = "🟡 HIGH ALERT"
            else:
                status = "🟢 NORMAL"
            st.metric("Current Status", status)
        with col3:
            avg_cases = dengue_df['dengue_cases'].tail(4).mean()
            st.metric("4-Week Average", f"{avg_cases:.0f}")
        
        st.markdown("---")
        st.subheader("📊 Dengue Cases Trend")
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(dengue_df['date'], dengue_df['dengue_cases'], linewidth=2, color='red', marker='o', markersize=3)
        ax.axhline(y=150, color='darkred', linestyle='--', linewidth=2, label='Outbreak Threshold')
        ax.axhline(y=100, color='orange', linestyle='--', linewidth=2, label='Alert Threshold')
        ax.fill_between(dengue_df['date'], 0, dengue_df['dengue_cases'], alpha=0.3, color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Dengue Cases')
        ax.set_title('Singapore Dengue Cases Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Dengue data not available.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Healthcare Resource Allocation System | Built with Streamlit | Data Science Portfolio Project</p>
</div>
""", unsafe_allow_html=True)
