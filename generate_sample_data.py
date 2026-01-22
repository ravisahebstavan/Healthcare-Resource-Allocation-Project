import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create directories
import os
os.makedirs('data/processed', exist_ok=True)

print("Generating sample data for Streamlit Cloud deployment...")

# 1. Integrated predictions (sample 1000 rows)
np.random.seed(42)
integrated_sample = pd.DataFrame({
    'readmit_risk': np.random.beta(2, 5, 1000),
    'predicted_cost': np.random.normal(40000, 15000, 1000),
    'actual_readmit': np.random.binomial(1, 0.11, 1000),
    'actual_cost': np.random.normal(40000, 15000, 1000),
    'expected_total_cost': np.random.normal(45000, 16000, 1000),
    'risk_category': np.random.choice(['Low Risk', 'Medium Risk', 'High Risk'], 1000, p=[0.6, 0.3, 0.1]),
    'cost_category': np.random.choice(['Low Cost', 'Medium Cost', 'High Cost'], 1000, p=[0.3, 0.5, 0.2]),
    'risk_score': np.random.uniform(10, 80, 1000),
    'cost_score': np.random.uniform(20, 90, 1000),
    'priority_score': np.random.uniform(30, 170, 1000)
})
integrated_sample.to_csv('data/processed/integrated_predictions.csv', index=False)
print("✓ integrated_predictions.csv")

# 2. 7-day forecast
dates = pd.date_range(datetime.now(), periods=7, freq='D')
forecast_sample = pd.DataFrame({
    'ds': dates,
    'yhat': np.random.normal(97, 8, 7),
    'yhat_lower': np.random.normal(85, 5, 7),
    'yhat_upper': np.random.normal(110, 5, 7)
})
forecast_sample.to_csv('data/processed/7day_forecast.csv', index=False)
print("✓ 7day_forecast.csv")

# 3. Bed allocation
allocation_sample = pd.DataFrame({
    'Department': ['General', 'ICU', 'Emergency', 'Surgery'],
    'Base_Allocation': [200, 75, 125, 100],
    'Optimized_Allocation': [210, 80, 120, 90],
    'Expected_Demand': [195.5, 72.3, 115.8, 88.4],
    'Utilization_Rate': [93.1, 90.4, 96.5, 98.2]
})
allocation_sample.to_csv('data/processed/bed_allocation_recommendations.csv', index=False)
print("✓ bed_allocation_recommendations.csv")

# 4. Dengue Singapore
dengue_dates = pd.date_range('2021-01-01', '2023-12-31', freq='W')
dengue_sample = pd.DataFrame({
    'date': dengue_dates,
    'week': [d.isocalendar()[1] for d in dengue_dates],
    'month': [d.month for d in dengue_dates],
    'year': [d.year for d in dengue_dates],
    'dengue_cases': np.random.poisson(100, len(dengue_dates)),
    'temperature': np.random.uniform(26, 32, len(dengue_dates)),
    'rainfall': np.random.uniform(100, 300, len(dengue_dates)),
    'humidity': np.random.uniform(75, 90, len(dengue_dates))
})
dengue_sample.to_csv('data/processed/dengue_singapore.csv', index=False)
print("✓ dengue_singapore.csv")

# 5. Dengue alerts (last 30 weeks)
alert_sample = dengue_sample.tail(30).copy()
alert_sample = alert_sample.rename(columns={'dengue_cases': 'actual_cases'})
alert_sample['predicted_cases'] = alert_sample['actual_cases'] + np.random.normal(0, 10, len(alert_sample))
alert_sample['actual_risk'] = alert_sample['actual_cases'].apply(
    lambda x: 'OUTBREAK' if x >= 150 else 'HIGH ALERT' if x >= 100 else 'NORMAL'
)
alert_sample['predicted_risk'] = alert_sample['predicted_cases'].apply(
    lambda x: 'OUTBREAK' if x >= 150 else 'HIGH ALERT' if x >= 100 else 'NORMAL'
)
alert_sample.to_csv('data/processed/dengue_alerts.csv', index=False)
print("✓ dengue_alerts.csv")

print("\n✅ All sample data files generated successfully!")
print("These are demo files for Streamlit Cloud deployment.")