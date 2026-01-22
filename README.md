\# 🏥 Healthcare Resource Allocation System



An AI-powered predictive analytics platform for optimizing hospital operations and resource management.



!\[Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[License](https://img.shields.io/badge/License-MIT-green)



\## 📋 Table of Contents

\- \[Overview](#overview)

\- \[Features](#features)

\- \[Tech Stack](#tech-stack)

\- \[Project Structure](#project-structure)

\- \[Installation](#installation)

\- \[Usage](#usage)

\- \[Model Performance](#model-performance)

\- \[Results](#results)

\- \[Future Enhancements](#future-enhancements)

\- \[Contact](#contact)



\## 🎯 Overview



This comprehensive healthcare analytics system integrates multiple machine learning models to address critical challenges in hospital resource management:



\- \*\*Predicting hospital readmissions\*\* to enable early intervention

\- \*\*Forecasting healthcare costs\*\* for better budget planning

\- \*\*Optimizing patient flow\*\* and bed allocation

\- \*\*Early warning system\*\* for dengue outbreaks in Singapore



Built as a portfolio project to demonstrate end-to-end data science capabilities for graduate school applications (NUS/NTU Data Science programs).



\## ✨ Features



\### 1️⃣ Readmission Prediction

\- Identifies patients at high risk of 30-day readmission

\- Uses 52 engineered features from 100k+ patient records

\- Risk stratification and intervention recommendations

\- \*\*Performance: AUC = 0.6857\*\*



\### 2️⃣ Cost Prediction

\- Estimates total healthcare costs based on patient characteristics

\- Integrates with readmission risk for expected cost calculation

\- Identifies high-priority patients (top 10% by risk-cost score)

\- \*\*Performance: R² = 0.8982\*\*



\### 3️⃣ Patient Flow Forecasting

\- Predicts daily hospital admissions using time series analysis

\- Optimizes bed allocation across departments

\- Accounts for weekly and seasonal patterns

\- \*\*Performance: MAPE = 4.14%\*\*



\### 4️⃣ Dengue Outbreak Prediction

\- Singapore-specific model using weather data

\- 2-week lag correlation with rainfall patterns

\- Early warning alert system for public health

\- \*\*Performance: MAPE = 17.34%\*\*



\### 5️⃣ Interactive Dashboard

\- Built with Streamlit for real-time predictions

\- 5 modules: Overview, Readmission, Cost, Flow, Dengue

\- Professional UI with visualizations and metrics



\## 🛠️ Tech Stack



\*\*Languages \& Libraries:\*\*

\- Python 3.10

\- Pandas, NumPy for data manipulation

\- Scikit-learn for machine learning

\- XGBoost, LightGBM for gradient boosting

\- Prophet for time series forecasting

\- SHAP for model interpretability

\- Matplotlib, Seaborn for visualization

\- Streamlit for web deployment



\*\*Development Tools:\*\*

\- Jupyter Notebook for analysis

\- Git for version control

\- Conda for environment management



\## 📁 Project Structure

```

healthcare\_resource\_allocation/

├── data/

│   ├── raw/                    # Original datasets

│   ├── processed/              # Cleaned and engineered data

│   └── external/               # External data sources

├── notebooks/

│   ├── 01\_data\_exploration.ipynb

│   ├── 02\_model\_development.ipynb

│   ├── 03\_cost\_prediction.ipynb

│   ├── 04\_integrated\_risk\_cost\_model.ipynb

│   ├── 05\_patient\_flow\_forecasting.ipynb

│   └── 06\_dengue\_outbreak\_prediction.ipynb

├── src/

│   ├── data\_processing/

│   ├── models/

│   ├── visualization/

│   └── deployment/

├── models/                     # Trained model files

├── results/                    # Figures and outputs

├── docs/                       # Documentation

├── app.py                      # Streamlit dashboard

├── requirements.txt            # Python dependencies

├── .gitignore

└── README.md

```



\## 🚀 Installation



\### Prerequisites

\- Python 3.10+

\- Conda or pip



\### Setup



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/YOUR\_USERNAME/healthcare-resource-allocation.git

cd healthcare-resource-allocation

```



2\. \*\*Create virtual environment\*\*

```bash

conda create -n healthcare\_project python=3.10

conda activate healthcare\_project

```



3\. \*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



4\. \*\*Download data and models\*\*

Due to file size limitations, trained models and processed data are available separately:

\- \[Download Models](YOUR\_LINK\_HERE) (Place in `models/` directory)

\- \[Download Processed Data](YOUR\_LINK\_HERE) (Place in `data/processed/` directory)



\## 💻 Usage



\### Run the Dashboard

```bash

streamlit run app.py

```



The dashboard will open at `http://localhost:8501`



\### Explore Notebooks

Navigate to `notebooks/` and open in Jupyter:

```bash

jupyter notebook

```



\## 📊 Model Performance



| Model | Task | Metric | Score | Status |

|-------|------|--------|-------|--------|

| LightGBM | Readmission Prediction | AUC-ROC | 0.6857 | ✅ Production |

| LightGBM | Cost Prediction | R² | 0.8982 | ✅ Production |

| Prophet | Patient Flow | MAPE | 4.14% | ✅ Production |

| LightGBM | Dengue Forecast | MAPE | 17.34% | ✅ Production |



\### Key Insights



\*\*Readmission Risk Factors:\*\*

1\. Number of procedures

2\. Time in hospital

3\. Total interactions (labs + meds + procedures)



\*\*Cost Drivers:\*\*

1\. Number of procedures

2\. Hospital stay length

3\. Total medical interactions



\*\*Patient Flow Patterns:\*\*

\- 17% lower admissions on weekends

\- Winter surge (Dec-Feb): +15% cases

\- Reliable 7-day forecast with 4.14% error



\*\*Dengue Prediction:\*\*

\- Rainfall (2-week lag) strongest predictor (r=0.331)

\- Optimal mosquito breeding: 28-32°C

\- Alert system accuracy: ~75%



\## 📈 Results



\### Business Value

\- \*\*High-Priority Patients Identified:\*\* 10,246 (10% of population)

\- \*\*Potential Intervention ROI:\*\* 512% return on investment

\- \*\*Estimated Annual Savings:\*\* $13.2M through reduced readmissions



\### Visualizations

!\[Risk-Cost Matrix](results/risk\_cost\_matrix.png)

!\[Patient Flow Forecast](results/forecasting\_comparison.png)

!\[Dengue Patterns](results/dengue\_patterns.png)



\## 🔮 Future Enhancements



\- \[ ] Incorporate real-time EMR data integration

\- \[ ] Deploy to cloud (AWS/Azure) for scalability

\- \[ ] Add A/B testing framework for model updates

\- \[ ] Multi-hospital support with federated learning

\- \[ ] Mobile app for clinicians

\- \[ ] Advanced NLP for clinical notes analysis

\- \[ ] Real-time alerting system (SMS/Email)



\## 👨‍💻 Author



\*\*Stavan Ravisaheb\*\*

\- LinkedIn: www.linkedin.com/in/stavanravisaheb

\- Email: ravisahebstavan@gmail.com



\## 📝 License



This project is licensed under the MIT License - see the LICENSE file for details.



\## 🙏 Acknowledgments



\- Dataset: Diabetes 130-US hospitals (UCI ML Repository)

\- Singapore weather data simulation based on public patterns

\- Inspiration: Real-world healthcare resource optimization challenges



\## 📧 Contact



For questions, collaboration, or graduate program inquiries:

\- Email: ravisahebstavan@gmail.com

\- LinkedIn: www.linkedin.com/in/stavanravisaheb



---



\*\*Built for NUS/NTU Data Science Graduate Program Applications\*\*



\*Demonstrating end-to-end ML capabilities: from data exploration to production deployment\*

