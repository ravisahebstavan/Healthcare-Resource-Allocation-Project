# 🏥 Healthcare Resource Allocation System

An AI-powered predictive analytics platform for optimizing hospital operations and resource management.

![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Model Performance](#model-performance)
* [Results](#results)
* [Future Enhancements](#future-enhancements)
* [Contact](#contact)

---

## 🎯 Overview

This healthcare analytics system integrates multiple machine learning models to address critical challenges in hospital resource management:

* **Predicting hospital readmissions** to enable early intervention
* **Forecasting healthcare costs** for improved budget planning
* **Optimizing patient flow** and bed allocation
* **Early warning system** for dengue outbreaks in Singapore

This project was developed as a **portfolio project** to demonstrate end-to-end data science and deployment skills for **NUS / NTU Data Science graduate program applications**.

---

## ✨ Features

### 1️⃣ Readmission Prediction

* Identifies patients at high risk of 30-day readmission
* Uses 52 engineered features from 100k+ patient records
* Risk stratification with intervention recommendations
* **Performance:** AUC = **0.6857**

### 2️⃣ Cost Prediction

* Estimates total healthcare costs from patient attributes
* Integrated risk-cost prioritization framework
* Identifies top 10% high-priority patients
* **Performance:** R² = **0.8982**

### 3️⃣ Patient Flow Forecasting

* Time-series forecasting of daily admissions
* Supports bed allocation optimization
* Captures weekly and seasonal patterns
* **Performance:** MAPE = **4.14%**

### 4️⃣ Dengue Outbreak Prediction

* Singapore-specific predictive model
* 2-week lag correlation with rainfall
* Early-warning alert framework
* **Performance:** MAPE = **17.34%**

### 5️⃣ Interactive Dashboard

* Built using **Streamlit**
* Five modules: Overview, Readmission, Cost, Flow, Dengue
* Interactive visualizations and metrics

---

## 🛠️ Tech Stack

### Languages & Libraries

* Python 3.10
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* Prophet (time series forecasting)
* SHAP (model interpretability)
* Matplotlib, Seaborn
* Streamlit

### Development Tools

* Jupyter Notebook
* Git
* Conda

---

## 📁 Project Structure

```text
healthcare_resource_allocation/
│
├── data/
│   ├── raw/                     # Original datasets
│   ├── processed/               # Cleaned and engineered data
│   └── external/                # External data sources
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_cost_prediction.ipynb
│   ├── 04_integrated_risk_cost_model.ipynb
│   ├── 05_patient_flow_forecasting.ipynb
│   └── 06_dengue_outbreak_prediction.ipynb
│
├── src/
│   ├── data_processing/
│   ├── models/
│   ├── visualization/
│   └── deployment/
│
├── models/                      # Trained model files
├── results/                     # Figures and outputs
├── docs/
│
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### Prerequisites

* Python 3.10+
* Conda or pip

### Setup

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-resource-allocation.git
cd healthcare-resource-allocation
```

#### 2️⃣ Create virtual environment

```bash
conda create -n healthcare_project python=3.10
conda activate healthcare_project
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Download data and models

Due to file size limits, trained models and processed data are hosted externally:

* **Models:** Place in `models/`
* **Processed data:** Place in `data/processed/`

---

## 💻 Usage

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The app will be available at:
👉 [http://localhost:8501](http://localhost:8501)

### Explore Notebooks

```bash
jupyter notebook
```

---

## 📊 Model Performance

| Model    | Task                   | Metric  | Score  | Status       |
| -------- | ---------------------- | ------- | ------ | ------------ |
| LightGBM | Readmission Prediction | AUC-ROC | 0.6857 | ✅ Production |
| LightGBM | Cost Prediction        | R²      | 0.8982 | ✅ Production |
| Prophet  | Patient Flow Forecast  | MAPE    | 4.14%  | ✅ Production |
| LightGBM | Dengue Forecast        | MAPE    | 17.34% | ✅ Production |

---

## 📈 Results

### Business Impact

* **High-priority patients identified:** 10,246
* **Intervention ROI:** 512%
* **Estimated annual savings:** $13.2M

### Visualizations

![Risk-Cost Matrix](results/risk_cost_matrix.png)
![Patient Flow Forecast](results/forecasting_comparison.png)
![Dengue Patterns](results/dengue_patterns.png)

---

## 🔮 Future Enhancements

* Real-time EMR data integration
* Cloud deployment (AWS / Azure)
* A/B testing framework for model updates
* Multi-hospital federated learning
* Mobile clinician dashboard
* NLP on clinical notes
* Real-time alerting (SMS / Email)

---

## 👨‍💻 Author

**Stavan Ravisaheb**

* LinkedIn: [https://www.linkedin.com/in/stavanravisaheb](https://www.linkedin.com/in/stavanravisaheb)
* Email: [ravisahebstavan@gmail.com](mailto:ravisahebstavan@gmail.com)

---

## 📝 License

This project is licensed under the **MIT License**.

---

## 📧 Contact

For collaboration, academic review, or graduate program inquiries, feel free to reach out via LinkedIn or email.
