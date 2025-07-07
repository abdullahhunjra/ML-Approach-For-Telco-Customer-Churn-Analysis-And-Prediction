# 📊 Telco Customer Churn Prediction

This project analyzes customer churn behavior using machine learning models to identify patterns and help telecom companies retain customers more effectively. It includes data cleaning, exploratory data analysis (EDA), feature selection, model comparison, hyperparameter tuning, and explainability using SHAP.

---

## 🎯 Problem Statement

Churn — the loss of customers — is a major challenge in the telecom industry. The goal of this project is to build a predictive model that identifies customers who are likely to churn, based on their demographic information and usage patterns.

---

## 📂 Dataset Overview

- **Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows:** 7043
- **Features:** 21 (including categorical, numerical, and target column `Churn`)

Key features include:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `MonthlyCharges`, `TotalCharges`
- `PhoneService`, `InternetService`, `OnlineSecurity`, etc.
- `Contract`, `PaymentMethod`
- `Churn` (target variable)

---

## ⚙️ Project Workflow

1. **Data Cleaning & Preprocessing**
2. **EDA & Visualization**
3. **Feature Selection (using SelectKBest + SHAP)**
4. **Model Training (Logistic Regression, Random Forest, XGBoost, LGBM, Gradient Boosting)**
5. **Hyperparameter Tuning**
6. **Model Evaluation & Comparison**
7. **Explainability with SHAP**
8. **Exporting Trained Model**

---

## 🧪 Models Used

- Logistic Regression
- Random Forest
- Gradient Boosting
- LightGBM
- XGBoost

Each model was evaluated based on:
- **Recall** (priority for churn)
- **Accuracy**
- **F1 Score**
- **Confusion Matrix**

---

## 📈 Best Model & Evaluation

### ✅ Final Chosen Model: **Gradient Boosting (Default Parameters)**

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 75.9%     |
| Recall     | **85.9%** ✅ |
| F1-Score   | 77.6%     |

**Confusion Matrix:**

```
[[254 128]
 [ 51 311]]
```

This means:
- ✅ **311 churners were correctly identified**
- ⚠️ Only **51 churners were missed**

---

## 🔍 Model Explainability (SHAP)

SHAP analysis was used to explain the impact of each feature on model predictions:

- 📉 **Contract Type**: Month-to-month contracts strongly increase churn risk
- 💰 **High Monthly Charges** & 📊 **Low Tenure**: Push model toward predicting churn
- 🔐 Lack of **OnlineSecurity** and **TechSupport**: Strong indicators of churn
- 💵 **TotalCharges**: Low total spend → likely new customer → higher churn risk

---

## 🧠 Key Insights & Strategic Recommendations

### 🔹 Key Drivers of Churn
- Month-to-month contract
- High monthly charges with short tenure
- Missing tech support or online security

### 📌 Business Recommendations
- Incentivize long-term contracts
- Bundle security & support features
- Engage new users quickly
- Flag high-value but new customers

---

## 📁 Project Structure

```
├── data/                    # Raw and cleaned data
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── eda/                     # EDA plots and insights
│   └── churn_distribution.png
│   └── senior_citizen_plot.png
│
├── shap/                    # SHAP visualizations
│   └── shap_summary_plot.png
│
├── models/                  # Saved models (.pkl)
│   └── logistic_regression_default.pkl
│
├── Telco-Customer-Churn.ipynb
│
│
├── requirements.txt         # Python dependencies
├── README.md                # Project summary
```

---

## 📦 Setup Instructions

1. Clone the repository  
2. Install packages:  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter or run script

OR

1-Clone the repository
2-Install dependencies
   pip install -r requirements.txt
3-Run the FastAPI server using command: uvicorn main:app --reload
4-Visit Swagger Docs: http://127.0.0.1:8000/docs

---

## 👨‍💻 Author

**Abdullah Shahzad**  
[GitHub](https://github.com//abdullahhunjra) • [LinkedIn](https://www.linkedin.com/in/abdullahhunjra)

---

## 📌 License

This project is open-source and available under the [MIT License](LICENSE).
