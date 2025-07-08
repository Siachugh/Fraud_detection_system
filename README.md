
A machine learning-powered fraud detection system built using Random Forest Classifier and Streamlit, designed for real-time transaction analysis, risk classification, and interpretability of predictions.

---

## 🚀 Project Overview

This system is developed to support banks, e-commerce platforms, payment gateways, and other financial entities in detecting potentially fraudulent transactions.

It helps reduce manual fraud investigations by providing:
- Risk-based classification of each transaction
- Real-time decision-making assistance
- Clear, interpretable feature breakdowns

---

## 🔍 How It Works

The model analyzes transactional data and classifies each entry into one of four risk categories based on its fraud probability score:

- 🟢 Low – Safe transaction, low fraud risk  
- 🟡 Medium– Needs review, moderate risk  
- 🔴 High– Likely fraud, should be investigated  
- ⚫ Critical– Highly suspicious, immediate attention required

These risk levels help investigators prioritize their workflow and identify true threats faster.

---

## 🧠 Key Features

✅ Random Forest Classifier  
- Trained on real-world anonymized financial transaction data  
- Balanced for fraud/non-fraud classes to handle data imbalance

✅ Explainable Features
- Replaces PCA-based V1–V28 with intuitive names like:
  - Device Risk Score
  - Login Frequency
  - Unusual Time of Transaction
  - Anomaly Score

✅ Dual Input Modes 
- 📁 CSV File Upload: Analyze large sets of transactions in bulk  
- ✍️ Manual Entry: Enter a single transaction for quick analysis  

✅ Fraud Scoring & Risk Classification
- Predicts a fraud probability score for each transaction  
- Categorizes results into Low, Medium, High, and Critical risk

✅ Visual Dashboards 
- 📊 Fraud probability distribution  
- 🥧 Risk level pie chart  
- 🔍 Top feature importance  
- 🕐 Time-of-day fraud pattern analysis  

✅ Downloadable Filtered Results
- Download only selected transactions post-analysis (e.g., high risk only)

✅ Interactive UI Built with Streamlit
- Clean, professional interface suitable for analysts and operations teams  
- Easy to navigate with dynamic charts and real-time updates

## 📁 Directory Structure
Fraud_Detection_System/
├── models/
│ └── random_forest_model.pkl # Pre-trained Random Forest model
├── app/
│ └── streamlit_app.py # Streamlit app frontend
| └── app.py
├── src/
│ └── train.py
│ └── predict.py
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)


## Acknowledgments
Kaggle Credit Card Fraud Dataset
Streamlit, Scikit-learn, Plotly, and Seaborn


