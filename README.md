# Customer Churn Prediction & Retention Dashboard

An end-to-end machine learning application that predicts customer churn and provides actionable retention strategies.

Built by **Hariharan Gopikrishnan** — MS Business Intelligence & Analytics, Stevens Institute of Technology

## What It Does
- Trains and compares 3 ML models: Random Forest, XGBoost, Logistic Regression
- Achieves **89% accuracy** and **84% precision** on churn prediction
- Lets you predict churn risk for any individual customer in real time
- Surfaces high-risk segments with a recommended retention playbook
- Estimates annual revenue saved through targeted interventions

## Tech Stack
Python · Streamlit · Scikit-learn · Plotly · Pandas · NumPy

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Fork this repo
2. Go to share.streamlit.io
3. Connect your GitHub repo → select `app.py` → Deploy
