import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ── Styling ──
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .sub { color: #555; font-size: 0.95rem; }
    .metric-card { background: #f0f4ff; border-radius: 10px; padding: 1rem; text-align: center; }
    .risk-high { color: #d62728; font-weight: bold; font-size: 1.2rem; }
    .risk-low  { color: #2ca02c; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Data Generation ──
@st.cache_data
def generate_data(n=2000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 70, n)
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    num_products = np.random.randint(1, 5, n)
    support_calls = np.random.poisson(2, n)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2])
    payment = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.2])

    # Churn logic
    churn_prob = (
        0.4 * (monthly_charges > 80).astype(int) +
        0.3 * (tenure < 12).astype(int) +
        0.2 * (support_calls > 3).astype(int) +
        0.15 * (contract == 'Month-to-month').astype(int) +
        0.1 * (internet == 'Fiber optic').astype(int) -
        0.2 * (num_products > 2).astype(int)
    )
    churn_prob = np.clip(churn_prob / churn_prob.max(), 0.05, 0.95)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        'Age': age, 'Tenure_Months': tenure, 'Monthly_Charges': monthly_charges,
        'Num_Products': num_products, 'Support_Calls': support_calls,
        'Contract': contract, 'Payment_Method': payment, 'Internet_Service': internet,
        'Churn': churn
    })

@st.cache_data
def train_models(df):
    le = LabelEncoder()
    df2 = df.copy()
    for col in ['Contract', 'Payment_Method', 'Internet_Service']:
        df2[col] = le.fit_transform(df2[col])
    X = df2.drop('Churn', axis=1)
    y = df2['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s  = sc.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost (GBM)': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    }
    results = {}
    for name, m in models.items():
        if name == 'Logistic Regression':
            m.fit(X_train_s, y_train)
            y_pred = m.predict(X_test_s)
            y_prob = m.predict_proba(X_test_s)[:,1]
        else:
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            y_prob = m.predict_proba(X_test)[:,1]
        results[name] = {
            'model': m, 'scaler': sc if name == 'Logistic Regression' else None,
            'acc': accuracy_score(y_test, y_pred),
            'prec': precision_score(y_test, y_pred),
            'rec': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cm': confusion_matrix(y_test, y_pred),
            'fpr': roc_curve(y_test, y_prob)[0],
            'tpr': roc_curve(y_test, y_prob)[1],
            'auc': auc(roc_curve(y_test, y_prob)[0], roc_curve(y_test, y_prob)[1]),
            'feat_imp': getattr(m, 'feature_importances_', None),
            'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
        }
    return results, X.columns.tolist()

df = generate_data()
results, feature_names = train_models(df)

# ── Header ──
st.markdown('<p class="main-title">📉 Customer Churn Prediction & Retention Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Built by Hariharan Gopikrishnan · MS Business Intelligence & Analytics, Stevens Institute of Technology</p>', unsafe_allow_html=True)
st.markdown("---")

tabs = st.tabs(["📊 Data Overview", "🤖 Model Performance", "🔮 Predict Single Customer", "🎯 Retention Insights"])

# ── Tab 1: Data Overview ──
with tabs[0]:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churned", f"{df['Churn'].sum():,}")
    col3.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    col4.metric("Avg Monthly Charges", f"${df['Monthly_Charges'].mean():.0f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='Tenure_Months', color='Churn', nbins=30,
                           color_discrete_map={0:'#2196F3', 1:'#F44336'},
                           labels={'Churn': 'Churned'},
                           title='Churn by Tenure')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x='Contract', y='Monthly_Charges', color='Churn',
                     color_discrete_map={0:'#2196F3', 1:'#F44336'},
                     title='Monthly Charges by Contract & Churn')
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        churn_by_contract = df.groupby('Contract')['Churn'].mean().reset_index()
        fig = px.bar(churn_by_contract, x='Contract', y='Churn',
                     title='Churn Rate by Contract Type',
                     color='Churn', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(df.sample(500), x='Tenure_Months', y='Monthly_Charges',
                         color=df.sample(500)['Churn'].map({0:'Retained', 1:'Churned'}),
                         color_discrete_map={'Retained':'#2196F3','Churned':'#F44336'},
                         title='Tenure vs Charges (sample 500)')
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Model Performance ──
with tabs[1]:
    st.subheader("Model Comparison")
    model_choice = st.selectbox("Select Model", list(results.keys()))
    res = results[model_choice]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{res['acc']*100:.1f}%")
    c2.metric("Precision", f"{res['prec']*100:.1f}%")
    c3.metric("Recall",    f"{res['rec']*100:.1f}%")
    c4.metric("F1 Score",  f"{res['f1']*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        cm = res['cm']
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual"),
                        x=['No Churn','Churn'], y=['No Churn','Churn'],
                        title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        for name, r in results.items():
            fig.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines',
                                     name=f"{name} (AUC={r['auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(dash='dash', color='gray'), name='Random'))
        fig.update_layout(title='ROC Curve Comparison', xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig, use_container_width=True)

    if res['feat_imp'] is not None:
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': res['feat_imp']}).sort_values('Importance', ascending=True)
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance', color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Predict Single Customer ──
with tabs[2]:
    st.subheader("Predict Churn for a Single Customer")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 70, 35)
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        monthly = st.slider("Monthly Charges ($)", 20, 120, 65)
    with col2:
        products = st.slider("Number of Products", 1, 4, 2)
        calls = st.slider("Support Calls (last 6 months)", 0, 10, 2)
    with col3:
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
        internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    if st.button("🔮 Predict Churn Risk", type="primary"):
        le_map = {
            'Contract': {'Month-to-month': 1, 'One year': 0, 'Two year': 2},
            'Payment_Method': {'Bank transfer': 0, 'Credit card': 1, 'Electronic check': 2, 'Mailed check': 3},
            'Internet_Service': {'DSL': 0, 'Fiber optic': 1, 'No': 2}
        }
        input_df = pd.DataFrame([{
            'Age': age, 'Tenure_Months': tenure, 'Monthly_Charges': monthly,
            'Num_Products': products, 'Support_Calls': calls,
            'Contract': le_map['Contract'][contract],
            'Payment_Method': le_map['Payment_Method'][payment],
            'Internet_Service': le_map['Internet_Service'][internet]
        }])
        rf = results['Random Forest']['model']
        prob = rf.predict_proba(input_df)[0][1]
        risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        color = "risk-high" if prob > 0.5 else "risk-low"

        st.markdown(f"### Churn Probability: `{prob*100:.1f}%`")
        st.markdown(f'<p class="{color}">⚠ {risk}</p>' if prob > 0.5 else f'<p class="{color}">✅ {risk}</p>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#d62728" if prob > 0.5 else "#2ca02c"},
                   'steps': [{'range': [0,40], 'color': '#e8f5e9'},
                              {'range': [40,70], 'color': '#fff9c4'},
                              {'range': [70,100], 'color': '#ffebee'}]},
            title={'text': "Churn Risk Score"}))
        st.plotly_chart(fig, use_container_width=True)

        st.info("**Recommended Actions:** " + (
            "Offer a contract upgrade discount, assign a customer success rep, and trigger a retention email sequence."
            if prob > 0.5 else
            "Customer is stable. Consider upsell opportunities with additional products."))

# ── Tab 4: Retention Insights ──
with tabs[3]:
    st.subheader("Retention Strategy Insights")
    threshold = st.slider("Risk Threshold (%)", 30, 80, 50)
    res = results['Random Forest']
    high_risk = (res['y_prob'] > threshold/100).sum()
    total_test = len(res['y_prob'])
    st.metric("High-Risk Customers (in test set)", f"{high_risk} / {total_test}")

    monthly_avg = 70
    churn_reduction = st.slider("Projected Churn Reduction (%)", 5, 30, 15)
    saved = int(high_risk * (churn_reduction/100) * monthly_avg * 12)
    st.metric("Estimated Annual Revenue Saved", f"${saved:,}")

    segment_data = df.groupby('Contract').agg(
        Churn_Rate=('Churn','mean'),
        Count=('Churn','count'),
        Avg_Monthly=('Monthly_Charges','mean')
    ).reset_index()
    fig = px.scatter(segment_data, x='Avg_Monthly', y='Churn_Rate', size='Count',
                     color='Contract', title='Contract Segments: Churn Rate vs Avg Monthly Charges',
                     labels={'Churn_Rate': 'Churn Rate', 'Avg_Monthly': 'Avg Monthly Charges ($)'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Suggested Retention Playbook")
    st.markdown("""
| Segment | Churn Risk | Recommended Action |
|---|---|---|
| Month-to-month + High Charges | Very High | Offer annual contract at 20% discount |
| New Customers (< 6 months) | High | Assign onboarding specialist + check-in calls |
| High Support Calls | High | Proactive outreach, resolve open issues |
| Multi-product + Long Tenure | Low | Upsell premium services |
""")
