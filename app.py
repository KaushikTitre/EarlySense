"""
Premium Customer Churn Prediction Dashboard
- LSTM Performance Metrics & Visualizations
- Behavioral Drift Analysis
- Explainable AI (SHAP)
- New Customer Prediction
- Churn Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
#  PREMIUM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0d1526 50%, #111b2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1724 0%, #1a2332 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }

    .metric-card h4 {
        color: #94a3b8 !important;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        -webkit-text-fill-color: #94a3b8;
        background: none;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Churn risk colors */
    .risk-critical { color: #ef4444 !important; -webkit-text-fill-color: #ef4444 !important; }
    .risk-high { color: #f97316 !important; -webkit-text-fill-color: #f97316 !important; }
    .risk-medium { color: #eab308 !important; -webkit-text-fill-color: #eab308 !important; }
    .risk-low { color: #22c55e !important; -webkit-text-fill-color: #22c55e !important; }

    /* Glass Panel */
    .glass-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
    }

    /* Drift severity badges */
    .drift-low {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .drift-medium {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .drift-high {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Offer Box */
    .offer-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        margin-top: 15px;
    }

    /* Architecture diagram */
    .arch-layer {
        background: rgba(99, 102, 241, 0.06);
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-family: 'Inter', monospace;
        font-size: 14px;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 24px 0;
    }

    /* Sidebar branding */
    .sidebar-brand {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(99, 102, 241, 0.15);
    }

    .sidebar-brand h2 {
        font-size: 22px;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
#  LOAD ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model('lstm_churn_model.h5')
    return model

@st.cache_data
def load_artifacts():
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    background = pd.read_csv('shap_background.csv')

    with open('training_results.json', 'r') as f:
        results = json.load(f)

    return scaler, feature_names, background, results

@st.cache_data
def load_dataset():
    if os.path.exists('churn_noisy_named.xlsx'):
        return pd.read_excel('churn_noisy_named.xlsx')
    elif os.path.exists('churn_noisy.xlsx'):
        df = pd.read_excel('churn_noisy.xlsx', header=None)
        df.columns = [
            'AccountAge', 'DataUsage', 'Age', 'MonthlySpend',
            'CallDuration', 'CallFrequency', 'SMSFrequency', 'DataFrequency',
            'PlanDuration', 'OverageFee', 'RoamingCalls',
            'CustomerSatisfaction', 'ComplaintRate', 'Churn',
            'AgeGroup', 'PlanType', 'Status'
        ]
        return df
    return pd.DataFrame()

# Load everything
try:
    model = load_model()
    scaler, feature_names, background, results = load_artifacts()
    dataset = load_dataset()
    model_loaded = True
except Exception as e:
    st.error(f"Model artifacts not found. Run `python train.py` first.\n\nError: {e}")
    model_loaded = False
    results = {}
    dataset = pd.DataFrame()


# ============================================================================
#  SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("""
<div class="sidebar-brand">
    <h2>📡 Churn AI</h2>
    <p style="color: #64748b; font-size: 12px; margin-top: 4px;">Behavioral Drift + XAI</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "🔮 New Prediction",
    "🔍 Customer Search",
    "📊 LSTM Performance",
    "🌊 Behavioral Drift",
    "🧠 Explainable AI",
    "📋 Churn Report",
    "⚙️ Model Overview"
])

st.sidebar.markdown("---")
if model_loaded and 'metrics' in results:
    m = results['metrics']
    st.sidebar.markdown(f"""
    <div class="glass-panel" style="padding: 16px;">
        <h4 style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">
            Quick Stats
        </h4>
        <p style="margin: 4px 0; font-size: 13px;">
            <span style="color: #64748b;">Acc:</span>
            <span style="color: #a78bfa; font-weight: 600;">{m['accuracy']*100:.1f}%</span>
        </p>
        <p style="margin: 4px 0; font-size: 13px;">
            <span style="color: #64748b;">F1:</span>
            <span style="color: #a78bfa; font-weight: 600;">{m['f1_score']*100:.1f}%</span>
        </p>
        <p style="margin: 4px 0; font-size: 13px;">
            <span style="color: #64748b;">AUC:</span>
            <span style="color: #a78bfa; font-weight: 600;">{m['roc_auc']*100:.1f}%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
#  HELPER: PREMIUM PLOT STYLE
# ============================================================================
def style_plot(fig, ax, title=""):
    fig.patch.set_facecolor('#0a0e17')
    ax.set_facecolor('#0f1724')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    if title:
        ax.set_title(title, color='#e2e8f0', fontsize=14, fontweight='bold', pad=12)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(True, alpha=0.1, color='#6366f1')
    return fig, ax


# ============================================================================
#  PAGE: DASHBOARD (HOME)
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown("# 📡 Customer Churn Prediction System")
    st.markdown("##### BiLSTM + Attention | Behavioral Drift Analysis | Explainable AI")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if model_loaded and 'metrics' in results:
        m = results['metrics']

        # Metric cards row
        cols = st.columns(5)
        metrics_display = [
            ("Accuracy", f"{m['accuracy']*100:.1f}%"),
            ("Precision", f"{m['precision']*100:.1f}%"),
            ("Recall", f"{m['recall']*100:.1f}%"),
            ("F1-Score", f"{m['f1_score']*100:.1f}%"),
            ("ROC-AUC", f"{m['roc_auc']*100:.1f}%"),
        ]
        for col, (name, val) in zip(cols, metrics_display):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <div class="metric-value">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Charts row
        col1, col2 = st.columns(2)

        with col1:
            # Training history
            hist = m.get('training_history', {})
            if hist:
                fig, ax = plt.subplots(figsize=(7, 4))
                epochs = range(1, len(hist['train_accuracy']) + 1)
                ax.plot(epochs, [v*100 for v in hist['train_accuracy']],
                        color='#6366f1', linewidth=2, label='Train Accuracy')
                ax.plot(epochs, [v*100 for v in hist['val_accuracy']],
                        color='#a78bfa', linewidth=2, linestyle='--', label='Val Accuracy')
                ax.fill_between(epochs, [v*100 for v in hist['train_accuracy']],
                                alpha=0.1, color='#6366f1')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.legend(facecolor='#0f1724', edgecolor='#1e293b', labelcolor='#94a3b8')
                style_plot(fig, ax, "Training History")
                st.pyplot(fig)
                plt.close()

        with col2:
            # ROC Curve
            roc = m.get('roc_curve', {})
            if roc:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(roc['fpr'], roc['tpr'], color='#6366f1', linewidth=2.5,
                        label=f"AUC = {m['roc_auc']*100:.1f}%")
                ax.plot([0, 1], [0, 1], color='#374151', linestyle='--', linewidth=1)
                ax.fill_between(roc['fpr'], roc['tpr'], alpha=0.1, color='#6366f1')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(facecolor='#0f1724', edgecolor='#1e293b', labelcolor='#94a3b8')
                style_plot(fig, ax, "ROC Curve")
                st.pyplot(fig)
                plt.close()

        # Dataset summary
        if not dataset.empty:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                churn_pct = dataset['Churn'].mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Dataset Size</h4>
                    <div class="metric-value">{len(dataset):,}</div>
                    <p style="color: #64748b; margin-top: 8px;">customers analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Churn Rate</h4>
                    <div class="metric-value risk-{'high' if churn_pct > 25 else 'medium' if churn_pct > 15 else 'low'}">{churn_pct:.1f}%</div>
                    <p style="color: #64748b; margin-top: 8px;">overall churn</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Features</h4>
                    <div class="metric-value">{len(feature_names)}</div>
                    <p style="color: #64748b; margin-top: 8px;">engineered features</p>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
#  PAGE: NEW PREDICTION
# ============================================================================
elif page == "🔮 New Prediction":
    st.markdown("# 🔮 New Customer Prediction")
    st.markdown("Enter customer details to predict churn risk with AI recommendations.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if model_loaded:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("#### Account Info")
            account_age = st.slider("Account Age (normalized)", 0.0, 1.0, 0.3, 0.01)
            age = st.number_input("Customer Age", 18, 70, 35)
            age_group = st.selectbox("Age Group", ['Young(<25)', 'Adult(25-35)', 'Middle(36-45)', 'Senior(46-55)', 'Elderly(>55)'])
            plan_type = st.selectbox("Plan Type", ['Pay-as-you-go', 'Contractual'])
            status = st.selectbox("Status", ['Active', 'Inactive'])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("#### Usage Metrics")
            data_usage = st.slider("Data Usage", 0.0, 1.0, 0.1, 0.01)
            call_duration = st.slider("Call Duration", 0.0, 1.0, 0.3, 0.01)
            call_frequency = st.slider("Call Frequency", 0.0, 1.0, 0.35, 0.01)
            sms_frequency = st.slider("SMS Frequency", 0.0, 1.0, 0.2, 0.01)
            data_frequency = st.slider("Data Frequency", 0.0, 1.0, 0.3, 0.01)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("#### Financial & Satisfaction")
            monthly_spend = st.number_input("Monthly Spend", 0.0, 10.0, 1.0, 0.1)
            plan_duration = st.number_input("Plan Duration", 0.5, 5.0, 3.0, 0.1)
            overage_fee = st.number_input("Overage Fee", 0.8, 2.5, 1.0, 0.05)
            roaming_calls = st.number_input("Roaming Calls", 0.8, 2.5, 1.0, 0.05)
            satisfaction = st.slider("Customer Satisfaction", 0.0, 1.0, 0.6, 0.01)
            complaint_rate = st.slider("Complaint Rate", 0.0, 1.0, 0.2, 0.01)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔮 Predict Churn Risk", use_container_width=True):
            # Encode categoricals
            age_group_map = {'Adult(25-35)': 0, 'Elderly(>55)': 1, 'Middle(36-45)': 2, 'Senior(46-55)': 3, 'Young(<25)': 4}
            plan_type_map = {'Contractual': 0, 'Pay-as-you-go': 1}
            status_map = {'Active': 0, 'Inactive': 1}

            # Build feature vector
            features = {
                'AccountAge': account_age,
                'DataUsage': data_usage,
                'Age': age,
                'MonthlySpend': monthly_spend,
                'CallDuration': call_duration,
                'CallFrequency': call_frequency,
                'SMSFrequency': sms_frequency,
                'DataFrequency': data_frequency,
                'PlanDuration': plan_duration,
                'OverageFee': overage_fee,
                'RoamingCalls': roaming_calls,
                'CustomerSatisfaction': satisfaction,
                'ComplaintRate': complaint_rate,
                'AgeGroup_enc': age_group_map.get(age_group, 2),
                'PlanType_enc': plan_type_map.get(plan_type, 1),
                'Status_enc': status_map.get(status, 0),
                'Usage_Intensity': data_usage * call_frequency,
                'Spend_Satisfaction': monthly_spend * satisfaction,
                'Complaint_Usage_Ratio': complaint_rate / (data_usage + 0.01),
                'Engagement_Score': (call_duration + sms_frequency + data_frequency) / 3,
                'Loyalty_Index': account_age * satisfaction,
                'Risk_Score': complaint_rate * roaming_calls / (satisfaction + 0.01),
            }

            input_df = pd.DataFrame([features])[feature_names]
            input_scaled = scaler.transform(input_df)
            input_lstm = input_scaled.reshape(1, 1, -1).astype(np.float32)

            prob = model.predict(input_lstm, verbose=0)[0][0]

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if prob > 0.7:
                    risk_class = "risk-critical"
                    risk_label = "CRITICAL RISK"
                elif prob > 0.5:
                    risk_class = "risk-high"
                    risk_label = "HIGH RISK"
                elif prob > 0.3:
                    risk_class = "risk-medium"
                    risk_label = "MODERATE RISK"
                else:
                    risk_class = "risk-low"
                    risk_label = "LOW RISK"

                st.markdown(f"""
                <div class="metric-card" style="padding: 40px;">
                    <h4>Churn Probability</h4>
                    <div class="metric-value {risk_class}" style="font-size: 48px;">{prob*100:.1f}%</div>
                    <p class="{risk_class}" style="margin-top: 12px; font-size: 16px; font-weight: 600;">{risk_label}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if prob > 0.7:
                    offer = "🚨 **Critical:** Offer 50% discount on next 3 months + free Premium add-on + dedicated account manager!"
                elif prob > 0.5:
                    offer = "⚠️ **High Risk:** Send personalized retention email with 25% discount coupon + loyalty rewards."
                elif prob > 0.3:
                    offer = "📋 **Moderate:** Schedule customer satisfaction survey + suggest plan upgrade benefits."
                else:
                    offer = "🌟 **Loyal Customer:** Recommend referral program + upsell to premium long-term contract."

                st.markdown(f"""
                <div class="offer-box">
                    <h3 style="color: white; -webkit-text-fill-color: white; margin-bottom: 12px;">AI Recommended Action</h3>
                    <p style="font-size: 15px;">{offer}</p>
                </div>
                """, unsafe_allow_html=True)

            # Quick SHAP for this prediction
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Feature Contribution Analysis")

            try:
                import shap
                def predict_wrapper(X):
                    X_3d = np.array(X, dtype=np.float32).reshape(-1, 1, X.shape[1])
                    return model.predict(X_3d, verbose=0).flatten()

                bg_summary = shap.kmeans(background, 10)
                explainer = shap.KernelExplainer(predict_wrapper, bg_summary)

                with st.spinner("Computing SHAP explanations..."):
                    sv = explainer.shap_values(input_df)
                    shap_vals = sv[0] if isinstance(sv, list) else sv[0]

                    feat_imp = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': shap_vals
                    })
                    feat_imp['Abs'] = feat_imp['SHAP Value'].abs()
                    feat_imp = feat_imp.sort_values('Abs', ascending=False).head(10)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['#ef4444' if v > 0 else '#22c55e' for v in feat_imp['SHAP Value'][::-1]]
                    ax.barh(feat_imp['Feature'][::-1], feat_imp['SHAP Value'][::-1], color=colors, height=0.6)
                    ax.axvline(x=0, color='#475569', linewidth=0.8)
                    ax.set_xlabel('SHAP Value (Impact on Churn)')
                    style_plot(fig, ax, "Top 10 Feature Contributions")
                    st.pyplot(fig)
                    plt.close()
            except Exception as ex:
                st.warning(f"SHAP computation skipped: {ex}")


# ============================================================================
#  PAGE: CUSTOMER SEARCH
# ============================================================================
elif page == "🔍 Customer Search":
    st.markdown("# 🔍 Customer Search")
    st.markdown("Search and analyze individual customers from the dataset.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if model_loaded and not dataset.empty:
        customer_idx = st.number_input("Customer Index", 0, len(dataset)-1, 0, step=1)
        customer = dataset.iloc[customer_idx]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("#### Customer Profile")
            st.markdown(f"**Age:** {customer['Age']:.0f}")
            st.markdown(f"**Age Group:** {customer['AgeGroup']}")
            st.markdown(f"**Plan:** {customer['PlanType']}")
            st.markdown(f"**Status:** {customer['Status']}")
            st.markdown(f"**Satisfaction:** {customer['CustomerSatisfaction']:.2f}")
            actual_churn = "Yes" if customer['Churn'] == 1 else "No"
            churn_color = "#ef4444" if customer['Churn'] == 1 else "#22c55e"
            st.markdown(f"**Actual Churn:** <span style='color:{churn_color};font-weight:bold;'>{actual_churn}</span>",
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### All Features")
            st.dataframe(pd.DataFrame(customer).T, use_container_width=True)

        if st.button("🔮 Predict & Explain", use_container_width=True):
            # Prepare features
            age_group_map = {'Adult(25-35)': 0, 'Elderly(>55)': 1, 'Middle(36-45)': 2, 'Senior(46-55)': 3, 'Young(<25)': 4}
            plan_type_map = {'Contractual': 0, 'Pay-as-you-go': 1}
            status_map = {'Active': 0, 'Inactive': 1}

            features = {
                'AccountAge': customer['AccountAge'],
                'DataUsage': customer['DataUsage'],
                'Age': customer['Age'],
                'MonthlySpend': customer['MonthlySpend'],
                'CallDuration': customer['CallDuration'],
                'CallFrequency': customer['CallFrequency'],
                'SMSFrequency': customer['SMSFrequency'],
                'DataFrequency': customer['DataFrequency'],
                'PlanDuration': customer['PlanDuration'],
                'OverageFee': customer['OverageFee'],
                'RoamingCalls': customer['RoamingCalls'],
                'CustomerSatisfaction': customer['CustomerSatisfaction'],
                'ComplaintRate': customer['ComplaintRate'],
                'AgeGroup_enc': age_group_map.get(customer['AgeGroup'], 2),
                'PlanType_enc': plan_type_map.get(customer['PlanType'], 1),
                'Status_enc': status_map.get(customer['Status'], 0),
                'Usage_Intensity': customer['DataUsage'] * customer['CallFrequency'],
                'Spend_Satisfaction': customer['MonthlySpend'] * customer['CustomerSatisfaction'],
                'Complaint_Usage_Ratio': customer['ComplaintRate'] / (customer['DataUsage'] + 0.01),
                'Engagement_Score': (customer['CallDuration'] + customer['SMSFrequency'] + customer['DataFrequency']) / 3,
                'Loyalty_Index': customer['AccountAge'] * customer['CustomerSatisfaction'],
                'Risk_Score': customer['ComplaintRate'] * customer['RoamingCalls'] / (customer['CustomerSatisfaction'] + 0.01),
            }

            input_df = pd.DataFrame([features])[feature_names]
            input_scaled = scaler.transform(input_df)
            input_lstm = input_scaled.reshape(1, 1, -1).astype(np.float32)

            prob = model.predict(input_lstm, verbose=0)[0][0]

            col1, col2 = st.columns(2)
            with col1:
                risk_class = "risk-critical" if prob > 0.7 else "risk-high" if prob > 0.5 else "risk-medium" if prob > 0.3 else "risk-low"
                st.markdown(f"""
                <div class="metric-card" style="padding: 30px;">
                    <h4>Predicted Churn Probability</h4>
                    <div class="metric-value {risk_class}" style="font-size: 42px;">{prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                correct = (prob > 0.5 and customer['Churn'] == 1) or (prob <= 0.5 and customer['Churn'] == 0)
                st.markdown(f"""
                <div class="metric-card" style="padding: 30px;">
                    <h4>Prediction vs Actual</h4>
                    <div class="metric-value {'risk-low' if correct else 'risk-critical'}" style="font-size: 42px;">
                        {'Correct ✓' if correct else 'Incorrect ✗'}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
#  PAGE: LSTM PERFORMANCE
# ============================================================================
elif page == "📊 LSTM Performance":
    st.markdown("# 📊 LSTM Model Performance")
    st.markdown("Comprehensive evaluation metrics and visualizations.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if model_loaded and 'metrics' in results:
        m = results['metrics']

        # Main metrics
        cols = st.columns(5)
        for col, (name, val) in zip(cols, [
            ("Accuracy", m['accuracy']),
            ("Precision", m['precision']),
            ("Recall", m['recall']),
            ("F1-Score", m['f1_score']),
            ("ROC-AUC", m['roc_auc']),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <div class="metric-value">{val*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Confusion Matrix
            cm = np.array(m['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation='nearest', cmap='PuBu')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Not Churned', 'Churned'], color='#94a3b8')
            ax.set_yticklabels(['Not Churned', 'Churned'], color='#94a3b8')
            ax.set_xlabel('Predicted', color='#94a3b8')
            ax.set_ylabel('Actual', color='#94a3b8')

            # Add text
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i][j]),
                            ha='center', va='center',
                            color='white' if cm[i][j] > cm.max()/2 else '#e2e8f0',
                            fontsize=20, fontweight='bold')
            style_plot(fig, ax, "Confusion Matrix")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()

        with col2:
            # ROC Curve
            roc = m.get('roc_curve', {})
            if roc:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(roc['fpr'], roc['tpr'], color='#6366f1', linewidth=2.5,
                        label=f"AUC = {m['roc_auc']*100:.1f}%")
                ax.plot([0, 1], [0, 1], color='#374151', linestyle='--')
                ax.fill_between(roc['fpr'], roc['tpr'], alpha=0.15, color='#6366f1')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(facecolor='#0f1724', edgecolor='#1e293b', labelcolor='#94a3b8',
                          fontsize=12)
                style_plot(fig, ax, "ROC Curve")
                st.pyplot(fig)
                plt.close()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Training Loss
            hist = m.get('training_history', {})
            if hist:
                fig, ax = plt.subplots(figsize=(6, 4))
                epochs = range(1, len(hist['train_loss']) + 1)
                ax.plot(epochs, hist['train_loss'], color='#6366f1', linewidth=2, label='Train Loss')
                ax.plot(epochs, hist['val_loss'], color='#a78bfa', linewidth=2, linestyle='--', label='Val Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend(facecolor='#0f1724', edgecolor='#1e293b', labelcolor='#94a3b8')
                style_plot(fig, ax, "Training & Validation Loss")
                st.pyplot(fig)
                plt.close()

        with col2:
            # Training Accuracy
            if hist:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(epochs, [v*100 for v in hist['train_accuracy']],
                        color='#22c55e', linewidth=2, label='Train Accuracy')
                ax.plot(epochs, [v*100 for v in hist['val_accuracy']],
                        color='#86efac', linewidth=2, linestyle='--', label='Val Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.legend(facecolor='#0f1724', edgecolor='#1e293b', labelcolor='#94a3b8')
                style_plot(fig, ax, "Training & Validation Accuracy")
                st.pyplot(fig)
                plt.close()

        # Precision-Recall Curve
        pr = m.get('pr_curve', {})
        if pr:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(pr['recall'], pr['precision'], color='#f59e0b', linewidth=2.5)
                ax.fill_between(pr['recall'], pr['precision'], alpha=0.1, color='#f59e0b')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                style_plot(fig, ax, "Precision-Recall Curve")
                st.pyplot(fig)
                plt.close()

            with col2:
                # Classification Report
                report = m.get('classification_report', {})
                if report:
                    st.markdown("#### Classification Report")
                    report_df = pd.DataFrame(report).T
                    report_df = report_df.round(4)
                    st.dataframe(report_df, use_container_width=True)


# ============================================================================
#  PAGE: BEHAVIORAL DRIFT ANALYSIS
# ============================================================================
elif page == "🌊 Behavioral Drift":
    st.markdown("# 🌊 Behavioral Drift Analysis")
    st.markdown("Monitor how customer behavior patterns shift across temporal segments using Population Stability Index (PSI).")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if 'drift_analysis' in results:
        drift = results['drift_analysis']

        # Churn rate drift
        churn_rates = drift.get('churn_rate_by_window', [])
        if churn_rates:
            st.markdown("### Churn Rate Across Temporal Windows")
            fig, ax = plt.subplots(figsize=(10, 4))
            windows = [f"Window {i+1}" for i in range(len(churn_rates))]
            bars = ax.bar(windows, churn_rates, color=['#6366f1', '#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd'],
                          width=0.5, edgecolor='none')
            for bar, val in zip(bars, churn_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', va='bottom', color='#e2e8f0',
                        fontweight='bold', fontsize=12)
            ax.set_ylabel('Churn Rate (%)')
            style_plot(fig, ax, "Churn Rate Drift Across Time Windows")
            st.pyplot(fig)
            plt.close()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # PSI analysis
        st.markdown("### Population Stability Index (PSI) by Feature")
        drift_features = {k: v for k, v in drift.items()
                         if isinstance(v, dict) and 'psi' in v}

        if drift_features:
            # Sort by PSI
            sorted_drift = sorted(drift_features.items(),
                                  key=lambda x: x[1]['psi'], reverse=True)

            # PSI bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            names = [d[0] for d in sorted_drift]
            psi_vals = [d[1]['psi'] for d in sorted_drift]
            colors = ['#ef4444' if p > 0.2 else '#eab308' if p > 0.1 else '#22c55e' for p in psi_vals]

            bars = ax.barh(names[::-1], psi_vals[::-1], color=colors[::-1], height=0.6)
            ax.axvline(x=0.1, color='#eab308', linestyle='--', alpha=0.7, label='Medium (0.1)')
            ax.axvline(x=0.2, color='#ef4444', linestyle='--', alpha=0.7, label='High (0.2)')
            ax.set_xlabel('PSI Value')
            ax.legend(facecolor='#0f1724', edgecolor='#1e293b', labelcolor='#94a3b8')
            style_plot(fig, ax, "Population Stability Index (PSI)")
            st.pyplot(fig)
            plt.close()

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Detailed drift table
            st.markdown("### Detailed Drift Metrics")
            drift_table_data = []
            for feat, d in sorted_drift:
                severity_badge = f'<span class="drift-{d["drift_severity"].lower()}">{d["drift_severity"]}</span>'
                drift_table_data.append({
                    'Feature': feat,
                    'PSI': f"{d['psi']:.4f}",
                    'Mean Drift': f"{d['mean_drift']:.4f}",
                    'Std Drift': f"{d['std_drift']:.4f}",
                    'Severity': d['drift_severity']
                })

            drift_df = pd.DataFrame(drift_table_data)
            st.dataframe(drift_df, use_container_width=True)

            # Feature mean trends
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Feature Mean Trends Across Windows")

            selected_feat = st.selectbox("Select Feature", list(drift_features.keys()))
            if selected_feat:
                feat_data = drift_features[selected_feat]
                fig, ax = plt.subplots(figsize=(10, 4))
                windows_x = [f"W{i+1}" for i in range(len(feat_data['window_means']))]

                ax.plot(windows_x, feat_data['window_means'], color='#6366f1', linewidth=2.5,
                        marker='o', markersize=8, markerfacecolor='#a78bfa', markeredgecolor='#6366f1')
                ax.fill_between(windows_x,
                                [m - s for m, s in zip(feat_data['window_means'], feat_data['window_stds'])],
                                [m + s for m, s in zip(feat_data['window_means'], feat_data['window_stds'])],
                                alpha=0.15, color='#6366f1')
                ax.set_xlabel('Temporal Window')
                ax.set_ylabel('Mean Value')
                style_plot(fig, ax, f"{selected_feat} - Mean Trend (with Std Band)")
                st.pyplot(fig)
                plt.close()

    else:
        st.warning("Drift analysis not available. Run training first.")


# ============================================================================
#  PAGE: EXPLAINABLE AI
# ============================================================================
elif page == "🧠 Explainable AI":
    st.markdown("# 🧠 Explainable AI (XAI)")
    st.markdown("SHAP-based feature importance analysis for model interpretability.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if 'shap_data' in results:
        shap_info = results['shap_data']
        feat_imp = shap_info.get('feature_importance', {})

        if feat_imp:
            # Global Feature Importance
            st.markdown("### Global Feature Importance (Mean |SHAP|)")

            fig, ax = plt.subplots(figsize=(10, 6))
            features_sorted = list(feat_imp.keys())
            values_sorted = list(feat_imp.values())

            # Color gradient
            n = len(features_sorted)
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, n))

            ax.barh(features_sorted[::-1], values_sorted[::-1],
                    color=colors[::-1], height=0.6, edgecolor='none')
            ax.set_xlabel('Mean |SHAP Value|')
            style_plot(fig, ax, "Global Feature Importance (SHAP)")
            st.pyplot(fig)
            plt.close()

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Top features details
            st.markdown("### Top Feature Analysis")

            top_features = list(feat_imp.items())[:5]
            cols = st.columns(5)
            for col, (feat, imp) in zip(cols, top_features):
                with col:
                    pct = imp / max(feat_imp.values()) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="font-size: 11px;">{feat}</h4>
                        <div class="metric-value" style="font-size: 24px;">{imp:.4f}</div>
                        <p style="color: #64748b; margin-top: 4px; font-size: 12px;">
                            {pct:.0f}% relative impact
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            # Feature importance table
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Complete Feature Importance Table")
            imp_df = pd.DataFrame([
                {'Rank': i+1, 'Feature': feat, 'Mean |SHAP|': f"{val:.6f}",
                 'Relative %': f"{val/max(feat_imp.values())*100:.1f}%"}
                for i, (feat, val) in enumerate(feat_imp.items())
            ])
            st.dataframe(imp_df, use_container_width=True)

            # Interpretation
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### How to Interpret SHAP Values")
            st.markdown("""
            <div class="glass-panel">
                <p><strong>SHAP (SHapley Additive exPlanations)</strong> assigns each feature an importance value
                for a particular prediction.</p>
                <ul style="color: #94a3b8;">
                    <li><span style="color: #ef4444;">Positive SHAP values</span> push the prediction towards <strong>Churn</strong></li>
                    <li><span style="color: #22c55e;">Negative SHAP values</span> push the prediction towards <strong>Not Churn</strong></li>
                    <li>Higher <strong>|SHAP|</strong> = stronger influence on the model's decision</li>
                    <li>Global importance = average |SHAP| across all explained samples</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("SHAP data not available. Run training first.")


# ============================================================================
#  PAGE: CHURN REPORT
# ============================================================================
elif page == "📋 Churn Report":
    st.markdown("# 📋 Churn Report")
    st.markdown("Detailed tabular analysis of the churn noisy dataset.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if not dataset.empty:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Customers</h4>
                <div class="metric-value">{len(dataset):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            churned = dataset['Churn'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h4>Churned</h4>
                <div class="metric-value risk-high">{int(churned):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            retained = len(dataset) - churned
            st.markdown(f"""
            <div class="metric-card">
                <h4>Retained</h4>
                <div class="metric-value risk-low">{int(retained):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Churn Rate</h4>
                <div class="metric-value risk-medium">{dataset['Churn'].mean()*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Churn by Age Group")
            if 'AgeGroup' in dataset.columns:
                churn_by_age = dataset.groupby('AgeGroup')['Churn'].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(churn_by_age.index, churn_by_age.values,
                              color='#6366f1', edgecolor='none', width=0.5)
                for bar, val in zip(bars, churn_by_age.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                            f'{val:.1f}%', ha='center', va='bottom', color='#e2e8f0',
                            fontsize=10, fontweight='bold')
                ax.set_ylabel('Churn Rate (%)')
                ax.tick_params(axis='x', rotation=30)
                style_plot(fig, ax, "Churn Rate by Age Group")
                st.pyplot(fig)
                plt.close()

        with col2:
            st.markdown("### Churn by Plan Type")
            if 'PlanType' in dataset.columns:
                churn_by_plan = dataset.groupby('PlanType')['Churn'].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#6366f1', '#a78bfa']
                bars = ax.bar(churn_by_plan.index, churn_by_plan.values,
                              color=colors[:len(churn_by_plan)], edgecolor='none', width=0.4)
                for bar, val in zip(bars, churn_by_plan.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                            f'{val:.1f}%', ha='center', va='bottom', color='#e2e8f0',
                            fontsize=12, fontweight='bold')
                ax.set_ylabel('Churn Rate (%)')
                style_plot(fig, ax, "Churn Rate by Plan Type")
                st.pyplot(fig)
                plt.close()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Data table with filtering
        st.markdown("### Customer Data Explorer")
        filter_churn = st.selectbox("Filter by Churn", ["All", "Churned Only", "Retained Only"])

        if filter_churn == "Churned Only":
            filtered = dataset[dataset['Churn'] == 1]
        elif filter_churn == "Retained Only":
            filtered = dataset[dataset['Churn'] == 0]
        else:
            filtered = dataset

        st.dataframe(filtered, use_container_width=True)
        st.caption(f"Showing {len(filtered):,} of {len(dataset):,} records")
    else:
        st.error("Dataset not found. Ensure churn_noisy.xlsx exists.")


# ============================================================================
#  PAGE: MODEL OVERVIEW
# ============================================================================
elif page == "⚙️ Model Overview":
    st.markdown("# ⚙️ Model Architecture Overview")
    st.markdown("Technical details of the Bidirectional LSTM with Self-Attention.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    model_info = results.get('model_info', {})

    # Architecture visual
    st.markdown("### Neural Network Architecture")
    layers = [
        ("Input Layer", f"Shape: (1, {model_info.get('n_features', 22)})", "#3b82f6"),
        ("Bidirectional LSTM #1", "256 units, return_sequences=True", "#6366f1"),
        ("BatchNorm + Dropout", "0.25 dropout rate", "#475569"),
        ("Bidirectional LSTM #2", "128 units, return_sequences=True", "#7c3aed"),
        ("BatchNorm + Dropout", "0.25 dropout rate", "#475569"),
        ("Bidirectional LSTM #3", "64 units, return_sequences=True", "#8b5cf6"),
        ("BatchNorm + Dropout", "0.2 dropout rate", "#475569"),
        ("Self-Attention", "Attention([x, x]) + Concatenate", "#a78bfa"),
        ("Flatten", "Reshape 3D to 2D", "#475569"),
        ("Dense Layer", "128 units, ReLU + BatchNorm + Dropout(0.2)", "#6366f1"),
        ("Dense Layer", "64 units, ReLU + BatchNorm + Dropout(0.15)", "#7c3aed"),
        ("Dense Layer", "32 units, ReLU", "#8b5cf6"),
        ("Output Layer", "1 unit, Sigmoid (Churn Probability)", "#22c55e"),
    ]

    for name, desc, color in layers:
        st.markdown(f"""
        <div class="arch-layer" style="border-left-color: {color};">
            <strong style="color: {color};">{name}</strong>
            <span style="color: #64748b; margin-left: 12px;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Training configuration
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Training Configuration")
        st.markdown("""
        <div class="glass-panel">
            <table style="width: 100%; color: #94a3b8;">
                <tr><td style="padding: 6px;"><strong>Optimizer</strong></td><td>Adam (lr=0.0005)</td></tr>
                <tr><td style="padding: 6px;"><strong>Loss Function</strong></td><td>Binary Crossentropy</td></tr>
                <tr><td style="padding: 6px;"><strong>Batch Size</strong></td><td>32</td></tr>
                <tr><td style="padding: 6px;"><strong>Max Epochs</strong></td><td>100</td></tr>
                <tr><td style="padding: 6px;"><strong>Early Stopping</strong></td><td>Patience=20 (on val_accuracy)</td></tr>
                <tr><td style="padding: 6px;"><strong>LR Scheduler</strong></td><td>ReduceOnPlateau (factor=0.5)</td></tr>
                <tr><td style="padding: 6px;"><strong>Class Balance</strong></td><td>SMOTE Oversampling</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Key Components")
        st.markdown("""
        <div class="glass-panel">
            <table style="width: 100%; color: #94a3b8;">
                <tr><td style="padding: 6px;"><strong>Model Type</strong></td><td>Bidirectional LSTM + Attention</td></tr>
                <tr><td style="padding: 6px;"><strong>XAI Method</strong></td><td>SHAP (KernelExplainer)</td></tr>
                <tr><td style="padding: 6px;"><strong>Drift Detection</strong></td><td>Population Stability Index (PSI)</td></tr>
                <tr><td style="padding: 6px;"><strong>Dataset</strong></td><td>churn_noisy.xlsx (2,850 records)</td></tr>
                <tr><td style="padding: 6px;"><strong>Features</strong></td><td>22 (13 original + 3 encoded + 6 engineered)</td></tr>
                <tr><td style="padding: 6px;"><strong>Target</strong></td><td>Churn (Binary: 0/1)</td></tr>
                <tr><td style="padding: 6px;"><strong>Interpretability</strong></td><td>Per-prediction + Global SHAP</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Feature list
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Engineered Features")
    feat_list = model_info.get('feature_names', feature_names if model_loaded else [])
    if feat_list:
        feat_df = pd.DataFrame({
            'Index': range(len(feat_list)),
            'Feature Name': feat_list,
            'Type': ['Original' if i < 13 else 'Encoded' if i < 16 else 'Engineered'
                     for i in range(len(feat_list))]
        })
        st.dataframe(feat_df, use_container_width=True)
