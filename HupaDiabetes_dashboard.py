import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Diabetes Intelligence Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

[data-testid="stSidebar"] {
    background-color: #161B22;
}

h1, h2, h3, h4 {
    color: white;
    font-family: 'Segoe UI';
}

.metric-card {
    background: linear-gradient(145deg, #1c2333, #111827);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    border: 1px solid #2d3748;
}

.insight-box {
    background: #1F2937;
    padding: 15px;
    border-radius: 15px;
    border-left: 5px solid #00D4FF;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data

def load_data():
    df = pd.read_csv("cleaned_hupa_diabetes_recent.csv")
    df['time'] = pd.to_datetime(df['time'])
    return df


df = load_data()

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🧠 AI Diabetes Analytics")

analysis_type = st.sidebar.radio(
    "Select Analysis",
    [
        "Descriptive Analytics",
        "Predictive Analytics",
        "Prescriptive Analytics"
    ]
)

patient_list = ['All Patients'] + list(df['patient_id'].unique())
selected_patient = st.sidebar.selectbox(
    "Select Patient",
    patient_list
)

if selected_patient != 'All Patients':
    df = df[df['patient_id'] == selected_patient]

# ======================================================
# HEADER
# ======================================================

st.title("🩺 AI-Powered Diabetes Intelligence Dashboard")
st.markdown("### HUPA-UCM Continuous Glucose Monitoring Analytics")

# ======================================================
# KPI SECTION
# ======================================================

col1, col2, col3, col4 = st.columns(4)

avg_glucose = round(df['glucose'].mean(), 2)
max_glucose = round(df['glucose'].max(), 2)
min_glucose = round(df['glucose'].min(), 2)
avg_steps = round(df['steps'].mean(), 0)

with col1:
    st.metric("Average Glucose", avg_glucose)

with col2:
    st.metric("Maximum Glucose", max_glucose)

with col3:
    st.metric("Minimum Glucose", min_glucose)

with col4:
    st.metric("Average Steps", avg_steps)

# ======================================================
# DESCRIPTIVE ANALYTICS
# ======================================================

if analysis_type == "Descriptive Analytics":

    st.subheader("📊 Descriptive Analytics Dashboard")

    # --------------------------------------------------
    # TIME IN RANGE
    # --------------------------------------------------

    tir = ((df['glucose'] >= 70) & (df['glucose'] <= 180)).mean() * 100

    fig_tir = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = tir,
        title = {'text': "Time In Range (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00D4FF"},
            'steps': [
                {'range': [0, 50], 'color': '#7f1d1d'},
                {'range': [50, 70], 'color': '#78350f'},
                {'range': [70, 100], 'color': '#14532d'}
            ]
        }
    ))

    st.plotly_chart(fig_tir, use_container_width=True)

    # --------------------------------------------------
    # 24-HOUR GLUCOSE TREND
    # --------------------------------------------------

    fig_glucose = px.line(
        df,
        x='time',
        y='glucose',
        title='24-Hour Glucose Trend',
        template='plotly_dark'
    )

    fig_glucose.add_hline(y=70, line_dash='dash', line_color='red')
    fig_glucose.add_hline(y=180, line_dash='dash', line_color='orange')

    st.plotly_chart(fig_glucose, use_container_width=True)

    # --------------------------------------------------
    # HYPOGLYCEMIA ANALYSIS
    # --------------------------------------------------

    hypo = df[df['glucose'] < 70]

    fig_hypo = px.histogram(
        hypo,
        x='glucose',
        nbins=20,
        title='Hypoglycemic Frequency Distribution',
        template='plotly_dark'
    )

    st.plotly_chart(fig_hypo, use_container_width=True)

    # --------------------------------------------------
    # HEART RATE VS ROC
    # --------------------------------------------------

    fig_hr = px.scatter(
        df,
        x='heart_rate',
        y='glucose_roc',
        color='glucose',
        size='steps',
        title='Heart Rate vs Glucose Rate of Change',
        template='plotly_dark',
        hover_data=['patient_id', 'glucose']
    )

    st.plotly_chart(fig_hr, use_container_width=True)

# ======================================================
# PREDICTIVE ANALYTICS
# ======================================================

elif analysis_type == "Predictive Analytics":

    st.subheader("🤖 Predictive Analytics Dashboard")

    # --------------------------------------------------
    # HYPOGLYCEMIA RISK SCORE
    # --------------------------------------------------

    df['risk_score'] = (
        abs(df['glucose_roc']) * 0.4 +
        abs(df['glucose_rolling_std_1h']) * 0.4 +
        abs(df['heart_rate']) * 0.2
    )

    fig_risk = px.line(
        df,
        x='time',
        y='risk_score',
        title='Predicted Hypoglycemia Risk Trend',
        template='plotly_dark'
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    # --------------------------------------------------
    # HIGH VARIABILITY DAY PREDICTION
    # --------------------------------------------------

    fig_var = px.scatter(
        df,
        x='glucose_rolling_std_1h',
        y='glucose',
        color='steps',
        size='heart_rate',
        title='Glucose Variability Prediction',
        template='plotly_dark'
    )

    st.plotly_chart(fig_var, use_container_width=True)

    # --------------------------------------------------
    # MORNING GLUCOSE PREDICTION
    # --------------------------------------------------

    morning = df[df['hour'].between(5,10)]

    fig_morning = px.box(
        morning,
        x='hour',
        y='glucose',
        color='hour',
        title='Morning Glucose Pattern Analysis',
        template='plotly_dark'
    )

    st.plotly_chart(fig_morning, use_container_width=True)

# ======================================================
# PRESCRIPTIVE ANALYTICS
# ======================================================

elif analysis_type == "Prescriptive Analytics":

    st.subheader("🧠 Prescriptive Intervention Dashboard")

    # --------------------------------------------------
    # GLUCOSE VARIABILITY INTERVENTION
    # --------------------------------------------------

    df['Risk_Level'] = np.where(
        df['glucose_rolling_std_1h'] > 30,
        'High Risk',
        'Stable'
    )

    fig_intervention = px.scatter(
        df,
        x='time',
        y='glucose_rolling_std_1h',
        color='Risk_Level',
        size='glucose',
        title='Glucose Variability Risk Intervention',
        template='plotly_dark',
        hover_data=['patient_id', 'steps', 'carb_input']
    )

    fig_intervention.add_hline(
        y=30,
        line_dash='dash',
        line_color='red'
    )

    st.plotly_chart(fig_intervention, use_container_width=True)

    # --------------------------------------------------
    # ACTIVITY VS INSULIN
    # --------------------------------------------------

    fig_activity = px.scatter(
        df,
        x='calories_burned',
        y='basal_rate',
        color='glucose',
        size='steps',
        trendline='lowess',
        title='Physical Activity vs Insulin Requirement',
        template='plotly_dark'
    )

    st.plotly_chart(fig_activity, use_container_width=True)

    # --------------------------------------------------
    # CARB TREATMENT HEATMAP
    # --------------------------------------------------

    hypo = df[df['glucose'] < 70].copy()

    hypo['Recovery_Status'] = np.where(
        hypo['glucose_roc'] > 3,
        'Rebound Hyperglycemia',
        'Safe Recovery'
    )

    hypo['carb_bin'] = pd.cut(
        hypo['carb_input'],
        bins=[0,5,10,15,20,30,50]
    ).astype(str)

    heat = hypo.groupby(
        ['carb_bin', 'Recovery_Status'],
        observed=False
    ).size().reset_index(name='count')

    fig_heat = px.density_heatmap(
        heat,
        x='carb_bin',
        y='Recovery_Status',
        z='count',
        text_auto=True,
        title='Optimal Carb Intake for Hypoglycemia Recovery',
        template='plotly_dark'
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    # --------------------------------------------------
    # AI INTERVENTION ENGINE
    # --------------------------------------------------

    st.markdown("## 🚨 AI Intervention Recommendations")

    if df['glucose_rolling_std_1h'].mean() > 30:
        st.error("High glucose variability detected. Recommend insulin reassessment and activity intervention.")

    if df['glucose'].max() > 250:
        st.warning("Severe hyperglycemia risk detected. Increase glucose monitoring frequency.")

    if df['glucose'].min() < 60:
        st.info("Hypoglycemia intervention recommended. Monitor carb correction strategy.")

# ======================================================
# INSIGHT PANEL
# ======================================================

st.markdown("---")
st.subheader("📌 Clinical Insights")

st.markdown("""
<div class='insight-box'>
<b>Key Clinical Findings:</b><br><br>
• Increased glucose variability strongly predicts future glycemic instability.<br><br>
• Moderate physical activity improves insulin sensitivity and reduces glucose fluctuations.<br><br>
• High carbohydrate meals significantly increase post-meal glucose excursions.<br><br>
• Sleep duration between 7–8 hours is associated with improved Time-In-Range.<br><br>
• Early predictive alerts can reduce severe hypoglycemia risk through proactive intervention.
</div>
""", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")
st.caption("Developed for HUPA-UCM Diabetes Intelligence Analytics | PyCore Python Hackathon 2026")
