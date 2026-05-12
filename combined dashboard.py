# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="GlucoAI Diabetes Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
}

.block-container {
    padding-top: 1 rem;
}

/* SIDEBAR WIDTH */
[data-testid="stSidebar"] {
    width: 320px;
    min-width: 320px;
    max-width: 320px;
    background: linear-gradient(180deg,#0B1F3A 0%, #102B50 100%);
    padding-top: 0rem;
}

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}



/* SIDEBAR BUTTON FULL WIDTH */
section[data-testid="stSidebar"] div.stButton > button {
    width: 240px;
    height: 55px;
    border-radius: 14px;
    background-color: #112D4E;
    color: white;
    border: 1px solid #1E4D7A;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 5px;
    text-align: left;
    padding-left: 18px;
    transition: 0.3s;
}

/* HOVER EFFECT */
section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #1B4F8C;
    border: 1px solid #4FC3F7;
    transform: translateX(3px);
}

/* REMOVE EXTRA SPACE */
section[data-testid="stSidebar"] {
    padding-top: 5px;
}


/* Single select dropdown styling */
[data-testid="stSidebar"] .stSelectbox {
    position: sticky;
    top: 10px;
    z-index: 999;
    background: transparent !important;
    padding-bottom: 10px;
}
/* Insight Box */

.insight-box {
    background: #EAF3FF;
    padding: 20px;
    border-radius: 20px;
    border-left: 6px solid #1B4F8C;
    color: #081229;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.12);
}

""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_hupa_diabetes_recent1.xlsb")
    demo = pd.read_csv("cleaned_demographics.csv")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if "patient_id" in demo.columns:
        df = df.merge(demo, on="patient_id", how="left")

    return df

df = load_data()

# ---------------------------------------------------
# COLUMN SAFETY
# ---------------------------------------------------

df = df.dropna(subset=["patient_id", "time", "glucose"])
df = df.sort_values(["patient_id", "time"])

bolus_col = "bolus_volume_delivered" if "bolus_volume_delivered" in df.columns else "bolus"

for col in ["steps", "heart_rate", "basal_rate", "carb_input", bolus_col]:
    if col not in df.columns:
        df[col] = 0

df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date
df["is_weekend"] = df["time"].dt.dayofweek.isin([5, 6]).astype(int)
df["is_night"] = df["hour"].between(0, 5).astype(int)

df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["glucose_rolling_mean_1h"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12)
    .mean()
    .reset_index(level=0, drop=True)
)

df["glucose_rolling_std_1h"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12)
    .std()
    .reset_index(level=0, drop=True)
)

df["is_hypoglycemia"] = (df["glucose"] < 70).astype(int)
df["is_in_range"] = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
df["is_hyperglycemia"] = (df["glucose"] > 180).astype(int)
# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------

#st.sidebar.title("🧭 Navigation Menu")

if "menu" not in st.session_state:
    st.session_state.menu = "Introduction"

if st.sidebar.button("🏠 Introduction"):
    st.session_state.menu = "Introduction"

if st.sidebar.button("📘 Overview"):
    st.session_state.menu = "Overview"

if st.sidebar.button("🧹 Data Cleaning"):
    st.session_state.menu = "Data Cleaning"

if st.sidebar.button("📊 Insights Dashboard"):
    st.session_state.menu = "Insights"

if st.sidebar.button("📌 Key Takeaways"):
    st.session_state.menu = "Key Takeaways"

if st.sidebar.button("✅ Conclusions"):
    st.session_state.menu = "Conclusions"

menu = st.session_state.menu

# ---------------------------------------------------
# PATIENT FILTER - SINGLE SELECT DROPDOWN
# ---------------------------------------------------

patients = sorted(df["patient_id"].unique())

patient_options = ["All Patients"] + patients

selected_patient = st.sidebar.selectbox(
"Select Patient",
patient_options,
index=0
)

if selected_patient == "All Patients":
 df_view = df.copy()
else:
 df_view = df[df["patient_id"] == selected_patient].copy()

if df_view.empty:
    st.warning("No data available for selected patient.")
    st.stop()


# ---------------------------------------------------
# DAILY SUMMARY
# ---------------------------------------------------

daily = (
    df_view
    .groupby(["patient_id", "date"])
    .agg(
        daily_tir=("is_in_range", "mean"),
        avg_glucose=("glucose", "mean"),
        glucose_variability=("glucose", "std"),
        daily_steps=("steps", "sum"),
        avg_hr=("heart_rate", "mean"),
        avg_basal=("basal_rate", "mean"),
        total_bolus=(bolus_col, "sum"),
        total_carbs=("carb_input", "sum"),
        hypo_rate=("is_hypoglycemia", "mean"),
        hyper_rate=("is_hyperglycemia", "mean")
    )
    .reset_index()
)

daily["daily_tir"] *= 100
# ===================================================
# INTRODUCTION PAGE
# ===================================================

if menu == "Introduction":

  st.image("Assets/Introduction.png", use_container_width=True)
# ===================================================
# OVERVIEW PAGE
# ===================================================

elif menu == "Overview":

    st.image("Assets/dataoverview.png", use_container_width=True)

  

# ===================================================
# DATA CLEANING PAGE
# ===================================================

elif menu == "Data Cleaning":

    st.title("🧹 Data Cleaning & Feature Engineering")
    st.markdown("""
        <div class='insight-box'>
        <b>Data Cleaning Steps:</b><br><br>
            ✅ Removed null glucose and timestamp values<br/><br/>
            ✅ Converted timestamps into datetime format<br><br>
            ✅ Created hourly and daily features<br><br>
            ✅ Generated glucose rolling mean and rolling standard deviation<br><br>
            ✅ Created glucose rate-of-change (ROC)<br><br>
            ✅ Generated hypoglycemia and hyperglycemia flags<br><br>
            ✅ Handled missing insulin and activity values
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🧠 Engineered Features")

    engineered = pd.DataFrame({
        "Feature": [
            "glucose_roc",
            "glucose_rolling_std_1h",
            "is_hypoglycemia",
            "daily_tir",
            "post_meal_spike",
            "risk_score"
        ],
        "Purpose": [
            "Rapid glucose change detection",
            "Glucose instability monitoring",
            "Hypoglycemia classification",
            "Daily glucose control",
            "Meal response monitoring",
            "Intervention prioritization"
        ]
    })

    st.dataframe(engineered, use_container_width=True)

# ===================================================
# INSIGHTS DASHBOARD
# ===================================================

elif menu == "Insights":

    st.title("📊 Diabetes Insights Dashboard")

    # ---------------------------------------------------
    # KPI ROW
    # ---------------------------------------------------

    #total_patients = df_view["patient_id"].nunique()
    #total_records = len(df_view)
    tir = df_view["is_in_range"].mean() * 100
    hypo = df_view["is_hypoglycemia"].mean() * 100
    hyper = df_view["is_hyperglycemia"].mean() * 100
    avg_glucose = df_view["glucose"].mean()

    c1, c2, c3, c4 = st.columns(4)

    #c1.metric("👥 Patients", total_patients)
    #c2.metric("📊 Records", f"{total_records:,}")
    c1.metric("✅ TIR", f"{tir:.1f}%")
    c2.metric("⚠️ Hypo", f"{hypo:.1f}%")
    c3.metric("🔥 Hyper", f"{hyper:.1f}%")
    c4.metric("🩸 Avg Glucose", f"{avg_glucose:.1f}")

 
    # ---------------------------------------------------
    # TABS
    # ---------------------------------------------------
    
    tabs = st.tabs([
        "👥 Demographics",
        "📊 Glucose Overview",
        "🍽️ Meal + Insulin",
        "🏃 Activity + Sleep",
        "🌙 Night Risk",
        "🤖 Predictive",
        "💊 Prescriptive Analytics"
    ])

   
# ---------------------------------------------------
# Demographics
# ---------------------------------------------------

    with tabs[0]:

         col1, col2 = st.columns(2)
         with col1:
            if 'gender' in df.columns:

                gender_chart = px.pie(
                    df,
                    names='gender',
                    title='Gender Distribution'
                )
        
                st.plotly_chart(
                    gender_chart,
                    use_container_width=True
                )
    
         with col2:
            if 'age' in df.columns:
    
                age_chart = px.histogram(
                    df,
                    x='age',
                    nbins=20,
                    title='Age Distribution'
                )
        
                st.plotly_chart(
                    age_chart,
                    use_container_width=True
                )
    
        
# ---------------------------------------------------
# GLUCOSE OVERVIEW
# ---------------------------------------------------

    with tabs[1]:
        st.subheader("Glucose Monitoring Overview")
        hourly_glucose = df.groupby("hour")["glucose"].mean().reset_index()

        fig = px.line(
            hourly_glucose,
            x="hour",
            y="glucose",
            markers=True,
            color_discrete_sequence=["#E63946"],
            title="24-Hour Glycemic Trend"
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Glucose (mg/dL)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            tir_summary = (
                df_view
                .groupby("patient_id")
                .agg(
                    TBR=("is_hypoglycemia", "mean"),
                    TIR=("is_in_range", "mean"),
                    TAR=("is_hyperglycemia", "mean")
                ) * 100
            ).reset_index()

            tir_melt = tir_summary.melt(
                id_vars="patient_id",
                var_name="Range",
                value_name="Percentage"
            )

            fig = px.bar(
                tir_melt,
                x="patient_id",
                y="Percentage",
                color="Range",
                title="TBR / TIR / TAR by Patient"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df_view,
                x="patient_id",
                y="glucose",
                title="Glucose Distribution by Patient"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # MEAL + INSULIN
    # ---------------------------------------------------

        with tabs[2]:
            st.subheader("Meal, Carbohydrate, and Insulin Response")

            meal_df = df_view.copy()

            meal_df["glucose_next_2h"] = (
                meal_df.groupby("patient_id")["glucose"].shift(-24)
            )

            meal_df["post_meal_spike"] = (
                meal_df["glucose_next_2h"] - meal_df["glucose"]
            )

            meal_df = meal_df[meal_df["carb_input"] > 0].dropna(
                subset=["carb_input", "post_meal_spike", bolus_col]
            )

            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(
                    meal_df.sample(min(5000, len(meal_df)), random_state=42),
                    x="carb_input",
                    y="post_meal_spike",
                    size=bolus_col,
                    color="glucose",
                    trendline="ols",
                    title="Carbohydrate Intake vs Post-Meal Spike"
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.density_heatmap(
                    meal_df,
                    x="carb_input",
                    y="post_meal_spike",
                    nbinsx=30,
                    nbinsy=30,
                    title="Carb Load vs Spike Risk Heatmap"
                )

                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Missed Bolus Detection")

            meal_df["missed_bolus"] = (
                (meal_df["carb_input"] > 20) &
                (meal_df[bolus_col] == 0) &
                (meal_df["glucose_next_2h"] > 180)
            ).astype(int)

            missed = meal_df["missed_bolus"].value_counts().reset_index()
            missed.columns = ["Missed Bolus", "Count"]

            fig = px.bar(
                missed,
                x="Missed Bolus",
                y="Count",
                title="Detected Missed Bolus Events"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # ACTIVITY + SLEEP
    # ---------------------------------------------------

        with tabs[3]:
            st.subheader("Activity Impact on Glucose Stability")
        
            activity_df = df_view.copy()
        
            activity_df["glucose_drift_1h"] = (
                activity_df.groupby("patient_id")["glucose"].diff(12)
            )
        
            activity_df["activity_group"] = pd.cut(
                activity_df["steps"],
                bins=[-1, 0, 50, 500, 100000],
                labels=["Sedentary", "Low", "Moderate", "High"]
            )
        
            act_summary = (
                activity_df
                .groupby("activity_group", observed=True)
                .agg(
                    avg_drift=("glucose_drift_1h", "mean"),
                    avg_instability=("glucose_rolling_std_1h", "mean"),
                    avg_glucose=("glucose", "mean")
                )
                .reset_index()
            )
        
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    act_summary,
                    x="activity_group",
                    y="avg_instability",
                    color="activity_group",
                    title="Activity Level vs Glucose Instability"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    daily,
                    x="daily_steps",
                    y="daily_tir",
                    size="glucose_variability",
                    color="avg_glucose",
                    hover_data=["patient_id", "date"],
                    title="Daily Steps vs Time-In-Range"
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# NIGHT RISK
# ---------------------------------------------------

    with tabs[4]:
        st.subheader("Nocturnal Hypoglycemia Risk")
    
        night_df = df_view[df_view["is_night"] == 1].copy()
        night_df["nocturnal_hypo"] = (night_df["glucose"] < 70).astype(int)
    
        risk_curve = (
            night_df
            .groupby("basal_rate")["nocturnal_hypo"]
            .mean()
            .reset_index()
        )
    
        col1, col2 = st.columns(2)
    
        with col1:
            fig = px.line(
                risk_curve,
                x="basal_rate",
                y="nocturnal_hypo",
                markers=True,
                title="Night Basal Rate vs Hypoglycemia Risk"
            )
            st.plotly_chart(fig, use_container_width=True)
    
        with col2:
            fig = px.box(
                night_df,
                x="nocturnal_hypo",
                y="basal_rate",
                title="Basal Rate Distribution by Night Hypoglycemia"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PREDICTIVE 
# ---------------------------------------------------

    with tabs[5]:
        st.subheader("Predictive Models")
    
        model_choice = st.selectbox(
            "Select Prediction Task",
            [
                "Hypoglycemia Next 30 Minutes",
                "Hyperglycemia >200 Within 2 Hours After Meal",
                "Next 15-Minute Glucose ROC"
            ]
        )
    
        if model_choice == "Hypoglycemia Next 30 Minutes":
            model_data = df_view.copy()
            model_data["target"] = (
                model_data.groupby("patient_id")["glucose"].shift(-6) < 70
            ).astype(int)
    
            features = [
                "glucose", "glucose_roc", "glucose_rolling_std_1h",
                "basal_rate", bolus_col, "steps", "heart_rate", "hour"
            ]
    
            task = "classification"
    
        elif model_choice == "Hyperglycemia >200 Within 2 Hours After Meal":
            model_data = df_view[df_view["carb_input"] > 0].copy()
            model_data["target"] = (
                model_data.groupby("patient_id")["glucose"].shift(-24) > 200
            ).astype(int)
    
            features = [
                "glucose", "carb_input", bolus_col,
                "basal_rate", "steps", "heart_rate", "hour"
            ]
    
            task = "classification"
    
        elif model_choice == "Next 15-Minute Glucose ROC":
            model_data = df_view.copy()
            model_data["target"] = (
                model_data.groupby("patient_id")["glucose_roc"].shift(-3)
            )
    
            features = [
                "glucose", "glucose_roc", "glucose_rolling_std_1h",
                "basal_rate", bolus_col, "steps", "heart_rate", "hour"
            ]
    
            task = "regression"
    
        else:
            model_data = daily.copy()
            model_data["future_tir"] = (
                model_data.groupby("patient_id")["daily_tir"].shift(-7)
            )
            model_data["target"] = (
                model_data["future_tir"] < model_data["daily_tir"] - 10
            ).astype(int)
    
            features = [
                "daily_tir", "avg_glucose", "glucose_variability",
                "daily_steps", "avg_hr", "avg_basal", "total_bolus"
            ]
    
            task = "classification"
    
        model_df = model_data[features + ["target"]].dropna()
    
        if len(model_df) > 20000:
            model_df = model_df.sample(20000, random_state=42)
    
        if len(model_df) < 50 or model_df["target"].nunique() < 2:
            st.warning("Not enough balanced data for this model.")
        else:
            X = model_df[features]
            y = model_df["target"]
    
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
    
            if task == "classification":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
    
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]
    
                st.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
                st.metric("ROC-AUC", f"{roc_auc_score(y_test, prob):.3f}")
    
            else:
                model = RandomForestRegressor(
                    n_estimators=80,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
    
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
    
                st.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
                st.metric("R² Score", f"{r2_score(y_test, pred):.3f}")
    
            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
    
            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Feature Importance: {model_choice}"
            )
    
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PRESCRIPTIVE ANALYTICS
# ---------------------------------------------------

    with tabs[6]:
        st.subheader("Prescriptive Insulin Effectiveness Score")
    
        score = (
            df_view
            .groupby("patient_id")
            .agg(
                tir=("is_in_range", "mean"),
                glucose_variability=("glucose", "std"),
                hypo_rate=("is_hypoglycemia", "mean"),
                hyper_rate=("is_hyperglycemia", "mean"),
                avg_steps=("steps", "mean"),
                total_bolus=(bolus_col, "sum"),
                avg_basal=("basal_rate", "mean")
            )
            .reset_index()
        )
    
        score["tir_score"] = score["tir"] * 40
    
        score["stability_score"] = (
            1 - score["glucose_variability"].rank(pct=True)
        ) * 25
    
        score["hypo_safety_score"] = (
            1 - score["hypo_rate"].rank(pct=True)
        ) * 20
    
        score["activity_score"] = (
            score["avg_steps"].rank(pct=True)
        ) * 15
    
        score["insulin_effectiveness_score"] = (
            score["tir_score"] +
            score["stability_score"] +
            score["hypo_safety_score"] +
            score["activity_score"]
        ).clip(0, 100)
    
        score = score.sort_values(
            "insulin_effectiveness_score",
            ascending=False
        )
    
        st.dataframe(score, use_container_width=True)
    
        fig = px.bar(
            score,
            x="patient_id",
            y="insulin_effectiveness_score",
            color="insulin_effectiveness_score",
            title="0–100 Insulin Effectiveness Score"
        )
    
        st.plotly_chart(fig, use_container_width=True)

# ===================================================
# KEY TAKEAWAYS
# ===================================================

elif menu == "Key Takeaways":

    st.title("📌 Key Takeaways")

    st.markdown("""
    <div class='insight-box'>
    <b>Key Clinical Findings:</b><br><br>
        ✅ Higher physical activity improves glucose stability<br/><br/>
        ✅ Missed bolus events significantly increase post-meal spikes<br><br>
        ✅ High glucose variability predicts future instability<br><br>
        ✅ Sleep duration influences next-day glucose response<br><br>
        ✅ Predictive AI can identify high-risk patients early<br><br>
        ✅ Prescriptive analytics supports personalized intervention
    </div>
    """, unsafe_allow_html=True)

# ===================================================
# CONCLUSIONS
# ===================================================

elif menu == "Conclusions":

    st.title("✅ Conclusions")
    st.markdown("""
    <div class='insight-box'>
        <b>Key Final Conclusion</b><br>
        This dashboard successfully combines descriptive,predictive, and prescriptive analytics into one unified healthcare platform.<br>
        The dashboard demonstrates how machine learning,CGM analytics, insulin intelligence,
        and behavioral monitoring can improve diabetes management and proactive care.<br><br>
        This platform supports:<br><br>
        ✅ early risk stratification<br><br>
        ✅ personalized intervention<br><br>
        ✅ glucose stability optimization<br><br>
        ✅ intelligent insulin decision support<br><br>
        ✅ real-time diabetes monitoring<br>
    </div>
    """, unsafe_allow_html=True)

   
