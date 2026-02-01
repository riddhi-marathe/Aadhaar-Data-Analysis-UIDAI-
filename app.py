import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Aadhaar Intelligence System", layout="wide")
# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Welcome"
st.title("Aadhaar Intelligence & Analytics Platform")
st.markdown("**Trends • Anomalies • Predictions • Policy Insights**")

# ---------------- PATHS ----------------
ENROLMENT_FOLDER = r"C:\Users\afroz alam\Documents\aadhaar_app\api_data_aadhar_enrolment"
BIOMETRIC_FOLDER = r"C:\Users\afroz alam\Documents\aadhaar_app\api_data_aadhar_biometric"
DEMOGRAPHIC_FOLDER = r"C:\Users\afroz alam\Documents\aadhaar_app\api_data_aadhar_demographic"

# ---------------- UTILS ----------------
@st.cache_data(show_spinner=False)
def load_dataset(folder):
    csvs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))

    if not csvs:
        return pd.DataFrame()

    frames = []
    bar = st.progress(0)
    for i, f in enumerate(csvs):
        frames.append(pd.read_csv(f, low_memory=False))
        bar.progress((i + 1) / len(csvs))
        time.sleep(0.05)

    return pd.concat(frames, ignore_index=True)

def sample_df(df, n=50000):
    return df.sample(n=min(len(df), n), random_state=42)

def detect_anomalies(series):
    mean = series.mean()
    std = series.std()
    return (series - mean).abs() > 2 * std

# ---------------- FILTERS ----------------
def apply_filters(df):
    st.sidebar.header("🔎 Filters")

    if "state" in df.columns:
        states = st.sidebar.multiselect(
            "State",
            sorted(df["state"].dropna().unique())
        )
        if states:
            df = df[df["state"].isin(states)]

    if "gender" in df.columns:
        genders = st.sidebar.multiselect(
            "Gender",
            sorted(df["gender"].dropna().unique())
        )
        if genders:
            df = df[df["gender"].isin(genders)]

    if "age" in df.columns:
        min_age = int(df["age"].min())
        max_age = int(df["age"].max())
        age_range = st.sidebar.slider(
            "Age Range",
            min_age,
            max_age,
            (min_age, max_age)
        )
        df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

    return df

# ---------------- MAP ----------------
def india_map(df, metric, title):
    if "state" not in df.columns:
        return

    map_df = df.groupby("state")[metric].sum().reset_index()
    map_df["anomaly"] = detect_anomalies(map_df[metric])

    fig = px.choropleth(
        map_df,
        geojson="https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson",
        locations="state",
        featureidkey="properties.NAME_1",
        color=metric,
        hover_name="state",
        color_continuous_scale="Turbo",
        title=title
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=550)

    st.plotly_chart(fig, use_container_width=True)

    anomalies = map_df[map_df["anomaly"]]
    if not anomalies.empty:
        st.error("⚠ Anomalies detected in: " + ", ".join(anomalies["state"]))

# ---------------- VISUALS ----------------
def visuals(df, name):
    df_s = sample_df(df)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found")
        return None

    metric = st.selectbox("Select Metric", num_cols)

    st.subheader("📊 Distribution")
    st.plotly_chart(px.histogram(df_s, x=metric), use_container_width=True)

    if "state" in df.columns:
        st.subheader("📍 State-wise Comparison (Red = Anomaly)")
        state_df = df.groupby("state")[metric].sum().reset_index()
        state_df["anomaly"] = detect_anomalies(state_df[metric])

        st.plotly_chart(
            px.bar(
                state_df,
                x="state",
                y=metric,
                color="anomaly",
                color_discrete_map={True: "blue", False: "red"}
            ),
            use_container_width=True
        )

        st.subheader("🗺️ India Map")
        india_map(df, metric, f"{name}: {metric}")

    if len(num_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        st.plotly_chart(
            px.imshow(df[num_cols].corr(), text_auto=True),
            use_container_width=True
        )

        st.subheader("🧠 Cluster Analysis")
        km = KMeans(n_clusters=4, random_state=42)
        df_s["cluster"] = km.fit_predict(df_s[num_cols[:2]].dropna())

        st.plotly_chart(
            px.scatter(
                df_s,
                x=num_cols[0],
                y=num_cols[1],
                color="cluster"
            ),
            use_container_width=True
        )

    if "year" in df.columns:
        st.subheader("📈 Trend & Prediction")
        t = df.groupby("year")[metric].sum().reset_index()
        X = t["year"].values.reshape(-1, 1)
        y = t[metric].values
        model = LinearRegression().fit(X, y)
        future = np.arange(X.max() + 1, X.max() + 6).reshape(-1, 1)
        pred = model.predict(future)

        fig = px.line(t, x="year", y=metric)
        fig.add_scatter(
            x=future.flatten(),
            y=pred,
            mode="lines+markers",
            name="Prediction",
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig, use_container_width=True)

    return metric

# ---------------- INSIGHTS ----------------
def generate_insights(df, name, metric):
    insights = []
    insights.append(f"Total records analyzed: {len(df)}")

    if "state" in df.columns:
        insights.append(f"Highest activity observed in {df['state'].value_counts().idxmax()}")

    if "gender" in df.columns:
        insights.append(f"Majority gender: {df['gender'].value_counts().idxmax()}")

    insights.append(f"Average {metric}: {df[metric].mean():.2f}")
    insights.append("Anomalies indicate operational or reporting irregularities.")
    insights.append("Predictive trends support proactive policy planning.")

    return insights

# ---------------- PDF ----------------
def generate_pdf(insights, name):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)

    text = c.beginText(40, 800)
    text.setFont("Helvetica", 11)
    text.textLine(f"Aadhaar {name} Insights Report")
    text.textLine("=" * 50)
    text.textLine("")

    for i in insights:
        text.textLine(f"- {i}")

    c.drawText(text)
    c.save()
    return tmp.name
# ---------------- welcome page ----------------

def welcome_page():
    st.warning("Welcome to Aadhaar Intelligence & Analytics System where numbers are turned into knowledge and knowledge into action")   # temporary debug line

    st.markdown(
        """
        <h1 style='text-align:center;'>Aadhaar Intelligence & Analytics System</h1>
        <h4 style='text-align:center; color:gray;'>
        Large-Scale Data Analytics • Anomaly Detection • Policy Insights
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 Problem Statement")
        st.write(
            """
            India’s Aadhaar system generates massive datasets across enrolment,
            biometrics, and demographics. Extracting meaningful insights,
            detecting anomalies, and predicting trends is challenging.
            """
        )

        st.subheader("🎯 Objective")
        st.write(
            """
            • Analyze enrolment, biometric, and demographic datasets  
            • Detect anomalies and irregular patterns  
            • Visualize trends interactively  
            • Support policy and operational decisions  
            """
        )

    with col2:
        st.subheader("🚀 Key Features")
        st.markdown(
            """
            ✅ Large-scale data handling (5L+ records)  
            ✅ Interactive dashboards  
            ✅ Anomaly detection  
            ✅ Trend forecasting  
            ✅ Auto-generated reports  
            """
        )

        st.subheader("📊 Dataset Scale")
        st.info(
            """
            • Enrolment Data  
            • Biometric Updates  
            • Demographic Updates  
            """
        )

    st.markdown("---")

    colA, colB, colC = st.columns(3)
    colA.metric("Datasets", "3")
    colB.metric("Records", "5L+")
    colC.metric("Visualizations", "10+")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Enter Dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()


# ---------------- WORKFLOW ----------------
def workflow(path, name):
    df = load_dataset(path)
    if df.empty:
        st.warning("No data found")
        return

    df = apply_filters(df)

    st.success(f"{len(df):,} records loaded")

    with st.expander("🔍 View Sample Records"):
        st.dataframe(df.head(5))

    metric = visuals(df, name)

    st.subheader("📝 Auto-generated Insights")
    insights = generate_insights(df, name, metric)
    for i in insights:
        st.markdown("✔ " + i)

    pdf = generate_pdf(insights, name)
    with open(pdf, "rb") as f:
        st.download_button(
            "📄 Download PDF Report",
            f,
            file_name=f"{name}_Insights.pdf"
        )

# ---------------- TABS ----------------
if st.session_state.page == "Welcome":
    welcome_page()
else:
    tabs = st.tabs(["Enrolment", "Biometric", "Demographic"])

    with tabs[0]:
        workflow(ENROLMENT_FOLDER, "Enrolment")

    with tabs[1]:
        workflow(BIOMETRIC_FOLDER, "Biometric")

    with tabs[2]:
        workflow(DEMOGRAPHIC_FOLDER, "Demographic")
