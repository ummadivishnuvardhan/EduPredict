import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import DATA_PATH, EARLY_FEATURES, LATE_FEATURES, load_data
from src.predict import load_model, predict_grades, risk_category, validate_batch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
st.set_page_config(page_title="EduPredict", page_icon="📘", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#3478F6", "#16A6A1", "#F4A340", "#8B6CE8"]

st.markdown("""
<style>
  :root { --ink:#17243a; --muted:#64748b; --line:#e4eaf2; --blue:#2563eb; }
  .stApp { background:#fff; color:var(--ink); }
  [data-testid="stAppViewContainer"] { background:#fff; }
  [data-testid="stHeader"] { background:rgba(255,255,255,.96); }
  .block-container { max-width:1400px; padding-top:2.1rem; padding-bottom:3rem; }
  [data-testid="stSidebar"] { background:#f8fafc; border-right:1px solid var(--line); }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label { color:var(--ink) !important; }
  h1,h2,h3,h4 { color:var(--ink) !important; letter-spacing:-.025em; }
  p,li,label,small,[data-testid="stCaptionContainer"] { color:#42536b; }
  .hero { padding:0; margin:0 0 1.15rem 0; background:transparent; border:0; box-shadow:none; }
  .hero h1 { color:var(--ink) !important; font-size:2.5rem; margin:0 0 .45rem 0; }
  .hero p { color:#475569 !important; font-size:1.08rem; margin:0; }
  .eyebrow { color:#64748b; text-transform:uppercase; letter-spacing:.1em; font-size:.74rem; font-weight:700; margin-bottom:.5rem; }
  .page-heading { margin:0 0 1.45rem 0; }
  .page-heading h1 { font-size:2.35rem; margin:0 0 .35rem 0; }
  .page-heading p { color:#526177; font-size:1.02rem; margin:0; }
  [data-testid="stMetric"] { background:#fff; padding:1.05rem 1.2rem; border:1px solid var(--line); border-radius:14px; box-shadow:0 3px 12px rgba(23,36,58,.035); }
  [data-testid="stMetricLabel"] p { color:#64748b !important; font-size:.87rem; }
  [data-testid="stMetricValue"] { color:#17243a !important; }
  [data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.25rem; }
  [data-testid="stAlert"] { border-radius:12px; }
  div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button { background:#2563eb; border-color:#2563eb; border-radius:10px; font-weight:650; padding:.55rem 1.1rem; }
  div.stButton > button[kind="primary"] p, div[data-testid="stFormSubmitButton"] button p { color:#fff !important; }
  [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
  .support-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; height:100%; }
  .support-card h3 { font-size:1.02rem; margin:.1rem 0 .45rem 0; }
  .support-card p { color:#64748b; margin:0; line-height:1.5; }
  .workflow-card { min-height:170px; }
  .workflow-card h3 { margin:.1rem 0 .45rem 0; }
  .workflow-card p { color:#64748b; line-height:1.55; min-height:48px; }
  [data-testid="stRadio"] label p { color:#24344b !important; }
  div[data-testid="stButton"] button { border-radius:9px; font-weight:600; }
  [data-testid="stHorizontalBlock"] { gap:1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_evaluation():
    path = MODEL_DIR / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def get_model(stage: str):
    return load_model(MODEL_DIR / f"{stage}.joblib")


def features_for(stage: str):
    return EARLY_FEATURES if stage == "early" else LATE_FEATURES


def page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"<div class='page-heading'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def navigate_to(page_name: str) -> None:
    st.session_state["navigation_page"] = page_name


FIELD_LABELS = {
    "G1": "First-period grade (G1)", "G2": "Second-period grade (G2)",
    "studytime": "Weekly study time", "absences": "School absences",
    "failures": "Previous class failures", "schoolsup": "Extra school support",
    "famsup": "Family study support", "paid": "Paid subject classes",
    "activities": "Extracurricular activities", "nursery": "Attended nursery school",
    "higher": "Plans for higher education", "internet": "Internet access at home",
    "traveltime": "Travel time to school",
}


def risk_note(row: pd.Series) -> str:
    notes = []
    if row.get("G1", 20) < 10:
        notes.append("first-period grade below 10")
    if row.get("G2", 20) < 10:
        notes.append("second-period grade below 10")
    if row.get("failures", 0) > 0:
        notes.append("previous class failures reported")
    if row.get("absences", 0) >= 10:
        notes.append("10 or more absences")
    if row.get("schoolsup") == "yes":
        notes.append("extra educational support is already reported")
    return ", ".join(notes) if notes else "No selected indicator crossed the review prompts."


st.sidebar.title("📘 EduPredict")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Individual prediction", "Batch prediction", "Model evaluation", "Data insights", "About & limitations"],
    key="navigation_page",
)
st.sidebar.caption("Prototype • UCI Mathematics data")

if page == "Home":
    st.markdown("""
    <div class='hero'>
      <h1>Student performance</h1>
      <p>Explore grade estimates and academic indicators for educator review.</p>
    </div>
    """, unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("<div class='workflow-card'><h3>Individual prediction</h3><p>Enter one set of academic indicators and review an estimated final grade.</p></div>", unsafe_allow_html=True)
            st.button("Open individual prediction", type="primary", width="stretch", on_click=navigate_to, args=("Individual prediction",))
    with right:
        with st.container(border=True):
            st.markdown("<div class='workflow-card'><h3>Batch prediction</h3><p>Validate a CSV of supported model inputs and download the estimates.</p></div>", unsafe_allow_html=True)
            st.button("Open batch prediction", width="stretch", on_click=navigate_to, args=("Batch prediction",))
    st.divider()
    st.subheader("Explore the project")
    browse = st.columns(3)
    for column, label, target in zip(
        browse,
        ["Model evaluation", "Data insights", "About and limitations"],
        ["Model evaluation", "Data insights", "About & limitations"],
    ):
        with column:
            st.button(label + " →", width="stretch", on_click=navigate_to, args=(target,))
    st.caption("Estimates are advisory and should be considered alongside educator judgment.")

elif page == "Individual prediction":
    page_heading("STUDENT SUPPORT", "Individual estimate", "Estimate a final grade from information available at a selected point in the year.")
    stage_label = st.radio("Prediction point", ["Earlier stage (before G2)", "Later stage (after G2)"], horizontal=True)
    stage = "early" if stage_label.startswith("Earlier") else "late"
    threshold = st.slider("Illustrative academic review threshold (grade out of 20)", 0.0, 20.0, 10.0, 0.5)
    st.caption("Align this threshold with your institution’s grading rules. It is a review prompt, not a pass/fail decision.")
    with st.form("individual_form", border=False):
        st.markdown("#### Student learning indicators")
        st.caption("Enter academic and support information only. Do not enter names or student IDs.")
        cols = st.columns(3)
        values = {}
        for index, feature in enumerate(features_for(stage)):
            with cols[index % 3]:
                if feature in {"G1", "G2"}:
                    values[feature] = st.number_input(FIELD_LABELS[feature], min_value=0.0, max_value=20.0, value=10.0, step=1.0)
                elif feature == "studytime":
                    values[feature] = st.select_slider("Weekly study time", options=[1, 2, 3, 4], value=2, format_func=lambda x: {1:"<2 hours",2:"2–5 hours",3:"5–10 hours",4:">10 hours"}[x])
                elif feature == "absences":
                    values[feature] = st.number_input(FIELD_LABELS[feature], 0, 93, 0)
                elif feature == "failures":
                    values[feature] = st.selectbox(FIELD_LABELS[feature], [0, 1, 2, 4], index=0, format_func=lambda x: "3 or more" if x == 4 else str(x))
                elif feature == "traveltime":
                    values[feature] = st.selectbox(FIELD_LABELS[feature], [1, 2, 3, 4], index=1, format_func=lambda x: {1:"<15 min",2:"15–30 min",3:"30–60 min",4:">60 min"}[x])
                else:
                    values[feature] = st.selectbox(FIELD_LABELS[feature], ["no", "yes"])
        submitted = st.form_submit_button("Estimate grade", type="primary")
    if submitted:
        try:
            row = pd.DataFrame([values], columns=features_for(stage))
            grade = float(predict_grades(get_model(stage), row)[0])
            risk = risk_category(grade, threshold)
            m1, m2 = st.columns(2)
            m1.metric("Estimated final grade", f"{grade:.1f} / 20")
            m2.metric("Teacher review prompt", risk)
            st.subheader("Input indicators to review")
            st.write(risk_note(pd.Series(values)))
            st.write("Consider a supportive check-in, reviewing recent learning needs, and offering relevant academic resources. Interpret this estimate alongside the student’s own context and teacher judgment.")
            st.warning("This estimate is not a diagnosis or definitive judgment. Regression output is not a calibrated probability of failure.")
        except Exception as exc:
            st.error(f"Could not create an estimate: {exc}")

elif page == "Batch prediction":
    page_heading("BATCH WORKFLOW", "Batch predictions", "Check a group of records without storing uploads or collecting identifiers.")
    stage_label = st.radio("Prediction point", ["Earlier stage (before G2)", "Later stage (after G2)"], horizontal=True, key="batch_stage")
    stage = "early" if stage_label.startswith("Earlier") else "late"
    threshold = st.slider("Illustrative review threshold", 0.0, 20.0, 10.0, 0.5, key="batch_threshold")
    st.caption("Upload a CSV with the model feature columns. Do not include names, IDs, or other identifying information.")
    st.code(",".join(features_for(stage)), language="text")
    upload = st.file_uploader("Choose CSV", type=["csv"])
    if upload:
        try:
            incoming = pd.read_csv(upload)
            clean = validate_batch(incoming, features_for(stage))
            st.write(f"Validated {len(clean)} rows")
            st.dataframe(clean.head(10), width="stretch")
            result = clean.copy()
            result["predicted_G3"] = predict_grades(get_model(stage), clean).round(2)
            result["risk_category"] = result["predicted_G3"].map(lambda x: risk_category(x, threshold))
            st.dataframe(result.head(20), width="stretch")
            st.download_button("Download predictions CSV", result.to_csv(index=False).encode("utf-8"), "edupredict_results.csv", "text/csv")
            st.caption("Uploaded rows are processed in memory by this app and are not stored by EduPredict.")
        except Exception as exc:
            st.error(f"Upload could not be processed: {exc}")

elif page == "Model evaluation":
    page_heading("MODEL QUALITY", "Evaluation and evidence", "Compare held-out model results before interpreting any estimate.")
    try:
        evaluation = get_evaluation()
        stage_label = st.radio("Evaluation stage", ["Earlier stage", "Later stage"], horizontal=True, key="eval_stage")
        stage = "early" if stage_label.startswith("Earlier") else "late"
        details = evaluation[stage]
        st.caption(f"{evaluation['rows']} records · {details['rows_train']} train / {details['rows_test']} held-out test · {evaluation['split']}")
        st.write(f"Selected on held-out MAE: **{details['selected_model']}**")
        table = pd.DataFrame(details["metrics"]).T.rename(columns={"mae":"MAE", "rmse":"RMSE", "r2":"R²"})
        st.dataframe(table.style.format("{:.3f}"), width="stretch")
        holdout = pd.read_csv(MODEL_DIR / f"{stage}-holdout.csv")
        left, right = st.columns(2)
        with left:
            fig = px.scatter(holdout, x="actual", y="predicted", labels={"actual":"Actual G3", "predicted":"Predicted G3"}, title="Held-out actual vs predicted")
            fig.add_shape(type="line", x0=0, y0=0, x1=20, y1=20, line={"dash":"dash", "color":"#64748b"})
            st.plotly_chart(fig, width="stretch")
        with right:
            residual = holdout.assign(error=holdout.predicted - holdout.actual)
            st.plotly_chart(px.histogram(residual, x="error", nbins=20, title="Held-out prediction errors", labels={"error":"Predicted − actual grade"}), width="stretch")
        model = get_model(stage)
        if details["selected_model"] == "RandomForestRegressor":
            transformed = model.named_steps["preprocess"].get_feature_names_out()
            importance = pd.DataFrame({"feature": transformed, "importance": model.named_steps["model"].feature_importances_})
            importance = importance.nlargest(15, "importance").sort_values("importance")
            st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="RandomForest split importance (top 15)", labels={"importance":"Relative split importance", "feature":"Transformed input"}), width="stretch")
            st.caption("Split importance shows how the fitted forest used features; it does not show causes or how a teacher should act.")
        else:
            st.caption("Feature importance is not displayed for this selected estimator because a simple feature ranking would be misleading after preprocessing.")
        st.info("All feature effects are associations in a small observational dataset, not causal evidence.")
    except Exception as exc:
        st.error(f"Evaluation files are unavailable. Run `python -m src.train` after installing requirements. Details: {exc}")

elif page == "Data insights":
    page_heading("SOURCE DATA", "Explore the dataset", "Understand the historical grade, attendance, study-time, and failure patterns.")
    try:
        data = get_data()
        st.caption(f"UCI Mathematics dataset · {len(data)} historical student records")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(data, x="G3", nbins=21, title="Final math grade (G3)", labels={"G3":"Grade out of 20"}), width="stretch")
        c2.plotly_chart(px.histogram(data, x="absences", nbins=25, title="School absences", labels={"absences":"Number of absences"}), width="stretch")
        c1.plotly_chart(px.histogram(data, x="studytime", title="Weekly study-time category", labels={"studytime":"Category (1 <2h, 2 2–5h, 3 5–10h, 4 >10h)"}), width="stretch")
        c2.plotly_chart(px.histogram(data, x="failures", title="Previous class failures", labels={"failures":"Reported failure count/category"}), width="stretch")
    except Exception as exc:
        st.error(f"Dataset could not be loaded: {exc}")

else:
    page_heading("RESPONSIBLE USE", "About and limitations", "What this prototype can support, and where its evidence ends.")
    st.write("EduPredict is a classroom support prototype using the UCI Student Performance dataset, mathematics subset. It estimates the 0–20 final grade (G3) from selected earlier academic and support indicators.")
    st.write("The earlier model excludes G2 and G3 from its inputs. The later model includes G2, a strong predictor measured later in the school year. Neither model provides a calibrated failure probability.")
    st.write("The data cover 395 Portuguese secondary-school math records. The random held-out split is reproducible, but this is not external validation; results may not generalize to other schools, times, or populations. Do not use outputs to penalize, rank, or exclude students. Use them only as one prompt for supportive human review.")
    st.write("Sex, age, and family background are omitted from model inputs. Uploaded data are processed in memory and are not intentionally persisted by this application. Avoid uploading identifying information.")
    st.write("Dataset: Cortez (2008), UCI Machine Learning Repository, DOI 10.24432/C5TG7T, CC BY 4.0. See data/README.md for attribution and details.")
    st.subheader("What each section does")
    st.markdown("""
    - **Home:** choose a workflow or open another section.
    - **Individual prediction:** enter one student's supported academic indicators and estimate a final grade.
    - **Batch prediction:** validate a CSV with the displayed feature headers, estimate grades for its rows, and download the results. Names and IDs are not accepted.
    - **Model evaluation:** compare DummyRegressor, LinearRegression, and RandomForestRegressor on the held-out test set, including error charts and RandomForest split importance.
    - **Data insights:** explore grade, absence, study-time, and prior-failure distributions in the historical dataset.
    - **About and limitations:** review the intended use, data source, and limits of this prototype.
    """)
    st.subheader("Input options")
    st.markdown("""
    | Input | Meaning |
    |---|---|
    | First/second-period grade | G1 and G2, each on the dataset's 0–20 grade scale. G2 appears only in the later model. |
    | Weekly study time | 1 = under 2 hours; 2 = 2–5 hours; 3 = 5–10 hours; 4 = over 10 hours. |
    | School absences | Count from 0 to 93 in the source dataset. |
    | Previous class failures | 0, 1, 2, or 3 or more. The source codes 3 or more as 4. |
    | Travel time | Under 15 minutes; 15–30 minutes; 30–60 minutes; or over 60 minutes. |
    | Yes/no indicators | Extra school support, family study support, paid subject classes, extracurricular activities, nursery attendance, higher-education plans, and internet access at home. |
    """)
    st.subheader("How to understand a student estimate")
    st.markdown("""
    - **Estimated final grade:** a regression estimate on the dataset's 0–20 scale. It is not a certainty or a probability of passing.
    - **Review recommended:** the estimate is below the threshold currently selected on the prediction page. The default 10/20 is illustrative; confirm the threshold with your institution.
    - **At/above illustrative threshold:** the estimate meets or exceeds that selected threshold. This does not guarantee success.
    - **Input indicators:** low period grades, previous failures, absences, and existing support may be useful prompts for a teacher conversation. They do not explain why a student is struggling and should not be used to blame or rank them.
    - **Earlier vs later:** the earlier model does not use G2. The later model includes G2, so its estimate is made later in the school year and often has more recent grade information.
    - **MAE and RMSE:** both are average error measures in grade points; lower is better. RMSE penalizes large errors more strongly.
    - **R²:** compares predictions with a mean-only baseline on this test split. Higher is better; a negative score means that model performed worse than predicting the test-set mean.
    """)
