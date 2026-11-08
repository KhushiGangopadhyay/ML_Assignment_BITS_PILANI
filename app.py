import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

TARGET = "NObeyesdad"
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

st.set_page_config(page_title="Obesity Level Classifier", page_icon="📊", layout="wide")
st.title("Obesity Level Classification - Model Explorer")
st.caption("Upload test data, pick a model, and inspect how it performs.")


@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(p) for n, p in MODEL_FILES.items()}
    le = joblib.load("model/label_encoder.joblib")
    feats = joblib.load("model/feature_columns.joblib")
    return models, le, feats


models, le, feature_cols = load_artifacts()

# ---------- Sidebar ----------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

if uploaded is None:
    st.info("Upload `test_data.csv` from the repo (it must include the "
            f"`{TARGET}` column) to get started.")
    try:
        st.subheader("Training-time comparison of all models")
        st.dataframe(pd.read_csv("model/metrics.csv"), use_container_width=True)
    except FileNotFoundError:
        pass
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in feature_cols + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()

with st.expander("Preview uploaded data"):
    st.write(f"{df.shape[0]} rows x {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

X = df[feature_cols]
y_true = le.transform(df[TARGET])
pipe = models[model_name]
y_pred = pipe.predict(X)
proba = pipe.predict_proba(X)

# ---------- Metrics ----------
st.subheader(f"Evaluation metrics - {model_name}")
try:
    auc = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
except ValueError:  # e.g. uploaded subset is missing some classes
    auc = float("nan")

vals = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": auc,
    "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "Recall": recall_score(y_true, y_pred, average="weighted"),
    "F1": f1_score(y_true, y_pred, average="weighted"),
    "MCC": matthews_corrcoef(y_true, y_pred),
}
cols = st.columns(6)
for col, (k, v) in zip(cols, vals.items()):
    col.metric(k, "N/A" if v != v else f"{v:.3f}")

# ---------- Confusion matrix + report ----------
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, labels=range(len(le.classes_)),
        target_names=le.classes_, output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("All models at a glance (training-time test split)")
st.dataframe(pd.read_csv("model/metrics.csv"), use_container_width=True)
