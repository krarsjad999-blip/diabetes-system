# IntelligentDiabetesDetectionSystem
# Author: Senior AI Developer
# Requirements: streamlit, pandas, numpy, scikit-learn, joblib

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES = COLUMNS[:-1]

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())
    return df


def prepare_features(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y


def save_artifacts(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


def train_model(df):
    X, y = prepare_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    save_artifacts(model, scaler)
    return model, scaler, metrics


def load_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None


def predict_diabetes(input_data, model, scaler):
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"


st.set_page_config(page_title="Diabetes Detection", layout="centered")
st.title("Intelligent Diabetes Detection System")
st.markdown(
    "Use a trained logistic regression model to predict whether a patient is likely to have diabetes. "
    "The model is cached after training and loaded automatically on subsequent runs."
)


df = load_dataset()

with st.expander("Dataset preview"):
    st.dataframe(df.head())
    st.write("Rows:", len(df))
    st.write("Columns:", df.columns.tolist())

model, scaler = load_artifacts()
metrics = None

if model is None or scaler is None:
    st.warning("No saved model found. Training a new model now.")
    model, scaler, metrics = train_model(df)
    st.success("Model trained and saved successfully.")
else:
    st.success("Loaded trained model from disk.")

if st.button("Retrain model"):
    model, scaler, metrics = train_model(df)
    st.success("Model retrained and saved successfully.")

if metrics is None:
    X, y = prepare_features(df)
    X_scaled = scaler.transform(X)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

with st.expander("Model evaluation"):
    st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
    st.write("**Confusion matrix:**")
    st.write(pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    st.write("**Classification report:**")
    st.dataframe(pd.DataFrame(metrics["classification_report"]).transpose())

st.sidebar.header("Patient information")
user_input = {}
for feature in FEATURE_NAMES:
    default = float(df[feature].median())
    if feature in ["Pregnancies", "Age"]:
        user_input[feature] = st.sidebar.number_input(feature, min_value=0.0, value=default, step=1.0)
    else:
        user_input[feature] = st.sidebar.number_input(feature, min_value=0.0, value=default)

if st.sidebar.button("Predict"):
    result = predict_diabetes(user_input, model, scaler)
    st.write(f"### Prediction: {result}")
    st.write("#### Provided values")
    st.json(user_input)

st.sidebar.markdown(
    "---\n"
    "If you change the dataset or want to update the model, click **Retrain model**."
)
