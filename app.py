
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
