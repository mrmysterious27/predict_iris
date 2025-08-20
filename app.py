import joblib
import numpy as np
import streamlit as st

# Page config
st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Classifier")

# Load model
model = joblib.load("models/iris_best_model.joblib")
labels = ["setosa", "versicolor", "virginica"]

# Input sliders
sl = st.slider("Sepal length", 4.0, 8.0, 5.1, 0.1)
sw = st.slider("Sepal width", 2.0, 4.5, 3.5, 0.1)
pl = st.slider("Petal length", 1.0, 7.0, 1.4, 0.1)
pw = st.slider("Petal width", 0.1, 2.5, 0.2, 0.1)

# Prediction button
if st.button("Predict"):
    x = np.array([[sl, sw, pl, pw]])
    pred_idx = int(model.predict(x)[0])
    st.success(f"🌼 Prediction: **{labels[pred_idx]}**")

    try:
        proba = model.predict_proba(x)[0]
        st.subheader("Class Probabilities")
        for name, p in zip(labels, proba):
            st.write(f"- {name}: {p:.3f}")
    except Exception:
        st.info("This model does not support probability predictions.")
