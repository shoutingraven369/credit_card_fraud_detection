import streamlit as st
import numpy as np
import pandas as pd
import joblib

xgb_model = joblib.load("xgb_model.pkl")
df = pd.read_csv("creditcard.csv")

st.title("Credit Card Fraud Detection")

st.write("You can either upload a row or use a sample row from the dataset.")


if st.button("Use a random real transaction"):
    sample = df[df["Class"] == 0].sample(1).drop(columns=["Class"]).values
    st.write("Sample used for prediction:", sample)

    prediction = xgb_model.predict(sample)[0]
    proba = xgb_model.predict_proba(sample)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ Transaction Safe (Confidence: {proba:.2%})")


