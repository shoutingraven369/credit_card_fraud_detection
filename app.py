import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load the model
xgb_model = joblib.load("xgb_model.pkl")

st.title("Credit Card Fraud Detection")

st.write("Enter transaction features to check if it's fraudulent.")

# Check if CSV exists for sample data
csv_exists = os.path.exists("creditcard.csv")

if csv_exists:
    df = pd.read_csv("creditcard.csv")
    
    if st.button("Use a random real transaction"):
        sample = df[df["Class"] == 0].sample(1).drop(columns=["Class"]).values
        st.write("Sample used for prediction:", sample)

        prediction = xgb_model.predict(sample)[0]
        proba = xgb_model.predict_proba(sample)[0][1]

        if prediction == 1:
            st.error(f"Fraud Detected! (Confidence: {proba:.2%})")
        else:
            st.success(f"Transaction Safe (Confidence: {1-proba:.2%})")

st.markdown("---")
st.subheader("Manual Input")
st.write("Enter the 30 feature values (V1-V28, Time, Amount) separated by commas:")

feature_input = st.text_area(
    "Features (30 values, comma-separated):",
    placeholder="Example: -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, 0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40, 0.25, -0.01, 0.27, -0.11, 0.06, 0.12, -0.18, 0.13, -0.02, -2.19, 0.22"
)

if st.button("Predict"):
    try:
        features = [float(x.strip()) for x in feature_input.split(",")]
        
        if len(features) != 30:
            st.error(f"Please enter exactly 30 features. You entered {len(features)}.")
        else:
            sample = np.array(features).reshape(1, -1)
            prediction = xgb_model.predict(sample)[0]
            proba = xgb_model.predict_proba(sample)[0][1]

            if prediction == 1:
                st.error(f"Fraud Detected! (Confidence: {proba:.2%})")
            else:
                st.success(f"Transaction Safe (Confidence: {1-proba:.2%})")
    except ValueError:
        st.error("Invalid input. Please enter numeric values separated by commas.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("**Note:** This model uses XGBoost trained on credit card transaction data.")
