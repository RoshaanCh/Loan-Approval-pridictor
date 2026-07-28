import os

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run train_model.py first.")
        st.stop()

    return joblib.load(MODEL_PATH)


def prepare_input_data(feature_data, feature_columns):
    input_df = pd.DataFrame([feature_data])
    input_df = pd.get_dummies(input_df, columns=["education", "self_employed"], drop_first=True)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df


def main():
    st.title("Loan Approval Prediction")
    st.write("Enter the applicant details below and click Predict.")

    saved_data = load_model()

    if isinstance(saved_data, dict):
        model = saved_data["model"]
        feature_columns = saved_data["feature_columns"]
    else:
        model = saved_data
        feature_columns = None

    with st.form("loan_form"):
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        income_annum = st.number_input("Annual Income", min_value=0, value=5000000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=15000000)
        loan_term = st.number_input("Loan Term (months)", min_value=1, value=12)
        cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900, value=700)
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=1000000)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=1000000)
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=1000000)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=1000000)

        submitted = st.form_submit_button("Predict")

    if submitted:
        feature_data = {
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
        }

        if feature_columns is not None:
            input_df = prepare_input_data(feature_data, feature_columns)
        else:
            input_df = pd.DataFrame([feature_data])

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        approved_probability = probabilities[1] * 100

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

        st.write(f"Prediction confidence: {approved_probability:.2f}%")


if __name__ == "__main__":
    main()