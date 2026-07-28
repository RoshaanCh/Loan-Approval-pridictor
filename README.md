Loan Approval Predictor
Project Overview

This project predicts whether a loan application is likely to be approved or rejected based on the applicant's details. It is a simple machine learning project built to understand the complete ML workflow, from training a model to making predictions through a web application.

Dataset

The dataset includes the following features:

loan_id
no_of_dependents
education
self_employed
income_annum
loan_amount
loan_term
cibil_score
residential_assets_value
commercial_assets_value
luxury_assets_value
bank_asset_value
loan_status

loan_status is the target column, and loan_id is removed because it is only used as an identifier.

Model

I used a Decision Tree Classifier because it is simple, easy to understand, and works well for this type of classification problem.

Project Structure
LoanApprovalPredictor/
├── data/
├── models/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

Installation

Install the required libraries:

pip install -r requirements.txt
Run the Project

Train the model:

python train_model.py

Run the Streamlit app:

streamlit run app.py

Workflow:

Load the dataset
Clean the data
Remove loan_id
Encode categorical columns
Train the model
Test the model
Save the trained model
Use the saved model in the Streamlit app