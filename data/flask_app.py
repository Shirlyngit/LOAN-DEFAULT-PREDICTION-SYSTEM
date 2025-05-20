# File: backend/app.py (Flask API)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
with open("loan_default_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define expected feature order
expected_features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education_0', 'Education_1', 'Education_2', 'Education_3',
    'EmploymentType_0', 'EmploymentType_1', 'EmploymentType_2', 'EmploymentType_3',
    'MaritalStatus_0', 'MaritalStatus_1', 'MaritalStatus_2',
    'HasMortgage_0', 'HasMortgage_1',
    'HasDependents_0', 'HasDependents_1',
    'LoanPurpose_0', 'LoanPurpose_1', 'LoanPurpose_2', 'LoanPurpose_3', 'LoanPurpose_4',
    'HasCoSigner_0', 'HasCoSigner_1'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_vector = [data[feature] for feature in expected_features]
        prediction = int(model.predict([input_vector])[0])
        probability = float(model.predict_proba([input_vector])[0][1])

        return jsonify({"prediction": prediction, "probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
