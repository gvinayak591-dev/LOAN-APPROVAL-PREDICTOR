"""
Loan Approval Predictor - Flask App
-------------------------------------
Name       : Vinayak Gautam
Enroll No  : 2503051240158
Division   : 2B3
Batch      : 3rd
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age            = int(request.form['age'])
        income         = float(request.form['income'])
        credit_score   = int(request.form['credit_score'])
        loan_amount    = float(request.form['loan_amount'])
        employed       = int(request.form['employed'])
        existing_loans = int(request.form['existing_loans'])

        features    = np.array([[age, income, credit_score, loan_amount, employed, existing_loans]])
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        label       = "APPROVED" if prediction == 1 else "REJECTED"
        confidence  = round(max(probability) * 100, 1)
        is_approved = prediction == 1

        return render_template('index.html',
            label=label, confidence=confidence, is_approved=is_approved,
            age=age, income=income, credit_score=credit_score,
            loan_amount=loan_amount, employed=employed, existing_loans=existing_loans
        )
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
