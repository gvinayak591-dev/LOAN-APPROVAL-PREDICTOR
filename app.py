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


def get_reasons(age, income, credit_score, loan_amount, employed, existing_loans, is_approved):
    """Generate realistic reasons for approval or rejection."""
    reasons = []

    if is_approved:
        if credit_score >= 700:
            reasons.append(("✅", "Excellent credit score of " + str(credit_score)))
        elif credit_score >= 650:
            reasons.append(("✅", "Good credit score of " + str(credit_score)))
        if employed:
            reasons.append(("✅", "Stable employment status"))
        if income >= 10:
            reasons.append(("✅", "Strong annual income of ₹" + str(income) + " LPA"))
        elif income >= 5:
            reasons.append(("✅", "Adequate income of ₹" + str(income) + " LPA"))
        if existing_loans <= 1:
            reasons.append(("✅", "Low existing loan burden"))
        if loan_amount <= income * 0.8:
            reasons.append(("✅", "Loan amount is proportional to income"))
    else:
        if credit_score < 650:
            reasons.append(("❌", f"Low credit score of {credit_score} — minimum required is 650"))
        if not employed:
            reasons.append(("❌", "No stable employment — income source unclear"))
        if income < 5:
            reasons.append(("❌", f"Insufficient income of ₹{income} LPA — minimum required is ₹5 LPA"))
        if existing_loans >= 3:
            reasons.append(("❌", f"{existing_loans} existing loans — too high a debt burden"))
        if loan_amount > income * 2:
            reasons.append(("❌", f"Loan amount ₹{loan_amount}L is too high compared to income ₹{income} LPA"))
        if not reasons:
            reasons.append(("❌", "Overall risk profile does not meet bank criteria"))

    return reasons


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

        is_approved = prediction == 1
        label       = "APPROVED" if is_approved else "REJECTED"
        confidence  = round(max(probability) * 100, 1)
        reasons     = get_reasons(age, income, credit_score, loan_amount, employed, existing_loans, is_approved)

        return render_template('index.html',
            label=label, confidence=confidence,
            is_approved=is_approved, reasons=reasons,
            age=age, income=income, credit_score=credit_score,
            loan_amount=loan_amount, employed=employed,
            existing_loans=existing_loans
        )
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
