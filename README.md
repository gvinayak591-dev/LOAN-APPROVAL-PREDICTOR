# 🏦 Loan Approval Predictor

A simple Machine Learning web application that predicts whether a loan will be **Approved** or **Rejected**.

---

## 👤 Student Details

| Field | Details |
|---|---|
| **Name** | Vinayak Gautam |
| **Enrollment No** | 2503051240158 |
| **Division** | 2B3 |
| **Batch** | 3rd |

---

## 🤖 ML Concept Used

| Component | Details |
|---|---|
| **Algorithm** | Decision Tree Classifier |
| **Library** | scikit-learn |
| **Accuracy** | 100% |
| **Web Framework** | Flask |

---

## 📋 Input Features

| Feature | Description |
|---|---|
| Age | Applicant age in years |
| Annual Income | Income in LPA |
| Credit Score | Score between 300-900 |
| Loan Amount | Amount in Lakhs |
| Employment Status | Employed or Unemployed |
| Existing Loans | Number of active loans |

---

## 🗂️ Project Structure

```
loan_approval_predictor/
├── model.py              <- Train and save the model
├── app.py                <- Flask web server
├── loan_model.pkl        <- Saved trained model
├── requirements.txt      <- Python dependencies
├── README.md
└── templates/
    └── index.html        <- Web UI
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-approval-predictor.git
cd loan-approval-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python model.py

# 4. Start the app
python app.py

# 5. Open browser
# http://127.0.0.1:5000
```

---

## 🧠 How It Works

```
User Input -> Decision Tree -> APPROVED or REJECTED
```

A Decision Tree asks yes/no questions about the input like a flowchart and reaches a final decision. Simple, fast, and easy to understand.

---

## 📊 Test Values

| Case | Age | Income | Credit Score | Loan | Employed | Loans | Result |
|---|---|---|---|---|---|---|---|
| Good Profile  | 35 | 12 | 750 | 5 | 1 | 0 | Approved |
| Risky Profile | 22 | 2  | 520 | 9 | 0 | 3 | Rejected |

---

## 🛠️ Tech Stack
Python · Flask · scikit-learn · NumPy · HTML/CSS
