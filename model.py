"""
Loan Approval Predictor
-----------------------
Name       : Vinayak Gautam
Enroll No  : 2503051240158
Division   : 2B3
Batch      : 3rd
Algorithm  : Decision Tree Classifier (sklearn)
"""

import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# ── 1. Dataset ─────────────────────────────────────────────────────
# Features: [age, income(LPA), credit_score, loan_amount(L), employed(1/0), existing_loans]
# Label   : 1 = Approved  |  0 = Rejected

approved = [
    [35, 12, 750, 5, 1, 0], [28, 8, 700, 3, 1, 0], [42, 20, 800, 10, 1, 0],
    [30, 10, 720, 4, 1, 0], [38, 15, 760, 7, 1, 1], [45, 25, 820, 15, 1, 0],
    [32, 9, 710, 3, 1, 0],  [50, 30, 850, 20, 1, 0], [27, 7, 690, 2, 1, 0],
    [36, 13, 740, 6, 1, 0], [40, 18, 780, 9, 1, 1], [33, 11, 730, 5, 1, 0],
    [29, 8, 705, 3, 1, 0],  [55, 35, 860, 25, 1, 0], [37, 14, 750, 6, 1, 0],
    [43, 22, 810, 12, 1, 0],[31, 9, 715, 4, 1, 0],  [48, 28, 840, 18, 1, 0],
    [26, 6, 680, 2, 1, 0],  [39, 16, 770, 8, 1, 0], [34, 12, 735, 5, 1, 0],
    [41, 19, 785, 10, 1, 1],[28, 8, 695, 3, 1, 0],  [52, 32, 855, 22, 1, 0],
    [36, 13, 745, 6, 1, 0], [30, 10, 718, 4, 1, 0], [44, 23, 815, 13, 1, 0],
    [33, 11, 728, 5, 1, 0], [38, 15, 758, 7, 1, 0], [47, 27, 838, 17, 1, 0],
] * 5

rejected = [
    [22, 2, 550, 8, 0, 3],  [19, 1, 480, 10, 0, 4], [25, 3, 580, 12, 0, 2],
    [21, 2, 520, 9, 0, 3],  [18, 1, 450, 15, 0, 5], [23, 2, 560, 7, 0, 3],
    [20, 1, 500, 11, 0, 4], [24, 3, 570, 6, 1, 4],  [22, 2, 540, 8, 0, 2],
    [19, 1, 460, 13, 0, 5], [26, 4, 600, 10, 0, 3], [21, 2, 510, 9, 0, 4],
    [23, 3, 555, 7, 0, 2],  [20, 1, 490, 14, 0, 5], [25, 3, 575, 11, 0, 3],
    [18, 1, 440, 16, 0, 4], [22, 2, 530, 8, 0, 3],  [24, 3, 565, 9, 1, 4],
    [19, 1, 470, 12, 0, 5], [21, 2, 505, 10, 0, 3], [23, 2, 545, 7, 0, 2],
    [20, 1, 495, 13, 0, 4], [26, 4, 595, 11, 0, 3], [18, 1, 455, 15, 0, 5],
    [22, 2, 535, 8, 0, 3],  [19, 1, 465, 14, 0, 4], [25, 3, 572, 10, 0, 2],
    [21, 2, 515, 9, 0, 3],  [23, 3, 558, 7, 0, 4],  [20, 1, 488, 12, 0, 5],
] * 5

X = np.array(approved + rejected)
y = np.array([1] * len(approved) + [0] * len(rejected))

combined = list(zip(X.tolist(), y.tolist()))
np.random.shuffle(combined)
X = np.array([c[0] for c in combined])
y = np.array([c[1] for c in combined])

# ── 2. Train / Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 3. Decision Tree Classifier ────────────────────────────────────
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# ── 4. Evaluate ────────────────────────────────────────────────────
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)
print(f"\nAccuracy : {acc * 100:.2f}%")
print(classification_report(y_test, preds, target_names=['Rejected', 'Approved']))

# ── 5. Save ────────────────────────────────────────────────────────
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved: loan_model.pkl")
