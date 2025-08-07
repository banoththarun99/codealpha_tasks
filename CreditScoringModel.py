# Credit Scoring Model - Predicting Creditworthiness

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 1: Simulated Dataset (or load your real dataset)
# For demonstration, create synthetic data
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,
                           n_features=8,
                           n_informative=6,
                           n_redundant=2,
                           random_state=42,
                           weights=[0.7, 0.3])

columns = ['income', 'debt', 'payment_history', 'credit_utilization',
           'loan_amount', 'num_credit_cards', 'num_loans', 'credit_inquiries']

df = pd.DataFrame(X, columns=columns)
df['creditworthy'] = y

# Step 2: Feature Engineering
# In real datasets, feature engineering includes processing dates, handling missing values, encoding, etc.

# Step 3: Train-Test Split
X = df.drop('creditworthy', axis=1)
y = df['creditworthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Step 6: Training and Evaluation
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

# Plot ROC Curve
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()
