import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("Libraries imported successfully.")

## 1. LOAD AND PREPARE THE DATA
# The dataset is from the UCI Machine Learning Repository.
# It uses '?' for missing values.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Define column names as they are not in the file
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load data into a pandas DataFrame
df = pd.read_csv(url, header=None, names=column_names, na_values="?")
print("Dataset loaded successfully.")

# --- Data Cleaning ---
# For simplicity, we'll drop rows with any missing values.
df.dropna(inplace=True)

# The 'target' column is 0 for no disease and 1, 2, 3, 4 for disease.
# We'll convert it to a binary classification problem: 0 (no disease) vs. 1 (disease).
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Display data info
print("\nDataset Information:")
df.info()
print("\nFirst 5 rows of the dataset:")
print(df.head())


## 2. DEFINE FEATURES (X) AND TARGET (y) AND SPLIT DATA
X = df.drop('target', axis=1)
y = df['target']

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


## 3. FEATURE SCALING
# Scale features to have zero mean and unit variance.
# This is important for algorithms like SVM and Logistic Regression.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeatures scaled successfully.")


## 4. INITIALIZE AND TRAIN MODELS
# We'll test four different classification algorithms.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "xgboost": xgb.XGBClassifier(eval_metric='logloss', random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")


## 5. DETAILED EVALUATION OF THE BEST PERFORMING MODEL
# Let's assume Random Forest performed well and look at its detailed report.
print("\n--- Detailed Report for Random Forest ---")
rf_model = models["Random Forest"]
y_pred_rf = rf_model.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred_rf, target_names=["No Disease", "Disease"]))