import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

mlflow.set_experiment("Titanic_ML_Experiment")

# Load cleaned dataset
df = pd.read_csv("titanic_cleaned.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Prepare input example for MLflow logging
input_example = X_train_scaled[0].reshape(1, -1)

# Start MLflow experiment
mlflow.set_experiment("Titanic_ML_Experiment")

with mlflow.start_run(run_name="LogReg_vs_RF"):

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)

    mlflow.log_param("lr_model", "LogisticRegression")
    mlflow.log_metric("lr_accuracy", accuracy_score(y_test, lr_preds))
    mlflow.log_metric("lr_f1", f1_score(y_test, lr_preds))
    mlflow.sklearn.log_model(lr, "logistic_model", input_example=input_example)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)

    mlflow.log_param("rf_model", "RandomForest")
    mlflow.log_metric("rf_accuracy", accuracy_score(y_test, rf_preds))
    mlflow.log_metric("rf_f1", f1_score(y_test, rf_preds))
    mlflow.sklearn.log_model(rf, "random_forest_model", input_example=input_example)

    # Save best model (Random Forest)
    joblib.dump(rf, "best_model.pkl")
