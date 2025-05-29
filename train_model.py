import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Set experiment
mlflow.set_experiment("Titanic_ML_Experiment")

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("titanic_cleaned.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

# ---------- EDA Visualizations ----------

# Feature Distributions
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig("figures/eda_distributions.png")
plt.close()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig("figures/correlation_matrix.png")
plt.close()

# Feature Importance (initial RF for EDA only)
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X, y)
importances = pd.Series(rf_temp.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title="Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/feature_importance_rf.png")
plt.close()

# ---------- Model Preparation ----------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Prepare input example for MLflow
input_example = X_train_scaled[0].reshape(1, -1)

# ---------- MLflow Experiment ----------

with mlflow.start_run(run_name="LogReg_vs_RF"):

    # ----- Logistic Regression -----
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)

    mlflow.log_param("lr_model", "LogisticRegression")
    mlflow.log_metric("lr_accuracy", accuracy_score(y_test, lr_preds))
    mlflow.log_metric("lr_f1", f1_score(y_test, lr_preds))
    mlflow.sklearn.log_model(lr, "logistic_model", input_example=input_example)

    # ----- Random Forest Classifier -----
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)

    mlflow.log_param("rf_model", "RandomForest")
    mlflow.log_metric("rf_accuracy", accuracy_score(y_test, rf_preds))
    mlflow.log_metric("rf_f1", f1_score(y_test, rf_preds))
    mlflow.sklearn.log_model(rf, "random_forest_model", input_example=input_example)

    # ----- Confusion Matrices -----
    # Logistic Regression CM
    cm_lr = confusion_matrix(y_test, lr_preds)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
    disp_lr.plot()
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig("figures/confusion_matrix_logreg.png")
    plt.close()

    # Random Forest CM
    cm_rf = confusion_matrix(y_test, rf_preds)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    disp_rf.plot()
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("figures/confusion_matrix_rf.png")
    plt.close()

    # ----- Save best model (Random Forest) -----
    joblib.dump(rf, "best_model.pkl")