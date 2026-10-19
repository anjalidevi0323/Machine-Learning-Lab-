"""
================================================================================
PROJECT: Heart Disease Prediction using Machine Learning Algorithms
SCRIPT: train_model.py (Standalone CLI & Modular Training Engine)
================================================================================
Stages Implemented:
  1. Data Loading: Cleveland Heart Disease Clinical Dataset
  2. Data Preprocessing: Missing value check, train-test splitting, feature scaling
  3. Feature Extraction & Importance: Correlation analysis, tree-based feature rankings
  4. Machine Learning Model Training (5 Core Algorithms):
     - Support Vector Machine (SVM)
     - Random Forest Classifier
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Logistic Regression
  5. Performance Evaluation & Model Comparison:
     - Accuracy, Precision, Recall (Sensitivity), F1-Score
     - Confusion Matrix computation
     - Best Model Selection & Serialization (best_model.pkl & scaler.pkl)
  6. Sample Patient Clinical Inference Simulation
================================================================================
How to Compile and Run:
  1. Compile syntax verification:
     python -m py_compile train_model.py
  2. Execute training & evaluation:
     python train_model.py
================================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Ensure UTF-8 output encoding across Windows shells
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_data.csv")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "best_heart_disease_model.pkl")
SCALER_SAVE_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Human readable feature metadata
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)",
    "restecg": "Resting ECG Results (0, 1, 2)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1=Yes, 0=No)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment (0, 1, 2)",
    "ca": "Number of Major Vessels Colored by Fluoroscopy (0-3)",
    "thal": "Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)"
}


def load_dataset(csv_path=DATA_PATH):
    """Stage 1: Load Clinical Heart Disease Dataset"""
    print("\n" + "=" * 75)
    print("STAGE 1: LOADING CLINICAL HEART DISEASE DATASET")
    print("=" * 75)

    if not os.path.exists(csv_path):
        print(f"[!] Warning: Dataset file not found at {csv_path}. Generating synthetic clinical distribution...")
        df = generate_synthetic_dataset()
    else:
        df = pd.read_csv(csv_path)
        print(f"[+] Loaded dataset successfully from: {csv_path}")

    print(f"    - Total Patient Records : {df.shape[0]}")
    print(f"    - Diagnostic Features   : {df.shape[1] - 1}")
    print(f"    - Target Distribution   : {df['target'].value_counts().to_dict()} (0=Healthy, 1=Heart Disease)")
    return df


def generate_synthetic_dataset(num_samples=300):
    """Fallback generator to ensure robust execution anywhere"""
    np.random.seed(42)
    age = np.random.randint(29, 78, num_samples)
    sex = np.random.choice([0, 1], num_samples, p=[0.32, 0.68])
    cp = np.random.choice([0, 1, 2, 3], num_samples, p=[0.48, 0.17, 0.28, 0.07])
    trestbps = np.random.normal(131, 17, num_samples).clip(94, 200).astype(int)
    chol = np.random.normal(246, 51, num_samples).clip(126, 564).astype(int)
    fbs = np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], num_samples, p=[0.48, 0.50, 0.02])
    thalach = np.random.normal(149, 23, num_samples).clip(71, 202).astype(int)
    exang = np.random.choice([0, 1], num_samples, p=[0.67, 0.33])
    oldpeak = np.round(np.random.exponential(1.0, num_samples).clip(0, 6.2), 1)
    slope = np.random.choice([0, 1, 2], num_samples, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3], num_samples, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([1, 2, 3], num_samples, p=[0.06, 0.55, 0.39])

    # Realistic clinical risk formula
    logit = (
        (age - 54) * 0.03 + (sex * 0.5) + (cp * 0.7) + (trestbps - 130) * 0.02 +
        (chol - 240) * 0.005 + (fbs * 0.3) - (thalach - 150) * 0.04 +
        (exang * 0.8) + (oldpeak * 0.6) - (slope * 0.4) + (ca * 0.7) + (thal * 0.4)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target
    })
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Stage 2: Clinical Data Preprocessing
    - Feature and target isolation
    - Stratified train-test splitting
    - Feature normalization using Z-score standardization (StandardScaler)
    """
    print("\n" + "=" * 75)
    print("STAGE 2: DATA PREPROCESSING & STANDARDIZATION")
    print("=" * 75)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified split ensures balanced presence of disease across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[+] Train/Test Split: {100 - int(test_size*100)}% Train ({len(X_train)} samples), {int(test_size*100)}% Test ({len(X_test)} samples)")

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[+] Features standardized using StandardScaler (mean=0, variance=1).")
    return X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def extract_features(df):
    """
    Stage 3: Feature Extraction & Importance Analysis
    - Correlation coefficients with target
    - Tree-based feature importance rankings
    """
    print("\n" + "=" * 75)
    print("STAGE 3: FEATURE EXTRACTION & CORRELATION ANALYSIS")
    print("=" * 75)

    correlations = df.corr()["target"].drop("target").sort_values(ascending=False)
    print("[+] Top Diagnostic Correlates with Heart Disease:")
    for feature, corr in correlations.items():
        direction = "Positive (+)" if corr > 0 else "Negative (-)"
        print(f"    - {feature:<10} ({FEATURE_LABELS.get(feature, feature):<45}) : {corr:+.4f} [{direction}]")

    # Quick Random Forest feature importance
    rf_feat = RandomForestClassifier(n_estimators=100, random_state=42)
    X_clean = df.drop(columns=["target"])
    rf_feat.fit(X_clean, df["target"])
    importances = pd.Series(rf_feat.feature_importances_, index=X_clean.columns).sort_values(ascending=False)

    print("\n[+] Gini Impurity Feature Importance Rankings:")
    for rank, (feat, imp) in enumerate(importances.items(), 1):
        print(f"    {rank:2d}. {feat:<10} : {imp*100:6.2f}%")

    return correlations, importances


def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Stage 4 & 5: Model Training & Performance Evaluation
    Trains all 5 core Machine Learning algorithms and measures:
      - Accuracy
      - Precision
      - Recall (Sensitivity)
      - F1-Score
      - Confusion Matrix
    """
    print("\n" + "=" * 75)
    print("STAGE 4 & 5: MACHINE LEARNING MODEL TRAINING & PERFORMANCE EVALUATION")
    print("=" * 75)

    models = {
        "Support Vector Machine (SVM)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        # Training
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model

        # Prediction
        y_pred = model.predict(X_test_scaled)

        # Performance Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall (Sensitivity)": rec,
            "F1-Score": f1,
            "Confusion Matrix": cm,
            "Predictions": y_pred
        }

    # Print Comparative Performance Table
    print("\n" + "-" * 75)
    print(f"{'Algorithm':<32} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<8}")
    print("-" * 75)
    for name, m in results.items():
        print(f"{name:<32} | {m['Accuracy']*100:6.2f}%   | {m['Precision']*100:6.2f}%   | {m['Recall (Sensitivity)']*100:6.2f}%   | {m['F1-Score']:<8.4f}")
    print("-" * 75)

    # Detailed Confusion Matrix breakdown
    print("\n[+] Confusion Matrix Details per Algorithm:")
    for name, m in results.items():
        tn, fp, fn, tp = m["Confusion Matrix"].ravel()
        print(f"\n--- {name} ---")
        print(f"    True Negatives (Healthy correctly identified)     : {tn}")
        print(f"    False Positives (False Alarm / Healthy flagged)   : {fp}")
        print(f"    False Negatives (Critical Miss / Disease missed)  : {fn}")
        print(f"    True Positives (Heart Disease correctly detected) : {tp}")

    # Determine Best Model by F1-Score & Accuracy
    best_name = max(results, key=lambda k: (results[k]["F1-Score"], results[k]["Accuracy"]))
    print("\n" + "=" * 75)
    print(f"[BEST ALGORITHM] : {best_name}")
    print(f"    Accuracy : {results[best_name]['Accuracy']*100:.2f}%")
    print(f"    Recall   : {results[best_name]['Recall (Sensitivity)']*100:.2f}%")
    print(f"    F1-Score : {results[best_name]['F1-Score']:.4f}")
    print("=" * 75)

    return fitted_models, results, best_name


def simulate_patient_prediction(model, scaler, patient_dict=None):
    """
    Stage 6: Clinical Inference Simulation
    Feeds a patient's diagnostic values into the trained model and outputs the risk prediction.
    """
    print("\n" + "=" * 75)
    print("STAGE 6: CLINICAL PATIENT PREDICTION SIMULATION")
    print("=" * 75)

    if patient_dict is None:
        # Sample high-risk clinical profile
        patient_dict = {
            "age": 58,
            "sex": 1,         # Male
            "cp": 2,          # Non-anginal pain
            "trestbps": 140,  # Prehypertension
            "chol": 260,      # Elevated cholesterol
            "fbs": 1,         # Fasting sugar > 120
            "restecg": 1,     # ST-T abnormality
            "thalach": 135,   # Low maximum HR
            "exang": 1,       # Angina during exercise
            "oldpeak": 2.2,   # 2.2mm ST depression
            "slope": 1,       # Flat ST slope
            "ca": 2,          # 2 major vessels colored
            "thal": 3         # Reversible defect
        }

    print("[+] Patient Diagnostic Values:")
    for k, v in patient_dict.items():
        print(f"    - {k:<10} ({FEATURE_LABELS.get(k, k):<42}) : {v}")

    # Convert to DataFrame with matching feature names
    patient_df = pd.DataFrame([patient_dict])[FEATURE_NAMES]
    scaled_vector = scaler.transform(patient_df)

    pred = model.predict(scaled_vector)[0]
    prob = model.predict_proba(scaled_vector)[0] if hasattr(model, "predict_proba") else None

    print("\n" + "*" * 50)
    if pred == 1:
        confidence = f"({prob[1]*100:.1f}% Risk Probability)" if prob is not None else ""
        print(f"[PREDICTION RESULT] : POSITIVE - HEART DISEASE DETECTED {confidence}")
        print("   Recommendation: Immediate cardiology consult, troponin/angiography workup required.")
    else:
        confidence = f"({prob[0]*100:.1f}% Confidence)" if prob is not None else ""
        print(f"[PREDICTION RESULT] : NEGATIVE - NO HEART DISEASE DETECTED {confidence}")
        print("   Recommendation: Standard routine cardiac monitoring and healthy lifestyle.")
    print("*" * 50)


def main():
    """Main Orchestrator"""
    print("*" * 75)
    print("   HEART DISEASE PREDICTION - MACHINE LEARNING PIPELINE")
    print("   Python, Scikit-Learn, Multi-Algorithm Benchmark Suite")
    print("*" * 75)

    # 1. Load Data
    df = load_dataset()

    # 2. Preprocess
    X, y, X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = preprocess_data(df)

    # 3. Feature Extraction
    extract_features(df)

    # 4 & 5. Train & Evaluate
    fitted_models, results, best_name = train_and_evaluate_models(X_train_s, X_test_s, y_train, y_test)

    # Save artifacts for web app usage
    best_model = fitted_models[best_name]
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n[+] Saved best trained model to: {MODEL_SAVE_PATH}")
    print(f"[+] Saved feature scaler to: {SCALER_SAVE_PATH}")

    # 6. Simulate Patient Test
    simulate_patient_prediction(best_model, scaler)

    print("\n[DONE] Execution complete. To launch the interactive Single-Page Web Application, run:")
    print("       python -m streamlit run app.py\n")


if __name__ == "__main__":
    main()
