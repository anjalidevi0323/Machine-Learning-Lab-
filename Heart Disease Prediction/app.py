"""
================================================================================
PROJECT: CardioVision - Heart Disease Prediction AI Studio
================================================================================
Unified Single-Page Web Application & Machine Learning Suite:
  1. Patient Input & Live Prediction (Clinical Vitals, Risk Gauge, Multi-Model Consensus, ASCVD Score, Care Plan, Printable Summary)
  2. Medical Cardiac ECG Waveform & Image Pipeline (Live Waveform Generator, Peak Detection, Filters, CLAHE, HOG & LBP Feature Extraction)
  3. Preprocessing & Exploratory Feature Intelligence (Data Explorer, Interactive Correlation Heatmap, 3D Feature Space, Violin Plots)
  4. Model Training & Comparison Studio (6 Classifiers + Voting Ensemble, Hyperparameter Tuning, ROC/PR Curves, K-Fold CV, Confusion Matrix)
  5. Interactive Python Code Editor & Sandbox (Preloaded Templates, Compile & Run, Live Terminal Capture, Auto-Plot Renderer)
  6. Clinical Knowledge Base & Viva Guide (Feature Dictionary, Metric Math, Clinical Protocols, Compilation Instructions)
================================================================================
"""

import os
import sys
import io
import time
import base64
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# Optional imports with graceful fallbacks
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image, ImageOps, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from skimage.feature import hog, local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioVision AI Studio - Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Ultra-Modern Medical Glassmorphism & High-Tech UI)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Animated Pulse Glow Line */
    .pulse-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 16px;
        background: rgba(220, 38, 38, 0.08);
        border: 1px solid rgba(220, 38, 38, 0.25);
        border-radius: 50px;
        width: fit-content;
        margin-bottom: 12px;
    }
    .pulse-dot {
        width: 10px;
        height: 10px;
        background-color: #EF4444;
        border-radius: 50%;
        box-shadow: 0 0 0 rgba(239, 68, 68, 0.7);
        animation: pulseAnimation 1.6s infinite;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .pulse-text {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #DC2626;
        text-transform: uppercase;
    }

    /* Main Hero Header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #EF4444 0%, #B91C1C 50%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }

    /* Glassmorphic Metric and Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.95) 100%);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.04), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: rgba(220, 38, 38, 0.3);
    }

    /* Diagnostic Outcome Cards */
    .diagnostic-card {
        padding: 1.8rem;
        border-radius: 16px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        border-left: 8px solid;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.06);
        animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .card-healthy {
        background: linear-gradient(145deg, #F0FDF4 0%, #DCFCE7 100%);
        border-color: #10B981;
        color: #064E3B;
    }
    .card-warning {
        background: linear-gradient(145deg, #FFFBEB 0%, #FEF3C7 100%);
        border-color: #F59E0B;
        color: #78350F;
    }
    .card-disease {
        background: linear-gradient(145deg, #FEF2F2 0%, #FEE2E2 100%);
        border-color: #EF4444;
        color: #7F1D1D;
    }

    /* Badges & Pills */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-success { background-color: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-danger { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-info { background-color: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }

    /* Code Terminal Box */
    .terminal-box {
        background-color: #0B0F19;
        color: #38BDF8;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #1E293B;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 0.88rem;
        line-height: 1.6;
        max-height: 480px;
        overflow-y: auto;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #F1F5F9;
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #475569;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #DC2626 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }

    /* Form styling enhancement */
    div[data-testid="stForm"] {
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        background-color: #FAFAFC;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Clinical Constants & Metadata Dictionary
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_data.csv")

FEATURE_LABELS = {
    "age": "Age (Years)",
    "sex": "Sex (1 = Male, 0 = Female)",
    "cp": "Chest Pain Type (0: Typical Angina, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1 = High, 0 = Normal)",
    "restecg": "Resting ECG (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)",
    "thalach": "Maximum Heart Rate Achieved (bpm)",
    "exang": "Exercise Induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression Induced by Exercise Relative to Rest (mm)",
    "slope": "Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)",
    "ca": "Number of Major Vessels Colored by Fluoroscopy (0 - 3)",
    "thal": "Thalassemia Stress Scan (1: Normal, 2: Fixed Defect, 3: Reversible Defect)"
}

FEATURE_DESCRIPTIONS = {
    "age": "Chronological age of patient. Risk of atherosclerosis increases sharply after age 45 in men and 55 in women.",
    "sex": "Biological sex. Statistically, men present earlier with coronary artery disease due to hormonal factors.",
    "cp": "Chest Pain presentation. Typical angina is retrosternal pressure provoked by exertion and relieved by rest or nitroglycerin.",
    "trestbps": "Systemic systolic blood pressure upon clinical admission. Values > 130 mm Hg signify Stage 1 hypertension.",
    "chol": "Total fasting serum cholesterol. Levels > 200 mg/dl contribute directly to arterial lipid plaque formation.",
    "fbs": "Elevated fasting blood sugar > 120 mg/dl indicates insulin resistance or diabetes mellitus, a major cardiovascular accelerator.",
    "restecg": "Electrocardiographic baseline rhythm: Normal vs repolarization anomalies vs Left Ventricular Hypertrophy (LVH).",
    "thalach": "Peak heart rate during treadmill stress test. Inability to reach age-predicted target indicates chronotropic incompetence.",
    "exang": "Occurrence of myocardial ischemia-induced angina during physical exercise test.",
    "oldpeak": "Electrocardiographic ST segment depression (mm) during peak stress relative to baseline. Key ischemic marker.",
    "slope": "Trajectory of ST segment at peak exercise: Upsloping (favorable), Flat (ischemic), Downsloping (severe ischemia).",
    "ca": "Number of epicardial coronary arteries exhibiting >= 50% luminal stenosis under fluoroscopic coronary angiography.",
    "thal": "Nuclear scintigraphy myocardial perfusion result: Fixed defect (prior necrosis) vs Reversible defect (active inducible ischemia)."
}


# -----------------------------------------------------------------------------
# Dataset Loader & Generator
# -----------------------------------------------------------------------------
@st.cache_data
def load_or_create_dataset():
    """Loads the benchmark Cleveland Clinic Heart Disease dataset or synthesizes realistic data."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        np.random.seed(42)
        n = 303
        age = np.random.randint(29, 78, n)
        sex = np.random.choice([0, 1], n, p=[0.32, 0.68])
        cp = np.random.choice([0, 1, 2, 3], n, p=[0.48, 0.17, 0.28, 0.07])
        trestbps = np.random.normal(131, 17, n).clip(94, 200).astype(int)
        chol = np.random.normal(246, 51, n).clip(126, 564).astype(int)
        fbs = np.random.choice([0, 1], n, p=[0.85, 0.15])
        restecg = np.random.choice([0, 1, 2], n, p=[0.48, 0.50, 0.02])
        thalach = np.random.normal(149, 23, n).clip(71, 202).astype(int)
        exang = np.random.choice([0, 1], n, p=[0.67, 0.33])
        oldpeak = np.round(np.random.exponential(1.0, n).clip(0, 6.2), 1)
        slope = np.random.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
        ca = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
        thal = np.random.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39])

        logit = (
            (age - 54) * 0.03 + (sex * 0.55) + (cp * 0.65) + (trestbps - 130) * 0.022 +
            (chol - 240) * 0.006 + (fbs * 0.35) - (thalach - 150) * 0.042 +
            (exang * 0.85) + (oldpeak * 0.65) - (slope * 0.45) + (ca * 0.72) + (thal * 0.42)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        target = (prob > 0.5).astype(int)

        df = pd.DataFrame({
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target
        })
        df.to_csv(DATA_PATH, index=False)
    return df


df_clinical = load_or_create_dataset()


# -----------------------------------------------------------------------------
# Model Suite Training, Cross-Validation & Calibration Engine
# -----------------------------------------------------------------------------
@st.cache_resource
def train_all_models(df, test_size=0.2, random_state=42, n_trees=100, max_depth=6, svm_c=1.0, knn_k=5):
    """
    Trains 6 Machine Learning algorithms + 1 Voting Ensemble Classifier.
    Computes performance metrics, confusion matrices, ROC/PR curves, and 5-fold cross-validation.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Algorithm collection
    models = {
        "Support Vector Machine (SVM)": SVC(kernel="rbf", C=svm_c, probability=True, random_state=random_state),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=random_state),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=knn_k),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state)
    }

    if HAS_XGBOOST:
        models["XGBoost Classifier"] = xgb.XGBClassifier(
            n_estimators=n_trees, max_depth=max_depth, learning_rate=0.08,
            random_state=random_state, eval_metric="logloss"
        )

    # Fit base models
    fitted = {}
    metrics = {}

    for name, m in models.items():
        m.fit(X_train_s, y_train)
        fitted[name] = m

        preds = m.predict(X_test_s)
        prob_y = m.predict_proba(X_test_s)[:, 1] if hasattr(m, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # ROC Curve & AUC
        if prob_y is not None:
            fpr, tpr, _ = roc_curve(y_test, prob_y)
            roc_auc = auc(fpr, tpr)
            pr_rec, pr_prec, _ = precision_recall_curve(y_test, prob_y)
        else:
            fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
            pr_rec, pr_prec = [0, 1], [1, 0]

        # 5-fold CV score on entire dataset
        cv_scores = cross_val_score(m, scaler.transform(X), y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring="accuracy")

        metrics[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "CV_Mean": cv_scores.mean(),
            "CV_Std": cv_scores.std(),
            "Confusion Matrix": cm,
            "Predictions": preds,
            "Probabilities": prob_y,
            "FPR": fpr,
            "TPR": tpr,
            "PR_Recall": pr_rec,
            "PR_Precision": pr_prec
        }

    # Build Soft Voting Ensemble
    voting_estimators = [
        ("svm", fitted["Support Vector Machine (SVM)"]),
        ("rf", fitted["Random Forest Classifier"]),
        ("lr", fitted["Logistic Regression"])
    ]
    if HAS_XGBOOST and "XGBoost Classifier" in fitted:
        voting_estimators.append(("xgb", fitted["XGBoost Classifier"]))

    ensemble = VotingClassifier(estimators=voting_estimators, voting="soft")
    ensemble.fit(X_train_s, y_train)
    fitted["Soft Voting Ensemble"] = ensemble

    ens_preds = ensemble.predict(X_test_s)
    ens_prob = ensemble.predict_proba(X_test_s)[:, 1]
    ens_acc = accuracy_score(y_test, ens_preds)
    ens_prec = precision_score(y_test, ens_preds, zero_division=0)
    ens_rec = recall_score(y_test, ens_preds, zero_division=0)
    ens_f1 = f1_score(y_test, ens_preds, zero_division=0)
    ens_cm = confusion_matrix(y_test, ens_preds)
    tn, fp, fn, tp = ens_cm.ravel()
    ens_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_test, ens_prob)
    roc_auc = auc(fpr, tpr)
    pr_rec, pr_prec, _ = precision_recall_curve(y_test, ens_prob)
    cv_scores = cross_val_score(ensemble, scaler.transform(X), y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring="accuracy")

    metrics["Soft Voting Ensemble"] = {
        "Accuracy": ens_acc,
        "Precision": ens_prec,
        "Recall": ens_rec,
        "Specificity": ens_spec,
        "F1-Score": ens_f1,
        "ROC-AUC": roc_auc,
        "CV_Mean": cv_scores.mean(),
        "CV_Std": cv_scores.std(),
        "Confusion Matrix": ens_cm,
        "Predictions": ens_preds,
        "Probabilities": ens_prob,
        "FPR": fpr,
        "TPR": tpr,
        "PR_Recall": pr_rec,
        "PR_Precision": pr_prec
    }

    return fitted, metrics, scaler, X_test, y_test, X_train, y_train


# -----------------------------------------------------------------------------
# ECG Signal Synthesis & Image Processing Pipeline
# -----------------------------------------------------------------------------
def synthesize_ecg_signal(condition="Normal Sinus Rhythm", duration=4.0, fs=250, bpm=72, noise_level=0.03):
    """Synthesizes high-fidelity 12-lead equivalent Lead II ECG electrical tracing."""
    t = np.linspace(0, duration, int(duration * fs))
    rr_sec = 60.0 / bpm
    signal = np.zeros_like(t)

    # Generate pulse train
    beat_times = np.arange(0.2, duration, rr_sec)

    for bt in beat_times:
        dt = t - bt

        if condition == "Normal Sinus Rhythm":
            # Normal P-Q-R-S-T
            p = 0.18 * np.exp(-((dt + 0.18) ** 2) / (2 * (0.025 ** 2)))
            q = -0.15 * np.exp(-((dt + 0.05) ** 2) / (2 * (0.012 ** 2)))
            r = 1.15 * np.exp(-(dt ** 2) / (2 * (0.016 ** 2)))
            s = -0.28 * np.exp(-((dt - 0.05) ** 2) / (2 * (0.012 ** 2)))
            t_wave = 0.32 * np.exp(-((dt - 0.22) ** 2) / (2 * (0.045 ** 2)))
            signal += (p + q + r + s + t_wave)

        elif condition == "Myocardial Infarction (STEMI)":
            # Pronounced pathological Q-wave, diminished R, marked ST elevation & inverted T
            q = -0.38 * np.exp(-((dt + 0.06) ** 2) / (2 * (0.020 ** 2)))
            r = 0.65 * np.exp(-(dt ** 2) / (2 * (0.015 ** 2)))
            st_elev = 0.45 * np.exp(-((dt - 0.12) ** 2) / (2 * (0.080 ** 2)))
            t_inv = -0.25 * np.exp(-((dt - 0.28) ** 2) / (2 * (0.040 ** 2)))
            signal += (q + r + st_elev + t_inv)

        elif condition == "Atrial Fibrillation":
            # Absent P-wave, irregular fibrillatory baseline, normal QRS
            f_wave = 0.08 * np.sin(2 * np.pi * 32 * dt)
            q = -0.12 * np.exp(-((dt + 0.04) ** 2) / (2 * (0.010 ** 2)))
            r = 1.05 * np.exp(-(dt ** 2) / (2 * (0.015 ** 2)))
            s = -0.22 * np.exp(-((dt - 0.04) ** 2) / (2 * (0.012 ** 2)))
            t_wave = 0.18 * np.exp(-((dt - 0.18) ** 2) / (2 * (0.035 ** 2)))
            signal += (f_wave + q + r + s + t_wave)

        else:  # Ventricular Tachycardia
            # Wide, bizarre monomorphic QRS without discernible P or T
            vt = 1.2 * np.sin(2 * np.pi * 3.5 * dt) * np.exp(-(dt ** 2) / 0.08)
            signal += vt

    # Add Gaussian artifact noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(t))
        signal += noise

    return t, signal


def generate_sample_ecg_image(condition="Normal", w=220, h=140):
    """Generates synthetic ECG image on standardized medical pink millimetric grid."""
    canvas = np.full((h, w, 3), (250, 245, 248), dtype=np.uint8)

    if HAS_OPENCV:
        # Medical grid
        for x in range(0, w, 10):
            cv2.line(canvas, (x, 0), (x, h), (235, 210, 220), 1)
        for y in range(0, h, 10):
            cv2.line(canvas, (0, y), (w, y), (235, 210, 220), 1)

    points = []
    baseline = h // 2 + 5
    for x in range(w):
        if condition.startswith("Normal"):
            p = 7.0 * np.exp(-((x % 55 - 12) ** 2) / 10.0)
            q = -6.0 * np.exp(-((x % 55 - 24) ** 2) / 3.0)
            r = 45.0 * np.exp(-((x % 55 - 28) ** 2) / 3.5)
            s = -10.0 * np.exp(-((x % 55 - 32) ** 2) / 3.0)
            t_val = 14.0 * np.exp(-((x % 55 - 42) ** 2) / 14.0)
            y = baseline - (p + q + r + s + t_val)
        elif condition.startswith("Myocardial") or "STEMI" in condition:
            q = -15.0 * np.exp(-((x % 60 - 20) ** 2) / 4.0)
            r = 32.0 * np.exp(-((x % 60 - 25) ** 2) / 4.0)
            st_elev = 22.0 * np.exp(-((x % 60 - 35) ** 2) / 28.0)
            y = baseline - (q + r + st_elev)
        else:
            spike = 30.0 * np.sin(x * 0.35) + 15.0 * np.cos(x * 0.7)
            y = baseline - spike

        points.append((x, int(np.clip(y, 6, h - 6))))

    if HAS_OPENCV:
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], (20, 20, 25), 2)
    return canvas


def preprocess_ecg_pipeline(img_bgr, target_size=(140, 140), filter_mode="Gaussian Blur", clahe_clip=2.5):
    """Executes multi-stage medical image enhancement."""
    if not HAS_OPENCV:
        pil_img = Image.fromarray(img_bgr).resize(target_size)
        gray = ImageOps.grayscale(pil_img)
        filtered = gray.filter(ImageFilter.GaussianBlur(radius=1))
        thresh = gray.point(lambda p: 255 if p < 130 else 0)
        return np.array(pil_img), np.array(gray), np.array(filtered), np.array(gray), np.array(thresh)

    resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    if filter_mode == "Gaussian Blur":
        filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    elif filter_mode == "Median Blur":
        filtered = cv2.medianBlur(gray, 5)
    elif filter_mode == "Bilateral Filter":
        filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    else:
        filtered = gray.copy()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return resized, gray, filtered, enhanced, thresholded


# -----------------------------------------------------------------------------
# Sidebar Navigation & Model Calibration Suite
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 12px;'>
            <img src='https://img.icons8.com/color/144/heart-with-pulse.png' width='75'/>
            <h2 style='margin: 4px 0 0 0; color: #DC2626; font-weight: 800; font-size: 1.5rem;'>CardioVision AI</h2>
            <p style='margin: 0; color: #64748B; font-size: 0.85rem; font-weight: 600;'>Heart Disease Diagnostic Studio</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Active Clinical Classifier")
    available_model_names = [
        "Support Vector Machine (SVM)",
        "Random Forest Classifier",
        "Soft Voting Ensemble",
        "K-Nearest Neighbors (KNN)",
        "Decision Tree Classifier",
        "Logistic Regression"
    ]
    if HAS_XGBOOST:
        available_model_names.insert(2, "XGBoost Classifier")

    selected_model_name = st.selectbox(
        "Select Primary Model:",
        available_model_names,
        index=1,
        help="Choose the algorithm used for patient evaluation and live diagnostic reporting."
    )

    st.markdown("---")
    with st.expander("🛠️ Hyperparameter Calibration", expanded=False):
        st.caption("Live retrain all models with custom training parameters:")
        tune_split = st.slider("Test Split Ratio:", 0.10, 0.40, 0.20, step=0.05)
        tune_trees = st.slider("Ensemble Trees (RF / XGB):", 20, 250, 100, step=10)
        tune_depth = st.slider("Max Tree Depth:", 2, 12, 6)
        tune_svm_c = st.slider("SVM C Regularization:", 0.1, 10.0, 1.0, step=0.1)
        tune_knn = st.slider("KNN Neighbors (k):", 1, 15, 5, step=2)

    st.markdown("---")
    st.markdown("#### ⚡ Quick Patient Presets")
    preset_choice = st.radio(
        "Load Clinical Profile:",
        [
            "Custom Input (Manual)",
            "Profile 1: Healthy Athlete (Low Risk)",
            "Profile 2: Moderate Risk (Hypertensive / Smoker)",
            "Profile 3: High Risk (Angina & ST Depression)",
            "Profile 4: Critical Acute STEMI / Infarct",
            "Profile 5: Diabetic Female (Atypical Symptoms)"
        ],
        index=0
    )

    st.markdown("---")
    st.info("💡 **Quick Viva Tip:** In cardiovascular ML, **Recall (Sensitivity)** is prioritized over Accuracy to prevent discharging at-risk patients untreated (minimizing False Negatives).")


# Train models with configured parameters
models_dict, metrics_dict, clinical_scaler, X_test_df, y_test_series, X_train_df, y_train_series = train_all_models(
    df_clinical,
    test_size=tune_split,
    n_trees=tune_trees,
    max_depth=tune_depth,
    svm_c=tune_svm_c,
    knn_k=tune_knn
)


# -----------------------------------------------------------------------------
# Main Hero Header & KPI Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class='pulse-container'>
    <div class='pulse-dot'></div>
    <span class='pulse-text'>Clinical AI Diagnostics Engine Active • Cleveland Clinic Dataset Ready</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>❤️ CardioVision: Heart Disease Prediction AI Studio</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>"
    "Next-Generation Clinical AI Intelligence Suite: <b>Patient Vitals Triage ➔ Live ECG Waveform Analyzer ➔ Exploratory Analytics ➔ Multi-Model ML Benchmarking ➔ In-Browser Python Sandbox</b>"
    "</div>",
    unsafe_allow_html=True
)

# Top KPI Metric Cards
best_algo_name = max(metrics_dict, key=lambda k: metrics_dict[k]["F1-Score"])
top_acc = metrics_dict[best_algo_name]["Accuracy"] * 100
top_rec = metrics_dict[best_algo_name]["Recall"] * 100
top_f1 = metrics_dict[best_algo_name]["F1-Score"]
top_auc = metrics_dict[best_algo_name]["ROC-AUC"]

col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
col_kpi1.metric("🏥 Patient Cohort", f"{len(df_clinical)} Records", "13 Clinical Features")
col_kpi2.metric("🏆 Top Model", best_algo_name.split()[0], f"{top_acc:.1f}% Accuracy")
col_kpi3.metric("🎯 Top Recall / Sens.", f"{top_rec:.1f}%", "Critical Safety Metric")
col_kpi4.metric("⚖️ Top F1-Score", f"{top_f1:.3f}", "Harmonic Mean")
col_kpi5.metric("📈 Top ROC-AUC", f"{top_auc:.3f}", "Discrimination Index")

st.markdown("---")


# -----------------------------------------------------------------------------
# Unified Tabbed Navigation
# -----------------------------------------------------------------------------
tab_patient, tab_image, tab_data, tab_train, tab_editor, tab_guide = st.tabs([
    "🩺 Patient Vitals & Live Diagnosis",
    "🔬 Cardiac ECG Waveform & Image Studio",
    "📊 Data Preprocessing & Feature Intelligence",
    "🏋️ Model Training & Benchmarking Studio",
    "💻 Live Python Code Sandbox",
    "📖 Clinical Knowledge & Viva Guide"
])


# =============================================================================
# TAB 1: PATIENT VITALS & LIVE DIAGNOSTIC STUDIO
# =============================================================================
with tab_patient:
    st.markdown("### 🩺 Patient Clinical Vitals & Live AI Diagnosis")
    st.write("Enter patient physiological parameters below or choose a clinical preset from the sidebar. The selected model will instantly evaluate cardiovascular risk.")

    # Preset values dictionary
    p_vals = {
        "age": 54, "sex": 1, "cp": 1, "trestbps": 130, "chol": 240, "fbs": 0,
        "restecg": 0, "thalach": 155, "exang": 0, "oldpeak": 0.8, "slope": 2, "ca": 0, "thal": 2
    }
    if preset_choice.startswith("Profile 1"):
        p_vals = {"age": 32, "sex": 0, "cp": 2, "trestbps": 112, "chol": 175, "fbs": 0, "restecg": 0, "thalach": 182, "exang": 0, "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1}
    elif preset_choice.startswith("Profile 2"):
        p_vals = {"age": 56, "sex": 1, "cp": 1, "trestbps": 142, "chol": 265, "fbs": 1, "restecg": 1, "thalach": 140, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 1, "thal": 2}
    elif preset_choice.startswith("Profile 3"):
        p_vals = {"age": 63, "sex": 1, "cp": 0, "trestbps": 155, "chol": 298, "fbs": 1, "restecg": 1, "thalach": 122, "exang": 1, "oldpeak": 2.6, "slope": 1, "ca": 2, "thal": 3}
    elif preset_choice.startswith("Profile 4"):
        p_vals = {"age": 69, "sex": 1, "cp": 0, "trestbps": 172, "chol": 360, "fbs": 1, "restecg": 2, "thalach": 98, "exang": 1, "oldpeak": 4.2, "slope": 2, "ca": 3, "thal": 3}
    elif preset_choice.startswith("Profile 5"):
        p_vals = {"age": 61, "sex": 0, "cp": 3, "trestbps": 148, "chol": 280, "fbs": 1, "restecg": 1, "thalach": 135, "exang": 1, "oldpeak": 2.1, "slope": 1, "ca": 1, "thal": 3}

    with st.form("clinical_patient_form"):
        st.markdown("##### 1. Demographics & Hemodynamics")
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        with f_c1:
            inp_age = st.slider("Patient Age (Years):", 18, 100, p_vals["age"])
        with f_c2:
            inp_sex = st.selectbox("Biological Sex:", [("Male (1)", 1), ("Female (0)", 0)], index=0 if p_vals["sex"] == 1 else 1)[1]
        with f_c3:
            inp_trestbps = st.slider("Resting BP (mm Hg):", 80, 220, p_vals["trestbps"], help="Normal: <120, Prehypertension: 120-139, Stage 1 HTN: 140-159, Stage 2: >=160")
        with f_c4:
            inp_chol = st.slider("Serum Cholesterol (mg/dl):", 100, 600, p_vals["chol"], help="Desirable: <200, Borderline: 200-239, High: >=240")

        st.markdown("##### 2. Symptom Presentation & Baseline ECG")
        f_c5, f_c6, f_c7, f_c8 = st.columns(4)
        with f_c5:
            inp_cp = st.selectbox(
                "Chest Pain Type (cp):",
                [
                    ("0: Typical Angina (Exertional retrosternal pressure)", 0),
                    ("1: Atypical Angina (Non-exertional chest pain)", 1),
                    ("2: Non-Anginal Pain (Musculoskeletal/GI)", 2),
                    ("3: Asymptomatic (Silent ischemia risk)", 3)
                ],
                index=p_vals["cp"]
            )[1]
        with f_c6:
            inp_fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl:",
                [("0: Normal (<= 120 mg/dl)", 0), ("1: High / Diabetic (> 120 mg/dl)", 1)],
                index=p_vals["fbs"]
            )[1]
        with f_c7:
            inp_restecg = st.selectbox(
                "Resting ECG Result:",
                [
                    ("0: Normal Sinus Rhythm", 0),
                    ("1: ST-T Wave Abnormality", 1),
                    ("2: Left Ventricular Hypertrophy (LVH)", 2)
                ],
                index=p_vals["restecg"]
            )[1]
        with f_c8:
            inp_thalach = st.slider("Max Heart Rate (bpm):", 60, 220, p_vals["thalach"], help="Age-predicted max HR ~ 220 - Age")

        st.markdown("##### 3. Treadmill Stress Test & Fluoroscopy")
        f_c9, f_c10, f_c11, f_c12 = st.columns(4)
        with f_c9:
            inp_exang = st.selectbox(
                "Exercise Induced Angina:",
                [("0: No Angina with Exercise", 0), ("1: Yes, Angina Provoked by Exercise", 1)],
                index=p_vals["exang"]
            )[1]
        with f_c10:
            inp_oldpeak = st.slider("ST Depression (oldpeak mm):", 0.0, 7.0, float(p_vals["oldpeak"]), step=0.1)
        with f_c11:
            inp_slope = st.selectbox(
                "ST Segment Slope:",
                [
                    ("0: Upsloping (Often Normal)", 0),
                    ("1: Flat (Ischemic Indicator)", 1),
                    ("2: Downsloping (Severe Ischemia)", 2)
                ],
                index=p_vals["slope"]
            )[1]
        with f_c12:
            inp_ca = st.slider("Fluoroscopy Major Vessels (ca 0-3):", 0, 3, p_vals["ca"])

        inp_thal = st.selectbox(
            "4. Thalassemia Scintigraphy Scan:",
            [
                ("1: Normal Blood Flow (No perfusion defect)", 1),
                ("2: Fixed Defect (Prior myocardial infarction scar)", 2),
                ("3: Reversible Defect (Active inducible ischemia under stress)", 3)
            ],
            index=p_vals["thal"] - 1
        )[1]

        submit_patient_btn = st.form_submit_button("🔍 Execute Machine Learning Cardiac Diagnosis", use_container_width=True, type="primary")

    # Patient vector assembly
    patient_df = pd.DataFrame([{
        "age": inp_age, "sex": inp_sex, "cp": inp_cp, "trestbps": inp_trestbps,
        "chol": inp_chol, "fbs": inp_fbs, "restecg": inp_restecg, "thalach": inp_thalach,
        "exang": inp_exang, "oldpeak": inp_oldpeak, "slope": inp_slope, "ca": inp_ca, "thal": inp_thal
    }])

    scaled_patient = clinical_scaler.transform(patient_df)
    active_model = models_dict[selected_model_name]
    patient_pred = active_model.predict(scaled_patient)[0]

    if hasattr(active_model, "predict_proba"):
        risk_probability = active_model.predict_proba(scaled_patient)[0][1] * 100
    else:
        risk_probability = 85.0 if patient_pred == 1 else 15.0

    # Diagnostic Output Section
    st.markdown("#### 🎯 Diagnostic Evaluation & Clinical Risk Report")

    diag_col1, diag_col2 = st.columns([1, 1])

    with diag_col1:
        # Plotly Gauge Indicator
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_probability,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>Cardiac Disease Risk Probability</b><br><span style='font-size:0.8em;color:gray;'>{selected_model_name}</span>", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 36, 'color': "#EF4444" if patient_pred == 1 else "#10B981"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#EF4444" if patient_pred == 1 else "#10B981"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.25)'},
                    {'range': [30, 65], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with diag_col2:
        if patient_pred == 1:
            st.markdown(
                f"""
                <div class='diagnostic-card card-disease'>
                    <h3 style='margin-top:0; color:#991B1B;'>🚨 POSITIVE FOR HEART DISEASE (HIGH RISK)</h3>
                    <p><b>Model Diagnosis:</b> The patient exhibits clinical markers strongly associated with obstructive coronary artery disease and myocardial ischemia.</p>
                    <p><b>Key Risk Drivers Identified:</b></p>
                    <ul>
                        <li><b>ST Depression (oldpeak):</b> {inp_oldpeak} mm (Normal < 1.0 mm)</li>
                        <li><b>Fluoroscopy Occlusion:</b> {inp_ca} major vessels colored</li>
                        <li><b>Max Heart Rate:</b> {inp_thalach} bpm (Target ~ {220-inp_age} bpm)</li>
                    </ul>
                    <span class='badge badge-danger'>Priority: Urgent Cardiology Triage</span>
                    <span class='badge badge-warning'>ASCVD Risk: Elevated</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='diagnostic-card card-healthy'>
                    <h3 style='margin-top:0; color:#065F46;'>✅ NORMAL / NEGATIVE FOR HEART DISEASE</h3>
                    <p><b>Model Diagnosis:</b> Patient hemodynamic and stress parameters fall within healthy physiological ranges. Low probability of active coronary occlusion.</p>
                    <p><b>Protective Factors Identified:</b></p>
                    <ul>
                        <li><b>Preserved Exercise Capacity:</b> {inp_thalach} bpm max heart rate</li>
                        <li><b>Minimal ST Depression:</b> {inp_oldpeak} mm</li>
                        <li><b>Vessel Patency:</b> {inp_ca} fluoroscopy obstructions</li>
                    </ul>
                    <span class='badge badge-success'>Risk Level: Low / Normal</span>
                    <span class='badge badge-info'>Routine Annual Follow-up</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Multi-Model Consensus Grid
    st.markdown("##### ⚡ Multi-Model Algorithm Consensus")
    consensus_cols = st.columns(len(models_dict))
    positive_votes = 0

    for idx, (m_name, m_obj) in enumerate(models_dict.items()):
        p_vote = m_obj.predict(scaled_patient)[0]
        if p_vote == 1:
            positive_votes += 1
        with consensus_cols[idx]:
            short_name = m_name.replace("Classifier", "").replace("Machine", "").strip()
            if p_vote == 1:
                st.error(f"**{short_name}**\n\n🚨 Disease")
            else:
                st.success(f"**{short_name}**\n\n✅ Healthy")

    consensus_pct = (positive_votes / len(models_dict)) * 100
    st.caption(f"**Consensus Score:** `{positive_votes}/{len(models_dict)}` models flagged elevated risk ({consensus_pct:.0f}% Agreement).")

    # Local Explainability Waterfall / Risk Factors
    st.markdown("---")
    st.markdown("##### 🔍 Local Feature Risk Attribution (Explainability)")
    
    # Calculate feature z-score deviation from healthy baseline
    healthy_means = df_clinical[df_clinical["target"] == 0].mean()
    healthy_stds = df_clinical[df_clinical["target"] == 0].std()
    
    feature_deviations = {}
    for feat in FEATURE_LABELS.keys():
        val = patient_df[feat].values[0]
        mean = healthy_means[feat]
        std = healthy_stds[feat] if healthy_stds[feat] > 0 else 1.0
        z_dev = (val - mean) / std
        feature_deviations[FEATURE_LABELS[feat].split("(")[0].strip()] = z_dev

    dev_df = pd.DataFrame(list(feature_deviations.items()), columns=["Feature", "Deviation from Healthy Baseline"])
    dev_df["Impact"] = dev_df["Deviation from Healthy Baseline"].apply(lambda x: "Increases Risk" if x > 0.5 else ("Protective / Normal" if x < -0.5 else "Neutral"))

    fig_dev = px.bar(
        dev_df.sort_values(by="Deviation from Healthy Baseline", ascending=True),
        x="Deviation from Healthy Baseline",
        y="Feature",
        orientation="h",
        color="Impact",
        color_discrete_map={"Increases Risk": "#EF4444", "Protective / Normal": "#10B981", "Neutral": "#94A3B8"},
        title="Patient Vitals Relative to Healthy Patient Cohort (Z-Score Deviation)"
    )
    fig_dev.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_dev, use_container_width=True)

    # Printable / Downloadable Diagnostic Summary
    st.markdown("---")
    report_text = f"""================================================================================
CARDIOVISION AI STUDIO - CLINICAL DIAGNOSTIC REPORT
================================================================================
Report Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Selected AI Model: {selected_model_name}
Calculated Risk Probability: {risk_probability:.1f}%
Diagnostic Classification: {'POSITIVE FOR HEART DISEASE' if patient_pred == 1 else 'NEGATIVE (HEALTHY / NORMAL)'}
Consensus Score: {positive_votes}/{len(models_dict)} algorithms agreement

PATIENT PHYSIOLOGICAL PROFILE:
- Age: {inp_age} years | Biological Sex: {'Male' if inp_sex == 1 else 'Female'}
- Resting Blood Pressure: {inp_trestbps} mm Hg
- Serum Cholesterol: {inp_chol} mg/dl
- Fasting Blood Sugar > 120 mg/dl: {'Yes' if inp_fbs == 1 else 'No'}
- Chest Pain Type: {inp_cp}
- Resting ECG: {inp_restecg}
- Max Heart Rate: {inp_thalach} bpm
- Exercise Induced Angina: {'Yes' if inp_exang == 1 else 'No'}
- ST Depression (oldpeak): {inp_oldpeak} mm
- ST Slope: {inp_slope}
- Major Fluoroscopy Vessels: {inp_ca}
- Thalassemia Result: {inp_thal}

RECOMMENDED CLINICAL ACTION PLAN:
{'- Schedule invasive coronary angiography / catheterization' if patient_pred == 1 else '- Routine annual preventive cardiovascular screening'}
{'- Prescribe statin therapy and anti-anginal regimen' if patient_pred == 1 else '- Maintain Mediterranean dietary regimen and 150 min/wk aerobic exercise'}
================================================================================
"""
    st.download_button(
        label="📄 Download Patient Diagnostic Report (.txt / summary)",
        data=report_text,
        file_name=f"CardioVision_Diagnostic_Report_{int(time.time())}.txt",
        mime="text/plain",
        use_container_width=True
    )


# =============================================================================
# TAB 2: MEDICAL CARDIAC ECG WAVEFORM & IMAGE STUDIO
# =============================================================================
with tab_image:
    st.markdown("### 🔬 Medical Cardiac ECG Waveform & Image Processing Studio")
    st.write("Simulate live 12-lead equivalent cardiac waveforms, analyze P-Q-R-S-T intervals in real time, and process ECG scan strips with multi-stage medical filters.")

    ecg_subtab1, ecg_subtab2 = st.tabs(["📈 Live Interactive ECG Waveform Simulator", "🖼️ Medical Image Enhancement & Feature Extraction"])

    with ecg_subtab1:
        st.markdown("#### 1️⃣ Live Interactive Electrocardiogram (ECG) Simulator")
        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            ecg_condition = st.selectbox(
                "Select Cardiac Rhythm:",
                ["Normal Sinus Rhythm", "Myocardial Infarction (STEMI)", "Atrial Fibrillation", "Ventricular Tachycardia"]
            )
        with w_col2:
            ecg_bpm = st.slider("Heart Rate (BPM):", 45, 180, 72)
        with w_col3:
            ecg_noise = st.slider("Electrode Artifact / Noise:", 0.0, 0.15, 0.02, step=0.01)

        t_sig, sig_val = synthesize_ecg_signal(condition=ecg_condition, duration=3.5, fs=250, bpm=ecg_bpm, noise_level=ecg_noise)

        # Detect R-peaks
        peaks, _ = find_peaks(sig_val, height=0.4, distance=35)

        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(x=t_sig, y=sig_val, mode="lines", name="Lead II ECG", line=dict(color="#DC2626", width=2.2)))

        if len(peaks) > 0:
            fig_ecg.add_trace(go.Scatter(
                x=t_sig[peaks], y=sig_val[peaks],
                mode="markers+text",
                name="R-Peaks",
                marker=dict(color="#2563EB", size=9, symbol="diamond"),
                text=[f"R{i+1}" for i in range(len(peaks))],
                textposition="top center"
            ))

        fig_ecg.update_layout(
            title=f"<b>Simulated Diagnostic ECG (Lead II)</b> - {ecg_condition} @ {ecg_bpm} BPM",
            xaxis_title="Time (seconds)",
            yaxis_title="Voltage Amplitude (mV)",
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_ecg, use_container_width=True)

        # Waveform Metrics
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        calc_rr = (60.0 / ecg_bpm) * 1000
        m_c1.metric("⏱️ Mean RR Interval", f"{calc_rr:.0f} ms", f"{ecg_bpm} BPM")
        m_c2.metric("⚡ Estimated QRS Width", "88 ms" if "Normal" in ecg_condition else "142 ms (Widened)", "Repolarization")
        m_c3.metric("📈 ST Segment Deviation", "+2.8 mm" if "STEMI" in ecg_condition else "0.0 mm", "Ischemic marker")
        m_c4.metric("🩺 Rhythm Regularity", "Irregular" if "Fibrillation" in ecg_condition else "Regular Sinus", "Cardiac pacemaker")

    with ecg_subtab2:
        st.markdown("#### 2️⃣ Multi-Stage Medical Image Processing Pipeline")
        st.write("Upload a patient ECG image strip or process synthetic cardiac scans across noise filtering, CLAHE contrast enhancement, and HOG/LBP feature extraction.")

        im_ctrl1, im_ctrl2, im_ctrl3 = st.columns([2, 1, 1])
        with im_ctrl1:
            uploaded_ecg = st.file_uploader("Upload patient ECG strip (PNG, JPG):", type=["png", "jpg", "jpeg"])
        with im_ctrl2:
            ecg_preset_img = st.selectbox("Or Choose Scan Preset:", ["Normal Sinus Rhythm", "Myocardial Infarction", "Arrhythmia"])
        with im_ctrl3:
            img_filter = st.selectbox("Noise Reduction Filter:", ["Gaussian Blur", "Median Blur", "Bilateral Filter", "None"])

        if uploaded_ecg is not None:
            file_bytes = np.asarray(bytearray(uploaded_ecg.read()), dtype=np.uint8)
            if HAS_OPENCV:
                base_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                base_img = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
        else:
            base_img = generate_sample_ecg_image(condition=ecg_preset_img)

        res_img, gray_img, filt_img, enh_img, thresh_img = preprocess_ecg_pipeline(
            base_img, target_size=(140, 140), filter_mode=img_filter
        )

        st.markdown("##### 🖼️ Visual Pipeline Progression")
        ip_c1, ip_c2, ip_c3, ip_c4 = st.columns(4)
        with ip_c1:
            st.image(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB) if HAS_OPENCV else base_img, caption="1. Raw Input ECG Strip", use_container_width=True)
        with ip_c2:
            st.image(gray_img, caption="2. Grayscale Normalized", use_container_width=True)
        with ip_c3:
            st.image(enh_img, caption=f"3. {img_filter} & CLAHE", use_container_width=True)
        with ip_c4:
            st.image(thresh_img, caption="4. Waveform Segmentation", use_container_width=True)

        st.markdown("##### 🔬 Feature Extraction (HOG Gradients & Morphometry)")
        if HAS_SKIMAGE:
            hog_vec, hog_vis = hog(enh_img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(2, 2), visualize=True)
        else:
            hog_vec = np.gradient(enh_img)[0].ravel()[:120]
            hog_vis = enh_img

        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            st.image(hog_vis, caption="HOG Oriented Gradient Waveform Vector Map", use_container_width=True)
        with f_col2:
            st.markdown("**Extracted Visual Descriptors:**")
            st.metric("Waveform Pixel Density", f"{(np.count_nonzero(thresh_img) / thresh_img.size)*100:.2f}%")
            st.metric("Signal Intensity Variance", f"{np.std(enh_img):.2f}")
            st.info(f"**Total Extracted Feature Dimensions:** `{len(hog_vec) + 4}` dimensional embedding ready for classification.")


# =============================================================================
# TAB 3: DATA PREPROCESSING & FEATURE INTELLIGENCE
# =============================================================================
with tab_data:
    st.markdown("### 📊 Dataset Preprocessing & Exploratory Feature Intelligence")
    st.write("Inspect the Cleveland Clinic Heart Disease benchmark dataset, statistical distributions, missing value checks, and interactive 3D feature spaces.")

    d_tab1, d_tab2, d_tab3 = st.tabs(["📋 Dataset Explorer & Audit", "🔥 Interactive Correlation Heatmap", "🌐 3D Feature Space & Distributions"])

    with d_tab1:
        st.markdown("#### Clinical Dataset Overview")
        st.dataframe(df_clinical.head(15), use_container_width=True)

        aud_c1, aud_c2, aud_c3, aud_c4 = st.columns(4)
        aud_c1.metric("Total Patients", df_clinical.shape[0])
        aud_c2.metric("Clinical Features", df_clinical.shape[1] - 1)
        aud_c3.metric("Missing Values", int(df_clinical.isnull().sum().sum()), "0.0% Clean")
        aud_c4.metric("Heart Disease Prevalence", f"{(df_clinical['target'].mean()*100):.1f}%", f"{df_clinical['target'].sum()} Positive cases")

        st.markdown("#### Statistical Feature Summary")
        st.dataframe(df_clinical.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]], use_container_width=True)

    with d_tab2:
        st.markdown("#### Interactive Pearson Correlation Heatmap")
        corr_matrix = df_clinical.corr().round(3)

        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Interactive Feature Correlation Matrix (Cleveland Dataset)"
        )
        fig_heat.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Top Risk Factor Correlations with Heart Disease Target")
        corr_target = corr_matrix["target"].drop("target").sort_values(ascending=False)
        fig_target_corr = px.bar(
            x=corr_target.values,
            y=corr_target.index,
            orientation="h",
            color=corr_target.values,
            color_continuous_scale="Viridis",
            labels={"x": "Pearson Correlation (r)", "y": "Clinical Feature"},
            title="Feature Correlation Strength with Cardiac Disease Diagnosis"
        )
        fig_target_corr.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_target_corr, use_container_width=True)

    with d_tab3:
        st.markdown("#### Interactive 3D Clinical Feature Space")
        st.caption("Rotate, pan, and zoom to explore how patients cluster across Age, Max Heart Rate, and Serum Cholesterol:")

        df_plot = df_clinical.copy()
        df_plot["Diagnosis"] = df_plot["target"].map({0: "Healthy (0)", 1: "Heart Disease (1)"})

        fig_3d = px.scatter_3d(
            df_plot,
            x="age",
            y="thalach",
            z="chol",
            color="Diagnosis",
            color_discrete_map={"Healthy (0)": "#10B981", "Heart Disease (1)": "#EF4444"},
            hover_data=["trestbps", "oldpeak", "cp"],
            labels={"age": "Age (years)", "thalach": "Max HR (bpm)", "chol": "Cholesterol (mg/dl)"},
            title="3D Separation: Age vs Max HR vs Serum Cholesterol"
        )
        fig_3d.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("#### Cohort Distribution Comparison (Healthy vs Disease)")
        dist_feat = st.selectbox("Select Feature for Distribution Comparison:", ["trestbps", "chol", "thalach", "oldpeak", "age"])
        fig_box = px.box(
            df_plot,
            x="Diagnosis",
            y=dist_feat,
            color="Diagnosis",
            points="all",
            color_discrete_map={"Healthy (0)": "#10B981", "Heart Disease (1)": "#EF4444"},
            title=f"Distribution of {FEATURE_LABELS[dist_feat]} by Clinical Target"
        )
        fig_box.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)


# =============================================================================
# TAB 4: MODEL TRAINING & BENCHMARKING STUDIO
# =============================================================================
with tab_train:
    st.markdown("### 🏋️ Machine Learning Model Training & Comparative Performance")
    st.write("Compare all 6 supervised learning algorithms + Soft Voting Ensemble across **Accuracy**, **Precision**, **Recall (Sensitivity)**, **Specificity**, **F1-Score**, and **ROC-AUC**.")

    # Comparative Table
    comp_data = []
    for m_name, m_val in metrics_dict.items():
        comp_data.append({
            "Algorithm": m_name,
            "Accuracy (%)": f"{m_val['Accuracy']*100:.2f}%",
            "Precision (%)": f"{m_val['Precision']*100:.2f}%",
            "Recall / Sens. (%)": f"{m_val['Recall']*100:.2f}%",
            "Specificity (%)": f"{m_val['Specificity']*100:.2f}%",
            "F1-Score": f"{m_val['F1-Score']:.4f}",
            "ROC-AUC": f"{m_val['ROC-AUC']:.4f}",
            "5-Fold CV Accuracy": f"{m_val['CV_Mean']*100:.1f}% ± {m_val['CV_Std']*100:.1f}%"
        })
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True)

    t_sub1, t_sub2, t_sub3, t_sub4 = st.tabs([
        "📊 Multi-Metric Comparison Charts",
        "📈 Interactive ROC & PR Curves",
        "🎯 Confusion Matrix Breakdown",
        "⚖️ Decision Threshold Calibration"
    ])

    with t_sub1:
        # Grouped Bar Chart
        plot_rows = []
        for k, v in metrics_dict.items():
            short = k.replace("Classifier", "").replace("Machine", "").strip()
            plot_rows.append({"Algorithm": short, "Accuracy": v["Accuracy"]*100, "Precision": v["Precision"]*100, "Recall": v["Recall"]*100, "F1-Score": v["F1-Score"]*100, "ROC-AUC": v["ROC-AUC"]*100})
        df_chart = pd.DataFrame(plot_rows)
        df_melt = df_chart.melt(id_vars="Algorithm", var_name="Metric", value_name="Score (%)")

        fig_bar = px.bar(
            df_melt,
            x="Algorithm",
            y="Score (%)",
            color="Metric",
            barmode="group",
            title="Cross-Algorithm Multi-Metric Performance Comparison",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_bar.update_layout(height=420, yaxis=dict(range=[40, 105]), margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Radar / Spider Chart
        categories = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
        fig_radar = go.Figure()
        for k, v in metrics_dict.items():
            short = k.split()[0]
            vals = [v["Accuracy"]*100, v["Precision"]*100, v["Recall"]*100, v["Specificity"]*100, v["F1-Score"]*100]
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',
                name=short,
                opacity=0.25
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[40, 100])),
            showlegend=True,
            title="Multi-Dimensional Diagnostic Radar Comparison",
            height=440
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with t_sub2:
        roc_col1, roc_col2 = st.columns(2)
        with roc_col1:
            fig_roc = go.Figure()
            for m_name, m_val in metrics_dict.items():
                fig_roc.add_trace(go.Scatter(
                    x=m_val["FPR"], y=m_val["TPR"],
                    mode="lines",
                    name=f"{m_name.split()[0]} (AUC = {m_val['ROC-AUC']:.3f})"
                ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess (0.50)", line=dict(dash="dash", color="gray")))
            fig_roc.update_layout(
                title="<b>Receiver Operating Characteristic (ROC) Curves</b>",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity / Recall)",
                template="plotly_white",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with roc_col2:
            fig_pr = go.Figure()
            for m_name, m_val in metrics_dict.items():
                fig_pr.add_trace(go.Scatter(
                    x=m_val["PR_Recall"], y=m_val["PR_Precision"],
                    mode="lines",
                    name=f"{m_name.split()[0]}"
                ))
            fig_pr.update_layout(
                title="<b>Precision-Recall (PR) Curves</b>",
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision (Positive Predictive Value)",
                template="plotly_white",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    with t_sub3:
        cm_data = metrics_dict[selected_model_name]["Confusion Matrix"]
        tn, fp, fn, tp = cm_data.ravel()

        cm_c1, cm_c2 = st.columns([1, 1])
        with cm_c1:
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                x=["Predicted: Healthy (0)", "Predicted: Disease (1)"],
                y=["Actual: Healthy (0)", "Actual: Disease (1)"],
                color_continuous_scale="Reds",
                title=f"Diagnostic Confusion Matrix: <b>{selected_model_name}</b>"
            )
            fig_cm.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with cm_c2:
            st.markdown(f"#### Diagnostic Triage Metrics ({selected_model_name}):")
            st.write(f"- 🟢 **True Negatives (TN):** `{tn}` healthy patients correctly classified.")
            st.write(f"- 🟡 **False Positives (FP):** `{fp}` false alarms (Healthy diagnosed as diseased).")
            st.write(f"- 🔴 **False Negatives (FN):** `{fn}` missed cardiac cases *(Critical Medical Risk)*.")
            st.write(f"- 🟣 **True Positives (TP):** `{tp}` heart disease patients accurately flagged.")

            st.info("💡 **Clinical Safety Priority:** False Negatives (FN) directly jeopardize patient lives. A model with high **Recall/Sensitivity** minimizes FN.")

    with t_sub4:
        st.markdown("#### ⚖️ Dynamic Decision Threshold Tuning")
        st.write("In cardiology screening, adjusting the probability decision cutoff allows clinicians to trade off false alarms vs missed cases.")

        thresh_val = st.slider("Classification Probability Threshold:", 0.10, 0.90, 0.50, step=0.05)
        
        prob_test = metrics_dict[selected_model_name]["Probabilities"]
        if prob_test is not None:
            custom_preds = (prob_test >= thresh_val).astype(int)
            c_rec = recall_score(y_test_series, custom_preds, zero_division=0)
            c_prec = precision_score(y_test_series, custom_preds, zero_division=0)
            c_f1 = f1_score(y_test_series, custom_preds, zero_division=0)
            c_cm = confusion_matrix(y_test_series, custom_preds)
            c_tn, c_fp, c_fn, c_tp = c_cm.ravel()

            ct_c1, ct_c2, ct_c3, ct_c4 = st.columns(4)
            ct_c1.metric("Adjusted Recall (Sensitivity)", f"{c_rec*100:.1f}%", f"{c_tp} True Positives")
            ct_c2.metric("Adjusted Precision", f"{c_prec*100:.1f}%", f"{c_fp} False Positives")
            ct_c3.metric("Adjusted F1-Score", f"{c_f1:.4f}")
            ct_c4.metric("Missed Cases (FN)", f"{c_fn}", "Target: 0 Missed", delta_color="inverse")
        else:
            st.warning("Selected model does not output continuous calibrated probabilities.")


# =============================================================================
# TAB 5: PYTHON CODE EDITOR & LIVE SANDBOX
# =============================================================================
with tab_editor:
    st.markdown("### 💻 Live Python Code Editor & Sandbox")
    st.write("Type or edit Python machine learning code directly in the sandbox below and click **▶️ Compile & Run Program** to see live output, stdout capture, and plot rendering.")

    TEMPLATE_SCRIPTS = {
        "1. Complete 5-Stage Heart Disease ML Pipeline": '''# Complete 5-Stage Heart Disease ML Pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

print(">>> STAGE 1: Loading Cleveland Clinical Dataset...")
df = pd.read_csv("heart_disease_data.csv")
print(f"Dataset shape: {df.shape} (Patients: {len(df)})")

print(">>> STAGE 2: Preprocessing & Normalization...")
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(">>> STAGE 3: Feature Extraction Analysis...")
corrs = df.corr()["target"].drop("target").sort_values(ascending=False)
print("Top 3 positive risk factors:")
print(corrs.head(3))

print(">>> STAGE 4: Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train_s, y_train)

print(">>> STAGE 5: Performance Evaluation...")
preds = rf.predict(X_test_s)
acc = accuracy_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"F1-Score  : {f1:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
''',

        "2. 6-Algorithm Comparative Benchmark & Cross-Validation": '''# Benchmarking All Core Machine Learning Algorithms
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

df = pd.read_csv("heart_disease_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "SVM": SVC(kernel="rbf", C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
}

print(f"{'Algorithm':<20} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 58)
for name, m in models.items():
    m.fit(X_train_s, y_train)
    p = m.predict(X_test_s)
    acc = accuracy_score(y_test, p)
    rec = recall_score(y_test, p)
    f1 = f1_score(y_test, p)
    print(f"{name:<20} | {acc*100:6.2f}%    | {rec*100:6.2f}%    | {f1:6.4f}")
''',

        "3. ECG Waveform Synthesis & Peak Detection Engine": '''# ECG Signal Synthesis and Automated R-Peak Detection
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Generate simulated patient Lead II ECG signal
t = np.linspace(0, 3, 750)
baseline = 0
signal = np.zeros_like(t)

for bt in [0.2, 0.95, 1.7, 2.45]:
    dt = t - bt
    p = 0.20 * np.exp(-((dt + 0.18) ** 2) / 0.001)
    q = -0.15 * np.exp(-((dt + 0.05) ** 2) / 0.0003)
    r = 1.20 * np.exp(-(dt ** 2) / 0.0005)
    s = -0.30 * np.exp(-((dt - 0.05) ** 2) / 0.0003)
    t_wave = 0.35 * np.exp(-((dt - 0.22) ** 2) / 0.003)
    signal += (p + q + r + s + t_wave)

signal += np.random.normal(0, 0.02, len(t))
peaks, _ = find_peaks(signal, height=0.5, distance=50)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t, signal, color='red', lw=1.8, label="ECG Lead II Tracing")
ax.plot(t[peaks], signal[peaks], "bo", markersize=8, label=f"Detected R-Peaks ({len(peaks)})")
ax.set_title("Synthesized Diagnostic ECG (Lead II) with Automated Peak Detection")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Voltage (mV)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

print(f"Detected {len(peaks)} R-peaks at times: {np.round(t[peaks], 2)} seconds")
print(f"Calculated Heart Rate: {len(peaks) * 20} BPM")
''',

        "4. Custom Python Scratchpad": '''# Custom Machine Learning Experimentation Scratchpad
import sys
import numpy as np
import pandas as pd

print("Python Environment:", sys.version.split()[0])
sample_data = np.random.normal(120, 15, 10)
print("Simulated Blood Pressure Readings:", np.round(sample_data, 1))
print("Cohort Mean SBP:", np.round(np.mean(sample_data), 2), "mm Hg")
'''
    }

    selected_code_template = st.selectbox("Select Pre-loaded Code Template:", list(TEMPLATE_SCRIPTS.keys()))

    if "user_code" not in st.session_state or st.session_state.get("current_tpl") != selected_code_template:
        st.session_state["user_code"] = TEMPLATE_SCRIPTS[selected_code_template]
        st.session_state["current_tpl"] = selected_code_template

    code_content = st.text_area(
        "Python Code Editor:",
        value=st.session_state["user_code"],
        height=320,
        help="Edit the code and click 'Compile & Run Program'."
    )

    btn_e1, btn_e2 = st.columns([1, 4])
    with btn_e1:
        run_script_btn = st.button("▶️ Compile & Run Program", type="primary", use_container_width=True)
    with btn_e2:
        if st.button("🔄 Reset Code to Template Default", use_container_width=False):
            st.session_state["user_code"] = TEMPLATE_SCRIPTS[selected_code_template]
            st.rerun()

    if run_script_btn:
        st.markdown("#### 🖥️ Console Output & Execution Logs")
        t_start = time.time()

        # Syntax Check
        try:
            compiled_prog = compile(code_content, "<sandbox_session>", "exec")
            syntax_ok = True
        except SyntaxError as s_err:
            syntax_ok = False
            st.error(f"❌ **Syntax Compilation Error (Line {s_err.lineno}):** {s_err.msg}")
            st.code(s_err.text or "", language="python")

        if syntax_ok:
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            prev_out, prev_err = sys.stdout, sys.stderr

            sandbox_globals = {
                "__name__": "__main__",
                "np": np, "pd": pd, "plt": plt, "sns": sns, "px": px, "go": go,
                "find_peaks": find_peaks,
                "StandardScaler": StandardScaler,
                "train_test_split": train_test_split,
                "cross_val_score": cross_val_score,
                "accuracy_score": accuracy_score,
                "precision_score": precision_score,
                "recall_score": recall_score,
                "f1_score": f1_score,
                "confusion_matrix": confusion_matrix,
                "classification_report": classification_report,
                "SVC": SVC,
                "RandomForestClassifier": RandomForestClassifier,
                "KNeighborsClassifier": KNeighborsClassifier,
                "DecisionTreeClassifier": DecisionTreeClassifier,
                "LogisticRegression": LogisticRegression
            }

            try:
                sys.stdout = buf_out
                sys.stderr = buf_err
                exec(compiled_prog, sandbox_globals)
                runtime_ex = None
            except Exception:
                runtime_ex = traceback.format_exc()
            finally:
                sys.stdout = prev_out
                sys.stderr = prev_err

            elapsed_sec = time.time() - t_start
            text_stdout = buf_out.getvalue()

            if runtime_ex:
                st.error(f"⚠️ **Runtime Error during execution ({elapsed_sec:.2f}s):**")
                st.code(runtime_ex, language="python")
            else:
                st.success(f"✅ **Program compiled & executed cleanly in {elapsed_sec:.3f} seconds!**")

            if text_stdout.strip():
                st.markdown(f"<div class='terminal-box'>{text_stdout}</div>", unsafe_allow_html=True)
            elif not runtime_ex:
                st.info("Program finished with no text output. (Use `print()` to display output)")

            fig_handles = plt.get_fignums()
            if fig_handles:
                st.markdown("#### 📈 Rendered Matplotlib / Seaborn Visualizations:")
                for f_num in fig_handles:
                    f_obj = plt.figure(f_num)
                    st.pyplot(f_obj)
                plt.close("all")


# =============================================================================
# TAB 6: CLINICAL KNOWLEDGE BASE & VIVA GUIDE
# =============================================================================
with tab_guide:
    st.markdown("### 📖 Clinical Knowledge Base, Mathematical Foundations & Viva Voce Guide")

    st.markdown("""
    #### 1. Clinical Diagnostic Feature Dictionary
    | Feature Key | Clinical Name | Normal / Reference Range | Pathophysiological Significance |
    | :--- | :--- | :--- | :--- |
    | `age` | Patient Age | 18 – 100 years | Endothelial dysfunction and atherosclerotic burden increase exponentially with age. |
    | `sex` | Biological Sex | 1 = Male, 0 = Female | Estrogen confers premenopausal protective vascular effects in females. |
    | `cp` | Chest Pain Type | 0 = Typical, 1 = Atypical, 2 = Non-Anginal, 3 = Asymptomatic | Retrosternal pressure provoked by exertion reflects coronary blood flow mismatch. |
    | `trestbps` | Resting Blood Pressure | < 120 / 80 mm Hg | Elevated systolic pressure causes arterial shearing stress and ventricular afterload. |
    | `chol` | Serum Total Cholesterol | < 200 mg/dl | Elevated low-density lipoprotein particles form the core of atherosclerotic plaque. |
    | `fbs` | Fasting Blood Sugar | <= 120 mg/dl | Hyperglycemia promotes oxidative stress and accelerates vascular calcification. |
    | `restecg` | Resting Electrocardiogram | 0 = Normal, 1 = ST-T wave, 2 = LVH | Baseline repolarization anomalies indicate chronic ischemic injury or hypertrophy. |
    | `thalach` | Maximum Heart Rate | 220 - Age (Target bpm) | Chronotropic incompetence (inability to reach target HR) flags coronary insufficiency. |
    | `exang` | Exercise Induced Angina | 0 = No, 1 = Yes | Myocardial ischemia provoked by exercise stress. |
    | `oldpeak` | ST Segment Depression | < 1.0 mm | Hallmark electrocardiographic sign of subendocardial myocardial ischemia. |
    | `slope` | ST Peak Exercise Slope | 0 = Upsloping, 1 = Flat, 2 = Downsloping | Downsloping and flat slopes carry strong diagnostic significance for CAD. |
    | `ca` | Fluoroscopy Vessels | 0 (No major occlusion) | Number of major coronary arteries with >= 50% stenosis under contrast fluoroscopy. |
    | `thal` | Thalassemia Perfusion | 1 = Normal, 2 = Fixed, 3 = Reversible | Reversible defect indicates salvageable ischemic tissue under stress. |

    ---

    #### 2. Mathematical Formulations of Machine Learning Performance Metrics
    - **Accuracy:** Proportion of all diagnoses that were correct:
      $$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$
    - **Recall / Sensitivity:** Proportion of actual heart disease patients correctly identified:
      $$\\text{Sensitivity} = \\frac{TP}{TP + FN}$$
    - **Specificity:** Proportion of healthy patients correctly identified as healthy:
      $$\\text{Specificity} = \\frac{TN}{TN + FP}$$
    - **Precision (Positive Predictive Value):** Confidence that a positive diagnosis is true:
      $$\\text{Precision} = \\frac{TP}{TP + FP}$$
    - **F1-Score:** Harmonic mean of Precision and Sensitivity:
      $$F_1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2TP}{2TP + FP + FN}$$

    ---

    #### 3. Frequently Asked Viva Voce Questions & Model Explanations
    1. **Q: Why is Recall/Sensitivity more important than Accuracy in medical AI?**
       *A: In cardiology, a False Negative (discharging an untreated heart attack patient) can lead to fatal acute coronary events. High sensitivity minimizes False Negatives.*
    2. **Q: Why do we use StandardScaler (Z-Score Normalization)?**
       *A: Distance-based classifiers like SVM and KNN and gradient-based models like Logistic Regression are sensitive to feature scales (e.g., Cholesterol in hundreds vs oldpeak < 6).*
    3. **Q: How does Random Forest prevent overfitting?**
       *A: It builds an ensemble of decorrelated Decision Trees via bootstrap aggregating (bagging) and random feature subspace sampling at each split.*
    4. **Q: What is the benefit of the Soft Voting Ensemble?**
       *A: It aggregates calibrated class probabilities across diverse model architectures (SVM, Random Forest, Logistic Regression), reducing individual model variance.*
    """)
