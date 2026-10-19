# ❤️ CardioVision: Heart Disease Prediction AI Studio

An end-to-end, single-frame interactive web application and machine learning engine for **Heart Disease Prediction & Cardiac Intelligence** built with **Python**, **Streamlit**, **Scikit-Learn**, **XGBoost**, and **Plotly**.

---

## 🌟 Key Features & Diagnostic Workflow

The application covers the complete 6-stage medical AI diagnostic suite in **one unified web application frame**:

1. **🩺 Patient Vitals & Live AI Diagnosis**:
   - Comprehensive input for all 13 Cleveland Clinic physiological vitals (Age, Sex, Chest Pain, Resting BP, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression oldpeak, ST Slope, Fluoroscopy Vessels, Thalassemia Scan).
   - 5 Quick Clinical Presets (Healthy Athlete, Hypertensive Moderate Risk, High-Risk Ischemia, Acute STEMI/Infarct, Diabetic Female).
   - Dynamic **Plotly Risk Speedometer Gauge** (0% to 100% Risk Probability).
   - **Multi-Model Consensus Bar** displaying individual diagnoses across all 7 algorithms.
   - **Local Feature Risk Attribution (Explainability)** showing patient Z-score deviations from the healthy cohort.
   - **Downloadable Clinical Diagnostic Summary Report (.txt / summary)** with physician recommendations.

2. **🔬 Cardiac ECG Waveform & Image Processing Studio**:
   - **Live Interactive ECG Waveform Simulator**: Adjustable Heart Rate (45–180 BPM), Cardiac Rhythm (Normal Sinus, STEMI Infarct, Atrial Fibrillation, Ventricular Tachycardia), and Noise artifacts.
   - **Automated P-Q-R-S-T R-Peak Detection**: Real-time interactive peak markers and calculated metrics (RR interval, QRS width, ST deviation).
   - **Multi-Stage Medical Image Pipeline**: Grayscale conversion, noise filtering (Gaussian, Median, Bilateral), CLAHE contrast equalization, Otsu waveform segmentation, and HOG (Histogram of Oriented Gradients) / LBP feature extraction.

3. **📊 Data Preprocessing & Feature Intelligence**:
   - Interactive Cleveland Clinic dataset table with missing value audit and statistical feature distributions.
   - **Interactive Pearson Correlation Heatmap** with hover tooltips and target correlation rankings.
   - **Interactive 3D Feature Space Scatter**: Age vs Max Heart Rate vs Serum Cholesterol with interactive 3D rotation and patient hover cards.
   - Cohort distribution box plots comparing healthy vs diseased distributions.

4. **🏋️ Model Training & Benchmarking Studio**:
   - Train and benchmark **6 Supervised Classifiers + 1 Soft Voting Ensemble**:
     - 🛡️ **Support Vector Machine (SVM)** (RBF Kernel)
     - 🌲 **Random Forest Classifier** (Bagged Decision Tree Ensemble)
     - 🚀 **XGBoost Classifier** (Gradient Boosted Decision Trees)
     - 👥 **K-Nearest Neighbors (KNN)** (Distance-Weighted)
     - 🌳 **Decision Tree Classifier** (Interpretable Clinical Rules)
     - 📈 **Logistic Regression** (Calibrated Probabilities)
     - 🗳️ **Soft Voting Ensemble** (Consensus Blending)
   - Live **Hyperparameter Tuning Sidebar** (Split ratio, Tree count, Depth, SVM C, KNN k).
   - Multi-metric comparative tables with medals and grouped bar charts.
   - **Interactive Multi-Dimensional Radar / Spider Chart**.
   - **Interactive ROC (Receiver Operating Characteristic) Curves & AUC Scores**.
   - **Interactive Precision-Recall (PR) Curves**.
   - **Diagnostic Confusion Matrix Heatmap** with breakdown of TN, FP, FN (critical misses), and TP.
   - **5-Fold Stratified Cross-Validation** with Mean ± Standard Deviation confidence intervals.
   - **Dynamic Decision Threshold Slider** (0.10 to 0.90) for sensitivity-specificity clinical calibration.

5. **💻 Live In-Browser Python Code Sandbox**:
   - Type or edit Python ML code directly in the web browser.
   - 4 pre-built runnable scripts (Full Pipeline, 6-Algorithm Benchmark, ECG Peak Detection, Custom Scratchpad).
   - **"▶️ Compile & Run Program"** button with live terminal stdout capture, error tracebacks, and automatic Matplotlib/Seaborn plot rendering.

6. **📖 Clinical Knowledge Base & Viva Guide**:
   - Full clinical definitions, reference ranges, and pathophysiological explanations for all 13 features.
   - Mathematical formulations of evaluation metrics (Accuracy, Sensitivity/Recall, Specificity, Precision, F1-Score).
   - Frequently asked Viva Voce interview questions and answers.

---

## 📁 Project File Structure

```
Heart predict/
│
├── app.py                   # Unified Single-Page Web Application (Streamlit + Plotly)
├── train_model.py           # Standalone CLI ML Training & Evaluation Engine
├── heart_disease_data.csv   # Cleveland Clinic Heart Disease benchmark dataset
├── best_heart_disease_model.pkl # Serialized best trained model
├── scaler.pkl               # Serialized StandardScaler
├── requirements.txt         # Required Python packages
├── run_app.bat              # 1-Click Windows Batch launcher
├── run_app.ps1              # 1-Click Windows PowerShell launcher
└── README.md                # Comprehensive documentation & Viva guide
```

---

## 🚀 How to Compile and Run the Code

### Step 1: Open Project Folder in Terminal
```powershell
cd "c:\Users\Lenovo\OneDrive\Desktop\Heart predict"
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Verify Syntax Compilation
Python bytecode syntax check:
```powershell
# 1. Compile standalone training script
python -m py_compile train_model.py

# 2. Compile Streamlit application
python -m py_compile app.py
```

### Step 4: Run the Standalone Training CLI Engine
```powershell
python train_model.py
```

### Step 5: Launch the Interactive Single-Page Web Application
```powershell
python -m streamlit run app.py
```
👉 Access the application at **`http://localhost:8501`** *(or simply double-click `run_app.bat` on Windows)*.

---

## 📋 Clinical Diagnostic Features (Cleveland Dataset)

| Feature | Clinical Name | Normal / Reference Range | Pathophysiological Significance |
| :--- | :--- | :--- | :--- |
| `age` | Patient Age | 18 – 100 years | Atherosclerotic plaque accumulation increases exponentially with age. |
| `sex` | Biological Sex | 1 = Male, 0 = Female | Premenopausal estrogen provides vascular protection in females. |
| `cp` | Chest Pain Type | 0 = Typical, 1 = Atypical, 2 = Non-Anginal, 3 = Asymptomatic | Retrosternal exertional discomfort signals coronary ischemia. |
| `trestbps` | Resting Blood Pressure | < 120 / 80 mm Hg | Elevated systolic pressure causes arterial shearing and myocardial hypertrophy. |
| `chol` | Serum Cholesterol | < 200 mg/dl | Elevated LDL atherogenic particles form the core of arterial plaques. |
| `fbs` | Fasting Blood Sugar | <= 120 mg/dl | Hyperglycemia promotes oxidative stress and accelerates endothelial dysfunction. |
| `restecg` | Resting ECG | 0 = Normal, 1 = ST-T Wave, 2 = LVH | Conduction anomalies indicate prior ischemic injury or ventricular hypertrophy. |
| `thalach` | Maximum Heart Rate | $220 - \text{Age}$ (Target bpm) | Inability to achieve target heart rate reflects chronotropic incompetence. |
| `exang` | Exercise Induced Angina | 0 = No, 1 = Yes | Myocardial ischemia provoked by exercise stress. |
| `oldpeak` | ST Depression | < 1.0 mm | Classic electrocardiographic indicator of subendocardial ischemia. |
| `slope` | ST Segment Slope | 0 = Upsloping, 1 = Flat, 2 = Downsloping | Downsloping/flat ST segments indicate high coronary artery disease probability. |
| `ca` | Fluoroscopy Vessels | 0 (No stenosis) | Major coronary vessels with $\ge 50\%$ occlusion under fluoroscopy. |
| `thal` | Thalassemia Perfusion | 1 = Normal, 2 = Fixed, 3 = Reversible | Reversible defect flags active inducible ischemia under treadmill stress. |
| `target` | Diagnosis | 0 = Healthy, 1 = Heart Disease | Ground truth clinical diagnosis. |

---

## 🎓 Viva Voce Cheat Sheet

1. **Q: Why is Recall / Sensitivity the most critical metric in cardiology AI?**
   - *A:* A False Negative means a heart disease patient is mistakenly diagnosed as healthy and discharged without treatment, risking fatal coronary events. Maximizing Recall ensures that true positive cardiac cases are detected.
2. **Q: Why is `StandardScaler` essential before feeding vitals into SVM/KNN/Logistic Regression?**
   - *A:* Features like Cholesterol (~250 mg/dl) and ST Depression (~1.5 mm) operate on vastly different scales. Without Z-score normalization ($\mu=0, \sigma=1$), distance-based models would be overwhelmed by high-magnitude features.
3. **Q: What is the advantage of the Soft Voting Ensemble?**
   - *A:* It aggregates predicted class probabilities from diverse models (SVM, Random Forest, XGBoost, Logistic Regression), reducing individual model variance and yielding more reliable clinical classifications.
