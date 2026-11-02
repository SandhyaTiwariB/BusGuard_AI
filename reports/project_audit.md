# BusGuard AI — Complete Project and Codebase Audit Report

**Audit Date**: November 2, 2026  
**System Target**: BusGuard AI — AI-Assisted Multi-Sensor Busbar Condition-Monitoring System  
**Audit Scope**: Repository Structure, Raw Datasets, Existing Machine Learning Code, Calculation Scripts, and Architectural Recommendations  

---

## 1. Project Overview & System Scope

**BusGuard AI** is engineered to provide continuous, automated condition monitoring of electrical busbar joints. Loose, vibrating, or oxidised joints introduce elevated electrical contact resistance ($R_c$), resulting in localized Joule heating ($P = I^2 R_c$), thermal degradation, mechanical loosening from thermal cycling, micro-arcing, and catastrophic switchgear fire hazards.

To detect degradation before physical failure occurs, BusGuard AI fuses three complementary sensor modalities:
1. **Thermal Sensing**: Monitoring local joint temperature and ambient-to-joint differential ($\Delta T$).
2. **Electrical Measurements**: Monitoring load current ($I$) and contact voltage drops ($\Delta V$) to quantify contact resistance ($R_c$).
3. **Acoustic & Vibration Sensing**: Capturing high-frequency acoustic emissions and micro-vibrations indicative of contact fretting, electrical discharge, and micro-arcing.

---

## 2. Repository Structure & Workspace State

```text
BusGuard_AI/
│
├── data/
│   ├── raw/                                                     [Populated with all uploaded datasets]
│   │   ├── busguard_physical_data.jsonl                         [Primary multi-sensor stream]
│   │   ├── connector_degradation_testing.xlsx                   [Degradation benchmark workbook]
│   │   ├── connector_degradation_sheets/                        [4 extracted CSV sheets]
│   │   ├── Fault Simulation Dataset for 110 kV.../              [110 kV grid dataset & PowerFactory files]
│   │   ├── plc_industrial_control/                              [PLC SCADA dataset (CSV & JSON)]
│   │   ├── Fault_Simulation_Dataset_for_110_kV...zip            [Container archive]
│   │   ├── plc_industrial_control_archive.zip                   [Container archive]
│   │   └── fault-prediction-using-decision-tree...ipynb         [Reference notebook]
│   └── processed/                                               [Empty — ready for verified preprocessed data]
│
├── notebooks/
│   └── fault-prediction-using-decision-tree...ipynb             [Reference ML notebook]
│
├── ml/                                                          [Empty — designated for feature engineering & models]
│
├── backend/                                                     [Empty — designated for FastAPI inference service]
│
├── dashboard/                                                   [Empty — designated for monitoring UI]
│
├── models/                                                      [Empty — designated for serialized model checkpoints]
│
├── reports/
│   ├── dataset_audit.md                                         [Detailed dataset-level inspection report]
│   └── project_audit.md                                         [Complete project & codebase audit report]
│
└── README.md                                                    [Project architecture & folder documentation]
```

---

## 3. Comprehensive Code Inspection & Technical Evaluation

### 3.1. Inspection of `fault-prediction-using-decision-tree-algorithm.ipynb`

- **Location**: [`notebooks/fault-prediction-using-decision-tree-algorithm.ipynb`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/notebooks/fault-prediction-using-decision-tree-algorithm.ipynb)
- **File Size**: 5,474,227 bytes (122 cells: 63 code cells, 59 markdown cells with embedded base64 plots).
- **Core Intent**: Implements multiclass classification for 3-phase high-voltage power transmission line faults using Kaggle's external `classData.csv`.

#### Machine Learning Algorithms Implemented
The notebook implements five supervised classification algorithms from `scikit-learn` and `xgboost`:
1. **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
2. **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`)
3. **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`, 100 estimators)
4. **XGBoost Classifier** (`xgboost.XGBClassifier`)
5. **Support Vector Classifier (SVC)** (`sklearn.svm.SVC`)

#### Preprocessing & Pipeline Logic
- **Data Ingestion**: `pd.read_csv("../input/electrical-fault-detection-and-classification/classData.csv")`.
- **Target Construction**: String-concatenating 4 binary ground-truth indicators:
  $$\text{Fault\_Type} = \text{str}(G) + \text{str}(C) + \text{str}(B) + \text{str}(A)$$
- **Label Encoding**: `sklearn.preprocessing.LabelEncoder` applied to `Fault_Type`.
- **Train/Test Splitting**: `train_test_split(X, y, test_size=0.20, random_state=21)`.
- **Evaluation**: Accuracy scores, confusion matrices, and decision tree structure plotting via `sklearn.tree.plot_tree`.

#### Critical Flaws & Discovered Errors

> [!CAUTION]
> **Critical Flaw 1: Severe Target / Data Leakage**
> In Code Cells 7–10, the input data contains binary columns `G`, `C`, `B`, and `A` representing ground, line C, line B, and line A fault indicators. The target variable `Fault_Type` was derived directly from these four columns.
> In Code Cell 41, the feature matrix was defined as:
> ```python
> X = df_class.drop(['Fault_Type'], axis=1)
> y = df_class['Fault_Type']
> ```
> Because `G`, `C`, `B`, and `A` were **not dropped from $X$**, the feature matrix $X$ contained the exact binary components of the target label $y$. As a result, the models achieved 100% accuracy simply by memorizing the target columns rather than learning from physical current and voltage waveforms (`Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`). This is a classic textbook data-leakage defect.

> [!WARNING]
> **Defect 2: Model Evaluation Summary Index Mismatch**
> In Code Cell 51, the author constructs a model performance summary table:
> ```python
> models = pd.DataFrame({
>     'Model': ['Support Vector Machines', 'Logistic Regression', 'Random Forest', 'Decision Tree', 'XGBClassifier'],
>     'Training Accuracy': [log_train, svc_train, decision_train, random_forest_train, xgb_train],
>     'Model Accuracy Score': [log_accuracy, svc_accuracy, decision_accuracy, random_forest_accuracy, xgb_accuracy]
> })
> ```
> The array order for `Model` does not match the variables: index 0 is labeled `"Support Vector Machines"` but receives `log_train` (Logistic Regression), and index 1 is labeled `"Logistic Regression"` but receives `svc_train`.

> [!WARNING]
> **Defect 3: Deprecated and Removed Library Functions**
> - The notebook calls `from sklearn.metrics import plot_confusion_matrix`. This function was deprecated in scikit-learn 1.0 and completely removed in version 1.2+ (replaced by `ConfusionMatrixDisplay.from_estimator`).
> - The notebook uses `sns.distplot`, which has been deprecated in modern Seaborn in favor of `sns.histplot` or `sns.displot`.

> [!NOTE]
> **Defect 4: Missing Dependency and Hardcoded Environment Path**
> The input path `../input/electrical-fault-detection-and-classification/classData.csv` is hardcoded to Kaggle's container directory. The underlying CSV is not bundled with the notebook.

#### Reusable Code Components
- **Visualization Palette & Matplotlib Plotting Structure**: Code for multi-class confusion matrix plotting and distribution styling can be adapted for BusGuard AI's future exploratory analysis.
- **Scikit-learn Model Evaluation Structure**: The workflow structure (instantiation, cross-validation, precision/recall/F1 reporting) provides a clean template once applied to genuine, leakage-free features.

---

### 3.2. Inspection of PowerFactory Simulation Scripts

- **Location**: `data/raw/Fault Simulation Dataset for 110 kV Power Transmission Lines/`
- **Files**: `shc_calc.py` (3,368 bytes) and `ldf_calc.py` (2,832 bytes).
- **Core Intent**:
  - `shc_calc.py`: Automates short-circuit calculations across 5 transmission lines in a 110 kV grid model (`110kV_Grid.pfd`) according to IEC 60909 standards. Iterates through 4 fault types (`3psc`, `2psc`, `spgf`, `2pfg`) at 5% position increments.
  - `ldf_calc.py`: Automates balanced AC load-flow calculations while sweeping active power demands across loads in 0.1 MW steps.
- **External Dependencies**:
  - Hardcoded proprietary path: `C:\Program Files\DIgSILENT\PowerFactory 2024 SP2\Python\3.12`.
  - Imports proprietary module: `import powerfactory as pf`.
- **Reusability Assessment**: **Zero Reusability for BusGuard AI**.
  - These scripts cannot execute without a commercial DIgSILENT PowerFactory license.
  - The physics of 110 kV transmission grid short-circuits (kA-level network overcurrents) has no connection to local busbar joint degradation, contact micro-resistance, or acoustic emissions.

---

## 4. Dataset Categorization & Suitability Matrix

```text
+-----------------------------------------------------------------------------------------------------+
|                                      PROJECT ASSET EVALUATION                                       |
+------------------------------------+--------------------------+--------------------+----------------+
| Asset Name                         | Primary Modalities       | Domain Match       | Recommendation |
+------------------------------------+--------------------------+--------------------+----------------+
| busguard_physical_data.jsonl       | Acoustic, Temp, Current  | Direct (100%)      | Retain (Core)  |
| connector_degradation_testing.xlsx | Contact Resistance (mΩ)  | Analogue (90%)     | Retain (Bench) |
| fault_dataset.csv (110 kV grid)    | Grid kV, kA Transients   | Unrelated (0%)     | Exclude        |
| plc_industrial_control             | Hydraulic bar, PLC I/O   | Unrelated (0%)     | Exclude        |
| fault-prediction...ipynb           | Decision Tree Code       | Reference (20%)    | Code Template  |
| shc_calc.py / ldf_calc.py          | PowerFactory Grid Sweep  | Unrelated (0%)     | Exclude        |
+------------------------------------+--------------------------+--------------------+----------------+
```

### 1. Directly Relevant Busbar Joint-Condition Data
- **`busguard_physical_data.jsonl`**: The only physical dataset measuring the actual sensor signals planned for BusGuard AI (temperature, acoustic RMS, acoustic peak-to-peak, microphone DC level).
- *Limitation*: The current channel `I_A` is strictly 0.0 A, and ground-truth fault state labels are absent.

### 2. Supplementary Connector Degradation Data
- **`connector_degradation_testing.xlsx`** (and CSV extracts): Empirical physical measurements of electrical contact resistance ($R_c$) degradation over thermal cycling (up to 1000 hrs / 200 cycles) and damp heat (up to 1416 hrs).
- *Relevance*: Provides real physical failure curves and resistance boundaries ($R_c < 2\text{ m}\Omega$ healthy, $2\text{–}10\text{ m}\Omega$ warning, $> 10\text{ m}\Omega$ critical failure) that govern electrical joint deterioration.

### 3. Transmission-Line Fault Simulation Data
- **`Fault Simulation Dataset for 110 kV...`**: High-voltage transmission line short-circuit simulations.
- *Limitation*: Excluded from BusGuard AI modeling due to complete domain incompatibility.

### 4. PLC Industrial Control Data
- **`plc_industrial_control`**: Factory automation log tracking hydraulic presses, valves, and manufacturing counts.
- *Limitation*: Excluded from BusGuard AI modeling.

### 5. Existing ML Notebooks & Project Code
- **`fault-prediction-using...ipynb`**: Useful purely as a reference for structuring Python data pipelines and scikit-learn models once data leakage is corrected.

---

## 5. Project Recommendations for Software & ML Development

### 5.1. Dataset Strategy for Development
1. **Primary Operational Dataset**: Adopt `busguard_physical_data.jsonl` for signal processing, feature extraction, and real-time streaming pipeline verification.
2. **Physical Degradation Calibration**: Use the contact resistance progression from `connector_degradation_testing.xlsx` to establish empirical degradation stages and threshold values:
   - **Normal / Healthy Joint**: $R_c \le 2.0\text{ m}\Omega$ (Nominal operating resistance).
   - **Degradation / Loosening Inception**: $2.0\text{ m}\Omega < R_c \le 10.0\text{ m}\Omega$ (Accelerating contact oxidation).
   - **Severe Hazard / Critical**: $R_c > 10.0\text{ m}\Omega$ (Imminent thermal runaway / dielectric failure).
3. **Data Isolation**: Strictly exclude `fault_dataset.csv` and `plc_industrial_control` from all BusGuard AI training code.

### 5.2. Machine Learning Architecture Strategy
Because `busguard_physical_data.jsonl` is unlabeled and contains zero active current, standard supervised classifiers (like those in the reference notebook) cannot be trained directly. The recommended ML architecture is:

```
                               RAW SENSOR STREAM
                     (Temperature, Acoustic RMS, P2P, Mic)
                                       │
                                       ▼
                           SIGNAL FEATURE EXTRACTION
               ┌───────────────────────┼───────────────────────┐
               ▼                       ▼                       ▼
      Time-Domain Statistics    Acoustic Energy Burst    Thermal Rate of Rise
      (Rolling Mean, Std Dev)   (Crest Factor, P2P/RMS)     (dT/dt, ΔT_ambient)
               │                       │                       │
               └───────────────────────┼───────────────────────┘
                                       │
                                       ▼
                          UNSUPERVISED ANOMALY MODEL
                     (Isolation Forest / Mahalanobis / ECDF)
                                       │
                                       ▼
                          PHYSICS-INFORMED HEALTH INDEX
                   (Fused Acoustic Burst Index + Thermal Score)
                                       │
                                       ▼
                       JOINT HEALTH STATUS & DISPATCH
                    [ NORMAL  |  WARNING  |  CRITICAL ]
```

1. **Feature Engineering**:
   - Rolling acoustic energy: $\text{RMS}_{\text{smooth}}$ and $\text{P2P}_{\text{smooth}}$.
   - Acoustic Crest Factor: $\text{P2P} / (2\sqrt{2} \cdot \text{RMS})$. Sharp spikes indicate discharge or mechanical chatter.
   - Acoustic baseline deviation: $|\text{mic\_average\_adc} - 1382.72|$.
   - Thermal rate-of-rise: $dT/dt$.
2. **Anomaly Detection**:
   - Train an unsupervised **Isolation Forest** or **One-Class SVM** on quiescent intervals of the stream.
   - Anomaly score directly maps to anomalous acoustic burst events.
3. **Fused Health Indicator (0–100% Health Index)**:
   - Fuses acoustic anomaly scores with thermal rise rates to output a clear health status: `NORMAL`, `WARNING`, or `CRITICAL`.

### 5.3. Software Architecture Strategy
- **`ml/` Directory**:
  - `features.py`: Signal extraction and rolling statistical metrics.
  - `anomaly_detector.py`: Unsupervised anomaly detection model and scorer.
  - `thresholds.py`: Physics-calibrated degradation thresholds.
- **`backend/` Directory**:
  - Lightweight FastAPI service ingesting sensor telemetry via JSON/REST endpoints.
  - Computes instantaneous health index and serves alert endpoints.
- **`dashboard/` Directory**:
  - Interactive condition monitoring UI (e.g., Streamlit or modern web dashboard) visualizing real-time joint temperature, acoustic burst telemetry, and health alerts.

---

## 6. Preprocessing & Development Roadmap

| Phase | Milestone | Scope & Deliverables | Prerequisites |
| :---: | :--- | :--- | :--- |
| **Phase 1** | **Data Cleaning & Feature Engineering** | Parse `busguard_physical_data.jsonl`, extract rolling statistical and acoustic burst features, save clean outputs to `data/processed/`. | User approval of audit report. |
| **Phase 2** | **ML Baseline Model Development** | Build unsupervised anomaly detection pipeline in `ml/` (Isolation Forest + acoustic thresholding); validate against physical sensor bursts. | Phase 1 completion. |
| **Phase 3** | **Backend Inference API** | Build FastAPI service in `backend/` to ingest telemetry payloads and return real-time condition assessments. | Phase 2 completion. |
| **Phase 4** | **Monitoring Dashboard** | Build real-time condition-monitoring UI in `dashboard/` displaying telemetry charts, joint status, and alerts. | Phase 3 completion. |

---

## 7. Compliance Verification

-  No original datasets were modified, moved, renamed, or deleted.
-  Incompatible datasets were strictly isolated and not combined.
-  No synthetic data was generated.
-  No machine learning models were trained.
-  No unverified accuracy claims were made.
-  No dashboard code or hardware integrations were implemented.

*Audit complete. Standing by for user approval before beginning Phase 1 data preprocessing.*

