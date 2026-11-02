# BusGuard AI — Intelligent Multi-Sensor Busbar Condition Monitoring & Predictive Maintenance System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20Scikit--Learn-orange.svg)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Functional%20MVP-emerald.svg)]()
[![System Type](https://img.shields.io/badge/System-Cyber--Physical%20Condition%20Monitoring-cyan.svg)]()

> **Advisory Condition-Monitoring Prototype**: Designed for low/medium voltage switchgear busbars and industrial power distribution infrastructure. Detects loose or degrading electrical joints using fused thermal, electrical, and acoustic sensor telemetry.

---

## 1. System Objective & Engineering Problem

Bolted busbar electrical connections in industrial switchgear, substations, and MCCs (Motor Control Centers) deteriorate over time due to:
- **Loss of Bolt Preload / Loosening**: Vibration and mechanical stress relax clamping torque.
- **Micro-Fretting & Oxidation**: Interfacial contact asperities (a-spots) corrode under environmental humidity and thermal expansion cycles.
- **Contact Resistance ($R_c$) Escalation**: According to Holm's Electrical Contact Theory, shrinking true contact area drives contact resistance from $< 2\text{ m}\Omega$ to $> 10\text{–}60\text{ m}\Omega$.
- **Joule Overheating Loop**: $P = I^2 R_c$ heating accelerates oxidation, triggering destructive thermal runaway, switchgear fires, and arc-flash hazards.

**BusGuard AI** continuous condition monitoring prevents catastrophic failures by fusing:
1. **Surface Temperature ($T$) & Temperature Rise ($\Delta T$)**: Tracking localized heating above ambient.
2. **Electrical Contact Resistance ($R_c$) & Voltage Drop ($\Delta V$)**: Quantifying electrical joint degradation.
3. **Acoustic Waveforms & Crest Factor**: Detecting high-frequency micro-arcing and mechanical contact chatter.

---

## 2. System Architecture

```text
BusGuard_AI/
│
├── data/
│   ├── raw/                                           # Immutable original sensor streams & benchmarks
│   │   ├── busguard_physical_data.jsonl               # Real physical multi-sensor stream (480 rows)
│   │   ├── connector_degradation_testing.xlsx         # Empirical degradation testing (1416 hrs / 200 cycles)
│   │   └── connector_degradation_sheets/              # 4 normalized laboratory benchmark CSVs
│   │
│   └── processed/                                     # Cleaned, feature-engineered datasets
│       ├── busguard_processed_physical.csv            # Time-domain thermal & acoustic engineered features
│       ├── connector_processed_degradation.csv        # Labeled contact degradation dataset (190 records)
│       └── multijoint_busbar_status.csv               # 8-node switchgear monitored busbar topology
│
├── ml/                                                # Machine Learning & Signal Processing Pipeline
│   ├── data_processing.py                             # Automated data cleaning & feature engineering
│   ├── train_model.py                                 # Anti-leakage Random Forest & Isolation Forest training
│   └── predict.py                                     # Real-time multi-sensor inference engine
│
├── backend/                                           # Core Condition Monitoring Business Logic
│   └── health_monitor.py                              # Physics-calibrated Joint Health Index (0-100%)
│
├── dashboard/                                         # Modern Industrial IoT Monitoring Interface
│   ├── app.py                                         # Multi-page Streamlit application
│   └── style.css                                      # Deep navy/charcoal SCADA dark theme
│
├── models/                                            # Serialized Model Checkpoints & Metadata
│   ├── busbar_rf_classifier.joblib                   # Trained Random Forest Classifier
│   ├── acoustic_anomaly_detector.joblib              # Trained Isolation Forest Detector
│   ├── metrics.json                                  # Quantitative validation metrics & confusion matrix
│   └── anomaly_metrics.json                          # Acoustic anomaly baseline metadata
│
├── reports/                                           # System Audit & Technical Documentation
│   ├── dataset_audit.md                               # Comprehensive raw dataset audit
│   ├── project_audit.md                               # Repository & codebase inspection report
│   └── model_evaluation.md                            # Complete ML model evaluation report
│
├── requirements.txt                                   # Python package dependencies
└── README.md                                          # Executive system documentation
```

---

## 3. Quick Start & Execution Instructions

### Prerequisites
- Python 3.10+ (Tested through Python 3.14)
- Git & PowerShell / Bash

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Data Processing Pipeline
Loads raw multi-sensor streams, engineers acoustic/thermal features, and generates clean datasets in `data/processed/`:
```bash
python ml/data_processing.py
```

### 3. Train Machine Learning Models
Trains the anti-leakage Random Forest Classifier and Isolation Forest Anomaly Detector, evaluates performance, and serializes checkpoints to `models/`:
```bash
python ml/train_model.py
```

### 4. Launch the BusGuard AI Monitoring Dashboard
Starts the interactive industrial dashboard on port 8501:
```bash
streamlit run dashboard/app.py
```
Open your browser at: `http://localhost:8501`

---

## 4. Machine Learning & Engineering Performance

### Supervised Joint Condition Classifier (Random Forest)
- **Algorithm**: `RandomForestClassifier` (100 estimators, regularized depth).
- **Target Classes**:
  - `Tight connection` (Nominal / Healthy, $R_c \le 2.0\text{ m}\Omega$)
  - `Intermediate connection` (Degradation / Loosening Inception, $2.0 < R_c \le 10.0\text{ m}\Omega$)
  - `Loose connection` (Critical Hazard / Severe High Resistance, $R_c > 10.0\text{ m}\Omega$)
- **Anti-Leakage Validation Strategy**: Group-aware split on `specimen_id` (`GroupShuffleSplit`). Specimens in the test partition are completely hidden during training to ensure zero autocorrelation leakage across aging time steps.

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Training Accuracy** | **99.30%** | Evaluated on 26 training specimens (142 samples) |
| **Testing Accuracy** | **100.00%** | Evaluated on 9 unseen test specimens (48 samples) |
| **Macro Precision** | **100.00%** | Unweighted across all 3 connection conditions |
| **Macro Recall** | **100.00%** | Maximum sensitivity to loose / intermediate states |
| **Macro F1-Score** | **100.00%** | Perfect harmonic balance on independent test set |

### Feature Importance Ranking
1. `contact_resistance_mOhm` (45.3%) — Direct physical root-cause indicator.
2. `resistance_ratio` (19.8%) — Multiplication factor over pristine baseline.
3. `resistance_delta_mOhm` (15.2%) — Absolute resistance growth above initial state.
4. `exposure_hours` (8.4%) — Duration under thermal cycling or damp heat stress.
5. `dielectric_pass` (4.9%) — High-voltage insulation integrity test status.

### Unsupervised Acoustic Anomaly Detector (Isolation Forest)
- **Dataset**: `busguard_processed_physical.csv` (480 temporal physical sensor records).
- **Features**: Temperature rise ($\Delta T$), Rate of temperature change ($dT/dt$), Acoustic RMS, Acoustic Peak-to-Peak, Acoustic Crest Factor, and Microphone DC Bias deviation.
- **Function**: Automatically detects transient micro-discharge spikes and mechanical contact chatter in live telemetry streams without requiring historical failure labels.

---

## 5. Transparent Joint Health Index (0–100%)

Rather than presenting an opaque black-box probability, BusGuard AI computes a transparent, physics-calibrated **Joint Health Index**:

$$\text{Health Index} = 100 - (\text{Resistance Penalty} + \text{Thermal Penalty} + \text{Acoustic Penalty})$$

### 1. Resistance Penalty (0 to 50 pts)
- $R_c \le 2.0\text{ m}\Omega$: $0\text{–}5\text{ pts}$ (Nominal contact asperities).
- $2.0 < R_c \le 10.0\text{ m}\Omega$: $10 + \frac{R_c - 2}{8} \times 25\text{ pts}$ (Moderate loosening).
- $R_c > 10.0\text{ m}\Omega$: $35 + \min\left(15, \frac{R_c - 10}{20} \times 15\right)\text{ pts}$ (Severe degradation).

### 2. Thermal Penalty (0 to 30 pts)
- $\Delta T \le 5^\circ\text{C}$: $0\text{ pts}$ (Ambient balance).
- $5^\circ\text{C} < \Delta T \le 25^\circ\text{C}$: $\frac{\Delta T - 5}{20} \times 15\text{ pts}$ (Noticeable heating).
- $\Delta T > 25^\circ\text{C}$: $15 + \min\left(15, \frac{\Delta T - 25}{35} \times 15\right)\text{ pts}$ (Hotspot hazard).

### 3. Acoustic Penalty (0 to 20 pts)
- Peak-to-Peak $> 1600\text{ counts}$ or Crest Factor $> 3.5$: $0\text{–}10\text{ pts}$.
- Crest Factor $> 4.5$: Additional $0\text{–}10\text{ pts}$ (Micro-arcing detection).

### Operational Health Categories
- **NORMAL** ($\ge 80.0\%$): Contact clean and tight. Standard quarterly inspection.
- **ATTENTION REQUIRED** ($50.0\%\text{–}79.9\%$): Moderate degradation. Schedule thermography scan & torque audit.
- **INSPECTION RECOMMENDED / CRITICAL** ($< 50.0\%$): High resistance hazard. Immediate retorquing to prevent thermal runaway.

---

## 6. Documented Hardware Integration Interface (ESP32)

BusGuard AI is decoupled from edge sensor hardware. An external hardware integration team can connect ESP32 microcontrollers or IoT gateways by streaming standard JSON payloads over REST or MQTT:

```json
{
  "node_id": "ESP32_BUSGUARD_NODE_01",
  "joint_id": "J01",
  "timestamp_ms": 1793575050866,
  "telemetry": {
    "current_A": 420.5,
    "temperature_C": 28.5,
    "voltage_drop_mV": 0.55,
    "ac_rms_adc": 136.37,
    "ac_peak_to_peak_adc": 921,
    "mic_average_adc": 1384.5
  },
  "diagnostics": {
    "battery_V": 3.28,
    "wifi_rssi_dBm": -62
  }
}
```

- **Protocol**: HTTP POST to `/api/v1/telemetry` or MQTT topic `busguard/telemetry/{joint_id}`.
- **Baud Rate (Serial CDC)**: 115200 bps.
- **ADC Resolution**: 12-bit ($0\text{–}4095$ counts) referenced to $3.3\text{ V}$.

---

## 7. Deliverables & Documentation

- [**`reports/dataset_audit.md`**](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/reports/dataset_audit.md): Deep-dive audit of all datasets in `data/raw/`.
- [**`reports/project_audit.md`**](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/reports/project_audit.md): Codebase audit, defect analysis of external notebooks, and architectural recommendations.
- [**`reports/model_evaluation.md`**](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/reports/model_evaluation.md): Full validation metrics, confusion matrix, and feature importance rankings.
- [**`reports/final_testing_report.md`**](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/reports/final_testing_report.md): Automated 12-point system verification and end-to-end testing report.
