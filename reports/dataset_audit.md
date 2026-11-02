# BusGuard AI — Comprehensive Dataset Audit Report

**Audit Date**: November 2, 2026  
**System Target**: BusGuard AI (AI-Assisted Multi-Sensor Busbar Condition-Monitoring System)  
**Location Audited**: `data/raw/`  

---

## 1. Executive Summary

BusGuard AI is designed to monitor electrical busbar joints and detect loosening, contact degradation, and micro-arcing by fusing three distinct physical sensor streams: **temperature**, **electrical measurements**, and **acoustic emissions**.

An exhaustive technical audit of all data files and archives located in [`data/raw/`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/data/raw) was performed. The primary objective is to verify schema consistency, sensor validity, data quality, and domain alignment before initiating preprocessing or model engineering.

### Summary Matrix

| Dataset / File | File Format | Records & Dimensions | Primary Features / Signals | Domain Categorization | Suitability for BusGuard AI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`busguard_physical_data.jsonl`** | JSON Lines (JSONL) | 480 rows × 6 columns | Timestamp, Current, Temperature, AC RMS, AC Peak-to-Peak, Mic DC | **Directly Relevant Busbar Data** | **Primary (High Priority)**: Directly captures the required sensor modalities, though current is 0.0 A and ground truth labels are absent. |
| **`connector_degradation_testing.xlsx`** (and CSV sheets) | Excel (.xlsx) / CSV | 4 sheets (11–28 rows × 10–23 cols) | Contact resistance ($m\Omega$), Field age, Thermal cycling, Damp heat | **Supplementary Degradation Data** | **High (Benchmark Calibration)**: Provides empirical joint contact resistance degradation curves under environmental and thermal stress. |
| **`Fault Simulation Dataset for 110 kV...`** | CSV, PFD, PY, PDF, ZIP | 618 rows × 27 columns | 3-phase line currents (kA), bus voltages (kV), fault types | **Transmission Line Simulation** | **Incompatible / Unrelated**: 110 kV transmission grid short-circuit/ground fault simulation. Excluded from busbar modeling. |
| **`plc_industrial_control`** | CSV, JSON, ZIP | 100,000 rows × 7 columns | Hydraulic pressure (bar), PLC I/O (0–10 V), valve status | **PLC Industrial Control** | **Incompatible / Unrelated**: Hydraulic pressing and machine tool automation. Excluded from busbar modeling. |
| **`fault-prediction-using...ipynb`** | Jupyter Notebook | 122 cells (5.47 MB) | Kaggle transmission line fault classification code | **ML Notebooks & Project Code** | **Code Reference Only**: Methodology template for decision tree workflows; contains no usable busbar dataset. |

---

## 2. Complete File Inventory

```text
data/raw/
│
├── busguard_physical_data.jsonl                                         [JSONL, 61,260 bytes]
├── connector_degradation_testing.xlsx                                   [XLSX, 42,852 bytes]
├── Fault_Simulation_Dataset_for_110_kV_Power_Transmission_Lines.zip     [ZIP, 121,479 bytes]
├── plc_industrial_control_archive.zip                                   [ZIP, 3,048,055 bytes]
├── fault-prediction-using-decision-tree-algorithm.ipynb                 [IPYNB, 5,474,227 bytes]
│
├── connector_degradation_sheets/                                        [Directory, Extracted CSVs]
│   ├── General_Testing_CR.csv                                           [CSV, 2,570 bytes]
│   ├── General_Testing_Misc.csv                                         [CSV, 1,640 bytes]
│   ├── Thermal_Cycling.csv                                              [CSV, 3,602 bytes]
│   └── Damp_Heat.csv                                                    [CSV, 3,382 bytes]
│
├── Fault Simulation Dataset for 110 kV Power Transmission Lines/        [Directory, Extracted Package]
│   ├── fault_dataset.csv                                                [CSV, 101,368 bytes]
│   ├── 110kV_Grid.pfd                                                   [PFD (PowerFactory Model), 79,152 bytes]
│   ├── diagram.pdf                                                      [PDF, 26,933 bytes]
│   ├── shc_calc.py                                                      [Python Script, 3,368 bytes]
│   └── ldf_calc.py                                                      [Python Script, 2,832 bytes]
│
└── plc_industrial_control/                                              [Directory, Extracted Package]
    ├── plc_industrial_control_dataset (1).csv                           [CSV, 6,352,115 bytes]
    └── plc_industrial_control_dataset (1).json                          [JSON, 26,052,814 bytes]
```

---

## 3. In-Depth Dataset Technical Inspections

### 3.1. Primary Stream: `busguard_physical_data.jsonl`

- **Location**: `data/raw/busguard_physical_data.jsonl`
- **Format**: JSON Lines (UTF-8, one valid JSON object per row).
- **Row & Column Count**: 480 rows, 6 columns.
- **Sampling Duration**: $427,268\text{ ms} \approx 427.3\text{ seconds}$ ($7.12\text{ minutes}$).
- **Sampling Uniformity**: Uniform $892\text{ ms}$ interval between consecutive rows ($f_s \approx 1.121\text{ Hz}$).
- **Missing Values**: $0$ across all 6 columns ($0.0\%$).
- **Duplicate Rows**: $0$ duplicate rows.
- **Available Labels/Classes**: None (unlabeled stream).

#### Detailed Column Audit

| Column | Data Type | Min | Max | Mean | Median | Std Dev | Identifiable Unit | Interpretation & Physical Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `t` | Integer | 3,011 | 430,279 | 216,645.0 | 216,645.0 | 123,450.4 | Milliseconds ($\text{ms}$) | Internal system timestamp. |
| `I_A` | Float | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 | Amperes ($\text{A}$) | Conductor load current. Strictly zero throughout. |
| `temperature_C` | Float | 25.75 | 31.75 | 27.65 | 27.00 | 1.67 | Celsius ($^\circ\text{C}$) | Joint surface temperature. |
| `ac_rms_adc` | Float | 31.44 | 560.56 | 162.47 | 132.65 | 71.30 | ADC Counts (raw) | RMS amplitude of acoustic/vibration signal. |
| `ac_peak_to_peak_adc` | Integer | 274 | 2,897 | 1,163.80 | 945.00 | 462.26 | ADC Counts (raw) | Peak-to-peak amplitude of acoustic waveform. |
| `mic_average_adc` | Float | 1,375.60 | 1,390.50 | 1,382.72 | 1382.60 | 2.27 | ADC Counts (raw) | DC baseline offset of acoustic microphone. |

#### Sensor Validity Analysis
- **Current (`I_A`) — Critical Finding**: The sensor channel registered strictly `0.0 A` for every sample. No active current flow was recorded during this capture. Consequently, direct calculation of Joule dissipation ($P = I^2 R_c$) or contact resistance ($R_c = \Delta V / I$) cannot be computed from this file alone.
- **Temperature (`temperature_C`)**: Ranges between $25.75^\circ\text{C}$ and $31.75^\circ\text{C}$. This is fully valid for ambient laboratory or low-stress testing, but does not yet exhibit the thermal runaway ($> 70\text{–}100^\circ\text{C}$) typical of severe high-current joint failure.
- **Acoustic Channels (`ac_rms_adc`, `ac_peak_to_peak_adc`, `mic_average_adc`)**: The microphone DC offset is highly stable ($1382.7 \pm 2.27$ counts), consistent with a hardware preamplifier biased at ~1.11 V (on a 3.3 V 12-bit ADC). The RMS and peak-to-peak channels show dynamic bursts (up to 2897 counts), confirming valid capture of transient acoustic and vibration phenomena.

---

### 3.2. Degradation Benchmark: `connector_degradation_testing.xlsx`

- **Location**: `data/raw/connector_degradation_testing.xlsx` (also exported to `data/raw/connector_degradation_sheets/`).
- **Format**: Microsoft Excel Spreadsheet (.xlsx), 4 worksheets.
- **Domain Relevance**: Directly captures the physical failure mode of electrical contact joints under accelerated stress testing (thermal cycling and damp heat).

#### Sheet 1: `General_Testing_CR.csv`
- **Dimensions**: 27 data rows × 10 columns. Missing values: 0. Duplicates: 0.
- **Columns**:
  - `Connector` (String): Specimen identifier (e.g., CC1, CS2, J8_B5).
  - `Cable Length with Connector mm` (Numeric, $0\text{–}970\text{ mm}$).
  - `Mfr origin` (Categorical: A, B, C, D, H, I, unknown).
  - `Mfr` (Categorical: Staubli MC4, TV Nord, Elmex, Renhesolar, TUV NORD, SUNTER, Xinhui, etc.).
  - `Climate` (Categorical: Neutral, Rainy, Cold, Warm, Hot).
  - `Location` (Categorical: Laboratory A, Leh, Deogarh, Pune, Dehradoon, Tirunelveli, Delhi, Cochin, Jaipur, Imphal, Jaisalmer).
  - `Material` (Categorical: PC, PPO, UNKNOWN).
  - `Field Age of connector years` (Numeric: $0\text{–}10\text{ years}$).
  - `Contact Resistance of cable Miliohm` (Numeric: $0.0\text{–}4.074\text{ m}\Omega$).
  - `Contact Resistance Connector Miliohm` (Numeric: $0.43\text{–}20.226\text{ m}\Omega$).
- **Identifiable Units**: Cable length in $\text{mm}$, field age in $\text{years}$, resistance in $\text{m}\Omega$ ($\text{milli-ohms}$).

#### Sheet 2: `General_Testing_Misc.csv`
- **Dimensions**: 27 data rows × 11 columns. Missing values: 0. Duplicates: 0.
- **Columns**: Specimen metadata + 4 binary electrical/safety test columns (`1` = Pass, `0` = Fail):
  - `Dielectric Voltage Withstand test`: Min 0, Max 1.
  - `Insulation Resistance test`: Min 0, Max 1.
  - `IP test` (Ingress Protection): Min 0, Max 1.
  - `Glow wire` (Flammability resistance): Min 0, Max 1.
- **Identifiable Units**: Binary test indicators ($0 / 1$).

#### Sheet 3: `Thermal_Cycling.csv`
- **Dimensions**: 28 data rows × 16 columns. Missing values: 0. Duplicates: 0.
- **Columns**: Specimen metadata + Contact resistance progression across thermal stress cycles:
  - `Contact Resistance of connector at 0 Hrs Miliohm`: Min $0.0\text{ m}\Omega$, Max $14.05\text{ m}\Omega$.
  - `Contact Resistance of connector After 100 cycles at 500 hrs Miliohm`: Min $0.0\text{ m}\Omega$, Max $24.54\text{ m}\Omega$.
  - `Contact Resistance of connector After 200 cycles at 1000 hrs Miliohm`: Min $0.0\text{ m}\Omega$, Max $36.35\text{ m}\Omega$.
  - `Contact Resistance of connector After Dielectric V and IP test Miliohm`: Min $0.0\text{ m}\Omega$, Max $41.55\text{ m}\Omega$.
  - Binary test indicators (`Dielectric Voltage Withstand test`, `IP test`, `Glow wire`).
- **Physical Trajectory**: Quantifies joint contact degradation driven by thermal expansion/contraction cycles (mean resistance climbs from $3.89\text{ m}\Omega$ at 0 hrs to $11.51\text{ m}\Omega$ post-test).

#### Sheet 4: `Damp_Heat.csv`
- **Dimensions**: 11 data rows × 23 columns. Missing values: 0. Duplicates: 0.
- **Columns**: Dense time-series resistance progression during continuous damp heat ($85^\circ\text{C} / 85\%\text{ RH}$) aging at:
  $0\text{ hrs}$, $168\text{ hrs}$, $360\text{ hrs}$, $600\text{ hrs}$, $888\text{ hrs}$, $1056\text{ hrs}$, $1224\text{ hrs}$, and $1416\text{ hrs}$.
  - Initial contact resistance ($0\text{ hrs}$): $0.56\text{–}24.55\text{ m}\Omega$.
  - Final contact resistance ($1416\text{ hrs}$): $3.15\text{–}60.20\text{ m}\Omega$ (surpassing $61.0\text{ m}\Omega$ post-dielectric test).

---

### 3.3. Unrelated Grid Dataset: `Fault Simulation Dataset for 110 kV Power Transmission Lines`

- **Location**: `data/raw/Fault Simulation Dataset for 110 kV Power Transmission Lines/`
- **Files in Package**:
  - `fault_dataset.csv` (Tabular simulation results, 101.4 KB).
  - `110kV_Grid.pfd` (PowerFactory grid model database, 79.2 KB).
  - `diagram.pdf` (Single-line diagram of the 110 kV transmission grid, 26.9 KB).
  - `shc_calc.py` (DIgSILENT PowerFactory Python script for short-circuit sweeps, 3.4 KB).
  - `ldf_calc.py` (DIgSILENT PowerFactory Python script for load flow sweeps, 2.8 KB).
- **Tabular Dimensions (`fault_dataset.csv`)**: 618 data rows × 27 columns.
- **Columns**:
  - 3-Phase Bus Voltages: `Ua1`, `Ub1`, `Uc1`, `Ua2`, `Ub2`, `Uc2`, `Ua3`, `Ub3`, `Uc3` (Numeric, $0.0\text{–}89.98\text{ kV}$).
  - 3-Phase Line Currents: `Ia1` through `Ic5` across 5 transmission lines (Numeric, $0.003\text{–}11.54\text{ kA}$).
  - Targets: `fault` (`no fault`, `spgf`, `2psc`, `3psc`), `line` (`Line1`–`Line5`), `position` ($0\text{–}100\%$).
- **Missing Values**:
  - `line`: 303 missing (49.03%) — empty when `fault == 'no fault'`.
  - `position`: 303 missing (49.03%) — empty when `fault == 'no fault'`.
- **Duplicates**: **73 duplicate rows** ($11.81\%$ of dataset).
- **Identifiable Units**: Voltage in $\text{kV}$, Current in $\text{kA}$, Distance in $\%$.
- **Suitability**: **Incompatible**. This dataset simulates macroscopic high-voltage network short-circuits. It lacks busbar joint contact resistance, local thermal dissipation, and acoustic emissions.

---

### 3.4. Unrelated Industrial Dataset: `plc_industrial_control`

- **Location**: `data/raw/plc_industrial_control/`
- **Files in Package**:
  - `plc_industrial_control_dataset (1).csv` (6.35 MB).
  - `plc_industrial_control_dataset (1).json` (26.05 MB).
- **Dimensions**: 100,000 data rows × 7 columns.
- **Columns & Metadata**:
  - `plc_input_signal` (Float, $0.0\text{–}8.94\text{ V}$, Analog sensor input 0–10 V).
  - `plc_output_signal` (Float, $0.0\text{–}6.99\text{ V}$, Actuator control signal 0–10 V).
  - `hydraulic_pressure` (Float, $2.0\text{–}183.5\text{ bar}$, Hydraulic system pressure).
  - `valve_status` (Categorical: `CLOSED`, `OPEN`, `PARTIAL`).
  - `emergency_stop` (Categorical: `False`, `True`).
  - `process_time` (Datetime: ISO 8601 with `+07:00` offset).
  - `production_count` (Float/Int: $0\text{–}23,094\text{ units}$, cumulative manufacturing piece count).
- **Missing Values & Duplicates**: 0 missing, 0 duplicates.
- **Suitability**: **Incompatible**. Tracks manufacturing automation (hydraulic pressing and part counters). Completely outside the physical scope of electrical joint condition monitoring.

---

### 3.5. Supplementary Reference: `fault-prediction-using-decision-tree-algorithm.ipynb`

- **Location**: `data/raw/fault-prediction-using-decision-tree-algorithm.ipynb` (also mirrored in `notebooks/`).
- **Format**: Jupyter Notebook (122 cells, 5.47 MB).
- **Content**: Kaggle notebook executing machine learning classification on high-voltage transmission line faults using Kaggle's external `classData.csv` (features: `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`; targets: `G`, `C`, `B`, `A`).
- **Suitability**: **Code Reference Only**. It contains no busbar data, and its source CSV is not in the repository.

---

## 4. Dataset Categorization & Relevance Matrix

```
                      ┌──────────────────────────────────────────────┐
                      │              Raw Data Assets                 │
                      └──────────────────────┬───────────────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             │                               │                               │
  ▼ Directly Relevant             ▼ Supplementary Benchmark        ▼ Incompatible / Excluded
  • busguard_physical_data.jsonl  • connector_degradation_         • fault_dataset.csv (110 kV)
    (Acoustic, Temp, Current)       testing.xlsx (Thermal          • plc_industrial_control (SCADA)
                                    Cycling & Damp Heat)           • ldf_calc.py, shc_calc.py
```

### Categorization Details

1. **Directly Relevant Busbar Joint-Condition Data**:
   - `busguard_physical_data.jsonl`: Directly measures the target multi-sensor modalities (temperature, acoustic RMS, peak-to-peak amplitude, and microphone DC level).
   - *Limitation*: The current channel is strictly zero (`I_A == 0.0`), and health labels are missing.
2. **Supplementary Connector Degradation Benchmark Data**:
   - `connector_degradation_testing.xlsx`: Empirically establishes how joint contact resistance degrades over thermal cycles and environmental aging.
   - *Limitation*: Measurements are taken on electrical connector joints rather than heavy rectangular copper/aluminum busbar bars; however, the contact resistance physics ($R_c$) is identical.
3. **Transmission-Line Fault Simulation Data**:
   - `fault_dataset.csv` + PowerFactory project files.
   - *Limitation*: Models 110 kV grid short circuits, not loose electrical joints or contact overheating.
4. **PLC Industrial Control Data**:
   - `plc_industrial_control_dataset (1).csv` and `.json`.
   - *Limitation*: Focuses on hydraulic valves and production counts; irrelevant to busbar condition monitoring.
5. **Existing ML Notebooks and Project Code**:
   - `fault-prediction-using-decision-tree-algorithm.ipynb`.
   - *Limitation*: References an external transmission line dataset; cannot be run directly.

---

## 5. Sensor Reading Validity Assessment

| Sensor Parameter | Expected Electrical / Physical Behavior | Observed in `busguard_physical_data.jsonl` | Validity Assessment |
| :--- | :--- | :--- | :--- |
| **Current ($I$)** | Active operating current ($10\text{–}1000\text{+ A}$) driving $I^2 R_c$ heating. | Strictly `0.0 A` across all 480 rows | ❌ **Invalid / Inactive**: Indicates sensor disconnect, zero-load idling, or unscaled ADC input. |
| **Temperature ($T$)** | Ambient ($20\text{–}30^\circ\text{C}$) to loose-joint hotspot ($60\text{–}120^\circ\text{C}$). | $25.75^\circ\text{C} \text{ to } 31.75^\circ\text{C}$ | ⚠️ **Valid Ambient**: Plausible for ambient baseline, but lacks high-temperature fault states. |
| **Acoustic RMS** | Baseline noise with sharp spikes during micro-arcing/vibration. | $31.44 \text{ to } 560.56\text{ counts}$ |  **Valid**: Active dynamic range; reveals distinct acoustic burst intervals. |
| **Acoustic P2P** | High-amplitude transient peaks during electrical discharge. | $274 \text{ to } 2,897\text{ counts}$ |  **Valid**: Excellent dynamic range on a 12-bit ADC ($0\text{–}4095$) without clipping. |
| **Microphone DC** | Stable analog reference midpoint ($V_{ref}/2 \approx 1.1\text{–}1.65\text{ V}$). | $1375.6 \text{ to } 1390.5\text{ counts}$ |  **Valid**: Stays tightly centered around $1382.7\text{ counts}$ ($\sim 1.11\text{ V}$). |
| **Contact Resistance ($R_c$)** | Healthy: $< 2\text{ m}\Omega$; Degraded: $10\text{–}60\text{ m}\Omega$. | $0.43 \text{ to } 61.0\text{ m}\Omega$ *(connector benchmark)* |  **Valid & Calibrated**: Reflects true physical contact degradation. |

---

## 6. Data Quality, Anomalies & Incompatibilities

1. **Zero-Current Anomaly**: Without current in `busguard_physical_data.jsonl`, thermal modeling cannot calculate electrical power dissipation ($P = I^2 R$). Current calibration or active load data must be confirmed before thermal resistance estimation is possible.
2. **Unlabeled Physical Data**: `busguard_physical_data.jsonl` contains no ground-truth annotations (`healthy`, `loose`, `degraded`). Consequently, supervised classification cannot be applied directly to this file. Unsupervised anomaly detection or semi-supervised clustering must be utilized.
3. **Data Duplication**: `fault_dataset.csv` contains 73 duplicate rows ($11.81\%$).
4. **Severe Modality Mismatch**: Attempting to train BusGuard AI on `fault_dataset.csv` (110 kV grid lines) or `plc_industrial_control` (hydraulic machines) would introduce severe domain mismatch and negative transfer. Both must be excluded from the pipeline.
