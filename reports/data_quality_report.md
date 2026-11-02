# BusGuard AI — Phase 4.1: Data Quality & Profiling Report

**Audit Target**: Physical Multi-Sensor Busbar Dataset (`data/raw/busguard_physical_data.jsonl`)  
**Processing Pipeline**: [`ml/data_processing.py`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/ml/data_processing.py)  
**Cleaned Output**: [`data/processed/busguard_cleaned.csv`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/data/processed/busguard_cleaned.csv)  
**Date**: November 2, 2026  

---

## 1. Executive Summary

This report documents the rigorous data validation, quality assessment, and cleaning performed on the primary physical multi-sensor recording for **BusGuard AI**. 

The raw recording represents an authentic physical sensor stream capturing:
1. **Surface Temperature** (`temperature_C`)
2. **AC Acoustic RMS Amplitude** (`ac_rms_adc`)
3. **AC Acoustic Peak-to-Peak Amplitude** (`ac_peak_to_peak_adc`)
4. **Microphone DC Baseline Bias** (`mic_average_adc`)
5. **Electrical Conductor Current** (`I_A`)
6. **Hardware System Timestamp** (`t`)

---

## 2. Ingestion & Completeness Profiling

| Profiling Parameter | Metric Value | Evaluation & Verification |
| :--- | :--- | :--- |
| **Raw File Path** | `data/raw/busguard_physical_data.jsonl` | File integrity preserved without alteration. |
| **File Format** | JSON Lines (newline-delimited JSON) | All 480 lines parsed successfully with zero syntax errors. |
| **Total Ingested Records** | **480 rows** | Complete contiguous recording. |
| **Total Features** | **6 attributes** | `t`, `I_A`, `temperature_C`, `ac_rms_adc`, `ac_peak_to_peak_adc`, `mic_average_adc`. |
| **Missing Values (NaN / Null)** | **0 (0.00%)** | 100% complete across all columns. |
| **Duplicate Records** | **0 (0.00%)** | All 480 time points are unique. |
| **Duration of Stream** | **427.27 seconds (~7.12 minutes)** | $t_{\text{start}} = 3,011\text{ ms}$, $t_{\text{end}} = 430,279\text{ ms}$. |
| **Sampling Interval ($\Delta t$)** | **892.0 milliseconds (constant)** | Exactly $892\text{ ms}$ interval across all samples ($f_s \approx 1.121\text{ Hz}$). |

---

## 3. Attribute-Level Statistical Distribution

```text
+----------------------+---------+------------+----------+----------+----------+----------+-----------------------+
| Attribute Name       | Dtype   | Min        | Median   | Mean     | Max      | Std Dev  | Identifiable Unit     |
+----------------------+---------+------------+----------+----------+----------+----------+-----------------------+
| t                    | int64   | 3,011      | 216,645  | 216,645  | 430,279  | 123,450  | Milliseconds (ms)     |
| I_A                  | float64 | 0.00       | 0.00     | 0.00     | 0.00     | 0.00     | Amperes (A)           |
| temperature_C        | float64 | 25.75      | 27.00    | 27.65    | 31.75    | 1.67     | Degrees Celsius (°C)  |
| ac_rms_adc           | float64 | 31.44      | 132.65   | 162.47   | 560.56   | 71.30    | Raw ADC Counts        |
| ac_peak_to_peak_adc  | int64   | 274        | 945      | 1,163.80 | 2,897    | 462.26   | Raw ADC Counts        |
| mic_average_adc      | float64 | 1,375.60   | 1,382.60 | 1,382.72 | 1,390.50 | 2.27     | Raw ADC Counts        |
+----------------------+---------+------------+----------+----------+----------+----------+-----------------------+
```

---

## 4. In-Depth Sensor Validity & Physical Diagnostics

### 4.1. Thermal Sensor Channel (`temperature_C`)
- **Observed Range**: $25.75^\circ\text{C}$ to $31.75^\circ\text{C}$ (Span: $6.0^\circ\text{C}$, Standard Deviation: $1.67^\circ\text{C}$).
- **Validity Assessment**: **Valid Ambient Baseline**.
- **Physical Interpretation**: The temperature starts at $26.9^\circ\text{C}$, rises steadily to peak at $31.75^\circ\text{C}$ at $t = 98.45\text{ s}$, and gently relaxes back to $25.75^\circ\text{C}$. This profile represents a real thermal transient (e.g., enclosure heat dissipation or ambient thermal drift), without clipping or erratic noise.

### 4.2. Acoustic Channels (`ac_rms_adc`, `ac_peak_to_peak_adc`, `mic_average_adc`)
- **Microphone DC Baseline (`mic_average_adc`)**: Stays remarkably centered at $1,382.72 \pm 2.27\text{ counts}$.
  - For a 12-bit ADC ($0\text{–}4095$ counts) referenced to $3.3\text{ V}$:
    $$V_{\text{DC}} = \frac{1,382.72}{4,095} \times 3.3\text{ V} \approx 1.114\text{ V}$$
  - This perfectly matches the hardware analog pre-amplifier virtual ground bias for an electret / MEMS acoustic transducer.
- **Acoustic RMS & Peak-to-Peak (`ac_rms_adc`, `ac_peak_to_peak_adc`)**:
  - `ac_rms_adc` has a baseline of $\sim 130\text{ counts}$ with dynamic surges up to $560.56\text{ counts}$.
  - `ac_peak_to_peak_adc` exhibits prominent bursts reaching $2,897\text{ counts}$.
  - Confirms active dynamic acoustic sensitivity without sensor saturation or rail-to-rail clipping ($< 4095$).

### 4.3. Conductor Current Channel (`I_A`) — Critical Finding
- **Observed Reading**: Strictly `0.0 A` across all 480 timestamps.
- **Root-Cause Analysis**:
  1. The hardware recording was conducted in a de-energized/idle busbar state, or
  2. The analog current transformer / Hall-effect sensor channel was unpowered or uncalibrated during this specific test run.
- **Data Safety Protocol Enforced**:
  - Contact resistance ($R = \Delta V / I$) and electrical power loss ($P = I^2 R$) **must not be calculated** on zero-current records.
  - Doing so would trigger divide-by-zero errors or generate fabricated zero-resistance artifacts.
  - The pipeline sets `is_current_active = False` and outputs `contact_resistance_calc_mOhm = NaN` and `power_loss_W = NaN`.

---

## 5. Dataset Cleaning & Output Deliverables

The cleaning function `validate_and_clean_data()` produces:
- **`data/processed/busguard_cleaned.csv`**:
  - Verified schema, explicit numerical types.
  - Normalised elapsed time: $t_{\text{sec}} = (t - t_0) / 1000.0$.
  - Explicit sampling interval $\Delta t = 892\text{ ms}$.
  - Zero missing values, zero duplicates.

