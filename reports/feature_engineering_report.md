# BusGuard AI — Phase 4.1: Feature Engineering Report

**Feature Dataset Output**: [`data/processed/busguard_features.csv`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/data/processed/busguard_features.csv)  
**Implementation**: [`ml/data_processing.py`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/ml/data_processing.py)  
**Date**: November 2, 2026  

---

## 1. Domain Background & Engineering Rationale

In high-power bolted electrical busbar joints, mechanical loosening and interfacial contact degradation manifest across three physical mechanisms:

1. **Joule Heating & Thermal Elevation**:
   As clamping bolt preload relaxes, the true microscopic contact area (a-spots) shrinks, causing localized contact resistance ($R_c$) to escalate. Current flow generates heat ($P = I^2 R_c$). Because heat dissipates into ambient air, the true indicator of degradation is **temperature rise above ambient ($\Delta T$)** and **rate of rise ($dT/dt$)**, rather than raw uncorrected temperature.

2. **Acoustic Emissions & Impulse Bursts**:
   Loose contacts subjected to electromagnetic forces ($50/60\text{ Hz}$ vibrational torque) experience mechanical fretting wear and intermittent micro-discharges (micro-arcing). Micro-arcing produces sharp acoustic impulse waveforms characterized by high **Crest Factor** ($\text{Peak} / \text{RMS}$) and elevated acoustic energy bursts.

3. **Electrical Contact Integrity**:
   Direct electrical assessment requires measuring contact voltage drop ($\Delta V$) and load current ($I$). In our physical sensor stream, the current channel recorded `0.0 A`. Feature engineering must handle this state safely without invalid mathematical operations.

---

## 2. Complete Engineered Feature Inventory

The feature-engineered dataset contains **23 columns** computed from the validated physical stream:

| Feature Name | Unit | Mathematical Definition | Physical & Diagnostic Role |
| :--- | :---: | :--- | :--- |
| `t_sec` | $\text{s}$ | $(t - t_0) / 1000.0$ | Normalized temporal baseline from stream inception. |
| `sampling_dt_ms` | $\text{ms}$ | $t_i - t_{i-1}$ | Confirms uniform telemetry rate ($892\text{ ms}$). |
| `ambient_temp_C` | $^\circ\text{C}$ | $\min(\text{temperature\_C}) = 25.75$ | Reference ambient baseline temperature. |
| `temp_rise_C` | $^\circ\text{C}$ | $\max(0, T_{\text{joint}} - T_{\text{ambient}})$ | Localized thermal elevation above ambient. |
| `temp_roc_C_per_sec` | $^\circ\text{C}/\text{s}$ | $(T_i - T_{i-1}) / \Delta t$ | Thermal derivative ($dT/dt$). Flags rapid heating onset. |
| `temp_rolling_mean_5` | $^\circ\text{C}$ | $\frac{1}{5}\sum_{k=0}^4 T_{i-k}$ | Smoothed 5-sample moving average ($\sim 4.46\text{ s}$). |
| `temp_rolling_std_5` | $^\circ\text{C}$ | $\sigma_5(T)$ | Thermal volatility / noise indicator. |
| `temp_smoothed_rise_C` | $^\circ\text{C}$ | $\frac{1}{5}\sum_{k=0}^4 \Delta T_{i-k}$ | Filtered thermal elevation above ambient. |
| `ac_rms_adc` | counts | $\sqrt{\frac{1}{N}\sum x^2}$ | Root-mean-square amplitude of acoustic channel. |
| `ac_peak_to_peak_adc` | counts | $x_{\max} - x_{\min}$ | Peak-to-peak amplitude of acoustic waveform. |
| `acoustic_crest_factor` | ratio | $\frac{\text{ac\_peak\_to\_peak\_adc}}{2\sqrt{2} \cdot \max(\text{ac\_rms\_adc}, 1.0)}$ | Ratio of peak amplitude to effective RMS. Values $> 4.5$ indicate impulse micro-arcing. |
| `acoustic_energy` | $\text{counts}^2$ | $\text{ac\_rms\_adc}^2$ | Instantaneous acoustic vibration energy. |
| `acoustic_p2p_to_rms_ratio` | ratio | $\text{ac\_peak\_to\_peak\_adc} / \text{ac\_rms\_adc}$ | Amplitude dynamic range ratio. |
| `acoustic_rms_smoothed` | counts | $\frac{1}{5}\sum_{k=0}^4 \text{ac\_rms}_{i-k}$ | Rolling smoothed acoustic background energy. |
| `mic_dc_bias_dev` | counts | $|\text{mic\_average\_adc} - 1382.72|$ | Analog pre-amplifier DC stability deviation. |
| `acoustic_burst_flag` | binary | $(\text{P2P} > 1800) \lor (\text{CF} > 4.5)$ | Trigger flag for acoustic discharge burst events. |
| `current_A` | $\text{A}$ | $I_A$ | Measured conductor current ($0.0\text{ A}$). |
| `is_current_active` | binary | $(I_A > 0.01) \rightarrow 0$ | Safety guard: prevents division-by-zero downstream. |
| `contact_resistance_calc_mOhm`| $\text{m}\Omega$ | $\text{NaN}$ | Safely uncomputed when current is zero. |
| `power_loss_W` | $\text{W}$ | $\text{NaN}$ | Safely uncomputed when current is zero. |

---

## 3. Mathematical & Diagnostic Deep Dive

### 3.1. Acoustic Crest Factor Formulation

In vibration and acoustic signal processing, the **Crest Factor (CF)** is defined as:

$$\text{CF} = \frac{|x_{\text{peak}}|}{x_{\text{RMS}}} = \frac{\text{Peak-to-Peak}}{2\sqrt{2} \cdot \text{RMS}}$$

- **Continuous Gaussian Noise**: For pure continuous sinusoidal noise or ambient ventilation hum, the theoretical Crest Factor is $\sim 1.414\text{ to } 2.5$.
- **Impulse Discharges (Micro-Arcing / Fretting)**: When a loose joint sparks or chatters mechanically, narrow high-amplitude voltage/acoustic spikes occur. The peak amplitude surges drastically while the RMS energy changes only marginally, driving $\text{CF} > 4.5\text{–}6.0$.
- In our dataset:
  - Median Crest Factor: **$2.45$** (Normal ambient baseline).
  - Maximum Crest Factor: **$6.44$** (Severe burst spike at $t = 358.9\text{ s}$).

### 3.2. Thermal Elevation & Rate-of-Change ($dT/dt$)

Thermal condition monitoring relies on two complementary metrics:
1. **Steady-State Thermal Elevation ($\Delta T = T_{\text{joint}} - T_{\text{ambient}}$)**:
   Normal busbars operate at $\Delta T < 10^\circ\text{C}$ to $20^\circ\text{C}$ above enclosure ambient. $\Delta T > 30^\circ\text{C}$ indicates warning-level resistance degradation, while $\Delta T > 50^\circ\text{C}$ indicates critical danger.
2. **Dynamic Thermal Rate of Rise ($dT/dt$)**:
   Sudden changes in contact area or high-current surges cause steep positive $dT/dt$. In our recording, $dT/dt$ peaked at $+0.28^\circ\text{C}/\text{s}$ during initial thermal elevation ($t = 40\text{–}60\text{ s}$).

### 3.3. Safe Zero-Current Handling

A common error in condition-monitoring software is computing:
$$R_c = \frac{\Delta V}{I}$$
When $I = 0$, this formula causes either a `ZeroDivisionError` or infinite floating-point values (`inf`). 
Similarly, power loss $P = I^2 R_c$ would evaluate to $0.0\text{ W}$, falsely implying a perfect 100% efficient joint.

**BusGuard AI's Protection Protocol**:
- The pipeline explicitly tests:
  ```python
  if current_A <= 0.01:
      is_current_active = False
      contact_resistance_calc_mOhm = np.nan
      power_loss_W = np.nan
  ```
- This ensures downstream machine learning and dashboard modules recognize that electrical current telemetry is inactive, relying on thermal and acoustic modalities without propagating mathematical corruption.

---

## 4. Cross-Feature Correlation Analysis

From `reports/figures/feature_correlations.png`:
- **`ac_rms_adc` vs `ac_peak_to_peak_adc`**: Moderately positive correlation ($r = 0.54$). Both respond to energetic vibration, but P2P captures impulsive peaks that RMS smooths over.
- **`temperature_C` vs `temp_roc_C_per_sec`**: Weak correlation ($r = 0.12$). The rate of temperature change leads temperature, acting as an early-warning predictor before peak temperature is reached.
- **`mic_dc_bias_dev`**: Shows near-zero correlation ($|r| < 0.08$) with thermal and acoustic parameters, confirming that DC bias drift is negligible and does not contaminate acoustic signal processing.

---

## 5. Generated Visual Artifacts

The following figures were generated and saved in [`reports/figures/`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/reports/figures):
1. **`temperature_trends.png`**: Multi-panel visualization of raw temperature, rolling 5-sample mean, ambient baseline, and rate-of-rise ($dT/dt$).
2. **`acoustic_features.png`**: Dual-axis plot of acoustic RMS, peak-to-peak amplitude, and Crest Factor burst dynamics against the $4.5$ threshold.
3. **`feature_correlations.png`**: Cross-sensor correlation matrix across thermal and acoustic features.
4. **`zero_current_and_sensor_ranges.png`**: Histograms of raw sensor distributions illustrating zero-current containment and acoustic energy distributions.

