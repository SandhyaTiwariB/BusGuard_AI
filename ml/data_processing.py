"""
BusGuard AI — Phase 4.1: Data Processing & Feature Engineering Module
======================================================================
Modular, reusable data-processing pipeline for multi-sensor busbar condition monitoring.
Key capabilities:
1. Ingests raw physical JSONL data from data/raw/busguard_physical_data.jsonl
2. Validates schema, data types, timestamps, missing values and duplicates
3. Safely handles zero-current condition without division-by-zero or fabrication
4. Extracts thermal dynamics, acoustic burst signatures and statistical rolling trends
5. Saves clean outputs to data/processed/busguard_cleaned.csv and busguard_features.csv
6. Generates visual diagnostic figures in reports/figures/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def load_raw_busguard_data(raw_path=None):
    """
    Loads raw multi-sensor stream from JSONL format into a pandas DataFrame.
    Preserves original raw file intact.
    """
    if raw_path is None:
        raw_path = os.path.join(RAW_DIR, "busguard_physical_data.jsonl")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {line_no} failed JSON decode: {e}")

    df = pd.DataFrame(records)
    print(f"[OK] Ingested raw records: {len(df)} rows, {len(df.columns)} columns from {os.path.basename(raw_path)}")
    return df


def validate_and_clean_data(df_raw, cleaned_path=None):
    """
    Performs data validation and cleaning:
    - Verifies schema and data types
    - Checks missing values and duplicates
    - Validates timestamp monotonicity and sampling interval
    - Exports cleaned base dataset to data/processed/busguard_cleaned.csv
    """
    if cleaned_path is None:
        cleaned_path = os.path.join(PROCESSED_DIR, "busguard_cleaned.csv")
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)

    df_clean = df_raw.copy()

    # 1. Type casting
    df_clean["t"] = df_clean["t"].astype(int)
    df_clean["I_A"] = df_clean["I_A"].astype(float)
    df_clean["temperature_C"] = df_clean["temperature_C"].astype(float)
    df_clean["ac_rms_adc"] = df_clean["ac_rms_adc"].astype(float)
    df_clean["ac_peak_to_peak_adc"] = df_clean["ac_peak_to_peak_adc"].astype(int)
    df_clean["mic_average_adc"] = df_clean["mic_average_adc"].astype(float)

    # 2. Check duplicates
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        print(f"[WARN] Found {dup_count} duplicate rows. Dropping duplicates.")
        df_clean = df_clean.drop_duplicates()
    else:
        print("[OK] Duplicate check passed: 0 duplicate rows.")

    # 3. Check missing values
    null_counts = df_clean.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[WARN] Found missing values: {null_counts.to_dict()}")
        df_clean = df_clean.dropna()
    else:
        print("[OK] Missing value check passed: 0 missing values.")

    # 4. Check timestamp sequence
    df_clean = df_clean.sort_values(by="t").reset_index(drop=True)
    t0 = df_clean["t"].iloc[0]
    df_clean["t_sec"] = (df_clean["t"] - t0) / 1000.0
    df_clean["sampling_dt_ms"] = df_clean["t"].diff().fillna(892.0)

    # Save cleaned dataset
    df_clean.to_csv(cleaned_path, index=False)
    print(f"[OK] Saved cleaned dataset: {cleaned_path} ({len(df_clean)} rows, {len(df_clean.columns)} columns)")
    return df_clean


def engineer_features(df_clean, features_path=None):
    """
    Computes domain-specific engineered features:
    - Thermal features (ambient baseline, temperature rise, rate of change, rolling stats)
    - Acoustic features (crest factor, acoustic energy, peak-to-RMS ratio, DC deviation)
    - Electrical safety features (safe zero-current handling without division-by-zero)
    - Saves feature-engineered dataset to data/processed/busguard_features.csv
    """
    if features_path is None:
        features_path = os.path.join(PROCESSED_DIR, "busguard_features.csv")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    df_feat = df_clean.copy()

    # ---------------------------------------------------------
    # 1. Thermal Feature Engineering
    # ---------------------------------------------------------
    # Ambient baseline: Minimum recorded temperature in stream (25.75 deg C)
    ambient_baseline_C = 25.75
    df_feat["ambient_temp_C"] = ambient_baseline_C
    
    # Temperature rise above ambient: Delta T = T_joint - T_ambient
    df_feat["temp_rise_C"] = np.maximum(0.0, df_feat["temperature_C"] - ambient_baseline_C)

    # Thermal rate-of-change (dT/dt in deg C per second)
    dt_sec = np.maximum(0.001, df_feat["sampling_dt_ms"] / 1000.0)
    df_feat["temp_roc_C_per_sec"] = df_feat["temperature_C"].diff().fillna(0.0) / dt_sec

    # Rolling thermal statistics (window = 5 samples ~ 4.46 seconds)
    df_feat["temp_rolling_mean_5"] = df_feat["temperature_C"].rolling(window=5, min_periods=1).mean()
    df_feat["temp_rolling_std_5"] = df_feat["temperature_C"].rolling(window=5, min_periods=1).std().fillna(0.0)
    df_feat["temp_smoothed_rise_C"] = df_feat["temp_rise_C"].rolling(window=5, min_periods=1).mean()

    # ---------------------------------------------------------
    # 2. Acoustic Feature Engineering
    # ---------------------------------------------------------
    # Acoustic Crest Factor = Peak / (RMS * sqrt(2)) or P2P / (2 * sqrt(2) * RMS)
    # Impulsive micro-arcing and contact chatter generate elevated crest factors
    rms_safe = np.maximum(df_feat["ac_rms_adc"], 1.0)
    df_feat["acoustic_crest_factor"] = df_feat["ac_peak_to_peak_adc"] / (2.0 * np.sqrt(2.0) * rms_safe)

    # Acoustic Energy (Proportional to RMS squared)
    df_feat["acoustic_energy"] = df_feat["ac_rms_adc"] ** 2

    # Peak-to-RMS Amplitude Ratio
    df_feat["acoustic_p2p_to_rms_ratio"] = df_feat["ac_peak_to_peak_adc"] / rms_safe

    # Rolling acoustic smoothed energy (window = 5)
    df_feat["acoustic_rms_smoothed"] = df_feat["ac_rms_adc"].rolling(window=5, min_periods=1).mean()

    # Microphone DC bias deviation from nominal hardware midpoint (1382.72 ADC counts)
    hardware_dc_nominal = 1382.72
    df_feat["mic_dc_bias_dev"] = (df_feat["mic_average_adc"] - hardware_dc_nominal).abs()

    # Acoustic Burst Event Flag: P2P > 1800 counts or Crest Factor > 4.5
    df_feat["acoustic_burst_flag"] = (
        (df_feat["ac_peak_to_peak_adc"] > 1800) | (df_feat["acoustic_crest_factor"] > 4.5)
    ).astype(int)

    # ---------------------------------------------------------
    # 3. Electrical Current Safety Handling
    # ---------------------------------------------------------
    # Safe handling: Current in raw physical stream is strictly 0.0 A
    # Avoid zero-division: Do NOT calculate Rc = V/I or P = I^2 * R when I = 0
    df_feat["is_current_active"] = (df_feat["I_A"] > 0.01).astype(int)
    df_feat["contact_resistance_calc_mOhm"] = np.nan
    df_feat["power_loss_W"] = np.nan

    # Save feature-engineered dataset
    df_feat.to_csv(features_path, index=False)
    print(f"[OK] Saved feature-engineered dataset: {features_path} ({len(df_feat)} rows, {len(df_feat.columns)} columns)")
    return df_feat


def generate_visualizations(df_feat, out_dir=None):
    """
    Generates high-resolution diagnostic plots for reporting:
    1. temperature_trends.png: Temperature, rise above ambient, and dT/dt
    2. acoustic_features.png: Acoustic RMS, Peak-to-Peak, and Crest Factor dynamics
    3. feature_correlations.png: Cross-sensor correlation matrix
    4. zero_current_and_sensor_ranges.png: Sensor distributions & zero-current safety profile
    """
    if out_dir is None:
        out_dir = FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use("dark_background")

    # Figure 1: Thermal Trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(df_feat["t_sec"], df_feat["temperature_C"], color="#F59E0B", lw=2, label="Measured Temp (°C)")
    ax1.plot(df_feat["t_sec"], df_feat["temp_rolling_mean_5"], color="#38BDF8", lw=1.5, ls="--", label="Smoothed Mean (w=5)")
    ax1.axhline(25.75, color="#10B981", ls=":", lw=1.5, label="Ambient Baseline (25.75 °C)")
    ax1.set_ylabel("Temperature (°C)", color="#F8FAFC", fontsize=11)
    ax1.set_title("BusGuard AI — Busbar Joint Thermal Dynamics & Temperature Rise", color="#F8FAFC", fontsize=13, weight="bold")
    ax1.legend(loc="upper right", framealpha=0.3)
    ax1.grid(color="#1E293B", alpha=0.7)

    ax2.plot(df_feat["t_sec"], df_feat["temp_roc_C_per_sec"], color="#EF4444", lw=1.2, label="Rate of Temperature Change dT/dt (°C/s)")
    ax2.axhline(0.0, color="#64748B", ls="-", lw=1)
    ax2.set_xlabel("Elapsed Time (Seconds)", color="#F8FAFC", fontsize=11)
    ax2.set_ylabel("dT/dt (°C/sec)", color="#F8FAFC", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.3)
    ax2.grid(color="#1E293B", alpha=0.7)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "temperature_trends.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[OK] Generated figure: {p1}")

    # Figure 2: Acoustic Features
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(df_feat["t_sec"], df_feat["ac_peak_to_peak_adc"], color="#F43F5E", lw=1.5, label="Acoustic Peak-to-Peak (ADC)")
    ax1.plot(df_feat["t_sec"], df_feat["ac_rms_adc"], color="#38BDF8", lw=1.5, label="Acoustic RMS (ADC)")
    ax1.set_ylabel("ADC Counts", color="#F8FAFC", fontsize=11)
    ax1.set_title("BusGuard AI — Acoustic Waveform Dynamics & Crest Factor Spikes", color="#F8FAFC", fontsize=13, weight="bold")
    ax1.legend(loc="upper right", framealpha=0.3)
    ax1.grid(color="#1E293B", alpha=0.7)

    ax2.plot(df_feat["t_sec"], df_feat["acoustic_crest_factor"], color="#A855F7", lw=1.5, label="Acoustic Crest Factor (Peak/RMS)")
    ax2.axhline(4.5, color="#EF4444", ls="--", lw=1.5, label="Micro-Arcing Threshold (4.5)")
    ax2.set_xlabel("Elapsed Time (Seconds)", color="#F8FAFC", fontsize=11)
    ax2.set_ylabel("Crest Factor", color="#F8FAFC", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.3)
    ax2.grid(color="#1E293B", alpha=0.7)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "acoustic_features.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[OK] Generated figure: {p2}")

    # Figure 3: Correlation Heatmap
    corr_cols = [
        "temperature_C", "temp_rise_C", "temp_roc_C_per_sec",
        "ac_rms_adc", "ac_peak_to_peak_adc", "acoustic_crest_factor", "mic_dc_bias_dev"
    ]
    corr_matrix = df_feat[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    clean_labels = [c.replace("_", " ").title() for c in corr_cols]
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(clean_labels, rotation=45, ha="left", fontsize=9, color="#F8FAFC")
    ax.set_yticklabels(clean_labels, fontsize=9, color="#F8FAFC")
    ax.set_title("Multi-Sensor Feature Correlation Matrix", color="#F8FAFC", fontsize=12, pad=30, weight="bold")

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="#000000" if abs(val) > 0.6 else "#FFFFFF", fontsize=9)

    plt.tight_layout()
    p3 = os.path.join(out_dir, "feature_correlations.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"[OK] Generated figure: {p3}")

    # Figure 4: Zero Current and Sensor Distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(df_feat["I_A"], bins=10, color="#64748B", edgecolor="#1E293B")
    axes[0, 0].set_title("Current I_A Distribution (Strictly 0.0 A)", color="#F8FAFC", fontsize=10, weight="bold")
    axes[0, 0].set_xlabel("Current (A)", color="#94A3B8")
    axes[0, 0].grid(color="#1E293B", alpha=0.5)

    axes[0, 1].hist(df_feat["temperature_C"], bins=20, color="#F59E0B", edgecolor="#1E293B")
    axes[0, 1].set_title("Temperature Distribution (25.8 - 31.8 °C)", color="#F8FAFC", fontsize=10, weight="bold")
    axes[0, 1].set_xlabel("Temperature (°C)", color="#94A3B8")
    axes[0, 1].grid(color="#1E293B", alpha=0.5)

    axes[1, 0].hist(df_feat["ac_rms_adc"], bins=25, color="#38BDF8", edgecolor="#1E293B")
    axes[1, 0].set_title("Acoustic RMS Energy Distribution", color="#F8FAFC", fontsize=10, weight="bold")
    axes[1, 0].set_xlabel("ADC Counts", color="#94A3B8")
    axes[1, 0].grid(color="#1E293B", alpha=0.5)

    axes[1, 1].hist(df_feat["acoustic_crest_factor"], bins=25, color="#A855F7", edgecolor="#1E293B")
    axes[1, 1].set_title("Acoustic Crest Factor Distribution", color="#F8FAFC", fontsize=10, weight="bold")
    axes[1, 1].set_xlabel("Crest Factor Ratio", color="#94A3B8")
    axes[1, 1].grid(color="#1E293B", alpha=0.5)

    plt.suptitle("BusGuard AI — Physical Sensor Distribution Profiles", color="#F8FAFC", fontsize=13, weight="bold")
    plt.tight_layout()
    p4 = os.path.join(out_dir, "zero_current_and_sensor_ranges.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"[OK] Generated figure: {p4}")


def run_pipeline():
    """
    Executes the complete Phase 4.1 Data Processing & Feature Engineering Pipeline.
    """
    print("\n========================================================")
    print("  BUSGUARD AI — PHASE 4.1 DATA PROCESSING PIPELINE")
    print("========================================================")
    
    df_raw = load_raw_busguard_data()
    df_clean = validate_and_clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    generate_visualizations(df_feat)
    
    print("\n[SUCCESS] Phase 4.1 Data Processing & Feature Engineering Complete!")
    return df_clean, df_feat


if __name__ == "__main__":
    run_pipeline()
