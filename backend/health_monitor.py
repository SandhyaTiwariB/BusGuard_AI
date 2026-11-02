"""
BusGuard AI — Joint Health Monitoring & Maintenance Engine
==========================================================
Computes transparent, physics-calibrated Joint Health Indices (0-100%),
assigns alert levels (NORMAL, ATTENTION, CRITICAL), and generates
explainable engineering recommendations.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def calculate_health_index(
    contact_resistance_mOhm,
    temperature_C,
    ambient_temp_C=25.75,
    acoustic_p2p_adc=950,
    acoustic_crest_factor=2.4
):
    """
    Computes a transparent, explainable 0-100% Joint Health Index:
    Health Index = 100 - (Resistance Penalty + Thermal Penalty + Acoustic Penalty)

    Thresholds:
      - Normal:               80.0% - 100.0%
      - Attention Required:   50.0% - 79.9%
      - Inspection Recommended: < 50.0%
    """
    # Safe handling for NaN / None / Missing values
    res_missing = False
    if contact_resistance_mOhm is None or (isinstance(contact_resistance_mOhm, float) and np.isnan(contact_resistance_mOhm)):
        rc = 0.0
        res_missing = True
    else:
        try:
            rc = max(0.0, float(contact_resistance_mOhm))
        except (ValueError, TypeError):
            rc = 0.0
            res_missing = True

    try:
        temp = float(temperature_C)
    except (ValueError, TypeError):
        temp = 25.75

    try:
        t_ambient = float(ambient_temp_C)
    except (ValueError, TypeError):
        t_ambient = 25.75

    temp_rise = max(0.0, temp - t_ambient)

    try:
        p2p = float(acoustic_p2p_adc)
    except (ValueError, TypeError):
        p2p = 950.0

    try:
        cf = float(acoustic_crest_factor)
    except (ValueError, TypeError):
        cf = 2.4

    # 1. Electrical Resistance Penalty (Max 50 points)
    # Contact resistance is the direct root cause of joint deterioration (Holm's theory)
    if res_missing:
        p_res = 0.0
    elif rc <= 2.0:
        p_res = (rc / 2.0) * 5.0  # 0 to 5 pts
    elif rc <= 10.0:
        p_res = 10.0 + ((rc - 2.0) / 8.0) * 25.0  # 10 to 35 pts
    else:
        p_res = 35.0 + min(15.0, ((rc - 10.0) / 20.0) * 15.0)  # 35 to 50 pts

    # 2. Thermal Penalty (Max 30 points)
    # Excess Joule heating (I^2 * Rc) produces temperature rise above ambient
    if temp_rise <= 5.0:
        p_therm = 0.0
    elif temp_rise <= 25.0:
        p_therm = ((temp_rise - 5.0) / 20.0) * 15.0  # 0 to 15 pts
    else:
        p_therm = 15.0 + min(15.0, ((temp_rise - 25.0) / 35.0) * 15.0)  # 15 to 30 pts

    # 3. Acoustic Penalty (Max 20 points)
    # Acoustic emissions indicate micro-arcing across loose contact interfaces or mechanical fretting
    p_ac = 0.0
    if p2p > 1600 or cf > 3.5:
        p_ac += min(10.0, ((p2p - 1600) / 1000.0) * 10.0)
    if cf > 4.5:
        p_ac += min(10.0, ((cf - 4.5) / 3.0) * 10.0)
    p_ac = min(20.0, p_ac)

    total_penalty = p_res + p_therm + p_ac
    health_index = max(5.0, min(100.0, 100.0 - total_penalty))
    health_index = round(health_index, 1)

    # Determine status & recommendation
    if res_missing:
        recommendation = "Resistance data not available (zero-current or inactive load). Health Index based on thermal and acoustic channels."
        if health_index >= 80.0:
            status = "NORMAL"
            badge_color = "#10B981"
        elif health_index >= 50.0:
            status = "ATTENTION"
            badge_color = "#F59E0B"
        else:
            status = "CRITICAL"
            badge_color = "#EF4444"
    elif health_index >= 80.0:
        status = "NORMAL"
        badge_color = "#10B981"  # Emerald green
        recommendation = "Nominal operational state. Joint contact asperities tight. Maintain standard quarterly maintenance schedule."
    elif health_index >= 50.0:
        status = "ATTENTION"
        badge_color = "#F59E0B"  # Amber
        recommendation = "Moderate degradation. Contact resistance has elevated into warning band. Schedule thermal imaging and torque check during next planned shutdown."
    else:
        status = "CRITICAL"
        badge_color = "#EF4444"  # Red
        recommendation = "CRITICAL HAZARD: Severe contact resistance escalation with high thermal or acoustic discharge. Immediate physical retorquing required to prevent thermal runaway."

    return {
        "health_index": health_index,
        "status": status,
        "badge_color": badge_color,
        "recommendation": recommendation,
        "penalties": {
            "resistance_penalty": round(p_res, 1),
            "thermal_penalty": round(p_therm, 1),
            "acoustic_penalty": round(p_ac, 1)
        },
        "metrics": {
            "contact_resistance_mOhm": rc if not res_missing else None,
            "temp_rise_C": round(temp_rise, 1),
            "acoustic_p2p_adc": p2p,
            "acoustic_crest_factor": round(cf, 2)
        }
    }


# Alias for backward and forward compatibility
calculate_joint_health_index = calculate_health_index



def load_monitored_joints():
    """
    Loads all monitored busbar joints from data/processed/multijoint_busbar_status.csv.
    Recalculates instantaneous health indicators.
    """
    path = os.path.join(PROCESSED_DIR, "multijoint_busbar_status.csv")
    if not os.path.exists(path):
        from ml.data_processing import run_pipeline
        run_pipeline()

    df = pd.read_csv(path)

    # Update health indices dynamically with transparent physics formula
    updated = []
    for _, row in df.iterrows():
        hi_res = calculate_health_index(
            contact_resistance_mOhm=row["contact_resistance_mOhm"],
            temperature_C=row["temperature_C"],
            acoustic_p2p_adc=row["acoustic_p2p_adc"],
            acoustic_crest_factor=row["acoustic_crest_factor"]
        )
        row_dict = row.to_dict()
        row_dict["health_index"] = hi_res["health_index"]
        row_dict["alert_status"] = hi_res["status"]
        row_dict["badge_color"] = hi_res["badge_color"]
        row_dict["recommendation"] = hi_res["recommendation"]
        row_dict["resistance_penalty"] = hi_res["penalties"]["resistance_penalty"]
        row_dict["thermal_penalty"] = hi_res["penalties"]["thermal_penalty"]
        row_dict["acoustic_penalty"] = hi_res["penalties"]["acoustic_penalty"]
        updated.append(row_dict)

    return pd.DataFrame(updated)


def get_system_summary():
    """
    Calculates executive system KPIs across the busbar switchgear network.
    """
    df = load_monitored_joints()
    total = len(df)
    normal = int((df["alert_status"] == "NORMAL").sum())
    attention = int((df["alert_status"] == "ATTENTION").sum())
    critical = int((df["alert_status"] == "CRITICAL").sum())
    avg_health = float(df["health_index"].mean())
    max_temp = float(df["temperature_C"].max())
    max_rc = float(df["contact_resistance_mOhm"].max())

    return {
        "total_joints": total,
        "normal_joints": normal,
        "attention_joints": attention,
        "critical_joints": critical,
        "avg_system_health": round(avg_health, 1),
        "max_temperature_C": round(max_temp, 1),
        "max_contact_resistance_mOhm": round(max_rc, 2),
        "df_joints": df
    }


if __name__ == "__main__":
    summary = get_system_summary()
    print("=== BusGuard AI Health Engine Test ===")
    print(f"Total Joints:      {summary['total_joints']}")
    print(f"Normal:            {summary['normal_joints']}")
    print(f"Attention:         {summary['attention_joints']}")
    print(f"Critical:          {summary['critical_joints']}")
    print(f"System Health:     {summary['avg_system_health']}%")

