"""
BusGuard AI — Prediction & Inference Engine
============================================
Provides unified inference routines for:
1. Supervised Connection Condition Classification (Random Forest)
2. Unsupervised Acoustic & Thermal Burst Anomaly Detection (Isolation Forest)
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_RF_MODEL = None
_ISO_MODEL = None


def load_rf_model():
    global _RF_MODEL
    if _RF_MODEL is None:
        path = os.path.join(MODELS_DIR, "busbar_rf_classifier.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}. Run ml/train_model.py first.")
        _RF_MODEL = joblib.load(path)
    return _RF_MODEL


def load_iso_model():
    global _ISO_MODEL
    if _ISO_MODEL is None:
        path = os.path.join(MODELS_DIR, "acoustic_anomaly_detector.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Anomaly detector not found at {path}. Run ml/train_model.py first.")
        _ISO_MODEL = joblib.load(path)
    return _ISO_MODEL


def predict_connection_condition(
    contact_resistance_mOhm,
    initial_resistance_mOhm=0.8,
    exposure_hours=100.0,
    exposure_cycles=20.0,
    prior_field_age_years=2.0,
    dielectric_pass=1,
    is_damp_heat=0
):
    """
    Predicts the physical connection condition using the trained Random Forest Classifier.
    Returns predicted class, class probabilities, confidence score, and explainable rationale.
    """
    model = load_rf_model()

    r0 = max(0.1, float(initial_resistance_mOhm))
    rc = max(0.0, float(contact_resistance_mOhm))
    r_delta = max(0.0, rc - r0)
    r_ratio = rc / r0

    features = [
        rc,
        r0,
        r_delta,
        r_ratio,
        float(exposure_hours),
        float(exposure_cycles),
        float(prior_field_age_years),
        int(dielectric_pass),
        int(is_damp_heat)
    ]

    feature_cols = [
        "contact_resistance_mOhm",
        "initial_resistance_mOhm",
        "resistance_delta_mOhm",
        "resistance_ratio",
        "exposure_hours",
        "exposure_cycles",
        "prior_field_age_years",
        "dielectric_pass",
        "stress_type_code"
    ]

    df_in = pd.DataFrame([features], columns=feature_cols)

    pred_class = model.predict(df_in)[0]
    probs = model.predict_proba(df_in)[0]
    classes = list(model.classes_)
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
    confidence = float(np.max(probs))

    # Generate explainable engineering rationale
    if pred_class == "Tight connection":
        explanation = (
            f"Contact resistance ({rc:.2f} mΩ) is within healthy nominal range (≤ 2.0 mΩ). "
            f"Resistance ratio ({r_ratio:.2f}x) confirms minimal oxidation of contact asperities."
        )
    elif pred_class == "Intermediate connection":
        explanation = (
            f"Contact resistance ({rc:.2f} mΩ) has risen into warning band (2.0 - 10.0 mΩ). "
            f"Resistance increased by {r_delta:.2f} mΩ ({r_ratio:.2f}x initial), indicating early joint loosening or thermal oxidation."
        )
    else:
        explanation = (
            f"CRITICAL: Severe contact resistance ({rc:.2f} mΩ > 10.0 mΩ). "
            f"Resistance is {r_ratio:.1f}x baseline with high Joule heating hazard. Immediate bolt retorquing required."
        )

    return {
        "predicted_condition": pred_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "explanation": explanation,
        "features_used": dict(zip(feature_cols, features))
    }


def predict_acoustic_anomaly(
    temp_rise_C=0.0,
    temp_roc_C_per_sec=0.0,
    ac_rms_adc=135.0,
    ac_peak_to_peak_adc=900,
    acoustic_crest_factor=2.35,
    mic_dc_bias_dev=0.0
):
    """
    Evaluates acoustic and thermal multi-sensor telemetry using the Isolation Forest.
    Returns binary anomaly status and severity.
    """
    model = load_iso_model()

    features = [
        float(temp_rise_C),
        float(temp_roc_C_per_sec),
        float(ac_rms_adc),
        float(ac_peak_to_peak_adc),
        float(acoustic_crest_factor),
        float(mic_dc_bias_dev)
    ]
    feature_cols = [
        "temp_rise_C",
        "temp_roc_C_per_sec",
        "ac_rms_adc",
        "ac_peak_to_peak_adc",
        "acoustic_crest_factor",
        "mic_dc_bias_dev"
    ]

    df_in = pd.DataFrame([features], columns=feature_cols)

    pred = model.predict(df_in)[0]  # 1 = normal, -1 = anomaly
    score = float(model.decision_function(df_in)[0])

    is_anomaly = bool(pred == -1)

    if is_anomaly:
        if ac_peak_to_peak_adc > 2200 or temp_rise_C > 20.0:
            severity = "CRITICAL"
        else:
            severity = "WARNING"
    else:
        severity = "NORMAL"

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": score,
        "severity": severity,
        "features_used": dict(zip(feature_cols, features))
    }


if __name__ == "__main__":
    print("Testing ML Inference...")
    res1 = predict_connection_condition(1.2, initial_resistance_mOhm=0.9)
    print("Test 1 (1.2 mOhm):", res1["predicted_condition"], f"({res1['confidence'] * 100:.1f}%)")

    res2 = predict_connection_condition(5.4, initial_resistance_mOhm=0.9)
    print("Test 2 (5.4 mOhm):", res2["predicted_condition"], f"({res2['confidence'] * 100:.1f}%)")

    res3 = predict_connection_condition(18.5, initial_resistance_mOhm=0.9)
    print("Test 3 (18.5 mOhm):", res3["predicted_condition"], f"({res3['confidence'] * 100:.1f}%)")

    anom = predict_acoustic_anomaly(temp_rise_C=25.0, ac_peak_to_peak_adc=2800)
    print("Acoustic Anomaly Test:", anom["severity"], f"(Score: {anom['anomaly_score']:.3f})")

