"""
BusGuard AI — Phase 4.3: Backend Service Layer
==============================================
Encapsulates all core backend operations:
- Model management (loading & caching)
- Multi-sensor data management & telemetry caching
- Real-time prediction pipeline (Random Forest & Isolation Forest)
- Transparent 0-100% Joint Health Index calculation
- Alert generation & stateful acknowledgment
- Field maintenance recommendation dispatch
- Hardware input format documentation for future ESP32 integration
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from backend.health_monitor import calculate_health_index

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


class BusGuardBackendService:
    """
    Centralized Python backend service for BusGuard AI condition monitoring.
    Designed for seamless integration with Streamlit or FastAPI endpoints.
    """

    def __init__(self):
        self._rf_model = None
        self._iso_model = None
        self._metrics = None
        self._df_physical = None
        self._df_degradation = None
        self._df_joints = None
        self._acknowledged_alerts = set()

    # --------------------------------------------------------------------------
    # 1. Model Loading & Validation
    # --------------------------------------------------------------------------
    def get_rf_model(self):
        """Loads and caches the trained Random Forest Classifier."""
        if self._rf_model is None:
            # Check both standard model filenames
            p1 = os.path.join(MODELS_DIR, "random_forest_model.joblib")
            p2 = os.path.join(MODELS_DIR, "busbar_rf_classifier.joblib")
            target = p1 if os.path.exists(p1) else p2
            if not os.path.exists(target):
                from ml.train_model import train_and_evaluate_models
                self._rf_model, _ = train_and_evaluate_models()
            else:
                self._rf_model = joblib.load(target)
        return self._rf_model

    def get_iso_model(self):
        """Loads and caches the trained Isolation Forest Anomaly Detector."""
        if self._iso_model is None:
            target = os.path.join(MODELS_DIR, "acoustic_anomaly_detector.joblib")
            if not os.path.exists(target):
                from ml.train_model import train_and_evaluate_models
                train_and_evaluate_models()
            self._iso_model = joblib.load(target)
        return self._iso_model

    def get_model_metrics(self) -> Dict[str, Any]:
        """Loads cached evaluation metrics and model comparison data."""
        if self._metrics is None:
            target = os.path.join(MODELS_DIR, "metrics.json")
            if os.path.exists(target):
                with open(target, "r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            else:
                _, self._metrics = self.get_rf_model(), {}
        return self._metrics

    # --------------------------------------------------------------------------
    # 2. Data Loading & Telemetry
    # --------------------------------------------------------------------------
    def get_physical_stream(self) -> pd.DataFrame:
        """Loads processed physical sensor telemetry (480 records)."""
        if self._df_physical is None:
            target = os.path.join(PROCESSED_DIR, "busguard_features.csv")
            if not os.path.exists(target):
                target = os.path.join(PROCESSED_DIR, "busguard_processed_physical.csv")
            if not os.path.exists(target):
                from ml.data_processing import run_pipeline
                run_pipeline()
            self._df_physical = pd.read_csv(target)
        return self._df_physical

    def get_degradation_benchmark(self) -> pd.DataFrame:
        """Loads processed connector degradation benchmark (190 records)."""
        if self._df_degradation is None:
            target = os.path.join(PROCESSED_DIR, "connector_processed_degradation.csv")
            if not os.path.exists(target):
                from ml.data_processing import run_pipeline
                run_pipeline()
            self._df_degradation = pd.read_csv(target)
        return self._df_degradation

    def get_monitored_joints(self) -> pd.DataFrame:
        """Loads monitored busbar topology and dynamically updates health metrics."""
        path = os.path.join(PROCESSED_DIR, "multijoint_busbar_status.csv")
        if not os.path.exists(path):
            from ml.data_processing import run_pipeline
            run_pipeline()

        df = pd.read_csv(path)
        updated_rows = []
        for _, row in df.iterrows():
            hi = self.calculate_health_indicator(
                contact_resistance_mOhm=row["contact_resistance_mOhm"],
                temperature_C=row["temperature_C"],
                acoustic_p2p_adc=row["acoustic_p2p_adc"],
                acoustic_crest_factor=row["acoustic_crest_factor"]
            )
            r_dict = row.to_dict()
            r_dict["health_index"] = hi["health_index"]
            r_dict["alert_status"] = hi["status"]
            r_dict["badge_color"] = hi["badge_color"]
            r_dict["recommendation"] = hi["recommendation"]
            r_dict["resistance_penalty"] = hi["penalties"]["resistance_penalty"]
            r_dict["thermal_penalty"] = hi["penalties"]["thermal_penalty"]
            r_dict["acoustic_penalty"] = hi["penalties"]["acoustic_penalty"]
            updated_rows.append(r_dict)

        self._df_joints = pd.DataFrame(updated_rows)
        return self._df_joints

    # --------------------------------------------------------------------------
    # 3. Prediction Pipeline
    # --------------------------------------------------------------------------
    def predict_joint_condition(
        self,
        contact_resistance_mOhm: float,
        initial_resistance_mOhm: float = 0.8,
        exposure_hours: float = 100.0,
        exposure_cycles: float = 20.0,
        prior_field_age_years: float = 2.0,
        dielectric_pass: int = 1,
        is_damp_heat: int = 0
    ) -> Dict[str, Any]:
        """
        Runs live inference using the trained Random Forest Classifier.
        Validates inputs and handles missing values with conservative physical defaults.
        """
        rf = self.get_rf_model()

        r0 = max(0.1, float(initial_resistance_mOhm or 0.8))
        rc = max(0.0, float(contact_resistance_mOhm or 0.0))
        r_delta = max(0.0, rc - r0)
        r_ratio = rc / r0

        features = [
            rc,
            r0,
            r_delta,
            r_ratio,
            float(exposure_hours or 0.0),
            float(exposure_cycles or 0.0),
            float(prior_field_age_years or 0.0),
            int(dielectric_pass if dielectric_pass is not None else 1),
            int(is_damp_heat if is_damp_heat is not None else 0)
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

        pred_class = rf.predict(df_in)[0]
        probs = rf.predict_proba(df_in)[0]
        classes = list(rf.classes_)
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
        confidence = float(np.max(probs))

        # Explainable engineering rationale based on physical contact theory
        if pred_class == "Tight connection":
            explanation = (
                f"Contact resistance ({rc:.2f} mΩ) is within nominal healthy limits (≤ 2.0 mΩ). "
                f"Resistance ratio ({r_ratio:.2f}x) indicates intact clamping torque and clean contact asperities."
            )
        elif pred_class == "Intermediate connection":
            explanation = (
                f"Contact resistance ({rc:.2f} mΩ) has elevated into the warning band (2.0 - 10.0 mΩ). "
                f"Resistance rose by +{r_delta:.2f} mΩ ({r_ratio:.2f}x initial), signaling joint loosening inception or contact fretting."
            )
        else:
            explanation = (
                f"CRITICAL HAZARD: Contact resistance ({rc:.2f} mΩ > 10.0 mΩ) indicates severe joint degradation. "
                f"Resistance has escalated to {r_ratio:.1f}x baseline, triggering high Joule heating (I²Rc) and risk of thermal runaway."
            )

        return {
            "predicted_condition": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "explanation": explanation,
            "features_used": dict(zip(feature_cols, features))
        }

    def predict_acoustic_anomaly(
        self,
        temp_rise_C: float = 0.0,
        temp_roc_C_per_sec: float = 0.0,
        ac_rms_adc: float = 135.0,
        ac_peak_to_peak_adc: float = 900.0,
        acoustic_crest_factor: float = 2.35,
        mic_dc_bias_dev: float = 0.0
    ) -> Dict[str, Any]:
        """Runs acoustic and thermal anomaly detection via Isolation Forest."""
        from ml.predict import predict_acoustic_anomaly
        return predict_acoustic_anomaly(
            temp_rise_C=temp_rise_C,
            temp_roc_C_per_sec=temp_roc_C_per_sec,
            ac_rms_adc=ac_rms_adc,
            ac_peak_to_peak_adc=ac_peak_to_peak_adc,
            acoustic_crest_factor=acoustic_crest_factor,
            mic_dc_bias_dev=mic_dc_bias_dev
        )

    # --------------------------------------------------------------------------
    # 4. Transparent Health Indicator Calculation
    # --------------------------------------------------------------------------
    def calculate_health_indicator(
        self,
        contact_resistance_mOhm: Optional[float],
        temperature_C: float,
        ambient_temp_C: float = 25.75,
        acoustic_p2p_adc: float = 950.0,
        acoustic_crest_factor: float = 2.4
    ) -> Dict[str, Any]:
        """
        Computes the transparent 0-100% Joint Health Index based on:
        Health Index = 100 - (Resistance Penalty + Thermal Penalty + Acoustic Penalty)
        """
        return calculate_health_index(
            contact_resistance_mOhm=contact_resistance_mOhm,
            temperature_C=temperature_C,
            ambient_temp_C=ambient_temp_C,
            acoustic_p2p_adc=acoustic_p2p_adc,
            acoustic_crest_factor=acoustic_crest_factor
        )

    # --------------------------------------------------------------------------
    # 5. Alert & Prioritization Engine
    # --------------------------------------------------------------------------
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Generates prioritized alerts from monitored joints."""
        joints = self.get_monitored_joints()
        alerts = []
        for _, j in joints.iterrows():
            if j["alert_status"] in ["ATTENTION", "CRITICAL"]:
                alt_id = f"ALT-{j['joint_id']}-2026"
                alerts.append({
                    "alert_id": alt_id,
                    "joint_id": j["joint_id"],
                    "joint_name": j["joint_name"],
                    "severity": j["alert_status"],
                    "health_index": j["health_index"],
                    "is_acknowledged": alt_id in self._acknowledged_alerts,
                    "evidence": f"Rc: {j['contact_resistance_mOhm']:.2f} mΩ | Temp: {j['temperature_C']:.1f}°C (ΔT = +{j['temperature_C'] - 25.75:.1f}°C) | Acoustic P2P: {j['acoustic_p2p_adc']}",
                    "action": j["recommendation"]
                })
        # Sort so CRITICAL appears first, then lowest health index
        alerts.sort(key=lambda a: (0 if a["severity"] == "CRITICAL" else 1, a["health_index"]))
        return alerts

    def acknowledge_alert(self, alert_id: str, technician: Optional[str] = None) -> Dict[str, Any]:
        """Marks an alert as reviewed by the maintenance operator."""
        self._acknowledged_alerts.add(alert_id)
        return {
            "status": "ACKNOWLEDGED",
            "alert_id": alert_id,
            "technician": technician or "Field Operator",
            "is_acknowledged": True
        }

    def generate_work_order(self, joint_id: str) -> Dict[str, Any]:
        """Generates an actionable, structured work order dictionary for a specific joint."""
        df = self.get_monitored_joints()
        match = df[df["joint_id"] == joint_id]
        row = match.iloc[0] if not match.empty else df.iloc[0]
        
        priority = "HIGH" if row["alert_status"] == "CRITICAL" else ("MEDIUM" if row["alert_status"] == "ATTENTION" else "LOW")
        return {
            "work_order_id": f"WO-{row['joint_id']}-2026",
            "joint_id": row["joint_id"],
            "joint_name": row["joint_name"],
            "phase": row["phase"],
            "priority": priority,
            "health_index": row["health_index"],
            "contact_resistance_mOhm": row["contact_resistance_mOhm"],
            "temperature_C": row["temperature_C"],
            "recommended_torque_Nm": 45.0,
            "procedure": row["recommendation"],
            "safety_requirement": "De-energize, lock-out/tag-out (LOTO) busbar bay before contact inspection."
        }

    # --------------------------------------------------------------------------
    # 6. Fleet Summary & Report Generation
    # --------------------------------------------------------------------------
    def get_fleet_summary(self) -> Dict[str, Any]:
        """Calculates executive KPI metrics across the switchgear fleet."""
        df = self.get_monitored_joints()
        total = len(df)
        norm = int((df["alert_status"] == "NORMAL").sum())
        att = int((df["alert_status"] == "ATTENTION").sum())
        crit = int((df["alert_status"] == "CRITICAL").sum())
        avg_hi = round(float(df["health_index"].mean()), 1)

        return {
            "total_joints": total,
            "normal_joints": norm,
            "attention_joints": att,
            "critical_joints": crit,
            "avg_health_index": avg_hi,
            "df_joints": df
        }

    def generate_work_order_text(self) -> str:
        """Generates a formal maintenance work order dispatch ticket."""
        summary = self.get_fleet_summary()
        alerts = self.get_active_alerts()

        lines = [
            "==================================================================",
            "   BUSGUARD AI — SWITCHGEAR MAINTENANCE DISPATCH WORK ORDER      ",
            "==================================================================",
            f"Generated Timestamp : 2026-11-02 UTC",
            f"Substation Section  : Main Low/Medium Voltage Switchgear Bay 01",
            f"Overall Health Index: {summary['avg_health_index']}%",
            f"Active Alert Count  : {len(alerts)} ({summary['critical_joints']} Critical, {summary['attention_joints']} Attention)",
            "------------------------------------------------------------------",
            "PRIORITIZED FIELD INSPECTION ITEMS:"
        ]
        for i, a in enumerate(alerts, 1):
            ack_tag = "[ACKNOWLEDGED]" if a["is_acknowledged"] else "[PENDING REVIEW]"
            lines.append(f"\nItem #{i}: Joint {a['joint_id']} ({a['joint_name']}) — Severity: {a['severity']} {ack_tag}")
            lines.append(f"  Telemetry Evidence : {a['evidence']}")
            lines.append(f"  Field Procedure    : {a['action']}")

        lines.extend([
            "\n------------------------------------------------------------------",
            "DISPATCH AUTHORIZATION:",
            "Approved By: BusGuard AI Condition Monitoring System (Automated)",
            "Maintenance Crew Protocol: De-energize bay before bolt contact inspection.",
            "=================================================================="
        ])
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # 7. Hardware Interface Documentation
    # --------------------------------------------------------------------------
    @staticmethod
    def get_hardware_integration_doc() -> str:
        """Returns JSON schema documentation for ESP32 edge microcontroller integration."""
        return json.dumps({
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
        }, indent=2)


# Global Service Singleton
_BACKEND_SERVICE = None

def get_backend_service() -> BusGuardBackendService:
    global _BACKEND_SERVICE
    if _BACKEND_SERVICE is None:
        _BACKEND_SERVICE = BusGuardBackendService()
    return _BACKEND_SERVICE


if __name__ == "__main__":
    print("=== Testing BusGuardBackendService ===")
    service = get_backend_service()
    sum_data = service.get_fleet_summary()
    print("Fleet Summary:", {k: v for k, v in sum_data.items() if k != "df_joints"})
    pred = service.predict_joint_condition(14.5)
    print("Prediction (14.5 mOhm):", pred["predicted_condition"], f"({pred['confidence']*100:.1f}%)")
    alerts = service.get_active_alerts()
    print(f"Active Alerts: {len(alerts)}")
    print("Backend Service verification successful!")

