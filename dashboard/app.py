"""
BusGuard AI — Industrial Multi-Sensor Busbar Condition Monitoring System
========================================================================
Interactive Industrial Monitoring Interface built with Streamlit & Plotly.
Connected to real datasets, trained Random Forest Classifier, and Isolation Forest.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.health_monitor import load_monitored_joints, get_system_summary, calculate_health_index
from backend.service import get_backend_service
from ml.predict import predict_connection_condition, predict_acoustic_anomaly, load_rf_model

# Page Configuration
st.set_page_config(
    page_title="BusGuard AI — Busbar Condition Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def inject_custom_css():
    css_path = os.path.join(BASE_DIR, "dashboard", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_custom_css()

# Color Palette
DARK_BG = "#0B1120"
CARD_BG = "#0F172A"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#F8FAFC"
ACCENT_BLUE = "#38BDF8"
ACCENT_TEAL = "#14B8A6"
STATUS_NORMAL = "#10B981"
STATUS_WARNING = "#F59E0B"
STATUS_CRITICAL = "#EF4444"

# Load Datasets
@st.cache_data
def load_datasets():
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    phys_p = os.path.join(processed_dir, "busguard_processed_physical.csv")
    deg_p = os.path.join(processed_dir, "connector_processed_degradation.csv")
    
    if not os.path.exists(phys_p) or not os.path.exists(deg_p):
        from ml.data_processing import run_pipeline
        run_pipeline()
        
    df_phys = pd.read_csv(phys_p)
    df_deg = pd.read_csv(deg_p)
    return df_phys, df_deg

df_physical, df_degradation = load_datasets()
system_summary = get_system_summary()
df_joints = system_summary["df_joints"]

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div style="font-size: 28px;">⚡</div>
        <div>
            <div class="brand-title">BusGuard AI</div>
            <div class="brand-subtitle">Condition Monitor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="system-status-badge">
        <div class="status-dot"></div>
        <span>TELEMETRY ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Joint Monitoring",
            "Sensor Analytics",
            "AI Predictions",
            "Alerts & Maintenance",
            "Reports",
            "About System"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size: 11px; color: #64748B; line-height: 1.6;">
        <div style="font-weight: 700; color: #94A3B8; margin-bottom: 4px;">SYSTEM SPECIFICATIONS</div>
        <div>Topology: <b>3-Phase Switchgear</b></div>
        <div>Monitored Joints: <b>{system_summary['total_joints']}</b></div>
        <div>Sampling Rate: <b>1.12 Hz (892 ms)</b></div>
        <div>Core Engine: <b>Random Forest & IF</b></div>
        <div style="margin-top: 8px; color: #38BDF8;">v1.0.4-MVP</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
if page == "Overview":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #F8FAFC; margin: 0 0 4px 0;">Switchgear Busbar Operational Overview</h1>
        <p style="font-size: 14px; color: #94A3B8; margin: 0;">Real-time continuous condition monitoring across electrical, thermal and acoustic sensor streams.</p>
    </div>
    """, unsafe_allow_html=True)

    # Metric KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-topbar" style="background: {ACCENT_BLUE};"></div>
            <div class="kpi-label">Monitored Joints</div>
            <div class="kpi-value">{system_summary['total_joints']}</div>
            <div class="kpi-sub">Active Sensor Nodes</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-topbar" style="background: {STATUS_NORMAL};"></div>
            <div class="kpi-label">Healthy / Tight</div>
            <div class="kpi-value" style="color: {STATUS_NORMAL};">{system_summary['normal_joints']}</div>
            <div class="kpi-sub">Rc ≤ 2.0 mΩ</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-topbar" style="background: {STATUS_WARNING};"></div>
            <div class="kpi-label">Attention Required</div>
            <div class="kpi-value" style="color: {STATUS_WARNING};">{system_summary['attention_joints']}</div>
            <div class="kpi-sub">Moderate Degradation</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-topbar" style="background: {STATUS_CRITICAL};"></div>
            <div class="kpi-label">Critical Alerts</div>
            <div class="kpi-value" style="color: {STATUS_CRITICAL};">{system_summary['critical_joints']}</div>
            <div class="kpi-sub">Immediate Inspection</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        health_color = STATUS_NORMAL if system_summary['avg_system_health'] >= 80 else (STATUS_WARNING if system_summary['avg_system_health'] >= 60 else STATUS_CRITICAL)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-topbar" style="background: {health_color};"></div>
            <div class="kpi-label">Fleet Health Index</div>
            <div class="kpi-value" style="color: {health_color};">{system_summary['avg_system_health']}%</div>
            <div class="kpi-sub">Weighted Multi-Sensor Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Busbar Topology Diagram
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-title">
            <span>🗺️</span> 3-Phase Busbar Topology & Joint Health Mapping
        </div>
        <p style="font-size: 13px; color: #94A3B8; margin-bottom: 16px;">
            Physical layout of switchgear main busbar and distribution feeders. Joint status indicators reflect fused multi-sensor telemetry.
        </p>
    """, unsafe_allow_html=True)

    # Interactive Topology Visualization with Plotly
    topo_nodes = [
        {"id": "J01", "name": "Main Incomer A", "x": 1.5, "y": 3.0, "status": df_joints.loc[df_joints['joint_id']=='J01', 'alert_status'].values[0], "temp": 28.5, "rc": 1.31},
        {"id": "J02", "name": "Main Incomer B", "x": 1.5, "y": 2.0, "status": df_joints.loc[df_joints['joint_id']=='J02', 'alert_status'].values[0], "temp": 48.2, "rc": 5.42},
        {"id": "J03", "name": "Main Incomer C", "x": 1.5, "y": 1.0, "status": df_joints.loc[df_joints['joint_id']=='J03', 'alert_status'].values[0], "temp": 78.6, "rc": 19.02},
        {"id": "J04", "name": "Bus Coupler", "x": 4.5, "y": 2.0, "status": df_joints.loc[df_joints['joint_id']=='J04', 'alert_status'].values[0], "temp": 29.1, "rc": 1.29},
        {"id": "J05", "name": "Feeder 1-A", "x": 7.5, "y": 3.0, "status": df_joints.loc[df_joints['joint_id']=='J05', 'alert_status'].values[0], "temp": 31.4, "rc": 1.73},
        {"id": "J06", "name": "Feeder 1-B", "x": 7.5, "y": 2.0, "status": df_joints.loc[df_joints['joint_id']=='J06', 'alert_status'].values[0], "temp": 44.7, "rc": 6.58},
        {"id": "J07", "name": "Feeder 2-C", "x": 7.5, "y": 1.0, "status": df_joints.loc[df_joints['joint_id']=='J07', 'alert_status'].values[0], "temp": 30.2, "rc": 1.62},
        {"id": "J08", "name": "Neutral Bus", "x": 4.5, "y": 0.2, "status": df_joints.loc[df_joints['joint_id']=='J08', 'alert_status'].values[0], "temp": 26.8, "rc": 1.11},
    ]
    
    color_map = {"NORMAL": STATUS_NORMAL, "ATTENTION": STATUS_WARNING, "CRITICAL": STATUS_CRITICAL}
    
    fig_topo = go.Figure()

    # Busbar Conductor Lines
    # Main Busbars (Phase A, B, C)
    for y_pos, phase in [(3.0, "Phase A Busbar"), (2.0, "Phase B Busbar"), (1.0, "Phase C Busbar")]:
        fig_topo.add_trace(go.Scatter(
            x=[0.5, 8.5], y=[y_pos, y_pos],
            mode="lines",
            line=dict(color="#334155", width=8),
            hoverinfo="text",
            text=f"Copper Busway {phase} (800A Rated)",
            showlegend=False
        ))

    # Neutral Busway
    fig_topo.add_trace(go.Scatter(
        x=[0.5, 8.5], y=[0.2, 0.2],
        mode="lines",
        line=dict(color="#1E293B", width=6, dash="dash"),
        hoverinfo="text",
        text="Neutral Ground Return Busbar",
        showlegend=False
    ))

    # Add Joint Nodes
    for n in topo_nodes:
        node_color = color_map.get(n["status"], STATUS_NORMAL)
        fig_topo.add_trace(go.Scatter(
            x=[n["x"]], y=[n["y"]],
            mode="markers+text",
            marker=dict(
                size=32,
                color=node_color,
                line=dict(color="#FFFFFF", width=2),
                symbol="square"
            ),
            text=[f"<b>{n['id']}</b>"],
            textposition="middle center",
            textfont=dict(color="#000000" if n["status"] != "CRITICAL" else "#FFFFFF", size=11, family="JetBrains Mono"),
            name=n["id"],
            hoverinfo="text",
            hovertext=f"<b>Joint {n['id']} — {n['name']}</b><br>Status: {n['status']}<br>Temperature: {n['temp']} °C<br>Contact Resistance: {n['rc']} mΩ",
            showlegend=False
        ))
        
        # Subtitle label below node
        fig_topo.add_annotation(
            x=n["x"], y=n["y"] - 0.35,
            text=f"<span style='font-size:10px; color:#94A3B8;'>{n['name']}</span>",
            showarrow=False,
            font=dict(color="#94A3B8", size=10)
        )

    fig_topo.update_layout(
        plot_bgcolor="#0A0F1D",
        paper_bgcolor="#0A0F1D",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.4, 3.6]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        dragmode=False
    )
    st.plotly_chart(fig_topo, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Split: Quick Trends & Maintenance Priority
    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.markdown("""
        <div class="glass-panel">
            <div class="panel-title">
                <span>📈</span> Live Physical Multi-Sensor Stream (Telemetry Baseline)
            </div>
        """, unsafe_allow_html=True)

        # Plotly multi-sensor stream from busguard_processed_physical.csv
        fig_stream = go.Figure()
        fig_stream.add_trace(go.Scatter(
            x=df_physical["t_sec"], y=df_physical["temperature_C"],
            name="Temperature (°C)",
            line=dict(color="#F59E0B", width=2)
        ))
        fig_stream.add_trace(go.Scatter(
            x=df_physical["t_sec"], y=df_physical["ac_rms_adc"],
            name="Acoustic RMS (ADC)",
            yaxis="y2",
            line=dict(color=ACCENT_BLUE, width=1.5)
        ))
        fig_stream.update_layout(
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR, family="Plus Jakarta Sans"),
            margin=dict(l=40, r=40, t=20, b=30),
            height=260,
            legend=dict(orientation="h", y=1.1, x=0.2),
            xaxis=dict(title="Time (seconds)", gridcolor=GRID_COLOR),
            yaxis=dict(title="Temp (°C)", gridcolor=GRID_COLOR, range=[24, 34]),
            yaxis2=dict(title="Acoustic RMS", overlaying="y", side="right", showgrid=False, range=[0, 600])
        )
        st.plotly_chart(fig_stream, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-panel">
            <div class="panel-title">
                <span>🚨</span> Immediate Maintenance Prioritization
            </div>
        """, unsafe_allow_html=True)

        # Filter joints by health index ascending
        df_prio = df_joints.sort_values(by="health_index", ascending=True)
        for _, j in df_prio.iterrows():
            st_color = color_map.get(j['alert_status'], STATUS_NORMAL)
            st.markdown(f"""
            <div style="background: #0A0F1D; border-left: 4px solid {st_color}; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 13px; color: #F8FAFC;">{j['joint_id']} — {j['joint_name']}</span>
                    <span style="font-size: 11px; font-weight: 700; color: {st_color};">{j['alert_status']} ({j['health_index']}%)</span>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">
                    Rc: <b>{j['contact_resistance_mOhm']} mΩ</b> | Temp: <b>{j['temperature_C']}°C</b> | P2P: <b>{j['acoustic_p2p_adc']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 2. JOINT MONITORING PAGE
# ==============================================================================
elif page == "Joint Monitoring":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #F8FAFC; margin: 0 0 4px 0;">Individual Joint Telemetry & Diagnostics</h1>
        <p style="font-size: 14px; color: #94A3B8; margin: 0;">Detailed multi-sensor inspection and explainable condition breakdown for individual bolted joints.</p>
    </div>
    """, unsafe_allow_html=True)

    # Joint Selector Dropdown
    joint_options = [f"{row['joint_id']} — {row['joint_name']} ({row['alert_status']})" for _, row in df_joints.iterrows()]
    selected_option = st.selectbox("SELECT MONITORED BUSBAR JOINT", joint_options, index=2)
    selected_id = selected_option.split(" ")[0]
    joint_data = df_joints[df_joints["joint_id"] == selected_id].iloc[0]

    # Joint Detail Cards
    badge_style = f"background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3);"
    if joint_data['alert_status'] == "ATTENTION":
        badge_style = f"background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3);"
    elif joint_data['alert_status'] == "CRITICAL":
        badge_style = f"background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3);"

    col_meta1, col_meta2 = st.columns([1.5, 1.0])
    with col_meta1:
        st.markdown(f"""
        <div class="glass-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <h3 style="margin: 0; color: #F8FAFC; font-size: 20px;">Joint {joint_data['joint_id']}: {joint_data['joint_name']}</h3>
                <span style="{badge_style} padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 12px;">{joint_data['alert_status']}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">
                <div><span style="color: #64748B; font-size: 11px;">BUSBAR SECTION</span><br><b>{joint_data['section']}</b></div>
                <div><span style="color: #64748B; font-size: 11px;">PHASE</span><br><b>{joint_data['phase']}</b></div>
                <div><span style="color: #64748B; font-size: 11px;">RATED TORQUE</span><br><b>{joint_data['torque_rating_Nm']} Nm</b></div>
                <div><span style="color: #64748B; font-size: 11px;">MEASURED TORQUE</span><br><b style="color: {STATUS_CRITICAL if joint_data['measured_torque_Nm'] < 30 else TEXT_COLOR};">{joint_data['measured_torque_Nm']} Nm</b></div>
            </div>
            <div style="background: #0A0F1D; padding: 14px; border-radius: 8px; border-left: 4px solid {joint_data['badge_color']};">
                <div style="font-size: 11px; font-weight: 700; color: #94A3B8; margin-bottom: 4px;">ENGINEERING RECOMMENDATION</div>
                <div style="font-size: 13px; color: #F8FAFC;">{joint_data['recommendation']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_meta2:
        # Gauge Chart for Health Index
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=joint_data["health_index"],
            title={"text": "Joint Health Index (%)", "font": {"size": 14, "color": "#94A3B8"}},
            number={"suffix": "%", "font": {"size": 32, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                "bar": {"color": joint_data["badge_color"], "thickness": 0.25},
                "bgcolor": "#1E293B",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [50, 80], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [80, 100], "color": "rgba(16, 185, 129, 0.2)"}
                ]
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            height=200,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Detailed Telemetry Tiles
    st.markdown("<h3 style='font-size: 16px; margin: 16px 0 12px 0;'>Measured Sensor Telemetry & Physical Indicators</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Contact Resistance (Rc)</div>
            <div class="kpi-value">{joint_data['contact_resistance_mOhm']:.2f} <span style="font-size:16px;">mΩ</span></div>
            <div class="kpi-sub">Threshold: Normal ≤ 2.0 mΩ</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Surface Temperature</div>
            <div class="kpi-value">{joint_data['temperature_C']:.1f} <span style="font-size:16px;">°C</span></div>
            <div class="kpi-sub">Ambient: 25.8 °C (ΔT = {joint_data['temperature_C'] - 25.75:.1f} °C)</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Load Current & Voltage Drop</div>
            <div class="kpi-value">{joint_data['current_A']:.0f} <span style="font-size:16px;">A</span></div>
            <div class="kpi-sub">Drop: {joint_data['voltage_drop_mV']:.2f} mV across contact</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Acoustic Crest Factor</div>
            <div class="kpi-value">{joint_data['acoustic_crest_factor']:.2f}</div>
            <div class="kpi-sub">RMS: {joint_data['acoustic_rms_adc']:.1f} | P2P: {joint_data['acoustic_p2p_adc']}</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 3. SENSOR ANALYTICS PAGE
# ==============================================================================
elif page == "Sensor Analytics":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #F8FAFC; margin: 0 0 4px 0;">Multi-Sensor Engineering Analytics</h1>
        <p style="font-size: 14px; color: #94A3B8; margin: 0;">Interactive exploratory trends for temperature, contact resistance aging, and acoustic burst waveforms.</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["⚡ Contact Resistance Degradation", "🌡️ Thermal Trends", "🔊 Acoustic Waveform & Bursts"])

    with tabs[0]:
        st.markdown("#### Empirical Contact Resistance Escalation under Accelerated Stress")
        st.caption("Extracted from laboratory thermal cycling (200 cycles / 1000 hrs) and damp heat (1416 hrs) aging datasets.")

        fig_deg = px.scatter(
            df_degradation,
            x="exposure_hours",
            y="contact_resistance_mOhm",
            color="connection_condition",
            symbol="stress_type",
            color_discrete_map={
                "Tight connection": STATUS_NORMAL,
                "Intermediate connection": STATUS_WARNING,
                "Loose connection": STATUS_CRITICAL
            },
            hover_data=["specimen_id", "prior_field_age_years", "dielectric_pass"],
            labels={
                "exposure_hours": "Stress Exposure Duration (Hours)",
                "contact_resistance_mOhm": "Contact Resistance (mΩ)",
                "connection_condition": "Connection Health State"
            }
        )
        # Add threshold reference lines
        fig_deg.add_hline(y=2.0, line_dash="dash", line_color=STATUS_NORMAL, annotation_text="Tight Threshold (2.0 mΩ)")
        fig_deg.add_hline(y=10.0, line_dash="dash", line_color=STATUS_CRITICAL, annotation_text="Critical Loose Threshold (10.0 mΩ)")

        fig_deg.update_layout(
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR),
            height=450,
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, type="log")
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    with tabs[1]:
        st.markdown("#### Real Recorded Temperature Rise & Rate of Change (dT/dt)")
        st.caption("Telemetry stream captured from physical BusGuard hardware at 892 ms sampling interval.")

        fig_temp = px.line(
            df_physical,
            x="t_sec",
            y=["temperature_C", "temp_smoothed_C"],
            labels={"t_sec": "Time (Seconds)", "value": "Temperature (°C)"},
            color_discrete_sequence=["#F59E0B", "#38BDF8"]
        )
        fig_temp.update_layout(
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR),
            height=420,
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR)
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with tabs[2]:
        st.markdown("#### Acoustic Dynamic Range & Crest Factor Spike Detection")
        st.caption("Acoustic energy bursts signify mechanical fretting or micro-discharge across degraded contact asperities.")

        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(
            x=df_physical["t_sec"], y=df_physical["ac_peak_to_peak_adc"],
            name="Peak-to-Peak (ADC)",
            line=dict(color="#EF4444", width=1.5)
        ))
        fig_ac.add_trace(go.Scatter(
            x=df_physical["t_sec"], y=df_physical["ac_rms_adc"],
            name="RMS (ADC)",
            line=dict(color=ACCENT_BLUE, width=1.5)
        ))
        fig_ac.update_layout(
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR),
            height=420,
            xaxis=dict(title="Time (Seconds)", gridcolor=GRID_COLOR),
            yaxis=dict(title="Raw ADC Amplitude", gridcolor=GRID_COLOR),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_ac, use_container_width=True)


# ==============================================================================
# 4. AI PREDICTIONS PAGE
# ==============================================================================
elif page == "AI Predictions":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #F8FAFC; margin: 0 0 4px 0;">AI Classification & Model Interpretability</h1>
        <p style="font-size: 14px; color: #94A3B8; margin: 0;">Interactive inference using trained Random Forest Classifier and Isolation Forest Anomaly Detector.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load Model Metadata
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        from ml.train_model import train_random_forest_classifier
        _, metrics = train_random_forest_classifier()

    col_sim, col_res = st.columns([1.1, 1.2])

    with col_sim:
        st.markdown("""
        <div class="glass-panel">
            <div class="panel-title">
                <span>🎛️</span> Joint Telemetry Simulator / Live Inference
            </div>
            <p style="font-size: 12px; color: #94A3B8;">Adjust physical joint parameters to test live machine learning classification.</p>
        """, unsafe_allow_html=True)

        sim_rc = st.slider("Measured Contact Resistance Rc (mΩ)", min_value=0.2, max_value=40.0, value=6.5, step=0.1)
        sim_r0 = st.slider("Baseline Initial Resistance R0 (mΩ)", min_value=0.2, max_value=3.0, value=0.9, step=0.1)
        sim_hours = st.slider("Environmental Stress Exposure (Hours)", min_value=0, max_value=1500, value=360, step=20)
        sim_age = st.slider("Prior Field Age (Years)", min_value=0, max_value=10, value=4, step=1)
        sim_stress = st.selectbox("Accelerated Stress Regime", ["Thermal Cycling", "Damp Heat"])
        is_dh = 1 if sim_stress == "Damp Heat" else 0

        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        # Run Real-Time Prediction
        pred_res = predict_connection_condition(
            contact_resistance_mOhm=sim_rc,
            initial_resistance_mOhm=sim_r0,
            exposure_hours=sim_hours,
            prior_field_age_years=sim_age,
            is_damp_heat=is_dh
        )

        pred_color = STATUS_NORMAL if pred_res["predicted_condition"] == "Tight connection" else (
            STATUS_WARNING if pred_res["predicted_condition"] == "Intermediate connection" else STATUS_CRITICAL
        )

        st.markdown(f"""
        <div class="glass-panel">
            <div class="panel-title">
                <span>🧠</span> Model Prediction Output
            </div>
            <div style="background: #0A0F1D; border-left: 5px solid {pred_color}; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                <div style="font-size: 12px; font-weight: 700; color: #94A3B8;">PREDICTED CONNECTION STATE</div>
                <div style="font-size: 24px; font-weight: 800; color: {pred_color}; margin: 4px 0;">{pred_res['predicted_condition']}</div>
                <div style="font-size: 12px; color: #F8FAFC;">Confidence Score: <b>{pred_res['confidence'] * 100:.1f}%</b></div>
            </div>
            <div style="font-size: 13px; color: #94A3B8; margin-bottom: 16px;">
                <b>Engineering Rationale:</b><br>{pred_res['explanation']}
            </div>
        """, unsafe_allow_html=True)

        # Probability distribution bar
        probs_df = pd.DataFrame([
            {"Condition": k, "Probability": v} for k, v in pred_res["probabilities"].items()
        ])
        fig_prob = px.bar(
            probs_df, x="Probability", y="Condition", orientation="h",
            color="Condition",
            color_discrete_map={
                "Tight connection": STATUS_NORMAL,
                "Intermediate connection": STATUS_WARNING,
                "Loose connection": STATUS_CRITICAL
            },
            range_x=[0, 1]
        )
        fig_prob.update_layout(
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR),
            height=160,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance & Model Validation Metrics
    st.markdown("<h3 style='font-size: 18px; margin: 24px 0 12px 0;'>Model Validation Metrics & Feature Importance</h3>", unsafe_allow_html=True)
    c_m1, c_m2 = st.columns([1.2, 1.0])

    with c_m1:
        st.markdown("""
        <div class="glass-panel">
            <div class="panel-title"><span>📊</span> Random Forest Feature Importance</div>
        """, unsafe_allow_html=True)
        imp_df = pd.DataFrame([
            {"Feature": k.replace("_", " ").title(), "Importance": v}
            for k, v in metrics["feature_importances"].items()
        ]).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color_discrete_sequence=[ACCENT_BLUE]
        )
        fig_imp.update_layout(
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c_m2:
        st.markdown("""
        <div class="glass-panel">
            <div class="panel-title"><span>🎯</span> Anti-Leakage Evaluation Summary</div>
        """, unsafe_allow_html=True)
        train_acc = metrics.get('rf_train_accuracy', metrics.get('train_accuracy', 0.993))
        test_acc = metrics.get('rf_test_accuracy', metrics.get('test_accuracy', 1.0))
        prec_macro = metrics.get('rf_precision_macro', metrics.get('test_precision_macro', 1.0))
        rec_macro = metrics.get('rf_recall_macro', metrics.get('test_recall_macro', 1.0))
        f1_macro = metrics.get('rf_f1_macro', metrics.get('test_f1_macro', 1.0))

        st.markdown(f"""
        <table style="width: 100%; font-size: 13px; color: #F8FAFC; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 8px 0; color: #94A3B8;">Train Specimens / Samples</td><td style="text-align: right; font-weight: 700;">26 specimens ({metrics.get('train_samples', 142)} samples)</td></tr>
            <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 8px 0; color: #94A3B8;">Test Specimens / Samples</td><td style="text-align: right; font-weight: 700;">9 specimens ({metrics.get('test_samples', 48)} samples)</td></tr>
            <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 8px 0; color: #94A3B8;">Training Accuracy</td><td style="text-align: right; font-weight: 700; color: {STATUS_NORMAL};">{train_acc * 100:.1f}%</td></tr>
            <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 8px 0; color: #94A3B8;">Test Set Accuracy</td><td style="text-align: right; font-weight: 700; color: {STATUS_NORMAL};">{test_acc * 100:.1f}%</td></tr>
            <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 8px 0; color: #94A3B8;">Macro Precision</td><td style="text-align: right; font-weight: 700;">{prec_macro * 100:.1f}%</td></tr>
            <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 8px 0; color: #94A3B8;">Macro Recall</td><td style="text-align: right; font-weight: 700;">{rec_macro * 100:.1f}%</td></tr>
            <tr><td style="padding: 8px 0; color: #94A3B8;">Macro F1-Score</td><td style="text-align: right; font-weight: 700; color: {ACCENT_BLUE};">{f1_macro * 100:.1f}%</td></tr>
        </table>
        <div style="font-size: 11px; color: #64748B; margin-top: 12px;">
            *GroupShuffleSplit enforces zero autocorrelation leakage between identical hardware specimens across train and test partitions.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 5. ALERTS & MAINTENANCE PAGE
# ==============================================================================
elif page == "Alerts & Maintenance":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #F8FAFC; margin: 0 0 4px 0;">Alert Log & Prescriptive Maintenance Orders</h1>
        <p style="font-size: 14px; color: #94A3B8; margin: 0;">Prioritized list of active anomaly events and recommended standard operating procedures (SOPs).</p>
    </div>
    """, unsafe_allow_html=True)

    service = get_backend_service()
    alerts = service.get_active_alerts()

    if not alerts:
        st.success("All monitored busbar joints are in NOMINAL healthy condition (Health Index >= 80%). No active alerts.")
    else:
        for alt in alerts:
            sev_color = STATUS_CRITICAL if alt["severity"] == "CRITICAL" else STATUS_WARNING
            ack_badge = (
                '<span style="background: rgba(16,185,129,0.15); color: #10B981; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 10px; margin-left: 8px;">ACKNOWLEDGED</span>'
                if alt["is_acknowledged"]
                else '<span style="background: rgba(239,68,68,0.15); color: #EF4444; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 10px; margin-left: 8px;">PENDING REVIEW</span>'
            )
            st.markdown(f"""
            <div style="background: {CARD_BG}; border: 1px solid #1E293B; border-left: 6px solid {sev_color}; border-radius: 10px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: 800; font-size: 16px; color: #F8FAFC;">{alt['joint_id']} — {alt['joint_name']}</span>
                        <span style="font-size: 12px; color: #64748B; margin-left: 12px;">[{alt['alert_id']}]</span>
                        {ack_badge}
                    </div>
                    <span style="background: rgba({ '239,68,68' if alt['severity']=='CRITICAL' else '245,158,11' }, 0.15); color: {sev_color}; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 11px;">{alt['severity']} ({alt['health_index']}%)</span>
                </div>
                <div style="font-size: 13px; color: #94A3B8; margin-bottom: 8px;">
                    <b>Sensor Evidence:</b> <span style="font-family: 'JetBrains Mono'; color: #F8FAFC;">{alt['evidence']}</span>
                </div>
                <div style="background: #0A0F1D; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: {ACCENT_BLUE}; border: 1px solid #1E293B;">
                    <b>Recommended Maintenance Action:</b> {alt['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c_btn1, c_btn2 = st.columns([1, 4])
            with c_btn1:
                if not alt["is_acknowledged"]:
                    if st.button(f"Acknowledge {alt['joint_id']}", key=f"btn_ack_{alt['alert_id']}"):
                        service.acknowledge_alert(alt['alert_id'])
                        st.rerun()


# ==============================================================================
# 6. REPORTS PAGE
# ==============================================================================
elif page == "Reports":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #F8FAFC; margin: 0 0 4px 0;">Auditing Summaries & Data Export</h1>
        <p style="font-size: 14px; color: #94A3B8; margin: 0;">Export processed monitoring logs, empirical degradation tables, and generate formal inspection tickets.</p>
    </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("""
        <div class="glass-panel">
            <div class="panel-title"><span>📥</span> Export Cleaned & Processed Datasets</div>
            <p style="font-size: 13px; color: #94A3B8;">Download the preprocessed datasets generated by the BusGuard AI data pipeline.</p>
        """, unsafe_allow_html=True)

        st.download_button(
            label="Download Monitored Joints Telemetry (CSV)",
            data=df_joints.to_csv(index=False).encode("utf-8"),
            file_name="busguard_monitored_joints.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="Download Physical Multi-Sensor Stream (CSV)",
            data=df_physical.to_csv(index=False).encode("utf-8"),
            file_name="busguard_physical_stream_processed.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="Download Connector Degradation Benchmark (CSV)",
            data=df_degradation.to_csv(index=False).encode("utf-8"),
            file_name="connector_degradation_processed.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r2:
        st.markdown("""
        <div class="glass-panel">
            <div class="panel-title"><span>📋</span> Generate Inspection Work Order</div>
            <p style="font-size: 13px; color: #94A3B8;">Generate a printable field maintenance dispatch ticket for electrical maintenance personnel.</p>
        """, unsafe_allow_html=True)

        service = get_backend_service()
        work_order_text = service.generate_work_order_text()
        st.text_area("WORK ORDER PREVIEW", work_order_text, height=180)
        st.download_button(
            label="Download Formal Maintenance Work Order (TXT)",
            data=work_order_text.encode("utf-8"),
            file_name="busguard_maintenance_work_order.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 7. ABOUT SYSTEM PAGE
# ==============================================================================
elif page == "About System":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #F8FAFC; margin: 0 0 4px 0;">System Architecture & Interface Specification</h1>
        <p style="font-size: 14px; color: #94A3B8; margin: 0;">Theoretical foundations, data integration architecture, and ESP32 hardware interface specification.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-panel">
        <div class="panel-title"><span>🔬</span> Physical & Theoretical Principles</div>
        <p style="font-size: 13px; color: #94A3B8; line-height: 1.6;">
            <b>Holm's Electrical Contact Theory:</b> Bolted electrical joints conduct current through microscopic asperities (a-spots). 
            When bolt preload is lost due to vibration or thermal cycling, the true contact area diminishes, causing contact resistance ($R_c$) to escalate. 
            Joule heating ($P = I^2 R_c$) accelerates copper oxidation, triggering a destructive thermal runaway loop.
        </p>
        <p style="font-size: 13px; color: #94A3B8; line-height: 1.6;">
            <b>Multi-Sensor Fusion:</b> BusGuard AI achieves superior robustness by correlating thermal rise (&Delta;T) with acoustic emissions (transient micro-arcing and mechanical fretting chatter), detecting loose joints long before catastrophic failure.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-panel">
        <div class="panel-title"><span>🔌</span> Documented ESP32 Hardware Integration Interface</div>
        <p style="font-size: 13px; color: #94A3B8;">
            A separate hardware integration team can stream real-time sensor packets from ESP32 edge microcontrollers using the following JSON payload schema:
        </p>
    """, unsafe_allow_html=True)

    json_spec = """{
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
}"""
    st.code(json_spec, language="json")
    st.caption("Standard REST endpoint: POST /api/v1/telemetry | Baud rate for Serial CDC: 115200 bps | ADC resolution: 12-bit (0-4095)")
    st.markdown("</div>", unsafe_allow_html=True)
