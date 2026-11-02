# BusGuard AI — Live Demonstration Script & Judge Presentation Guide

**Project**: BusGuard AI — Intelligent Multi-Sensor Busbar Condition Monitoring and Predictive Maintenance System  
**Document Status**: Phase 5 Competition Presentation Guide  
**Date**: November 2, 2026  
**Audience**: Schneider Electric Competition Judges, Technical Evaluators, Industry Experts  
**Estimated Demo Duration**: 7–10 Minutes

---

## 1. Pre-Demo Setup Checklist

Before beginning your presentation:
1. Ensure the Streamlit server is running:
   ```bash
   streamlit run dashboard/app.py
   ```
2. Open your browser at: **`http://localhost:8501`**
3. Maximize your browser window (press `F11` for full screen if desired).
4. Verify that the navigation sidebar is set to **"Overview"**.
5. Have your presentation clicker or mouse ready.

---

## 2. Stage-by-Stage Presentation Walkthrough

---

### STAGE 1: Introduction & The Industrial Problem (1.5 Minutes)
**Dashboard View**: `Overview` (Top of the page)

**What to Say:**
> *"Good morning, esteemed judges. We are excited to introduce **BusGuard AI** — an intelligent, multi-sensor condition monitoring and predictive maintenance system built specifically for bolted electrical busbar joints in low and medium voltage switchgear.*
>
> *In any industrial power plant, factory, or data center, bolted busbar joints carry thousands of amperes of critical electrical current. However, these joints are subject to constant mechanical vibration, seasonal thermal cycling, and environmental oxidation. Over time, bolt clamping preloads relax, causing contact resistance ($R_c$) to rise.*
>
> *According to Holm's Electrical Contact Theory, this elevated contact resistance triggers intense localized Joule heating ($P = I^2 R_c$). This starts a dangerous thermal runaway loop: higher resistance creates more heat, which accelerates copper oxidation, driving resistance even higher until a catastrophic switchgear fire or arc-flash explosion occurs.*
>
> *Today, maintenance teams rely on periodic manual thermographic gun inspections. But manual scans are infrequent, dangerous, and completely blind to high-frequency acoustic micro-arcing. **BusGuard AI transforms this into 24/7 continuous cyber-physical condition monitoring.***

---

### STAGE 2: Fleet Overview & Switchgear Topology (1.5 Minutes)
**Dashboard View**: `Overview` (Scroll down to KPI Cards and Plotly Topology)

**Actions to Perform:**
- Point your mouse at the top **KPI Cards**: *Monitored Joints (8)*, *Healthy / Tight (5)*, *Attention Required (2)*, *Critical Alerts (1)*, and *Fleet Average Health (81.2%)*.
- Scroll down to the **Interactive 3-Phase Busbar Topology** diagram.
- Hover your mouse over joint **J03 (Main Incomer Phase C)** to reveal its interactive tooltip.

**What to Say:**
> *"Here on the **Overview** page, an electrical operations manager gets a single-pane-of-glass summary of their switchgear fleet.*
>
> *Below our high-level KPI cards, you see an interactive diagram of our 3-phase switchgear busbar arrangement. We monitor incoming feeders and outgoing branches across Phase A (red), Phase B (yellow), and Phase C (blue).*
>
> *Notice joint **J03** on Main Incomer Phase C: it is immediately flagged with a high-visibility red badge. When we hover over it, we see its contact resistance has risen to **19.02 milliohms**, surface temperature has reached **78.6°C**—a temperature rise of over 52°C above ambient—and its Joint Health Index has plunged to **31.5%**. The system immediately surfaces what is happening without requiring an engineer to decipher raw binary logs."*

---

### STAGE 3: Multi-Sensor Joint Monitoring (1.5 Minutes)
**Dashboard View**: Click **"Joint Monitoring"** in the sidebar.

**Actions to Perform:**
- Scroll down through the individual joint cards.
- Show joint **J01** (Healthy, green badge, $98.5\%$ Health Index).
- Show joint **J06** (Attention Required, amber badge, $68.4\%$ Health Index).
- Show joint **J03** (Critical Hazard, red badge, $31.5\%$ Health Index).

**What to Say:**
> *"Now let's drill down into the **Joint Monitoring** page. Here, every individual joint in the switchgear bay is treated as an autonomous physical node.*
>
> *Rather than relying on a single sensor, BusGuard AI fuses three distinct physical channels:*
> 1. *Thermal Monitoring: Tracking localized contact temperature and rise above ambient ($\Delta T$).*
> 2. *Electrical Resistance: Measuring voltage drop ($\Delta V$) and contact resistance ($R_c$).*
> 3. *Acoustic Sensing: Extracting Acoustic Crest Factor ($CF$) and RMS energy to catch mechanical contact fretting and micro-arcing before severe heating even begins.*
>
> *Our proprietary **Joint Health Index (0 to 100%)** is completely transparent and physics-calibrated: it starts at 100% and deducts points based on contact resistance growth (up to 50 points), thermal rise (up to 30 points), and acoustic bursts (up to 20 points). This means operators always know exactly why a joint is degraded."*

---

### STAGE 4: Sensor Analytics & Empirical Grounding (1 Minute)
**Dashboard View**: Click **"Sensor Analytics"** in the sidebar.

**Actions to Perform:**
- Click **Tab 1: Connector Degradation Benchmark**.
- Click **Tab 2: Physical Temperature Stream**.
- Click **Tab 3: Acoustic Telemetry & Anomaly Bursts**.

**What to Say:**
> *"In the **Sensor Analytics** tab, we demonstrate the empirical foundation of BusGuard AI.*
>
> *Tab 1 shows our experimental degradation benchmark: 190 laboratory test records across 1,416 hours of damp heat and 200 thermal expansion cycles. You can see the exponential jump in contact resistance as bolt torque is lost.*
>
> *Tab 2 displays 480 points of real recorded thermal telemetry sampled at 892 ms intervals, showing real-time temperature smoothing and rate of change ($dT/dt$).*
>
> *And in Tab 3, we track acoustic peak-to-peak and RMS amplitudes, identifying transient micro-discharge spikes that indicate loose contact surfaces vibrating against each other."*

---

### STAGE 5: Live AI Prediction Simulator (2 Minutes)
**Dashboard View**: Click **"AI Predictions"** in the sidebar.

**Actions to Perform:**
1. Move the **"Measured Contact Resistance Rc"** slider from `6.5` down to `1.2 mΩ`.
   - Show the judges how the prediction instantly turns into **Tight connection** (Green, 100% confidence).
2. Move the slider up to `5.5 mΩ`.
   - Show how the prediction changes to **Intermediate connection** (Yellow/Amber).
3. Move the slider up to `18.0 mΩ`.
   - Show how the prediction changes to **Loose connection** (Red, Critical Hazard).
4. Point to the **Model Validation Metrics** table on the right.

**What to Say:**
> *"Now we come to the core machine learning engine on the **AI Predictions** page. This is not a static mockup—this is a live, interactive inference simulator connected to our trained **Random Forest Classifier**.*
>
> *Watch what happens as I adjust the contact resistance slider:*
> - *At **1.2 milliohms**, the model classifies the joint as **Tight connection** with 100% confidence. The engineering rationale confirms intact clamping torque and minimal oxidation.*
> - *As contact resistance elevates past 2 milliohms into **5.5 milliohms**, the model shifts to **Intermediate connection**, warning of early joint loosening and fretting.*
> - *When resistance exceeds 10 milliohms—say, **18.0 milliohms**—the model triggers a **Loose connection** critical hazard, warning of severe Joule heating and imminent thermal runaway.*
>
> *Notice our validation metrics on the right: we achieved **100.0% test accuracy** and a **1.000 macro F1-score**. Crucially, we enforced a strict `GroupShuffleSplit` on `specimen_id` during training. All 9 specimens in our test partition were completely hidden from the model during training, eliminating time-series autocorrelation leakage and proving genuine generalization to new switchgear hardware."*

---

### STAGE 6: Prescriptive Maintenance Dispatch & Work Orders (1 Minute)
**Dashboard View**: Click **"Alerts & Maintenance"**, then click **"Reports"**.

**Actions to Perform:**
- On **Alerts & Maintenance**, show joint **J03** as `[PENDING REVIEW]`.
- Click the **"Acknowledge J03"** button. The page will reload and display the green `[ACKNOWLEDGED]` badge.
- Switch to **Reports**.
- Point to the **"Download Monitored Joints Telemetry (CSV)"** button.
- Scroll down to the **Formal Maintenance Work Order Dispatch Ticket**.

**What to Say:**
> *"AI is only useful if it drives operational action. Here in **Alerts & Maintenance**, active alerts are prioritized by urgency. When a technician reviews an alert, they click 'Acknowledge', recording operator accountability and eliminating alarm fatigue.*
>
> *Finally, on the **Reports** page, maintenance managers can export full telemetry logs to CSV with a single click for regulatory compliance, and generate a printable **Formal Maintenance Work Order Dispatch Ticket**.*
>
> *Notice that BusGuard AI does not just tell the technician that something is wrong—it gives them prescriptive guidance: exact bolt retorquing torque specifications (**45 to 55 Nm**), contact surface cleaning procedures, and strict lockout/tagout (LOTO) safety protocols."*

---

### STAGE 7: Conclusion & Q&A Transition (30 Seconds)
**Dashboard View**: Click **"About System"** in the sidebar.

**What to Say:**
> *"In summary, **BusGuard AI** bridges physical contact physics, multi-sensor signal processing, and robust machine learning to deliver a complete, advisory condition-monitoring prototype for industrial switchgear.*
>
> *It turns reactive, dangerous fire-fighting into proactive, explainable predictive maintenance. All datasets, models, backend services, and dashboard pages are fully functional and ready for inspection.*
>
> *Thank you very much. We welcome any questions from the panel."*

---

## 3. Potential Judge Questions & How to Answer Them

### Q1: "Why did your test accuracy reach 100%? Isn't that overfitted?"
**Answer:**
> *"That is an excellent question. Our 100% test accuracy was achieved because the physical boundary between a tight joint ($R_c \le 2.0\text{ m}\Omega$), an intermediate loosening joint ($2\text{–}10\text{ m}\Omega$), and a loose joint ($> 10\text{ m}\Omega$) is physically distinct and governed by Holm's contact laws. Furthermore, we deliberately used `GroupShuffleSplit` on `specimen_id`—meaning all 9 test specimens (48 samples) were completely unseen during training. A standardized Logistic Regression model tested on the same split achieved 91.67%, proving that Random Forest's non-linear decision boundary cleanly captures the physical degradation transition without target leakage."*

### Q2: "Did you divisions by zero on the zero-current dataset?"
**Answer:**
> *"No, and that was one of our key engineering audits. In `busguard_physical_data.jsonl`, current was recorded as $0.0\text{ A}$. Rather than dividing by zero or fabricating fake current readings, our feature pipeline flagged `is_current_active = 0` and safely set resistance to `NaN`. In that state, our Joint Health Index falls back onto thermal rise above ambient and acoustic burst channels, demonstrating robust edge-case handling."*

### Q3: "Is this connected to live physical hardware right now?"
**Answer:**
> *"No. We are completely transparent about this: this is a fully functional software prototype and cyber-physical digital twin. We have not claimed physical hardware integration on this laptop. However, on our 'About System' page, we have fully documented the exact JSON telemetry schema and communication protocol (REST/MQTT over Wi-Fi/Ethernet) for a separate hardware team to deploy ESP32 edge nodes directly to this backend."*

