# ParkSight BLR — AI-Driven Parking Congestion Intelligence

> **Flipkart Gridlock Hackathon 2.0 · Round 2 (Prototype)**  
> Problem: *Poor Visibility on Parking-Induced Congestion*

Patrol-based parking enforcement in Bengaluru is reactive and blind: enforcement teams know violations happen, but they do not have a reliable way to decide **which junction to police first, at what hour, and why that spot matters for traffic flow**.

**ParkSight** turns **298,431 real Bengaluru Traffic Police parking-violation records** into an operational decision-support dashboard that:

1. **Detects chronic illegal-parking hotspots** using geospatial clustering.
2. **Forecasts hour-wise enforcement demand** using a supervised ML model.
3. **Estimates traffic-flow obstruction risk** using an interpretable congestion-risk proxy.
4. **Ranks patrol zones by predicted enforcement value** for the selected hour.
5. **Schedules limited enforcement units** using ParkSched, an MLFQ-inspired priority scheduler with aging.

![Dashboard](shot_top.png)

---

## What is real data vs modelled

The violation locations, timestamps, vehicle types, police stations, junctions and violation counts come from the provided anonymized BTP CSV. The traffic-flow impact values are not measured speed loss because the dataset does not include speed, queue length or lane occupancy. They are transparent risk proxies that can be calibrated later with CCTV/traffic-speed feeds.

---

## Why this wins the brief

| Judging axis | How ParkSight delivers |
|---|---|
| **Robustness** | Built on 298K real BTP records from Nov 2023-Apr 2024. Rejected and duplicate records are validation-discounted. |
| **ML / AI** | Uses DBSCAN for geospatial hotspot discovery and a supervised `RandomForestRegressor` to forecast future zone-hour parking pressure. |
| **Innovation** | Combines historical hotspot intelligence, hour-wise ML prediction, traffic-flow obstruction-risk scoring and an MLFQ-inspired patrol scheduler. |
| **Clarity** | Every zone explains the reason: violation mix, vehicle type, peak patrol window, recurrence and impact evidence. |
| **Operational value** | Output is directly actionable: which zone, what hour, what offence, what vehicle type and what enforcement action. |

### Headline insight

**84.1% of detected CIS-weighted hotspot impact is concentrated in the top 20 enforcement zones.**
This means enforcement does not only need more patrols; it needs better targeting. KR Market Junction ranks first by flow-impact risk and peaks around **08:00-11:00**.

---

## Layer 1: Supervised ML hotspot forecast

The stronger ML layer trains a supervised model at **zone × hour** level.

```text
Features from earlier dates  ->  future parking pressure for that zone-hour
```

Model used:

```text
RandomForestRegressor
```

Features include:

- historical weighted violations per day
- raw historical violations per day
- active-day ratio
- hour cyclic features
- peak-hour flag
- average severity
- average obstruction score
- junction share
- CIRS / flow-impact score
- recurrence
- zone size and location

Backtest design:

```text
Training sample:    first 50% dates -> next 20% dates
Validation sample:  first 70% dates -> last 30% dates
```

Current holdout result:

```text
ML recall@20:      85.0%
ML coverage@20:    78.6%
MAE:               0.0863 violations per zone-hour-day
RMSE:              0.3914 violations per zone-hour-day
```

The dashboard uses the model's hourly predictions to re-rank zones when the deployment slider is moved.

---

## Layer 4: Traffic-flow obstruction risk

Since the CSV does not contain traffic speeds or queue lengths, ParkSight does not claim exact measured delay. Instead, it estimates obstruction risk from factors available in the violation data.

The score combines:

```text
Traffic-flow risk = CIRS + obstruction type + peak timing + junction spillback risk
```

Examples:

| Violation type | Why it matters |
|---|---|
| Parking near road crossing | Can block turning traffic and cause intersection spillback. |
| Parking near traffic light/zebra crossing | Can reduce signal discharge and pedestrian safety. |
| Parking in main road | Direct carriageway obstruction. |
| Double parking | Effective lane loss. |
| Footpath parking | Lower direct vehicle-flow impact but higher pedestrian risk. |

---

## Layer 5: ParkSched patrol scheduler

ParkSight now converts predictions into a deployable patrol plan. It treats:

| Operating-system concept | ParkSight equivalent |
|---|---|
| CPU | Patrol / towing / e-challan unit |
| Process | Hotspot enforcement zone |
| Priority | ML forecast + flow-impact risk + recurrence + confidence |
| Time quantum | 30-minute patrol window |
| Aging | Waiting zones receive a boost so they are not starved |

The scheduler uses an MLFQ-inspired policy:

```text
Q0 Critical  -> severe/high-need zones, tow + e-challan first
Q1 High      -> patrol/challan in current or next slot
Q2 Watch     -> round-robin rotation if units remain
Q3 Low       -> passive monitoring
```

Need score:

```text
Need = 0.35 × ML forecast
     + 0.25 × flow-impact risk
     + 0.15 × recurrence
     + 0.15 × ML confidence
     + 0.10 × lane-blockage risk
```

The dashboard lets the user choose the number of available enforcement units and number of 30-minute slots. ParkSched then outputs assignments like:

```text
10:00-10:30
Unit 1 -> KR Market Junction
Unit 2 -> Safina Plaza Junction
Unit 3 -> Hosahalli Metro Station
```

This turns ParkSight from a map into a control-room scheduling tool.

---

## Run it

```bash
pip install -r requirements.txt
python3 pipeline/build_data.py
python3 -m http.server 8765 -d web
```

Then open:

```text
http://localhost:8765
```

On Windows PowerShell, use:

```powershell
py -m pip install -r requirements.txt
py pipeline\build_data.py
py -m http.server 8765 -d web
```

---

## Architecture

```text
raw CSV (298K rows)
      │
      │  pipeline/build_data.py
      │  pandas + DBSCAN + RandomForestRegressor
      ▼
web/data/*.json
summary · hotspots · zones · temporal · breakdowns
      │
      ▼
web dashboard
Leaflet heatmap + zone ranking + ML deployment slider + ParkSched patrol scheduler + Chart.js analytics
```

| File | Role |
|---|---|
| `pipeline/build_data.py` | Clean -> grid-bin -> CIRS -> DBSCAN zones -> ML forecast -> flow-risk scoring -> scheduler metadata -> JSON. |
| `web/app.js` | Loads JSON, ranks zones by ML forecast, builds ParkSched patrol schedule, renders map/list/modals/charts. |
| `web/index.html / styles.css` | Dashboard layout and styling. |
| `web/data/*.json` | Precomputed analytics artifacts. |

---

## 90-second demo script

1. **Open dashboard**: "We use 298K real BTP parking-violation records."
2. **Show heatmap**: "This is not just volume; heat is weighted by traffic-flow obstruction risk."
3. **Show ML panel**: "Layer 1 predicts which zones need enforcement at the selected hour."
4. **Drag slider to 10:00**: "The ranking changes using ML-predicted zone-hour demand."
5. **Show ParkSched**: "With five units, the scheduler creates a 30-minute patrol plan using MLFQ priority and aging."
6. **Click KR Market Junction**: "High violations, peak 08:00-11:00, severe flow risk, specific enforcement action."
7. **Be honest on impact**: "The flow score is a calibrated risk proxy, not measured speed loss. In production, speed/CCTV feeds calibrate actual delay."

---

*Prototype built for Flipkart Gridlock Hackathon 2.0. Data: anonymized Bengaluru Traffic Police parking-violation records.*

## Layer 6 — AI Early-Warning Detector

The final prototype adds a second real ML layer beyond the RandomForest forecast. `pipeline/build_data.py` trains an `IsolationForest` anomaly detector on the historical baseline zone-hour profile and scores the most recent window against it.

This produces `web/data/ai_alerts.json`, which powers the **Layer 6 · AI early-warning detector** panel in the dashboard. These alerts are not hardcoded. They are generated when a zone-hour shows an abnormal recent surge relative to its own baseline, then weighted with flow-impact risk so enforcement can pre-position units before the surge becomes a chronic choke.

Honest limitation: this is still based on the provided historical violation stream, not CCTV/live sensors. It is a real ML early-warning layer for historical/recent data, and can be connected to live violation feeds in production.
