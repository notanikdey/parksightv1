/* ParkSight BLR — dashboard logic */
const DATA = {};
const FILES = ["summary", "hotspots", "zones", "temporal", "breakdowns", "ai_alerts"];
let map, heatLayer, zoneLayer, zoneMarkers = {}, activeHour = -1, activeZoneId = null;
let patrolUnits = 5, patrolSlots = 3, patrolQuantum = 30;

const WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/* ---------- utils ---------- */
const $ = (s) => document.querySelector(s);
const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : "" + n;
const pad = (h) => String(h).padStart(2, "0");
const hourLabel = (h) => `${pad(h)}:00`;

/* CIS -> color on a cool→hot gradient */
function cisColor(v) {
  const stops = [
    [0, [43, 111, 255]], [40, [55, 214, 179]],
    [62, [255, 208, 0]], [80, [255, 138, 61]], [100, [255, 59, 87]],
  ];
  v = Math.max(0, Math.min(100, v));
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i];
      const t = (v - a) / (b - a);
      const c = ca.map((x, j) => Math.round(x + (cb[j] - x) * t));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(255,59,87)";
}

/* group sorted hours into ranges -> "09:00–12:00, 17:00–18:00" */
function windowText(hours) {
  if (!hours || !hours.length) return "—";
  const s = [...hours].sort((a, b) => a - b);
  const out = []; let st = s[0], pv = s[0];
  for (let i = 1; i < s.length; i++) {
    if (s[i] === pv + 1) { pv = s[i]; }
    else { out.push([st, pv]); st = pv = s[i]; }
  }
  out.push([st, pv]);
  return out.map(([a, b]) => `${pad(a)}:00–${pad(b + 1)}:00`).join(", ");
}

/* ---------- load ---------- */
async function boot() {
  await Promise.all(FILES.map(async (f) => {
    try {
      const res = await fetch(`data/${f}.json`, { cache: "no-store" });
      if (!res.ok) throw new Error(`${f}.json not found`);
      DATA[f] = await res.json();
    } catch (err) {
      console.warn(`Skipping optional data file: ${f}.json`, err);
      if (f === "ai_alerts") DATA[f] = { alerts: [], summary: { enabled: false } };
      else throw err;
    }
  }));
  renderHeader();
  renderKPIs();
  initMap();
  renderZoneList(DATA.zones.zones);
  renderForecastPanel();
  renderImpactBoard();
  renderSchedulerPanel();
  renderAIAlertsPanel();
  renderTemporal();
  renderCharts();
  wireControls();

  // Deep-link: ?zone=<rank> auto-opens a zone (handy for live demos).
  const rank = +new URLSearchParams(location.search).get("zone");
  if (rank) {
    const z = DATA.zones.zones.find((x) => x.rank === rank);
    if (z) openZone(z.id, true);
  }
}

/* ---------- header + kpis ---------- */
function renderHeader() {
  const s = DATA.summary;
  $("#date-range").textContent = `${s.date_start} → ${s.date_end}`;
  $("#row-count").textContent = `${s.generated_from_rows.toLocaleString()} records`;
  $("#foot-rows").textContent = s.generated_from_rows.toLocaleString();
}

function renderKPIs() {
  const s = DATA.summary;
  const backtest = s.ml_prediction_backtest || s.prediction_backtest || {};
  const cards = [
    { v: fmt(s.generated_from_rows), l: "Parking records", spark: `${s.span_days} days of BTP data`, cls: "" },
    { v: s.n_zones, l: "Enforcement zones", spark: `${s.n_hot_cells.toLocaleString()} scored grid cells`, cls: "" },
    { v: (backtest.recall_at_k_pct ?? "—") + "<small>%</small>", l: "ML recall@20", spark: "future-window backtest", cls: "", sparkCls: "up" },
    { v: hourLabel(s.peak_hour), l: "Peak hour", spark: `${s.peak_share_pct}% in rush windows`, cls: "", sparkCls: "warn" },
    { v: `${s.ai_alert_count || 0}`, l: "AI surge alerts", spark: s.ai_early_warning?.enabled ? "IsolationForest" : "no alerts", cls: "", sparkCls: "danger" },
  ];
  $("#kpi-strip").innerHTML = cards.map((c) => `
    <div class="kpi ${c.cls}">
      <div class="k-val">${c.v}</div>
      <div class="k-lab">${c.l}</div>
      <div class="k-spark ${c.sparkCls || ""}">${c.spark}</div>
    </div>`).join("");
}

/* ---------- map ---------- */
function initMap() {
  const s = DATA.summary;
  map = L.map("map", { zoomControl: true, attributionControl: false }).setView(s.city_center, 12);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19, subdomains: "abcd",
  }).addTo(map);

  heatLayer = L.heatLayer(DATA.hotspots.points, {
    radius: 18, blur: 22, maxZoom: 15, max: 1.0,
    gradient: { 0.0: "#2b6fff", 0.35: "#37d6b3", 0.6: "#ffd000", 0.8: "#ff8a3d", 1.0: "#ff3b57" },
  }).addTo(map);

  zoneLayer = L.layerGroup();
  DATA.zones.zones.forEach((z) => {
    const m = L.circleMarker([z.lat, z.lon], markerStyle(z, z.cis))
      .bindPopup(popupHTML(z))
      .on("click", () => openZone(z.id, false));
    zoneMarkers[z.id] = m;
    zoneLayer.addLayer(m);
  });
}

function markerStyle(z, intensity) {
  const val = intensity ?? z.flow_impact_score ?? z.cis;
  return {
    radius: 6 + Math.sqrt(val) * 1.9,
    fillColor: cisColor(z.flow_impact_score || z.cis),
    color: "#fff", weight: 1.4, opacity: .9, fillOpacity: .82, className: "zmark",
  };
}

function popupHTML(z) {
  return `<div class="pop-name">#${z.rank} · ${z.name}</div>
    <div class="pop-meta">${z.violations.toLocaleString()} violations · ${z.top_violation}</div>
    <div class="pop-cis" style="color:${cisColor(z.flow_impact_score || z.cis)}">Flow risk ${z.flow_impact_score || z.cis} · CIRS ${z.cis}</div>`;
}

function setLayer(layer) {
  document.querySelectorAll("#layer-toggle button").forEach((b) =>
    b.classList.toggle("active", b.dataset.layer === layer));
  if (layer === "heat") { map.addLayer(heatLayer); map.removeLayer(zoneLayer); }
  else { map.removeLayer(heatLayer); map.addLayer(zoneLayer); }
}

/* ---------- enforcement list ---------- */
function recommendation(z) {
  const win = windowText(z.forecast_peak_hours || z.peak_hours);
  const veh = z.top_vehicle ? z.top_vehicle.toLowerCase() : "vehicles";
  const band = z.flow_impact_band || "High";
  const evidence = (z.impact_evidence || []).join(", ");
  return `Deploy a towing + e-challan unit during <b>${win}</b>. Dominant offence: <b>${z.top_violation}</b> by <b>${veh}</b>. Flow-impact risk is <b>${band}</b>${evidence ? ` (${evidence})` : ""}. Recurs on ${Math.round(z.recurrence * 100)}% of days — a chronic choke, not a one-off.`;
}

function forecastScore(z, h) {
  if (h < 0) return z.flow_impact_score || z.cis;
  if (z.ml_forecast_scores && z.ml_forecast_scores[h] != null) return z.ml_forecast_scores[h];
  return (z.forecast_scores && z.forecast_scores[h] != null) ? z.forecast_scores[h] : (z.hourly[h] || 0);
}
function expectedHourly(z, h) {
  if (z.ml_expected_hourly && z.ml_expected_hourly[h] != null) return z.ml_expected_hourly[h];
  return (z.expected_hourly && z.expected_hourly[h] != null) ? z.expected_hourly[h] : 0;
}

function renderZoneList(zones) {
  const hourMode = activeHour >= 0;
  let list = zones.slice();
  if (hourMode) {
    list = list.filter((z) => forecastScore(z, activeHour) > 0)
      .sort((a, b) => forecastScore(b, activeHour) - forecastScore(a, activeHour));
  }
  $("#zone-count").textContent = hourMode
    ? `${list.length} zones forecast at ${hourLabel(activeHour)} · ranked by predicted patrol value`
    : `${zones.length} zones ranked by traffic-flow impact risk`;

  $("#zone-list").innerHTML = list.map((z, i) => {
    const metric = hourMode ? forecastScore(z, activeHour) : (z.flow_impact_score || z.cis);
    const col = cisColor(z.flow_impact_score || z.cis);
    const rkBg = hourMode ? "var(--card)" : col + "22";
    const expected = hourMode ? ` · ${expectedHourly(z, activeHour).toFixed(2)} exp/hr/day` : "";
    const aiBadge = z.ai_alert_level && z.ai_alert_level !== "Normal" ? ` · AI ${z.ai_alert_level}` : "";
    return `<div class="zone ${z.id === activeZoneId ? "active" : ""}" data-id="${z.id}">
      <div class="rk" style="background:${rkBg};color:${col}">${hourMode ? i + 1 : z.rank}</div>
      <div class="zinfo">
        <div class="zname">${z.name}</div>
        <div class="zmeta">${z.flow_impact_band || "Risk"} impact · ${z.top_violation} · ${z.top_vehicle}${expected}${aiBadge}</div>
      </div>
      <div class="cis-wrap">
        <div class="cis-num" style="color:${col}">${hourMode ? Math.round(metric) : metric}</div>
        <div class="cis-lab">${hourMode ? "PRED" : "FLOW"}</div>
      </div>
    </div>`;
  }).join("");

  document.querySelectorAll(".zone").forEach((el) =>
    el.addEventListener("click", () => openZone(+el.dataset.id, true)));
}

function openZone(id, fly) {
  activeZoneId = id;
  const z = DATA.zones.zones.find((x) => x.id === id);
  if (!z) return;
  if (fly) { setLayer("zones"); map.flyTo([z.lat, z.lon], 15, { duration: .8 }); }
  zoneMarkers[id].openPopup();
  renderZoneList(DATA.zones.zones);
  document.querySelector(`.zone[data-id="${id}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  showModal(z);
}

/* ---------- modal ---------- */
function ring(cis) {
  const r = 40, c = 2 * Math.PI * r, off = c * (1 - cis / 100);
  return `<svg class="ring" viewBox="0 0 96 96">
    <circle cx="48" cy="48" r="${r}" fill="none" stroke="#243154" stroke-width="9"/>
    <circle cx="48" cy="48" r="${r}" fill="none" stroke="${cisColor(cis)}" stroke-width="9"
      stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${off}"
      transform="rotate(-90 48 48)"/>
    <text x="48" y="52" text-anchor="middle" fill="#e8edf9" font-size="22" font-weight="800" font-family="JetBrains Mono">${cis}</text>
  </svg>`;
}

function showModal(z) {
  const maxMix = Math.max(...z.violation_mix.map((m) => m.n), 1);
  const mix = z.violation_mix.map((m) =>
    `<div class="mix-row"><span class="mix-lab">${m.label}</span>
      <div class="mix-bar" style="width:${20 + 120 * m.n / maxMix}px;background:${cisColor(z.cis)}"></div>
      <span class="mix-n">${m.n.toLocaleString()}</span></div>`).join("");

  $("#modal-body").innerHTML = `
    <div class="m-head">
      <div class="m-rank">PRIORITY RANK #${z.rank} OF ${DATA.zones.zones.length}</div>
      <div class="m-name">${z.name}</div>
      <div class="m-loc">${z.locality || ""}${z.station ? " · " + z.station + " PS" : ""}</div>
      <button class="m-close" onclick="closeModal()">✕</button>
    </div>
    <div class="m-body">
      <div class="m-cis">
        ${ring(z.cis)}
        <div><div style="font-weight:800;font-size:15px">Congestion Impact Risk Score</div>
        <div class="ring-lab">Composite of density, road-criticality, peak concentration,<br>junction proximity & day-to-day recurrence.</div></div>
      </div>
      <div class="m-stats">
        <div class="m-stat"><div class="v">${z.violations.toLocaleString()}</div><div class="l">Total violations</div></div>
        <div class="m-stat"><div class="v">${windowText(z.ml_forecast_peak_hours || z.forecast_peak_hours || z.peak_hours)}</div><div class="l">ML patrol window</div></div>
        <div class="m-stat"><div class="v">${z.flow_impact_score || z.cis}</div><div class="l">Traffic-flow impact risk</div></div>
        <div class="m-stat"><div class="v">${z.flow_impact_band || "—"}</div><div class="l">Impact band</div></div>
        <div class="m-stat"><div class="v">${z.lane_blockage_risk_pct || "—"}%</div><div class="l">Lane-blockage risk proxy</div></div>
        <div class="m-stat"><div class="v">${z.ml_prediction_confidence || z.prediction_confidence || "—"}%</div><div class="l">ML prediction confidence</div></div>
      </div>
      <div class="m-rec">
        <div class="r-kick">RECOMMENDED ENFORCEMENT ACTION</div>
        <div class="r-txt">${recommendation(z)}</div>
      </div>
      <div class="m-rec ai-modal-note">
        <div class="r-kick">AI EARLY WARNING</div>
        <div class="r-txt">${z.ai_alert_level && z.ai_alert_level !== "Normal" ? `${z.ai_alert_level} surge at ${windowText(z.ai_alert_hours || [])}: ${z.ai_alert_reason}` : "No abnormal recent surge detected for this zone-hour profile."}</div>
      </div>
      <div class="m-mix">
        <div class="sub" style="margin-bottom:8px">Violation composition</div>${mix}
      </div>
    </div>`;
  $("#modal").classList.add("show");
}
function closeModal() { $("#modal").classList.remove("show"); }


/* ---------- decision panels ---------- */
function renderForecastPanel() {
  const h = activeHour >= 0 ? activeHour : DATA.summary.peak_hour;
  const zones = DATA.zones.zones.slice()
    .sort((a, b) => forecastScore(b, h) - forecastScore(a, h))
    .slice(0, 5);
  $("#forecast-sub").textContent = `Top predicted enforcement zones for ${hourLabel(h)}–${hourLabel((h + 1) % 24)}`;
  $("#forecast-board").innerHTML = `
    <div class="decision-note">Supervised <b>${DATA.summary.ml_prediction_backtest?.model || "ML"}</b> forecast trained on zone×day×hour demand. Rankings combine predicted pressure with traffic-flow risk.</div>
    ${zones.map((z, i) => `
      <div class="forecast-row" data-id="${z.id}">
        <div class="forecast-rank">${i + 1}</div>
        <div class="forecast-main">
          <div class="forecast-name">${z.name}</div>
          <div class="forecast-meta">${expectedHourly(z, h).toFixed(2)} expected viol/hr/day · ${z.flow_impact_band} flow risk · ${z.top_violation}</div>
        </div>
        <div class="forecast-score" style="color:${cisColor(forecastScore(z, h))}">${Math.round(forecastScore(z, h))}</div>
      </div>`).join("")}`;
  document.querySelectorAll(".forecast-row").forEach((el) =>
    el.addEventListener("click", () => openZone(+el.dataset.id, true)));
}

function renderImpactBoard() {
  const s = DATA.summary;
  const zones = DATA.zones.zones.slice()
    .sort((a, b) => (b.flow_impact_score || 0) - (a.flow_impact_score || 0))
    .slice(0, 4);
  const backtest = s.ml_prediction_backtest || s.prediction_backtest || {};
  $("#impact-board").innerHTML = `
    <div class="impact-kpis">
      <div><b>${s.avg_flow_impact_score || "—"}</b><span>Avg flow-risk score</span></div>
      <div><b>${s.severe_impact_zones || 0}</b><span>Severe zones</span></div>
      <div><b>${backtest.validation_coverage_pct ?? "—"}%</b><span>ML coverage@20</span></div>
    </div>
    <div class="decision-note">This is an obstruction-risk proxy, not measured speed loss. It uses offence type, peak timing and junction spillback context until speed or CCTV feeds are connected.</div>
    ${zones.map((z) => `
      <div class="impact-row" data-id="${z.id}">
        <div class="impact-band ${String(z.flow_impact_band || "").toLowerCase()}">${z.flow_impact_band}</div>
        <div class="impact-main"><b>${z.name}</b><span>${z.lane_blockage_risk_pct}% lane-blockage risk proxy · ${(z.impact_evidence || []).join(" · ")}</span></div>
        <div class="impact-score">${z.flow_impact_score}</div>
      </div>`).join("")}`;
  document.querySelectorAll(".impact-row").forEach((el) =>
    el.addEventListener("click", () => openZone(+el.dataset.id, true)));
}


/* ---------- ParkSched: MLFQ-inspired patrol scheduler ---------- */
function haversineKm(a, b) {
  const R = 6371;
  const dLat = (b.lat - a.lat) * Math.PI / 180;
  const dLon = (b.lon - a.lon) * Math.PI / 180;
  const lat1 = a.lat * Math.PI / 180;
  const lat2 = b.lat * Math.PI / 180;
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

function schedulerNeed(z, h) {
  const ml = forecastScore(z, h);
  const flow = z.flow_impact_score || z.cis || 0;
  const rec = 100 * (z.recurrence || 0);
  const conf = z.ml_prediction_confidence || z.prediction_confidence || 70;
  const block = z.lane_blockage_risk_pct || 0;
  const expected = expectedHourly(z, h);
  let need = 0.35 * ml + 0.25 * flow + 0.15 * rec + 0.15 * conf + 0.10 * block;
  // Demand multiplier keeps a zone with only normalized score but tiny expected demand from dominating.
  need *= Math.min(1.18, 0.86 + Math.log1p(expected) / 5);
  return Math.max(0, Math.min(120, need));
}

function queueForNeed(z, need) {
  if (need >= 85 || z.flow_impact_band === "Severe") return 0;
  if (need >= 65) return 1;
  if (need >= 40) return 2;
  return 3;
}

function queueName(q) {
  return ["Q0 Critical", "Q1 High", "Q2 Watch", "Q3 Low"][q] || "Q3 Low";
}

function queueAction(q) {
  if (q === 0) return "Tow + e-challan";
  if (q === 1) return "Patrol + challan";
  if (q === 2) return "Rotation patrol";
  return "Passive watch";
}

function buildPatrolSchedule(hour, units = 5, slots = 3) {
  const h = hour >= 0 ? hour : DATA.summary.peak_hour;
  const candidates = DATA.zones.zones
    .map((z) => {
      const need = schedulerNeed(z, h);
      return {
        zone: z,
        baseNeed: need,
        queue: queueForNeed(z, need),
        served: 0,
        wait: 0,
      };
    })
    .filter((c) => c.baseNeed >= 20)
    .sort((a, b) => a.queue - b.queue || b.baseNeed - a.baseNeed);

  const schedule = [];
  const covered = new Set();

  for (let slot = 0; slot < slots; slot++) {
    const chosen = [];
    const pool = candidates.map((c) => {
      const agingBoost = c.wait * 4.0;
      const repeatPenalty = c.served * 10.0;
      // Feedback: critical zones can return quickly, lower queues rotate more.
      const criticalReturn = c.queue === 0 && c.served > 0 ? 4.0 : 0.0;
      return { ...c, effectiveNeed: c.baseNeed + agingBoost - repeatPenalty + criticalReturn };
    }).sort((a, b) => a.queue - b.queue || b.effectiveNeed - a.effectiveNeed);

    for (const c of pool) {
      if (chosen.length >= units) break;
      // Avoid placing two units on nearly the same non-critical cluster in the same slot.
      const tooClose = chosen.some((x) => haversineKm(x.zone, c.zone) < 0.35);
      if (tooClose && c.queue > 0) continue;
      chosen.push(c);
    }

    // If spacing filtered too aggressively, fill remaining units by pure priority.
    for (const c of pool) {
      if (chosen.length >= units) break;
      if (!chosen.some((x) => x.zone.id === c.zone.id)) chosen.push(c);
    }

    const ids = new Set(chosen.map((c) => c.zone.id));
    candidates.forEach((c) => {
      if (ids.has(c.zone.id)) { c.served += 1; c.wait = 0; covered.add(c.zone.id); }
      else c.wait += 1;
    });

    schedule.push({
      slot,
      startMinute: slot * patrolQuantum,
      endMinute: (slot + 1) * patrolQuantum,
      assignments: chosen.map((c, i) => ({
        unit: i + 1,
        zone: c.zone,
        queue: c.queue,
        queueName: queueName(c.queue),
        need: Math.round(c.effectiveNeed),
        action: queueAction(c.queue),
      })),
    });
  }

  const totalPressure = candidates.reduce((s, c) => s + c.baseNeed, 0);
  const coveredPressure = candidates.filter((c) => covered.has(c.zone.id)).reduce((s, c) => s + c.baseNeed, 0);
  const severeZones = candidates.filter((c) => c.zone.flow_impact_band === "Severe");
  const severeCovered = severeZones.filter((c) => covered.has(c.zone.id)).length;

  return {
    hour: h,
    units,
    slots,
    quantum: patrolQuantum,
    schedule,
    coveragePct: Math.round(100 * coveredPressure / Math.max(totalPressure, 1)),
    severeCovered,
    severeTotal: severeZones.length,
    candidates: candidates.length,
    algorithm: "MLFQ-inspired priority scheduling with aging",
  };
}

function slotLabel(hour, startMinute, endMinute) {
  const startTotal = hour * 60 + startMinute;
  const endTotal = hour * 60 + endMinute;
  const sh = Math.floor((startTotal % 1440) / 60), sm = startTotal % 60;
  const eh = Math.floor((endTotal % 1440) / 60), em = endTotal % 60;
  return `${pad(sh)}:${String(sm).padStart(2, "0")}–${pad(eh)}:${String(em).padStart(2, "0")}`;
}

function renderSchedulerPanel() {
  const h = activeHour >= 0 ? activeHour : DATA.summary.peak_hour;
  const plan = buildPatrolSchedule(h, patrolUnits, patrolSlots);
  const firstSlot = plan.schedule[0]?.assignments || [];

  const tableRows = plan.schedule.map((slot) => `
    <div class="sched-slot">
      <div class="sched-time">${slotLabel(plan.hour, slot.startMinute, slot.endMinute)}</div>
      <div class="sched-assignments">
        ${slot.assignments.map((a) => `
          <div class="sched-assignment" data-id="${a.zone.id}">
            <div class="sched-unit">Unit ${a.unit}</div>
            <div class="sched-main">
              <b>${a.zone.name}</b>
              <span>${a.queueName} · need ${a.need} · ${a.action}</span>
            </div>
            <div class="sched-chip ${a.queue === 0 ? "critical" : a.queue === 1 ? "high" : "watch"}">${a.zone.flow_impact_band}</div>
          </div>`).join("")}
      </div>
    </div>`).join("");

  $("#scheduler-sub").textContent = `${plan.algorithm} · ${hourLabel(plan.hour)} start`;
  $("#scheduler-board").innerHTML = `
    <div class="sched-controls">
      <label>Available units
        <input type="number" id="unit-input" min="1" max="12" value="${patrolUnits}" />
      </label>
      <label>30-min slots
        <input type="number" id="slot-input" min="1" max="6" value="${patrolSlots}" />
      </label>
    </div>
    <div class="impact-kpis sched-kpis">
      <div><b>${plan.coveragePct}%</b><span>Predicted pressure covered</span></div>
      <div><b>${plan.severeCovered}/${plan.severeTotal}</b><span>Severe zones covered</span></div>
      <div><b>${firstSlot.length}</b><span>Units assigned now</span></div>
    </div>
    <div class="decision-note">
      ParkSched treats hotspot zones like processes and patrol/towing teams like CPUs.
      Q0 severe zones are served first, Q1/Q2 rotate next and aging boosts zones that wait too long so they are not starved.
    </div>
    <div class="sched-table">${tableRows}</div>`;

  $("#unit-input").addEventListener("input", (e) => {
    patrolUnits = Math.max(1, Math.min(12, +e.target.value || 1));
    renderSchedulerPanel();
  });
  $("#slot-input").addEventListener("input", (e) => {
    patrolSlots = Math.max(1, Math.min(6, +e.target.value || 1));
    renderSchedulerPanel();
  });
  document.querySelectorAll(".sched-assignment").forEach((el) =>
    el.addEventListener("click", () => openZone(+el.dataset.id, true)));
}

/* ---------- AI early-warning anomaly detector ---------- */
function renderAIAlertsPanel() {
  const payload = DATA.ai_alerts || { alerts: [], summary: {} };
  const summary = payload.summary || {};
  const allAlerts = (payload.alerts || []).slice();
  const hourMode = activeHour >= 0;

  // Time-aware alerts.
  // All-day mode shows the strongest recent anomalies city-wide.
  // Hour mode shows only anomalies for the selected patrol hour.
  let alerts = hourMode
    ? allAlerts.filter((a) => +a.hour === activeHour)
    : allAlerts;

  alerts = alerts
    .sort((a, b) => (b.early_warning_score || 0) - (a.early_warning_score || 0))
    .slice(0, 12);

  if (!alerts.length) {
    const msg = hourMode
      ? `No IsolationForest surge alerts for ${hourLabel(activeHour)}–${hourLabel((activeHour + 1) % 24)}. Move the hour slider to inspect another patrol window.`
      : `No AI early-warning alerts were generated for the latest window.`;
    $("#ai-sub").textContent = summary.enabled
      ? (hourMode ? `${summary.model || "IsolationForest"} · no alerts for selected hour` : "No abnormal recent surges found")
      : "AI detector unavailable";
    $("#ai-alerts-board").innerHTML = `<div class="decision-note">${msg}</div>`;
    return;
  }

  const high = alerts.filter((a) => a.alert_level === "Critical" || a.alert_level === "High").length;
  const scope = hourMode
    ? `${hourLabel(activeHour)}–${hourLabel((activeHour + 1) % 24)}`
    : "all hours";

  $("#ai-sub").textContent = `${summary.model || "IsolationForest"} · ${high} high/critical alerts for ${scope}`;
  $("#ai-alerts-board").innerHTML = `
    <div class="impact-kpis ai-kpis">
      <div><b>${alerts.length}</b><span>${hourMode ? "Selected-hour alerts" : "AI alerts"}</span></div>
      <div><b>${high}</b><span>High/Critical</span></div>
      <div><b>${summary.model || "IForest"}</b><span>Unsupervised model</span></div>
    </div>
    <div class="decision-note">
      IsolationForest learns normal zone-hour behavior from the older baseline window and flags recent abnormal surges for the selected patrol hour.
    </div>
    <div class="ai-alert-list">
      ${alerts.map((a) => `
        <div class="ai-alert ${a.alert_level.toLowerCase()}" data-id="${a.zone_id}">
          <div class="ai-level">${a.alert_level}</div>
          <div class="ai-main">
            <b>${a.zone_name}</b>
            <span>${hourLabel(a.hour)} · ${a.reason}</span>
          </div>
          <div class="ai-score">
            <b>${Math.round(a.early_warning_score)}</b>
            <span>${a.surge_ratio}×</span>
          </div>
        </div>`).join("")}
    </div>`;

  document.querySelectorAll(".ai-alert").forEach((el) =>
    el.addEventListener("click", () => openZone(+el.dataset.id, true)));
}


/* ---------- temporal heatmap ---------- */
function renderTemporal() {
  const m = DATA.temporal.matrix;
  const max = Math.max(...m.flat(), 1);
  let html = `<div class="th-corner"></div>`;
  for (let h = 0; h < 24; h++) {
    html += `<div class="th-hour">${h % 3 === 0 ? pad(h) : ""}</div>`;
  }
  for (let d = 0; d < 7; d++) {
    html += `<div class="th-lab">${WD[d]}</div>`;
    for (let h = 0; h < 24; h++) {
      const v = m[d][h];
      const t = v / max;
      const bg = t < 0.02 ? "#11182d" : cisColor(t * 100);
      const op = t < 0.02 ? 0.35 : 0.32 + 0.68 * t;
      const hot = t > 0.62 ? `<span>${fmt(v)}</span>` : "";
      html += `<div class="th-cell ${activeHour === h ? "selected" : ""}" title="${WD[d]} ${pad(h)}:00 — ${v.toLocaleString()} violations" style="background:${bg};opacity:${op}">${hot}</div>`;
    }
  }
  $("#temporal-heat").innerHTML = html;
  const peak = DATA.summary.peak_hour;
  $("#temporal-axis").innerHTML = `Peak hour: <b>${hourLabel(peak)}</b>. Use the hour slider to highlight a patrol window.`;
}

/* ---------- charts ---------- */
Chart.defaults.color = "#8b97b8";
Chart.defaults.font.family = "Inter";
Chart.defaults.borderColor = "rgba(36,49,84,.6)";

function renderCharts() {
  const b = DATA.breakdowns, t = DATA.temporal;
  const palette = ["#2874f0", "#4f93ff", "#37d6b3", "#ffd000", "#ff8a3d", "#ff3b57", "#a78bfa", "#f472b6", "#22d3ee", "#94a3b8"];

  new Chart($("#chart-violations"), {
    type: "doughnut",
    data: { labels: b.violations.map((d) => d.label), datasets: [{ data: b.violations.map((d) => d.n), backgroundColor: palette, borderWidth: 0 }] },
    options: { plugins: { legend: { position: "right", labels: { boxWidth: 10, font: { size: 10 } } } }, cutout: "58%" },
  });

  new Chart($("#chart-hourly"), {
    type: "line",
    data: { labels: [...Array(24).keys()].map(pad), datasets: [{ data: t.hourly, fill: true, borderColor: "#4f93ff", backgroundColor: "rgba(79,147,255,.18)", tension: .4, pointRadius: 0, borderWidth: 2 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { grid: { color: "rgba(36,49,84,.4)" } } } },
  });


  new Chart($("#chart-stations"), {
    type: "bar",
    data: { labels: b.stations.map((d) => d.label), datasets: [{ data: b.stations.map((d) => d.n), backgroundColor: "#ffd000", borderRadius: 5 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { font: { size: 9 }, maxRotation: 60, minRotation: 45 } }, y: { grid: { color: "rgba(36,49,84,.4)" } } } },
  });

  if ($("#chart-impact") && b.impact_bands) {
    new Chart($("#chart-impact"), {
      type: "bar",
      data: { labels: b.impact_bands.map((d) => d.label), datasets: [{ data: b.impact_bands.map((d) => d.n), backgroundColor: ["#ff3b57", "#ff8a3d", "#ffd000", "#37d6b3"], borderRadius: 5 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { grid: { color: "rgba(36,49,84,.4)" } } } },
    });
  }
}

/* ---------- controls ---------- */
function wireControls() {
  document.querySelectorAll("#layer-toggle button").forEach((b) =>
    b.addEventListener("click", () => setLayer(b.dataset.layer)));

  const slider = $("#hour-slider");
  slider.addEventListener("input", () => {
    activeHour = +slider.value;
    if (activeHour < 0) { $("#deploy-time").textContent = "All day"; }
    else {
      $("#deploy-time").textContent = `${hourLabel(activeHour)}–${hourLabel((activeHour + 1) % 24)}`;
      setLayer("zones");
      updateMarkersForHour();
    }
    renderZoneList(DATA.zones.zones);
    renderForecastPanel();
    renderSchedulerPanel();
    renderAIAlertsPanel();
    renderTemporal();
  });
  $("#reset-hour").addEventListener("click", () => {
    slider.value = -1; activeHour = -1;
    $("#deploy-time").textContent = "All day";
    resetMarkers(); renderZoneList(DATA.zones.zones); renderForecastPanel(); renderSchedulerPanel(); renderAIAlertsPanel(); renderTemporal();
  });

  $("#zone-filter").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    const filtered = DATA.zones.zones.filter((z) =>
      (z.name + z.locality + z.station + z.top_violation).toLowerCase().includes(q));
    renderFilteredList(filtered);
  });

  $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
}

function renderFilteredList(zones) {
  // keep CIS ordering for search results
  const prevHour = activeHour; activeHour = -1;
  renderZoneList(zones);
  activeHour = prevHour;
}

function updateMarkersForHour() {
  DATA.zones.zones.forEach((z) => {
    const score = forecastScore(z, activeHour);
    const m = zoneMarkers[z.id];
    if (score <= 0) { m.setStyle({ opacity: .12, fillOpacity: .06 }); m.setRadius(4); return; }
    m.setStyle({ opacity: .95, fillOpacity: .85, fillColor: cisColor(score) });
    m.setRadius(5 + Math.sqrt(score) * 1.35);
  });
}

function resetMarkers() {
  DATA.zones.zones.forEach((z) => {
    const st = markerStyle(z, z.cis);
    zoneMarkers[z.id].setStyle(st);
    zoneMarkers[z.id].setRadius(st.radius);
  });
}

window.closeModal = closeModal;
boot();
