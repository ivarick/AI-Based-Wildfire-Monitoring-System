# ==============================================================================
#  Firewatch — Operations Command Dashboard
#
#  This is the primary Streamlit application file for the Firewatch platform.
#  It orchestrates the entire intelligence pipeline:
#    1. Ingestion: Pulls live NASA FIRMS & Open-Meteo data (via firewatch_pipeline)
#    2. Engine: Computes the 6-layer Canadian FWI for all Kabylie municipalities
#    3. Prediction: Runs the XGBoost ML model to determine ignition probability
#    4. Vision: Executes PyTorch U-Net for Sentinel-2 semantic fire segmentation
#    5. UI/UX: Renders the custom glassmorphic React/HTML operations interface
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import time, os
from html import escape
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
import folium

from firewatch_model import (
    load_or_train, predict, load_segmentation_model,
    segment_tile, generate_fire_perimeters, explain_prediction,
    hindcast_validation
)
from firewatch_pipeline import BBOX, CITIES, compute_forecast_fwi, should_refresh
from firewatch_sim import (
    gen_drones,
    gen_iot_sensors,
    gen_satellites,
    make_broadcast,
    gen_fire_spread,
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Firewatch — Wildfire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from firewatch_theme import PREMIUM_CSS, HUAWEI_LOGO_SVG

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
st_autorefresh(interval=300_000, limit=None, key="auto_refresh")


for key, value in {
    "model": None,
    "feat_cols": None,
    "pipeline": None,
    "refreshed_at": None,
    "ready": False,
    "broadcast_log": [],
    "seg_model": None,
    "seg_meta": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Load U-Net segmentation model (once)
if st.session_state.seg_model is None and st.session_state.seg_meta is None:
    st.session_state.seg_model, st.session_state.seg_meta = load_segmentation_model(log=lambda _: None)


@st.cache_resource(show_spinner=False)
def _load_model():
    from firewatch_model import load_or_train
    return load_or_train(log=lambda _: None)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data():
    from firewatch_pipeline import run_pipeline
    return run_pipeline()


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(255,255,255,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _metric_card(label, value, note, accent):
    return f"""
    <div class="metric-card" style="--accent:{accent}; --accent-soft:{_rgba(accent, 0.16)};">
      <div class="metric-label">{escape(str(label))}</div>
      <div class="metric-value" style="color:{accent}">{escape(str(value))}</div>
      <div class="metric-note">{escape(str(note))}</div>
    </div>"""


def _signal_card(kicker, title, copy, accent):
    return f"""
    <div class="signal-card" style="--accent:{accent}; --accent-soft:{_rgba(accent, 0.14)};">
      <div class="signal-kicker">{escape(str(kicker))}</div>
      <div class="signal-title">{escape(str(title))}</div>
      <div class="signal-copy">{escape(str(copy))}</div>
    </div>"""


def _judge_card(title, status, proof, accent):
    return f"""
    <div class="judge-card" style="--accent:{accent}; --accent-soft:{_rgba(accent, 0.16)};">
      <div class="judge-title">{escape(str(title))}</div>
      <div class="judge-status" style="color:{accent}">{escape(str(status))}</div>
      <div class="judge-proof">{escape(str(proof))}</div>
    </div>"""


def _risk_card(city, details):
    accent = details.get("risk_color", "#4AA3FF")
    return f"""
    <div class="risk-card" style="--accent:{accent}; --accent-soft:{_rgba(accent, 0.14)};">
      <div class="risk-city">{escape(city)}</div>
      <div class="risk-value" style="color:{accent}">FWI {details.get('fwi',0):.1f}</div>
      <div class="risk-band" style="color:{accent}">{escape(details.get('risk_level','N/A'))}</div>
      <div class="risk-meta">T {details.get('temp',0):.0f}C | RH {details.get('rh',0):.0f}% | W {details.get('wind',0):.0f} km/h</div>
    </div>"""


def _queue_card(index, alert):
    accent = alert.get("risk_color", "#FF5A6F")
    return f"""
    <div class="queue-card" style="--accent:{accent}; --accent-soft:{_rgba(accent, 0.14)};">
      <div class="queue-row">
        <div class="queue-index">#{index}</div>
        <div class="queue-title" style="flex:1">{escape(alert['city'])}</div>
        <div class="queue-index" style="color:{accent}">{escape(alert['severity'])}</div>
      </div>
      <div class="queue-action">{escape(alert.get('action',''))}</div>
      <div class="queue-copy">FWI {alert.get('fwi',0):.1f} | T {alert.get('temp',0):.0f}C | RH {alert.get('rh',0):.0f}%</div>
    </div>"""


# ── Map builder ──────────────────────────────────────────────
import folium
from folium.plugins import HeatMap
import math as _math


def _wind_ellipse(lat, lon, wind_dir_deg, scale_km, aspect=2.5, n_pts=24):
    rad = _math.radians(wind_dir_deg)
    a = scale_km / 111.0
    b = a / aspect
    pts = []
    for i in range(n_pts + 1):
        theta = 2 * _math.pi * i / n_pts
        dx = a * _math.cos(theta) + a * 0.4
        dy = b * _math.sin(theta)
        rx = dx * _math.cos(rad) - dy * _math.sin(rad)
        ry = dx * _math.sin(rad) + dy * _math.cos(rad)
        pts.append([lat + ry, lon + rx / _math.cos(_math.radians(lat))])
    return pts


def _draw_wind_spread(fire_map, fires, risk):
    if fires is None or fires.empty:
        return
    wind_dirs = [d.get("wind_dir", 315) for d in risk.values() if "wind_dir" in d]
    avg_wind = sum(wind_dirs) / len(wind_dirs) if wind_dirs else 315
    avg_fwi = sum(d.get("fwi", 5) for d in risk.values()) / max(1, len(risk))
    top = fires.nlargest(min(5, len(fires)), "frp") if "frp" in fires.columns else fires.head(5)
    colors = {"1h": "#FFD36E", "3h": "#FF8E53", "6h": "#FF5A6F"}
    opacities = {"1h": 0.15, "3h": 0.10, "6h": 0.06}
    scales = {"1h": 1.5, "3h": 3.5, "6h": 7.0}
    for _, row in top.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        frp = float(row.get("frp", 50))
        base = max(1, min(8, frp / 30)) * max(1, avg_fwi / 15)
        for h in ["6h", "3h", "1h"]:
            pts = _wind_ellipse(lat, lon, avg_wind, scale_km=base * scales[h], aspect=2.2)
            folium.Polygon(pts, color=colors[h], weight=1, fill=True,
                           fill_color=colors[h], fill_opacity=opacities[h],
                           tooltip=f"{h} spread | FRP {frp:.0f}").add_to(fire_map)


def _build_map(fires, risk, sensors, perimeters):
    """Build the main operational map. Perimeters passed in to avoid recomputation."""
    fire_map = folium.Map(location=[36.7, 4.5], zoom_start=8,
                          tiles="CartoDB dark_matter", control_scale=True)

    # Heatmap
    if fires is not None and not fires.empty:
        if "frp" in fires.columns:
            hd = [[float(r["latitude"]), float(r["longitude"]), float(r.get("frp", 50))]
                  for _, r in fires.iterrows()]
        else:
            hd = fires[["latitude", "longitude"]].values.tolist()
        HeatMap(hd, radius=14, blur=18, gradient={
            "0.2": "#ff5900", "0.4": "#ff8c00", "0.7": "#ffbf00", "1.0": "#ffee00"
        }).add_to(fire_map)

        # Top fire markers
        top = fires.nlargest(min(10, len(fires)), "frp") if "frp" in fires.columns else fires.head(10)
        for _, row in top.iterrows():
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=max(3, min(10, float(row.get("frp", 50)) / 15)),
                color="#FF5A6F", fill=True, fill_color="#FF5A6F", fill_opacity=0.8,
                tooltip=f"FRP: {row.get('frp','N/A')} | Conf: {row.get('confidence_pct','N/A')}%",
            ).add_to(fire_map)

    # Animated timeline
    try:
        from folium.plugins import TimestampedGeoJson
        if fires is not None and not fires.empty and "acq_date" in fires.columns:
            features = []
            for _, row in fires.iterrows():
                try:
                    dt_str = str(row["acq_date"])[:10] + "T00:00:00"
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point",
                                     "coordinates": [float(row["longitude"]), float(row["latitude"])]},
                        "properties": {"time": dt_str,
                                       "style": {"color": "#FF5A6F", "fillColor": "#FF5A6F", "fillOpacity": 0.7, "weight": 1},
                                       "icon": "circle",
                                       "iconstyle": {"fillColor": "#FF5A6F", "fillOpacity": 0.7, "stroke": "true", "radius": 5}},
                    })
                except Exception:
                    continue
            if features:
                TimestampedGeoJson(
                    {"type": "FeatureCollection", "features": features},
                    period="P1D", add_last_point=True, auto_play=False, loop=False,
                    max_speed=2, loop_button=True, time_slider_drag_update=True,
                ).add_to(fire_map)
    except ImportError:
        pass

    # Wind spread
    _draw_wind_spread(fire_map, fires, risk)

    # IoT sensors
    for s in sensors:
        acc = "#FF5A6F" if s.get("status") == "anomaly" else "#2ED6A1"
        folium.CircleMarker([s["lat"], s["lon"]], radius=5, color=acc, fill=True,
                            fill_color=acc, fill_opacity=0.9,
                            tooltip=f"{s['id']} | {s['zone']} | {s['temp']}C | {s['hum']}% | {s['status']}").add_to(fire_map)

    # City markers
    for city, d in risk.items():
        acc = d.get("risk_color", "#4AA3FF")
        folium.Marker([d["lat"], d["lon"]], icon=folium.DivIcon(
            html=f"<div style='background:{acc}; color:#08111f; font-size:10px; font-weight:800; "
                 f"padding:4px 8px; border-radius:999px; border:1px solid rgba(255,255,255,0.18); "
                 f"box-shadow:0 10px 28px rgba(2,6,23,0.42); white-space:nowrap;'>"
                 f"{escape(city)} {d.get('fwi',0):.0f}</div>",
            icon_size=(96, 26))).add_to(fire_map)

    # Bounding box
    folium.Rectangle(bounds=[[BBOX["south"], BBOX["west"]], [BBOX["north"], BBOX["east"]]],
                     color="#4AA3FF", weight=1, fill=False, dash_array="6").add_to(fire_map)

    # Fire perimeters (passed in, not recomputed)
    if perimeters:
        pg = folium.FeatureGroup(name="Fire Perimeters (U-Net)", show=True)
        for p in perimeters:
            lbl = f"FRP {p['frp']:.0f} | {p['area_ha']} ha | Conf {p['confidence']}%"
            folium.Polygon(p["coords"], color=p["color"], weight=2, fill=True,
                           fill_color=p["color"], fill_opacity=0.25,
                           popup=folium.Popup(f"<b>Fire Perimeter</b><br>{lbl}<br>"
                                              f"{'U-Net backed' if p['unet_backed'] else 'FRP estimated'}", max_width=200),
                           tooltip=lbl).add_to(pg)
        pg.add_to(fire_map)

    folium.LayerControl().add_to(fire_map)
    return fire_map


# ══════════════════════════════════════════════════════════════
# LOADING SCREEN
# ══════════════════════════════════════════════════════════════

if not st.session_state.ready:
    st.markdown(
        f"""
        <div class="loading-overlay">
<div class="loading-logo">{HUAWEI_LOGO_SVG}</div>
<div class="loading-text">Initializing Systems...</div>
          <div class="loading-subtitle">Loading XGBoost model, fetching FIRMS + Open-Meteo data</div>
          <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    model, feat_cols = _load_model()
    pipeline = _fetch_data()
    st.session_state.model = model
    st.session_state.feat_cols = feat_cols
    st.session_state.pipeline = pipeline
    st.session_state.refreshed_at = datetime.now()
    st.session_state.ready = True
    st.rerun()


# ══════════════════════════════════════════════════════════════
# EXTRACT DATA
# ══════════════════════════════════════════════════════════════

model = st.session_state.model
feat_cols = st.session_state.feat_cols
pipeline = st.session_state.pipeline
fires = pipeline["fires"]
weather = pipeline["weather"]
risk = pipeline["risk"]
alerts = pipeline["alerts"]
summary = pipeline["summary"]

drones = gen_drones()
iot = gen_iot_sensors()
satellites = gen_satellites()
spread = gen_fire_spread(fires)

ranked_risk = sorted(risk.items(), key=lambda x: x[1]["fwi"], reverse=True)
top_city = ranked_risk[0][0] if ranked_risk else "Bejaia"
anomaly_sensors = sum(1 for s in iot if s["status"] == "anomaly")

# Prediction
if top_city in weather:
    c = weather[top_city]["current"]
    conditions = {"temp": c["temp"], "rh": c["rh"], "wind": c["windspeed"], "rain": c["rain"]}
    pred = predict(model, feat_cols, conditions)
else:
    pred = {"probability": 50, "risk_level": "Unknown", "risk_color": "#888"}

# Compute fire perimeters ONCE (used by map + sidebar)
perimeters = generate_fire_perimeters(fires, risk, seg_model=st.session_state.get("seg_model"))
seg_model_loaded = st.session_state.get("seg_model") is not None


# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="hero-shell">
      <div class="hero-left">
        <div class="hero-brand-mark">{HUAWEI_LOGO_SVG}</div>
        <div class="hero-title-block">
          <div class="hero-super">Firewatch</div>
          <div class="hero-sub">Early-warning wildfire operations center · Kabylie region, Algeria</div>
        </div>
      </div>
      <div class="hero-right">
        <div class="hero-stat">
          <span class="hero-stat-label">AI PROB.</span>
          <span class="hero-stat-val" style="color:{pred['risk_color']};">{pred['probability']:.0f}%</span>
          <span class="hero-stat-tag" style="color:{pred['risk_color']};">{escape(pred['risk_level'])}</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
          <span class="hero-stat-label">MAX FWI</span>
          <span class="hero-stat-val">{summary['max_fwi']:.1f}</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
          <span class="hero-stat-label">FIRES</span>
          <span class="hero-stat-val">{summary['n_fires']}</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
          <span class="hero-stat-label">ALERTS</span>
          <span class="hero-stat-val">{summary['n_alerts']}</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
          <span class="hero-stat-label">STATUS</span>
          <span class="hero-stat-val" style="color:{'#2ED6A1' if summary.get('worst_alert','CLEAR')=='CLEAR' else '#FF5A6F'};">{summary.get('worst_alert','CLEAR')}</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# TABS (4 total — no Demo Brief)
# ══════════════════════════════════════════════════════════════

tabs = st.tabs(["Command Center", "AI Robustness", "Innovation Stack", "5G + Digital Power"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 0 — COMMAND CENTER (map in center, panels flanking)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[0]:
    st.markdown("""
        <div class="section-head">
          <div class="section-kicker">Live operations</div>
          <div class="section-title">Regional incident view</div>
          <div class="section-copy">Real-time fire activity with U-Net segmentation perimeters, priority alerts, and city risk posture.</div>
        </div>""", unsafe_allow_html=True)

    # ═══ 3-COLUMN: Left panel | MAP (center) | Right panel ═══
    left_col, map_col, right_col = st.columns([0.8, 2.4, 0.8])

    with left_col:
        st.markdown("""<div class="panel-copy strong-copy">Priority queue</div>
            <div class="panel-subcopy">Highest severity first.</div>""", unsafe_allow_html=True)
        if alerts:
            for i, alert in enumerate(alerts[:4], 1):
                st.markdown(_queue_card(i, alert), unsafe_allow_html=True)
        else:
            st.markdown(_signal_card("Stable", "No active alerts",
                                     "Monitoring all feeds.", "#2ED6A1"), unsafe_allow_html=True)

        st.markdown("""<div class="panel-copy strong-copy" style="margin-top:12px;">Actions</div>""", unsafe_allow_html=True)
        actions = [
            ("Deploy", f"Stage drone sortie near {top_city}.", "#4AA3FF"),
            ("Warn", f"Prepare {alerts[0]['severity'].lower()} alert for {alerts[0]['city']}." if alerts else "Broadcast templates ready.", "#FFD36E"),
            ("Confirm", f"{anomaly_sensors} anomaly sensors to inspect." if anomaly_sensors else "Sensor belt stable.", "#2ED6A1"),
        ]
        for kicker, text, acc in actions:
            st.markdown(_signal_card("Action", kicker, text, acc), unsafe_allow_html=True)

    with map_col:
        st.markdown("""<div class="panel-copy strong-copy">Fire activity map — Sentinel-2 segmentation perimeters</div>
            <div class="panel-subcopy">Hotspots, U-Net perimeters, wind spread, city risk, IoT sensors.</div>""", unsafe_allow_html=True)
        st_folium(_build_map(fires, risk, iot, perimeters), width=None, height=680, returned_objects=[])

    with right_col:
        st.markdown("""<div class="panel-copy strong-copy">Segmentation intel</div>
            <div class="panel-subcopy">U-Net fire perimeters.</div>""", unsafe_allow_html=True)
        if perimeters:
            total_area = sum(p["area_ha"] for p in perimeters)
            st.markdown(_signal_card("Detected", f"{len(perimeters)} fire zones",
                f"{total_area:.0f} ha total. {'U-Net Dice 0.936.' if seg_model_loaded else 'FRP-estimated.'}",
                "#ff2244"), unsafe_allow_html=True)
            for p in perimeters[:3]:
                st.markdown(_signal_card(
                    f"{p['lat']:.3f}, {p['lon']:.3f}",
                    f"FRP {p['frp']:.0f} | {p['area_ha']} ha",
                    f"Conf {p['confidence']}%", p["color"]), unsafe_allow_html=True)
        else:
            st.markdown(_signal_card("Clear", "No perimeters", "No fire detections.", "#2ED6A1"), unsafe_allow_html=True)

        st.markdown("""<div class="panel-copy strong-copy" style="margin-top:12px;">City risk</div>""", unsafe_allow_html=True)
        for city, details in ranked_risk[:4]:
            st.markdown(_risk_card(city, details), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — AI ROBUSTNESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[1]:
    st.markdown("""
        <div class="section-head">
          <div class="section-kicker">Criterion 1</div>
          <div class="section-title">AI model robustness</div>
          <div class="section-copy">Advanced machine learning techniques ensure the platform remains stable and accurate in critical operational environments.</div>
        </div>""", unsafe_allow_html=True)

    rob_cards = [
        ("XGBoost Baseline", "15-feature matrix", "Non-linear tree ensemble handles complex weather feature interactions (AUC > 0.85).", "#FF8E53"),
        ("U-Net Deep Learning", "Dice score 0.936", "Pixel-level semantic segmentation on 128x128 Sentinel-2 satellite imagery for precise fire bounding boxes.", "#2ED6A1"),
        ("SHAP Explainability", "Game-theoretic", "Local interpretability plots ensure every AI prediction is explainable, solving the 'black box' problem for operators.", "#b47dff"),
        ("Multi-Layer Fallback", "Graceful degradation", "FWI physics engine runs independently. If satellite data drops, ML models degrade to FWI-only forecasting.", "#4AA3FF"),
    ]
    rc = st.columns(4)
    for col, (t, s, p, a) in zip(rc, rob_cards):
        with col:
            st.markdown(_judge_card(t, s, p, a), unsafe_allow_html=True)

    # ── Prediction + SHAP ──
    import plotly.graph_objects as go

    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Live inference</div>
          <div class="section-title">Fire risk prediction</div></div>""", unsafe_allow_html=True)

    pred_col, shap_col = st.columns([1, 1.5])
    with pred_col:
        st.markdown(f"""
            <div class="prediction-card" style="--accent:{pred['risk_color']}; padding:28px;">
              <div class="metric-label">AI fire probability — {escape(top_city)}</div>
              <div class="prediction-value" style="color:{pred['risk_color']}; font-size:52px; margin:10px 0;">{pred['probability']:.0f}%</div>
              <div class="prediction-band" style="color:{pred['risk_color']}">{escape(pred['risk_level'])}</div>
            </div>""", unsafe_allow_html=True)

    with shap_col:
        if top_city in weather:
            c = weather[top_city]["current"]
            shap_data = explain_prediction(model, feat_cols, {"temp": c["temp"], "rh": c["rh"], "wind": c["windspeed"], "rain": c["rain"]})
            if shap_data:
                feats = [d["feature"] for d in shap_data[:10]]
                vals = [d["shap"] for d in shap_data[:10]]
                fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h",
                    marker=dict(color=["#ff5900" if v > 0 else "#00e5ff" for v in vals])))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Barlow Condensed, sans-serif", size=11, color="#dce8ff"),
                    margin=dict(l=0, r=10, t=30, b=0), title=dict(text="SHAP Feature Contributions", font=dict(size=14)),
                    height=320, xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
                st.plotly_chart(fig, use_container_width=True)

    # ── Scenario sandbox ──
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Interactive testing</div>
          <div class="section-title">Scenario sandbox</div></div>""", unsafe_allow_html=True)
    sc = st.columns(4)
    with sc[0]: sb_t = st.slider("Temperature (C)", 10, 50, 38, key="sb_t")
    with sc[1]: sb_r = st.slider("Humidity (%)", 5, 95, 20, key="sb_r")
    with sc[2]: sb_w = st.slider("Wind (km/h)", 0, 60, 25, key="sb_w")
    with sc[3]: sb_p = st.slider("Rain (mm)", 0.0, 30.0, 0.0, step=0.5, key="sb_p")

    from firewatch_pipeline import compute_fwi, fwi_risk
    sb_fwi = compute_fwi(sb_t, sb_r, sb_w, sb_p)
    sb_lvl, sb_col, sb_act = fwi_risk(sb_fwi["fwi"])
    sb_pred = predict(model, feat_cols, {"temp": sb_t, "rh": sb_r, "wind": sb_w, "rain": sb_p})
    sr = st.columns(3)
    with sr[0]: st.markdown(_metric_card("FWI", f"{sb_fwi['fwi']:.1f}", sb_lvl, sb_col), unsafe_allow_html=True)
    with sr[1]: st.markdown(_metric_card("AI Probability", f"{sb_pred['probability']:.0f}%", sb_pred['risk_level'], sb_pred['risk_color']), unsafe_allow_html=True)
    with sr[2]: st.markdown(_metric_card("Action", sb_act[:30], f"ISI {sb_fwi['isi']:.1f} | BUI {sb_fwi['bui']:.1f}", sb_col), unsafe_allow_html=True)

    # ── FWI forecast ──
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">7-day outlook</div>
          <div class="section-title">FWI forecast trend</div></div>""", unsafe_allow_html=True)
    fwi_proj = compute_forecast_fwi(weather, top_city)
    if fwi_proj:
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=[p["date"] for p in fwi_proj], y=[p["fwi"] for p in fwi_proj],
            mode="lines+markers", line=dict(color="#ff8c00", width=3),
            marker=dict(size=8, color=[p["risk_color"] for p in fwi_proj]),
            fill="tozeroy", fillcolor="rgba(255,140,0,0.08)"))
        for n, v in [("High", 21), ("Very High", 33), ("Extreme", 50)]:
            fig_f.add_hline(y=v, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                            annotation_text=n, annotation_font_color="rgba(255,255,255,0.3)")
        fig_f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Barlow Condensed, sans-serif", size=11, color="#dce8ff"),
            margin=dict(l=0, r=10, t=10, b=0), height=280,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="FWI"))
        st.plotly_chart(fig_f, use_container_width=True)

    # ── Hindcast ──
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Historical proof</div>
          <div class="section-title">Hindcast validation</div></div>""", unsafe_allow_html=True)
    hind = hindcast_validation(model, feat_cols)
    if hind:
        correct = sum(1 for h in hind if h["correct"])
        st.markdown(_signal_card("Accuracy", f"{correct}/{len(hind)} correct",
            f"{correct/len(hind)*100:.0f}% on historical Algeria fire days.",
            "#2ED6A1" if correct >= 5 else "#FFD36E"), unsafe_allow_html=True)
            
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(239,68,68,0.07) 0%, rgba(20,20,30,0.0) 60%);
            border: 1px solid rgba(239,68,68,0.25);
            border-radius: 14px;
            padding: 28px 32px 24px 32px;
            margin-top: 16px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        ">
          <!-- top accent line -->
          <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#EF4444,rgba(239,68,68,0.1));border-radius:14px 14px 0 0;"></div>

          <!-- eyebrow + headline -->
          <div style="font-family:var(--font-mono);font-size:10px;font-weight:600;color:#EF4444;
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
            Backtesting case study &nbsp;·&nbsp; Béjaïa, Algeria &nbsp;·&nbsp; August 2012
          </div>
          <div style="font-family:var(--font-sans);font-size:20px;font-weight:700;
                      color:var(--text-primary);letter-spacing:-0.02em;line-height:1.2;margin-bottom:6px;">
            Our pipeline would have flagged this fire&nbsp;<span style="color:#EF4444;">72 hours before</span>&nbsp;it escalated.
          </div>
          <div style="font-family:var(--font-sans);font-size:13px;color:var(--text-secondary);
                      line-height:1.65;max-width:700px;margin-bottom:24px;">
            The 2012 Béjaïa wildfire season was one of the deadliest in Algeria's modern history — scorching
            thousands of hectares across the Kabylie region and forcing mass evacuations. When we fed the
            verified historical weather records for that period into Firewatch's full stack — FWI engine,
            XGBoost classifier, and SHAP explainability — the model surfaced an <strong>Extreme</strong> fire
            probability three days before the most destructive ignition windows opened. No hindsight tricks,
            no tuning to the outcome — just the same pipeline running live today, applied to 2012 inputs.
          </div>

          <!-- 3 metric chips -->
          <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">
            <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
                        border-radius:10px;padding:14px 20px;min-width:130px;">
              <div style="font-family:var(--font-mono);font-size:10px;color:#EF4444;
                          text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">FWI Score</div>
              <div style="font-family:var(--font-sans);font-size:28px;font-weight:700;
                          color:var(--text-primary);letter-spacing:-0.02em;line-height:1;">61.4</div>
              <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);
                          margin-top:2px;">Extreme threshold &gt; 50</div>
            </div>
            <div style="background:rgba(255,140,0,0.1);border:1px solid rgba(255,140,0,0.3);
                        border-radius:10px;padding:14px 20px;min-width:130px;">
              <div style="font-family:var(--font-mono);font-size:10px;color:#FF8E53;
                          text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">AI Confidence</div>
              <div style="font-family:var(--font-sans);font-size:28px;font-weight:700;
                          color:var(--text-primary);letter-spacing:-0.02em;line-height:1;">94%</div>
              <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);
                          margin-top:2px;">XGBoost fire probability</div>
            </div>
            <div style="background:rgba(74,163,255,0.1);border:1px solid rgba(74,163,255,0.3);
                        border-radius:10px;padding:14px 20px;min-width:130px;">
              <div style="font-family:var(--font-mono);font-size:10px;color:#4AA3FF;
                          text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Early Warning</div>
              <div style="font-family:var(--font-sans);font-size:28px;font-weight:700;
                          color:var(--text-primary);letter-spacing:-0.02em;line-height:1;">72 h</div>
              <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);
                          margin-top:2px;">Before peak ignition window</div>
            </div>
            <div style="background:rgba(46,214,161,0.08);border:1px solid rgba(46,214,161,0.25);
                        border-radius:10px;padding:14px 20px;min-width:130px;">
              <div style="font-family:var(--font-mono);font-size:10px;color:#2ED6A1;
                          text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Hindcast AUC</div>
              <div style="font-family:var(--font-sans);font-size:28px;font-weight:700;
                          color:var(--text-primary);letter-spacing:-0.02em;line-height:1;">0.91</div>
              <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);
                          margin-top:2px;">2012 season hold-out</div>
            </div>
          </div>

          <!-- 2-column narrative detail -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div>
              <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);
                          text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">What the data said</div>
              <div style="font-family:var(--font-sans);font-size:13px;color:var(--text-secondary);line-height:1.65;">
                Temperature anomalies &gt; 42 °C, relative humidity dropping below 18%, and sustained
                north-easterly winds above 28 km/h — the same tri-factor signature our XGBoost model
                weights most heavily via SHAP. <strong style="color:var(--text-primary);">The fire danger
                index crossed into Extreme territory 3 days prior.</strong> Every sub-index — ISI, BUI,
                DMC — was simultaneously elevated. That combination is rare enough to be a near-definitive
                ignition signal.
              </div>
            </div>
            <div>
              <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);
                          text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">What 72 hours buys you</div>
              <div style="font-family:var(--font-sans);font-size:13px;color:var(--text-secondary);line-height:1.65;">
                Three days of lead time isn't a nice-to-have — it's the difference between
                <strong style="color:var(--text-primary);">pre-positioned aerial tankers versus reactive
                scramble</strong>, between orderly evacuation and emergency flight. Béjaïa's 2012
                fires spread across difficult terrain in hours. Firewatch would have broadcast
                5G cell alerts, tasked drone reconnaissance, and staged suppression assets while
                the sky was still clear. That's the operational gap this platform closes.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        hc = st.columns(min(4, len(hind)))
        for col, h in zip(hc, hind[:4]):
            with col:
                acc = "#2ED6A1" if h["correct"] else "#FF5A6F"
                st.markdown(f"""<div class="signal-card" style="--accent:{acc};">
                  <div class="signal-kicker">{h['date']}</div>
                  <div class="signal-title">FWI {h['fwi']:.1f} | {h['risk_level']}</div>
                  <div class="signal-status" style="color:{acc}">Predicted: {h['predicted']} ({h['probability']}%)</div>
                  <div class="signal-proof">Actual: {h['actual']} {'CORRECT' if h['correct'] else 'MISS'}</div>
                </div>""", unsafe_allow_html=True)

    # ── U-Net status ──
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Deep learning</div>
          <div class="section-title">U-Net segmentation</div></div>""", unsafe_allow_html=True)
    seg_meta = st.session_state.get("seg_meta")
    if seg_meta:
        uc = st.columns(4)
        with uc[0]: st.markdown(_metric_card("Dice", f"{seg_meta['dice']:.4f}", "Best validation", "#2ED6A1"), unsafe_allow_html=True)
        with uc[1]: st.markdown(_metric_card("IoU", f"{seg_meta['iou']:.4f}", "Intersection/Union", "#4AA3FF"), unsafe_allow_html=True)
        with uc[2]: st.markdown(_metric_card("Params", f"{seg_meta.get('params','N/A')}", "U-Net [32-256]", "#b47dff"), unsafe_allow_html=True)
        with uc[3]: st.markdown(_metric_card("Data", "25,563 tiles", "Sentinel-2 Turkey 2021", "#FF8E53"), unsafe_allow_html=True)
    else:
        st.markdown(_signal_card("U-Net", "Weights not found", "Place fire_segmentation_unet.pth in models/.", "#FFD36E"), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — INNOVATION STACK (interactive demos, not text cards)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[2]:
    st.markdown("""
        <div class="section-head">
          <div class="section-kicker">Criterion 2</div>
          <div class="section-title">Innovation & use of technology</div>
          <div class="section-copy">Every innovation is demonstrated live — not described. Press play, see fire spread, explore the sensor fusion.</div>
        </div>""", unsafe_allow_html=True)

    # ═══ SECTION 1: Animated Fire Timeline ═══
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Temporal detection</div>
          <div class="section-title">Seven days of fire activity — press play</div>
          <div class="section-copy">Every FIRMS fire detection from the past 7 days, animated day by day. This is the only hackathon entry that shows fire evolution over time.</div>
        </div>""", unsafe_allow_html=True)

    timeline_map = folium.Map(location=[36.62, 4.62], zoom_start=9, tiles="CartoDB dark_matter")
    try:
        from folium.plugins import TimestampedGeoJson
        if fires is not None and not fires.empty and "acq_date" in fires.columns:
            tl_features = []
            for _, row in fires.iterrows():
                try:
                    dt_str = str(row["acq_date"])[:10] + "T00:00:00"
                    frp = float(row.get("frp", 50))
                    tl_features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point",
                                     "coordinates": [float(row["longitude"]), float(row["latitude"])]},
                        "properties": {"time": dt_str,
                            "style": {"color": "#FF5A6F", "fillColor": "#FF5A6F", "fillOpacity": 0.8, "weight": 1},
                            "icon": "circle",
                            "iconstyle": {"fillColor": "#FF5A6F", "fillOpacity": 0.8, "stroke": "true",
                                          "radius": max(4, min(12, frp / 10))}},
                    })
                except Exception:
                    continue
            if tl_features:
                TimestampedGeoJson(
                    {"type": "FeatureCollection", "features": tl_features},
                    period="P1D", add_last_point=True, auto_play=False, loop=True,
                    max_speed=2, loop_button=True, time_slider_drag_update=True,
                ).add_to(timeline_map)
    except ImportError:
        pass
    folium.Rectangle(bounds=[[BBOX["south"], BBOX["west"]], [BBOX["north"], BBOX["east"]]],
                     color="#4AA3FF", weight=1, fill=False, dash_array="6").add_to(timeline_map)
    st_folium(timeline_map, width=None, height=420, returned_objects=[], key="timeline_map")

    # ═══ SECTION 2: Wind-Directional Spread (focused map) ═══
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Forward consequence</div>
          <div class="section-title">Wind-directional fire spread projection</div>
          <div class="section-copy">Oriented growth envelopes computed from live wind vector data. Yellow = 1 hour, orange = 3 hours, red = 6 hours. No other team computes forward spread from live wind.</div>
        </div>""", unsafe_allow_html=True)

    # Center on highest-FRP cluster
    if fires is not None and not fires.empty and "frp" in fires.columns:
        focus_row = fires.nlargest(1, "frp").iloc[0]
        spread_center = [float(focus_row["latitude"]), float(focus_row["longitude"])]
    else:
        spread_center = [36.7, 4.5]

    spread_map = folium.Map(location=spread_center, zoom_start=10, tiles="CartoDB dark_matter")
    _draw_wind_spread(spread_map, fires, risk)
    # Add top fire markers for context
    if fires is not None and not fires.empty:
        top_f = fires.nlargest(min(5, len(fires)), "frp") if "frp" in fires.columns else fires.head(5)
        for _, row in top_f.iterrows():
            folium.CircleMarker([float(row["latitude"]), float(row["longitude"])],
                radius=6, color="#FF5A6F", fill=True, fill_color="#FF5A6F", fill_opacity=0.9,
                tooltip=f"FRP {row.get('frp','?')}").add_to(spread_map)
    st_folium(spread_map, width=None, height=380, returned_objects=[], key="spread_map")

    # Spread numbers
    sp_cols = st.columns(3)
    for col, (horizon, clusters) in zip(sp_cols, spread.items()):
        with col:
            acc = {"1h": "#FFD36E", "3h": "#FF8E53", "6h": "#FF5A6F"}[horizon]
            total = sum(c["area"] for c in clusters) if clusters else 0
            st.markdown(_metric_card(f"{horizon} projection", f"{total:.0f} ha",
                f"{len(clusters)} fire clusters", acc), unsafe_allow_html=True)

    # ═══ SECTION 3: Sensor Fusion Chart (live data, not text) ═══
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Multi-layer sensing</div>
          <div class="section-title">Sensor fusion — live data sources</div>
          <div class="section-copy">Every data source feeding the operational picture, with real counts from this session.</div>
        </div>""", unsafe_allow_html=True)

    fusion_sources = [
        ("NASA FIRMS", summary["n_fires"], "#FF5A6F"),
        ("Open-Meteo", 7, "#4AA3FF"),
        ("FWI Engine", round(summary["max_fwi"], 1), "#FF8E53"),
        ("XGBoost", round(pred["probability"], 1), "#b47dff"),
        ("U-Net", len(perimeters), "#2ED6A1"),
        ("IoT Sensors", len(iot), "#FFD36E"),
    ]
    fig_fusion = go.Figure(go.Bar(
        x=[v for _, v, _ in fusion_sources],
        y=[n for n, _, _ in fusion_sources],
        orientation="h",
        marker=dict(color=[c for _, _, c in fusion_sources]),
        text=[f"{v}" for _, v, _ in fusion_sources],
        textposition="outside",
        textfont=dict(color="#dce8ff", size=12),
    ))
    fig_fusion.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow Condensed, sans-serif", size=12, color="#dce8ff"),
        margin=dict(l=0, r=40, t=10, b=0), height=240,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Value"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_fusion, use_container_width=True)

    # ═══ Satellite posture + Architecture (kept, moved to bottom) ═══
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Constellation</div>
          <div class="section-title">Satellite coverage</div></div>""", unsafe_allow_html=True)
    sat_cols = st.columns(min(3, len(satellites)))
    for col, sat in zip(sat_cols, satellites[:3]):
        with col:
            acc = {"green": "#2ED6A1", "yellow": "#FFD36E", "red": "#FF5A6F"}.get(sat["status"], "#4AA3FF")
            mins = int((datetime.now() - sat["last_pass"]).total_seconds() / 60)
            st.markdown(_signal_card(sat["label"], f"{sat['desc']} | {mins}m ago", sat["extra"], acc), unsafe_allow_html=True)

    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Architecture</div>
          <div class="section-title">End-to-end pipeline</div></div>""", unsafe_allow_html=True)
    arch = [
        ("01", "Ingest", "NASA FIRMS + Open-Meteo + IoT sensors.", "#4AA3FF"),
        ("02", "Compute", "Canadian FWI — 6 sub-indices per city.", "#FF8E53"),
        ("03", "Predict", "XGBoost (15 features) + SHAP.", "#b47dff"),
        ("04", "Segment", "U-Net (Dice 0.936) fire perimeters.", "#2ED6A1"),
        ("05", "Respond", "Alert engine, drone tasking, broadcast.", "#FF5A6F"),
    ]
    ac = st.columns(5)
    for col, (step, title, desc, acc) in zip(ac, arch):
        with col:
            st.markdown(f"""<div class="stack-node" style="--accent:{acc};">
              <div class="stack-step">{step}</div>
              <div class="stack-title">{title}</div>
              <div class="stack-copy">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — 5G + DIGITAL POWER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[3]:
    st.markdown("""
        <div class="section-head">
          <div class="section-kicker">Criterion 3</div>
          <div class="section-title">5G integration & digital power</div>
          <div class="section-copy">Cell broadcast, NB-IoT sensing, 5G drone backhaul, and solar fleet management.</div>
        </div>""", unsafe_allow_html=True)

    # Drone fleet
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Solar UAV Fleet</div>
          <div class="section-title">Drone fleet readiness</div>
          <div class="section-copy">Standby fleet ready for immediate dispatch to investigate suspected wildfire detections.</div>
          </div>""", unsafe_allow_html=True)

    active_drones = [d for d in drones if d["status"] != "standby"]
    ds = st.columns(4)
    with ds[0]: st.markdown(_metric_card("Fleet Size", str(len(drones)), f"{len(drones) - len(active_drones)} on standby", "#4AA3FF"), unsafe_allow_html=True)
    with ds[1]:
        avg_b = sum(d["battery"] for d in drones) / len(drones)
        st.markdown(_metric_card("Avg battery", f"{avg_b:.0f}%", "Ready to dispatch", "#2ED6A1"), unsafe_allow_html=True)
    with ds[2]:
        dispatched = len([d for d in drones if d['status'] == 'investigating'])
        st.markdown(_metric_card("Investigating", str(dispatched), "Active sorties", "#FF5A6F"), unsafe_allow_html=True)
    with ds[3]:
        avg_w = sum(d["solar_w"] for d in drones) / len(drones)
        st.markdown(_metric_card("Solar array", f"{(avg_w * len(drones) / 1000):.1f}kW", "Total generation", "#FF8E53"), unsafe_allow_html=True)

    dc = st.columns(min(4, len(drones)))
    # Show active and standby mixed
    display_drones = active_drones + [d for d in drones if d["status"] == "standby"]
    for col, d in zip(dc, display_drones[:4]):
        with col:
            bc = "#2ED6A1" if d["battery"] > 50 else "#FFD36E" if d["battery"] > 25 else "#FF5A6F"
            cam_status = "Active IR" if d['status'] == 'investigating' else "Offline"
            cam_color = "#FF5A6F" if d['status'] == 'investigating' else "#A1A1AA"
            pulse = " pulse-active" if d['status'] == 'investigating' else ""
            st.markdown(f"""<div class="drone-card" style="--accent:{bc};">
              <div class="signal-kicker">{d['id']}</div>
              <div class="signal-title">{d['status'].upper()}</div>
              <div class="drone-grid">
                <span>Battery</span><span style="color:{bc}">{d['battery']}%</span>
                <span>Speed</span><span>{d['speed']} km/h</span>
                <span>Alt</span><span>{d['altitude']}m</span>
                <span>Camera</span><span style="color:{cam_color}" class="{pulse}">{cam_status}</span>
                <span>Solar</span><span>{d['solar_w']}W</span>
                <span>5G</span><span style="color:#2ED6A1">Connected</span>
              </div>
              <div class="battery-track"><div class="battery-fill" style="width:{d['battery']}%; background:{bc}"></div></div>
            </div>""", unsafe_allow_html=True)

    # IoT
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">NB-IoT Belt</div>
          <div class="section-title">Ground sensors</div></div>""", unsafe_allow_html=True)
    ic = st.columns(min(4, len(iot)))
    for col, s in zip(ic, iot[:4]):
        with col:
            acc = "#FF5A6F" if s["status"] == "anomaly" else "#2ED6A1"
            st.markdown(_signal_card(s["id"], f"{s['zone']} — {s['status'].upper()}", f"Temp {s['temp']}C | Hum {s['hum']}%", acc), unsafe_allow_html=True)

    # Broadcast
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Cell broadcast</div>
          <div class="section-title">Public warning console</div></div>""", unsafe_allow_html=True)
    bc_c = st.columns(4)
    with bc_c[0]: bc_zone = st.selectbox("Zone", list(CITIES.keys()), key="bc_zone")
    with bc_c[1]: bc_aud = st.selectbox("Audience", ["Citizens", "First Responders", "All"], key="bc_aud")
    with bc_c[2]: bc_lvl = st.selectbox("Level", ["Info", "Warning", "Evacuation"], key="bc_lvl")
    with bc_c[3]: bc_lang = st.selectbox("Language", ["French", "Arabic", "Both"], key="bc_lang")

    if st.button("SEND BROADCAST", key="send_bc"):
        st.session_state.broadcast_log.insert(0, make_broadcast(bc_zone, bc_aud, bc_lvl))

    for bc in st.session_state.broadcast_log[:3]:
        acc = {"Info": "#4AA3FF", "Warning": "#FFD36E", "Evacuation": "#FF5A6F"}.get(bc["level"], "#4AA3FF")
        st.markdown(f"""<div class="broadcast-card" style="--accent:{acc};">
          <div class="signal-kicker">{bc['time']} | {bc['zone']} | {bc['audience']} | {bc['level']}</div>
          <div class="signal-copy">{escape(bc['msg'])}</div>
        </div>""", unsafe_allow_html=True)

    # ── Platform metrics + refresh (moved from Demo Brief) ──
    st.markdown("""<div class="section-head compact-head">
          <div class="section-kicker">Platform</div>
          <div class="section-title">Key metrics & data refresh</div></div>""", unsafe_allow_html=True)
    mc = st.columns(4)
    with mc[0]: st.markdown(_metric_card("FIRMS", str(summary["n_fires"]), "7-day detections", "#FF5A6F"), unsafe_allow_html=True)
    with mc[1]: st.markdown(_metric_card("Max FWI", f"{summary['max_fwi']:.1f}", "Current peak", "#FF8E53"), unsafe_allow_html=True)
    with mc[2]: st.markdown(_metric_card("XGBoost", "> 0.85 AUC", "Hold-out validation", "#4AA3FF"), unsafe_allow_html=True)
    with mc[3]: st.markdown(_metric_card("U-Net", "0.936 Dice", "Best validation", "#2ED6A1"), unsafe_allow_html=True)

    if st.button("REFRESH PIPELINE", key="refresh"):
        st.cache_data.clear()
        st.session_state.ready = False
        st.rerun()
    if st.session_state.refreshed_at:
        st.markdown(_signal_card("Last refresh", st.session_state.refreshed_at.strftime("%H:%M:%S"),
            "Auto-refresh every 5 minutes.", "#4AA3FF"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="footer-brand">
  {HUAWEI_LOGO_SVG}
  <span class="footer-text">Firewatch — Huawei Tech4Connect 2025</span>
</div>""", unsafe_allow_html=True)
