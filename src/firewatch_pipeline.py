# ============================================================
#  firewatch_pipeline.py
#  Real-time data engine: NASA FIRMS + Open-Meteo + FWI + Alerts
#  No API keys. No authentication. Offline fallback built in.
# ============================================================

import warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Region config ─────────────────────────────────────────────
BBOX = {"west": 3.50, "south": 36.20, "east": 6.50, "north": 37.20}
CITIES = {
    "Bejaia":        {"lat": 36.7525, "lon": 5.0843},
    "Tizi Ouzou":    {"lat": 36.7117, "lon": 4.0450},
    "Akbou":         {"lat": 36.4667, "lon": 4.5333},
    "Sidi Aich":     {"lat": 36.6333, "lon": 4.8500},
    "Bordj Menaiel": {"lat": 36.7400, "lon": 3.7200},
    "Bouira":        {"lat": 36.3800, "lon": 3.9000},
    "Boumerdès":     {"lat": 36.7600, "lon": 3.4700},
}

# Alert thresholds (FWI-based)
ALERT_THRESHOLDS = {
    "WATCH":     11,   # Moderate
    "WARNING":   21,   # High
    "CRITICAL":  33,   # Very High
    "EMERGENCY": 50,   # Extreme
}

CACHE = Path(__file__).resolve().parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

AUTO_REFRESH_SECONDS = 300  # 5 minutes


# ── 1. NASA FIRMS ─────────────────────────────────────────────

def fetch_firms(days: int = 7) -> pd.DataFrame:
    """Real VIIRS S-NPP fire detections for Kabylie. No key needed."""
    sensor = "VIIRS_SNPP_NRT"
    url = (f"https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
           f"{sensor.lower()}/csv/{sensor}_{days}d.csv")
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            mask = ((df.latitude  >= BBOX["south"]) & (df.latitude  <= BBOX["north"]) &
                    (df.longitude >= BBOX["west"])  & (df.longitude <= BBOX["east"]))
            fires = df[mask].copy()
            if "confidence" in fires.columns:
                fires["confidence_pct"] = (
                    fires["confidence"].map({"l":33,"n":66,"h":95}).fillna(66)
                    if fires["confidence"].dtype == object else fires["confidence"])
            if "acq_date" in fires.columns:
                fires["acq_date"] = pd.to_datetime(fires["acq_date"])
            fires["data_source"] = "NASA FIRMS — Live"
            fires.to_csv(CACHE / "firms.csv", index=False)
            return fires
    except Exception:
        pass

    cache = CACHE / "firms.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        df["data_source"] = "NASA FIRMS — Cached"
        return df

    return _demo_fires()


def _demo_fires() -> pd.DataFrame:
    np.random.seed(2024)
    hotspots = [(36.75,5.08),(36.71,4.05),(36.65,4.80),
                (36.80,5.30),(36.60,4.30),(36.85,4.95),
                (36.38,3.90),(36.76,3.47)]
    rows = []
    today = datetime.now()
    for blat, blon in hotspots:
        for _ in range(np.random.randint(3, 9)):
            rows.append({
                "latitude":       blat + np.random.normal(0, 0.04),
                "longitude":      blon + np.random.normal(0, 0.04),
                "brightness":     float(np.random.uniform(310, 420)),
                "confidence_pct": int(np.random.randint(60, 96)),
                "frp":            float(np.random.uniform(5, 120)),
                "acq_date":       today - timedelta(days=int(np.random.randint(0, 7))),
                "daynight":       np.random.choice(["D", "N"]),
                "data_source":    "Demo Data — FIRMS Unavailable",
            })
    return pd.DataFrame(rows)


# ── 2. Open-Meteo Weather ─────────────────────────────────────

def fetch_weather() -> dict:
    """
    Fetches live weather and 7-day forecasting for all monitored Kabylie cities.
    
    Data Source: Open-Meteo API (High-resolution local models).
    Architecture Note: This function operates entirely key-less and free-tier compliant, 
    making the pipeline trivial to deploy in new command centers without API credential 
    management. If the API drops, it automatically degrades to a seasonal baseline 
    (`_fallback_weather`) to ensure the dashboard never crashes during an operation.
    """
    results = {}
    for city, coords in CITIES.items():
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":        coords["lat"],
                    "longitude":       coords["lon"],
                    "current_weather": "true",
                    "hourly":  "relative_humidity_2m,precipitation,winddirection_10m",
                    "daily":   "temperature_2m_max,temperature_2m_min,"
                               "precipitation_sum,windspeed_10m_max,"
                               "relative_humidity_2m_min,relative_humidity_2m_max,"
                               "weathercode",
                    "forecast_days": 7,
                    "timezone": "Africa/Algiers",
                }, timeout=10)
            if resp.status_code == 200:
                raw = resp.json()
                cur = raw.get("current_weather", {})
                hourly = raw.get("hourly", {})
                daily  = raw.get("daily",  {})
                now_str = datetime.now().strftime("%Y-%m-%dT%H:00")
                times   = hourly.get("time", [])
                hidx    = times.index(now_str) if now_str in times else 0
                results[city] = {
                    "lat": coords["lat"], "lon": coords["lon"],
                    "current": {
                        "temp":      float(cur.get("temperature",  30)),
                        "windspeed": float(cur.get("windspeed",    15)),
                        "rh":        float(hourly.get("relative_humidity_2m", [40])[hidx]),
                        "rain":      float(hourly.get("precipitation",       [0 ])[hidx]),
                        "wcode":     int(cur.get("weathercode", 0)),
                        "is_day":    int(cur.get("is_day", 1)),
                        "wind_dir":  float(hourly.get("winddirection_10m", [0])[hidx]),
                    },
                    "forecast": {
                        "dates":    daily.get("time", []),
                        "tmax":     daily.get("temperature_2m_max",  []),
                        "tmin":     daily.get("temperature_2m_min",  []),
                        "rain":     daily.get("precipitation_sum",   []),
                        "wind_max": daily.get("windspeed_10m_max",   []),
                        "rh_min":   daily.get("relative_humidity_2m_min", []),
                        "rh_max":   daily.get("relative_humidity_2m_max", []),
                        "wcode":    daily.get("weathercode",         []),
                    },
                    "source": "Open-Meteo — Live",
                }
                continue
        except Exception:
            pass
        results[city] = _fallback_weather(coords)
    return results


def _fallback_weather(coords: dict) -> dict:
    m = datetime.now().month
    summer = 6 <= m <= 9
    return {
        "lat": coords["lat"], "lon": coords["lon"],
        "current": {"temp": 36 if summer else 18, "windspeed": 22,
                    "rh": 22 if summer else 60, "rain": 0, "wcode": 0, "is_day": 1,
                    "wind_dir": 315.0},
        "forecast": {"dates":[],"tmax":[],"tmin":[],"rain":[],
                     "wind_max":[],"rh_min":[],"rh_max":[],"wcode":[]},
        "source": "Fallback — Offline Estimate",
    }


# ── 3. Canadian FWI (Van Wagner 1987) ─────────────────────────

def compute_fwi(temp: float, rh: float, wind: float, rain: float,
                ffmc0: float = 85.0, dmc0: float = 6.0, dc0: float = 15.0) -> dict:
    """
    Computes the full Canadian Forest Fire Weather Index (FWI) System (Van Wagner 1987).
    
    This function calculates all 6 standard components of the FWI system:
      1. FFMC (Fine Fuel Moisture Code): Tracks moisture in surface litter.
      2. DMC (Duff Moisture Code): Tracks moisture in loosely compacted organic layers.
      3. DC (Drought Code): Tracks moisture in deep, compact organic layers.
      4. ISI (Initial Spread Index): Combines wind and FFMC to predict fire spread rate.
      5. BUI (Buildup Index): Combines DMC and DC to represent total fuel available.
      6. FWI (Fire Weather Index): Combines ISI and BUI into a final intensity rating.

    Args:
        temp:  Temperature (°C)
        rh:    Relative humidity (%)
        wind:  Wind speed (km/h) at 10m height
        rain:  24-hour precipitation (mm)
        ffmc0, dmc0, dc0: Previous day's indices (using standardized mid-range defaults 
                          to provide instantaneous risk assessment without requiring 
                          continuous historical data tracking).

    Returns:
        Dictionary containing all 6 calculated indices.
    """
    # FFMC
    mo = 147.2*(101-ffmc0)/(59.5+ffmc0)
    if rain > 0.5:
        rf = rain-0.5
        mo = min(mo + 42.5*rf*np.exp(-100/(251-mo))*(1-np.exp(-6.93/rf)), 250.0)
    ed = 0.942*rh**0.679 + 11*np.exp((rh-100)/10) + 0.18*(21.1-temp)*(1-np.exp(-0.115*rh))
    ew = 0.618*rh**0.753 + 10*np.exp((rh-100)/10) + 0.18*(21.1-temp)*(1-np.exp(-0.115*rh))
    if mo > ed:
        kd = (0.424*(1-(rh/100)**1.7) + 0.0694*wind**0.5*(1-(rh/100)**8)) * 0.581*np.exp(0.0365*temp)
        m  = ed + (mo-ed)*10**(-kd)
    else:
        kw = (0.424*(1-((100-rh)/100)**1.7) + 0.0694*wind**0.5*(1-((100-rh)/100)**8)) * 0.581*np.exp(0.0365*temp)
        m  = ew - (ew-mo)*10**(-kw)
    ffmc = 59.5*(250-m)/(147.2+m)

    # DMC
    if rain > 1.5:
        re  = 0.92*rain-1.27
        mo2 = 20+np.exp(5.6348-dmc0/43.43)
        b   = 100/(0.5+0.3*dmc0) if dmc0<=33 else 14-1.3*np.log(dmc0) if dmc0<=65 else 6.2*np.log(dmc0)-17.2
        dmc0 = max(244.72-43.43*np.log(mo2+1000*re/(48.77+b*re)-20), 0)
    dl  = [6.5,7.5,9,12.5,13.5,13.5,12.5,11,9,7.5,6.5,5.5][datetime.now().month-1]
    dmc = dmc0 + 100*max(0, 1.894*(temp+1.1)*(100-rh)*dl*1e-6)

    # DC
    if rain > 2.8:
        qr  = 800*np.exp(-dc0/400) + 3.937*(0.83*rain-1.27)
        dc0 = max(400*np.log(800/qr), 0)
    lf  = [-1.6,-1.6,-1.6,0.9,3.8,5.8,6.4,5.0,2.4,0.4,-1.6,-1.6][datetime.now().month-1]
    dc  = dc0 + 0.5*max(0, 0.36*(temp+2.8)+lf)

    # ISI, BUI, FWI
    fw  = np.exp(0.05039*wind)
    ff  = 91.9*np.exp(-0.1386*m)*(1+(m**5.31)/4.93e7)
    isi = 0.208*fw*ff
    bui = (0.8*dmc*dc/(dmc+0.4*dc) if dmc <= 0.4*dc
           else dmc-(1-0.8*dc/(dmc+0.4*dc))*(0.92+(0.0114*dmc)**1.7))
    bb  = (0.1*isi*(1000/(25+108.64*np.exp(-0.023*bui))) if bui > 80
           else 0.1*isi*(0.626*bui**0.809+2))
    fwi = np.exp(2.72*(0.434*np.log(bb))**0.647) if bb > 1 else bb

    return {"ffmc":round(ffmc,2),"dmc":round(dmc,2),"dc":round(dc,2),
            "isi":round(isi,2),"bui":round(bui,2),"fwi":round(fwi,2)}


def fwi_risk(fwi: float) -> tuple:
    """Return (label, color, action_text) for the given FWI value."""
    if fwi < 5:  return "Very Low",  "#27ae60", "No action required"
    if fwi < 11: return "Low",       "#2ecc71", "Monitor conditions"
    if fwi < 21: return "Moderate",  "#f39c12", "Alert forest brigades"
    if fwi < 33: return "High",      "#e67e22", "Pre-position firefighting resources"
    if fwi < 50: return "Very High", "#e74c3c", "Maximum alert — evacuate risk areas"
    return              "Extreme",   "#8e44ad", "EMERGENCY — all resources mobilized"


def compute_all_risks(weather: dict) -> dict:
    """Compute FWI and risk for every city from live weather data."""
    out = {}
    for city, w in weather.items():
        c = w["current"]
        idx = compute_fwi(c["temp"], c["rh"], c["windspeed"], c["rain"])
        lvl, col, act = fwi_risk(idx["fwi"])
        out[city] = {**idx, "risk_level":lvl, "risk_color":col, "action":act,
                     "lat":w["lat"], "lon":w["lon"],
                     "temp":c["temp"], "rh":c["rh"], "wind":c["windspeed"], "rain":c["rain"],
                     "wind_dir":c.get("wind_dir", 315)}
    return out


# ── 4. Forecast FWI (uses real humidity) ──────────────────────

def compute_forecast_fwi(weather: dict, city: str) -> list:
    """Compute 7-day projected FWI for a city using ACTUAL forecast humidity
    from Open-Meteo (not estimated). Returns list of dicts with date, fwi, risk."""
    if city not in weather:
        return []
    fc = weather[city]["forecast"]
    dates = fc.get("dates", [])
    if not dates:
        return []

    projections = []
    n = min(7, len(dates))
    for i in range(n):
        tmax   = fc["tmax"][i]     if fc.get("tmax")     and i < len(fc["tmax"])     else 30
        wmax   = fc["wind_max"][i] if fc.get("wind_max") and i < len(fc["wind_max"]) else 15
        rain   = fc["rain"][i]     if fc.get("rain")     and i < len(fc["rain"])     else 0

        # Use real forecast humidity from Open-Meteo (average of min/max)
        rh_min = fc["rh_min"][i] if fc.get("rh_min") and i < len(fc["rh_min"]) else None
        rh_max = fc["rh_max"][i] if fc.get("rh_max") and i < len(fc["rh_max"]) else None

        if rh_min is not None and rh_max is not None:
            # Use midday humidity estimate (closer to min for fire danger)
            est_rh = rh_min * 0.6 + rh_max * 0.4  # weight toward the drier part of the day
        else:
            # Fallback only if API didn't provide humidity
            est_rh = max(15, 65 - (tmax - 20) * 1.5)

        idx = compute_fwi(tmax, est_rh, wmax, rain)
        lvl, col, act = fwi_risk(idx["fwi"])
        projections.append({
            "date": dates[i],
            "risk_level": lvl, "risk_color": col,
            "tmax": tmax, "rain": rain,
            "wind": wmax, "rh": round(est_rh, 1),
            **idx,
        })
    return projections


# ── 5. Alert System ───────────────────────────────────────────

def check_alerts(risk: dict) -> list:
    """Evaluate all cities against FWI thresholds and generate alerts.
    Returns list of alert dicts sorted by severity (worst first)."""
    alerts = []
    severity_order = {"EMERGENCY": 0, "CRITICAL": 1, "WARNING": 2, "WATCH": 3}

    for city, r in risk.items():
        fwi_val = r["fwi"]
        if fwi_val >= ALERT_THRESHOLDS["EMERGENCY"]:
            severity = "EMERGENCY"
        elif fwi_val >= ALERT_THRESHOLDS["CRITICAL"]:
            severity = "CRITICAL"
        elif fwi_val >= ALERT_THRESHOLDS["WARNING"]:
            severity = "WARNING"
        elif fwi_val >= ALERT_THRESHOLDS["WATCH"]:
            severity = "WATCH"
        else:
            continue  # no alert

        alerts.append({
            "city": city,
            "severity": severity,
            "fwi": fwi_val,
            "risk_level": r["risk_level"],
            "risk_color": r["risk_color"],
            "action": r["action"],
            "temp": r["temp"],
            "rh": r["rh"],
            "wind": r["wind"],
            "timestamp": datetime.now().isoformat(),
            "icon": {"EMERGENCY":"🚨","CRITICAL":"🔴","WARNING":"🟠","WATCH":"🟡"}[severity],
        })

    alerts.sort(key=lambda a: severity_order.get(a["severity"], 99))
    return alerts


# ── 6. Orchestrator ───────────────────────────────────────────

def run_pipeline(use_gee: bool = False) -> dict:
    """Run the full pipeline: fires + weather + risk + alerts.
    Returns a dict with all data needed by the dashboard."""
    fires   = fetch_firms()
    weather = fetch_weather()
    risk    = compute_all_risks(weather)
    alerts  = check_alerts(risk)
    imagery = None

    if use_gee:
        try:
            import ee, geemap
            ee.Initialize()
            imagery = _fetch_gee()
        except Exception as e:
            imagery = {"error": str(e)}

    max_fwi       = max((r["fwi"] for r in risk.values()), default=0)
    lvl, col, act = fwi_risk(max_fwi)
    return {
        "fires": fires, "weather": weather, "risk": risk,
        "alerts": alerts, "imagery": imagery,
        "summary": {
            "n_fires":    len(fires),
            "max_fwi":    max_fwi,
            "risk_level": lvl, "risk_color": col, "action": act,
            "n_alerts":   len(alerts),
            "worst_alert": alerts[0]["severity"] if alerts else "NONE",
            "firms_live": "Live" in fires.get("data_source", pd.Series([""])).iloc[0] if len(fires) else False,
            "timestamp":  datetime.now().isoformat(),
        },
    }


def should_refresh(last_refresh: datetime) -> bool:
    """Check if enough time has passed for an auto-refresh."""
    if last_refresh is None:
        return True
    return (datetime.now() - last_refresh).total_seconds() >= AUTO_REFRESH_SECONDS


def _fetch_gee() -> dict:
    import ee, geemap
    from pathlib import Path
    out = Path("data/imagery"); out.mkdir(parents=True, exist_ok=True)
    region = ee.Geometry.Rectangle([BBOX["west"],BBOX["south"],BBOX["east"],BBOX["north"]])
    end    = datetime.now().strftime("%Y-%m-%d")
    start  = (datetime.now()-timedelta(days=30)).strftime("%Y-%m-%d")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .sort("CLOUDY_PIXEL_PERCENTAGE").first())
    paths = {}
    for name, bands in [("rgb",["B4","B3","B2"]),("swir",["B12","B8","B4"])]:
        p = out/f"kabylie_{name}.tif"
        geemap.ee_export_image(s2.select(bands), filename=str(p),
                               scale=100, region=region, file_per_band=False)
        paths[name] = str(p)
    try:
        paths["image_date"] = ee.Date(s2.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    except Exception:
        paths["image_date"] = "Recent"
    return paths
