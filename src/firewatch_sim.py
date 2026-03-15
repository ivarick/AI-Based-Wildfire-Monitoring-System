# ============================================================
#  firewatch_sim.py — Simulated data generators
#  Drone fleet, IoT sensors, satellites, SMS broadcast
# ============================================================
import numpy as np
from datetime import datetime, timedelta

rand = lambda lo, hi: np.random.uniform(lo, hi)
rint = lambda lo, hi: int(np.random.randint(lo, hi))

# ── Drone Fleet ───────────────────────────────────────────────
def gen_drones(n=12):
    statuses = ["investigating", "returning", "standby"]
    return [{"id": f"DRN-{i+1:03d}",
             "battery": rint(15,99), "altitude": 0 if i >= 5 else rint(80,450),
             "speed": 0 if i >= 5 else rint(30,65), "signal": rint(70,100),
             "status": statuses[0 if i<2 else 1 if i<5 else 2],
             "solar_eff": round(rand(0.7,0.95),2),
             "solar_w": round(rand(50,200),1),
             "charging": bool(rint(0,2)) if i >= 5 else False,
             "lat": round(36.5+rand(0,0.8),4),
             "lon": round(3.5+rand(0,4.5),4)} for i in range(n)]

# ── IoT Ground Sensors ───────────────────────────────────────
def gen_iot_sensors():
    return [
        {"id":"IOT-01","lat":36.70,"lon":4.10,"temp":48,"hum":15,"status":"anomaly","zone":"Tizi Ouzou"},
        {"id":"IOT-02","lat":36.85,"lon":6.80,"temp":42,"hum":22,"status":"anomaly","zone":"Skikda"},
        {"id":"IOT-03","lat":36.75,"lon":4.50,"temp":28,"hum":45,"status":"normal","zone":"Bejaia"},
        {"id":"IOT-04","lat":36.80,"lon":6.20,"temp":31,"hum":40,"status":"normal","zone":"Jijel"},
        {"id":"IOT-05","lat":36.38,"lon":3.90,"temp":26,"hum":55,"status":"normal","zone":"Bouira"},
        {"id":"IOT-06","lat":36.76,"lon":3.47,"temp":44,"hum":18,"status":"anomaly","zone":"Boumerdes"},
        {"id":"IOT-07","lat":36.65,"lon":4.80,"temp":29,"hum":42,"status":"normal","zone":"Akbou"},
        {"id":"IOT-08","lat":36.72,"lon":5.20,"temp":33,"hum":35,"status":"normal","zone":"Sidi Aich"},
    ]

# ── Satellite Sources ────────────────────────────────────────
def gen_satellites():
    now = datetime.now()
    def spark(n=12, base=50, var=20):
        return [max(0, base+rand(-var,var)) for _ in range(n)]
    return [
        {"id":"S2-MSI","label":"Sentinel-2 MSI","desc":"Optical - 10m","status":"green",
         "last_pass":now-timedelta(minutes=12),"extra":"Cloud: 8%","sparkline":spark(12,70,15)},
        {"id":"MSG-SEV","label":"MSG/SEVIRI","desc":"Geostationary - 15min","status":"green",
         "last_pass":now-timedelta(minutes=4),"extra":"FRP: 2847 MW","sparkline":spark(12,55,25)},
        {"id":"VIIRS","label":"VIIRS S-NPP","desc":"Thermal - 375m","status":"yellow",
         "last_pass":now-timedelta(minutes=38),"extra":"Active px: 127","sparkline":spark(12,40,30)},
        {"id":"S1-SAR","label":"Sentinel-1 SAR","desc":"SAR - Cloud-penetrating","status":"green",
         "last_pass":now-timedelta(hours=2),"extra":"Coherence: 0.74","sparkline":spark(12,60,10)},
        {"id":"PLANET","label":"Planet SuperDove","desc":"On-demand - 3m","status":"yellow",
         "last_pass":now-timedelta(hours=6),"extra":"Tasked","sparkline":spark(12,30,15)},
        {"id":"IOT","label":"IoT Ground Sensors","desc":"48 nodes - Forest belt","status":"green",
         "last_pass":now-timedelta(seconds=45),"extra":"45/48 Active","sparkline":spark(12,80,12)},
    ]

# ── Broadcast log ────────────────────────────────────────────
def make_broadcast(zone, audience, level, lang="Both"):
    # Base instructions based on severity level
    safety = {
        "Info": "Monitor official channels for updates. Avoid open flames in forested areas.",
        "Warning": "Prepare for possible evacuation. Secure livestock and valuables. Keep emergency supplies ready.",
        "Evacuation": "EVACUATE IMMEDIATELY. Move away from forested areas. Follow designated evacuation routes. Contact emergency services if trapped.",
    }.get(level, "")
    
    # Construct bilingual emergency message
    if lang == "French":
        msg = f"ALERTE INCENDIE ({level.upper()}) — Zone: {zone}. Risque d'incendie détecté dans la région de {zone}. {safety}"
    elif lang == "Arabic":
        msg = f"تحذير من حريق ({level.upper()}) — المنطقة: {zone}. تم اكتشاف خطر حريق في منطقة {zone}. {safety}"
    else: # Both
        msg = (
            f"ALERTE INCENDIE ({level.upper()}) — Zone: {zone}. Risque d'incendie détecté dans la région de {zone}. {safety}\n\n"
            f"تحذير من حريق ({level.upper()}) — المنطقة: {zone}. تم اكتشاف خطر حريق في منطقة {zone}. {safety}"
        )

    return {"id": int(datetime.now().timestamp()*1000),
            "zone": zone, "audience": audience, "level": level, "lang": lang,
            "time": datetime.now().strftime("%H:%M:%S"),
            "msg": msg}

# ── Fire spread predictions ──────────────────────────────────
def gen_fire_spread(fires_df):
    """Generate 1h/3h/6h fire area predictions from current detections."""
    if fires_df is None or len(fires_df) == 0:
        return {"1h":[],"3h":[],"6h":[]}
    # Group by proximity into clusters
    clusters = []
    if "frp" in fires_df.columns:
        top = fires_df.nlargest(min(5, len(fires_df)), "frp")
    else:
        top = fires_df.head(5)
    for i, (_, r) in enumerate(top.iterrows()):
        area = float(r.get("frp", 50)) * 0.8  # approximate hectares from FRP
        clusters.append({"id": f"F-{i+1:03d}", "area": round(area,1),
                         "lat": float(r["latitude"]), "lon": float(r["longitude"])})
    return {
        "1h": [{"id":c["id"],"area":round(c["area"]*1.15,1)} for c in clusters],
        "3h": [{"id":c["id"],"area":round(c["area"]*1.45,1)} for c in clusters],
        "6h": [{"id":c["id"],"area":round(c["area"]*1.90,1)} for c in clusters],
    }
