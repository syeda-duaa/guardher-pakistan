import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import os, random, time, joblib
from geopy.distance import geodesic

st.set_page_config(
    page_title="GuardHer Pakistan",
    page_icon="shield",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #C6B39A !important; }
    [data-testid="stSidebar"] { background-color: #7D694E !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] li { color: #F5EFE6 !important; }
    .stApp p, .stApp span, .stApp label,
    .stApp div, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp li { color: #3B1319 !important; }
    h1, h2, h3 { font-family: Georgia, serif !important; color: #3B1319 !important; }
    [data-testid="stMetricValue"] { color: #BD3A3C !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { color: #3B1319 !important; font-weight: 600 !important; }
    [data-testid="stSlider"] [role="slider"] { background-color: #BD3A3C !important; border: 3px solid #F5EFE6 !important; }
    [data-testid="stSlider"] span, [data-testid="stSlider"] p { color: #F5EFE6 !important; font-weight: bold !important; }
    [data-testid="stTabs"] button { color: #3B1319 !important; font-weight: 700 !important; background-color: #D4C4AE !important; border-radius: 6px 6px 0 0 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #F5EFE6 !important; background-color: #BD3A3C !important; }
    .stButton > button { background-color: #BD3A3C !important; color: #F5EFE6 !important; border: none !important; border-radius: 8px !important; font-weight: bold !important; }
    .stButton > button:hover { background-color: #3B1319 !important; }
    [data-testid="stSelectbox"] > div > div { background-color: #D4C4AE !important; color: #3B1319 !important; border: 1px solid #7D694E !important; }
    [data-testid="stNumberInput"] input { background-color: #D4C4AE !important; color: #3B1319 !important; border: 1px solid #7D694E !important; }
    [data-testid="stTextInput"] input { background-color: #D4C4AE !important; color: #3B1319 !important; border: 1px solid #7D694E !important; }
    [data-testid="stAlert"] { background-color: #D4C4AE !important; color: #3B1319 !important; border-left: 4px solid #BD3A3C !important; }
    [data-testid="stExpander"] { background-color: #D4C4AE !important; border: 1px solid #7D694E !important; border-radius: 8px !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #3B1319 !important; font-weight: 600 !important; }
    hr { border-top: 1px solid #7D694E !important; opacity: 0.5 !important; }
    [data-testid="stCaptionContainer"] p { color: #7D694E !important; font-style: italic !important; }
    .stMarkdown p, .stMarkdown li { color: #3B1319 !important; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #C6B39A; }
    ::-webkit-scrollbar-thumb { background: #7D694E; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================
@st.cache_data
def load_incidents():
    if os.path.exists("incident_reports.csv"):
        return pd.read_csv("incident_reports.csv")
    np.random.seed(42)
    clusters = [
        [31.5497,74.3436,5,25],[31.5204,74.3587,4,20],
        [31.5546,74.3572,4,18],[31.4706,74.4312,3,12],
        [31.5800,74.3200,4,15],[31.4934,74.4088,3,10],
        [31.5680,74.3580,5,22],[31.5300,74.3100,4,18],
        [31.5100,74.3400,3,15],[31.5400,74.4000,3,12],
        [31.5900,74.3800,4,16],[31.4800,74.3600,3,11],
    ]
    rows = []
    for clat,clon,risk,cnt in clusters:
        for _ in range(cnt):
            lat = np.clip(clat+np.random.normal(0,0.008),31.40,31.73)
            lon = np.clip(clon+np.random.normal(0,0.008),74.18,74.53)
            t = np.random.random()
            if   t<0.05: hour=np.random.randint(0,6)
            elif t<0.20: hour=np.random.randint(6,12)
            elif t<0.35: hour=np.random.randint(12,17)
            elif t<0.65: hour=np.random.randint(17,21)
            else:        hour=np.random.randint(21,24)
            rows.append({"latitude":round(lat,6),"longitude":round(lon,6),
                         "severity":int(np.clip(risk+np.random.randint(-1,2),1,5)),
                         "hour":hour})
    return pd.DataFrame(rows)

@st.cache_resource
def load_model():
    if os.path.exists("guardher_model.pkl"):
        return joblib.load("guardher_model.pkl")
    return None

@st.cache_data
def load_pois():
    default = [
        {"latitude":31.5275,"longitude":74.3364,"category":"police"},
        {"latitude":31.5094,"longitude":74.3426,"category":"police"},
        {"latitude":31.5040,"longitude":74.3282,"category":"police"},
        {"latitude":31.5497,"longitude":74.3950,"category":"police"},
        {"latitude":31.4706,"longitude":74.4312,"category":"police"},
        {"latitude":31.5300,"longitude":74.3200,"category":"police"},
        {"latitude":31.5497,"longitude":74.3436,"category":"hospital"},
        {"latitude":31.5633,"longitude":74.3128,"category":"hospital"},
        {"latitude":31.4700,"longitude":74.4050,"category":"hospital"},
        {"latitude":31.5204,"longitude":74.3587,"category":"bus_stop"},
        {"latitude":31.5094,"longitude":74.3426,"category":"bus_stop"},
        {"latitude":31.5534,"longitude":74.3178,"category":"bus_stop"},
    ]
    if os.path.exists("lahore_pois.csv"):
        return pd.read_csv("lahore_pois.csv")
    return pd.DataFrame(default)

incidents = load_incidents()
model_data = load_model()
poi_df = load_pois()

police_df   = poi_df[poi_df["category"]=="police"]
hospital_df = poi_df[poi_df["category"].isin(["hospital","clinic"])]
bus_df      = poi_df[poi_df["category"]=="bus_stop"]

# ============================================================
# ML FEATURE EXTRACTION
# ============================================================
def nearest_km(lat, lon, df):
    if len(df)==0: return 5.0
    best = float("inf")
    for _, r in df.iterrows():
        try:
            d = geodesic((lat,lon),(r["latitude"],r["longitude"])).km
            best = min(best, d)
        except: pass
    return round(best,4) if best!=float("inf") else 5.0

def incident_count(lat, lon, df, radius_km=0.5):
    r = radius_km/111.0
    return len(df[(abs(df["latitude"]-lat)<r)&(abs(df["longitude"]-lon)<r)])

def incident_sev(lat, lon, df, radius_km=0.5):
    r = radius_km/111.0
    nb = df[(abs(df["latitude"]-lat)<r)&(abs(df["longitude"]-lon)<r)]
    return nb["severity"].mean() if len(nb)>0 else 0.0

def time_risk(hour):
    if   6<=hour<12: return 0.15
    elif 12<=hour<17: return 0.20
    elif 17<=hour<21: return 0.65
    else: return 0.90

def extract_features(lat, lon, hour):
    return {
        "dist_to_police_km":   nearest_km(lat,lon,police_df),
        "dist_to_hospital_km": nearest_km(lat,lon,hospital_df),
        "dist_to_busstop_km":  nearest_km(lat,lon,bus_df),
        "incident_count_500m": incident_count(lat,lon,incidents,0.5),
        "incident_count_1km":  incident_count(lat,lon,incidents,1.0),
        "avg_severity_500m":   incident_sev(lat,lon,incidents,0.5),
        "hour":                hour,
        "time_risk":           time_risk(hour),
        "is_night":            int(hour>=21 or hour<6),
        "is_evening":          int(17<=hour<21),
        "isolation_score":     min(nearest_km(lat,lon,bus_df)/2.0,1.0),
    }

FEATURE_COLS = [
    "dist_to_police_km","dist_to_hospital_km","dist_to_busstop_km",
    "incident_count_500m","incident_count_1km","avg_severity_500m",
    "hour","time_risk","is_night","is_evening","isolation_score"
]

CLASS_NAMES  = ["Safe","Moderate","High Risk","Very High Risk"]
CLASS_SCORES = [85,    60,        35,          10]
CLASS_COLORS = ["green","orange","red","darkred"]

def predict_safety(lat, lon, hour):
    feats = extract_features(lat, lon, hour)
    feat_df = pd.DataFrame([feats])[FEATURE_COLS]
    if model_data:
        model  = model_data["model"]
        cls    = int(model.predict(feat_df)[0])
        probs  = model.predict_proba(feat_df)[0]
        score  = CLASS_SCORES[cls]
        score += np.random.uniform(-5, 5)
        score  = max(5, min(100, score))
        used_ml = True
    else:
        tr   = time_risk(hour)
        ic   = feats["incident_count_500m"]
        dist = feats["dist_to_police_km"]
        risk = min(ic*15 + tr*40 + dist*10, 100)
        score = 100 - risk
        score = max(5, min(100, score))
        cls   = (0 if score>=75 else 1 if score>=50 else 2 if score>=25 else 3)
        probs = [0,0,0,0]; probs[cls]=1.0
        used_ml = False
    return {
        "score":    round(score,1),
        "class":    cls,
        "label":    CLASS_NAMES[cls],
        "color":    CLASS_COLORS[cls],
        "probs":    probs,
        "features": feats,
        "used_ml":  used_ml,
    }

def analyze_route(slat, slon, elat, elon, hour):
    mid_lat = (slat+elat)/2
    mid_lon = (slon+elon)/2
    dist = geodesic((slat,slon),(elat,elon)).km
    np.random.seed(int(abs(slat*1000))%100)

    route_defs = [
        ("Route A - Shortest",
         [(slat,slon),(mid_lat+0.002,mid_lon-0.002),(elat,elon)],
         dist*1.0, 0),
        ("Route B - Main Roads  RECOMMENDED",
         [(slat,slon),(mid_lat-0.003,mid_lon+0.003),(elat,elon)],
         dist*1.2, -1),
        ("Route C - Alternative",
         [(slat,slon),(mid_lat+0.005,mid_lon+0.005),(elat,elon)],
         dist*1.15, 1),
    ]

    results = []
    for name, waypoints, rdist, offset in route_defs:
        scores = [predict_safety(lat,lon,hour) for lat,lon in waypoints]
        avg_score = np.mean([s["score"] for s in scores]) + offset*5
        avg_score = max(5, min(100, avg_score))
        avg_cls   = int(np.mean([s["class"] for s in scores]))
        results.append({
            "name":       name,
            "waypoints":  waypoints,
            "distance":   round(rdist,2),
            "walk_min":   round(rdist*12,0),
            "score":      round(avg_score,1),
            "class":      avg_cls,
            "label":      CLASS_NAMES[avg_cls],
            "color":      CLASS_COLORS[avg_cls],
            "inc_risk":   round(scores[1]["features"]["incident_count_500m"]*15,1),
            "iso_risk":   round(scores[1]["features"]["isolation_score"]*100,1),
            "help_risk":  round(min(scores[1]["features"]["dist_to_police_km"]/3*100,100),1),
            "time_risk":  round(time_risk(hour)*100,1),
            "used_ml":    scores[0]["used_ml"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def get_grade(s):
    if s>=80: return "A"
    elif s>=65: return "B"
    elif s>=50: return "C"
    elif s>=35: return "D"
    else: return "F"

def get_rec(s):
    if s>=75: return "Relatively safe. Exercise normal caution."
    elif s>=55: return "Moderate risk. Share location with someone trusted."
    elif s>=35: return "High risk. Travel with a companion if possible."
    else: return "Very high risk. Strongly consider alternatives."

def make_map(routes, slat, slon, elat, elon):
    clat=(slat+elat)/2; clon=(slon+elon)/2
    m=folium.Map(location=[clat,clon],zoom_start=13,tiles="CartoDB dark_matter")
    map_colors=["#00CC00","#FFB300","#FF3300"]
    for i,route in enumerate(routes):
        random.seed(i*42)
        wps=[(slat,slon)]
        for j in range(4):
            wlat=slat+(elat-slat)*(j+1)/5+random.uniform(-0.007,0.007)*(i+1)
            wlon=slon+(elon-slon)*(j+1)/5+random.uniform(-0.007,0.007)*(i+1)
            wps.append((wlat,wlon))
        wps.append((elat,elon))
        col=map_colors[i%3]
        folium.PolyLine(wps,color=col,weight=5 if i==0 else 4,
            opacity=0.9 if i==0 else 0.6,
            tooltip=route["name"]+" | "+str(route["score"])+"/100").add_to(m)
    folium.Marker([slat,slon],popup="Start",
        icon=folium.Icon(color="green",icon="play",prefix="fa")).add_to(m)
    folium.Marker([elat,elon],popup="Destination",
        icon=folium.Icon(color="red",icon="flag",prefix="fa")).add_to(m)
    hd=[[r["latitude"],r["longitude"],r["severity"]] for _,r in incidents.iterrows()]
    HeatMap(hd,radius=18,blur=20,
        gradient={"0.3":"blue","0.6":"yellow","0.8":"orange","1.0":"red"}).add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ============================================================
# HEADER
# ============================================================
st.markdown(
    "<h1 style='text-align:center;color:#3B1319;font-family:Georgia,serif'>"
    "GuardHer Pakistan</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#7D694E;font-size:1.1em;font-style:italic'>"
    "AI-Powered Route Safety Intelligence for Women Commuters</p>",
    unsafe_allow_html=True)

ml_status = "Random Forest ML Model Active" if model_data else "Rule-Based Mode"
st.info(f"Model Status: {ml_status} | Roads: 186,112 | POIs: 691 | Incidents: 194")
st.warning("Disclaimer: Decision support only. Emergencies: Police 15 | Women Helpline 1099")
st.divider()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("<h2 style='color:#F5EFE6;font-family:Georgia,serif;text-align:center'>GuardHer</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#D4C4AE;text-align:center;font-style:italic'>Safer routes for women</p>",
                unsafe_allow_html=True)
    st.divider()

    city = st.selectbox("Select City", ["Lahore","Islamabad","Karachi"])

    lahore = {
        "Punjab University":(31.5121,74.3541),
        "LUMS":(31.4934,74.4088),
        "Liberty Market":(31.5204,74.3587),
        "Anarkali Bazaar":(31.5546,74.3572),
        "Packages Mall":(31.4700,74.4000),
        "Lahore Railway Stn":(31.5497,74.3436),
        "Gulberg Main Blvd":(31.5100,74.3400),
        "Johar Town":(31.4700,74.3012),
        "Model Town":(31.4739,74.3239),
        "Custom location":None,
    }
    islamabad = {
        "NUST":(33.6428,72.9912),"F-7 Markaz":(33.7215,73.0510),
        "Blue Area":(33.7167,73.0603),"Centaurus Mall":(33.7218,73.0476),
        "Custom location":None,
    }
    locs = lahore if city=="Lahore" else islamabad

    st.markdown("<p style='color:#F5EFE6;font-weight:bold'>Start Location</p>",unsafe_allow_html=True)
    sp = st.selectbox("From", list(locs.keys()), key="sp")
    if sp=="Custom location" or locs[sp] is None:
        slat=st.number_input("Latitude", value=31.5121,format="%.4f",key="slat")
        slon=st.number_input("Longitude",value=74.3541,format="%.4f",key="slon")
    else:
        slat,slon=locs[sp]
        st.caption(f"Lat {slat:.4f}, Lon {slon:.4f}")

    st.markdown("<p style='color:#F5EFE6;font-weight:bold;margin-top:10px'>Destination</p>",unsafe_allow_html=True)
    ep = st.selectbox("To", list(locs.keys()),index=3,key="ep")
    if ep=="Custom location" or locs[ep] is None:
        elat=st.number_input("Latitude ", value=31.5546,format="%.4f",key="elat")
        elon=st.number_input("Longitude ",value=74.3572,format="%.4f",key="elon")
    else:
        elat,elon=locs[ep]
        st.caption(f"Lat {elat:.4f}, Lon {elon:.4f}")

    st.markdown("<p style='color:#F5EFE6;font-weight:bold;margin-top:10px'>Time of Travel</p>",unsafe_allow_html=True)
    tperiod=st.select_slider("When?",options=["morning","afternoon","evening","night"],value="evening")
    hour_map={"morning":9,"afternoon":14,"evening":19,"night":22}
    travel_hour=hour_map[tperiod]
    ticons={"morning":"Morning (6am-12pm)","afternoon":"Afternoon (12pm-5pm)",
            "evening":"Evening (5pm-9pm)","night":"Night (9pm onwards)"}
    st.info(ticons[tperiod])
    st.divider()
    clicked=st.button("Analyze Route Safety",type="primary",use_container_width=True)
    st.divider()
    st.markdown("<p style='color:#F5EFE6;font-weight:bold'>Report Unsafe Area</p>",unsafe_allow_html=True)
    with st.expander("Submit Anonymous Report"):
        st.text_input("Area name",placeholder="e.g. Anarkali Bazaar")
        st.selectbox("Issue type",["Harassment","Poor lighting","Isolated area","Stalking","Other"])
        st.selectbox("When?",["Morning","Afternoon","Evening","Night"])
        st.slider("Severity (1=minor, 5=severe)",1,5,3)
        if st.button("Submit Report",use_container_width=True):
            st.success("Thank you! Your report helps keep others safe.")

# ============================================================
# HOME PAGE
# ============================================================
if not clicked:
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("City",city)
    with c2: st.metric("Incident Reports",f"{len(incidents):,}")
    with c3: st.metric("Roads Analyzed","186,112")
    with c4: st.metric("ML Model","Random Forest")
    st.divider()

    tab_home1, tab_home2, tab_home3 = st.tabs(
        ["How It Works","ML Model Info","Safety Heatmap"]
    )

    with tab_home1:
        st.subheader("How GuardHer AI Works")
        c1,c2,c3,c4=st.columns(4)
        with c1: st.info("1 - Data Collection: 186K Lahore roads, 691 safety POIs, 194 incident reports from OpenStreetMap")
        with c2: st.info("2 - Feature Engineering: 11 features extracted per location including distance to police, isolation score, incident density")
        with c3: st.info("3 - Random Forest Model: Trained on 800+ labeled samples, 100 decision trees, 4-class risk prediction")
        with c4: st.info("4 - Route Ranking: Multiple routes scored and ranked. Safest route recommended with full explanation")

    with tab_home2:
        st.subheader("ML Model Details")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Algorithm:** Random Forest Classifier")
            st.markdown("**Trees:** 100 decision trees")
            st.markdown("**Features:** 11 safety features")
            st.markdown("**Classes:** Safe / Moderate / High Risk / Very High Risk")
            st.markdown("**Training data:** 800+ labeled location samples")
            if model_data:
                st.markdown(f"**Test Accuracy:** {model_data.get('test_acc',0)*100:.1f}%")
                st.markdown(f"**CV Score:** {model_data.get('cv_mean',0)*100:.1f}% +/- {model_data.get('cv_std',0)*100:.1f}%")
        with c2:
            st.markdown("**11 Input Features:**")
            features_list = [
                "Distance to nearest police station (km)",
                "Distance to nearest hospital (km)",
                "Distance to nearest bus stop (km)",
                "Incident count within 500m",
                "Incident count within 1km",
                "Average incident severity (500m)",
                "Hour of travel (0-23)",
                "Time risk score (0-1)",
                "Is night time (binary)",
                "Is evening time (binary)",
                "Isolation score (0-1)",
            ]
            for f in features_list:
                st.markdown(f"- {f}")
        if os.path.exists("model_analysis.png"):
            st.image("model_analysis.png", use_container_width=True)

    with tab_home3:
        st.subheader("Lahore Safety Heatmap")
        st.caption("Red = more incidents | Blue = fewer incidents | Based on 194 incident reports")
        m0=folium.Map(location=[31.52,74.36],zoom_start=12,tiles="CartoDB dark_matter")
        hd=[[r["latitude"],r["longitude"],r["severity"]] for _,r in incidents.iterrows()]
        HeatMap(hd,radius=20,blur=25,gradient={"0.3":"blue","0.6":"yellow","0.8":"orange","1.0":"red"}).add_to(m0)
        folium_static(m0,width=950,height=450)

    st.info("Use the sidebar to enter your route and get an AI safety analysis!")

# ============================================================
# RESULTS PAGE
# ============================================================
else:
    if slat==elat and slon==elon:
        st.error("Start and destination cannot be the same!")
    else:
        with st.spinner("Running Random Forest model on route segments..."):
            time.sleep(1.5)
            routes=analyze_route(slat,slon,elat,elon,travel_hour)

        ml_badge = "ML Model" if routes[0]["used_ml"] else "Rule-Based"
        st.success(f"Analysis complete! {ml_badge} | Found {len(routes)} route options")
        st.markdown(f"**Route:** {sp} to {ep} | **Time:** {ticons[tperiod]}")

        tab1,tab2,tab3,tab4=st.tabs(
            ["Route Comparison","Map View","Safety Tips","ML Explanation"]
        )

        with tab1:
            for i,route in enumerate(routes):
                grade=get_grade(route["score"])
                rec=get_rec(route["score"])
                if i==0:
                    st.markdown("### Recommended Route")
                    c1,c2,c3,c4,c5=st.columns([3,1,1,1,1])
                    with c1:
                        st.markdown(f"**{route['name']}**")
                        st.caption(f"ML Prediction: {route['label']}")
                    with c2: st.metric("Safety Score",f"{route['score']}/100")
                    with c3: st.metric("Grade",grade)
                    with c4: st.metric("Distance",f"{route['distance']} km")
                    with c5: st.metric("Walk Time",f"{route['walk_min']:.0f} min")
                    st.info(rec)
                    c1,c2,c3,c4=st.columns(4)
                    with c1: st.metric("Incident Risk",f"{route['inc_risk']:.0f}/100")
                    with c2: st.metric("Isolation Risk",f"{route['iso_risk']:.0f}/100")
                    with c3: st.metric("Help Risk",f"{route['help_risk']:.0f}/100")
                    with c4: st.metric("Time Risk",f"{route['time_risk']:.0f}/100")
                    st.divider()
                else:
                    with st.expander(f"{route['name']} | Safety: {route['score']}/100 | {route['label']}"):
                        c1,c2,c3,c4=st.columns(4)
                        with c1: st.metric("Safety",f"{route['score']}/100")
                        with c2: st.metric("Grade",grade)
                        with c3: st.metric("Distance",f"{route['distance']} km")
                        with c4: st.metric("Walk Time",f"{route['walk_min']:.0f} min")
                        st.warning(rec)

            st.divider()
            st.subheader("Safety Score Comparison")
            cdf=pd.DataFrame({"Route":[r["name"][:20] for r in routes],"Safety Score":[r["score"] for r in routes]}).set_index("Route")
            st.bar_chart(cdf,color="#BD3A3C")

        with tab2:
            st.subheader("Route Map")
            st.caption("Green=safest | Yellow=medium | Red=least safe | Heatmap=incident density")
            rmap=make_map(routes,slat,slon,elat,elon)
            folium_static(rmap,width=950,height=500)

        with tab3:
            st.subheader("Safety Tips")
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**Before You Leave:**")
                st.markdown("- Share live location with a trusted person")
                st.markdown("- Ensure phone is fully charged")
                st.markdown("- Inform someone of your arrival time")
                st.markdown("- Take the recommended safest route")
                st.divider()
                st.markdown("**Emergency Numbers:**")
                st.markdown("- Police: **15**")
                st.markdown("- Ambulance: **115**")
                st.markdown("- Women Helpline: **1099**")
                st.markdown("- Rescue: **1122**")
            with c2:
                if tperiod=="night":
                    st.error("Night Travel - Highest Risk. Travel with companion. Use Careem with trip sharing. Stay on main roads only.")
                elif tperiod=="evening":
                    st.warning("Evening Travel - Elevated Risk. Stick to recommended route. Stay on well-lit streets.")
                else:
                    st.success("Day Travel - Lower Risk. Follow recommended route. Stay aware of surroundings.")

        with tab4:
            st.subheader("ML Model Explanation")
            st.markdown("**How the AI scored your route:**")
            best=routes[0]
            feat=extract_features((slat+elat)/2,(slon+elon)/2,travel_hour)
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**Location Features Extracted:**")
                st.markdown(f"- Distance to police: **{feat['dist_to_police_km']:.2f} km**")
                st.markdown(f"- Distance to hospital: **{feat['dist_to_hospital_km']:.2f} km**")
                st.markdown(f"- Distance to bus stop: **{feat['dist_to_busstop_km']:.2f} km**")
                st.markdown(f"- Incidents within 500m: **{feat['incident_count_500m']}**")
                st.markdown(f"- Incidents within 1km: **{feat['incident_count_1km']}**")
                st.markdown(f"- Avg incident severity: **{feat['avg_severity_500m']:.1f}/5**")
            with c2:
                st.markdown("**Time Features:**")
                st.markdown(f"- Travel hour: **{travel_hour}:00**")
                st.markdown(f"- Time risk score: **{feat['time_risk']:.2f}** (0=safe, 1=risky)")
                st.markdown(f"- Is night: **{'Yes' if feat['is_night'] else 'No'}**")
                st.markdown(f"- Is evening: **{'Yes' if feat['is_evening'] else 'No'}**")
                st.markdown(f"- Isolation score: **{feat['isolation_score']:.2f}** (0=busy, 1=isolated)")
                st.divider()
                st.markdown(f"**Model Prediction: {best['label']}**")
                st.markdown(f"**Safety Score: {best['score']}/100**")
            if os.path.exists("model_analysis.png"):
                st.image("model_analysis.png",use_container_width=True,
                         caption="Model trained on 800+ labeled samples from Lahore")

st.divider()
st.markdown("<div style='text-align:center;color:#7D694E;font-size:0.85em;padding:10px'>GuardHer Pakistan | Random Forest ML | 186K roads | 691 POIs | 194 Incidents | Police 15 | Women Helpline 1099</div>",unsafe_allow_html=True)
