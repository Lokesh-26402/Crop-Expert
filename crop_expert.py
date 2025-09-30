import streamlit as st
import numpy as np
import pandas as pd
import random
import joblib
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import openmeteo_requests
import requests_cache
from retry_requests import retry
from streamlit_folium import st_folium
import folium
from datetime import datetime, timedelta

# --- Optional Libraries for UI Features ---
try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try:
    from gtts import gTTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# -----------------------------
# 1. ML MODEL & CORE LOGIC
# -----------------------------
MODEL_PATH = "crop_model.pkl"
FEATURE_NAMES = ["pH", "Moisture (%)", "Nitrogen (kg/ha)", "Phosphorus (kg/ha)", "Potassium (kg/ha)",
                 "Temperature (°C)", "Rainfall (mm/hr)", "Humidity (%)"]


def train_dummy_model(path=MODEL_PATH):
    """Creates and saves a dummy RandomForest model if one doesn't exist."""
    st.info("Training a new dummy model for demonstration...")
    n = 500
    features = np.random.rand(n, 8)
    # Scale features to be in a realistic range
    features[:, 0] = features[:, 0] * 3.5 + 5.0  # pH
    features[:, 1] = features[:, 1] * 55 + 15  # Moisture
    features[:, 2] = features[:, 2] * 180 + 20  # Nitrogen
    features[:, 3] = features[:, 3] * 110 + 10  # Phosphorus
    features[:, 4] = features[:, 4] * 270 + 30  # Potassium
    features[:, 5] = features[:, 5] * 35 + 5  # Temperature
    features[:, 6] = features[:, 6] * 10  # Rainfall
    features[:, 7] = features[:, 7] * 90 + 10  # Humidity

    y = np.random.choice(["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Pulses", "Sugarcane", "Barley"], size=n)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(features, y)
    joblib.dump(model, path)
    st.success(f"Dummy model trained and saved to {path}")


# Load the model, or train one if it doesn't exist
if not os.path.exists(MODEL_PATH):
    train_dummy_model(MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load the prediction model. Error: {e}")
    model = None


# -----------------------------
# 2. DATA & FEATURE FUNCTIONS
# -----------------------------
def get_soil_data(lat, lon):
    """Simulates fetching soil data. Made deterministic based on location."""
    random.seed(int(lat) + int(lon))
    return {
        "pH": round(random.uniform(5.5, 8.5), 2),
        "Moisture (%)": round(random.uniform(20, 60), 2),
        "Nitrogen (kg/ha)": random.randint(30, 180),
        "Phosphorus (kg/ha)": random.randint(15, 100),
        "Potassium (kg/ha)": random.randint(40, 250)
    }


def get_weather_and_climate_data(lat, lon):
    """Fetches real-time weather and 30-day historical climate data. Falls back to simulated data on failure."""
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=1800)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        params = {
            "latitude": lat, "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation"],
            "daily": ["temperature_2m_mean", "precipitation_sum"], "timezone": "auto",
            "start_date": start_date, "end_date": end_date
        }
        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        res = responses[0]
        current, daily = res.Current(), res.Daily()

        current_weather = {
            "Temperature (°C)": round(current.Variables(0).Value(), 2),
            "Humidity (%)": round(current.Variables(1).Value(), 2),
            "Rainfall (mm/hr)": round(current.Variables(2).Value(), 2)
        }

        # Build daily dataframe
        daily_data = {"date": pd.to_datetime(daily.Time(), unit="s")}
        daily_data["temperature_2m_mean"] = daily.Variables(0).ValuesAsNumpy()
        daily_data["precipitation_sum"] = daily.Variables(1).ValuesAsNumpy()

        historical_climate_summary = {
            "30-Day Avg Temp (°C)": round(np.mean(daily_data["temperature_2m_mean"]), 2),
            "30-Day Total Rainfall (mm)": round(np.sum(daily_data["precipitation_sum"]), 2)
        }

        return current_weather, historical_climate_summary, pd.DataFrame(data=daily_data)
    except Exception as e:
        st.warning(f"Could not fetch live weather data: {e}. Using simulated data.")
        dummy_df = pd.DataFrame()
        return {"Temperature (°C)": 28.5, "Humidity (%)": 65, "Rainfall (mm/hr)": 0.2}, \
            {"30-Day Avg Temp (°C)": 29.1, "30-Day Total Rainfall (mm)": 25.5}, dummy_df


def get_market_trends():
    """Simulates market data. Made deterministic for consistency."""
    random.seed(42)
    crops = {"Wheat": 2200, "Rice": 2100, "Maize": 2000, "Soybean": 4500, "Cotton": 6000, "Pulses": 5500,
             "Sugarcane": 350, "Barley": 1800}
    return {crop: {"demand_score": random.randint(80, 150),
                   "avg_price_quintal": round(base_price * random.uniform(0.9, 1.1)),
                   "trend": random.choice(['📈', '📉', '→'])}
            for crop, base_price in crops.items()}


def get_dynamic_yield(crop, soil, weather):
    """Calculates a dynamic crop yield based on conditions."""
    base_yields = {"Rice": 45, "Wheat": 40, "Maize": 50, "Soybean": 25, "Cotton": 20, "Pulses": 15, "Sugarcane": 80,
                   "Barley": 35}
    base = base_yields.get(crop, 30) * 0.8
    if 80 <= soil.get("Nitrogen (kg/ha)", 0) <= 160:
        base += 3
    if 40 <= soil.get("Phosphorus (kg/ha)", 0) <= 80:
        base += 3
    if 6.0 <= soil.get("pH", 7.0) <= 7.5:
        base += 2
    if 20 <= weather.get("Temperature (°C)", 0) <= 32:
        base += 2
    return round(base + random.uniform(-2, 2))


def get_crop_recommendations(soil, weather, market):
    """Returns top 3 crop recommendations based on provided data using the loaded model."""
    if model is None:
        st.error("Model is not loaded. Cannot generate recommendations.")
        return []

    input_data = {**soil, **weather}
    features = np.array([input_data.get(name, 0) for name in FEATURE_NAMES]).reshape(1, -1)

    # If model supports predict_proba
    try:
        probabilities = model.predict_proba(features)[0]
    except Exception:
        # fallback: use predict and give 100% to predicted class
        preds = model.predict(features)
        probabilities = np.zeros(len(model.classes_))
        idx = list(model.classes_).index(preds[0])
        probabilities[idx] = 1.0

    top_3_indices = np.argsort(probabilities)[::-1][:3]

    recommendations = []
    for i in top_3_indices:
        crop = model.classes_[i]
        yield_quintals = get_dynamic_yield(crop, soil, weather)
        market_info = market.get(crop, {})
        recommendations.append({
            "crop": crop,
            "confidence": f"{probabilities[i] * 100:.2f}%",
            "yield_quintals": yield_quintals,
            "expected_yield": f"{yield_quintals} quintals/acre",
            "sustainability": random.choice(["High", "Medium", "Low"]),
            "market_demand": f"{market_info.get('demand_score', 'N/A')}/150",
            "market_price": market_info.get('avg_price_quintal', 0),
            "market_trend": market_info.get('trend', '→')
        })
    return recommendations


# -----------------------------
# 3. UI HELPER & FEATURE FUNCTIONS
# -----------------------------
CROP_PROFILES = {
    "Wheat": {"desc": "A key cereal crop, grown worldwide. It's a staple for bread and pasta.", "sow": "Rabi (Oct-Dec)",
              "harvest": "Feb-May",
              "img": "https://upload.wikimedia.org/wikipedia/commons/7/76/Feld_mit_Weizen_im_Sonnenuntergang_2015-08-01.jpg"},
    "Rice": {"desc": "A staple for over half the world's population, typically grown in flooded paddies.",
             "sow": "Kharif (Jun-Jul)", "harvest": "Nov-Dec",
             "img": "https://upload.wikimedia.org/wikipedia/commons/b/b2/A_terraced_paddy_field_in_Munnar%2C_Kerala.jpg"},
    "Maize": {"desc": "A versatile crop used for food, animal feed, and biofuel. Known for its tall stalks.",
              "sow": "Kharif/Rabi", "harvest": "Sep-Oct / Jan-Feb",
              "img": "https://upload.wikimedia.org/wikipedia/commons/b/b8/Zea_mays_-_Köhler–s_Medizinal-Pflanzen-154.jpg"},
    "Soybean": {"desc": "A legume valued for its high protein and oil content. A key crop in global agriculture.",
                "sow": "Kharif (Jun-Jul)", "harvest": "Oct-Nov",
                "img": "https://upload.wikimedia.org/wikipedia/commons/8/82/Soybean_flowering_-_Black_Jet.jpg"},
    "Cotton": {"desc": "A fiber crop crucial for the textile industry. It grows in warm climates.",
               "sow": "Kharif (Apr-Jun)", "harvest": "Oct-Dec",
               "img": "https://upload.wikimedia.org/wikipedia/commons/1/18/Cotton_plant_with_boll_in_Andhra_Pradesh.jpg"},
    "Sugarcane": {"desc": "A tall perennial grass that is the primary source of the world's sugar.", "sow": "Jan-Mar",
                  "harvest": "Dec-Mar (after 10-18 months)",
                  "img": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Saccharum_officinarum_-_Köhler–s_Medizinal-Pflanzen-128.jpg"},
    "Pulses": {
        "desc": "Includes crops like lentils, chickpeas, and beans. They are rich in protein and fix nitrogen in the soil.",
        "sow": "Rabi (Oct-Nov)", "harvest": "Feb-Mar",
        "img": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Lentils_and_beans.jpg"},
    "Barley": {"desc": "A cereal grain used for animal fodder, brewing (beer), and health foods.",
               "sow": "Rabi (Oct-Nov)", "harvest": "Mar-Apr",
               "img": "https://upload.wikimedia.org/wikipedia/commons/1/1f/Barley_field_-_geograph.org.uk_-_494632.jpg"}
}


def get_financial_analysis(crop, yield_quintals, market_price):
    costs = {"Rice": 35000, "Wheat": 30000, "Maize": 28000, "Soybean": 25000, "Cotton": 40000, "Pulses": 22000,
             "Sugarcane": 50000, "Barley": 27000}
    cost = costs.get(crop, 30000)
    revenue = market_price * yield_quintals
    return {"Gross Revenue": revenue, "Est. Cultivation Cost": cost, "Estimated Net Profit": revenue - cost}


def get_pest_and_disease_info(crop):
    advisories = {"Rice": {"Pests": "Stem Borer, Brown Planthopper.", "Diseases": "Sheath Blight, Blast."}}
    info = advisories.get(crop, {"Pests": "Monitor for common local pests.", "Diseases": "Ensure good soil health."})
    return f"- **Common Pests**: {info['Pests']}\n- **Common Diseases**: {info['Diseases']}"


def get_soil_health_card(soil_data):
    card = []
    ph = soil_data.get("pH", 7.0)
    status = "Optimal" if 6.5 <= ph <= 7.5 else ("Slightly Acidic" if ph < 6.5 else "Slightly Alkaline")
    card.append({"Parameter": "pH", "Value": ph, "Status": status})
    npk_map = {"Nitrogen": (80, 160), "Phosphorus": (40, 80), "Potassium": (100, 200)}
    for param, (low, high) in npk_map.items():
        val = soil_data.get(f"{param} (kg/ha)", 0)
        status = "Low" if val < low else ("High" if val > high else "Optimal")
        card.append({"Parameter": f"{param} (kg/ha)", "Value": val, "Status": status})
    return pd.DataFrame(card)


def get_fertilizer_recommendation(crop, soil_data):
    optimal_npk = {"Rice": (120, 60, 60), "Wheat": (140, 70, 50), "Maize": (160, 80, 50)}
    target_n, target_p, target_k = optimal_npk.get(crop, (100, 50, 50))
    n, p, k = soil_data.get("Nitrogen (kg/ha)", 0), soil_data.get("Phosphorus (kg/ha)", 0), soil_data.get(
        "Potassium (kg/ha)", 0)
    return (f"- **N:** Apply approx. **{max(0, round(target_n - n))} kg/ha**.\n"
            f"- **P:** Apply approx. **{max(0, round(target_p - p))} kg/ha**.\n"
            f"- **K:** Apply approx. **{max(0, round(target_k - k))} kg/ha**.")


def get_soil_amendment_advice(soil_data):
    """Provides actionable advice for soil correction."""
    advice = []
    ph = soil_data.get("pH", 7.0)
    if ph < 6.0:
        advice.append("• **pH is Acidic**: Apply agricultural lime to raise the pH towards neutral.")
    elif ph > 7.8:
        advice.append("• **pH is Alkaline**: Apply gypsum or elemental sulfur to lower the pH.")
    if soil_data.get("Nitrogen (kg/ha)", 100) < 60:
        advice.append("• **Nitrogen is Low**: Incorporate compost/manure or use Urea fertilizer.")
    if soil_data.get("Phosphorus (kg/ha)", 50) < 30:
        advice.append("• **Phosphorus is Low**: Use phosphate fertilizers (e.g., DAP) or bone meal.")
    if not advice:
        return "✅ Your soil appears well-balanced. Maintain good practices like crop rotation."
    return "\n".join(advice)


def get_irrigation_advice(crop, soil, climate):
    if climate.get("30-Day Total Rainfall (mm)", 0) > 75:
        return "✅ **Hold Irrigation**: Significant recent rainfall."
    if soil.get("Moisture (%)", 0) > 40:
        return "✅ **Monitor**: Soil is currently moist."
    return "💧 **Irrigate Soon**: Soil is becoming dry, and recent rainfall has been low."


def generate_text_report(results):
    rec, soil, weather, climate = results['recommendations'], results['soil'], results['weather'], results['climate_summary']
    report = ["=" * 60 + "\n     Crop Expert — An AI-based Crop Recommendation Report\n" + "=" * 60,
              f"\nPlan for: {st.session_state.location_name} on {datetime.now().strftime('%Y-%m-%d')}\n",
              "--- 1. Environmental Snapshot ---",
              f"Live Temp: {weather['Temperature (°C)']}°C | Humidity: {weather['Humidity (%)']}%",
              f"30-Day Avg Temp: {climate['30-Day Avg Temp (°C)']}°C | 30-Day Rain: {climate['30-Day Total Rainfall (mm)']} mm\n",
              "--- 2. Soil Health Card ---\n" + get_soil_health_card(soil).to_string(index=False) + "\n",
              "--- 3. Top Crop Recommendations ---"]
    for i, r in enumerate(rec):
        profit = get_financial_analysis(r['crop'], r['yield_quintals'], r['market_price'])['Estimated Net Profit']
        report.extend([f"\n#{i + 1}: {r['crop'].upper()} (Confidence: {r['confidence']})",
                       f"  - Yield Estimate: {r['expected_yield']} | Est. Net Profit: ₹{profit:,.0f}",
                       f"  - Fertilizer: {get_fertilizer_recommendation(r['crop'], soil).replace('- ', '').replace('*', '').replace('**', '').replace('.', ',')}",
                       f"  - Irrigation: {get_irrigation_advice(r['crop'], soil, climate)}"])
    return "\n".join(report)


def translate_text(text, dest_lang="hi"):
    if dest_lang in ("en", "", None) or not HAS_TRANSLATOR:
        return text
    try:
        return GoogleTranslator(source="auto", target=dest_lang).translate(text)
    except:
        return text


def text_to_speech(text, lang="hi"):
    if not HAS_TTS:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except:
        return None


def geocode_location(location_name):
    if not HAS_GEOPY:
        st.error("`geopy` not found. Please install it.")
        return None, None
    try:
        geolocator = Nominatim(user_agent="crop_expert_app_v1")
        loc = geolocator.geocode(location_name)
        if loc:
            return loc.latitude, loc.longitude
        st.warning("Location not found.")
        return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None


def style_soil_card_dataframe(df):
    def _color_status(row):
        status, color, bgcolor = row['Status'], 'white', 'gray'
        if status == "Optimal":
            bgcolor = '#2E8B57'
        elif status == "Low":
            bgcolor = '#DAA520'
        elif status == "High":
            bgcolor = '#CD5C5C'
        else:
            color = 'black'
            bgcolor = '#D3D3D3'
        return [f'background-color: {bgcolor}; color: {color}'] * len(row)

    return df.style.apply(_color_status, axis=1)


# -----------------------------
# 4. STREAMLIT APP LAYOUT
# -----------------------------
def apply_custom_styling():
    st.markdown("""<style>
        h1 { color: #004d00; border-bottom: 3px solid #2E8B57; }
        h2 { color: #006400; } h3 { color: #2E8B57; }
        [data-testid="stSidebar"] { background-color: #F0FFF0; }
        .stButton>button { background-color: #2E8B57; color: white; font-weight: bold; border-radius: 8px;}
        .stMetric { background-color: #F5FFFA; border: 1px solid #2E8B57; border-radius: 8px; padding: 10px; }
        [data-testid="stExpander"] summary { font-weight: bold; background-color: #F0FFF0; border-radius: 8px;}
    </style>""", unsafe_allow_html=True)


# --- Main App Execution ---
st.set_page_config(page_title="Crop Expert — AI Crop Recommendation", layout="wide", page_icon="🌾")
apply_custom_styling()

# --- Initialize Session State ---
DEFAULT_LAT, DEFAULT_LON = 17.3850, 78.4867  # Hyderabad
for key, value in {
    "latitude": DEFAULT_LAT, "longitude": DEFAULT_LON, "location_name": "Hyderabad",
    "show_results": False, "results": {}, "manual_override": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- SIDEBAR ---
with st.sidebar:
    st.header("📍 Location Input")
    location_query = st.text_input("Enter Location", st.session_state.location_name)
    if st.button("Find Location"):
        lat, lon = geocode_location(location_query)
        if lat and lon:
            st.session_state.latitude, st.session_state.longitude, st.session_state.location_name = lat, lon, location_query
            st.session_state.show_results = False  # Reset results on new location

    st.markdown("---")
    with st.expander("🔬 Manual Data Override (What-If)"):
        st.session_state.manual_override = st.checkbox("Enable Manual Input", value=st.session_state.manual_override)
        st.session_state.manual_ph = st.slider("Soil pH", 4.0, 9.0, 6.8, 0.1)
        st.session_state.manual_n = st.slider("Nitrogen (kg/ha)", 10, 200, 90)
        st.session_state.manual_p = st.slider("Phosphorus (kg/ha)", 5, 120, 45)
        st.session_state.manual_k = st.slider("Potassium (kg/ha)", 20, 280, 150)
        st.session_state.manual_temp = st.slider("Temperature (°C)", 5.0, 45.0, 28.0)
        st.session_state.manual_humidity = st.slider("Humidity (%)", 10, 100, 65)

    st.header("⚙️ User Settings")
    st.session_state.lang = st.selectbox("Language", ["en", "hi", "ta", "te"])

    if st.button("🛰️ Generate Command Plan", use_container_width=True, type="primary"):
        with st.spinner("Processing satellite and ground data..."):
            lat, lon = st.session_state.latitude, st.session_state.longitude
            market = get_market_trends()

            if st.session_state.manual_override:
                st.info("Using manually entered data for prediction.")
                soil = {"pH": st.session_state.manual_ph, "Moisture (%)": random.uniform(30, 50),
                        "Nitrogen (kg/ha)": st.session_state.manual_n, "Phosphorus (kg/ha)": st.session_state.manual_p,
                        "Potassium (kg/ha)": st.session_state.manual_k}
                weather = {"Temperature (°C)": st.session_state.manual_temp,
                           "Humidity (%)": st.session_state.manual_humidity,
                           "Rainfall (mm/hr)": random.uniform(0, 2)}
                _, climate_summary, climate_daily = get_weather_and_climate_data(lat, lon)
            else:
                soil = get_soil_data(lat, lon)
                weather, climate_summary, climate_daily = get_weather_and_climate_data(lat, lon)

            recommendations = get_crop_recommendations(soil, weather, market)
            st.session_state.results = {
                "soil": soil, "weather": weather, "climate_summary": climate_summary, "climate_daily": climate_daily,
                "market": market, "recommendations": recommendations, "lang": st.session_state.lang
            }
            st.session_state.show_results = True
        st.success("Plan Generated!")

# --- MAIN PAGE ---
st.title("🌾 Crop Expert — An AI-based Crop Recommendation")

# --- MAP DISPLAY in Main Area ---
st.markdown(f"### 📍 Analysis for: **{st.session_state.location_name}**")
with st.container():
    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=10)
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup=st.session_state.location_name,
        tooltip=st.session_state.location_name
    ).add_to(m)
    st_folium(m, use_container_width=True, height=400)


if not st.session_state.show_results:
    st.info("👋 Welcome! Use the sidebar to select a location and click **'Generate Command Plan'** to begin.")
else:
    res = st.session_state.results
    soil, weather, climate_summary, climate_daily, market, recommendations, ui_lang = res.values()

    # --- Header with Download Button ---
    rec_header, dl_button = st.columns([0.75, 0.25])
    with rec_header:
        st.header(f"🏆 Top Crop Recommendations")
    with dl_button:
        report_data = generate_text_report(st.session_state.results)
        st.download_button(label="📄 Download Report", data=report_data,
                           file_name=f"CropExpert_Plan_{st.session_state.location_name.replace(' ', '_')}.txt",
                           mime="text/plain", use_container_width=True)

    for i, rec in enumerate(recommendations):
        with st.expander(f"**#{i + 1}: {rec['crop']}** (Confidence: {rec['confidence']})", expanded=(i == 0)):
            tabs = st.tabs(
                ["🌾 Crop Profile", "📈 Metrics & Market", "💰 Financials", "🌱 Fertilizers & Soil", "🐛 Pest Advisory"])

            with tabs[0]:  # Crop Profile
                profile = CROP_PROFILES.get(rec['crop'], {})
                if profile:
                    col1, col2 = st.columns([0.6, 0.4])
                    with col1:
                        st.subheader(rec['crop'])
                        st.write(profile.get("desc", "No description available."))
                        st.markdown(
                            f"- **Sowing Season:** {profile.get('sow', 'N/A')}\n- **Harvesting Season:** {profile.get('harvest', 'N/A')}")
                    with col2:
                        st.image(profile.get('img'), caption=f"Image of {rec['crop']}", use_container_width=True)
                else:
                    st.info("No detailed profile available for this crop.")

            with tabs[1]:
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Dynamic Yield Estimate", rec['expected_yield'])
                m_col2.metric("Market Demand", rec['market_demand'])
                m_col3.metric("Avg. Market Price", f"₹ {rec['market_price']:,}/quintal", delta=rec['market_trend'])

            with tabs[2]:  # Financials with Chart
                analysis = get_financial_analysis(rec['crop'], rec['yield_quintals'], rec['market_price'])
                fin_col1, fin_col2 = st.columns([0.4, 0.6])
                with fin_col1:
                    for key, val in analysis.items():
                        st.metric(key, f"₹ {val:,.0f}")
                with fin_col2:
                    df_fin = pd.DataFrame([{"Category": k, "Amount": v} for k, v in analysis.items()])
                    # note: color_discrete_map uses color names; plotly will auto-handle if not available
                    fig = px.bar(df_fin, x='Category', y='Amount', color='Category', title='Financial Breakdown per Acre',
                                 template='plotly_white',
                                 color_discrete_map={'Gross Revenue': 'green', 'Est. Cultivation Cost': 'orange',
                                                     'Estimated Net Profit': '#2E8B57'})
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:  # Fertilizers and Soil Amendments
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.subheader("Nutrient Requirements")
                    st.markdown(get_fertilizer_recommendation(rec['crop'], soil))
                    st.info(get_irrigation_advice(rec['crop'], soil, climate_summary))
                with f_col2:
                    st.subheader("Soil Amendment Advice")
                    st.markdown(get_soil_amendment_advice(soil))

            with tabs[4]:
                st.markdown(get_pest_and_disease_info(rec['crop']))

    st.markdown("---")
    st.header("🔬 Model & Environment Dashboard")
    dash_col1, dash_col2 = st.columns([0.6, 0.4])
    with dash_col1:
        st.subheader("🧠 Model Insights (Explainable AI)")
        if model:
            importance_df = pd.DataFrame(
                {'Feature': FEATURE_NAMES, 'Importance': model.feature_importances_}).sort_values('Importance',
                                                                                                  ascending=True)
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title='Which Factors Matter Most?', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    with dash_col2:
        st.subheader("🌿 Soil Health Card")
        st.dataframe(style_soil_card_dataframe(get_soil_health_card(soil)), use_container_width=True, hide_index=True)
        st.subheader("🌦️ Live Weather")
        weather_col1, weather_col2, weather_col3 = st.columns(3)
        weather_col1.metric("Temperature", f"{weather['Temperature (°C)']}°C")
        weather_col2.metric("Humidity", f"{weather['Humidity (%)']}%")
        weather_col3.metric("Rainfall", f"{weather['Rainfall (mm/hr)']} mm/hr")

    st.markdown("---")
    st.header("📊 Climate & Market Analysis")
    climate_col, market_col = st.columns(2)
    with climate_col:  # Historical Climate Chart
        st.subheader("🗓️ 30-Day Climate History")
        if not climate_daily.empty:
            # Build a line chart and a bar for rainfall — plotly express combined
            fig = px.line(climate_daily, x='date', y='temperature_2m_mean', title='Avg Temp vs. Rainfall',
                          labels={'date': 'Date'})
            # add rainfall as a bar using underlying graph objects via fig.add_bar is allowed
            fig.add_bar(x=climate_daily['date'], y=climate_daily['precipitation_sum'], name='Rainfall (mm)', yaxis='y2')
            fig.update_layout(yaxis_title='Avg Temp (°C)',
                              yaxis2=dict(title='Rainfall (mm)', overlaying='y', side='right'),
                              template='plotly_white', legend=dict(x=0, y=1.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical climate chart not available for simulated data.")

    with market_col:
        st.subheader("📈 Current Market Demand")
        df_market = pd.DataFrame.from_dict(market, orient='index').reset_index().rename(
            columns={'index': 'Crop', 'demand_score': 'Demand Score'})
        fig = px.bar(df_market, x="Crop", y="Demand Score", color="Crop", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with st.container():  # Voice Summary
        st.subheader("🗣️ Voice Summary")
        top_rec = recommendations[0]
        profit_val = get_financial_analysis(top_rec['crop'], top_rec['yield_quintals'], top_rec['market_price'])[
            "Estimated Net Profit"]
        profit_str = f"rupees {profit_val:,}"
        base_text = f"The top recommended crop for {st.session_state.location_name} is {top_rec['crop']} with an estimated net profit of {profit_str} per acre."
        translated = translate_text(base_text, dest_lang=ui_lang)
        st.write(translated)
        audio_path = text_to_speech(translated, lang=ui_lang)
        if audio_path:
            st.audio(audio_path)
