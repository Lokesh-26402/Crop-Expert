# 🌾Crop Expert — An AI-based Crop Recommendation for farmers

Crop Expert is an AI-powered decision-support system for farmers that recommends suitable crops, fertilizer plans, irrigation advice, pest guidance, and financial estimates based on soil, weather and market context. Built with Streamlit, it combines explainable ML (RandomForest), live weather, geolocation and simple agronomic heuristics into a lightweight, usable dashboard.

# 🔍 Abstract

•	Agriculture is one of the most crucial sectors of the Indian economy, yet many farmers continue to face the challenge of crop misselection — choosing crops that are not well-suited to their soil, climate, or market conditions. “Crop Expert” is an AI-powered smart agriculture assistant that recommends the most suitable crops for cultivation based on soil characteristics, weather conditions, and market demand.

•	The system integrates machine learning (Random Forest Classifier) with real-time environmental data (using Open Meteo API) and geo-intelligence (via Geopy). It empowers farmers and agricultural planners with actionable insights — from fertilizer guidance and irrigation scheduling to profit estimation and pest control suggestions.

•	With its user-friendly Streamlit-based interface, Crop Expert not only predicts suitable crops but also generates a comprehensive digital report, complete with financial analytics, visual dashboards, and voice summaries, making agriculture smarter, data-driven, and sustainable.

# 🌱 Introduction

In modern agriculture, the decision of which crop to cultivate can determine the success or failure of a farming season. Traditionally, these decisions are based on experience, intuition, or local advice, often ignoring scientific data such as soil fertility, moisture levels, rainfall patterns, and market trends.
The purpose of this project is to bridge this gap by building an AI-based system that recommends the best crops for a given region, backed by machine learning models trained on environmental data.

**Problem Context**

•	Unscientific crop selection: Many farmers still rely on traditional practices, leading to low yields.

•	Climate variability: Changing rainfall and temperature patterns affect productivity.

•	Soil degradation: Lack of knowledge about soil health causes nutrient imbalance.

•	Market unawareness: Farmers often grow crops that are not in demand, leading to losses.

**Objective**

Crop Expert aims to help farmers make data-driven cultivation decisions by providing:

•	Accurate crop recommendations based on soil, weather, and market data.

•	Fertilizer and irrigation advice to enhance yield.

•	Pest and disease prevention tips.

•	Profit and cost analysis for better economic planning.

•	Multilingual & voice-enabled accessibility for ease of use in rural regions.

# ⚙️ System Overview

Crop Expert combines AI, data analytics, and geospatial intelligence into a unified application.

**•	Key Technologies**


⦁	**Frontend UI	Streamlit:**	Web-based user interface

⦁	**Machine Learning Model:**	Random Forest Classifier	Crop prediction based on environmental inputs

⦁	**Data Sources**	Open Meteo API:	Live weather and climate data

⦁	**Location Intelligence:**	Geopy & Folium	Geocoding and map visualization

⦁	**Data Storage**:	Pandas, Joblib	Local caching and model persistence

⦁	**Visualization:**	Plotly Express	Data-driven charts and dashboards

⦁	**Speech Interface:**	Google Text-to-Speech	Audio summary of recommendations

⦁	**Translation Engine:**	Deep Translator	Multilingual support (English, Hindi, Telugu, Tamil)

# 📊 Visualization and User Experience

The dashboard provides real-time interactivity with the following features:

•	**Live Map Integration:**
Displays the user’s selected area with precise latitude and longitude.

•	**Soil Health Card:**
A color-coded table indicating whether each parameter is Low, Optimal, or High.

•	**Weather & Climate Dashboard:**
Graphical visualization of 30-day temperature and rainfall trends using Plotly.

**•	Market Demand Analysis:**
Bar charts showing which crops are currently in high demand.

**•	Explainable AI View:**
Feature importance bar chart explaining which factors influenced the model’s decisions.



# 🔊 Multilingual and Voice Integration

To ensure accessibility for farmers across India:

•	The app supports English, Hindi, Telugu, and Tamil.

•	Text content can be automatically translated using the Google Translator API.

•	A Text-to-Speech engine (gTTS) generates an audio summary of the top recommendation in the selected language.

Example (in Telugu):
“మీ మట్టి మరియు వాతావరణ పరిస్థితుల ఆధారంగా రైస్ పంట అత్యంత అనుకూలంగా ఉంది. అంచనా లాభం ₹72,000.”




# 🌍 Social and Environmental Impact

•	Empowers Small Farmers: Offers scientific decision support to maximize yield and income.

•	Promotes Sustainable Agriculture: Recommends soil-friendly crops and balanced fertilizer use.

•	Optimizes Resource Usage: Reduces unnecessary water and fertilizer wastage.

•	Bridges the Digital Divide: Through multilingual voice-enabled features accessible even in low-literacy areas.

•	Boosts Agricultural Productivity: Enables data-driven planning and risk mitigation.

# 🧩 Future Enhancements

1.	Real Soil Sensor Integration (IoT devices for live NPK and moisture readings).

2.	Satellite Imagery for Crop Health Detection.

3.	Crop Disease Detection via Image Recognition (CNN).

4.	Farmer Chatbot using LLM (like Gemini or GPT).

5.	Offline Android App with Local Language UI.

6.	Blockchain-based Market Linkage for Fair Pricing.

## 📸 Screenshots
<https://github.com/Lokesh-26402/Crop-Expert/blob/main/Screenshot%202025-09-30%20102320.png>

Rythu Mitra Home Dashboard

---


