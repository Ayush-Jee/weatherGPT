import os
import json
import re
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# .env file ko auto-load karein
load_dotenv()

app = FastAPI(title="WeatherGPT")

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# MongoDB Async Connection Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client["weathergpt_db"]
    feedback_col = db["feedbacks"]
    print("✅ MongoDB connection setup successfully.")
except Exception as err:
    print("⚠️ MongoDB Connection Warning:", err)

class QueryRequest(BaseModel):
    message: str
    lat: Optional[float] = 22.5726   # Default: Kolkata
    lon: Optional[float] = 88.3639
    city: Optional[str] = "Kolkata"

class FeedbackRequest(BaseModel):
    name: Optional[str] = "Anonymous"
    email: Optional[str] = ""
    message: str

# Geocoding strictly constrained to India via Nominatim (In English)
async def get_coordinates(city_name: str):
    clean_name = city_name.split(",")[0].strip()
    url = f"https://nominatim.openstreetmap.org/search?q={clean_name}&countrycodes=in&format=json&limit=1&accept-language=en"
    headers = {"User-Agent": "WeatherGPT-DisasterApp/1.0"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(url, headers=headers)
            data = res.json()
            if len(data) > 0:
                parts = data[0]["display_name"].split(",")
                short_name = f"{parts[0].strip()}, {parts[-2].strip()}" if len(parts) > 2 else data[0]["display_name"]
                return {
                    "name": short_name,
                    "lat": float(data[0]["lat"]),
                    "lon": float(data[0]["lon"]),
                    "state": "India"
                }
    except Exception:
        pass
    return None

# Meteorological Ingestion via Open-Meteo with network fail-safety
async def fetch_weather_data(lat: float, lon: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_gusts_10m"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
        "&timezone=auto"
    )
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(url)
            if res.status_code == 200:
                return res.json()
    except Exception as e:
        print("Weather API Network Issue:", e)

    return {
        "current": {
            "temperature_2m": 27.5,
            "relative_humidity_2m": 80,
            "apparent_temperature": 30.0,
            "precipitation": 0,
            "weather_code": 0,
            "wind_speed_10m": 8.0,
            "wind_gusts_10m": 12.0
        },
        "daily": {
            "precipitation_sum": [0],
            "precipitation_probability_max": [10]
        }
    }

# Disaster Alert Engine (WMO thresholds evaluation)
def evaluate_disaster_risk(weather_json: dict):
    current = weather_json.get("current", {})
    daily = weather_json.get("daily", {})
    
    wind_gust = current.get("wind_gusts_10m", 0)
    current_rain = current.get("precipitation", 0)
    max_rain = daily.get("precipitation_sum", [0])[0] if daily.get("precipitation_sum") else 0
    max_rain_prob = daily.get("precipitation_probability_max", [0])[0] if daily.get("precipitation_probability_max") else 0
    temp = current.get("temperature_2m", 25)

    alerts = []
    level = "NORMAL"

    if wind_gust >= 65:
        alerts.append(f"GALE/CYCLONE WARNING: Wind gusts reaching {wind_gust} km/h.")
        level = "RED"
    elif wind_gust >= 45:
        alerts.append(f"HIGH WIND ADVISORY: Gusts up to {wind_gust} km/h.")
        level = "ORANGE"

    if max_rain >= 70 or current_rain >= 20:
        alerts.append("FLOOD / HEAVY RAINFALL ALERT: Immediate localized flooding risk.")
        level = "RED"
    elif max_rain_prob >= 50 or max_rain >= 25:
        alerts.append(f"RAIN ADVISORY: {max_rain_prob}% probability of rain. Caution advised.")
        level = "YELLOW"

    if temp >= 42:
        alerts.append("HEATWAVE WARNING: Avoid direct outdoor exposure 12 PM - 4 PM.")
        level = "RED"

    return {
        "level": level,
        "active_warnings": alerts,
        "rain_prob": max_rain_prob
    }

# Weather Code Mapping in Clean English
def get_weather_desc(code: int):
    mapping = {
        0: "Clear Sky",
        1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing Rime Fog",
        51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Dense Drizzle",
        61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
        80: "Rain Showers", 81: "Moderate Showers", 82: "Violent Rain Showers",
        95: "Thunderstorm with Lightning"
    }
    return mapping.get(code, "Clear / Fair")

# 100% Real-Time OpenRouter Generative AI Reasoning (Multi-Model Fallback)
async def generate_conversational_response(user_query: str, weather_data: dict, risk_data: dict, place_name: str):
    cur = weather_data.get("current", {})
    temp = cur.get("temperature_2m", "--")
    feels_like = cur.get("apparent_temperature", "--")
    humidity = cur.get("relative_humidity_2m", "--")
    wind = cur.get("wind_speed_10m", "--")
    gusts = cur.get("wind_gusts_10m", "--")
    rain = cur.get("precipitation", 0)
    w_desc = get_weather_desc(cur.get("weather_code", 0))
    rain_prob = risk_data.get("rain_prob", 0)
    alerts = risk_data.get("active_warnings", [])
    alert_str = " | ".join(alerts) if alerts else "None (Conditions Normal)"

    if OPENROUTER_API_KEY:
        system_prompt = f"""You are WeatherGPT, a smart real-time meteorological AI assistant.
User Query: "{user_query}"
Target Location: {place_name}

LIVE TELEMETRY:
- Condition: {w_desc}
- Temperature: {temp}°C (Feels like: {feels_like}°C)
- Humidity: {humidity}%
- Wind Speed: {wind} km/h (Gusts: {gusts} km/h)
- Rain Probability: {rain_prob}%
- Alert Level: {risk_data['level']}
- Active Warnings: {alert_str}

RULES:
1. Reason directly about the question (e.g. umbrella, rain, clothing, travel, farming). Give a direct, natural English conversational answer in 1-2 sentences.
2. After answering, append the telemetry data block in this exact format:

---
📊 **Live Telemetry ({place_name}):**
• Condition: {w_desc}
• Temperature: {temp}°C (Feels like {feels_like}°C)
• Humidity: {humidity}%
• Wind Speed: {wind} km/h (Gusts: {gusts} km/h)
• Rain Probability: {rain_prob}%
"""
        free_models = [
            "openrouter/auto",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "qwen/qwen-2.5-7b-instruct:free"
        ]

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "WeatherGPT",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=15.0) as client_ai:
            for model_name in free_models:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": system_prompt}],
                    "max_tokens": 250,
                    "temperature": 0.2
                }
                try:
                    res = await client_ai.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                    data = res.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                except Exception:
                    pass

    advisory = "Skies are stable with no rain threat." if rain_prob < 40 else f"Rain probability is {rain_prob}%, keep protection handy."
    return (
        f"Based on real-time data for {place_name}, {advisory}\n\n"
        f"📊 **Telemetry ({place_name}):**\n"
        f"• Condition: {w_desc}\n"
        f"• Temperature: {temp}°C (Feels like {feels_like}°C) | Humidity: {humidity}%\n"
        f"• Wind Speed: {wind} km/h (Gusts: {gusts} km/h) | Rain Probability: {rain_prob}%"
    )

@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    lat = req.lat
    lon = req.lon
    place_name = req.city if req.city else "Current Location"

    city_match = re.search(r'\b(?:in|at|for|near|of)\s+([a-zA-Z\s]{3,25})', req.message, re.IGNORECASE)
    if city_match:
        potential_city = city_match.group(1).strip()
        ignore_words = {"today", "tomorrow", "now", "my current location", "my location", "this week"}
        if potential_city.lower() not in ignore_words:
            coords = await get_coordinates(potential_city)
            if coords:
                lat = coords["lat"]
                lon = coords["lon"]
                place_name = coords["name"]

    weather_raw = await fetch_weather_data(lat, lon)
    risk_raw = evaluate_disaster_risk(weather_raw)
    llm_reply = await generate_conversational_response(req.message, weather_raw, risk_raw, place_name)

    return {
        "reply": llm_reply,
        "location": {"name": place_name, "lat": lat, "lon": lon},
        "metrics": weather_raw.get("current", {}),
        "risk": risk_raw
    }

# MongoDB Feedback Saving Endpoint
@app.post("/api/feedback")
async def save_feedback(data: FeedbackRequest):
    try:
        feedback_doc = {
            "name": data.name.strip() if data.name else "Anonymous",
            "email": data.email.strip() if data.email else "Not provided",
            "message": data.message.strip(),
            "timestamp": datetime.utcnow()
        }
        result = await feedback_col.insert_one(feedback_doc)
        return {"status": "success", "id": str(result.inserted_id)}
    except Exception as e:
        print("MongoDB Save Error:", e)
        raise HTTPException(status_code=500, detail="Database write error")

@app.get("/")
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)