import streamlit as st
import numpy as np
import joblib
import folium
import json
from streamlit_folium import st_folium
from shapely.geometry import shape, Polygon
import pandas as pd

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🌍 GEOAI - Modélisation avancée de la salinisation")
st.subheader("📍 Gandiol - Saint-Louis, Sénégal")

# =========================
# MODÈLE
# =========================
model = joblib.load("salinity_model_gandiol.pkl")

# =========================
# CARTE INTERACTIVE
# =========================
m = folium.Map(location=[15.95, -16.45], zoom_start=11, tiles="OpenStreetMap")

folium.plugins.Draw(
    export=True,
    draw_options={
        "polygon": True,
        "rectangle": True,
        "circle": False,
        "polyline": False,
        "marker": False
    }
).add_to(m)

map_data = st_folium(m, height=500)

# =========================
# EXTRACTION POLYGONE
# =========================
st.markdown("## 🧭 Zone sélectionnée")

if map_data and map_data.get("last_active_drawing"):
    geojson = map_data["last_active_drawing"]
    st.success("Zone détectée ✔")

    geometry = shape(geojson["geometry"])
    st.write("Surface analysée :", geometry.area)

    # =========================
    # SIMULATION GRID SPATIALE
    # =========================
    st.markdown("## 🔬 Simulation raster salinité")

    n_points = 200

    lats = np.random.uniform(15.7, 16.2, n_points)
    lons = np.random.uniform(-16.8, -16.2, n_points)

    # variables simulées GEOAI
    ndvi = np.random.uniform(-0.2, 0.8, n_points)
    ndwi = np.random.uniform(-0.3, 0.6, n_points)
    bsi  = np.random.uniform(-0.5, 0.7, n_points)
    distance_mer = np.random.uniform(0, 30, n_points)
    altitude = np.random.uniform(0, 40, n_points)
    humidite = np.random.uniform(0, 1, n_points)
    pluie = np.random.uniform(0, 200, n_points)

    # stack features
    X = np.column_stack([
        ndvi, ndwi, bsi,
        distance_mer,
        altitude,
        humidite,
        pluie
    ])

    # =========================
    # PRÉDICTION SPATIALE
    # =========================
    preds = model.predict(X)

    df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "salinite": preds
    })

    st.map(df)

    # =========================
    # STATISTIQUES
    # =========================
    st.markdown("## 📊 Analyse spatiale")

    st.write("Zone saine :", (preds == 0).sum())
    st.write("Modérée :", (preds == 1).sum())
    st.write("Forte :", (preds == 2).sum())

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧠 GEOAI Advanced")
st.sidebar.write("✔ Sentinel-2 + DEM + Climate")
st.sidebar.write("✔ Machine Learning spatial")
st.sidebar.write("✔ Polygon-based prediction")
st.sidebar.write("✔ Gandiol coastal system")