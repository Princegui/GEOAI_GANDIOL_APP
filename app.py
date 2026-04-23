# ============================================================
# 🌍 GEOAI - SALINISATION DES SOLS (VERSION FINALE)
# ============================================================

import streamlit as st
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="GEOAI Salinisation",
    page_icon="🌍",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
h1 {text-align:center; color:#0b3d91;}
.stApp {background-color:#f4f9f9;}
</style>
""", unsafe_allow_html=True)

# =========================
# LOGO USSEIN
# =========================
st.image(
    "https://www.ussein.sn/wp-content/uploads/2024/12/USSEIN-LOGO-copie.png",
    width=120
)

# =========================
# CHARGEMENT MODELE
# =========================
# 👉 Choisir UN des deux modèle


model = joblib.load("geoai_model.pkl")

# =========================
# TITRE
# =========================
st.title("🌍 GEOAI - Cartographie de la salinisation des sols")
st.subheader("📍 Sénégal - Zone de Gandiol")

# =========================
# SLIDER TEMPS
# =========================
annee = st.slider("📅 Année d’analyse", 2016, 2024, 2022)

# =========================
# CARTE
# =========================
st.subheader("🗺️ Sélection de zone")

m = folium.Map(location=[15.8, -16.5], zoom_start=11)

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google Satellite + Labels",
    name="Satellite"
).add_to(m)

Draw(export=True).add_to(m)

map_data = st_folium(m, height=550, width=1000)

# =========================
# ANALYSE
# =========================
if map_data and map_data.get("last_active_drawing"):

    st.success("📍 Zone sélectionnée")

    # =========================
    # SIMULATION INDICES (ou remplacer par GEE)
    # =========================
    np.random.seed(annee)

    ndvi = np.random.uniform(0.1, 0.8)
    ndwi = np.random.uniform(-0.4, 0.4)
    bsi  = np.random.uniform(0.2, 0.9)
    sar  = np.random.uniform(0.2, 1.0)

    X = np.array([[ndvi, ndwi, bsi, sar]])

    # =========================
    # PREDICTION
    # =========================
    if USE_DL:
        X = scaler.transform(X)
        pred = np.argmax(model.predict(X))
    else:
        pred = model.predict(X)[0]

    # =========================
    # DASHBOARD
    # =========================
    col1, col2 = st.columns([2, 1])

    with col2:

        st.subheader("📊 Indices")

        st.metric("NDVI", round(ndvi, 3))
        st.metric("NDWI", round(ndwi, 3))
        st.metric("BSI", round(bsi, 3))
        st.metric("SAR", round(sar, 3))

        st.markdown("---")
        st.subheader("🧠 Salinisation")

        if pred == 2:
            st.error("🔴 Forte salinisation")
        elif pred == 1:
            st.warning("🟠 Modérée")
        else:
            st.success("🟢 Stable")

        # =========================
        # ANALYSE SPATIALE
        # =========================
        st.markdown("---")
        st.subheader("🧭 Analyse spatiale")

        if ndvi < 0.3:
            st.write("🌱 Faible végétation")
        if ndwi < 0:
            st.write("💧 Stress hydrique")
        if bsi > 0.5:
            st.write("🧂 Sol potentiellement salin")

else:
    st.info("✏️ Dessine une zone sur la carte pour analyser")

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; margin-top:40px; color:gray;">
🌍 GEOAI - USSEIN | Salinisation des sols<br>
✍️ Magatte GUEYE
</div>
""", unsafe_allow_html=True)