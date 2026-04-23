# ============================================================
# 🌍 GEOAI STREAMLIT APP FINAL CLOUD VERSION
# ============================================================

import streamlit as st
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import os

# =========================
# CONFIG PAGE
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
# LOAD MODEL SAFE
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "geoai_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Impossible de charger geoai_model.pkl")
    st.stop()

# =========================
# TITRE
# =========================

st.title("🌍 GEOAI - Cartographie de la salinisation des sols")
st.subheader("📍 Zone test : Gandiol (Sénégal)")

# =========================
# SLIDER TEMPOREL
# =========================

annee = st.slider("📅 Année d’analyse", 2016, 2024, 2022)

# =========================
# CARTE INTERACTIVE
# =========================

st.subheader("🗺️ Dessine une zone d’analyse")

m = folium.Map(location=[15.8, -16.5], zoom_start=11)

Draw(export=True).add_to(m)

map_data = st_folium(m, height=550)

# =========================
# ANALYSE
# =========================

if map_data and map_data.get("last_active_drawing"):

    st.success("Zone sélectionnée")

    np.random.seed(annee)

    ndvi = np.random.uniform(0.1, 0.8)
    ndwi = np.random.uniform(-0.4, 0.4)
    bsi  = np.random.uniform(0.2, 0.9)
    sar  = np.random.uniform(0.2, 1.0)

    X = np.array([[ndvi, ndwi, bsi, sar]])

    pred = model.predict(X)[0]

    col1, col2 = st.columns([2,1])

    with col2:

        st.subheader("Indices")

        st.metric("NDVI", round(ndvi,3))
        st.metric("NDWI", round(ndwi,3))
        st.metric("BSI", round(bsi,3))
        st.metric("SAR", round(sar,3))

        st.subheader("Diagnostic salinité")

        if pred == 2:
            st.error("🔴 Forte salinisation")
        elif pred == 1:
            st.warning("🟠 Salinisation modérée")
        else:
            st.success("🟢 Zone stable")

else:

    st.info("✏️ Dessine un polygone sur la carte")

# =========================
# FOOTER
# =========================

st.markdown("""
<div style="text-align:center;color:gray;margin-top:40px">
🌍 GEOAI – USSEIN (Master Géomatique)  
Magatte GUEYE
</div>
""", unsafe_allow_html=True)