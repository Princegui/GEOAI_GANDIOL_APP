import streamlit as st
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="GEOAI - Salinité des sols",
    page_icon="🌍",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main { background-color: #f4f9f9; }
h1 { color: #0b3d91; text-align: center; }

.stButton>button {
    background-color: #0b3d91;
    color: white;
    border-radius: 10px;
    padding: 10px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CHARGEMENT MODELE
# =========================
model = joblib.load("salinity_model_gandiol.pkl")

# =========================
# TITRE
# =========================
st.title("🌍 GEOAI - Cartographie de la salinisation des sols")
st.subheader("📍 Zone d’étude : Gandiol - Sénégal")

st.markdown("---")

# =========================
# CARTE INTERACTIVE
# =========================
st.subheader("🗺️ Carte interactive - Sélection de zone")

m = folium.Map(location=[15.8, -16.5], zoom_start=11, tiles="OpenStreetMap")

# Satellite Google
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google",
    name="Satellite",
    overlay=False,
    control=True
).add_to(m)

# Outil dessin
draw = Draw(
    export=True,
    draw_options={
        "polyline": False,
        "rectangle": True,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": True
    }
)
draw.add_to(m)

map_data = st_folium(m, height=550, width=1000)

st.markdown("---")

# =========================
# ANALYSE ZONE
# =========================
if map_data and map_data.get("last_active_drawing"):

    st.success("📍 Zone sélectionnée détectée")

    geom = map_data["last_active_drawing"]["geometry"]
    st.write("📌 Géométrie du polygone :")
    st.json(geom)

    # =========================
    # SIMULATION INDICES (remplacer par Sentinel-2 plus tard)
    # =========================
    ndvi_zone = np.random.uniform(0.1, 0.7)
    ndwi_zone = np.random.uniform(-0.4, 0.4)
    bsi_zone  = np.random.uniform(0.2, 0.8)

    st.subheader("📊 Indices spectraux moyens de la zone")

    col1, col2, col3 = st.columns(3)

    col1.metric("NDVI", round(ndvi_zone, 3))
    col2.metric("NDWI", round(ndwi_zone, 3))
    col3.metric("BSI", round(bsi_zone, 3))

    # =========================
    # PREDICTION MODELE
    # =========================
    X_zone = np.array([[ndvi_zone, ndwi_zone, bsi_zone]])
    pred = model.predict(X_zone)[0]

    st.subheader("🧠 Résultat de prédiction GEOAI")

    if pred == 2:
        st.error("🔴 Forte salinisation détectée")
        st.write("Zone fortement dégradée — risque élevé pour agriculture")
    elif pred == 1:
        st.warning("🟠 Salinisation modérée")
        st.write("Zone en transition — surveillance recommandée")
    else:
        st.success("🟢 Zone stable")
        st.write("Faible risque de salinisation")

else:
    st.info("✏️ Dessine un polygone sur la carte pour lancer l’analyse GEOAI")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
🌍 Projet GEOAI - Salinisation des sols (Gandiol)<br>
✍️ Magatte GUEYE
</div>
""", unsafe_allow_html=True)