import streamlit as st
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import ee
import geemap
import json

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="GEOAI - Salinité des sols au Sénégal",
    page_icon="🌍",
    layout="wide"
)

# =========================
# LOGO USSEIN
# =========================
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://www.ussein.sn/wp-content/uploads/2024/12/USSEIN-LOGO-copie.png" width="140">
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main { background-color: #f4f9f9; }

h1 {
    color: #0b3d91;
    text-align: center;
}

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
# MODELE
# =========================
model = joblib.load("salinity_model_gandiol.pkl")

# =========================
# TITRE
# =========================
st.title("🌍 GEOAI - Cartographie de la salinisation des sols")
st.subheader("📍 Sénégal")

st.markdown("---")

# =========================
# MESSAGE INSTRUCTION CARTE
# =========================
st.info("✏️ Dessine un polygone sur la carte pour lancer l’analyse GEOAI")

ee.Authenticate()
ee.Initialize(project="mapathon-479420")
# =========================
# SELECTION PAR COORDONNEES
# =========================
st.subheader("📍 Sélection optionnelle par coordonnées GPS")

coord_text = st.text_area(
    "Entrer les coordonnées (format : lat,lon par ligne)",
    placeholder="Exemple :\n15.82,-16.50\n15.82,-16.45\n15.78,-16.45\n15.78,-16.50"
)

polygon_coords = []

if coord_text.strip() != "":
    try:
        lines = coord_text.strip().split("\n")

        polygon_coords = [
            [float(line.split(",")[0]), float(line.split(",")[1])]
            for line in lines
        ]

        st.success("📍 Polygone chargé avec succès")

    except:
        st.error("Format invalide. Utiliser : latitude,longitude")
        
        
    # =========================
# SELECTION TEMPORELLE
# =========================
st.subheader("📅 Évolution temporelle de la salinisation")

annee = st.slider(
    "Choisir une année d’analyse",
    min_value=2016,
    max_value=2025,
    value=2020,
    step=1
)

st.caption(f"Analyse simulée pour l'année : {annee}")


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

# =========================
# AJOUT POLYGONE COORDONNEES
# =========================
if polygon_coords:

    folium.Polygon(
        locations=polygon_coords,
        color="red",
        fill=True,
        fill_opacity=0.3,
        tooltip="Zone sélectionnée par coordonnées"
    ).add_to(m)

    # Centrage automatique carte
    m.location = polygon_coords[0]
    
# Outil dessin polygonal
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
# =========================
# LABELS (TOPONYMIE SEULE)
# =========================
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}",
    attr="Google Labels",
    name="Toponymie",
    overlay=True,
    control=True
).add_to(m)

# =========================
# CONVERSION POLYGONE STREAMLIT → GEE
# =========================

gee_polygon = None

if map_data and map_data.get("last_active_drawing"):

    geom = map_data["last_active_drawing"]["geometry"]

    if geom["type"] == "Polygon":

        coords = geom["coordinates"]

        gee_polygon = ee.Geometry.Polygon(coords)
        
# =========================
# CALCUL INDICES SENTINEL-2
# =========================

def compute_indices(year, region):

    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )

    ndvi = collection.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = collection.normalizedDifference(["B3", "B8"]).rename("NDWI")

    bsi = collection.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {
            "SWIR": collection.select("B11"),
            "RED": collection.select("B4"),
            "NIR": collection.select("B8"),
            "BLUE": collection.select("B2"),
        }
    ).rename("BSI")

    stats = (
        ndvi.addBands(ndwi)
        .addBands(bsi)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            maxPixels=1e9
        )
    )

    return stats.getInfo()


map_data = st_folium(m, height=550, width=1000)

st.markdown("---")

folium.LayerControl().add_to(m)
# =========================
# ANALYSE GEOAI
# =========================
if (map_data and map_data.get("last_active_drawing")) or polygon_coords:

    st.success("📍 Zone sélectionnée détectée")

    # =========================
    # SIMULATION INDICES (remplaçable Sentinel-2)
    # =========================
    if gee_polygon:

        indices = compute_indices(annee, gee_polygon)

        ndvi_zone = indices["NDVI"]
        ndwi_zone = indices["NDWI"]
        bsi_zone  = indices["BSI"]
    # prédiction
    X_zone = np.array([[ndvi_zone, ndwi_zone, bsi_zone]])
   
    pred = model.predict(X_zone)[0]
    # =========================
# ANALYSE SPATIALE GEOAI
# =========================

    st.markdown("---")
    st.subheader("🧭 Analyse spatiale de la salinisation")
    # =========================
    # DASHBOARD
    # =========================
    col_map, col_stats = st.columns([2, 1])
    
    if map_data and map_data.get("last_active_drawing"):

        geom = map_data["last_active_drawing"]["geometry"]

    if geom["type"] == "Polygon":

        coords = geom["coordinates"][0]

        # estimation surface approximative (km²)
        import math

        def polygon_area(coords):
            area = 0
            for i in range(len(coords)-1):
                x1, y1 = coords[i]
                x2, y2 = coords[i+1]
                area += x1*y2 - x2*y1
            return abs(area)/2

        surface_estimee = polygon_area(coords)

        st.metric("📐 Surface estimée analysée (approx.)", f"{round(surface_estimee,3)} km²")
    if pred == 2:

     st.error("Zone prioritaire d'intervention")

    st.write("""
Analyse spatiale :
- Sol fortement dégradé
- Risque de perte agricole élevé
- Probable intrusion saline
- Surveillance urgente recommandée
""")

elif pred == 1:

    st.warning("Zone de transition écologique")

    st.write("""
Analyse spatiale :
- Dégradation progressive observée
- Stress hydrique potentiel
- Zone tampon agricole fragile
""")

else:

    st.success("Zone stable écologiquement")

    st.write("""
Analyse spatiale :
- Couverture végétale satisfaisante
- Bonne humidité du sol
- Faible pression saline détectée
""")
st.markdown("### 📊 Diagnostic spectral")

if ndvi_zone < 0.3:
    st.write("🌱 Faible activité végétale détectée")

if ndwi_zone < 0:
    st.write("💧 Déficit hydrique probable")

if bsi_zone > 0.4:
    st.write("🧂 Forte exposition du sol nu (signature saline possible)")
    
st.markdown("### 📈 Tendance temporelle estimée")

if annee < 2019:
    st.write("Situation historique antérieure à intensification saline récente")

elif 2019 <= annee <= 2021:
    st.write("Phase intermédiaire de transformation du paysage")

else:
    st.write("Conditions récentes — pression saline actuelle probable")
    # -------------------------
    # -------------------------
    # RESULTATS
    # -------------------------
    with col_stats:
        st.subheader("📊 Indices spectraux")

        st.metric("NDVI", round(ndvi_zone, 3))
        st.metric("NDWI", round(ndwi_zone, 3))
        st.metric("BSI", round(bsi_zone, 3))

        st.markdown("---")
        st.subheader("📈 Evolution temporelle")

        st.write(f"Analyse de la salinité estimée pour l'année {annee}")
        st.subheader("🧠 Prédiction GEOAI")

        if pred == 2:
            st.error("🔴 Forte salinisation")
            st.write("Zone fortement dégradée — risque élevé")
        elif pred == 1:
            st.warning("🟠 Salinisation modérée")
            st.write("Zone en transition — surveillance nécessaire")
        else:
            st.success("🟢 Zone stable")
            st.write("Faible risque de salinisation")


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
🌍 Projet GEOAI - Salinisation des sols <br>
🏫 Université : USSEIN (Master Géomatique) <br>
✍️ Magatte GUEYE
</div>
""", unsafe_allow_html=True)