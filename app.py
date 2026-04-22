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
st.subheader("📍 Gandiol - Sénégal")

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

map_data = st_folium(m, height=550, width=1000)

st.markdown("---")

# =========================
# ANALYSE GEOAI
# =========================
if map_data and map_data.get("last_active_drawing"):

    st.success("📍 Zone sélectionnée détectée")

    # =========================
    # SIMULATION INDICES (remplaçable Sentinel-2)
    # =========================
    ndvi_zone = np.random.uniform(0.1, 0.7)
    ndwi_zone = np.random.uniform(-0.4, 0.4)
    bsi_zone  = np.random.uniform(0.2, 0.8)

    # prédiction
    X_zone = np.array([[ndvi_zone, ndwi_zone, bsi_zone]])
    pred = model.predict(X_zone)[0]

    # =========================
    # DASHBOARD
    # =========================
    col_map, col_stats = st.columns([2, 1])

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

else:
    st.info("✏️ Dessine un polygone sur la carte pour lancer l’analyse GEOAI")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
🌍 Projet GEOAI - Salinisation des sols (Gandiol)<br>
🏫 Université : USSEIN<br>
✍️ Magatte GUEYE
</div>
""", unsafe_allow_html=True)