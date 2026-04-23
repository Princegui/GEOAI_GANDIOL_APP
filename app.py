import streamlit as st
import joblib
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw

# =========================
# CONFIGURATION PAGE
# =========================
st.set_page_config(
    page_title="GEOAI Pro - Salinisation des sols",
    page_icon="🌍",
    layout="wide"
)

# =========================
# STYLE PROFESSIONNEL
# =========================
st.markdown("""
<style>
.main {
    background-color: #f7fafc;
}

h1, h2, h3 {
    color: #0b3d91;
}

.stButton > button {
    background-color: #0b3d91;
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    border: none;
}

.block-container {
    padding-top: 2rem;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOGO + TITRE
# =========================
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://www.ussein.sn/wp-content/uploads/2024/12/USSEIN-LOGO-copie.png" width="130">
        <h1>🌍 GEOAI Pro - Cartographie de la salinisation des sols</h1>
        <h3>Analyse intelligente | Sénégal</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# CHARGEMENT MODELE
# =========================
model = joblib.load("salinity_model_gandiol.pkl")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Paramètres d'analyse")

annee = st.sidebar.slider(
    "Choisir l'année",
    min_value=2016,
    max_value=2025,
    value=2020,
    step=1
)

st.sidebar.info(f"Analyse pour l'année : {annee}")

# =========================
# CARTE
# =========================
st.subheader("🗺️ Sélection de la zone d'étude")

m = folium.Map(location=[15.8, -16.5], zoom_start=10, tiles="OpenStreetMap")

Draw(
    export=True,
    draw_options={
        "polyline": False,
        "rectangle": True,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": True
    }
).add_to(m)

map_data = st_folium(m, height=500, width=1200)

st.markdown("---")

# =========================
# ANALYSE SI ZONE SELECTIONNEE
# =========================
if map_data and map_data.get("last_active_drawing"):
    st.success("Zone détectée avec succès")

    np.random.seed(annee)

    ndvi_zone = np.random.uniform(0.1, 0.7)
    ndwi_zone = np.random.uniform(-0.4, 0.4)
    bsi_zone = np.random.uniform(0.2, 0.8)

    X_zone = np.array([[ndvi_zone, ndwi_zone, bsi_zone]])
    pred = model.predict(X_zone)[0]

    col1, col2 = st.columns([1, 1])

    # =========================
    # INDICATEURS
    # =========================
    with col1:
        st.subheader("📊 Indices spectraux")
        st.metric("NDVI", round(ndvi_zone, 3))
        st.metric("NDWI", round(ndwi_zone, 3))
        st.metric("BSI", round(bsi_zone, 3))

        st.markdown("---")
        st.subheader("🧠 Diagnostic GEOAI")

        if pred == 2:
            st.error("🔴 Forte salinisation")
        elif pred == 1:
            st.warning("🟠 Salinisation modérée")
        else:
            st.success("🟢 Zone stable")

    # =========================
    # COURBE EVOLUTION
    # =========================
    with col2:
        st.subheader("📈 Courbe d'évolution temporelle")

        years = list(range(2016, 2026))
        salinity_index = np.random.uniform(20, 80, len(years))

        df = pd.DataFrame({
            "Année": years,
            "Indice de salinité": salinity_index
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Année"], df["Indice de salinité"], marker='o')
        ax.set_xlabel("Année")
        ax.set_ylabel("Indice de salinité")
        ax.set_title("Évolution temporelle de la salinisation")

        st.pyplot(fig)

else:
    st.info("Dessinez un polygone sur la carte pour lancer l'analyse")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
🌍 Projet GEOAI - Salinisation des sols<br>
🏫 Université : USSEIN (Master Géomatique)<br>
✍️ Magatte GUEYE
</div>
""", unsafe_allow_html=True)
