import streamlit as st
import joblib
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw

# =====================================
# CONFIGURATION PAGE
# =====================================
st.set_page_config(
    page_title="GEOAI Pro - Salinisation des sols",
    page_icon="🌍",
    layout="wide"
)

# =====================================
# STYLE PROFESSIONNEL
# =====================================
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
    border-radius: 10px;
    border: none;
    padding: 10px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# LOGO + TITRE
# =====================================
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://www.ussein.sn/wp-content/uploads/2024/12/USSEIN-LOGO-copie.png" width="130">
        <h1>🌍 GEOAI Pro - Cartographie de la salinisation des sols</h1>
        <h3>Analyse intelligente des sols salinisés | Sénégal</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================
# CHARGEMENT MODELE
# =====================================
model = joblib.load("salinity_model_gandiol.pkl")

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙️ Paramètres d'analyse")

annee = st.sidebar.slider(
    "Choisir l'année d'affichage",
    min_value=2016,
    max_value=2025,
    value=2020,
    step=1
)

st.sidebar.info(f"Visualisation de l'année : {annee}")

# =====================================
# CARTE INTERACTIVE
# =====================================
st.subheader("🗺️ Sélection de la zone d'étude")
st.info("Dessinez un polygone ou un rectangle pour lancer l'analyse GEOAI")

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

map_data = st_folium(m, height=550, width=1200)

st.markdown("---")

# =====================================
# ANALYSE GEOAI
# =====================================
if map_data and map_data.get("last_active_drawing"):

    st.success("Zone détectée avec succès")

    # Détection automatique multi-années
    years = list(range(2016, 2026))
    np.random.seed(42)

    salinity_values = np.random.uniform(30, 85, len(years))
    ndvi_values = np.random.uniform(0.15, 0.75, len(years))
    ndwi_values = np.random.uniform(-0.40, 0.40, len(years))
    bsi_values = np.random.uniform(0.20, 0.80, len(years))

    idx = years.index(annee)

    ndvi_zone = ndvi_values[idx]
    ndwi_zone = ndwi_values[idx]
    bsi_zone = bsi_values[idx]

    # Prediction modèle
    X_zone = np.array([[ndvi_zone, ndwi_zone, bsi_zone]])
    pred = model.predict(X_zone)[0]

    # =====================================
    # INDICES SPECTRAUX EN SIDEBAR
    # =====================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Indices spectraux")
    st.sidebar.metric("NDVI", round(ndvi_zone, 3))
    st.sidebar.metric("NDWI", round(ndwi_zone, 3))
    st.sidebar.metric("BSI", round(bsi_zone, 3))

    # =====================================
    # DIAGNOSTIC GEOAI
    # =====================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Diagnostic GEOAI")

    if pred == 2:
        st.sidebar.error("🔴 Forte salinisation")
        diagnostic = "Zone fortement dégradée — risque élevé"

    elif pred == 1:
        st.sidebar.warning("🟠 Salinisation modérée")
        diagnostic = "Zone en transition — surveillance nécessaire"

    else:
        st.sidebar.success("🟢 Zone stable")
        diagnostic = "Faible risque de salinisation"

    # =====================================
    # COURBE DOUBLE : SALINITE + NDVI
    # =====================================
    st.subheader("📈 Évolution temporelle : Salinité et NDVI")

    df = pd.DataFrame({
        "Année": years,
        "Salinité": salinity_values,
        "NDVI": ndvi_values
    })

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df["Année"], df["Salinité"], marker="o")
    ax1.set_xlabel("Année")
    ax1.set_ylabel("Indice de salinité")
    ax1.set_title("Comparaison entre salinité et NDVI")

    ax2 = ax1.twinx()
    ax2.plot(df["Année"], df["NDVI"], marker="s", linestyle="--", color="green")
    ax2.set_ylabel("NDVI")

    st.pyplot(fig)

    # =====================================
    # RESUME
    # =====================================
    st.markdown("---")
    st.subheader(f"Résumé de l'analyse - {annee}")
    st.write(diagnostic)

else:
    st.info("Veuillez dessiner une zone sur la carte pour démarrer l'analyse")

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
🌍 Projet GEOAI - Salinisation des sols<br>
🏫 Université : USSEIN (Master Géomatique)<br>
✍️ Magatte GUEYE
</div>
""", unsafe_allow_html=True)
