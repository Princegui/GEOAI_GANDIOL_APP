import streamlit as st
import joblib
import numpy as np

# =========================
# CHARGEMENT DU MODÈLE
# =========================
model = joblib.load("salinity_model_gandiol.pkl")

# =========================
# TITRE DE L’APP
# =========================
st.title("🌍 GEOAI - Détection de la salinisation des sols")
st.subheader("Zone : Gandiol (Saint-Louis, Sénégal)")

# =========================
# DESCRIPTION
# =========================
st.write("""
Application basée sur Sentinel-2 + Machine Learning pour estimer la salinité des sols.
""")

# =========================
# INPUTS UTILISATEUR
# =========================
ndvi = st.slider("NDVI (végétation)", -1.0, 1.0, 0.2)
ndwi = st.slider("NDWI (humidité)", -1.0, 1.0, 0.1)
bsi  = st.slider("BSI (sol nu / salinité)", -1.0, 1.0, 0.3)

# =========================
# PRÉDICTION
# =========================
if st.button("🔍 Lancer la prédiction"):
    features = np.array([[ndvi, ndwi, bsi]])
    prediction = model.predict(features)[0]

    # =========================
    # RÉSULTATS
    # =========================
    if prediction == 2:
        st.error("⚠ Forte salinisation détectée")
    elif prediction == 1:
        st.warning("⚠ Salinisation modérée")
    else:
        st.success("✔ Zone saine")

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.title("Projet GEOAI")
st.sidebar.info("Sentinel-2 Copernicus")
st.sidebar.info("Modèle : Random Forest")
st.sidebar.info("Université / Mémoire / Master GEOAI")