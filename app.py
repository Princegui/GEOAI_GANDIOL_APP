# =========================
# IMPORTS
# =========================
import streamlit as st
import joblib
import json
import numpy as np


# =========================
# CONFIGURATION DE LA PAGE
# =========================
st.set_page_config(
    page_title="GEOAI - Salinité des sols",
    page_icon="🌍",
    layout="centered"
)

# =========================
# STYLE CSS (UI COLOREE)
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f9f9;
    }

    h1 {
        color: #0b3d91;
        text-align: center;
    }

    h3 {
        color: #1b7f5a;
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
        margin-top: 50px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# CHARGEMENT DU MODELE ML
# =========================
model = joblib.load("salinity_model_gandiol.pkl")


# =========================
# CHARGEMENT DES METADONNEES (FEATURES OFFICIELLES)
# =========================
with open("model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]


# =========================
# INTERFACE UTILISATEUR
# =========================
st.title("🌍 GEOAI - Détection de la salinisation des sols")
st.subheader("📍 Gandiol - Sénégal")

st.markdown("---")

st.markdown("## 🔬 Variables environnementales")


# =========================
# INPUT UTILISATEUR (SLIDERS)
# ⚠️ DOIVENT RESPECTER FEATURES EXACTES
# =========================
inputs = {}

inputs["ndvi"] = st.slider("NDVI (végétation)", -1.0, 1.0, 0.2)
inputs["ndwi"] = st.slider("NDWI (humidité)", -1.0, 1.0, 0.1)
inputs["bsi"] = st.slider("BSI (sol nu)", -1.0, 1.0, 0.3)

inputs["distance_mer"] = st.slider("Distance à la mer (km)", 0, 50, 5)
inputs["altitude"] = st.slider("Altitude (m)", 0, 100, 10)
inputs["humidite_sol"] = st.slider("Humidité du sol", 0.0, 1.0, 0.4)
inputs["pluie"] = st.slider("Pluie (mm)", 0, 300, 120)

inputs["landcover"] = st.selectbox(
    "Occupation du sol",
    [0, 1, 2, 3]  # codage numérique obligatoire pour ML
)


st.markdown("---")


# =========================
# SIDEBAR : INFO MODELE
# =========================
st.sidebar.title("🔒 Model Info")
st.sidebar.write("Features attendues :")
st.sidebar.write(FEATURES)


# =========================
# CONSTRUCTION DU VECTEUR D'ENTREE
# =========================
X = np.array([[ndvi, ndwi, bsi]])

st.write("DEBUG shape X :", X.shape)


# =========================
# PREDICTION
# =========================
if st.button("🔍 Lancer la prédiction"):

    # Vérification sécurité (anti crash sklearn)
    if X.shape[1] == len(FEATURES):

        pred = model.predict(X)[0]

        # =========================
        # INTERPRETATION RESULTAT
        # =========================
        if pred == 2:
            st.error("🔴 Forte salinisation détectée")
            st.write("Risque élevé d’intrusion saline côtière.")

        elif pred == 1:
            st.warning("🟠 Salinisation modérée")
            st.write("Zone en transition écologique.")

        else:
            st.success("🟢 Zone stable")
            st.write("Faible risque de salinisation.")

    else:
        st.error("❌ Incompatibilité modèle / features")


# =========================
# FOOTER (SIGNATURE)
# =========================
st.markdown(
    """
    <div class="footer">
    ---<br>
    🌍 Projet GEOAI - Analyse de la salinisation des sols<br>
    ✍️ Magatte GUEYE
    </div>
    """,
    unsafe_allow_html=True
)