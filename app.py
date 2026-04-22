import streamlit as st
import joblib
import numpy as np
import json

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="GEOAI - Salinité des sols",
    page_icon="🌍",
    layout="centered"
)

# =========================
# STYLE UI
# =========================
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# =========================
# CHARGEMENT MODELE
# =========================
model = joblib.load("salinity_model_gandiol.pkl")

# =========================
# TITRE
# =========================
st.title("🌍 GEOAI - Salinisation des sols")
st.subheader("📍 Gandiol - Sénégal")

st.markdown("---")

# =========================
# INPUTS (⚠️ UNIQUEMENT VARIABLES DU MODELE)
# =========================
ndvi = st.slider("NDVI (végétation)", -1.0, 1.0, 0.2)
ndwi = st.slider("NDWI (humidité)", -1.0, 1.0, 0.1)
bsi  = st.slider("BSI (sol nu)", -1.0, 1.0, 0.3)

# =========================
# DEBUG IMPORTANT
# =========================
st.write("🔎 Modèle attend :", model.n_features_in_, "features")

# =========================
# PREDICTION
# =========================
if st.button("🔍 Lancer la prédiction"):

    # ⚠️ IMPORTANT : EXACT MATCH MODELE
    X = np.array([[ndvi, ndwi, bsi]])

    st.write("DEBUG X shape:", X.shape)

    pred = model.predict(X)[0]

    # =========================
    # RESULTATS
    # =========================
    if pred == 2:
        st.error("🔴 Forte salinisation")
        st.write("Risque élevé d'intrusion saline")

    elif pred == 1:
        st.warning("🟠 Salinisation modérée")
        st.write("Zone en transition")

    else:
        st.success("🟢 Zone stable")

# =========================
# FOOTER / SIGNATURE
# =========================
st.markdown(
    """
    <div class="footer">
    🌍 Projet GEOAI - Salinité des sols<br>
    ✍️ Magatte GUEYE
    </div>
    """,
    unsafe_allow_html=True
)