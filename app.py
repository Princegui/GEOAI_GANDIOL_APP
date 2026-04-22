import streamlit as st
import joblib
import numpy as np

# =========================
# CONFIG PAGE (THEME)
# =========================
st.set_page_config(
    page_title="GEOAI - Salinité des sols",
    page_icon="🌍",
    layout="centered"
)

# =========================
# STYLE CSS (COULEURS)
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
# CHARGEMENT MODELE
# =========================
model = joblib.load("salinity_model_gandiol.pkl")

# =========================
# TITRE
# =========================
st.title("🌍 GEOAI - Détection de la salinisation des sols")
st.subheader("📍 Gandiol - Sénégal")

st.markdown("---")

# =========================
# INPUTS
# =========================
st.markdown("## 🔬 Variables environnementales")

col1, col2 = st.columns(2)

with col1:
    ndvi = st.slider("NDVI (végétation)", -1.0, 1.0, 0.2)
    ndwi = st.slider("NDWI (humidité)", -1.0, 1.0, 0.1)
    bsi = st.slider("BSI (sol nu)", -1.0, 1.0, 0.3)
    distance_mer = st.slider("Distance mer (km)", 0, 50, 5)

with col2:
    altitude = st.slider("Altitude (m)", 0, 100, 10)
    humidite_sol = st.slider("Humidité sol", 0.0, 1.0, 0.4)
    pluie = st.slider("Pluie (mm)", 0, 300, 120)
    landcover = st.selectbox("Occupation du sol", {
        "Mangrove": 0,
        "Agriculture": 1,
        "Urbain": 2,
        "Sol nu": 3
    })

st.markdown("---")

# =========================
# PRÉDICTION
# =========================
if st.button("🔍 Lancer la prédiction GEOAI"):

    X = np.array([[
        ndvi,
        ndwi,
        bsi,
        distance_mer,
        altitude,
        humidite_sol,
        pluie,
        landcover
    ]])

    pred = model.predict(X)[0]

    st.markdown("## 📊 Résultat")

    if pred == 2:
        st.error("🔴 Forte salinisation détectée")
        st.write("Zone à haut risque d'intrusion saline côtière.")

    elif pred == 1:
        st.warning("🟠 Salinisation modérée")
        st.write("Zone en transition écologique.")

    else:
        st.success("🟢 Zone stable")
        st.write("Faible risque de salinisation.")

# =========================
# FOOTER / SIGNATURE
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