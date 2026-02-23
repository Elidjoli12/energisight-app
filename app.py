import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Configuration de la page ─────────────────────────────────────
st.set_page_config(
    page_title="EnergiSight — GreenSight Lomé",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style CSS personnalisé ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background-color: #0a0f1e;
        color: #e8f0fe;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1f3c 50%, #071a2e 100%);
    }

    /* Header principal */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #00d4aa, #00a8ff, #00d4aa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 0;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        font-size: 1.1rem;
        color: #7cb9e8;
        margin-top: 0.2rem;
        letter-spacing: 0.05em;
    }

    /* Cartes métriques */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s;
    }

    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.6);
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #00d4aa;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #7cb9e8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    /* Résultat principal */
    .result-box {
        background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(0,168,255,0.1));
        border: 2px solid #00d4aa;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-value {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #00d4aa;
    }

    .result-label {
        font-size: 1rem;
        color: #7cb9e8;
        margin-top: 0.5rem;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(0, 212, 170, 0.15) !important;
    }

    /* Bouton */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #00a8ff);
        color: #0a0f1e;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
        letter-spacing: 0.05em;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.4);
    }

    /* Sliders et inputs */
    .stSlider > div > div > div > div {
        background: #00d4aa !important;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,170,0.4), transparent);
        margin: 1.5rem 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        border: 1px solid rgba(0, 212, 170, 0.4);
        color: #00d4aa;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #7cb9e8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Chargement du modèle ─────────────────────────────────────────
@st.cache_resource
def charger_modele():
    return joblib.load('energisight_model_sans_estar_optimise.pkl')

modele = charger_modele()

# ── Types de bâtiments disponibles ──────────────────────────────
TYPES_BATIMENTS = [
    'Office', 'Warehouse', 'K-12 School', 'Retail Store', 'Hotel',
    'Worship Facility', 'Distribution Center', 'Mixed Use Property',
    'Medical Office', 'Self-Storage Facility', 'University',
    'Senior Care Community', 'Refrigerated Warehouse', 'Restaurant',
    'Hospital', 'Laboratory', 'Other', 'Supermarket / Grocery Store',
    'Small- and Mid-Sized Office', 'Large Office', 'Residence Hall'
]

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("# 🌍", unsafe_allow_html=True)

with col_title:
    st.markdown('<p class="hero-title">EnergiSight</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Prédiction de Consommation Énergétique — GreenSight Lomé &nbsp;'
        '<span class="badge">IA • Bâtiments</span></p>',
        unsafe_allow_html=True
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SIDEBAR — Paramètres du bâtiment
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏗️ Caractéristiques du Bâtiment")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    type_batiment = st.selectbox(
        "Type d'usage principal",
        options=TYPES_BATIMENTS,
        index=0
    )

    st.markdown("<br>", unsafe_allow_html=True)
    surface_totale = st.number_input(
        "Surface totale (m²)",
        min_value=100,
        max_value=500000,
        value=5000,
        step=100
    )

    surface_parking = st.number_input(
        "Surface parking (m²)",
        min_value=0,
        max_value=100000,
        value=500,
        step=50
    )

    surface_usage = st.number_input(
        "Surface usage principal (m²)",
        min_value=100,
        max_value=500000,
        value=4000,
        step=100
    )

    st.markdown("<br>", unsafe_allow_html=True)
    nb_etages = st.slider("Nombre d'étages", 1, 50, 5)
    nb_batiments = st.slider("Nombre de bâtiments", 1, 20, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    annee_construction = st.slider(
        "Année de construction",
        min_value=1900,
        max_value=2016,
        value=1990
    )

    st.markdown("<br>", unsafe_allow_html=True)
    distance_centre = st.slider(
        "Distance au centre-ville (km)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.1
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Prédire la Consommation")

# ════════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ════════════════════════════════════════════════════════════════

# Conversion m² → sqft (le modèle a été entraîné en sqft)
M2_TO_SQFT = 10.7639

if predict_btn:
    # ── Préparation des données ──────────────────────────────────
    age_batiment = 2016 - annee_construction

    input_data = pd.DataFrame({
        'PropertyGFATotal':          [surface_totale * M2_TO_SQFT],
        'PropertyGFAParking':        [surface_parking * M2_TO_SQFT],
        'NumberofFloors':            [nb_etages],
        'NumberofBuildings':         [nb_batiments],
        'BuildingAge':               [age_batiment],
        'DistanceCentre':            [distance_centre],
        'LargestPropertyUseTypeGFA': [surface_usage * M2_TO_SQFT],
        'LargestPropertyUseType':    [type_batiment]
    })

    # ── Prédiction ───────────────────────────────────────────────
    log_pred = modele.predict(input_data)[0]
    consommation_kbtu = np.expm1(log_pred)
    consommation_kwh  = consommation_kbtu * 0.293071
    consommation_mwh  = consommation_kwh / 1000
    co2_estime        = consommation_kbtu * 0.053  # facteur moyen Seattle

    # ── Résultat principal ───────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-value">{consommation_mwh:,.0f} MWh</div>
            <div class="result-label">Consommation énergétique annuelle estimée</div>
            <br>
            <div style="color:#7cb9e8; font-size:0.9rem;">
                soit &nbsp;<strong style="color:#00d4aa">{consommation_kwh:,.0f} kWh</strong>&nbsp;
                / an
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{co2_estime:,.0f}</div>
            <div class="metric-label">Tonnes CO₂ estimées / an</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        eui = consommation_kbtu / (surface_totale * M2_TO_SQFT) if surface_totale > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eui:.1f}</div>
            <div class="metric-label">EUI (kBtu/sqft/an)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Graphique jauge ──────────────────────────────────────────
    col_gauge, col_breakdown = st.columns(2)

    with col_gauge:
        st.markdown("#### 📊 Niveau de Consommation")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=eui,
            title={'text': "EUI (kBtu/sqft/an)", 'font': {'color': '#7cb9e8', 'size': 14}},
            number={'font': {'color': '#00d4aa', 'size': 36}},
            gauge={
                'axis': {'range': [0, 300], 'tickcolor': '#7cb9e8'},
                'bar': {'color': '#00d4aa'},
                'bgcolor': 'rgba(255,255,255,0.05)',
                'bordercolor': 'rgba(0,212,170,0.3)',
                'steps': [
                    {'range': [0, 50],   'color': 'rgba(0,212,170,0.15)'},
                    {'range': [50, 150], 'color': 'rgba(255,165,0,0.15)'},
                    {'range': [150, 300],'color': 'rgba(255,50,50,0.15)'}
                ],
                'threshold': {
                    'line': {'color': '#ff6b6b', 'width': 3},
                    'thickness': 0.75,
                    'value': 150
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e8f0fe'},
            height=280,
            margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_breakdown:
        st.markdown("#### 🔍 Profil du Bâtiment")
        categories = ['Surface', 'Étages', 'Âge', 'Distance', 'Usage']
        # Normalisation 0-100 pour radar
        values = [
            min(surface_totale / 5000 * 100, 100),
            min(nb_etages / 50 * 100, 100),
            min(age_batiment / 116 * 100, 100),
            min(distance_centre / 14 * 100, 100),
            min(surface_usage / surface_totale * 100 if surface_totale > 0 else 50, 100)
        ]
        values += [values[0]]
        categories += [categories[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 212, 170, 0.15)',
            line=dict(color='#00d4aa', width=2),
            marker=dict(color='#00d4aa', size=6)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(255,255,255,0.03)',
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickcolor='#7cb9e8', gridcolor='rgba(124,185,232,0.2)'
                ),
                angularaxis=dict(tickcolor='#7cb9e8', gridcolor='rgba(124,185,232,0.2)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e8f0fe'},
            height=280,
            margin=dict(t=40, b=10, l=40, r=40),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Recommandations ──────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 💡 Recommandations EnergiSight")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        if eui > 150:
            st.error("🔴 **Consommation élevée**\n\nAudit énergétique urgent recommandé. Isolation et systèmes CVC à réviser.")
        elif eui > 80:
            st.warning("🟡 **Consommation modérée**\n\nOptimisation possible. Éclairage LED et gestion intelligente recommandés.")
        else:
            st.success("🟢 **Bonne performance**\n\nBâtiment efficace. Maintenir les bonnes pratiques actuelles.")

    with col_r2:
        potentiel_eco = consommation_kwh * 0.20
        st.info(f"💰 **Potentiel d'économie**\n\nAvec 20% d'optimisation :\n**{potentiel_eco:,.0f} kWh/an** économisés")

    with col_r3:
        arbres = int(co2_estime * 45)
        st.info(f"🌳 **Équivalent carbone**\n\n{co2_estime:,.0f} tonnes CO₂/an\n≈ **{arbres:,} arbres** à planter")

else:
    # ── État initial — instructions ──────────────────────────────
    st.markdown("### Comment utiliser EnergiSight ?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">📋</div>
            <div style="font-family:'Syne',sans-serif; font-weight:600; 
                        color:#e8f0fe; margin:0.5rem 0">Étape 1</div>
            <div style="color:#7cb9e8; font-size:0.9rem">
                Renseignez les caractéristiques physiques du bâtiment dans le panneau de gauche
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">🔮</div>
            <div style="font-family:'Syne',sans-serif; font-weight:600; 
                        color:#e8f0fe; margin:0.5rem 0">Étape 2</div>
            <div style="color:#7cb9e8; font-size:0.9rem">
                Cliquez sur "Prédire la Consommation" pour obtenir une estimation instantanée
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">📊</div>
            <div style="font-family:'Syne',sans-serif; font-weight:600; 
                        color:#e8f0fe; margin:0.5rem 0">Étape 3</div>
            <div style="color:#7cb9e8; font-size:0.9rem">
                Analysez les résultats et recommandations pour optimiser votre bâtiment
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Métriques du modèle
    st.markdown("### 🤖 Performance du Modèle")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">72%</div>
            <div class="metric-label">Précision R²</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">1,620</div>
            <div class="metric-label">Bâtiments analysés</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">0</div>
            <div class="metric-label">Audit ENERGY STAR requis</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">Lomé</div>
            <div class="metric-label">Ville cible</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#3a5068; font-size:0.8rem; padding:1rem 0">
    🌍 EnergiSight by <strong style="color:#00d4aa">GreenSight Lomé</strong> &nbsp;•&nbsp; 
    Modèle : Gradient Boosting &nbsp;•&nbsp; Dataset : Seattle 2016 &nbsp;•&nbsp;
    Formation D-CLIC OIF
</div>
""", unsafe_allow_html=True)
