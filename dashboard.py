import sys
import asyncio

# --- CORRECTIF SYSTÈME (WINDOWS EVENT LOOP) ---
# Nécessaire pour éviter les conflits asynchrones sur Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# ----------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import nest_asyncio
from datetime import datetime, timedelta

# Importation des bibliothèques d'Intelligence Artificielle (Hugging Face)
from transformers import pipeline

# Importation du client API (Générateur de données)
from api_client import TwitterAPIClient

# Application du patch asynchrone pour Streamlit
nest_asyncio.apply()

# --- CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="Système d'Analyse IA & Crisis Room", layout="wide")

# Injection CSS pour une esthétique professionnelle (Style 'War Room')
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #1DA1F2; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #0d8ddb; color: white; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1DA1F2; }
</style>
""", unsafe_allow_html=True)

# Palette de couleurs standardisée
COLOR_MAP = {'Positif': '#00CC96', 'Négatif': '#EF553B', 'Neutre': '#7f7f7f'}

# --- 1. CHARGEMENT DU MODÈLE IA (CACHE GLOBAL) ---
@st.cache_resource
def load_sentiment_model():
    """
    Initialise et met en cache le modèle de Deep Learning (RoBERTa).
    Ce modèle est spécifiquement pré-entraîné sur des tweets multilingues (EN, FR, AR).
    """
    # Modèle SOTA (State-of-the-Art) pour l'analyse de tweets
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, top_k=None)

# --- 2. BARRE LATÉRALE : CONFIGURATION DE L'EXTRACTION ---
with st.sidebar:
    st.header("Paramètres de Recherche")
    st.caption("Moteur d'extraction via TwitterAPI.io")
    
    with st.form("api_form"):
        st.subheader("Filtres Sémantiques")
        all_words = st.text_input("Mots-clés (Intersection)", placeholder="ex: Crise Banque")
        exact_phrase = st.text_input("Phrase Exacte")
        hashtags = st.text_input("Hashtags", placeholder="#Finance")
        lang = st.selectbox("Langue Cible", ["Tout", "fr", "en", "ar"], index=1)

        st.subheader("Filtres Techniques & Temporels")
        with st.expander("Options Avancées"):
            st.info("Conseil : Une plage de dates large maximise la pertinence statistique.")
            c1, c2 = st.columns(2)
            since_date = c1.date_input("Date Début", datetime.now() - timedelta(days=30))
            until_date = c2.date_input("Date Fin", datetime.now())
            min_faves = st.number_input("Seuil Min. Engagement", 0)
            from_accts = st.text_input("Comptes Sources (@)")

        limit = st.number_input("Volume Cible (Tweets)", 10, 2000, 50)
        
        submitted = st.form_submit_button("🚀 Lancer l'Analyse IA")

    if submitted:
        client = TwitterAPIClient()
        
        params = {
            "all_words": all_words, "exact_phrase": exact_phrase,
            "hashtags": hashtags, "lang": lang,
            "min_faves": min_faves, "from_accounts": from_accts,
            "since": since_date.strftime("%Y-%m-%d"),
            "until": until_date.strftime("%Y-%m-%d")
        }

        # --- EXÉCUTION DYNAMIQUE AVEC FEEDBACK TEMPS RÉEL ---
        with st.status("Démarrage du protocole d'extraction...", expanded=True) as status:
            final_data = []
            
            for progress in client.fetch_tweets_generator(params, limit):
                
                if "error" in progress:
                    status.update(label="Erreur Critique API", state="error")
                    st.error(progress["error"])
                    break
                
                curr = progress['current_count']
                tgt = progress['target']
                
                # Mise à jour visuelle de la progression
                status.update(label=f"Acquisition des données ({curr}/{tgt}) - Veuillez patienter...", state="running")
                
                final_data = progress['data']
                
                if progress.get('finished'):
                    status.update(label="Extraction terminée avec succès.", state="complete", expanded=False)

            if final_data:
                st.success(f"Acquisition terminée : {len(final_data)} tweets indexés.")
                # Persistance locale des données brutes
                with open("api_data.json", "w", encoding="utf-8") as f:
                    json.dump(final_data, f, ensure_ascii=False)
                # Invalidation du cache pour forcer le retraitement IA
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Aucune donnée correspondante trouvée. Vérifiez vos filtres.")

# --- 3. TRAITEMENT ANALYTIQUE (IA + PANDAS) ---

@st.cache_data
def load_and_process_data():
    """
    Charge les données JSON et applique l'inférence du modèle IA.
    Cette fonction est mise en cache pour éviter de recalculer l'IA à chaque interaction.
    """
    if not os.path.exists("api_data.json"): return pd.DataFrame()
    try:
        with open("api_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except: return pd.DataFrame()
            
    if not data: return pd.DataFrame()
    
    df = pd.json_normalize(data)
    df['date'] = pd.to_datetime(df['date_iso'], errors='coerce')
    
    # Normalisation des métriques
    for col in ['metrics.likes', 'metrics.retweets', 'metrics.replies']:
        if col not in df.columns: df[col] = 0

    df['engagement'] = df['metrics.likes'] + df['metrics.retweets']

    # --- INFÉRENCE INTELLIGENCE ARTIFICIELLE ---
    sentiment_pipeline = load_sentiment_model()

    def get_ai_sentiment(text):
        """Analyse le texte via RoBERTa (Deep Learning)."""
        if not isinstance(text, str) or not text.strip(): 
            return 0.0, 'Neutre'
        
        try:
            # Troncature à 512 tokens (limite technique BERT)
            truncated_text = text[:512]
            results = sentiment_pipeline(truncated_text)[0]
            
            # Extraction des probabilités
            scores = {res['label']: res['score'] for res in results}
            
            # Mapping des labels du modèle (cardiffnlp utilise: positive, negative, neutral)
            pos_score = scores.get('positive', 0)
            neg_score = scores.get('negative', 0)
            neu_score = scores.get('neutral', 0)

            # Logique de décision (Argmax)
            if pos_score > neg_score and pos_score > neu_score:
                return pos_score, 'Positif'
            elif neg_score > pos_score and neg_score > neu_score:
                return -neg_score, 'Négatif' # Score négatif pour l'axe Y
            else:
                return 0.0, 'Neutre'
        except Exception:
            return 0.0, 'Neutre'
    
    if 'text' in df.columns:
        # Application du modèle (peut prendre quelques secondes selon le volume)
        with st.spinner("Analyse sémantique par Intelligence Artificielle en cours..."):
            result_series = df['text'].apply(lambda x: pd.Series(get_ai_sentiment(x)))
            df[['sentiment_score', 'sentiment_cat']] = result_series
    
    return df

df_raw = load_and_process_data()

# --- 4. TABLEAU DE BORD STRATÉGIQUE (UI) ---

st.title("🛡️ War Room : Analyse de Crise (AI Powered)")

if not df_raw.empty:
    
    # --- FILTRAGE DYNAMIQUE ---
    st.markdown("### 🔍 Segmentation des Données")
    col_filter, _ = st.columns([1, 2])
    with col_filter:
        selected_sentiments = st.multiselect(
            "Afficher les sentiments :",
            options=["Positif", "Négatif", "Neutre"],
            default=["Positif", "Négatif", "Neutre"]
        )
    
    # Filtrage du DataFrame
    if 'sentiment_cat' in df_raw.columns:
        df = df_raw[df_raw['sentiment_cat'].isin(selected_sentiments)]
    else:
        df = df_raw

    st.divider()

    # --- INDICATEURS DE PERFORMANCE (KPIs) ---
    k1, k2, k3 = st.columns(3)
    k1.metric("Volume Analysé", len(df))
    k2.metric("Engagement Total", int(df['engagement'].sum()))
    
    if 'sentiment_cat' in df.columns:
        neg_count = len(df[df['sentiment_cat'] == 'Négatif'])
        k3.metric("Signaux Négatifs", neg_count, delta_color="inverse")

        # --- VISUALISATIONS GRAPHIQUES ---
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Répartition Sémantique")
            if not df.empty:
                st.plotly_chart(
                    px.pie(df, names='sentiment_cat', color='sentiment_cat', color_discrete_map=COLOR_MAP), 
                    use_container_width=True
                )

        with c2:
            st.subheader("Matrice Impact / Sentiment")
            if not df.empty:
                st.plotly_chart(
                    px.scatter(df, x="engagement", y="sentiment_score", 
                               color="sentiment_cat", color_discrete_map=COLOR_MAP, 
                               hover_data=['text', 'handle'], size_max=40), 
                    use_container_width=True
                )

        st.divider()
        
        # --- ANALYSE TEMPORELLE AVANCÉE (SOLDE NET 4H) ---
        st.subheader("📉 Flux de Tendance Net (Périodicité : 4 Heures)")
        st.caption("Formule : Solde = [Volume Positif] - [Volume Négatif]. Les barres vertes indiquent une domination positive, les rouges une domination négative.")
        
        if 'date' in df.columns and not df.empty:
            # 1. Exclusion des Neutres pour le calcul du solde polarisé
            df_polar = df[df['sentiment_cat'] != 'Neutre'].copy()
            
            if not df_polar.empty:
                # 2. Agrégation temporelle par tranches de 4 heures
                # Pivot table pour compter Positifs et Négatifs séparément
                df_agg = df_polar.groupby([pd.Grouper(key='date', freq='4H'), 'sentiment_cat']).size().unstack(fill_value=0)
                
                # Initialisation des colonnes si absentes
                if 'Positif' not in df_agg.columns: df_agg['Positif'] = 0
                if 'Négatif' not in df_agg.columns: df_agg['Négatif'] = 0
                
                # 3. Calcul du Solde Net
                df_agg['net_score'] = df_agg['Positif'] - df_agg['Négatif']
                
                # 4. Étiquetage pour la couleur (Vert > 0, Rouge < 0)
                df_agg['trend_label'] = df_agg['net_score'].apply(lambda x: 'Positif' if x >= 0 else 'Négatif')
                
                # Préparation pour Plotly
                df_final = df_agg.reset_index()

                # 5. Génération du Graphique en Barres
                fig_bar = px.bar(
                    df_final, 
                    x="date", 
                    y="net_score", 
                    color="trend_label", # Couleur dynamique selon le solde
                    color_discrete_map=COLOR_MAP,
                    title="Évolution du Solde d'Opinion (Net Sentiment Balance)",
                    labels={"date": "Créneaux de 4h", "net_score": "Solde Net (Pos - Neg)"}
                )
                
                # 6. Mise en forme esthétique
                fig_bar.add_hline(y=0, line_color="white", opacity=0.8, line_width=2)
                fig_bar.update_layout(
                    yaxis_title="Solde (Négatif ↓ / Positif ↑)",
                    bargap=0.1, # Espacement entre les colonnes
                    showlegend=False, # Légende redondante
                    height=500
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Données insuffisantes pour calculer un solde polarisé (Absence de tweets positifs/négatifs).")

        # --- DONNÉES BRUTES ---
        st.subheader("📋 Registre des Données")
        st.dataframe(
            df[['date', 'handle', 'text', 'engagement', 'sentiment_cat']], 
            use_container_width=True
        )
    else:
        st.info("Les données récupérées ne contiennent pas de texte exploitable.")

else:
    st.info("Veuillez configurer les paramètres d'extraction dans le menu latéral.")