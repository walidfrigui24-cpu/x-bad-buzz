import sys
import asyncio

# --- CORRECTIF SYSTÈME (WINDOWS EVENT LOOP) ---
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

# Importation des bibliothèques d'Intelligence Artificielle
from transformers import pipeline

# Importation du client API
from api_client import TwitterAPIClient

# Patch asynchrone
nest_asyncio.apply()

# --- CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="Système d'Analyse IA (Pro)", layout="wide")

# Injection CSS (Style 'War Room')
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #1DA1F2; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #0d8ddb; color: white; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1DA1F2; }
</style>
""", unsafe_allow_html=True)

# Palette de couleurs
COLOR_MAP = {'Positif': '#00CC96', 'Négatif': '#EF553B', 'Neutre': '#7f7f7f'}

# --- 1. CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_sentiment_model():
    """Initialise le modèle RoBERTa (Multilingue)"""
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, top_k=None)

# --- 2. BARRE LATÉRALE (MODIFIÉE SELON VOS BESOINS) ---
with st.sidebar:
    st.header("Paramètres de Recherche")
    
    with st.form("api_form"):
        st.subheader("1. Ciblage Sémantique")
        all_words = st.text_input("Mots-clés (Tous)", placeholder="ex: Crise Banque")
        exact_phrase = st.text_input("Phrase Exacte")
        hashtags = st.text_input("Hashtags", placeholder="#Finance")
        lang = st.selectbox("Langue", ["Tout", "fr", "en", "ar"], index=1)

        # --- MODIFICATION: DATES VISIBLES DIRECTEMENT ---
        st.subheader("2. Période d'Analyse")
        c1, c2 = st.columns(2)
        since_date = c1.date_input("Début", datetime.now() - timedelta(days=30))
        until_date = c2.date_input("Fin", datetime.now())
        # -----------------------------------------------

        st.subheader("3. Filtres Techniques")
        with st.expander("Options Avancées (Comptes & Engagement)"):
            min_faves = st.number_input("Min. J'aime", 0)
            from_accts = st.text_input("Depuis ces comptes (@)")

        # --- MODIFICATION: LIMITE 10,000 + PAS DE 100 ---
        st.subheader("4. Volume")
        limit = st.number_input(
            "Nombre de tweets à extraire", 
            min_value=10, 
            max_value=10000,  # Max augmenté à 10,000
            value=100, 
            step=100          # Ajout par pas de 100
        )
        # -----------------------------------------------
        
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

        # --- EXÉCUTION DYNAMIQUE ---
        with st.status("Démarrage du protocole d'extraction...", expanded=True) as status:
            final_data = []
            
            for progress in client.fetch_tweets_generator(params, limit):
                
                if "error" in progress:
                    status.update(label="Erreur Critique API", state="error")
                    st.error(progress["error"])
                    break
                
                curr = progress['current_count']
                tgt = progress['target']
                
                status.update(label=f"Acquisition des données ({curr}/{tgt}) - Veuillez patienter...", state="running")
                
                final_data = progress['data']
                
                if progress.get('finished'):
                    status.update(label="Extraction terminée.", state="complete", expanded=False)

            if final_data:
                st.success(f"Acquisition terminée : {len(final_data)} tweets indexés.")
                with open("api_data.json", "w", encoding="utf-8") as f:
                    json.dump(final_data, f, ensure_ascii=False)
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Aucune donnée trouvée. Vérifiez vos filtres.")

# --- 3. TRAITEMENT ANALYTIQUE (IA) ---

@st.cache_data
def load_and_process_data():
    if not os.path.exists("api_data.json"): return pd.DataFrame()
    try:
        with open("api_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except: return pd.DataFrame()
            
    if not data: return pd.DataFrame()
    
    df = pd.json_normalize(data)
    df['date'] = pd.to_datetime(df['date_iso'], errors='coerce')
    
    for col in ['metrics.likes', 'metrics.retweets', 'metrics.replies']:
        if col not in df.columns: df[col] = 0

    df['engagement'] = df['metrics.likes'] + df['metrics.retweets']

    # Inférence IA
    sentiment_pipeline = load_sentiment_model()

    def get_ai_sentiment(text):
        if not isinstance(text, str) or not text.strip(): 
            return 0.0, 'Neutre'
        
        try:
            truncated_text = text[:512]
            results = sentiment_pipeline(truncated_text)[0]
            scores = {res['label']: res['score'] for res in results}
            
            pos_score = scores.get('positive', 0)
            neg_score = scores.get('negative', 0)
            neu_score = scores.get('neutral', 0)

            if pos_score > neg_score and pos_score > neu_score:
                return pos_score, 'Positif'
            elif neg_score > pos_score and neg_score > neu_score:
                return -neg_score, 'Négatif'
            else:
                return 0.0, 'Neutre'
        except Exception:
            return 0.0, 'Neutre'
    
    if 'text' in df.columns:
        with st.spinner("Analyse sémantique IA en cours (Deep Learning)..."):
            result_series = df['text'].apply(lambda x: pd.Series(get_ai_sentiment(x)))
            df[['sentiment_score', 'sentiment_cat']] = result_series
    
    return df

df_raw = load_and_process_data()

# --- 4. TABLEAU DE BORD (DASHBOARD) ---

st.title("🛡️ War Room : Analyse de Crise (IA Powered)")

if not df_raw.empty:
    
    # Filtres
    st.markdown("### 🔍 Segmentation")
    col_filter, _ = st.columns([1, 2])
    with col_filter:
        selected_sentiments = st.multiselect(
            "Afficher les sentiments :",
            options=["Positif", "Négatif", "Neutre"],
            default=["Positif", "Négatif", "Neutre"]
        )
    
    if 'sentiment_cat' in df_raw.columns:
        df = df_raw[df_raw['sentiment_cat'].isin(selected_sentiments)]
    else:
        df = df_raw

    st.divider()

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Volume Analysé", len(df))
    k2.metric("Engagement Total", int(df['engagement'].sum()))
    
    if 'sentiment_cat' in df.columns:
        neg_count = len(df[df['sentiment_cat'] == 'Négatif'])
        k3.metric("Signaux Négatifs", neg_count, delta_color="inverse")

        # Graphiques Circulaires & Nuages
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Répartition")
            if not df.empty:
                st.plotly_chart(px.pie(df, names='sentiment_cat', color='sentiment_cat', color_discrete_map=COLOR_MAP), use_container_width=True)
        with c2:
            st.subheader("Impact / Sentiment")
            if not df.empty:
                st.plotly_chart(px.scatter(df, x="engagement", y="sentiment_score", color="sentiment_cat", color_discrete_map=COLOR_MAP, hover_data=['text'], size_max=40), use_container_width=True)

        st.divider()
        
        # --- SOLDE NET 4H (Net Sentiment Balance) ---
        st.subheader("📉 Flux de Tendance Net (Périodicité : 4 Heures)")
        st.caption("Solde = [Volume Positif] - [Volume Négatif]")
        
        if 'date' in df.columns and not df.empty:
            df_polar = df[df['sentiment_cat'] != 'Neutre'].copy()
            
            if not df_polar.empty:
                df_agg = df_polar.groupby([pd.Grouper(key='date', freq='4H'), 'sentiment_cat']).size().unstack(fill_value=0)
                
                if 'Positif' not in df_agg.columns: df_agg['Positif'] = 0
                if 'Négatif' not in df_agg.columns: df_agg['Négatif'] = 0
                
                df_agg['net_score'] = df_agg['Positif'] - df_agg['Négatif']
                df_agg['trend_label'] = df_agg['net_score'].apply(lambda x: 'Positif' if x >= 0 else 'Négatif')
                
                df_final = df_agg.reset_index()

                fig_bar = px.bar(
                    df_final, 
                    x="date", 
                    y="net_score", 
                    color="trend_label",
                    color_discrete_map=COLOR_MAP,
                    title="Solde d'Opinion (Net Balance)",
                    labels={"date": "Créneaux 4h", "net_score": "Solde Net"}
                )
                fig_bar.add_hline(y=0, line_color="white", opacity=0.8, line_width=2)
                fig_bar.update_layout(bargap=0.1, showlegend=False, height=500)
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Données insuffisantes pour le solde polarisé.")

        # Données Brutes
        st.subheader("📋 Registre des Données")
        st.dataframe(df[['date', 'handle', 'text', 'engagement', 'sentiment_cat']], use_container_width=True)

else:
    st.info("Veuillez configurer les paramètres (Barre latérale).")
