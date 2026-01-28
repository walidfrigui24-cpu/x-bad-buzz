import sys
import asyncio

# --- FIX WINDOWS ASYNCIO ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import nest_asyncio
from datetime import datetime, timedelta
from transformers import pipeline
from api_client import TwitterAPIClient

nest_asyncio.apply()

# --- CONFIG PAGE ---
st.set_page_config(page_title="War Room IA (Final)", layout="wide")
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #1DA1F2; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #0d8ddb; color: white; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1DA1F2; }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'Positif': '#00CC96', 'Négatif': '#EF553B', 'Neutre': '#7f7f7f'}

# --- CHARGEMENT IA (CACHE) ---
@st.cache_resource
def load_sentiment_model():
    # Modèle RoBERTa multilingue optimisé pour Twitter
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, top_k=None)

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres de Recherche")
    
    with st.form("api_form"):
        # 1. Mots-clés
        st.subheader("1. Sémantique")
        all_words = st.text_input("Tous ces mots (AND)", placeholder="ex: Crise Banque")
        exact_phrase = st.text_input("Phrase exacte", placeholder="ex: Effondrement du marché")
        any_words = st.text_input("N'importe lequel (OR)", placeholder="ex: peur panique")
        none_words = st.text_input("Exclure (NOT)", placeholder="ex: bitcoin")
        hashtags = st.text_input("Hashtags", placeholder="#Crise")
        lang = st.selectbox("Langue", ["Tout", "fr", "en", "ar"], index=1)

        # 2. Comptes
        with st.expander("2. Comptes & Personnes"):
            from_accts = st.text_input("De (@)", placeholder="elonmusk")
            to_accts = st.text_input("À (@)", placeholder="support")
            mention_accts = st.text_input("Mentionnant (@)", placeholder="google")

        # 3. Métriques
        with st.expander("3. Engagement & Filtres"):
            c1, c2, c3 = st.columns(3)
            min_faves = c1.number_input("Min Likes", 0)
            min_retweets = c2.number_input("Min RTs", 0)
            min_replies = c3.number_input("Min Reps", 0)
            
            links_filter = st.radio("Liens", ["Tous", "Exclure les liens", "Uniquement avec liens"], index=0)
            replies_filter = st.radio("Réponses", ["Tous", "Exclure les réponses", "Uniquement les réponses"], index=0)

        # 4. Dates
        st.subheader("4. Période")
        d1, d2 = st.columns(2)
        since_date = d1.date_input("Début", datetime.now() - timedelta(days=30))
        until_date = d2.date_input("Fin", datetime.now())

        # 5. Volume
        st.subheader("5. Limites")
        limit = st.number_input("Volume cible (Max 10k)", 10, 10000, 100, step=100)
        
        submitted = st.form_submit_button("🚀 Lancer l'Analyse IA")

    if submitted:
        client = TwitterAPIClient()
        
        params = {
            "all_words": all_words, "exact_phrase": exact_phrase,
            "any_words": any_words, "none_words": none_words,
            "hashtags": hashtags, "lang": lang,
            "from_accounts": from_accts, "to_accounts": to_accts, "mention_accounts": mention_accts,
            "min_faves": min_faves, "min_retweets": min_retweets, "min_replies": min_replies,
            "links_filter": links_filter, "replies_filter": replies_filter,
            "since": since_date.strftime("%Y-%m-%d"),
            "until": until_date.strftime("%Y-%m-%d")
        }

        with st.status("Exécution du protocole...", expanded=True) as status:
            final_data = []
            for progress in client.fetch_tweets_generator(params, limit):
                if "error" in progress:
                    status.update(label="Erreur API", state="error")
                    st.error(progress["error"])
                    break
                
                curr = progress['current_count']
                tgt = progress['target']
                status.update(label=f"Acquisition en cours ({curr}/{tgt})...", state="running")
                final_data = progress['data']
                
                if progress.get('finished'):
                    status.update(label="Terminé.", state="complete", expanded=False)

            if final_data:
                st.success(f"Terminé : {len(final_data)} tweets indexés.")
                with open("api_data.json", "w", encoding="utf-8") as f:
                    json.dump(final_data, f, ensure_ascii=False)
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Aucune donnée trouvée. Essayez d'élargir la période ou de réduire les mots-clés.")

# --- TRAITEMENT DES DONNÉES ---

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

    # --- INFÉRENCE IA ---
    sentiment_pipeline = load_sentiment_model()

    def get_ai_sentiment(text):
        if not isinstance(text, str) or not text.strip(): return 0.0, 'Neutre'
        try:
            # Troncature pour éviter l'erreur "Sequence too long"
            res = sentiment_pipeline(text[:512])[0]
            scores = {r['label']: r['score'] for r in res}
            
            p = scores.get('positive', 0)
            n = scores.get('negative', 0)
            z = scores.get('neutral', 0)

            if p > n and p > z: return p, 'Positif'
            elif n > p and n > z: return -n, 'Négatif'
            else: return 0.0, 'Neutre'
        except: return 0.0, 'Neutre'
    
    if 'text' in df.columns:
        with st.spinner("Analyse sémantique IA (Deep Learning)..."):
            res = df['text'].apply(lambda x: pd.Series(get_ai_sentiment(x)))
            df[['sentiment_score', 'sentiment_cat']] = res
    return df

df_raw = load_and_process_data()

# --- TABLEAU DE BORD ---

st.title("🛡️ War Room : Analyse de Crise (AI + Full Search)")

if not df_raw.empty:
    
    st.markdown("### 🔍 Segmentation")
    col_filter, _ = st.columns([1, 2])
    with col_filter:
        selected_sentiments = st.multiselect("Filtrer par sentiment :", ["Positif", "Négatif", "Neutre"], default=["Positif", "Négatif", "Neutre"])
    
    df = df_raw[df_raw['sentiment_cat'].isin(selected_sentiments)] if 'sentiment_cat' in df_raw.columns else df_raw

    st.divider()

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Volume Analysé", len(df))
    k2.metric("Engagement Cumulé", int(df['engagement'].sum()))
    neg_count = len(df[df['sentiment_cat'] == 'Négatif']) if 'sentiment_cat' in df.columns else 0
    k3.metric("Alertes Négatives", neg_count, delta_color="inverse")

    # Graphiques Standards
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Répartition")
        if not df.empty: st.plotly_chart(px.pie(df, names='sentiment_cat', color='sentiment_cat', color_discrete_map=COLOR_MAP), use_container_width=True)
    with c2:
        st.subheader("Impact / Sentiment")
        if not df.empty: st.plotly_chart(px.scatter(df, x="engagement", y="sentiment_score", color="sentiment_cat", color_discrete_map=COLOR_MAP, hover_data=['text'], size_max=40), use_container_width=True)

    st.divider()
    
    # --- GRAPHIQUE SOLDE NET (LOGIQUE CORRECTIVE) ---
    st.subheader("📉 Solde Net de Sentiment (4 Heures)")
    st.caption("Barre VERTE = Majorité Positive / Barre ROUGE = Majorité Négative")
    
    if 'date' in df.columns and not df.empty:
        # On exclut les Neutres pour ne pas diluer le signal
        df_polar = df[df['sentiment_cat'] != 'Neutre'].copy()
        
        if not df_polar.empty:
            # Agrégation temporelle + Pivot
            df_agg = df_polar.groupby([pd.Grouper(key='date', freq='4H'), 'sentiment_cat']).size().unstack(fill_value=0)
            
            # Sécurisation des colonnes (si un sentiment manque totalement)
            if 'Positif' not in df_agg.columns: df_agg['Positif'] = 0
            if 'Négatif' not in df_agg.columns: df_agg['Négatif'] = 0
            
            # Calcul du solde
            df_agg['net_score'] = df_agg['Positif'] - df_agg['Négatif']
            
            # Étiquetage
            df_agg['trend_label'] = df_agg['net_score'].apply(lambda x: 'Positif' if x >= 0 else 'Négatif')
            
            # Visualisation
            fig = px.bar(
                df_agg.reset_index(), 
                x="date", 
                y="net_score", 
                color="trend_label", 
                color_discrete_map=COLOR_MAP,
                labels={"net_score": "Solde Net (Pos - Neg)"}
            )
            fig.add_hline(y=0, line_color="white", opacity=0.8)
            fig.update_layout(showlegend=False, height=500, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données insuffisantes pour calculer le solde (Aucun tweet positif ou négatif).")

    st.subheader("📋 Registre des Données")
    st.dataframe(df[['date', 'handle', 'text', 'engagement', 'sentiment_cat']], use_container_width=True)

else:
    st.info("Veuillez lancer une recherche depuis le menu latéral.")
