import sys
import asyncio

# Correctif Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import requests
import time
import nest_asyncio
from datetime import datetime, timedelta
from api_client import TwitterAPIClient

nest_asyncio.apply()

st.set_page_config(page_title="War Room (Cloud Edition)", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #1DA1F2; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #0d8ddb; color: white; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1DA1F2; }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'Positif': '#00CC96', 'Négatif': '#EF553B', 'Neutre': '#7f7f7f'}

# --- 1. CONFIGURATION CLOUD AI (HUGGING FACE) ---
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    HF_API_KEY = None 

# --- تصحيح الرابط: إضافة /hf-inference/ للمسار الجديد ---
API_URL_SENTIMENT = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"

def query_huggingface_api(payload):
    """Envoi avec gestion d'erreur ROBUSTE (JSON & TEXT)"""
    if not HF_API_KEY: return {"error": "Clé manquante (Missing Key)"}
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(API_URL_SENTIMENT, headers=headers, json=payload)
        
        # محاولة قراءة JSON
        try:
            return response.json()
        except json.JSONDecodeError:
            # إذا فشل الـ JSON، نعيد النص الخام لمعرفة سبب المشكلة (غالباً HTML error)
            return {"error": f"Server Error ({response.status_code}): {response.text[:200]}"}
            
    except Exception as e:
        return {"error": str(e)}

# --- 2. BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres de Recherche")
    
    # Vérification des clés
    if not HF_API_KEY or "TWITTER_API_KEY" not in st.secrets:
        st.error("⚠️ Clés API manquantes dans `.streamlit/secrets.toml` !")
        st.stop()

    with st.form("api_form"):
        st.subheader("1. Sémantique")
        all_words = st.text_input("Tous ces mots (AND)", placeholder="ex: Crise Banque")
        exact_phrase = st.text_input("Phrase exacte")
        any_words = st.text_input("N'importe lequel (OR)")
        none_words = st.text_input("Exclure (NOT)")
        hashtags = st.text_input("Hashtags")
        lang = st.selectbox("Langue", ["Tout", "fr", "en", "ar"], index=1)

        with st.expander("2. Comptes & Filtres"):
            from_accts = st.text_input("De (@)")
            to_accts = st.text_input("À (@)")
            mention_accts = st.text_input("Mentionnant (@)")
            min_faves = st.number_input("Min Likes", 0)
            min_retweets = st.number_input("Min RTs", 0)
            links_filter = st.radio("Liens", ["Tous", "Exclure", "Inclure"], index=0)
            replies_filter = st.radio("Réponses", ["Tous", "Exclure", "Inclure"], index=0)

        st.subheader("3. Période & Volume")
        d1, d2 = st.columns(2)
        since_date = d1.date_input("Début", datetime.now() - timedelta(days=7))
        until_date = d2.date_input("Fin", datetime.now())
        
        limit = st.number_input("Cible (Max 10k)", 10, 10000, 100, step=100)
        
        submitted = st.form_submit_button("🚀 Lancer l'Analyse Cloud")

    if submitted:
        client = TwitterAPIClient()
        params = {
            "all_words": all_words, "exact_phrase": exact_phrase,
            "any_words": any_words, "none_words": none_words,
            "hashtags": hashtags, "lang": lang,
            "from_accounts": from_accts, "to_accounts": to_accts, "mention_accounts": mention_accts,
            "min_faves": min_faves, "min_retweets": min_retweets,
            "links_filter": links_filter, "replies_filter": replies_filter,
            "since": since_date.strftime("%Y-%m-%d"), "until": until_date.strftime("%Y-%m-%d")
        }

        with st.status("Extraction & Analyse en cours...", expanded=True) as status:
            final_data = []
            for progress in client.fetch_tweets_generator(params, limit):
                if "error" in progress:
                    status.update(label="Erreur API", state="error")
                    st.error(progress["error"])
                    break
                curr = progress['current_count']
                tgt = progress['target']
                status.update(label=f"Acquisition ({curr}/{tgt})...", state="running")
                final_data = progress['data']
                if progress.get('finished'):
                    status.update(label="Extraction terminée.", state="complete", expanded=False)

            if final_data:
                st.success(f"{len(final_data)} tweets récupérés. Démarrage de l'analyse IA externe...")
                with open("api_data.json", "w", encoding="utf-8") as f:
                    json.dump(final_data, f, ensure_ascii=False)
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Aucune donnée trouvée.")

# --- 3. TRAITEMENT VIA CLOUD API (SYSTEME INTELLIGENT) ---
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

    # --- FONCTION D'ANALYSE (PATIENCE + DEBUG) ---
    def get_cloud_sentiment(text_list):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty() 
        total = len(text_list)
        error_shown = False 

        for i, text in enumerate(text_list):
            if not isinstance(text, str) or not text.strip():
                results.append((0.0, 'Neutre'))
                continue
                
            payload = {"inputs": text[:512]}
            sentiment_found = False
            
            # On essaie jusqu'à 10 fois
            for attempt in range(10):
                api_response = query_huggingface_api(payload)
                
                # Cas d'erreur ou chargement
                if isinstance(api_response, dict) and "error" in api_response:
                    err_msg = api_response["error"]
                    
                    # 1. Modèle en chargement
                    if "loading" in err_msg.lower():
                        status_text.warning(f"⏳ Le modèle IA démarre... ({attempt+1}/10)")
                        time.sleep(5) 
                        continue
                    
                    # 2. Erreur fatale (imprimée pour debugging)
                    elif not error_shown:
                        st.error(f"🛑 Erreur Hugging Face : {err_msg}")
                        error_shown = True # On ne montre l'erreur qu'une fois pour ne pas spammer
                        break
                
                # Cas de succès
                if isinstance(api_response, list) and len(api_response) > 0:
                    if isinstance(api_response[0], list):
                        scores = {item['label']: item['score'] for item in api_response[0]}
                        p = scores.get('positive', 0)
                        n = scores.get('negative', 0)
                        z = scores.get('neutral', 0)
                        
                        if p > n and p > z: results.append((p, 'Positif'))
                        elif n > p and n > z: results.append((-n, 'Négatif'))
                        else: results.append((0.0, 'Neutre'))
                        
                        sentiment_found = True
                        status_text.empty()
                        break
            
            if not sentiment_found:
                results.append((0.0, 'Neutre'))
            
            progress_bar.progress((i + 1) / total)
            
        progress_bar.empty()
        status_text.empty()
        return results

    if 'text' in df.columns and not df.empty:
        texts = df['text'].tolist()
        sentiments = get_cloud_sentiment(texts)
        df['sentiment_score'] = [s[0] for s in sentiments]
        df['sentiment_cat'] = [s[1] for s in sentiments]
        
    return df

df_raw = load_and_process_data()

# --- 4. DASHBOARD FINAL ---
st.title("🛡️ War Room : Cloud AI Analysis")

if not df_raw.empty:
    st.markdown("### 🔍 Segmentation")
    col_filter, _ = st.columns([1, 2])
    with col_filter:
        selected_sentiments = st.multiselect("Filtre :", ["Positif", "Négatif", "Neutre"], default=["Positif", "Négatif", "Neutre"])
    
    df = df_raw[df_raw['sentiment_cat'].isin(selected_sentiments)] if 'sentiment_cat' in df_raw.columns else df_raw
    st.divider()

    k1, k2, k3 = st.columns(3)
    k1.metric("Tweets", len(df))
    k2.metric("Engagement", int(df['engagement'].sum()))
    neg_count = len(df[df['sentiment_cat'] == 'Négatif']) if 'sentiment_cat' in df.columns else 0
    k3.metric("Négatifs", neg_count, delta_color="inverse")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Répartition")
        if not df.empty: st.plotly_chart(px.pie(df, names='sentiment_cat', color='sentiment_cat', color_discrete_map=COLOR_MAP), use_container_width=True)
    with c2:
        st.subheader("Impact / Sentiment")
        if not df.empty: st.plotly_chart(px.scatter(df, x="engagement", y="sentiment_score", color="sentiment_cat", color_discrete_map=COLOR_MAP, size_max=40), use_container_width=True)

    st.divider()
    
    # --- GRAPHIQUE SOLDE NET 4H ---
    st.subheader("📉 Solde Net (Périodicité : 4 Heures)")
    
    if 'date' in df.columns and not df.empty:
        df_polar = df[df['sentiment_cat'] != 'Neutre'].copy()
        if not df_polar.empty:
            df_agg = df_polar.groupby([pd.Grouper(key='date', freq='4H'), 'sentiment_cat']).size().unstack(fill_value=0)
            if 'Positif' not in df_agg.columns: df_agg['Positif'] = 0
            if 'Négatif' not in df_agg.columns: df_agg['Négatif'] = 0
            df_agg['net_score'] = df_agg['Positif'] - df_agg['Négatif']
            df_agg['trend_label'] = df_agg['net_score'].apply(lambda x: 'Positif' if x >= 0 else 'Négatif')
            
            fig = px.bar(df_agg.reset_index(), x="date", y="net_score", color="trend_label", color_discrete_map=COLOR_MAP)
            fig.add_hline(y=0, line_color="white", opacity=0.8)
            fig.update_layout(showlegend=False, height=500, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données insuffisantes pour le solde.")

    st.dataframe(df[['date', 'handle', 'text', 'sentiment_cat']], use_container_width=True)
else:
    st.info("Configuration requise (Menu latéral).")
