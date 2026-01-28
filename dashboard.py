import sys
import asyncio

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

# --- 1. إعدادات HUGGING FACE API ---
# ضع مفتاحك هنا مباشرة للتجربة، أو في Streamlit Secrets للأمان
# مثال: "hf_xxxxxxxxxxxxxxxxxxxx"
# نحاول جلب المفتاح من خزنة الأسرار الآمنة
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    st.error("⚠️ المفتاح السري غير موجود! يرجى إضافته في إعدادات Streamlit.")
    st.stop()
API_URL_SENTIMENT = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"

def query_huggingface_api(payload):
    """دالة خفيفة تتصل بسيرفرات Hugging Face للتحليل"""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(API_URL_SENTIMENT, headers=headers, json=payload)
        return response.json()
    except:
        return None

# --- 2. الواجهة الجانبية (نفس السابق) ---
with st.sidebar:
    st.header("Paramètres de Recherche")
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
            links_filter = st.radio("Liens", ["Tous", "Exclure", "Inclure"], index=0)
            replies_filter = st.radio("Réponses", ["Tous", "Exclure", "Inclure"], index=0)

        st.subheader("3. Période & Volume")
        d1, d2 = st.columns(2)
        since_date = d1.date_input("Début", datetime.now() - timedelta(days=7))
        until_date = d2.date_input("Fin", datetime.now())
        limit = st.number_input("Cible (Max 2000)", 10, 2000, 50, step=50) # قللنا الحد الأقصى للحفاظ على سرعة الـ API
        
        submitted = st.form_submit_button("🚀 Lancer l'Analyse Cloud")

    if submitted:
        client = TwitterAPIClient()
        params = {
            "all_words": all_words, "exact_phrase": exact_phrase,
            "any_words": any_words, "none_words": none_words,
            "hashtags": hashtags, "lang": lang,
            "from_accounts": from_accts, "to_accounts": to_accts, "mention_accounts": mention_accts,
            "min_faves": min_faves, "links_filter": links_filter, "replies_filter": replies_filter,
            "since": since_date.strftime("%Y-%m-%d"), "until": until_date.strftime("%Y-%m-%d")
        }

        with st.status("Extraction & Analyse Cloud...", expanded=True) as status:
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
                    status.update(label="Terminé.", state="complete", expanded=False)

            if final_data:
                st.success(f"{len(final_data)} tweets récupérés. Démarrage de l'analyse IA externe...")
                with open("api_data.json", "w", encoding="utf-8") as f:
                    json.dump(final_data, f, ensure_ascii=False)
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Aucune donnée.")

# --- 3. TRAITEMENT VIA API EXTERNE ---
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

    # --- دالة التحليل السحابي ---
    def get_cloud_sentiment(text_list):
        # Hugging Face API يقبل قائمة نصوص لتحليلها دفعة واحدة
        # لكن لكي لا نضغط على السيرفر المجاني، سنرسل واحداً تلو الآخر أو مجموعات صغيرة
        results = []
        
        # إنشاء بار التقدم
        progress_bar = st.progress(0)
        total = len(text_list)
        
        for i, text in enumerate(text_list):
            if not isinstance(text, str) or not text.strip():
                results.append((0.0, 'Neutre'))
                continue
                
            payload = {"inputs": text[:512]} # قص النص الطويل
            
            # محاولة الاتصال (مع إعادة المحاولة في حالة انشغال السيرفر)
            for _ in range(3):
                api_response = query_huggingface_api(payload)
                
                # التحقق من الخطأ (Model Loading)
                if isinstance(api_response, dict) and "error" in api_response:
                    time.sleep(2) # السيرفر يحمل الموديل، ننتظر قليلاً
                    continue
                    
                if isinstance(api_response, list) and len(api_response) > 0:
                    # استخراج النتائج
                    # الهيكل: [[{'label': 'positive', 'score': 0.9}, ...]]
                    scores = {item['label']: item['score'] for item in api_response[0]}
                    
                    p = scores.get('positive', 0)
                    n = scores.get('negative', 0)
                    z = scores.get('neutral', 0)
                    
                    if p > n and p > z: results.append((p, 'Positif'))
                    elif n > p and n > z: results.append((-n, 'Négatif'))
                    else: results.append((0.0, 'Neutre'))
                    break
            else:
                results.append((0.0, 'Neutre')) # في حالة الفشل
            
            # تحديث البار
            progress_bar.progress((i + 1) / total)
            
            # احترام حدود السرعة المجانية
            time.sleep(0.1) 
            
        progress_bar.empty()
        return results

    if 'text' in df.columns and not df.empty:
        # إذا كانت البيانات موجودة ولم تحلل بعد أو نريد إعادة التحليل
        # ملاحظة: لتحسين الأداء، نقوم بالتحليل فقط إذا طلب المستخدم ذلك أو نعتمد الكاش
        # هنا سنقوم بالتحليل المباشر
        
        # لتقليل الوقت، نأخذ النصوص كقائمة
        texts = df['text'].tolist()
        
        # استدعاء الدالة السحابية
        sentiments = get_cloud_sentiment(texts)
        
        # تفريغ النتائج في الداتا فريم
        df['sentiment_score'] = [s[0] for s in sentiments]
        df['sentiment_cat'] = [s[1] for s in sentiments]
        
    return df

df_raw = load_and_process_data()

# --- 4. DASHBOARD (نفس السابق تماماً) ---
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
        if not df.empty: st.plotly_chart(px.pie(df, names='sentiment_cat', color='sentiment_cat', color_discrete_map=COLOR_MAP), use_container_width=True)
    with c2:
        if not df.empty: st.plotly_chart(px.scatter(df, x="engagement", y="sentiment_score", color="sentiment_cat", color_discrete_map=COLOR_MAP, size_max=40), use_container_width=True)

    st.divider()
    
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
            st.info("Pas assez de données pour le solde.")

    st.dataframe(df[['date', 'handle', 'text', 'sentiment_cat']], use_container_width=True)
else:
    st.info("Veuillez lancer l'analyse.")
