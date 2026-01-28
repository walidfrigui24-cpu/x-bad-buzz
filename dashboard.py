# --- استبدل دالة الاتصال القديمة بهذه الدالة التي تكشف الأخطاء ---
def query_huggingface_api(payload):
    """Envoi avec gestion d'erreur explicite"""
    if not HF_API_KEY: return {"error": "Mising Key"}
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(API_URL_SENTIMENT, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- استبدل دالة التحليل القديمة داخل load_and_process_data بهذه ---
    def get_cloud_sentiment(text_list):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty() # مكان لعرض حالة الموديل
        total = len(text_list)
        
        # متغير لمعرفة هل تم تنبيه المستخدم أم لا
        error_shown = False 

        for i, text in enumerate(text_list):
            if not isinstance(text, str) or not text.strip():
                results.append((0.0, 'Neutre'))
                continue
                
            payload = {"inputs": text[:512]}
            
            # --- التعديل الجوهري: الصبر الذكي (Intelligent Retry) ---
            # نحاول 10 مرات (10 * 5 ثواني = 50 ثانية انتظار كحد أقصى)
            # هذا ضروري لأن الموديل المجاني يحتاج وقتاً "ليصحو من النوم"
            sentiment_found = False
            
            for attempt in range(10):
                api_response = query_huggingface_api(payload)
                
                # 1. حالة الخطأ الصريح (المفتاح خطأ أو غيره)
                if isinstance(api_response, dict) and "error" in api_response:
                    err_msg = api_response["error"]
                    
                    # إذا كان الخطأ هو "Model is loading" (الموديل يحمل)
                    if "loading" in err_msg.lower():
                        status_text.warning(f"⏳ الموديل يتم تحميله في سيرفرات Hugging Face... (المحاولة {attempt+1}/10)")
                        time.sleep(5) # ننتظر 5 ثواني ثم نحاول مجدداً
                        continue
                    
                    # إذا كان خطأ آخر (مثل المفتاح غلط)
                    elif not error_shown:
                        st.error(f"🛑 خطأ في API: {err_msg}")
                        error_shown = True # نعرض الخطأ مرة واحدة فقط
                        break
                
                # 2. حالة النجاح (قائمة نتائج)
                if isinstance(api_response, list) and len(api_response) > 0:
                    # تفكيك النتائج [[{label:..., score:...}]]
                    if isinstance(api_response[0], list):
                        scores = {item['label']: item['score'] for item in api_response[0]}
                        p = scores.get('positive', 0)
                        n = scores.get('negative', 0)
                        z = scores.get('neutral', 0)
                        
                        if p > n and p > z: results.append((p, 'Positif'))
                        elif n > p and n > z: results.append((-n, 'Négatif'))
                        else: results.append((0.0, 'Neutre'))
                        
                        sentiment_found = True
                        status_text.empty() # نخفي رسالة التحميل إذا نجح
                        break
            
            if not sentiment_found:
                # إذا فشلت كل المحاولات، نسجل محايد (للأسف)
                results.append((0.0, 'Neutre'))
            
            progress_bar.progress((i + 1) / total)
            
        progress_bar.empty()
        status_text.empty()
        return results
