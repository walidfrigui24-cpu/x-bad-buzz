import requests
import time
from datetime import datetime
from typing import Dict, Any, Generator

# --- CONFIGURATION (Free Tier) ---
# مفتاح API (يمكن تغييره لاحقاً ليأتي من Streamlit Secrets)
API_KEY = "new1_c4a4317b0a7f4669b7a0baf181eb4861" 
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API V4 (Final Logic).
    Features:
    1. Strict Date Filtering: Rejet immédiat des tweets hors période.
    2. Retry Mechanism: Survie aux pages vides (jusqu'à 3 tentatives).
    3. Pagination Robuste: Navigation forcée via curseurs.
    """
    
    def build_query(self, p: Dict[str, Any]) -> str:
        """Construction de la requête booléenne."""
        parts = []
        
        # 1. Sémantique
        if p.get('all_words'): parts.append(p['all_words'])
        if p.get('exact_phrase'): parts.append(f'"{p["exact_phrase"]}"')
        
        if p.get('any_words'): 
            words = p['any_words'].split()
            if len(words) > 1: parts.append(f"({' OR '.join(words)})")
            else: parts.append(words[0])
        
        if p.get('none_words'): 
            for w in p['none_words'].split(): parts.append(f"-{w}")
        
        if p.get('hashtags'): parts.append(p['hashtags'])
        if p.get('lang') and p['lang'] != "Tout": parts.append(f"lang:{p['lang']}")

        # 2. Comptes
        if p.get('from_accounts'): parts.append(f"from:{p['from_accounts'].replace('@', '')}")
        if p.get('to_accounts'): parts.append(f"to:{p['to_accounts'].replace('@', '')}")
        if p.get('mention_accounts'): parts.append(f"@{p['mention_accounts'].replace('@', '')}")

        # 3. Engagement & Filtres
        if p.get('min_faves') and int(p['min_faves']) > 0: parts.append(f"min_faves:{p['min_faves']}")
        if p.get('min_retweets') and int(p['min_retweets']) > 0: parts.append(f"min_retweets:{p['min_retweets']}")
        if p.get('min_replies') and int(p['min_replies']) > 0: parts.append(f"min_replies:{p['min_replies']}")

        if p.get('links_filter') == "Exclure les liens": parts.append("-filter:links")
        elif p.get('links_filter') == "Uniquement avec liens": parts.append("filter:links")
        if p.get('replies_filter') == "Exclure les réponses": parts.append("exclude:replies")
        elif p.get('replies_filter') == "Uniquement les réponses": parts.append("filter:replies")

        # 4. Dates (Indispensable pour l'API)
        if p.get('since'): parts.append(f"since:{p['since']}")
        if p.get('until'): parts.append(f"until:{p['until']}")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], limit: int = 50) -> Generator[Dict, None, None]:
        
        query_string = self.build_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        next_cursor = None
        start_time = time.time()
        empty_retries = 0  # Compteur de sécurité pour les pages vides
        
        # --- LOGIQUE DE DATE STRICTE (CLIENT-SIDE) ---
        # L'API est parfois approximative, nous filtrons nous-mêmes pour garantir 100% de précision.
        try:
            strict_start = datetime.strptime(params['since'], "%Y-%m-%d")
            strict_end = datetime.strptime(params['until'], "%Y-%m-%d")
            # On ajoute un jour à la fin pour inclure la journée entière (23:59:59)
            print(f"[SYSTEM] Filtrage strict activé : {strict_start.date()} -> {strict_end.date()}")
        except:
            strict_start = None
            print("[SYSTEM] Attention : Filtrage strict désactivé (Format de date invalide)")

        print(f"[SYSTEM] Démarrage extraction... Cible : {limit}")

        while len(all_tweets) < limit:
            
            payload = {"query": query_string, "limit": 20}
            if next_cursor:
                payload["cursor"] = next_cursor

            try:
                response = requests.get(API_URL, params=payload, headers=headers)
                
                # Gestion Rate Limit (Pause forcée)
                if response.status_code == 429:
                    print("[API] 🛑 Rate Limit atteint. Pause de 10s...")
                    time.sleep(10)
                    continue 

                if response.status_code != 200:
                    yield {"error": f"Erreur API ({response.status_code})"}
                    break

                data = response.json()
                batch = data.get('tweets', [])
                
                # --- LOGIQUE ANTI-ARRÊT PRÉMATURÉ ---
                if not batch:
                    empty_retries += 1
                    print(f"[API] ⚠️ Page vide ({empty_retries}/3). Tentative de forcing...")
                    
                    if empty_retries >= 3:
                        print("[API] Arrêt définitif : Trop de pages vides.")
                        break
                    
                    # On tente de forcer le curseur suivant même sans résultats
                    if data.get('has_next_page') and data.get('next_cursor'):
                        next_cursor = data.get('next_cursor')
                        time.sleep(2)
                        continue
                    else:
                        break
                else:
                    empty_retries = 0 # Reset si on trouve des données

                # --- TRAITEMENT ET FILTRAGE ---
                added_in_batch = 0
                for t in batch:
                    if len(all_tweets) >= limit: break
                    
                    if any(existing['id'] == t.get('id') for existing in all_tweets): continue

                    # 1. Vérification Date Stricte
                    if strict_start:
                        try:
                            t_date_str = t.get('createdAt', '').split('T')[0]
                            t_date = datetime.strptime(t_date_str, "%Y-%m-%d")
                            # Rejet silencieux si hors date
                            if t_date < strict_start or t_date > strict_end:
                                continue
                        except: pass 
                    
                    # 2. Construction Objet
                    author = t.get('author') or {}
                    tweet_obj = {
                        "id": t.get('id'),
                        "date_iso": t.get('createdAt'),
                        "text": t.get('text', ""),
                        "handle": author.get('userName', 'Inconnu'),
                        "url": t.get('url') or t.get('twitterUrl', ""),
                        "metrics": {
                            "likes": t.get('likeCount', 0),
                            "retweets": t.get('retweetCount', 0),
                            "replies": t.get('replyCount', 0)
                        }
                    }
                    all_tweets.append(tweet_obj)
                    added_in_batch += 1

                print(f"[API] +{added_in_batch} ajoutés (Total: {len(all_tweets)}/{limit})")

                # Mise à jour Interface
                duration = time.time() - start_time
                yield {
                    "current_count": len(all_tweets),
                    "target": limit,
                    "data": all_tweets,
                    "duration": round(duration, 2),
                    "finished": False
                }

                # Pagination
                next_cursor = data.get('next_cursor')
                if not next_cursor or not data.get('has_next_page'):
                    print("[API] Fin de pagination (Plus de curseur).")
                    break
                
                if len(all_tweets) >= limit:
                    break

                # Pause de courtoisie (4s pour éviter le bannissement temporaire)
                time.sleep(4) 

            except Exception as e:
                yield {"error": str(e)}
                break

        # Envoi Final
        duration = time.time() - start_time
        yield {
            "current_count": len(all_tweets),
            "target": limit,
            "data": all_tweets[:limit],
            "duration": round(duration, 2),
            "finished": True
        }
