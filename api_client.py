import requests
import time
import math
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, Generator

# --- AUTHENTIFICATION SÉCURISÉE ---
# Le client cherche la clé dans les secrets Streamlit.
try:
    API_KEY = st.secrets["TWITTER_API_KEY"]
except:
    API_KEY = None # Sera géré plus tard par une erreur explicite

API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API V6 (Final Production).
    Intègre :
    1. Time Slicing : Répartition équitable des tweets sur la période.
    2. Gestion des erreurs : Rate limits (429) et quotas (402).
    3. Sécurité : Utilisation des variables d'environnement.
    """
    
    def build_base_query(self, p: Dict[str, Any]) -> str:
        """Construction de la syntaxe de requête complexe."""
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

        # 3. Métriques & Filtres
        if p.get('min_faves') and int(p['min_faves']) > 0: parts.append(f"min_faves:{p['min_faves']}")
        if p.get('min_retweets') and int(p['min_retweets']) > 0: parts.append(f"min_retweets:{p['min_retweets']}")
        if p.get('min_replies') and int(p['min_replies']) > 0: parts.append(f"min_replies:{p['min_replies']}")

        if p.get('links_filter') == "Exclure les liens": parts.append("-filter:links")
        elif p.get('links_filter') == "Uniquement avec liens": parts.append("filter:links")
        if p.get('replies_filter') == "Exclure les réponses": parts.append("exclude:replies")
        elif p.get('replies_filter') == "Uniquement les réponses": parts.append("filter:replies")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], total_limit: int = 50) -> Generator[Dict, None, None]:
        
        if not API_KEY:
            yield {"error": "Clé API Twitter manquante dans les Secrets !"}
            return

        base_query = self.build_base_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        start_time_global = time.time()
        
        # 1. Calcul de la durée (Time Distribution Logic)
        try:
            d_start = datetime.strptime(params['since'], "%Y-%m-%d")
            d_end = datetime.strptime(params['until'], "%Y-%m-%d")
        except:
            d_start = datetime.now() - timedelta(days=7)
            d_end = datetime.now()

        delta = (d_end - d_start).days
        if delta <= 0: delta = 1
        
        # 2. Calcul du Quota Journalier (Total / Jours)
        daily_quota = math.ceil(total_limit / delta)
        # Minimum technique pour éviter les appels inutiles
        if daily_quota < 10: daily_quota = 10
        
        print(f"[SYSTEM] Stratégie temporelle : {delta} jours | {daily_quota} tweets/jour")

        # --- BOUCLE PRINCIPALE (JOUR PAR JOUR) ---
        current_day = d_start
        
        while current_day < d_end:
            # Arrêt global si objectif atteint
            if len(all_tweets) >= total_limit:
                break

            day_str = current_day.strftime("%Y-%m-%d")
            next_day = current_day + timedelta(days=1)
            next_day_str = next_day.strftime("%Y-%m-%d")
            
            # Requête partitionnée
            daily_query = f"{base_query} since:{day_str} until:{next_day_str}"
            
            day_tweets = []
            next_cursor = None
            
            # Boucle interne (Pagination du jour)
            while len(day_tweets) < daily_quota:
                
                payload = {"query": daily_query, "limit": 20}
                if next_cursor: payload["cursor"] = next_cursor

                try:
                    response = requests.get(API_URL, params=payload, headers=headers)
                    
                    # Gestion Rate Limit (429)
                    if response.status_code == 429:
                        time.sleep(10)
                        continue 
                    
                    # Gestion Quota Épuisé (402)
                    if response.status_code == 402:
                        yield {"error": "Crédit API épuisé (Erreur 402). Veuillez changer la clé."}
                        return

                    if response.status_code != 200:
                        yield {"error": f"Erreur API ({response.status_code})"}
                        break

                    data = response.json()
                    batch = data.get('tweets', [])

                    if not batch:
                        # Jour vide ou fini, on passe au suivant
                        break

                    for t in batch:
                        if any(existing['id'] == t.get('id') for existing in all_tweets): continue
                        
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
                        day_tweets.append(tweet_obj)

                    # Mise à jour UI en temps réel
                    duration = time.time() - start_time_global
                    yield {
                        "current_count": len(all_tweets),
                        "target": total_limit,
                        "data": all_tweets,
                        "duration": round(duration, 2),
                        "finished": False
                    }

                    next_cursor = data.get('next_cursor')
                    if not next_cursor or not data.get('has_next_page'):
                        break
                    
                    if len(day_tweets) >= daily_quota:
                        break 

                    time.sleep(1) # Pause légère

                except Exception as e:
                    yield {"error": str(e)}
                    break
            
            current_day += timedelta(days=1)
            time.sleep(2) # Pause inter-jours pour sécurité

        # Envoi Final
        duration = time.time() - start_time_global
        yield {
            "current_count": len(all_tweets),
            "target": total_limit,
            "data": all_tweets[:total_limit],
            "duration": round(duration, 2),
            "finished": True
        }
