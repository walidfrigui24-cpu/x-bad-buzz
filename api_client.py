import requests
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Generator

# --- CONFIGURATION ---
API_KEY = "new1_e1b45ea37988449dbebfea70c1740126" 
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API V5 (Time-Distributed).
    Divise l'objectif total par le nombre de jours pour forcer
    une extraction équitable sur toute la période sélectionnée.
    """
    
    def build_base_query(self, p: Dict[str, Any]) -> str:
        """Construit la requête SANS les dates (elles seront ajoutées dans la boucle)."""
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

        # 3. Filtres
        if p.get('min_faves') and int(p['min_faves']) > 0: parts.append(f"min_faves:{p['min_faves']}")
        if p.get('min_retweets') and int(p['min_retweets']) > 0: parts.append(f"min_retweets:{p['min_retweets']}")
        if p.get('min_replies') and int(p['min_replies']) > 0: parts.append(f"min_replies:{p['min_replies']}")

        if p.get('links_filter') == "Exclure les liens": parts.append("-filter:links")
        elif p.get('links_filter') == "Uniquement avec liens": parts.append("filter:links")
        if p.get('replies_filter') == "Exclure les réponses": parts.append("exclude:replies")
        elif p.get('replies_filter') == "Uniquement les réponses": parts.append("filter:replies")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], total_limit: int = 50) -> Generator[Dict, None, None]:
        
        base_query = self.build_base_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        start_time_global = time.time()
        
        # 1. Calcul de la durée (Nombre de jours)
        try:
            d_start = datetime.strptime(params['since'], "%Y-%m-%d")
            d_end = datetime.strptime(params['until'], "%Y-%m-%d")
        except:
            d_start = datetime.now() - timedelta(days=7)
            d_end = datetime.now()

        delta = (d_end - d_start).days
        if delta <= 0: delta = 1
        
        # 2. Calcul du Quota par jour (Ex: 100 tweets / 5 jours = 20 tweets/jour)
        daily_quota = math.ceil(total_limit / delta)
        # Sécurité: on ne fait pas de requête pour moins de 10 tweets (perte de temps)
        if daily_quota < 10: daily_quota = 10
        
        print(f"[SYSTEM] Période: {delta} jours. Quota journalier: {daily_quota} tweets.")

        # --- BOUCLE JOUR PAR JOUR ---
        # On itère du plus récent au plus ancien (ou l'inverse, ici du début à la fin)
        current_day = d_start
        
        while current_day < d_end:
            # Si on a déjà dépassé la limite TOTALE (pas journalière), on arrête tout
            if len(all_tweets) >= total_limit:
                break

            day_str = current_day.strftime("%Y-%m-%d")
            # Le jour suivant pour la borne 'until'
            next_day = current_day + timedelta(days=1)
            next_day_str = next_day.strftime("%Y-%m-%d")
            
            # Requête spécifique pour CE jour
            # since:JOUR until:LENDEMAIN
            daily_query = f"{base_query} since:{day_str} until:{next_day_str}"
            
            print(f"[API] Scanning {day_str} -> {next_day_str} (Cible: {daily_quota})")
            
            day_tweets = []
            next_cursor = None
            empty_retries = 0

            # Boucle pour remplir le quota DU JOUR
            while len(day_tweets) < daily_quota:
                
                payload = {"query": daily_query, "limit": 20}
                if next_cursor: payload["cursor"] = next_cursor

                try:
                    response = requests.get(API_URL, params=payload, headers=headers)
                    
                    if response.status_code == 429:
                        time.sleep(10)
                        continue 

                    if response.status_code != 200:
                        yield {"error": f"Err {day_str}: {response.status_code}"}
                        break

                    data = response.json()
                    batch = data.get('tweets', [])

                    if not batch:
                        # Si page vide, on passe DIRECTEMENT au jour suivant (Gain de temps)
                        # Pas besoin de retry 3 fois si on partitionne par jour
                        print(f"[API] Fin des données pour {day_str}.")
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

                    # Update UI
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
                        break # Quota du jour atteint, on passe au lendemain

                    time.sleep(2) # Petite pause intra-jour

                except Exception as e:
                    print(f"Erreur: {e}")
                    break
            
            # Passage au jour suivant
            current_day += timedelta(days=1)
            
            # Pause entre les jours pour éviter le Rate Limit
            time.sleep(3)

        # Envoi Final
        duration = time.time() - start_time_global
        yield {
            "current_count": len(all_tweets),
            "target": total_limit,
            "data": all_tweets[:total_limit],
            "duration": round(duration, 2),
            "finished": True
        }

