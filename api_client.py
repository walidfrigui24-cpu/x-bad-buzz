import requests
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Generator

# --- CONFIGURATION (Free Tier) ---
API_KEY = "new1_c4a4317b0a7f4669b7a0baf181eb4861" 
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API Intelligent avec 'Time Slicing'.
    Divise la requête globale en sous-requêtes journalières pour garantir
    une répartition équitable des tweets sur toute la période.
    """
    
    def build_query(self, p: Dict[str, Any]) -> str:
        """Construit la chaîne de requête (SANS les dates, gérées par la boucle)."""
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

        # 3. Engagement
        if p.get('min_faves') and int(p['min_faves']) > 0: parts.append(f"min_faves:{p['min_faves']}")
        if p.get('min_retweets') and int(p['min_retweets']) > 0: parts.append(f"min_retweets:{p['min_retweets']}")
        if p.get('min_replies') and int(p['min_replies']) > 0: parts.append(f"min_replies:{p['min_replies']}")

        # 4. Filtres
        if p.get('links_filter') == "Exclure les liens": parts.append("-filter:links")
        elif p.get('links_filter') == "Uniquement avec liens": parts.append("filter:links")
            
        if p.get('replies_filter') == "Exclure les réponses": parts.append("exclude:replies")
        elif p.get('replies_filter') == "Uniquement les réponses": parts.append("filter:replies")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], total_limit: int = 50) -> Generator[Dict, None, None]:
        """
        Générateur intelligent qui itère jour par jour.
        """
        base_query = self.build_query(params)
        headers = {"X-API-Key": API_KEY}
        all_tweets = []
        start_time = time.time()

        # 1. Calcul de la plage de dates
        try:
            d_start = datetime.strptime(params['since'], "%Y-%m-%d")
            d_end = datetime.strptime(params['until'], "%Y-%m-%d")
        except:
            # Fallback si erreur de date : mode classique
            d_start = datetime.now() - timedelta(days=7)
            d_end = datetime.now()
        
        # Nombre de jours total
        delta_days = (d_end - d_start).days
        if delta_days <= 0: delta_days = 1
        
        # 2. Calcul du Quota par jour (Répartition équitable)
        # Ex: 100 tweets sur 10 jours = 10 tweets/jour
        # On utilise ceil pour s'assurer d'atteindre le but même avec des arrondis
        daily_quota = math.ceil(total_limit / delta_days)
        
        # Sécurité : Minimum 10 tweets par jour pour rentabiliser la requête API
        if daily_quota < 10: daily_quota = 10

        print(f"[SYSTEM] Stratégie : {delta_days} jours, {daily_quota} tweets/jour")

        # --- BOUCLE TEMPORELLE (JOUR APRÈS JOUR) ---
        current_day = d_start
        
        while current_day < d_end:
            # Si on a déjà atteint l'objectif global, on arrête tout
            if len(all_tweets) >= total_limit:
                break

            # Définition de la fenêtre de 24h
            day_str = current_day.strftime("%Y-%m-%d")
            next_day_str = (current_day + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Ajout des dates à la requête
            # Note: Twitter API utilise 'until' comme exclusif, donc c'est parfait
            full_query = f"{base_query} since:{day_str} until:{next_day_str}"
            
            print(f"[API] Traitement du jour : {day_str}")

            # --- BOUCLE PAGINATION (POUR CE JOUR PRÉCIS) ---
            day_tweets = []
            next_cursor = None
            
            while len(day_tweets) < daily_quota:
                
                payload = {
                    "query": full_query,
                    "limit": 20, # Toujours 20 par page technique
                }
                if next_cursor:
                    payload["cursor"] = next_cursor

                try:
                    response = requests.get(API_URL, params=payload, headers=headers)
                    
                    if response.status_code == 429:
                        time.sleep(10)
                        continue 

                    if response.status_code != 200:
                        yield {"error": f"Erreur jour {day_str}: {response.status_code}"}
                        break

                    data = response.json()
                    batch = data.get('tweets', [])
                    
                    if not batch:
                        break # Pas de tweets ce jour-là, on passe

                    for t in batch:
                        # Déduplication globale
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

                    # Mise à jour de l'interface à chaque lot
                    duration = time.time() - start_time
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
                    
                    # Pause légère entre les pages d'un même jour
                    time.sleep(2) 

                except Exception as e:
                    yield {"error": str(e)}
                    break
            
            # --- FIN DE LA JOURNÉE ---
            current_day += timedelta(days=1)
            
            # Pause de sécurité entre les jours (Rate Limit Free Tier)
            # Important : On attend 5s avant de passer au jour suivant
            time.sleep(5)

        # Envoi Final
        duration = time.time() - start_time
        yield {
            "current_count": len(all_tweets),
            "target": total_limit,
            "data": all_tweets[:total_limit], # Coupe finale exacte
            "duration": round(duration, 2),
            "finished": True
        }
