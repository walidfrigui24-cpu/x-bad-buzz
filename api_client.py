import requests
import time
from typing import Dict, Any, Generator

# --- CONFIGURATION (Free Tier) ---
API_KEY = "new1_81efcd1da3a14aa5919e3082b164b068" 
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API Optimisé pour la Vitesse et le Volume.
    Récupère les données en continu (Pagination Smart) sans fragmentation temporelle
    pour éviter les délais inutiles sur les jours vides.
    """
    
    def build_query(self, p: Dict[str, Any]) -> str:
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

        # 5. Dates (Global Range)
        if p.get('since'): parts.append(f"since:{p['since']}")
        if p.get('until'): parts.append(f"until:{p['until']}")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], limit: int = 50) -> Generator[Dict, None, None]:
        """
        Générateur Rapide (Global Search).
        """
        query_string = self.build_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        next_cursor = None
        start_time = time.time()
        
        print(f"[SYSTEM] Mode Rapide activé. Cible : {limit}")

        # Boucle continue jusqu'à atteindre la limite
        while len(all_tweets) < limit:
            
            payload = {"query": query_string, "limit": 20}
            if next_cursor:
                payload["cursor"] = next_cursor

            try:
                response = requests.get(API_URL, params=payload, headers=headers)
                
                # Gestion Rate Limit (Pause et reprise automatique)
                if response.status_code == 429:
                    time.sleep(10)
                    continue 

                if response.status_code != 200:
                    yield {"error": f"Erreur API ({response.status_code})"}
                    break

                data = response.json()
                batch = data.get('tweets', [])
                
                # Si pas de tweets, on vérifie si c'est vraiment la fin
                if not batch:
                    # Parfois l'API renvoie une page vide mais il y a une suite
                    if not data.get('has_next_page'):
                        break
                    else:
                        # On essaie de forcer la page suivante
                        next_cursor = data.get('next_cursor')
                        time.sleep(2)
                        continue

                # Ajout des tweets
                for t in batch:
                    if len(all_tweets) >= limit: break # Arrêt précis
                    
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

                # Mise à jour UI
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
                    break
                
                if len(all_tweets) >= limit:
                    break

                # Pause optimisée (5s est le strict minimum pour éviter le blocage)
                time.sleep(5) 

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
