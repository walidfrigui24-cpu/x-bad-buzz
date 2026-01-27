import requests
import time
from typing import Dict, Any, Generator

# --- CONFIGURATION ET AUTHENTIFICATION ---
# Clé API fournie (Free Tier)
API_KEY = "new1_81efcd1da3a14aa5919e3082b164b068" 
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Classe cliente pour l'API Twitter (TwitterAPI.io).
    Utilise un modèle de GÉNÉRATEUR (yield) pour envoyer des mises à jour 
    en temps réel à l'interface utilisateur.
    """
    
    def build_query(self, p: Dict[str, Any]) -> str:
        """
        Construction de la chaîne de requête booléenne complexe.
        """
        parts = []
        # Sémantique
        if p.get('all_words'): parts.append(p['all_words'])
        if p.get('exact_phrase'): parts.append(f'"{p["exact_phrase"]}"')
        if p.get('any_words'):
            words = p['any_words'].split()
            parts.append(f"({' OR '.join(words)})" if len(words) > 1 else p['any_words'])
        if p.get('none_words'):
            for w in p['none_words'].split(): parts.append(f"-{w}")
        
        # Filtres Techniques
        if p.get('hashtags'): parts.append(p['hashtags'])
        if p.get('lang') and p['lang'] != "Tout": parts.append(f"lang:{p['lang']}")
        if p.get('from_accounts'): parts.append(f"from:{p['from_accounts'].replace('@', '')}")
        
        # Filtres Temporels
        if p.get('since'): parts.append(f"since:{p['since']}")
        if p.get('until'): parts.append(f"until:{p['until']}")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], limit: int = 50) -> Generator[Dict, None, None]:
        """
        Exécute l'extraction de manière itérative.
        Renvoie l'état après chaque page pour mettre à jour la barre de progression.
        
        Yields:
            Dict: État actuel {current, target, data, finished}
        """
        query_string = self.build_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        next_cursor = None
        page_num = 1
        start_time = time.time()
        
        print(f"[SYSTEM] Initialisation... Cible : {limit}")

        while len(all_tweets) < limit:
            
            # Préparation de la requête (Toujours par lots de 20)
            payload = {"query": query_string, "limit": 20}
            if next_cursor:
                payload["cursor"] = next_cursor

            try:
                # Exécution HTTP
                response = requests.get(API_URL, params=payload, headers=headers)
                
                # Gestion Rate Limit (Erreur 429)
                if response.status_code == 429:
                    # Si le serveur dit stop, on attend 10s et on réessaie
                    time.sleep(10)
                    continue 

                if response.status_code != 200:
                    yield {"error": f"Erreur API: {response.status_code}"}
                    break

                data = response.json()
                batch = data.get('tweets', [])
                
                # Arrêt si aucune donnée
                if not batch:
                    break 

                # Traitement et Déduplication
                for t in batch:
                    if any(existing['id'] == t.get('id') for existing in all_tweets):
                        continue
                    
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

                # --- YIELD : Envoi de la mise à jour à l'interface ---
                duration = time.time() - start_time
                yield {
                    "current_count": len(all_tweets),
                    "target": limit,
                    "data": all_tweets, # On renvoie tout ce qu'on a jusqu'ici
                    "duration": round(duration, 2),
                    "finished": False
                }

                # Pagination
                next_cursor = data.get('next_cursor')
                has_next = data.get('has_next_page')

                if not next_cursor or not has_next:
                    break
                
                if len(all_tweets) >= limit:
                    break

                # --- PAUSE CRITIQUE (Free Tier) ---
                # 6 secondes pour garantir la stabilité
                time.sleep(6)
                page_num += 1

            except Exception as e:
                yield {"error": str(e)}
                break

        # Envoi final
        duration = time.time() - start_time
        yield {
            "current_count": len(all_tweets),
            "target": limit,
            "data": all_tweets[:limit], # Coupe exacte
            "duration": round(duration, 2),
            "finished": True
        }