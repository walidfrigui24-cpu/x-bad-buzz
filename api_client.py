import requests
import time
from typing import Dict, Any, Generator

# --- CONFIGURATION (Free Tier) ---
API_KEY = "new1_c4a4317b0a7f4669b7a0baf181eb4861" 
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API Twitter avec constructeur de requête AVANCÉ.
    Gère tous les opérateurs booléens, filtres de comptes et métriques.
    """
    
    def build_query(self, p: Dict[str, Any]) -> str:
        """Construit la chaîne de requête complète (Syntaxe Twitter Standard)."""
        parts = []
        
        # 1. Mots-clés (Sémantique)
        if p.get('all_words'): 
            parts.append(p['all_words'])
        
        if p.get('exact_phrase'): 
            parts.append(f'"{p["exact_phrase"]}"')
        
        if p.get('any_words'): 
            # Syntaxe : (mot1 OR mot2)
            words = p['any_words'].split()
            if len(words) > 1:
                parts.append(f"({' OR '.join(words)})")
            else:
                parts.append(words[0])
        
        if p.get('none_words'): 
            # Syntaxe : -mot1 -mot2
            for w in p['none_words'].split():
                parts.append(f"-{w}")
        
        if p.get('hashtags'): 
            parts.append(p['hashtags'])
        
        if p.get('lang') and p['lang'] != "Tout": 
            parts.append(f"lang:{p['lang']}")

        # 2. Comptes (People)
        if p.get('from_accounts'): 
            parts.append(f"from:{p['from_accounts'].replace('@', '')}")
        
        if p.get('to_accounts'): 
            parts.append(f"to:{p['to_accounts'].replace('@', '')}")
            
        if p.get('mention_accounts'): 
            parts.append(f"@{p['mention_accounts'].replace('@', '')}")

        # 3. Engagement (Metrics)
        if p.get('min_faves'): parts.append(f"min_faves:{p['min_faves']}")
        if p.get('min_retweets'): parts.append(f"min_retweets:{p['min_retweets']}")
        if p.get('min_replies'): parts.append(f"min_replies:{p['min_replies']}")

        # 4. Filtres Techniques (Links/Replies)
        # Gestion des liens
        if p.get('links_filter') == "Exclure les liens":
            parts.append("-filter:links")
        elif p.get('links_filter') == "Uniquement avec liens":
            parts.append("filter:links")
            
        # Gestion des réponses
        if p.get('replies_filter') == "Exclure les réponses":
            parts.append("exclude:replies")
        elif p.get('replies_filter') == "Uniquement les réponses":
            parts.append("filter:replies")

        # 5. Dates
        if p.get('since'): parts.append(f"since:{p['since']}")
        if p.get('until'): parts.append(f"until:{p['until']}")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], limit: int = 50) -> Generator[Dict, None, None]:
        """Générateur d'extraction avec pause de sécurité (6s)."""
        query_string = self.build_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        next_cursor = None
        start_time = time.time()
        
        print(f"[SYSTEM] Query : {query_string}") # Debug dans la console

        while len(all_tweets) < limit:
            payload = {"query": query_string, "limit": 20}
            if next_cursor:
                payload["cursor"] = next_cursor

            try:
                response = requests.get(API_URL, params=payload, headers=headers)
                
                if response.status_code == 429:
                    time.sleep(10)
                    continue 

                if response.status_code != 200:
                    yield {"error": f"Erreur API ({response.status_code})"}
                    break

                data = response.json()
                batch = data.get('tweets', [])
                
                if not batch: break 

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

                duration = time.time() - start_time
                yield {
                    "current_count": len(all_tweets),
                    "target": limit,
                    "data": all_tweets,
                    "duration": round(duration, 2),
                    "finished": False
                }

                next_cursor = data.get('next_cursor')
                if not next_cursor or not data.get('has_next_page'): break
                if len(all_tweets) >= limit: break

                time.sleep(6) # Pause obligatoire Free Tier

            except Exception as e:
                yield {"error": str(e)}
                break

        duration = time.time() - start_time
        yield {
            "current_count": len(all_tweets),
            "target": limit,
            "data": all_tweets[:limit],
            "duration": round(duration, 2),
            "finished": True
        }
