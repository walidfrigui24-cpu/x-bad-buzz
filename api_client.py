import requests
import time
from datetime import datetime
from typing import Dict, Any, Generator

# --- CONFIGURATION (Free Tier) ---
API_KEY = "new1_c4a4317b0a7f4669b7a0baf181eb4861" 
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API Optimisé V3 (Strict Mode).
    1. Force le respect des dates (Filtrage Client-Side).
    2. Retry Mechanism : Ne s'arrête pas au premier échec vide.
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

        # 5. Dates (Pour l'API)
        if p.get('since'): parts.append(f"since:{p['since']}")
        if p.get('until'): parts.append(f"until:{p['until']}")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], limit: int = 50) -> Generator[Dict, None, None]:
        
        query_string = self.build_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        next_cursor = None
        start_time = time.time()
        empty_retries = 0  # Compteur pour les essais si page vide
        
        # Conversion des dates pour le filtrage strict (Client-Side)
        try:
            strict_start = datetime.strptime(params['since'], "%Y-%m-%d")
            strict_end = datetime.strptime(params['until'], "%Y-%m-%d")
            print(f"[SYSTEM] Filtrage strict activé : {strict_start.date()} -> {strict_end.date()}")
        except:
            strict_start = None
            print("[SYSTEM] Pas de filtrage date strict (Format invalide)")

        print(f"[SYSTEM] Démarrage... Cible : {limit}")

        while len(all_tweets) < limit:
            
            payload = {"query": query_string, "limit": 20}
            if next_cursor:
                payload["cursor"] = next_cursor

            try:
                response = requests.get(API_URL, params=payload, headers=headers)
                
                # Gestion Rate Limit
                if response.status_code == 429:
                    print("[API] Pause forcée (Rate Limit)...")
                    time.sleep(10)
                    continue 

                if response.status_code != 200:
                    yield {"error": f"Erreur API ({response.status_code})"}
                    break

                data = response.json()
                batch = data.get('tweets', [])
                
                # --- LOGIQUE DE RETRY (POUR NE PAS S'ARRÊTER TROP TÔT) ---
                if not batch:
                    empty_retries += 1
                    print(f"[API] Page vide reçue. Tentative {empty_retries}/3...")
                    if empty_retries >= 3:
                        print("[API] Arrêt : Trop de pages vides consécutives.")
                        break # Vraiment fini
                    
                    # On essaie de forcer le curseur suivant même si vide
                    if data.get('has_next_page') and data.get('next_cursor'):
                        next_cursor = data.get('next_cursor')
                        time.sleep(2)
                        continue
                    else:
                        break
                else:
                    empty_retries = 0 # Reset du compteur si on trouve des tweets

                # --- TRAITEMENT ET FILTRAGE STRICT ---
                added_in_batch = 0
                for t in batch:
                    if len(all_tweets) >= limit: break
                    
                    # Déduplication ID
                    if any(existing['id'] == t.get('id') for existing in all_tweets): continue

                    # --- FILTRE DATE STRICT ---
                    if strict_start:
                        try:
                            # Format Twitter: "2023-05-24T12:00:00.000Z"
                            # On prend juste la partie date YYYY-MM-DD
                            t_date_str = t.get('createdAt', '').split('T')[0]
                            t_date = datetime.strptime(t_date_str, "%Y-%m-%d")
                            
                            # Si le tweet est HORS de la plage, on l'ignore
                            if t_date < strict_start or t_date > strict_end:
                                continue
                        except:
                            pass # Si erreur de parsing, on garde par sécurité
                    # --------------------------
                    
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

                print(f"[API] +{added_in_batch} tweets valides (Total: {len(all_tweets)})")

                # Mise à jour UI
                duration = time.time() - start_time
                yield {
                    "current_count": len(all_tweets),
                    "target": limit,
                    "data": all_tweets,
                    "duration": round(duration, 2),
                    "finished": False
                }

                next_cursor = data.get('next_cursor')
                if not next_cursor or not data.get('has_next_page'):
                    print("[API] Plus de pages disponibles.")
                    break
                
                if len(all_tweets) >= limit:
                    break

                # Pause Optimisée
                time.sleep(4) 

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
