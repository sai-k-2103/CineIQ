from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import re
import ast
from src.recommender import search_movie, get_hybrid_recommendations, cineiq_df
from src.sentiment import apply_sentiment_reranking, load_imdb_dataset

app = FastAPI(title="CINEIQ: Core Microservice API Engine")

IMDB_CSV_PATH = "CineIq_Data/imdb-50k/imdb_reviews.csv"
DEMO_RATINGS_PATH = "CineIq_Data/demo_ratings.csv"
RAW_RATINGS_PATH = "CineIq_Data/ml-25m/ratings.csv"
LINKS_CSV_PATH = "CineIq_Data/ml-25m/links.csv"
CREDITS_CSV_PATH = "CineIq_Data/tmdb-45k/credits.csv"

if not os.path.exists(DEMO_RATINGS_PATH) and os.path.exists(RAW_RATINGS_PATH):
    print("-> Slicing performance sample from raw ratings matrix...")
    chunk = pd.read_csv(RAW_RATINGS_PATH, nrows=1000000)
    chunk.to_csv(DEMO_RATINGS_PATH, index=False)

VALID_IDS = cineiq_df['movieId'].unique().tolist()
REVIEW_DATASTORE = load_imdb_dataset(csv_path=IMDB_CSV_PATH, valid_movie_ids=VALID_IDS, reviews_per_movie=3)
try:
    RATINGS_MATRIX = pd.read_csv(DEMO_RATINGS_PATH)
    print("-> SUCCESS: Ratings matrix loaded into RAM.")
except Exception:
    RATINGS_MATRIX = pd.DataFrame()

@app.get("/")
def read_root():
    return {"status": "online"}

@app.get("/search")
def api_search_movie(Movie_name: str):
    results = search_movie(Movie_name)
    if results is None:
        raise HTTPException(status_code=404, detail="No matches.")
    return {"results": [{"matrix_index": idx, "movieId": row["movieId"], "title": row["title"]} for idx, row in results.iterrows()]}

@app.get("/recommend")
def api_get_recommendations(User_Id: int, Movie_name: str):
    try:
        search_results = search_movie(Movie_name)
        if search_results is None:
            raise HTTPException(status_code=404, detail="Target missing.")
        
        best_match_idx = search_results.index[0]
        actual_movie_title = search_results.iloc[0]['title']

        if not RATINGS_MATRIX.empty:
            user_history = RATINGS_MATRIX[RATINGS_MATRIX['userId'] == User_Id]
            disliked_movie_ids = user_history[user_history['rating'] <= 2.5]['movieId'].tolist()
        else:
            disliked_movie_ids = []

        hybrid_results = get_hybrid_recommendations(user_id=User_Id, target_idx=best_match_idx).copy()
        
        penalty_applied = False
        
        if disliked_movie_ids:
            hated_movies_df = cineiq_df[cineiq_df['movieId'].isin(disliked_movie_ids)]
            hated_genres_set = set()
            for genres_str in hated_movies_df['genres'].dropna():
                for g in (genres_str.split('|') if '|' in genres_str else [genres_str]):
                    hated_genres_set.add(g.strip().lower())
            
            penalized_scores = []
            for _, row in hybrid_results.iterrows():
                current_genres = set([g.strip().lower() for g in str(row['genres']).split('|')])
                genre_collision = current_genres.intersection(hated_genres_set)
                
                penalty = 1.0
                if genre_collision and len(current_genres) > 0:
                    if (len(genre_collision) / len(current_genres)) > 0.5:
                        penalty = 0.70  
                        penalty_applied = True
                penalized_scores.append(row['hybrid_score'] * penalty)
            hybrid_results['hybrid_score'] = penalized_scores

        final_reranked_results = apply_sentiment_reranking(hybrid_results, REVIEW_DATASTORE)
        
        return {
            "requested_by_user": User_Id,
            "anchor_movie_used": actual_movie_title,
            "negative_filter_applied": penalty_applied,
            "recommendations": final_reranked_results[['movieId', 'title', 'hybrid_score', 'final_score']].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/profile")
def get_user_profile(User_Id: int):
    try:
        if RATINGS_MATRIX.empty:
             raise HTTPException(status_code=500, detail="Matrix unavailable.")
             
        user_history = RATINGS_MATRIX[(RATINGS_MATRIX['userId'] == User_Id) & (RATINGS_MATRIX['rating'] >= 3.5)]
        if user_history.empty:
            raise HTTPException(status_code=404, detail="No history found.")
            
        merged = user_history.merge(cineiq_df, on='movieId', how='inner')
        
        genre_counts = {}
        for g_str in merged['genres'].dropna():
            for g in (g_str.split('|') if '|' in g_str else [g_str]):
                genre_counts[g.strip()] = genre_counts.get(g.strip(), 0) + 1
                
        top_genres = dict(sorted(genre_counts.items(), key=lambda item: item[1], reverse=True)[:8])
                
        decade_counts = {}
        for title in merged['title']:
            match = re.search(r'\((\d{4})\)', title)
            if match:
                decade_str = f"{(int(match.group(1)) // 10) * 10}s"
                decade_counts[decade_str] = decade_counts.get(decade_str, 0) + 1

        director_counts = {}
        actor_counts = {}
        
        if os.path.exists(LINKS_CSV_PATH) and os.path.exists(CREDITS_CSV_PATH):
            links_df = pd.read_csv(LINKS_CSV_PATH)
            liked_movie_ids = merged['movieId'].unique()
            user_links = links_df[links_df['movieId'].isin(liked_movie_ids)]
            user_tmdb_ids = user_links['tmdbId'].dropna().astype(int).tolist()
            
            credits_df = pd.read_csv(CREDITS_CSV_PATH)
            user_credits = credits_df[credits_df['id'].isin(user_tmdb_ids)]
            
            for _, row in user_credits.iterrows():
                try:
                    crew_data = ast.literal_eval(row['crew']) if isinstance(row['crew'], str) else []
                    for member in crew_data:
                        if member.get('job') == 'Director':
                            d_name = member.get('name')
                            if d_name:
                                director_counts[d_name] = director_counts.get(d_name, 0) + 1
                except Exception:
                    pass
                
                try:
                    cast_data = ast.literal_eval(row['cast']) if isinstance(row['cast'], str) else []
                    for member in cast_data[:5]:  
                        a_name = member.get('name')
                        if a_name:
                            actor_counts[a_name] = actor_counts.get(a_name, 0) + 1
                except Exception:
                    pass

        top_directors = sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_liked_movies": len(merged),
            "genres": top_genres,
            "decades": decade_counts,
            "directors": [{"Director": d, "Count": c} for d, c in top_directors],
            "actors": [{"Actor": a, "Count": c} for a, c in top_actors]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))