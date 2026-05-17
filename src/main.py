from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.recommender import search_movie, get_hybrid_recommendations, cineiq_df
from src.sentiment import apply_sentiment_reranking, load_imdb_dataset

app = FastAPI(title="CINEIQ: Advanced Hybrid Recommendation Engine API")

IMDB_CSV_PATH = "CineIq_Data/imdb-50k/imdb_reviews.csv" 
VALID_IDS = cineiq_df['movieId'].unique().tolist()
REVIEW_DATASTORE = load_imdb_dataset(csv_path=IMDB_CSV_PATH, valid_movie_ids=VALID_IDS, reviews_per_movie=3)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the CINEIQ Recommendation Engine Core API API Server"}

@app.get("/search")
def api_search_movie(Movie_name: str):
    """Standalone search to just browse the database."""
    results = search_movie(Movie_name)
    if results is None:
        raise HTTPException(status_code=404, detail=f"No movies matching query '{Movie_name}' found.")
    
    output = []
    for idx, row in results.iterrows():
        output.append({"matrix_index": idx, "movieId": row["movieId"], "title": row["title"]})
    return {"query": Movie_name, "total_matches": len(output), "results": output}

@app.get("/recommend")
def api_get_recommendations(User_Id: int, Movie_name: str):
    """One-click recommendation: Type a movie name, get recommendations instantly."""
    try:
        search_results = search_movie(Movie_name)
        if search_results is None:
            raise HTTPException(status_code=404, detail=f"Could not find any movie matching '{Movie_name}'. Please try another title.")
        
        best_match_idx = search_results.index[0]
        actual_movie_title = search_results.iloc[0]['title']

        hybrid_results = get_hybrid_recommendations(
            user_id=User_Id,
            target_idx=best_match_idx
        )
        
        final_reranked_results = apply_sentiment_reranking(hybrid_results, REVIEW_DATASTORE)
        
        return {
            "requested_by_user": User_Id,
            "anchor_movie_used": actual_movie_title, # Let the user know exactly what movie we matched with!
            "recommendations": final_reranked_results[['movieId', 'title', 'hybrid_score', 'final_score']].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))