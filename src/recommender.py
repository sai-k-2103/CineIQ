import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

modify_path = "CineIq_Data/modify/"
cineiq_df = pd.read_csv(modify_path + "cineiq_metadata.csv").fillna("")

with open(modify_path + "svd_model.pkl","rb") as f:
    svd_model=pickle.load(f)

cineiq_df["soup"] = (
    cineiq_df["overview"] + " " +
    (cineiq_df["genres"] + " ") * 2 +
    (cineiq_df["keywords"] + " ") * 3
)
cineiq_df = cineiq_df.fillna("")

tfidf = TfidfVectorizer(stop_words="english", min_df=4,max_features=20000)
tfidf_matrix = tfidf.fit_transform(cineiq_df["soup"])

def search_movie(query: str):
    matches = cineiq_df[cineiq_df["title"].str.lower().str.contains(query.lower(), na=False, regex=False)].copy()
    
    if matches.empty:
        return None
    
    matches['title_length'] = matches['title'].str.len()
    matches = matches.sort_values('title_length')
    
    return matches[["movieId", "title"]].head(10)

def get_hybrid_recommendations(user_id: int, target_idx: int, n=10, svd_weight = 0.7, content_weight = 0.3):
    target_movie_id = cineiq_df.iloc[target_idx]["movieId"]
    matched_title = cineiq_df.iloc[target_idx]["title"]

    movie_vector = tfidf_matrix[target_idx]
    sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()

    candidate_indices = sim_scores.argsort()[::-1]
    candidate_indices = candidate_indices[1:51]

    candidates = cineiq_df.iloc[candidate_indices][["movieId","title"]].copy()
    candidates["content_score"] = sim_scores[candidate_indices]

    svd_preds = []
    for m_id in candidates["movieId"]:
        raw_pred = svd_model.predict(uid=user_id, iid=m_id).est
        normalised_svd = (raw_pred - 0.5)/4.5
        svd_preds.append(normalised_svd)

    candidates["svd_score"] = svd_preds

    candidates["hybrid_score"] = (candidates["svd_score"] * svd_weight) + (candidates["content_score"] * content_weight)

    final_recs = candidates.sort_values(by="hybrid_score",ascending=False).head(n)

    return final_recs[["movieId","title","content_score","svd_score","hybrid_score"]].reset_index(drop=True)

