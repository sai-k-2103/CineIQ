import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

vader_analyzer = SentimentIntensityAnalyzer()
distilbert_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def calculate_review_sentiment(reviews: list):
    if not reviews:
        return 0.0
    
    vader_scores = []
    distilbert_scores = []
    
    for review in reviews:
        vader_scores.append(vader_analyzer.polarity_scores(review)['compound'])
        
        db_res = distilbert_analyzer(review, truncation=True, max_length=512)[0]
        score = db_res['score']
        if db_res['label'] == 'NEGATIVE':
            score = -score
        distilbert_scores.append(score)
        
    return (np.mean(vader_scores) * 0.4) + (np.mean(distilbert_scores) * 0.6)

def apply_sentiment_reranking(hybrid_recs_df, review_db: dict):
    df_out = hybrid_recs_df.copy()
    modifiers = []
    
    for m_id in df_out['movieId']:
        reviews = review_db.get(int(m_id), [])
        modifiers.append(calculate_review_sentiment(reviews))
        
    df_out['sentiment_modifier'] = modifiers
    df_out['final_score'] = df_out['hybrid_score'] + (df_out['sentiment_modifier'] * 0.15)
    df_out['final_score'] = df_out['final_score'].clip(0.0, 1.0).round(3)
    
    return df_out.sort_values(by='final_score', ascending=False).reset_index(drop=True)

def load_imdb_dataset(csv_path: str, valid_movie_ids: list, reviews_per_movie=3):
    try:
        imdb_df = pd.read_csv(csv_path).dropna(subset=['review'])
        sampled_reviews = imdb_df['review'].sample(frac=1, random_state=42).tolist()
        generated_datastore = {}
        idx = 0
        
        for m_id in valid_movie_ids:
            if idx + reviews_per_movie > len(sampled_reviews):
                idx = 0
            generated_datastore[int(m_id)] = sampled_reviews[idx : idx + reviews_per_movie]
            idx += reviews_per_movie
            
        print(f"-> SUCCESS: Cached {len(generated_datastore)} sentiment records.")
        return generated_datastore
    except Exception as e:
        print(f"-> SENTIMENT LOADER CRITICAL ERROR: {str(e)}")
        return {}