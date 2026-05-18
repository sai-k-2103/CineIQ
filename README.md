# CINEIQ

Explainable Hybrid Movie Recommendation Engine built using FastAPI, Streamlit, Scikit-Surprise, and Hugging Face Transformers.

---

## Overview

CINEIQ is a full-stack hybrid movie recommendation system that combines collaborative filtering, content-based filtering, and NLP sentiment analysis to generate personalized and explainable movie recommendations.

The system integrates:
- SVD-based collaborative filtering
- TF-IDF content similarity
- Transformer-based sentiment analysis
- Explainable recommendation scoring
- Interactive analytics dashboard

---

## Features

- Hybrid Recommendation Engine
- Personalized Recommendations
- Content-Based Similarity Search
- Sentiment-Based Re-ranking
- User Analytics Dashboard
- MLflow Experiment Tracking
- FastAPI REST APIs
- Streamlit Frontend Interface
- Hybrid Recommendation Scoring

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| Machine Learning | Scikit-Surprise |
| NLP | Hugging Face Transformers |
| Sentiment Analysis | VADER |
| Experiment Tracking | MLflow |
| Visualization | Plotly |
| Deep Learning Backend | PyTorch |

---

## Project Structure

```text
CineIQ/
│
├── CineIq_Data/
|   ├── demo_ratings.csv
│   ├── imdb-50k/
│   │   └── imdb_reviews.csv
│   │
│   ├── ml-25m/
│   │   ├── genome-scores.csv
│   │   ├── genome-tags.csv
│   │   ├── links.csv
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   ├── README.txt
│   │   └── tags.csv
│   │
│   ├── modify/
│   │   ├── cineiq_metadata.csv
│   │   └── svd_model.pkl
│   │
│   └── tmdb-45k/
│       ├── credits.csv
│       ├── keywords.csv
│       ├── links.csv
│       ├── links_small.csv
│       ├── movies_metadata.csv
│       ├── ratings.csv
│       └── ratings_small.csv
│
├── src/
│   ├── main.py
│   ├── recommender.py
│   ├── sentiment.py
│   └── train.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/sai-k-2103/CineIQ.git
cd CineIQ
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

> Always activate the virtual environment before running any command.

CINEIQ runs 3 services in separate terminals.

---

## Terminal 1 — Train Model & MLflow

```bash
python src/train.py
mlflow ui
```

MLflow:
```
http://127.0.0.1:5000
```

---

## Terminal 2 — FastAPI Backend

```bash
uvicorn src.main:app --reload
```

Backend:
```
http://127.0.0.1:8000
```

Swagger:
```
http://127.0.0.1:8000/docs
```

---

## Terminal 3 — Streamlit Frontend

```bash
streamlit run app.py
```

Frontend:
```
http://localhost:8501
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /search` | Search movies |
| `GET /recommend` | Generate recommendations |
| `GET /similar` | Find similar movies |
| `GET /user/profile` | User analytics profile |

---

## Model Metrics

| Metric | Score |
|---|---|
| RMSE | 0.8456 |
| MAE | 0.6454 |

Demo Link : https://drive.google.com/drive/folders/1C87OmPHEJhonLR9CPUPIPx8hGdMjy4OS