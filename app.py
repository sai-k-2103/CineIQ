import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="CINEIQ Dashboard", layout="wide")

st.title("CINEIQ: Explainable AI Recommendation Engine")
st.markdown("Discover movies powered by Hybrid Matrix Factorization (SVD + TF-IDF) and DistilBERT Sentiment Analysis.")
st.divider()

with st.sidebar:
    st.header("Engine Parameters")
    user_id = st.number_input("Simulate User ID", min_value=1, max_value=5000, value=2)
    st.markdown("---")
    st.markdown("**Architecture Stack:**")
    st.markdown("- **Matrix Math:** Scikit-Surprise (SVD)")
    st.markdown("- **NLP Text:** Hugging Face (DistilBERT)")
    st.markdown("- **API:** FastAPI (Python)")

col1, col2 = st.columns([2, 1])
with col1:
    search_query = st.text_input("Search for a movie to anchor your recommendations:", placeholder="Doctor Strange")

if search_query:
    with st.spinner("Querying Database..."):
        
        search_res = requests.get(f"{API_URL}/search", params={"Movie_name": search_query})
    
    if search_res.status_code == 200:
        movies = search_res.json().get("results", [])
        
        if movies:
            movie_titles = [m["title"] for m in movies]
            with col1:
                selected_title = st.selectbox("Select the exact release:", movie_titles)
            
            with col2:
                st.write("")
                st.write("") 
                generate_btn = st.button("Generate AI Predictions", type="primary", use_container_width=True)
            
            if generate_btn:
                with st.spinner("Running Matrix Factorization & NLP Sentiment Pipelines..."):
                    rec_res = requests.get(f"{API_URL}/recommend", params={"User_Id": user_id, "Movie_name": selected_title})
                    
                    if rec_res.status_code == 200:
                        data = rec_res.json()
                        recs_df = pd.DataFrame(data["recommendations"])
                        
                        st.success(f"Pipeline Complete! Engine Anchor: **{data['anchor_movie_used']}**")
                        
                        top_movie = recs_df.iloc[0]
                        nlp_boost = top_movie['final_score'] - top_movie['hybrid_score']
                        
                        st.subheader(f"Top Match: {top_movie['title']}")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Final CINEIQ Score", f"{top_movie['final_score']*100:.1f}%")
                        m2.metric("Base Algorithm Score", f"{top_movie['hybrid_score']*100:.1f}%")
                        m3.metric("NLP Sentiment Modifier", f"{nlp_boost*100:+.1f}%")

                        with st.expander("Explainable AI: Why was this recommended?"):
                            st.write(f"Our hybrid engine (combining user history and genre similarity) initially scored **{top_movie['title']}** at **{top_movie['hybrid_score']*100:.1f}%**.")
                            if nlp_boost > 0:
                                st.write(f"However, our **DistilBERT NLP model** scanned recent web reviews and detected highly positive sentiment, giving this movie a **+{nlp_boost*100:.1f}% boost** to its final ranking.")
                            elif nlp_boost < 0:
                                st.write(f"However, our **DistilBERT NLP model** scanned recent web reviews and detected negative sentiment, heavily penalizing this movie by **{nlp_boost*100:.1f}%**, though it remained strong enough mathematically to be your top choice.")
                            else:
                                st.write("The NLP sentiment analysis found neutral web reviews, leaving the mathematical matrix score unchanged.")

                        st.divider()

                        st.subheader("Pipeline Breakdown (Top 10 Matches)")
                        
                        plot_df = recs_df.head(10).copy()
                        plot_df['Sentiment Impact'] = plot_df['final_score'] - plot_df['hybrid_score']
                        plot_df = plot_df.rename(columns={'hybrid_score': 'Base Score', 'title': 'Movie'})
                        
                        fig = px.bar(
                            plot_df, 
                            x='Movie', 
                            y=['Base Score', 'Sentiment Impact'], 
                            title="Base Mathematical Score vs. NLP Sentiment Adjustments",
                            color_discrete_map={'Base Score': '#1f77b4', 'Sentiment Impact': '#ff7f0e'}
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("View Raw Matrix Data (JSON Output)"):
                            st.dataframe(recs_df, use_container_width=True)
                            
                    else:
                        st.error("Engine failed to generate recommendations.")
        else:
            st.warning("No movies found matching that query.")