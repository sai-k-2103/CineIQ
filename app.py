import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="CINEIQ Dashboard", layout="wide")

st.title("🎬 CINEIQ: Explainable AI Recommendation Dashboard")
st.markdown("Multi-algorithmic delivery platform powered by Collaborative SVD matrices, Content TF-IDFs, and Sentiment Processing.")
st.divider()

with st.sidebar:
    st.header("Engine Parameters")
    user_id = st.number_input("Simulate User ID", min_value=1, max_value=5000, value=2)
    st.markdown("---")
    st.markdown("**Architecture Stack:**")
    st.markdown("- **Matrix Math:** Surprise (SVD)")
    st.markdown("- **NLP Text Engine:** Hugging Face (DistilBERT)")
    st.markdown("- **Serving Gateway:** FastAPI (Python)")

tab1, tab2 = st.tabs(["🎯 Get AI Recommendations", "📊 My Taste Analytics"])

with tab1:
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
                    st.write(""); st.write("")
                    generate_btn = st.button("Generate AI Predictions", type="primary", use_container_width=True)
                
                if generate_btn:
                    with st.spinner("Executing Inference Pipelines..."):
                        rec_res = requests.get(f"{API_URL}/recommend", params={"User_Id": user_id, "Movie_name": selected_title})
                        if rec_res.status_code == 200:
                            data = rec_res.json()
                            recs_df = pd.DataFrame(data["recommendations"])
                            
                            st.success(f"Pipeline Complete! Engine Anchor: **{data['anchor_movie_used']}**")
                            if data.get("negative_filter_applied"):
                                st.info("ℹ️ Negative Preference Filter Engaged: Recommendations containing historical hate markers suppressed.")

                            top_movie = recs_df.iloc[0]
                            nlp_boost = top_movie['final_score'] - top_movie['hybrid_score']
                            
                            st.subheader(f"🏆 Top Match: {top_movie['title']}")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Final CINEIQ Score", f"{top_movie['final_score']*100:.1f}%")
                            m2.metric("Base Algorithm Score", f"{top_movie['hybrid_score']*100:.1f}%")
                            m3.metric("NLP Sentiment Modifier", f"{nlp_boost*100:+.1f}%")

                            with st.expander("Explainable AI: Why was this recommended?"):
                                st.write(f"Our hybrid matrix scored **{top_movie['title']}** at **{top_movie['hybrid_score']*100:.1f}%** based on collaborative traits.")
                                if nlp_boost > 0:
                                    st.write(f"DistilBERT evaluated text reviews, applying a **+{nlp_boost*100:.1f}% positive adjustment**.")
                                elif nlp_boost < 0:
                                    st.write(f"DistilBERT identified audience churn signals, applying a **{nlp_boost*100:.1f}% reduction penalty**.")
                            
                            st.divider()
                            st.subheader("Pipeline Breakdown")
                            plot_df = recs_df.head(10).copy()
                            plot_df['Sentiment Impact'] = plot_df['final_score'] - plot_df['hybrid_score']
                            plot_df = plot_df.rename(columns={'hybrid_score': 'Base Score', 'title': 'Movie'})
                            
                            fig = px.bar(plot_df, x='Movie', y=['Base Score', 'Sentiment Impact'], barmode='stack', color_discrete_map={'Base Score': '#1f77b4', 'Sentiment Impact': '#ff7f0e'})
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No matches discovered.")

with tab2:
    st.header(f"Taste Footprint Analysis — User Baseline Profile #{user_id}")
    profile_res = requests.get(f"{API_URL}/user/profile", params={"User_Id": user_id})
    
    if profile_res.status_code == 200:
        p_data = profile_res.json()
        st.metric("Total Historical Liked Samples Extracted", p_data["total_liked_movies"])
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🕸️ Top Core Genre Distribution")
            g_df = pd.DataFrame(list(p_data["genres"].items()), columns=["Genre", "Count"])
            
            if not g_df.empty:
                r_values = g_df['Count'].tolist()
                r_values.append(r_values[0])
                theta_values = g_df['Genre'].tolist()
                theta_values.append(theta_values[0])
                
                fig_radar = go.Figure(data=go.Scatterpolar(r=r_values, theta=theta_values, fill='toself', line_color='#e74c3c'))
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.write("Insufficient data for radar.")
        with c2:
            st.subheader("⏳ Temporal Preferences Distribution")
            d_df = pd.DataFrame(list(p_data["decades"].items()), columns=["Decade", "Count"]).sort_values("Decade")
            st.plotly_chart(px.bar(d_df, x='Decade', y='Count', color_discrete_sequence=['#2ecc71']), use_container_width=True)
            
        st.divider()
        st.subheader("👥 Top Professional Talent Affinities")
        st.markdown("Calculated dynamically by parsing crew and top-billed cast listings across underlying metadata nodes.")
        
        ca1, ca2 = st.columns(2)
        with ca1:
            st.markdown("🎬 **Favorite Directors**")
            dir_df = pd.DataFrame(p_data["directors"])
            if not dir_df.empty:
                dir_df.columns = ["Director Name", "Watch Count"]
                st.dataframe(dir_df, use_container_width=True, hide_index=True)
            else:
                st.write("No historical director data mapping found.")
                
        with ca2:
            st.markdown("🎭 **Favorite Actors**")
            act_df = pd.DataFrame(p_data["actors"])
            if not act_df.empty:
                act_df.columns = ["Actor Name", "Watch Count"]
                st.dataframe(act_df, use_container_width=True, hide_index=True)
            else:
                st.write("No historical actor data mapping found.")
    else:
        st.info("Adjust User ID index to re-initialize layout processing parameters.")