import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="FlickFinder Dashboard", page_icon="🎬", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, 'data', 'processed', 'processed_movies.csv')
    recs_path = os.path.join(base_dir, 'outputs', 'recs_content.csv')
    eval_path = os.path.join(base_dir, 'outputs', 'evaluation_report.json')
    
    # Load Movies
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)
    else:
        movies_df = pd.DataFrame(columns=['movieId', 'title', 'genres'])
        
    # Load Recs
    if os.path.exists(recs_path):
        recs_df = pd.read_csv(recs_path)
    else:
        recs_df = pd.DataFrame(columns=['userId', 'movieId', 'predicted_rating', 'rank', 'method'])
        
    # Load Eval
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)
    else:
        eval_data = {}
        
    return movies_df, recs_df, eval_data

movies_df, recs_df, eval_data = load_data()

st.title("🎬 FlickFinder: Movie Recommendation Engine")
st.markdown("Welcome to the **FlickFinder Model Interactive Dashboard**. Use the sidebar to navigate through system evaluation and user-based recommendations.")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Top Recommendations", "System Metrics", "Data Exploration"])

if page == "Top Recommendations":
    st.header("👤 Personalized Recommendations")
    
    if not recs_df.empty:
        # Get unique users
        unique_users = sorted(recs_df['userId'].unique())
        selected_user = st.selectbox("Select User ID Profile:", unique_users)
        
        # Filter recs
        user_recs = recs_df[recs_df['userId'] == selected_user].sort_values(by='rank')
        
        # Merge with movie metadata to show titles instead of just IDs
        if not movies_df.empty:
            merged_recs = pd.merge(user_recs, movies_df[['movieId', 'title', 'genres', 'release_year']], on='movieId', how='left')
            
            # Format display
            display_df = merged_recs[['rank', 'title', 'genres', 'release_year', 'predicted_rating', 'method']]
            display_df.columns = ['Rank', 'Movie Title', 'Genres', 'Release', 'Predicted Score', 'Origin Strategy']
            
            st.markdown(f"### Top {len(display_df)} Matches for User `{selected_user}`")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.info("💡 **Note**: Because the Collaborative SVD component bypassed successfully during pipeline compile constraint on Windows, these metrics flow dynamically via the Content-Based TF-IDF Natural Language System backbone!")
        else:
            st.dataframe(user_recs)
    else:
        st.warning("No recommendation data found. Ensure the pipeline ran successfully.")

elif page == "System Metrics":
    st.header("📊 Pipeline Evaluation Metrics")
    
    if eval_data:
        # Display key metrics using Streamlit Columns
        st.subheader("Key Model Vital Stats (Content-Based Engine)")
        
        col1, col2, col3 = st.columns(3)
        beyond_acc = eval_data.get("Beyond_Accuracy", {}).get("Content", {})
        ranking = eval_data.get("Ranking_Metrics", {}).get("Content", {})
        
        with col1:
            st.metric("NDCG @ 10", f"{ranking.get('NDCG@10', 0):.4f}")
            st.metric("Precision @ 10", f"{ranking.get('P@10', 0):.4f}")
        with col2:
            st.metric("Catalog Coverage", f"{beyond_acc.get('Catalog Coverage (%)', 0)}%")
            st.metric("Recall @ 10", f"{ranking.get('Recall@10', 0):.4f}")
        with col3:
            st.metric("Intra-List Diversity", f"{beyond_acc.get('Intra-List Diversity', 0):.4f}")
            st.metric("Serendipity Quotient", f"{beyond_acc.get('Serendipity Quotient', 0):.4f}")
            
        st.divider()
        st.subheader("Raw JSON Evaluation Report File")
        st.json(eval_data)
        
    else:
        st.warning("evaluation_report.json is completely empty. Run pipeline evaluation first (Task 11).")

elif page == "Data Exploration":
    st.header("📈 Data Universe Exploration")
    
    if not movies_df.empty:
        st.subheader(f"Total Movies Available: {len(movies_df):,}")
        
        search_query = st.text_input("🔍 Search Movie Database by Title", "")
        if search_query:
            results = movies_df[movies_df['title'].str.contains(search_query, case=False, na=False)]
            st.markdown(f"Found **{len(results)}** matches.")
            st.dataframe(results[['movieId', 'title', 'release_year', 'genres', 'director']].head(50), use_container_width=True, hide_index=True)
        else:
            st.dataframe(movies_df[['movieId', 'title', 'release_year', 'genres']].head(100), use_container_width=True, hide_index=True)
    else:
        st.error("Processed Movies dataset is missing.")

st.sidebar.markdown("---")
st.sidebar.caption("Data Mining Capstone Project")
