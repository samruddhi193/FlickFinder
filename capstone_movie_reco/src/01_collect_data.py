"""
Module for data collection and TMDB API metadata enrichment.
"""
import os
import time
import yaml
import requests
import pandas as pd
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 500, 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def fetch_movie_metadata(tmdb_id, api_key, base_url, session):
    """
    Fetch movie metadata (overview, cast, director) from TMDB.
    Uses empty strings if the request fails.
    """
    if pd.isna(tmdb_id):
        return "", "", ""
        
    url = f"{base_url}/movie/{int(tmdb_id)}"
    credits_url = f"{base_url}/movie/{int(tmdb_id)}/credits"
    params = {"api_key": api_key}
    
    overview = ""
    cast_str = ""
    director = ""
    
    try:
        # Fetch overview
        response = session.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            overview = data.get('overview', "")
        
        # Fetch credits
        cred_resp = session.get(credits_url, params=params, timeout=10)
        if cred_resp.status_code == 200:
            cred_data = cred_resp.json()
            cast = cred_data.get('cast', [])[:5]
            cast_str = " ".join([c.get('name', '') for c in cast])
            crew = cred_data.get('crew', [])
            directors = [c.get('name', '') for c in crew if c.get('job') == 'Director']
            director = directors[0] if directors else ""
            
    except Exception as e:
        pass
        
    return overview, cast_str, director

def collect_data():
    """
    Reads local MovieLens data, fetching additional metadata from TMDB, 
    and saves processed raw data.
    """
    config = load_config()
    api_key = config.get('tmdb_api', {}).get('api_key')
    base_url = config.get('tmdb_api', {}).get('base_url', 'https://api.themoviedb.org/3')
    raw_dir = config.get('paths', {}).get('raw_data', 'data/raw/')
    
    base_raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), raw_dir)
    os.makedirs(base_raw_dir, exist_ok=True)
    
    ratings_path = os.path.join(base_raw_dir, 'ratings.csv')
    movies_path = os.path.join(base_raw_dir, 'movies.csv')
    tags_path = os.path.join(base_raw_dir, 'tags.csv')
    links_path = os.path.join(base_raw_dir, 'links.csv')
    metadata_path = os.path.join(base_raw_dir, 'metadata.csv')
    out_movies_path = os.path.join(base_raw_dir, 'raw_movies.csv')
    out_ratings_path = os.path.join(base_raw_dir, 'raw_ratings.csv')

    print("Reading CSV files from data/raw/...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)
    links = pd.read_csv(links_path)
    
    # Merge movies with links
    merged = pd.merge(movies, links, on='movieId', how='left')
    
    # Load existing metadata
    existing_metadata = pd.DataFrame()
    if os.path.exists(metadata_path):
        existing_metadata = pd.read_csv(metadata_path)
        existing_ids = set(existing_metadata['movieId'])
    else:
        existing_ids = set()
        
    all_metadata = existing_metadata.to_dict('records')
    batch_metadata = []
    
    success_count = 0
    attempt_count = 0
    
    print("Fetching TMDB metadata...")
    session = get_session()
    for idx, row in merged.iterrows():
        movie_id = row['movieId']
        if movie_id in existing_ids:
            continue
            
        tmdb_id = row['tmdbId']
        attempt_count += 1
        
        overview, cast, director = fetch_movie_metadata(tmdb_id, api_key, base_url, session)
        if overview or cast or director:
            success_count += 1
            
        meta_dict = {
            'movieId': movie_id,
            'overview': overview,
            'cast': cast,
            'director': director
        }
        batch_metadata.append(meta_dict)
        all_metadata.append(meta_dict)
        
        time.sleep(0.02)
        
        if len(batch_metadata) >= 100:
            pd.DataFrame(batch_metadata).to_csv(metadata_path, mode='a', 
                                                header=not os.path.exists(metadata_path), 
                                                index=False)
            batch_metadata = []
            
    # Save remaining
    if batch_metadata:
        pd.DataFrame(batch_metadata).to_csv(metadata_path, mode='a', 
                                            header=not os.path.exists(metadata_path), 
                                            index=False)
                                            
    # Create final raw_movies
    # Need to extract release year. The instructions for Task 4 mention extracting release year with regex,
    # but Task 3 says "Saves final merged file to data/raw/raw_movies.csv with columns: movieId, title, genres, release_year, overview, cast, director".
    # I should extract it now if it's expected in Task 3.
    final_meta = pd.read_csv(metadata_path)
    final_movies = pd.merge(merged, final_meta, on='movieId', how='left')
    
    # extract release year if not present
    if 'release_year' not in final_movies.columns:
        final_movies['release_year'] = final_movies['title'].str.extract(r'\((\d{4})\)')
        
    final_movies = final_movies[['movieId', 'title', 'genres', 'release_year', 'overview', 'cast', 'director']]
    final_movies.to_csv(out_movies_path, index=False)
    
    ratings.to_csv(out_ratings_path, index=False)
    
    total_movies = len(final_movies)
    total_ratings = len(ratings)
    unique_users = ratings['userId'].nunique()
    
    # Optional: Convert timestamp safely for calculating range
    try:
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        date_range = f"{ratings['datetime'].min()} to {ratings['datetime'].max()}"
    except:
        date_range = f"{ratings['timestamp'].min()} to {ratings['timestamp'].max()}"
        
    success_rate = (success_count / attempt_count * 100) if attempt_count > 0 else 100.0
    
    print(f"Total Movies: {total_movies}")
    print(f"Total Ratings: {total_ratings}")
    print(f"Unique Users: {unique_users}")
    print(f"Date Range: {date_range}")
    print(f"TMDB Fetch Success Rate: {success_rate:.2f}%")

if __name__ == "__main__":
    collect_data()
