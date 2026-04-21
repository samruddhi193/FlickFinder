import os
import requests
import pandas as pd
import time
import zipfile
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_movielens(output_dir):
    """Downloads and extracts the MovieLens 25M dataset."""
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = os.path.join(output_dir, "ml-25m.zip")
    
    if not os.path.exists(os.path.join(output_dir, "raw_ratings.csv")):
        print(f"Downloading MovieLens 25M from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                
            print("Renaming and formatting...")
            ml_dir = os.path.join(output_dir, "ml-25m")
            os.rename(os.path.join(ml_dir, "ratings.csv"), os.path.join(output_dir, "raw_ratings.csv"))
            os.rename(os.path.join(ml_dir, "movies.csv"), os.path.join(output_dir, "raw_movies.csv"))
            os.rename(os.path.join(ml_dir, "tags.csv"), os.path.join(output_dir, "raw_tags.csv"))
            os.rename(os.path.join(ml_dir, "links.csv"), os.path.join(output_dir, "raw_links.csv"))
            
            os.remove(zip_path) # cleanup
            print("MovieLens 25M download complete.")
        except Exception as e:
            print(f"Failed to download MovieLens dataset: {e}")
    else:
        print("Real MovieLens dataset already exists in data/raw/. Skipping download...")

def fetch_tmdb_metadata(links_path, output_path, api_key):
    """Fetches TMDB metadata for API enrichment respecting the 0.25s rate limit."""
    if os.path.exists(output_path):
        print(f"Metadata file already exists at {output_path}.")
        return

    links_df = pd.read_csv(links_path)
    print(f"Loaded {len(links_df)} movies from links.csv.")
    print("Initiating TMDB metadata fetch with 0.25s delays...")
    
    metadata_list = []
    
    for idx, row in links_df.iterrows():
        tmdb_id = row['tmdbId']
        movie_id = row['movieId']
        
        if pd.isna(tmdb_id):
            continue
            
        try:
            url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&append_to_response=credits"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Plot Overview
                overview = data.get('overview', '')
                
                # Top 5 Cast
                cast = data.get('credits', {}).get('cast', [])[:5]
                cast_names = "|".join([a['name'] for a in cast])
                
                # Director
                crew = data.get('credits', {}).get('crew', [])
                directors = [c['name'] for c in crew if c['job'] == 'Director']
                director_name = directors[0] if directors else "Unknown"
                
                # Poster URL
                poster_path = data.get('poster_path', '')
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ''
                
                # Original Language
                org_lang = data.get('original_language', '')
                
                metadata_list.append({
                    'movieId': movie_id,
                    'overview': overview,
                    'cast': cast_names,
                    'director': director_name,
                    'poster_url': poster_url,
                    'original_language': org_lang
                })
            else:
                print(f"Skipping tmdbId {int(tmdb_id)} (API returned status {response.status_code})")
                
        except Exception as e:
            print(f"Error for tmdbId {int(tmdb_id)}: {str(e)}")
            
        # Respect TMDB rate limits
        time.sleep(0.25)
        
        if (idx + 1) % 100 == 0:
            print(f"Progress: {idx + 1}/{len(links_df)}")
            
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(output_path, index=False)
    print(f"TMDB Meta-data saved successfully to: {output_path}")

if __name__ == "__main__":
    print("--- Phase 1: Data Collection ---")
    config = load_config()
    api_key = config.get('api_keys', {}).get('tmdb')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('raw', 'data/raw'))
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. Start MovieLens primary source extraction
    # WARNING: This downloads a large ~260MB zip. It is commented out so it doesn't accidentally execute.
    # Uncomment the below line to authorize the download.
    # download_movielens(raw_dir)
    print("Skipping downloading 250MB MovieLens zip file! To enable real download, uncomment `download_movielens(raw_dir)` in src/data_collection.py.")
    
    # 2. Extract secondary enrichment from TMDB API
    links_file = os.path.join(raw_dir, "raw_links.csv")
    metadata_file = os.path.join(raw_dir, "metadata.csv")
    
    if api_key and api_key != "YOUR_TMDB_API_KEY_HERE":
        if os.path.exists(links_file):
            fetch_tmdb_metadata(links_file, metadata_file, api_key)
        else:
            print(f"Error: {links_file} not found. Please run mock_data_generator.py first or enable the MovieLens download.")
    else:
        print("TMDB API key is not configured in config.yaml. Please add it to fetch metadata enrichment.")
