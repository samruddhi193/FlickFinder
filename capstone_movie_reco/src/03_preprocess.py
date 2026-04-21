import pandas as pd
import numpy as np
import os
import yaml
from scipy.sparse import csr_matrix
import pickle

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    if not os.path.exists(config_path):
        return {'dataset_paths': {'raw': '../data/raw', 'processed': '../data/processed'}}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(raw_dir):
    print("Loading raw CSV files...")
    ratings_path = os.path.join(raw_dir, 'raw_ratings.csv')
    movies_path = os.path.join(raw_dir, 'raw_movies.csv')
    metadata_path = os.path.join(raw_dir, 'metadata.csv')
    
    ratings_df = pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    movies_df = pd.read_csv(movies_path) if os.path.exists(movies_path) else pd.DataFrame(columns=['movieId', 'title', 'genres'])
    metadata_df = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
    
    # Verify expectations
    if not ratings_df.empty:
        required_ratings_cols = {'userId', 'movieId', 'rating', 'timestamp'}
        if not required_ratings_cols.issubset(ratings_df.columns):
            raise AssertionError(f"Ratings CSV is missing columns! Expected {required_ratings_cols}")
        print("Verified Ratings columns and dtypes:")
        print(ratings_df.dtypes)
        
    if not movies_df.empty:
        required_movies_cols = {'movieId', 'title', 'genres'}
        if not required_movies_cols.issubset(movies_df.columns):
            raise AssertionError(f"Movies CSV is missing columns! Expected {required_movies_cols}")
            
    print(f"System Loaded: {len(ratings_df)} ratings, {len(movies_df)} movies.")
    return ratings_df, movies_df, metadata_df

def filter_data(ratings, movies):
    print("\nFiltering sparse data structures...")
    initial_users = ratings['userId'].nunique()
    initial_movies = ratings['movieId'].nunique()
    
    # 1. Remove movies with fewer than 5 ratings
    movie_counts = ratings['movieId'].value_counts()
    valid_movies = movie_counts[movie_counts >= 5].index
    ratings = ratings[ratings['movieId'].isin(valid_movies)]
    
    # 2. Remove users with fewer than 20 ratings
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= 20].index
    ratings = ratings[ratings['userId'].isin(valid_users)]
    
    # Sync movies dataset to keep only movies that survived the cut
    final_movies = ratings['movieId'].unique()
    movies = movies[movies['movieId'].isin(final_movies)]
    
    dropped_users = initial_users - ratings['userId'].nunique()
    dropped_movies = initial_movies - len(final_movies)
    
    print(f"Filtering Results:")
    print(f" -> Dropped {dropped_users} low-volume users (< 20 ratings).")
    print(f" -> Dropped {dropped_movies} sparse movies (< 5 ratings).")
    
    return ratings, movies

def clean_data(ratings, movies):
    print("\nExecuting Data Cleansing...")
    
    # Drop exact duplicated rows mathematically
    ratings = ratings.drop_duplicates()
    movies = movies.drop_duplicates()
    
    # Parse the genres string into a list structure via pipe separator
    if 'genres' in movies.columns:
        movies['genres'] = movies['genres'].str.split('|')
        
    # Convert UNIX timestamps to genuine pandas DateTime objects
    if 'timestamp' in ratings.columns:
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
    # Strip whitespace automatically structurally from all object/string columns
    string_columns = movies.select_dtypes(include=['object']).columns
    for col in string_columns:
        # Check if column is a Pandas series of strings (skip lists)
        if type(movies[col].iloc[0]) is str:
            movies[col] = movies[col].str.strip()
            
    # Regex extract the Release Year from the title column (e.g. "Toy Story (1995)")
    if 'title' in movies.columns:
        movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)')
        
    return ratings, movies

def handle_missing(metadata):
    print("\nHandling missing and NaN values...")
    if metadata.empty:
        return metadata
    
    total_len = len(metadata)
    missing_pct = (metadata.isnull().sum() / total_len) * 100
    print("Percentage of missing data per column:")
    print(missing_pct.to_string())
    
    # Fill neutral TMDB defaults
    if 'overview' in metadata.columns:
        metadata['overview'] = metadata['overview'].fillna('')
    if 'cast' in metadata.columns:
        metadata['cast'] = metadata['cast'].fillna('Unknown')
    if 'director' in metadata.columns:
        metadata['director'] = metadata['director'].fillna('Unknown Director')
    if 'poster_url' in metadata.columns:
        metadata['poster_url'] = metadata['poster_url'].fillna('')
        
    return metadata

def split_data(ratings):
    print("\nExecuting 80/20 Time-based Split...")
    # Sort rigorously by exactly when the preference was supplied simulating real future testing
    ratings_sorted = ratings.sort_values(by='timestamp')
    
    split_index = int(len(ratings_sorted) * 0.8)
    
    train = ratings_sorted.iloc[:split_index].copy()
    test = ratings_sorted.iloc[split_index:].copy()
    
    print(f"Data Splitted -> Train dataset: {len(train):,}, Test dataset: {len(test):,}")
    return train, test

def build_matrix(train_df):
    print("\nConstructing Sparse CSR Matrices...")
    
    unique_users = train_df['userId'].unique()
    unique_movies = train_df['movieId'].unique()
    
    # Fast 0-based index lookup mapping
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
    idx_to_movie = {idx: movie for movie, idx in movie_to_idx.items()}
    
    row_idx = train_df['userId'].map(user_to_idx).to_numpy()
    col_idx = train_df['movieId'].map(movie_to_idx).to_numpy()
    vals = train_df['rating'].to_numpy() # the raw scalar ratings
    
    # Instantiate Scientific Sparse Matrix 
    s_matrix = csr_matrix((vals, (row_idx, col_idx)), shape=(len(unique_users), len(unique_movies)))
    
    mappings = {
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'movie_to_idx': movie_to_idx,
        'idx_to_movie': idx_to_movie
    }
    
    print(f"Successfully generated matrix with dimensions: {s_matrix.shape}")
    return s_matrix, mappings

def normalize_ratings(train_df, mappings, shape):
    print("\nExecuting Mean-Centering Standardization...")
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # Mean-centering: value = specific rating - average rating of that user across all their movies
    centered_values = []
    for _, row in train_df.iterrows():
        n_val = row['rating'] - user_means.get(row['userId'], 0)
        centered_values.append(n_val)
        
    row_idx = train_df['userId'].map(mappings['user_to_idx']).to_numpy()
    col_idx = train_df['movieId'].map(mappings['movie_to_idx']).to_numpy()
    
    normalized_matrix = csr_matrix((centered_values, (row_idx, col_idx)), shape=shape)
    
    print("Standardization Complete. User BIAS corrected.")
    return normalized_matrix, user_means

def generate_quality_report(ratings, movies, file_path):
    print("\nGenerating HTML Export Data Quality Report...")
    missing_ratings = ratings.isnull().sum().to_frame("Missing Count").to_html()
    missing_movies = movies.isnull().sum().to_frame("Missing Count").to_html()
    
    html = f"""
    <html>
        <head><title>Data Quality Preprocessing Report</title></head>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h1 style="color: #2E86C1;">Data Quality Report</h1>
            <hr>
            <h2>Summary</h2>
            <ul>
                <li><strong>Total Cleaned Users:</strong> {ratings['userId'].nunique():,}</li>
                <li><strong>Total Cleaned Movies:</strong> {movies['movieId'].nunique():,}</li>
                <li><strong>Total Usable Ratings:</strong> {len(ratings):,}</li>
                <li><strong>System Rating Value Range:</strong> {ratings['rating'].min()} -> {ratings['rating'].max()}</li>
            </ul>
            
            <h2>Missing Value Anomalies (Ratings)</h2>
            {missing_ratings}
            
            <h2>Missing Value Anomalies (Movies)</h2>
            {missing_movies}
            
            <h2>Ratings DTYPE Summary</h2>
            {ratings.dtypes.to_frame("DTYPE").to_html()}
            
            <h2>Movies DTYPE Summary</h2>
            {movies.dtypes.to_frame("DTYPE").to_html()}
        </body>
    </html>
    """
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Interactive Quality Report Published structurally to: {file_path}")

def main():
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Folder pointers
    raw_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('raw', 'data/raw'))
    proc_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('processed', 'data/processed'))
    outputs_dir = os.path.join(base_dir, 'outputs')
    warehouse_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('warehouse', 'data/warehouse'))
    
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(warehouse_dir, exist_ok=True)

    print("=== Commencing Phase 3: Data Preprocessing Pipeline ===")
    ratings, movies, metadata = load_data(raw_dir)
    
    if ratings.empty or movies.empty:
        print("CRITICAL: raw_ratings.csv or raw_movies.csv not found in data/raw. Cannot preprocess.")
        return

    ratings, movies = filter_data(ratings, movies)
    ratings, movies = clean_data(ratings, movies)
    metadata = handle_missing(metadata)

    # Re-assemble strings for safe saving
    movies_export = movies.copy()
    if 'genres' in movies_export.columns:
        movies_export['genres'] = movies_export['genres'].apply(lambda x: "|".join(x) if isinstance(x, list) else x)

    # Deliverables (A): Base clean sets
    ratings.to_csv(os.path.join(proc_dir, 'cleaned_ratings.csv'), index=False)
    movies_export.to_csv(os.path.join(proc_dir, 'processed_movies.csv'), index=False)

    train_df, test_df = split_data(ratings)

    # Deliverables (B): Time-Splits
    train_df.to_csv(os.path.join(proc_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(proc_dir, 'test.csv'), index=False)

    # Matrix math and Normalization mappings
    sparse_matrix, mappings = build_matrix(train_df)
    norm_sparse_matrix, user_means = normalize_ratings(train_df, mappings, sparse_matrix.shape)
    
    # Saving mathematically heavy matrices correctly using binaries instead of strings to avoid RAM death
    with open(os.path.join(warehouse_dir, 'sparse_matrix.pkl'), 'wb') as f:
        pickle.dump(sparse_matrix, f)
    with open(os.path.join(warehouse_dir, 'norm_sparse_matrix.pkl'), 'wb') as f:
        pickle.dump(norm_sparse_matrix, f)
    with open(os.path.join(warehouse_dir, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    with open(os.path.join(warehouse_dir, 'user_means.pkl'), 'wb') as f:
        pickle.dump(user_means, f)
        
    # Deliverables (C): Quality Metric Report HTML
    generate_quality_report(ratings, movies, os.path.join(outputs_dir, 'data_quality_report.html'))

    print("=== Pipeline Succeeded! Data is warehouse standardized. ===")

if __name__ == "__main__":
    main()
