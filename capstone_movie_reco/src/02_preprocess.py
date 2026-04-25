"""
Module for preprocessing data, filtering, feature engineering (year/decade),
time-based train/test splitting, and matrix creation.
"""
import os
import json
import yaml
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data():
    """Preprocess movie and ratings data and create splits."""
    config = load_config()
    raw_dir = config.get('paths', {}).get('raw_data', 'data/raw/')
    proc_dir = config.get('paths', {}).get('processed_data', 'data/processed/')
    outputs_dir = 'outputs/'
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, raw_dir)
    proc_dir = os.path.join(base_dir, proc_dir)
    outputs_dir = os.path.join(base_dir, outputs_dir)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    min_user_ratings = config.get('filtering', {}).get('min_user_ratings', 20)
    min_movie_ratings = config.get('filtering', {}).get('min_movie_ratings', 5)
    
    ratings_path = os.path.join(raw_dir, 'raw_ratings.csv')
    movies_path = os.path.join(raw_dir, 'raw_movies.csv')
    
    print("Loading data...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Clean commas in potential numeric columns
    for col in ratings.columns:
        if ratings[col].dtype == 'object' and ratings[col].str.contains(',').any():
            ratings[col] = ratings[col].str.replace(',', '').astype(float)
            
    for col in movies.columns:
        if movies[col].dtype == 'object' and movies[col].str.contains(',').any() and col != 'genres' and col != 'overview' and col != 'cast' and col != 'director' and col != 'title':
            # safely convert numeric-looking columns like 'streams'
            try:
                movies[col] = movies[col].str.replace(',', '').astype(float)
            except:
                pass

    print("Filtering ratings...")
    # Filter users
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings['userId'].isin(valid_users)]
    
    # Filter movies
    movie_counts = ratings['movieId'].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings = ratings[ratings['movieId'].isin(valid_movies)]
    
    # Make sure to filter movies df as well
    movies = movies[movies['movieId'].isin(ratings['movieId'])]
    
    print("Preprocessing movies...")
    # Parse pipe-separated genres
    movies['genres'] = movies['genres'].str.split('|')
    
    # Extract release_year from title
    movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['release_year'] = pd.to_numeric(movies['release_year'], errors='coerce')
    
    # Derive decade
    def get_decade(year):
        if pd.isna(year):
            return "Unknown"
        return f"{int(year // 10 * 10)}s"
        
    movies['decade'] = movies['release_year'].apply(get_decade)
    
    print("Processing timestamps and splitting data...")
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings.sort_values(by=['userId', 'datetime'])
    
    # 80/20 Time-based split per user
    def time_based_split(group):
        split_idx = int(len(group) * 0.8)
        train_idx = group.index[:split_idx]
        test_idx = group.index[split_idx:]
        return train_idx, test_idx
        
    train_indices = []
    test_indices = []
    
    # Fast vectorized split
    ratings['rank'] = ratings.groupby('userId')['datetime'].rank(method='first')
    ratings['count'] = ratings.groupby('userId')['userId'].transform('count')
    ratings['split'] = ratings['rank'] <= (ratings['count'] * 0.8)
    
    train = ratings[ratings['split']].copy()
    test = ratings[~ratings['split']].copy()
    
    train = train.drop(columns=['rank', 'count', 'split'])
    test = test.drop(columns=['rank', 'count', 'split'])
    ratings = ratings.drop(columns=['rank', 'count', 'split'])
    
    # Mean-centering ratings per user
    user_means = train.groupby('userId')['rating'].mean()
    train['user_mean_rating'] = train['userId'].map(user_means)
    train['normalized_rating'] = train['rating'] - train['user_mean_rating']
    
    # Map to test as well
    test['user_mean_rating'] = test['userId'].map(user_means)
    test['normalized_rating'] = test['rating'] - test['user_mean_rating']
    
    print("Building CSR matrix...")
    # Build scipy.sparse.csr_matrix from training data
    unique_users = train['userId'].unique()
    unique_movies = train['movieId'].unique()
    
    user_to_idx = {int(user): i for i, user in enumerate(unique_users)}
    movie_to_idx = {int(movie): i for i, movie in enumerate(unique_movies)}
    
    row_idx = train['userId'].map(user_to_idx).values
    col_idx = train['movieId'].map(movie_to_idx).values
    data = train['normalized_rating'].values
    
    user_item_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(len(unique_users), len(unique_movies)))
    
    # Save files
    with open(os.path.join(proc_dir, 'user_to_idx.json'), 'w') as f:
        json.dump(user_to_idx, f)
    with open(os.path.join(proc_dir, 'movie_to_idx.json'), 'w') as f:
        json.dump(movie_to_idx, f)
        
    ratings.to_csv(os.path.join(proc_dir, 'cleaned_ratings.csv'), index=False)
    train.to_csv(os.path.join(proc_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(proc_dir, 'test.csv'), index=False)
    movies.to_csv(os.path.join(proc_dir, 'processed_movies.csv'), index=False)
    
    # Generate simple HTML data quality report
    missing_summary = movies.isnull().sum().to_frame(name='Missing Values').join(
                      ratings.isnull().sum().to_frame(name='Missing Values'), lsuffix='_movies', rsuffix='_ratings', how='outer')
                      
    with open(os.path.join(outputs_dir, 'data_quality_report.html'), 'w') as f:
        f.write("<html><body><h1>Data Quality Report</h1>")
        f.write("<h2>Missing Values</h2>")
        f.write(missing_summary.to_html())
        f.write("</body></html>")
        
    sparsity = 1.0 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
    
    print(f"Rows after filtering (ratings): {len(ratings)}")
    print(f"Sparsity of user-item matrix: {sparsity * 100:.2f}%")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print("Missing value summary:")
    print(missing_summary)

if __name__ == "__main__":
    preprocess_data()
