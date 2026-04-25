"""
Module for Content Feature Engineering using TF-IDF and user profiles.
"""
import os
import yaml
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_features():
    """Build item features and user profiles based on TF-IDF representation."""
    config = load_config()
    proc_dir = config.get('paths', {}).get('processed_data', 'data/processed/')
    models_dir = config.get('paths', {}).get('models', 'outputs/models/')
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    proc_dir = os.path.join(base_dir, proc_dir)
    models_dir = os.path.join(base_dir, models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    movies_path = os.path.join(proc_dir, 'processed_movies.csv')
    train_path = os.path.join(proc_dir, 'train.csv')
    
    print("Loading processed data...")
    movies = pd.read_csv(movies_path)
    train = pd.read_csv(train_path)
    
    # Fill NA for text fields
    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('[]')
    movies['cast'] = movies['cast'].fillna('')
    
    # Safe eval for python lists saved as strings, if necessary
    import ast
    def safe_parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return []
        return []
        
    movies['genres_list'] = movies['genres'].apply(safe_parse_list)
    
    # Replace "|" spaces in cast if generated that way
    movies['cast_space'] = movies['cast'].str.replace('|', ' ')
    
    print("Building combined text field...")
    # concatenated: overview + " " + genres joined by space + " " + cast joined by space
    movies['genres_space'] = movies['genres_list'].apply(lambda x: " ".join(x))
    movies['combined_text'] = movies['overview'] + " " + movies['genres_space'] + " " + movies['cast_space']
    
    print("Applying TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(movies['combined_text'])
    
    print("One-hot encoding genres...")
    mlb = MultiLabelBinarizer()
    genre_binary = mlb.fit_transform(movies['genres_list'])
    genre_sparse = sp.csr_matrix(genre_binary)
    
    print("Concatenating features...")
    item_features = sp.hstack([tfidf_matrix, genre_sparse], format='csr')
    
    # Build a movieId to row index mapping
    movie_to_idx = {movie_id: i for i, movie_id in enumerate(movies['movieId'])}
    
    print("Computing user profiles...")
    # Weighted average of TF-IDF vectors (item_features) using rating value
    user_profiles_list = []
    user_lookup_map = {}
    
    for i, (user_id, group) in enumerate(train.groupby('userId')):
        indices = [movie_to_idx[m_id] for m_id in group['movieId'] if m_id in movie_to_idx]
        weights = group['rating'].values[:len(indices)] # align
        
        if len(indices) == 0:
            continue
            
        user_movies_matrix = item_features[indices]
        
        weight_diag = sp.diags(weights)
        weighted_features = weight_diag.dot(user_movies_matrix)
        
        profile_vector = np.asarray(weighted_features.mean(axis=0)).flatten()
        user_profiles_list.append(profile_vector)
        user_lookup_map[user_id] = i
        
    user_profiles = {
        'matrix': np.array(user_profiles_list),
        'user_lookup_map': user_lookup_map
    }
        
    print("Saving outputs...")
    with open(os.path.join(models_dir, 'item_features.pkl'), 'wb') as f:
        pickle.dump(item_features, f)
        
    with open(os.path.join(models_dir, 'user_profiles.pkl'), 'wb') as f:
        pickle.dump(user_profiles, f)
        
    # Save a dense sample (first 1000 rows)
    dense_sample = pd.DataFrame(item_features[:1000].toarray())
    dense_sample.to_csv(os.path.join(proc_dir, 'feature_matrix.csv'), index=False)
    
    sparsity = 1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    
    print(f"Feature matrix shape (Movies x Features): {item_features.shape}")
    print(f"TF-IDF matrix sparsity: {sparsity * 100:.2f}%")
    print(f"User profiles built: {len(user_profiles)}")

if __name__ == "__main__":
    build_features()
