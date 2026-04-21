import pandas as pd
import numpy as np
import os
import yaml
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_processed_data(proc_dir):
    print("Loading Phase 3 finalized datasets...")
    train = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
    movies = pd.read_csv(os.path.join(proc_dir, 'processed_movies.csv'))
    
    metadata_path = os.path.join(proc_dir, 'cleaned_metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
    else:
        # Fallback silently or initialize an empty DF structure if TMDB failed completely
        metadata = pd.DataFrame(columns=['movieId', 'overview', 'cast'])
        
    return train, movies, metadata

def engineer_text_features(movies, metadata):
    print("\n[1/6] Text Matrix: Term Frequency-Inverse Document Frequency (TF-IDF)...")
    
    if not metadata.empty:
        df = pd.merge(movies, metadata[['movieId', 'overview', 'cast']], on='movieId', how='left')
        df['overview'] = df['overview'].fillna('')
        df['cast'] = df['cast'].fillna('').str.replace('|', ' ')
    else:
        df = movies.copy()
        df['overview'] = ''
        df['cast'] = ''
    
    # Process Genres (replace pipes with spacing for standard unigram parsings)
    df['genres_str'] = df['genres'].fillna('').str.replace('|', ' ')
    
    # Concatenate specific features requested into unified target textual matrix
    df['combined_text'] = df['overview'] + " " + df['genres_str'] + " " + df['cast']
    
    # Generate 5,000 width limit for performance matrix
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    
    print(f"Generated NLP TF-IDF representation array shape: {tfidf_matrix.shape}")
    return tfidf_matrix, df

def encode_genres(df):
    print("\n[2/6] Genre Matrix: MultiLabel Binarizer (One-Hot Formatting)...")
    # Cast safe split back out
    df['genre_list'] = df['genres'].fillna('').str.split('|')
    
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genre_list'])
    
    genre_sparse = csr_matrix(genre_matrix)
    print(f"Generated Structural One-Hot dense array: {genre_sparse.shape} representing {len(mlb.classes_)} unique genres.")
    return genre_sparse

def encode_decades(df):
    print("\n[3/6] Decade Encoding: Transforming Year Data into Style Eras...")
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    
    # Example computation (1994 -> 1990.0) -> '1990s' safely avoiding NAT issues
    df['decade_val'] = (np.floor(df['release_year'] / 10) * 10).astype('Int64').astype(str) + 's'
    df['decade_val'] = df['decade_val'].replace('<NA>s', 'Unknown Era')
    
    decade_dummies = pd.get_dummies(df['decade_val'], prefix='decade')
    decade_sparse = csr_matrix(decade_dummies.values)
    
    print(f"Categorized decades into Matrix Array shape: {decade_sparse.shape}")
    return decade_sparse

def compute_user_bias(train):
    print("\n[4/6] Biometrics: Calculating Aggregated User Behaviours...")
    user_bias = train.groupby('userId')['rating'].agg(
        mean_rating='mean',
        std_rating='std',
        rating_count='count'
    ).reset_index()
    
    # Users with exactly 1 rating result in standard dev of NaN. Safety fix.
    user_bias['std_rating'] = user_bias['std_rating'].fillna(0)
    print(f"Computed mathematical biases and variances for {len(user_bias)} users.")
    return user_bias

def build_user_profiles(train, tfidf_matrix, df_movies):
    print("\n[5/6] Embedding Generation: Computing Weighted User Density Features...")
    
    movie_id_to_idx = {m_id: idx for idx, m_id in enumerate(df_movies['movieId'])}
    users = train['userId'].unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    
    rows, cols, vals = [], [], []
    for row in train.itertuples():
        if row.movieId in movie_id_to_idx:
            rows.append(user_to_idx[row.userId])
            cols.append(movie_id_to_idx[row.movieId])
            vals.append(row.rating)
            
    rating_sparse = csr_matrix((vals, (rows, cols)), shape=(len(users), len(df_movies)))
    
    # Weight normalizing the averages structurally (so heavy variance doesn't explode out proportions)
    sums = rating_sparse.sum(axis=1).A1
    sums[sums == 0] = 1.0 # Denom zero error handling structure
    inv_diag = csr_matrix((1.0/sums, (np.arange(len(users)), np.arange(len(users)))))
    
    normalized_ratings = inv_diag.dot(rating_sparse)
    
    # Weighted calculation combining their normalized rating scales DIRECTLY multiplied by TF-IDF properties
    user_profile_matrix = normalized_ratings.dot(tfidf_matrix)
    
    print(f"Successfully processed User-Item context Matrix via dot factorization. Shape: {user_profile_matrix.shape}")
    return user_profile_matrix, user_to_idx

def main():
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    proc_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('processed', 'data/processed'))
    models_dir = os.path.join(base_dir, 'outputs', 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("=== Commencing Phase 4: Content and BIAS Feature Engineering ===")
    try:
        train, movies, metadata = load_processed_data(proc_dir)
    except FileNotFoundError:
        print("CRITICAL: train.csv or processed_movies.csv not located. Please finalize Phase 3 first.")
        return

    # Engineer Features
    tfidf_matrix, df_movies = engineer_text_features(movies, metadata)
    genre_matrix = encode_genres(df_movies)
    decade_matrix = encode_decades(df_movies)
    user_bias_df = compute_user_bias(train)
    user_profiles_matrix, user_mapping_dist = build_user_profiles(train, tfidf_matrix, df_movies)
    
    print("\n[6/6] Final Matrix Stack Concatenations...")
    # Vertically stack / horizontally glue the items features. 
    item_features_matrix = hstack([tfidf_matrix, genre_matrix, decade_matrix])
    print(f"Glued Item Contextual Array Dimensionality: {item_features_matrix.shape}")
    
    # Output saves processing limits
    user_bias_df.to_csv(os.path.join(proc_dir, 'user_bias_features.csv'), index=False)
    
    # Feature matrix reference table matching ID -> Matrix Index Position mathematically
    feature_matrix_df = pd.DataFrame({
        'movieId': df_movies['movieId'].values,
        'feature_index': range(len(df_movies))
    })
    feature_matrix_df.to_csv(os.path.join(proc_dir, 'feature_matrix.csv'), index=False)
    
    # Output Models
    print("\nPersisting Binary Matrices using JOBLIB format...")
    joblib.dump(item_features_matrix, os.path.join(models_dir, 'item_features.pkl'))
    
    user_package = {
        'matrix': user_profiles_matrix,
        'user_lookup_map': user_mapping_dist
    }
    joblib.dump(user_package, os.path.join(models_dir, 'user_profiles.pkl'))
    
    print("=== Pipeline Succeeded! Advanced Content vectors established. ===")

if __name__ == "__main__":
    main()
