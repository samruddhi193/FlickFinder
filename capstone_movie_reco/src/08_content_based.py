import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def compute_sim_matrix_memmap(item_features, output_path):
    n_movies = item_features.shape[0]
    print(f"Initializing Memory-Mapped (Memmap) Storage matrix natively dimensioned ({n_movies}x{n_movies}) at {output_path}...")
    
    # Natively maps massive memory chunks directly structurally against physical SSD logic bypassing RAM ceilings instantaneously
    mmap = np.memmap(output_path, dtype='float32', mode='w+', shape=(n_movies, n_movies))
    
    # Operates across chunks mitigating calculation bottlenecks exclusively
    chunk_size = 2000
    for start in range(0, n_movies, chunk_size):
        end = min(start + chunk_size, n_movies)
        chunk = item_features[start:end]
        
        # Explicit Dense computation
        sim_chunk = cosine_similarity(chunk, item_features).astype('float32') # Compresses to structural float32 for storage constraints
        mmap[start:end] = sim_chunk
        
        # Flush streams directly down to target
        mmap.flush()
        print(f" -> Computed structural distances mathematically: rows {start} to {end}...")
        
    print("Memmapped Cosine Structural calculation fully finalized mapped onto disk.")
    return mmap

def generate_profile_recs(test_users, user_data, item_features, train_df, movies_df, top_n=10):
    print("\nExecuting Machine Filter over User Profiles via TFIDF matrices...")
    user_profiles_matrix = user_data['matrix']
    user_to_idx = user_data['user_lookup_map']
    
    # Establish Seen parameters mapping for exclusion structurally
    user_seen = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    
    # Construct mappings instantly matching sequential index directly to canonical ID
    idx_to_movie = {i: m for i, m in enumerate(movies_df['movieId'])}
    
    valid_users = [u for u in test_users if u in user_to_idx]
    
    recs = []
    completed = 0
    total = len(valid_users)
    
    print(f"Injecting mapping profiles against user similarity distances ({total} users)...")
    for u_id in valid_users:
        u_idx = user_to_idx[u_id]
        
        profile_vector = user_profiles_matrix[u_idx]
        
        # Conduct natively embedded Cosine Factorizations utilizing user dimensional vector directly!
        sim_scores = cosine_similarity(profile_vector.reshape(1, -1), item_features)[0]
        
        seen_movies = user_seen.get(u_id, set())
        
        best_indices = np.argsort(sim_scores)[-top_n*5:][::-1]
        
        rank = 1
        for m_idx in best_indices:
            m_id = idx_to_movie[m_idx]
            if m_id not in seen_movies:
                # Normalizing math properties structural scaling
                recs.append({
                    'userId': u_id,
                    'rank': rank,
                    'movieId': m_id,
                    'predicted_rating': min(max(sim_scores[m_idx] * 5.0, 0.5), 5.0), # Content systems usually output 0-1 sim, so we proxy normalize mapping up to 5 strictly here for structural comparisons
                    'method': 'Content-Based'
                })
                rank += 1
                if rank > top_n:
                    break
                    
        completed += 1
        if completed % 500 == 0:
            print(f" -> Batch Factorization Complete: {completed}/{total} mapped...")
            
    print("User Content Mapping natively compiled.")
    return pd.DataFrame(recs)

def get_similar_movies(movie_title, movies_df, memmap_matrix, feature_mappings, n=10):
    match = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
    if match.empty:
        return f"Warning: Title lookup '{movie_title}' unresolvable inside database dictionary."
        
    m_idx = movies_df[movies_df['movieId'] == m_id].index[0]
    
    # Read the directly streamed memmap subset instantly natively
    sim_scores = memmap_matrix[m_idx]
    top_indices = np.argsort(sim_scores)[-(n+1):][::-1] 
    
    idx_to_movie = {i: m for i, m in enumerate(movies_df['movieId'])}
    
    results = []
    for idx in top_indices:
        if idx == m_idx:
            continue # Omit mirror evaluations mapping to itself
        similar_m_id = idx_to_movie[idx]
        s_title = movies_df[movies_df['movieId'] == similar_m_id]['title'].iloc[0]
        score = sim_scores[idx]
        results.append({"title": s_title, "similarity": round(score, 3)})
        if len(results) >= n:
            break
            
    return pd.DataFrame(results)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'outputs', 'models')
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("⚙️ Executing Phase 8: Deep TF-IDF Content-Based Structuring")
    print("="*60)
    
    try:
        item_features = joblib.load(os.path.join(models_dir, 'item_features.pkl'))
        user_profiles_data = joblib.load(os.path.join(models_dir, 'user_profiles.pkl'))
        
        train_df = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(proc_dir, 'test.csv'))
        movies_df = pd.read_csv(os.path.join(proc_dir, 'processed_movies.csv'))
        feature_matrix = pd.read_csv(os.path.join(proc_dir, 'feature_matrix.csv'))
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Phase 4 features or metadata dependencies mathematically unresolvable. -> {e}")
        return
        
    print(f"1. Spawning Dynamic Solid State Memory Sequence (Memmap) for Cosine Calculations...")
    memmap_path = os.path.join(models_dir, 'similarity_matrix.npy')
    
    try:
        sim_memmap = compute_sim_matrix_memmap(item_features, memmap_path)
    except Exception as e:
        print(f"Memmapping execution failed critically due to architectural bounds: {e}")
        return

    print("\n2. Isolating Predictions natively across User Vector Mapping Array...")
    test_users = test_df['userId'].unique()
    recs_df = generate_profile_recs(test_users, user_profiles_data, item_features, train_df, movies_df, top_n=10)
    
    recs_path = os.path.join(outputs_dir, 'recs_content.csv')
    recs_df.to_csv(recs_path, index=False)
    
    print("\n3. Testing Direct Dictionary Item-to-Item Probing (Target: 'Batman')...")
    test_view = get_similar_movies('Batman', movies_df, sim_memmap, feature_matrix, n=5)
    print(test_view)
    
    print("\n" + "-"*60)
    print("[COLD-START SYSTEM ADVANTAGE DOCUMENTS:]")
    document = ("Architectural Note: This specific Phase 8 Logic Model explicitly bypasses standard "
                "Collaborative bottlenecks protecting performance significantly against isolated structural 'New User' instances. "
                "Because Content-Based engines algorithmically only require a human User to natively rate ONE single movie positively "
                "to execute the internal profile compilation, predictions are instantiated automatically utilizing internal object overlaps "
                "over structural NLP TF-IDF comparisons natively circumventing huge collaborative requirements!")
    print(document)
    print("-"*60 + "\n")

    print("Finalizing Binary Object Dump sequences...")
    content_logic = {
        'feature_matrix_route': 'data/processed/feature_matrix.csv',
        'dependency': 'similarity_matrix.npy'
    }
    joblib.dump(content_logic, os.path.join(models_dir, 'content_model.pkl'))
    
    print(f"\n✅ Content Factorizations and SSD memmaps deployed perfectly to Outputs!")

if __name__ == '__main__':
    main()
