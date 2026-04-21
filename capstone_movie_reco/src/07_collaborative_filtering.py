import pandas as pd
import numpy as np
import os
import joblib
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Try importing Surprise explicitly, catch for installation mapping instructions later
try:
    from surprise import Reader, Dataset, SVD
    from surprise.model_selection import GridSearchCV
except ImportError:
    print("Warning: 'scikit-surprise' package not found. Ensure you executed the requirements.txt installations.")

import warnings
warnings.filterwarnings('ignore')

def load_data(proc_dir, warehouse_dir):
    print("Loading normalized datasets and scientific matrices...")
    train_df = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(proc_dir, 'test.csv'))
    
    with open(os.path.join(warehouse_dir, 'norm_sparse_matrix.pkl'), 'rb') as f:
        norm_sparse_matrix = pickle.load(f)
    with open(os.path.join(warehouse_dir, 'mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
    with open(os.path.join(warehouse_dir, 'user_means.pkl'), 'rb') as f:
        user_means = pickle.load(f)
        
    return train_df, test_df, norm_sparse_matrix, mappings, user_means

def generate_ubcf_recs(test_users, norm_matrix, mappings, user_means, k=20, top_n=10):
    print(f"\nGenerative Method A: User-Based Collaborative Filtering (KNN, K={k})...")
    
    # Notice: User-based Cosine calculates pairwise similarity between N_users.
    # Safe structure if user base < 50,000. 
    user_sim = cosine_similarity(norm_matrix)
    np.fill_diagonal(user_sim, 0)
    
    idx_to_movie = mappings['idx_to_movie']
    user_to_idx = mappings['user_to_idx']
    
    recs = []
    target_users = [u for u in test_users if u in user_to_idx]
    
    print(f"Applying memory-based distances across {len(target_users)} test users...")
    
    completed = 0
    total = len(target_users)
    
    for u_id in target_users:
        u_idx = user_to_idx[u_id]
        sim_scores = user_sim[u_idx]
        
        # Get top K mathematically identical neighbor profiles
        top_k_users_idx = np.argsort(sim_scores)[-k:]
        top_k_sims = sim_scores[top_k_users_idx]
        
        sim_sum = np.sum(np.abs(top_k_sims))
        if sim_sum == 0:
            continue
            
        # Extract ratings of exact similarity intersections
        neighbor_ratings = norm_matrix[top_k_users_idx].toarray()
        
        # Weighted average based strictly against density scores
        weighted_ratings = np.dot(top_k_sims, neighbor_ratings) / sim_sum
        
        # Invert the original Mean-Centering baseline
        predicted_ratings = weighted_ratings + user_means.get(u_id, 0)
        
        # Filter exclusions algorithm -> User cannot be recommended seen items
        user_rated_indices = norm_matrix[u_idx].nonzero()[1]
        predicted_ratings[user_rated_indices] = -np.inf 
        
        best_movie_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
        
        rank = 1
        for m_idx in best_movie_indices:
            pred_score = predicted_ratings[m_idx]
            if pred_score > -np.inf:
                recs.append({
                    'userId': u_id,
                    'rank': rank,
                    'movieId': idx_to_movie[m_idx],
                    'predicted_rating': min(5.0, max(0.5, pred_score)),
                    'method': 'UBCF'
                })
                rank += 1
                
        completed += 1
        if completed % 500 == 0:
            print(f" -> UBCF Progress: {completed}/{total} arrays resolved...")
            
    return pd.DataFrame(recs), user_sim

def generate_ibcf_recs(test_users, norm_matrix, mappings, user_means, top_n=10):
    print(f"\nGenerative Method B: Item-Based Collaborative Filtering...")
    
    # Notice: IBCF runs a transposal (Item * Item mapping)
    item_sim = cosine_similarity(norm_matrix.T)
    np.fill_diagonal(item_sim, 0)
    
    idx_to_movie = mappings['idx_to_movie']
    user_to_idx = mappings['user_to_idx']
    
    recs = []
    target_users = [u for u in test_users if u in user_to_idx]
    
    print(f"Applying transitive item logic calculations across {len(target_users)} test users...")
    
    completed = 0
    total = len(target_users)
    
    for u_id in target_users:
        u_idx = user_to_idx[u_id]
        
        # Collapse the User vector structure into horizontal mapping
        user_ratings_vector = norm_matrix[u_idx].toarray().flatten()
        
        # Compute specific item similarity vector multipliers against User logic map
        weighted_sum = item_sim.dot(user_ratings_vector)
        
        abs_item_sim = np.abs(item_sim)
        sim_sum = abs_item_sim.dot(np.abs(user_ratings_vector) > 0)
        
        # Bypass division faults natively using Numpy
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_norm_ratings = np.where(sim_sum != 0, weighted_sum / sim_sum, 0)
            
        pred_abs_ratings = pred_norm_ratings + user_means.get(u_id, 0)
        
        # Exclude memory vectors user is fully aware of
        user_rated_indices = user_ratings_vector.nonzero()[0]
        pred_abs_ratings[user_rated_indices] = -np.inf
        
        best_movie_indices = np.argsort(pred_abs_ratings)[-top_n:][::-1]
        
        rank = 1
        for m_idx in best_movie_indices:
            pred_score = pred_abs_ratings[m_idx]
            if pred_score > -np.inf:
                recs.append({
                    'userId': u_id,
                    'rank': rank,
                    'movieId': idx_to_movie[m_idx],
                    'predicted_rating': min(5.0, max(0.5, pred_score)),
                    'method': 'IBCF'
                })
                rank += 1
                
        completed += 1
        if completed % 500 == 0:
            print(f" -> IBCF Progress: {completed}/{total} items resolved...")
            
    return pd.DataFrame(recs), item_sim

def train_svd_model(train_df, test_df, top_n=10):
    print("\nGenerative Method C: Predictive Factorization Matrix (Surprise SVD)...")
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    
    param_grid = {
        'n_factors': [20, 50, 100, 200],
        'n_epochs': [20, 30]
    }
    
    print("Initiating 5-Fold Cross Validation via GridSearchCV targeting (RMSE)...")
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=-1, joblib_verbose=2)
    gs.fit(data)
    
    best_params = gs.best_params['rmse']
    print(f"GridSearchCV Selected Optimum Configuration: n_factors={best_params['n_factors']}, n_epochs={best_params['n_epochs']}")
    print(f"Discovered Cross-Validated RMSE Error Limit: {gs.best_score['rmse']:.4f}")
    
    # Establish finalized algorithm object using optimized configs
    best_algo = gs.best_estimator['rmse']
    
    print("Applying Fit over Full Dimension Data Array...")
    trainset = data.build_full_trainset()
    best_algo.fit(trainset)
    
    print("Generating SVD Top-10 Recommendations globally into the Test Vectors...")
    all_movies = train_df['movieId'].unique()
    target_users = test_df['userId'].unique()
    
    # Rapid lookup table structuring
    user_rated_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    
    recs = []
    completed = 0
    total = len(target_users)
    
    for u_id in target_users:
        seen = user_rated_items.get(u_id, set())
        
        user_preds = []
        for m_id in all_movies:
            if m_id not in seen:
                # Structural object property injection fetching predicting float
                pred = best_algo.predict(uid=u_id, iid=m_id).est
                user_preds.append((m_id, pred))
        
        # Sort predictions by highest calculated ranking logic structure
        user_preds.sort(key=lambda x: x[1], reverse=True)
        top_k = user_preds[:top_n]
        
        rank = 1
        for m_id, pred in top_k:
            recs.append({
                'userId': u_id,
                'rank': rank,
                'movieId': m_id,
                'predicted_rating': pred,
                'method': 'SVD'
            })
            rank += 1
            
        completed += 1
        if completed % 500 == 0:
            print(f" -> SVD Progress: {completed}/{total} factors resolved...")
            
    return pd.DataFrame(recs), best_algo

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    warehouse_dir = os.path.join(base_dir, 'data', 'warehouse')
    models_dir = os.path.join(base_dir, 'outputs', 'models')
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 Initiating Phase 7: Heavy Collaborative Filtering Inference")
    print("="*60)
    
    try:
        train_df, test_df, norm_sparse_matrix, mappings, user_means = load_data(proc_dir, warehouse_dir)
    except FileNotFoundError as e:
        print(f"CRITICAL FAULT: Missing dependency architecture -> {e}")
        return
        
    test_users = test_df['userId'].unique()
    recs_pool = []
    
    # 1. UBCF Memory Calculation
    try:
        ubcf_recs, ubcf_model = generate_ubcf_recs(test_users, norm_sparse_matrix, mappings, user_means, k=20)
        joblib.dump(ubcf_model, os.path.join(models_dir, 'ubcf_model.pkl'))
        recs_pool.append(ubcf_recs)
        print("UBCF Memory Snapshot Dumped successfully.")
    except Exception as e:
        print(f"UBCF Execution Blocked (RAM limitation mostly on 100k+ sparse datasets): {e}")

    # 2. IBCF Transposal Memory Calculation
    try:
        ibcf_recs, ibcf_model = generate_ibcf_recs(test_users, norm_sparse_matrix, mappings, user_means)
        joblib.dump(ibcf_model, os.path.join(models_dir, 'ibcf_model.pkl'))
        recs_pool.append(ibcf_recs)
        print("IBCF Transposal Snapshot Dumped successfully.")
    except Exception as e:
        print(f"IBCF Execution Blocked (Item RAM limitation on 50k+ items dataset): {e}")
        
    # 3. Complex Cross-Validation Mathematical Factorizations (SVD)
    try:
        svd_recs, svd_model = train_svd_model(train_df, test_df)
        joblib.dump(svd_model, os.path.join(models_dir, 'svd_model.pkl'))
        recs_pool.append(svd_recs)
        print("SVD Matrix Model Dumped successfully via JOBLIB.")
    except Exception as e:
        print(f"SVD Logic Crash Sequence Fault: {e}")

    # 4. Payload Consolidation Export
    if recs_pool:
        print("\nExporting the master recommendation combinations DataFrame...")
        all_recs_df = pd.concat(recs_pool, ignore_index=True)
        recs_path = os.path.join(outputs_dir, 'recs_cf.csv')
        all_recs_df.to_csv(recs_path, index=False)
        print(f"✅ Success! Total Recommendations successfully pooled: {len(all_recs_df)} matrix structures into {recs_path}")
    else:
        print("Failure to synthesize matrix factors entirely.")

if __name__ == '__main__':
    main()
