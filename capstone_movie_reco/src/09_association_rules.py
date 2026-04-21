import pandas as pd
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    print("Warning: mlxtend not fundamentally installed in environment.")

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(proc_dir):
    print("Loading fundamental structural dependencies...")
    train = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
    movies = pd.read_csv(os.path.join(proc_dir, 'processed_movies.csv'))
    user_bias = pd.read_csv(os.path.join(proc_dir, 'user_bias_features.csv'))
    return train, movies, user_bias

def build_baskets(df, threshold=3.5):
    """Converts the ratings paradigm into a 'Market Basket' structure grouping strictly movies rated positively."""
    liked = df[df['rating'] >= threshold]
    baskets = []
    
    # Accelerate iterations grouping string sequences collectively
    grouped = liked.groupby('userId')['genres'].apply(list)
    
    for g_list in grouped:
        basket = set() # Automatically isolates specific genres mitigating volume duplications
        for g_str in g_list:
            if pd.notna(g_str):
                for genre in g_str.split('|'):
                    genre = genre.strip()
                    if genre and genre != 'Unknown':
                        basket.add(genre)
        baskets.append(list(basket))
        
    return baskets

def mine_rules(baskets, min_support=0.05, min_confidence=0.4):
    if not baskets:
        return pd.DataFrame()
        
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Core mathematical iteration logic execution detecting common sets structurally
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    # Output isolated strictly via the 'Lift' factor mitigating base popularity skewing
    return rules.sort_values(by='lift', ascending=False)

def build_network_graph(rules, output_path):
    print(f"\nConstructing physical NetworkX Visual Vector mapping -> {output_path}")
    if rules.empty:
        print("Empty nodes structure. Plot sequence aborted.")
        return
        
    # Heavily restrict rendering density bypassing visual cluster-storms
    top_rules = rules.head(35)
    
    G = nx.DiGraph()
    
    for _, row in top_rules.iterrows():
        # Clean formatting replacing strict memory structures inside Frozensets
        antecedent = list(row['antecedents'])[0] if isinstance(row['antecedents'], frozenset) else list(row['antecedents'])[0]
        consequent = list(row['consequents'])[0] if isinstance(row['consequents'], frozenset) else list(row['consequents'])[0]
        
        weight = row['lift']
        G.add_edge(antecedent, consequent, weight=weight)
        
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.9, iterations=50) # Expands layout spread structurally
    
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='#82E0AA', edgecolors='black', linewidths=1.5, alpha=0.95)
    nx.draw_networkx_edges(G, pos, width=[max(1, (d['weight']-1)*1.5) for (u,v,d) in G.edges(data=True)], 
                           arrowsize=25, edge_color='#AAB7B8', connectionstyle='arc3, rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif', font_weight='bold')
    
    plt.title("Genre Recommendation Linkages (Market Basket Rules mapped via Ascending Lift)", fontweight='bold', fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
def main():
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    outputs_dir = os.path.join(base_dir, 'outputs')
    fig_dir = os.path.join(outputs_dir, 'figures')
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("🛒 Commencing Phase 9: Apriori Association Rule Extraction")
    print("="*60)
    
    try:
        train, movies, user_bias = load_data(proc_dir)
    except FileNotFoundError as e:
        print(f"CRITICAL DEPENDENCY FAULT: Base components isolated missing -> {e}")
        return
        
    df = pd.merge(train, movies[['movieId', 'genres']], on='movieId', how='left')
    
    config_thresh = config.get('apriori_thresholds', {})
    min_sup = config_thresh.get('min_support', 0.05)
    min_conf = config_thresh.get('min_confidence', 0.4)
    
    print(f"\n1. Global Matrix Mining [Constraints -> Base Support: {min_sup*100}%, Confidence: {min_conf*100}%]")
    baskets_overall = build_baskets(df, threshold=3.5)
    print(f" -> Constructed {len(baskets_overall)} isolated User Baskets structurally.")
    
    rules_overall = mine_rules(baskets_overall, min_support=min_sup, min_confidence=min_conf)
    
    if not rules_overall.empty:
        # Resolve array formats securely targeting string conversions
        rules_overall['antecedents'] = rules_overall['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_overall['consequents'] = rules_overall['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_overall['segment'] = 'Overall System'
    
    print("\n2. Isolating Structural Splits for Event-Type Analysis (Divergent Matrices)...")
    
    high_raters = user_bias[user_bias['mean_rating'] >= 4.0]['userId'].unique()
    low_raters = user_bias[user_bias['mean_rating'] <= 3.0]['userId'].unique()
    
    df_high = df[df['userId'].isin(high_raters)]
    df_low = df[df['userId'].isin(low_raters)]
    
    # Execution Block H
    print(f" -> Segment Alpha: Processing {len(high_raters)} strict High-Raters...")
    baskets_high = build_baskets(df_high, threshold=3.5)
    rules_high = mine_rules(baskets_high, min_support=min_sup, min_confidence=min_conf)
    if not rules_high.empty:
        rules_high['antecedents'] = rules_high['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_high['consequents'] = rules_high['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_high['segment'] = 'High-Rater Profile'
        
    # Execution Block L
    print(f" -> Segment Beta: Processing {len(low_raters)} strict Low-Raters...")
    baskets_low = build_baskets(df_low, threshold=3.5) 
    rules_low = mine_rules(baskets_low, min_support=min_sup, min_confidence=min_conf)
    if not rules_low.empty:
        rules_low['antecedents'] = rules_low['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_low['consequents'] = rules_low['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_low['segment'] = 'Low-Rater Profile'
        
    print("\n3. Synthesizing Dimensional Extractions and Generating Graphics...")
    frames = []
    if not rules_overall.empty: frames.append(rules_overall)
    if not rules_high.empty: frames.append(rules_high)
    if not rules_low.empty: frames.append(rules_low)
    
    if frames:
        final_rules = pd.concat(frames, ignore_index=True)
        # Select explicit variables specifically mapping exact analytic requirements
        final_rules = final_rules[['segment', 'antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']]
        
        final_rules.to_csv(os.path.join(outputs_dir, 'association_rules.csv'), index=False)
        print(f"✅ Final CSV Structuration saved {len(final_rules)} individual math rules securely.")
        
        if not rules_overall.empty:
            print("\nPreview Base Logic Combinations (Top 5 Globally):")
            print(rules_overall[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5).to_string(index=False))
            
            # Map graph explicitly
            build_network_graph(rules_overall, os.path.join(fig_dir, 'rules_network.png'))
    else:
        print("Warning: Algorithm hit zero extractions. Modify the confidence config limiters lower natively.")

if __name__ == '__main__':
    main()
