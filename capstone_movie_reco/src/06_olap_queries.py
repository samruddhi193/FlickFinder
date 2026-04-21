import pandas as pd
import sqlite3
import os
import yaml

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_rollup_query(conn):
    print("\n" + "="*50)
    print("--- 1. OLAP Roll-Up ---")
    print("Concept: Average rating grouped by genre and year, rolled up to genre and decade.")
    print("="*50)
    
    query = """
    SELECT 
        m.primary_genre,
        m.decade,
        ROUND(AVG(f.rating), 2) AS avg_rating,
        COUNT(f.fact_id) AS rating_count
    FROM fact_rating f
    JOIN dim_movie m ON f.movie_id = m.movie_id
    WHERE m.primary_genre != 'Unknown' AND m.primary_genre IS NOT NULL
    GROUP BY m.primary_genre, m.decade
    ORDER BY m.primary_genre, m.decade
    """
    df = pd.read_sql_query(query, conn)
    print(df.head(15).to_string(index=False))
    print("... (Showing top 15 rows)")
    return df

def run_drilldown_query(conn):
    print("\n" + "="*50)
    print("--- 2. OLAP Drill-Down ---")
    print("Concept: View overall system avg -> zoom to a specific Genre -> zoom to specific Movies.")
    print("="*50)
    
    # [Level 1]
    overall_query = "SELECT ROUND(AVG(rating), 2) FROM fact_rating"
    overall_avg = pd.read_sql_query(overall_query, conn).iloc[0, 0]
    print(f"[Level 1 - OVERALL] System Average Rating = {overall_avg}\n")
    
    # [Level 2]
    target_genre = 'Action'
    genre_query = f"""
    SELECT ROUND(AVG(f.rating), 2) 
    FROM fact_rating f 
    JOIN dim_movie m ON f.movie_id = m.movie_id 
    WHERE m.primary_genre = '{target_genre}'
    """
    genre_avg = pd.read_sql_query(genre_query, conn).iloc[0, 0]
    print(f"[Level 2 - GENRE DRILL] '{target_genre}' Average Rating = {genre_avg}\n")
    
    # [Level 3]
    movie_query = f"""
    SELECT 
        m.title, 
        ROUND(AVG(f.rating), 2) AS movie_avg, 
        COUNT(f.fact_id) AS total_ratings
    FROM fact_rating f
    JOIN dim_movie m ON f.movie_id = m.movie_id
    WHERE m.primary_genre = '{target_genre}'
    GROUP BY m.title
    HAVING total_ratings >= 50
    ORDER BY movie_avg DESC
    LIMIT 10
    """
    movies_df = pd.read_sql_query(movie_query, conn)
    print(f"[Level 3 - MOVIE DRILL] Top 10 '{target_genre}' Highly Rated Movies:")
    print(movies_df.to_string(index=False))
    return movies_df

def run_slice_query(conn):
    print("\n" + "="*50)
    print("--- 3. OLAP Slice ---")
    print("Concept: Extract subset of Fact Table isolating solely the 'Drama' genre across all time dimensions.")
    print("="*50)
    
    query = """
    SELECT 
        m.title,
        t.year,
        f.rating,
        u.activity_level
    FROM fact_rating f
    JOIN dim_movie m ON f.movie_id = m.movie_id
    JOIN dim_time t ON f.time_id = t.time_id
    JOIN dim_user u ON f.user_id = u.user_id
    WHERE m.primary_genre = 'Drama'
    ORDER BY f.rating DESC, t.year ASC
    LIMIT 15
    """
    df = pd.read_sql_query(query, conn)
    print("Drama Sub-Cube Snapshot (Top 15 Rows):")
    print(df.to_string(index=False))
    return df

def run_dice_query(conn):
    print("\n" + "="*50)
    print("--- 4. OLAP Dice ---")
    print("Concept: Multi-Dimensional Filter -> (Genre=Drama OR Thriller) + (Release Year>2000) + (Rating>4.0). Top 20 Count.")
    print("="*50)
    
    query = """
    SELECT 
        m.title,
        m.primary_genre,
        m.release_year,
        COUNT(f.fact_id) AS popularity_count
    FROM fact_rating f
    JOIN dim_movie m ON f.movie_id = m.movie_id
    WHERE (m.primary_genre IN ('Drama', 'Thriller'))
      AND m.release_year > 2000
      AND f.rating > 4.0
    GROUP BY m.title, m.primary_genre, m.release_year
    ORDER BY popularity_count DESC
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    return df

def run_pivot_query(conn, df_rollup):
    print("\n" + "="*50)
    print("--- 5. OLAP Pivot Table ---")
    print("Concept: Flipping Axis. Genres mapped as Rows vs Decades mapped as Columns (Values = Avg Rating).")
    print("="*50)
    
    if df_rollup.empty:
        print("Rollup dataset was empty, cannot compute pivot axis.")
        return pd.DataFrame()
        
    pivot_df = df_rollup.pivot(index='primary_genre', columns='decade', values='avg_rating')
    pivot_df.fillna('-', inplace=True)
    
    # Print preview
    print(pivot_df.head(15).to_string())
    return pivot_df

def main():
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    warehouse_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('warehouse', 'data/warehouse'))
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(outputs_dir, exist_ok=True)
    
    db_path = os.path.join(warehouse_dir, 'movie_warehouse.db')
    
    if not os.path.exists(db_path):
        print(f"CRITICAL ERROR: Data warehouse not located at --> {db_path}")
        print("Ensure you execute your Phase 5 Database Engine (05_build_warehouse.py) correctly before running OLAP queries.")
        return
        
    print(f"Accessing OLAP Data Warehouse Storage Runtime via read_sql -> {db_path} ...")
    conn = sqlite3.connect(db_path)
    
    try:
        df_rollup = run_rollup_query(conn)
        df_drilldown = run_drilldown_query(conn)
        df_slice = run_slice_query(conn)
        df_dice = run_dice_query(conn)
        df_pivot = run_pivot_query(conn, df_rollup)
        
        # Save exact requested outputs
        df_rollup.to_csv(os.path.join(outputs_dir, 'olap_results_rollup.csv'), index=False)
        df_drilldown.to_csv(os.path.join(outputs_dir, 'olap_results_drilldown.csv'), index=False)
        df_slice.to_csv(os.path.join(outputs_dir, 'olap_results_slice.csv'), index=False)
        df_dice.to_csv(os.path.join(outputs_dir, 'olap_results_dice.csv'), index=False)
        df_pivot.to_csv(os.path.join(outputs_dir, 'olap_results_pivot.csv'), index=True)
        
        print(f"\n✅ SUCCESS: Mathematical structures saved to CSV binaries inside the {outputs_dir} directory.")
        
    except Exception as e:
        print(f"Error encountered during OLAP SQL Queries: {e}")
    finally:
        conn.close()
        
if __name__ == '__main__':
    main()
