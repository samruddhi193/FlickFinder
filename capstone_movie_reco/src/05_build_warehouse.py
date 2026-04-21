import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Date, Index
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

# --- Dimensional Models Target Schemas ---

class DimUser(Base):
    __tablename__ = 'dim_user'
    user_id = Column(Integer, primary_key=True)
    user_index = Column(Integer, nullable=True) # Used computationally inside sparse mappings
    rating_count = Column(Integer)
    avg_rating = Column(Float)
    activity_level = Column(String(50)) # Low, Medium, High

class DimMovie(Base):
    __tablename__ = 'dim_movie'
    movie_id = Column(Integer, primary_key=True)
    tmdb_id = Column(Integer, nullable=True)
    title = Column(String(255), index=True)
    release_year = Column(Integer, nullable=True)
    decade = Column(String(10), nullable=True)
    primary_genre = Column(String(100), nullable=True)
    all_genres = Column(String(500), nullable=True)
    overview_length = Column(Integer, nullable=True)

class DimGenre(Base):
    __tablename__ = 'dim_genre'
    genre_id = Column(Integer, primary_key=True)  # Populated sequentially
    genre_name = Column(String(100), unique=True, index=True)

class DimTime(Base):
    __tablename__ = 'dim_time'
    time_id = Column(Integer, primary_key=True)
    full_date = Column(Date)
    year = Column(Integer)
    month = Column(Integer)
    quarter = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)

class FactRating(Base):
    __tablename__ = 'fact_rating'
    fact_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('dim_user.user_id'), index=True)
    movie_id = Column(Integer, ForeignKey('dim_movie.movie_id'), index=True)
    time_id = Column(Integer, ForeignKey('dim_time.time_id'), index=True)
    rating = Column(Float)
    normalized_rating = Column(Float, nullable=True)
    is_train = Column(Boolean)

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- ETL Logic Mappings ---

def generate_time_dimension(ratings_df):
    """Parses all distinct timestamps globally into analytical time blocks."""
    time_df = pd.DataFrame()
    time_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s', errors='coerce').dt.normalize()
    unique_dates = time_df['datetime'].dropna().unique()
    
    dim_time_records = []
    date_to_id = {}
    time_id_counter = 1
    
    for d in sorted(unique_dates):
        py_date = pd.Timestamp(d).date()
        date_to_id[py_date] = time_id_counter
        
        dim_time_records.append(DimTime(
            time_id=time_id_counter,
            full_date=py_date,
            year=py_date.year,
            month=py_date.month,
            quarter=(py_date.month - 1) // 3 + 1,
            day_of_week=py_date.weekday(), # 0 is Monday, 6 is Sunday
            is_weekend=py_date.weekday() >= 5
        ))
        time_id_counter += 1
    return dim_time_records, date_to_id

def populate_dim_user(session, user_bias_df):
    print("Upserting dim_user...")
    def determine_activity(count):
        if count < 50: return 'Low'
        if count < 500: return 'Medium'
        return 'High'
        
    for _, row in user_bias_df.iterrows():
        user = DimUser(
            user_id=int(row['userId']),
            rating_count=int(row['rating_count']),
            avg_rating=float(row['mean_rating']),
            activity_level=determine_activity(row['rating_count'])
        )
        session.merge(user) # Upserts flawlessly without duplication failures
    session.commit()

def populate_dim_genre(session, unique_genres):
    print("Upserting dim_genre...")
    for idx, g_name in enumerate(unique_genres, start=1):
        genre = DimGenre(
            genre_id=idx,
            genre_name=g_name
        )
        session.merge(genre)
    session.commit()

def populate_dim_movie(session, movies_df, metadata_df):
    print("Upserting dim_movie...")
    movies_meta = pd.merge(movies_df, metadata_df[['movieId', 'tmdbId', 'overview']], on='movieId', how='left') if not metadata_df.empty else movies_df.copy()
    
    for _, row in movies_meta.iterrows():
        genres_list = str(row.get('genres', '')).split('|')
        primary = genres_list[0] if genres_list and genres_list[0] else 'Unknown'
        
        overview_text = str(row.get('overview', ''))
        o_len = len(overview_text) if overview_text != 'nan' else 0
        
        release_yr = row.get('release_year', None)
        yr_val = int(release_yr) if pd.notna(release_yr) else None
        decade_val = f"{(yr_val // 10) * 10}s" if yr_val else 'Unknown'

        tmdb_field = row.get('tmdbId')

        movie = DimMovie(
            movie_id=int(row['movieId']),
            tmdb_id=int(tmdb_field) if pd.notna(tmdb_field) else None,
            title=str(row['title']),
            release_year=yr_val,
            decade=decade_val,
            primary_genre=primary,
            all_genres=str(row.get('genres', '')),
            overview_length=o_len
        )
        session.merge(movie)
    session.commit()

def populate_dim_time(session, time_records):
    print("Upserting dim_time...")
    for record in time_records:
        session.merge(record)
    session.commit()

def populate_fact_rating(session, all_ratings, date_to_id, user_means_dict):
    print("Connecting constraints and migrating fact_rating (Batches of 5000)...")
    batch_size = 5000
    batch = []
    inserted_count = 0
    
    # Pre-caching conversions
    all_ratings['mapped_time'] = pd.to_datetime(all_ratings['timestamp'], unit='s').dt.date
    
    for _, row in all_ratings.iterrows():
        date_obj = row['mapped_time']
        t_id = date_to_id.get(date_obj)
        
        mean_rating = user_means_dict.get(row['userId'], 0.0)
        n_rating = float(row['rating']) - mean_rating
        
        fact = FactRating(
            user_id=int(row['userId']),
            movie_id=int(row['movieId']),
            time_id=t_id,
            rating=float(row['rating']),
            normalized_rating=n_rating,
            is_train=bool(row.get('is_train', True))
        )
        batch.append(fact)
        
        # Bulk optimizations logic
        if len(batch) >= batch_size:
            session.bulk_save_objects(batch)
            session.commit()
            inserted_count += len(batch)
            batch = []
            
    if batch:
        session.bulk_save_objects(batch)
        session.commit()
        inserted_count += len(batch)
        
    print(f"Fact Extraction Finalized! Total Rating relationships established: {inserted_count:,}")

def main():
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('processed', 'data/processed'))
    warehouse_dir = os.path.join(base_dir, config.get('dataset_paths', {}).get('warehouse', 'data/warehouse'))
    
    os.makedirs(warehouse_dir, exist_ok=True)
    db_path = os.path.join(warehouse_dir, 'movie_warehouse.db')
    
    print(f"=== Commencing Phase 5: SQLite Database Schema Migrations ===")
    print(f"Initializing SQL Backend Schema -> {db_path}")
    
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        train = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
        test = pd.read_csv(os.path.join(proc_dir, 'test.csv'))
        train['is_train'] = True
        test['is_train'] = False
        all_ratings = pd.concat([train, test])
        
        movies = pd.read_csv(os.path.join(proc_dir, 'processed_movies.csv'))
        user_bias = pd.read_csv(os.path.join(proc_dir, 'user_bias_features.csv'))
        
        metadata_path = os.path.join(proc_dir, 'cleaned_metadata.csv')
        metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
        
        # Set Extract mappings
        all_genres = set()
        for g_str in movies['genres'].dropna():
            for g in g_str.split('|'):
                if g: all_genres.add(g.strip())
                
        user_means_dict = dict(zip(user_bias['userId'], user_bias['mean_rating']))
        dim_time_records, date_to_id = generate_time_dimension(all_ratings)
        
        # Step-1: Dimension Loadings
        populate_dim_user(session, user_bias)
        populate_dim_genre(session, sorted(list(all_genres)))
        populate_dim_movie(session, movies, metadata)
        populate_dim_time(session, dim_time_records)
        
        # Step-2: Fact Loading
        populate_fact_rating(session, all_ratings, date_to_id, user_means_dict)
        
        print("✅ Success! The Dimension Tables and Fact Database indices have been written and stabilized.")
        
    except Exception as e:
        session.rollback()
        print(f"CRITICAL: SQL Pipeline Fault: {e}")
    finally:
        session.close()

if __name__ == '__main__':
    main()
