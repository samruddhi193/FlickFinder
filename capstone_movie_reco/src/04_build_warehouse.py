"""
Module to build SQLite warehouse for structured querying and reporting.
"""
import os
import yaml
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Date
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class DimUser(Base):
    __tablename__ = 'dim_user'
    user_id = Column(Integer, primary_key=True)
    rating_count = Column(Integer)
    avg_rating = Column(Float)
    activity_level = Column(String(50))

class DimMovie(Base):
    __tablename__ = 'dim_movie'
    movie_id = Column(Integer, primary_key=True)
    title = Column(String(255), index=True)
    release_year = Column(Integer, nullable=True)
    decade = Column(String(10), nullable=True)
    primary_genre = Column(String(100), nullable=True)
    all_genres = Column(String(500), nullable=True)
    overview_length = Column(Integer, nullable=True)

class DimGenre(Base):
    __tablename__ = 'dim_genre'
    genre_id = Column(Integer, primary_key=True, autoincrement=True)
    genre_name = Column(String(100), unique=True, index=True)

class DimTime(Base):
    __tablename__ = 'dim_time'
    time_id = Column(Integer, primary_key=True, autoincrement=True)
    full_date = Column(Date, unique=True, index=True)
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
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_warehouse():
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    proc_dir = os.path.join(base_dir, config.get('paths', {}).get('processed_data', 'data/processed/'))
    warehouse_dir = os.path.join(base_dir, config.get('paths', {}).get('warehouse', 'data/warehouse/'))
    os.makedirs(warehouse_dir, exist_ok=True)
    
    db_path = os.path.join(warehouse_dir, 'movie_warehouse.db')
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    print("Loading data...")
    train = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(proc_dir, 'test.csv'))
    train['is_train'] = True
    test['is_train'] = False
    all_ratings = pd.concat([train, test])
    
    movies = pd.read_csv(os.path.join(proc_dir, 'processed_movies.csv'))

    print("Building dim_user...")
    user_stats = all_ratings.groupby('userId').agg(
        rating_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    def get_activity(count):
        if count < 50: return 'Low'
        if count < 500: return 'Medium'
        return 'High'
        
    user_stats['activity_level'] = user_stats['rating_count'].apply(get_activity)
    
    dim_users = [
        DimUser(
            user_id=row['userId'],
            rating_count=row['rating_count'],
            avg_rating=row['avg_rating'],
            activity_level=row['activity_level']
        ) for _, row in user_stats.iterrows()
    ]
    session.bulk_save_objects(dim_users)
    session.commit()

    print("Building dim_movie...")
    dim_movies = []
    # Movies may have genres stored as string "['Action', 'Adventure']"
    import ast
    def parse_genres(g_str):
        if isinstance(g_str, str):
            try:
                g_list = ast.literal_eval(g_str)
                if isinstance(g_list, list):
                    return g_list
            except:
                pass
            return g_str.split('|')
        return []

    unique_genres = set()
    for _, row in movies.iterrows():
        g_list = parse_genres(row.get('genres', '[]'))
        primary = g_list[0] if g_list else 'Unknown'
        all_g = "|".join(g_list)
        for g in g_list:
            unique_genres.add(g)
            
        o_len = len(str(row.get('overview', ''))) if pd.notna(row.get('overview')) else 0
        ry = int(row['release_year']) if pd.notna(row.get('release_year')) else None
        
        dim_movies.append(DimMovie(
            movie_id=int(row['movieId']),
            title=str(row['title']),
            release_year=ry,
            decade=str(row.get('decade', 'Unknown')),
            primary_genre=primary,
            all_genres=all_g,
            overview_length=o_len
        ))
    session.bulk_save_objects(dim_movies)
    session.commit()

    print("Building dim_genre...")
    dim_genres = [DimGenre(genre_name=g) for g in sorted(unique_genres)]
    session.add_all(dim_genres)
    session.commit()

    print("Building dim_time...")
    all_ratings['mapped_time'] = pd.to_datetime(all_ratings['timestamp'], unit='s', errors='coerce').dt.date
    unique_dates = all_ratings['mapped_time'].dropna().unique()
    
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
            day_of_week=py_date.weekday(),
            is_weekend=py_date.weekday() >= 5
        ))
        time_id_counter += 1
        
    session.bulk_save_objects(dim_time_records)
    session.commit()

    print("Building fact_rating...")
    batch_size = 5000
    batch = []
    inserted_count = 0
    
    for _, row in all_ratings.iterrows():
        t_id = date_to_id.get(row['mapped_time'])
        
        fact = FactRating(
            user_id=int(row['userId']),
            movie_id=int(row['movieId']),
            time_id=t_id,
            rating=float(row['rating']),
            normalized_rating=float(row['normalized_rating']) if pd.notna(row['normalized_rating']) else None,
            is_train=bool(row['is_train'])
        )
        batch.append(fact)
        
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
    session.close()

if __name__ == "__main__":
    build_warehouse()
