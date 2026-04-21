import pandas as pd
import numpy as np
import os
import random

def generate_mock_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    random.seed(42)
    
    num_movies = 1000
    num_users = 500
    num_ratings = 50000
    
    # 1. Generate Movies (raw_movies.csv)
    print("Generating movies...")
    movie_ids = np.arange(1, num_movies + 1)
    genres_list = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller', 'Romance', 'Horror', 'Documentary', 'Animation', 'Adventure']
    movies = pd.DataFrame({
        'movieId': movie_ids,
        'title': [f"Mock Movie {i}" for i in movie_ids],
        'genres': ["|".join(random.sample(genres_list, k=random.randint(1, 4))) for _ in range(num_movies)]
    })
    movies.to_csv(os.path.join(output_dir, 'raw_movies.csv'), index=False)
    
    # 2. Generate Links (raw_links.csv) for TMDB integration
    print("Generating links...")
    links = pd.DataFrame({
        'movieId': movie_ids,
        'imdbId': [f"tt{random.randint(100000, 999999):07d}" for _ in range(num_movies)],
        'tmdbId': [random.randint(1000, 99999) for _ in range(num_movies)]
    })
    links.to_csv(os.path.join(output_dir, 'raw_links.csv'), index=False)
    
    # 3. Generate Ratings (raw_ratings.csv)
    print("Generating ratings...")
    user_ids = np.random.randint(1, num_users + 1, size=num_ratings)
    rated_movie_ids = np.random.choice(movie_ids, size=num_ratings)
    ratings_vals = np.round(np.random.normal(loc=3.5, scale=1.0, size=num_ratings) * 2) / 2
    ratings_vals = np.clip(ratings_vals, 0.5, 5.0)
    timestamps = np.random.randint(1500000000, 1700000000, size=num_ratings)
    
    ratings = pd.DataFrame({
        'userId': user_ids,
        'movieId': rated_movie_ids,
        'rating': ratings_vals,
        'timestamp': timestamps
    })
    ratings.drop_duplicates(subset=['userId', 'movieId'], inplace=True)
    ratings.to_csv(os.path.join(output_dir, 'raw_ratings.csv'), index=False)
    
    # 4. Generate Tags (raw_tags.csv)
    print("Generating tags...")
    num_tags = 5000
    sample_tags = ['runaway', 'time travel', 'boring', 'masterpiece', 'long', 'funny', 'dark', 'plot twist', 'predictable']
    tags = pd.DataFrame({
        'userId': np.random.randint(1, num_users + 1, size=num_tags),
        'movieId': np.random.choice(movie_ids, size=num_tags),
        'tag': np.random.choice(sample_tags, size=num_tags),
        'timestamp': np.random.randint(1500000000, 1700000000, size=num_tags)
    })
    tags.drop_duplicates(subset=['userId', 'movieId', 'tag'], inplace=True)
    tags.to_csv(os.path.join(output_dir, 'raw_tags.csv'), index=False)
    
    # 5. Generate TMDB Metadata (metadata.csv)
    print("Generating metadata with TMDB fields...")
    languages = ['en', 'fr', 'es', 'de', 'ja', 'ko']
    metadata = pd.DataFrame({
        'movieId': movie_ids,
        # Intentionally dropping tmdbId here so it directly maps, or keeping it for verification
        'overview': [f"This is a rich mocked plot overview for movie {i}." for i in movie_ids],
        'cast': [f"Actor {random.randint(1,100)}|Actor {random.randint(101,200)}|Actor {random.randint(201,300)}|Actor {random.randint(301,400)}|Actor {random.randint(401,500)}" for _ in range(num_movies)],
        'director': [f"Director {random.randint(1,50)}" for _ in range(num_movies)],
        'poster_url': [f"https://image.tmdb.org/t/p/w500/mock_poster_{i}.jpg" for i in movie_ids],
        'original_language': np.random.choice(languages, size=num_movies, p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.1])
    })
    metadata.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    print(f"✅ Synthetic datasets mock generation complete! Files saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    generate_mock_data(raw_data_dir)
