"""
Script tự động train model từ dữ liệu TMDB
Chạy file này trước khi chạy Streamlit app
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("🎬 TMDB MOVIE RECOMMENDER - TRAINING MODEL")
print("=" * 60)

# 1. Load dữ liệu
print("\n[1/6] Đang load dữ liệu...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
print(f"✓ Đã load {len(movies)} phim từ movies.csv")
print(f"✓ Đã load {len(credits)} records từ credits.csv")

# 2. Merge và làm sạch dữ liệu
print("\n[2/6] Đang xử lý và làm sạch dữ liệu...")
credits_renamed = credits.rename(columns={'movie_id': 'id'})
movies_merged = movies.merge(credits_renamed[['id', 'cast', 'crew']], on='id', how='left')

# Loại bỏ duplicate
before_dups = movies_merged.shape[0]
movies_merged = movies_merged.drop_duplicates(subset=['title'])
print(f"✓ Đã loại bỏ {before_dups - len(movies_merged)} phim trùng lặp")

# Xử lý missing values
text_cols = ['overview', 'tagline', 'cast', 'crew', 'keywords', 'genres']
for col in text_cols:
    if col in movies_merged.columns:
        movies_merged[col] = movies_merged[col].fillna('')

num_cols = ['vote_average', 'vote_count', 'popularity', 'runtime']
for col in num_cols:
    if col in movies_merged.columns:
        movies_merged[col] = movies_merged[col].fillna(movies_merged[col].median())

# Xử lý outliers
low, high = movies_merged['vote_count'].quantile([0.01, 0.99])
movies_merged['vote_count_clipped'] = movies_merged['vote_count'].clip(lower=low, upper=high)

# Chuẩn hóa
scaler = MinMaxScaler()
movies_merged[['vote_avg_scaled', 'popularity_scaled', 'vote_count_scaled']] = scaler.fit_transform(
    movies_merged[['vote_average', 'popularity', 'vote_count_clipped']]
)
print("✓ Đã xử lý missing values, outliers và chuẩn hóa dữ liệu")

# 3. Feature Engineering
print("\n[3/6] Đang tạo features...")

def extract_genres(genres_str):
    try:
        genres_list = json.loads(genres_str)
        return [g['name'] for g in genres_list]
    except:
        return []

def extract_keywords(keywords_str):
    try:
        keywords_list = json.loads(keywords_str)
        return ' '.join([k['name'] for k in keywords_list])
    except:
        return ''

def extract_cast(cast_str):
    try:
        cast_list = json.loads(cast_str)
        return ' '.join([c['name'].replace(' ', '') for c in cast_list[:5]])
    except:
        return ''

def extract_director(crew_str):
    try:
        crew_list = json.loads(crew_str)
        for person in crew_list:
            if person.get('job') == 'Director':
                return person['name'].replace(' ', '')
        return ''
    except:
        return ''

movies_merged['genres_list'] = movies_merged['genres'].apply(extract_genres)
movies_merged['keywords_clean'] = movies_merged['keywords'].apply(extract_keywords)
movies_merged['cast_clean'] = movies_merged['cast'].apply(extract_cast)
movies_merged['director_clean'] = movies_merged['crew'].apply(extract_director)
movies_merged['genres_clean'] = movies_merged['genres_list'].apply(lambda x: ' '.join([g.replace(' ', '') for g in x]))

# Kết hợp features
movies_merged['combined_features'] = (
    movies_merged['overview'].fillna('') + ' ' +
    movies_merged['genres_clean'] + ' ' +
    movies_merged['keywords_clean'] + ' ' +
    movies_merged['cast_clean'] + ' ' +
    movies_merged['director_clean']
)
print("✓ Đã tạo combined features từ overview, genres, keywords, cast, director")

# 4. Vector hóa với TF-IDF
print("\n[4/6] Đang vector hóa với TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf.fit_transform(movies_merged['combined_features'])
print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

# 5. Tính Cosine Similarity
print("\n[5/6] Đang tính Cosine Similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"✓ Cosine similarity matrix shape: {cosine_sim.shape}")

# Tạo mapping
indices = pd.Series(movies_merged.index, index=movies_merged['title']).drop_duplicates()

# 6. Lưu model
print("\n[6/6] Đang lưu model...")
data_to_save = {
    'movies_data': movies_merged[['id', 'title', 'vote_average', 'vote_count', 'popularity', 
                                   'genres_clean', 'overview', 'release_date', 'runtime']],
    'cosine_sim': cosine_sim,
    'indices': indices
}

with open('movie_recommender_model.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

file_size = os.path.getsize('movie_recommender_model.pkl') / (1024*1024)
print(f"✓ Đã lưu model vào 'movie_recommender_model.pkl' ({file_size:.2f} MB)")

print("\n" + "=" * 60)
print("✅ HOÀN THÀNH! Model đã sẵn sàng.")
print("=" * 60)
print("\n📌 Bước tiếp theo:")
print("   Chạy lệnh: streamlit run app.py")
print("=" * 60)

