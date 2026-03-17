import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Dataset
# -----------------------------
movies = pd.read_csv("../data/ml-latest-small/movies.csv")
ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")

# Fill missing genres
movies['genres'] = movies['genres'].fillna('')

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
tfidf = TfidfVectorizer(token_pattern=r"[^|]+")  # split genres by |

tfidf_matrix = tfidf.fit_transform(movies['genres'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# -----------------------------
# Cosine Similarity
# -----------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create mapping from movie title to index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie_title, top_n=5):
    if movie_title not in indices:
        print(f"Movie '{movie_title}' not found!")
        return

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip itself
    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]

    results = movies.iloc[movie_indices][['title']].copy()
    results['similarity'] = [i[1] for i in sim_scores]

    return results

# -----------------------------
# Test Recommendation
# -----------------------------
if __name__ == "__main__":
    movie_name = "Toy Story (1995)"
    print(f"\nTop recommendations for: {movie_name}\n")

    recommendations = recommend(movie_name, top_n=5)

    if recommendations is not None:
        print(recommendations)

# -----------------------------
# User Profile Based Recommender
# -----------------------------

def build_user_profile(user_id):
    # Get movies rated by user
    user_ratings = ratings[ratings['userId'] == user_id]

    if user_ratings.empty:
        print("User not found!")
        return None

    # Merge with movies to get indices
    user_data = user_ratings.merge(movies, on='movieId')

    # Get indices of movies
    movie_indices = user_data.index

    # Get corresponding TF-IDF vectors
    user_tfidf = tfidf_matrix[user_data.index]

    # Get ratings as weights
    weights = user_data['rating'].values.reshape(-1, 1)

    # Weighted sum
    user_profile = (user_tfidf.multiply(weights)).sum(axis=0)

    # Normalize
    # Weighted sum
    user_profile = (user_tfidf.multiply(weights)).sum(axis=0)

    # Convert to numpy array (FIX)
    user_profile = user_profile.A  # or np.asarray(user_profile)

    # Normalize
    user_profile = user_profile / weights.sum()

    return user_profile


def recommend_for_user(user_id, top_n=5):
    user_profile = build_user_profile(user_id)

    if user_profile is None:
        return

    # Compute similarity with all movies
    sim_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # Get movies already watched
    watched_movies = ratings[ratings['userId'] == user_id]['movieId']

    # Exclude watched movies
    movie_indices = movies[~movies['movieId'].isin(watched_movies)].index

    # Sort remaining movies
    sim_scores_filtered = [(i, sim_scores[i]) for i in movie_indices]
    sim_scores_filtered = sorted(sim_scores_filtered, key=lambda x: x[1], reverse=True)

    top_movies = sim_scores_filtered[:top_n]

    results = movies.iloc[[i[0] for i in top_movies]][['title']].copy()
    results['score'] = [i[1] for i in top_movies]

    return results

# -----------------------------
# Evaluation Metrics
# -----------------------------

def precision_recall_at_k(user_id, k=5):
    # Get recommended movies
    recommendations = recommend_for_user(user_id, top_n=k)

    if recommendations is None:
        return None

    recommended_titles = set(recommendations['title'])

    # Ground truth: movies user rated >= 4
    user_data = ratings[ratings['userId'] == user_id]
    relevant_movies = user_data[user_data['rating'] >= 4]

    relevant_movie_ids = set(relevant_movies['movieId'])

    # Map movieId to title
    relevant_titles = set(
        movies[movies['movieId'].isin(relevant_movie_ids)]['title']
    )

    # Intersection
    relevant_and_recommended = recommended_titles.intersection(relevant_titles)

    precision = len(relevant_and_recommended) / k
    recall = len(relevant_and_recommended) / len(relevant_titles) if len(relevant_titles) > 0 else 0

    return precision, recall


# -----------------------------
# Test User-Based Recommendation
# -----------------------------
if __name__ == "__main__":
    # Movie recommendation test
    movie_name = "Toy Story (1995)"
    print(f"\nTop recommendations for: {movie_name}\n")
    print(recommend(movie_name, top_n=5))

    # User recommendation test
    user_id = 1
    print("\nUser-based recommendations:\n")
    user_recommendations = recommend_for_user(user_id, top_n=5)

    if user_recommendations is not None:
        print(user_recommendations)

    # Evaluation
    print("\nEvaluation Metrics:\n")
    precision, recall = precision_recall_at_k(user_id, k=5)

    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")