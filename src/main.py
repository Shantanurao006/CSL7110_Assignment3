import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse.linalg import svds

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

    # Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Compute user-user similarity
user_similarity = cosine_similarity(user_item_matrix)

# Convert to DataFrame for easy handling
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def predict_rating(user_id, movie_id, k=5):
    if movie_id not in user_item_matrix.columns:
        return 0

    # Similar users
    sim_users = user_similarity_df[user_id].sort_values(ascending=False)[1:k+1]

    numerator = 0
    denominator = 0

    for sim_user, similarity in sim_users.items():
        rating = user_item_matrix.loc[sim_user, movie_id]

        if rating > 0:
            numerator += similarity * rating
            denominator += similarity

    if denominator == 0:
        return 0

    return numerator / denominator


def recommend_cf(user_id, top_n=5):
    movies_not_rated = user_item_matrix.loc[user_id]
    movies_not_rated = movies_not_rated[movies_not_rated == 0].index

    predictions = []

    for movie_id in movies_not_rated:
        pred = predict_rating(user_id, movie_id)
        predictions.append((movie_id, pred))

    # Sort predictions
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    movie_ids = [i[0] for i in predictions]
    scores = [i[1] for i in predictions]

    results = movies[movies['movieId'].isin(movie_ids)][['title']].copy()
    results['predicted_rating'] = scores

    return results

    # -----------------------------
# Item-Based Collaborative Filtering
# -----------------------------

# Transpose user-item matrix to item-user
item_item_matrix = user_item_matrix.T

# Compute item-item similarity
item_similarity = cosine_similarity(item_item_matrix)

# Convert to DataFrame
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_item_matrix.index,
    columns=item_item_matrix.index
)


def predict_rating_item_based(user_id, movie_id, k=5):
    if movie_id not in item_similarity_df.index:
        return 0

    # Movies already rated by user
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]

    # Similar items
    similar_items = item_similarity_df[movie_id].sort_values(ascending=False)[1:k+1]

    numerator = 0
    denominator = 0

    for sim_movie, similarity in similar_items.items():
        if sim_movie in rated_movies.index:
            rating = rated_movies[sim_movie]
            numerator += similarity * rating
            denominator += similarity

    if denominator == 0:
        return 0

    return numerator / denominator


def recommend_item_based(user_id, top_n=5):
    user_ratings = user_item_matrix.loc[user_id]
    movies_not_rated = user_ratings[user_ratings == 0].index

    predictions = []

    for movie_id in movies_not_rated:
        pred = predict_rating_item_based(user_id, movie_id)
        predictions.append((movie_id, pred))

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    movie_ids = [i[0] for i in predictions]
    scores = [i[1] for i in predictions]

    results = movies[movies['movieId'].isin(movie_ids)][['title']].copy()
    results['predicted_rating'] = scores

    return results

def svd_recommend(user_id, top_n=5):
    # Convert to numpy matrix
    R = user_item_matrix.values

    # Normalize by subtracting user mean
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    # Apply SVD
    U, sigma, Vt = svds(R_demeaned, k=50)

    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)

    # Reconstruct ratings
    R_pred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    # Convert to DataFrame
    preds_df = pd.DataFrame(R_pred, columns=user_item_matrix.columns, index=user_item_matrix.index)

    # Get user's predicted ratings
    user_row = preds_df.loc[user_id]

    # Remove already rated movies
    already_rated = user_item_matrix.loc[user_id]
    user_row = user_row[already_rated == 0]

    # Sort predictions
    top_movies = user_row.sort_values(ascending=False).head(top_n)

    # Map movie IDs to titles
    results = movies[movies['movieId'].isin(top_movies.index)][['title']].copy()
    results['predicted_rating'] = top_movies.values

    return results


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

    # -----------------------------
    # User-Based CF Test
    # -----------------------------
    print("\nUser-Based Collaborative Filtering Recommendations:\n")

    cf_recommendations = recommend_cf(user_id=1, top_n=5)

    print(cf_recommendations)

    # -----------------------------
    # Item-Based CF Test
    # -----------------------------
    print("\nItem-Based Collaborative Filtering Recommendations:\n")

    item_cf_recommendations = recommend_item_based(user_id=1, top_n=5)

    print(item_cf_recommendations)

    # -----------------------------
    # SVD Recommendation
    # -----------------------------
    print("\nSVD Recommendations:\n")

    svd_results = svd_recommend(user_id=1, top_n=5)
    print(svd_results)


