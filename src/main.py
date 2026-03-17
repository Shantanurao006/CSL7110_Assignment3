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