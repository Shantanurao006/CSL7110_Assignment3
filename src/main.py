import pandas as pd

# Load dataset
movies = pd.read_csv("../data/ml-latest-small/movies.csv")
ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")

# Basic info
print("Movies Dataset:")
print(movies.head())

print("\nRatings Dataset:")
print(ratings.head())

print("\nShape of movies:", movies.shape)
print("Shape of ratings:", ratings.shape)