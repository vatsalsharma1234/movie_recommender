"""
Movie Recommendation System
============================
A self-contained, runnable movie recommender that combines two classic
approaches:

1. Content-Based Filtering
   - Recommends movies similar to a movie you already like, based on
     genres/plot keywords (TF-IDF + cosine similarity).

2. Collaborative Filtering (user-based)
   - Recommends movies based on what similar *users* have liked
     (user-item ratings matrix + cosine similarity).

No external dataset download is required — a small built-in sample
dataset is included so this runs out of the box. Swap in your own
CSV data (see `load_your_own_data()` at the bottom) to scale it up.

Requirements: pandas, numpy, scikit-learn
    pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Sample data (replace with your own — see bottom of file)
# ---------------------------------------------------------------------------

MOVIES = pd.DataFrame([
    {"movie_id": 1, "title": "The Matrix",          "genres": "Action Sci-Fi",          "tags": "hacker simulation dystopia virtual reality"},
    {"movie_id": 2, "title": "John Wick",            "genres": "Action Thriller",        "tags": "assassin revenge gun-fu crime"},
    {"movie_id": 3, "title": "Inception",            "genres": "Action Sci-Fi Thriller", "tags": "dreams heist mind-bending"},
    {"movie_id": 4, "title": "The Notebook",         "genres": "Romance Drama",          "tags": "love letters memory war"},
    {"movie_id": 5, "title": "La La Land",           "genres": "Romance Musical Drama",  "tags": "jazz dreams love hollywood"},
    {"movie_id": 6, "title": "Interstellar",         "genres": "Sci-Fi Drama",           "tags": "space time travel family survival"},
    {"movie_id": 7, "title": "The Conjuring",        "genres": "Horror Thriller",        "tags": "haunted house ghost supernatural"},
    {"movie_id": 8, "title": "Get Out",              "genres": "Horror Thriller",        "tags": "psychological social commentary twist"},
    {"movie_id": 9, "title": "Toy Story",            "genres": "Animation Family",       "tags": "toys friendship adventure childhood"},
    {"movie_id": 10, "title": "Finding Nemo",        "genres": "Animation Family",       "tags": "ocean father son adventure"},
    {"movie_id": 11, "title": "The Dark Knight",     "genres": "Action Crime Thriller",  "tags": "batman joker gotham chaos"},
    {"movie_id": 12, "title": "Mad Max: Fury Road",  "genres": "Action Sci-Fi",          "tags": "wasteland chase survival war"},
])

# user_id, movie_id, rating (1-5)
RATINGS = pd.DataFrame([
    (1, 1, 5), (1, 2, 4), (1, 3, 5), (1, 11, 4), (1, 12, 5),
    (2, 4, 5), (2, 5, 5), (2, 6, 3), (2, 9, 4),
    (3, 1, 4), (3, 3, 5), (3, 6, 5), (3, 11, 5),
    (4, 7, 5), (4, 8, 4), (4, 2, 3),
    (5, 9, 5), (5, 10, 5), (5, 4, 3),
    (6, 1, 5), (6, 12, 4), (6, 2, 5), (6, 11, 5),
], columns=["user_id", "movie_id", "rating"])


# ---------------------------------------------------------------------------
# 2. Content-based recommender
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    """Recommends movies similar to a given title using genres + tags."""

    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        self.movies["combined_features"] = (
            self.movies["genres"] + " " + self.movies["tags"]
        )

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.movies["combined_features"])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

        self.title_to_index = {
            title.lower(): idx for idx, title in enumerate(self.movies["title"])
        }

    def recommend(self, title: str, top_n: int = 5) -> pd.DataFrame:
        key = title.lower()
        if key not in self.title_to_index:
            raise ValueError(
                f"'{title}' not found. Available titles: "
                f"{list(self.movies['title'])}"
            )

        idx = self.title_to_index[key]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        result_idx = [i for i, _ in scores]
        result_scores = [round(s, 3) for _, s in scores]

        result = self.movies.iloc[result_idx][["title", "genres"]].copy()
        result["similarity"] = result_scores
        return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Collaborative filtering (user-based)
# ---------------------------------------------------------------------------

class CollaborativeRecommender:
    """Recommends movies based on ratings from similar users."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.movies = movies
        self.user_item_matrix = ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def recommend(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"user_id {user_id} not found in ratings data.")

        # similarity of this user to every other user (excluding self)
        sim_scores = self.user_similarity[user_id].drop(user_id)

        # weighted sum of other users' ratings, weighted by similarity
        other_users_matrix = self.user_item_matrix.drop(index=user_id)
        weighted_ratings = other_users_matrix.T.dot(sim_scores)
        sim_sum = sim_scores.sum()
        predicted_scores = weighted_ratings / (sim_sum if sim_sum != 0 else 1)

        # exclude movies the user already rated
        already_rated = self.user_item_matrix.loc[user_id]
        predicted_scores = predicted_scores[already_rated == 0]

        top_movie_ids = predicted_scores.sort_values(ascending=False).head(top_n)

        result = self.movies[self.movies["movie_id"].isin(top_movie_ids.index)].copy()
        result["predicted_score"] = result["movie_id"].map(top_movie_ids).round(3)
        return result[["title", "genres", "predicted_score"]].sort_values(
            "predicted_score", ascending=False
        ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CONTENT-BASED: movies similar to 'Inception'")
    print("=" * 60)
    content_recommender = ContentBasedRecommender(MOVIES)
    print(content_recommender.recommend("Inception", top_n=5))

    print("\n" + "=" * 60)
    print("COLLABORATIVE: recommendations for user_id=1")
    print("(user 1 likes action/sci-fi movies)")
    print("=" * 60)
    collab_recommender = CollaborativeRecommender(RATINGS, MOVIES)
    print(collab_recommender.recommend(user_id=1, top_n=5))

    print("\n" + "=" * 60)
    print("COLLABORATIVE: recommendations for user_id=4")
    print("(user 4 likes horror movies)")
    print("=" * 60)
    print(collab_recommender.recommend(user_id=4, top_n=5))


def load_your_own_data():
    """
    To use your own data instead of the sample set:

    movies = pd.read_csv("movies.csv")   # needs: movie_id, title, genres, tags
    ratings = pd.read_csv("ratings.csv") # needs: user_id, movie_id, rating

    content_recommender = ContentBasedRecommender(movies)
    collab_recommender = CollaborativeRecommender(ratings, movies)
    """
    pass


if __name__ == "__main__":
    main()
