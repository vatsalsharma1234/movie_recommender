# Movie Recommendation System

A self-contained Python movie recommender combining two classic approaches:

- **Content-Based Filtering** — recommends movies similar to one you already like, based on genre/plot tags (TF-IDF + cosine similarity).
- **Collaborative Filtering** — recommends movies based on what similar *users* have rated highly (user-item matrix + cosine similarity).

No external dataset download is required — a small sample dataset (12 movies, 6 users) is built into the script, so it runs out of the box.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

## Setup

1. Make sure Python 3 is installed:
   ```bash
   python3 --version
   ```

2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn
   ```

## Running the Code

From the folder containing `movie_recommender.py`, run:

```bash
python3 movie_recommender.py
```

This runs the built-in demo, which prints:
1. Content-based recommendations for movies similar to **"Inception"**
2. Collaborative-filtering recommendations for a user who likes action/sci-fi
3. Collaborative-filtering recommendations for a user who likes horror

## Using It in Your Own Code

```python
from movie_recommender import ContentBasedRecommender, CollaborativeRecommender, MOVIES, RATINGS

# Content-based: movies similar to a given title
recommender = ContentBasedRecommender(MOVIES)
print(recommender.recommend("The Matrix", top_n=5))

# Collaborative: recommendations for a specific user_id
collab = CollaborativeRecommender(RATINGS, MOVIES)
print(collab.recommend(user_id=1, top_n=5))
```

## Using Your Own Data

Replace the sample `MOVIES` and `RATINGS` DataFrames with your own CSV files:

```python
import pandas as pd

movies = pd.read_csv("movies.csv")    # columns: movie_id, title, genres, tags
ratings = pd.read_csv("ratings.csv")  # columns: user_id, movie_id, rating

content_recommender = ContentBasedRecommender(movies)
collab_recommender = CollaborativeRecommender(ratings, movies)
```

- **movies.csv** needs: `movie_id`, `title`, `genres`, `tags` (free-text keywords describing the movie)
- **ratings.csv** needs: `user_id`, `movie_id`, `rating` (numeric, e.g. 1–5)

A good real-world dataset to try this on is [MovieLens](https://grouplens.org/datasets/movielens/).

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| `'X' not found. Available titles: [...]` | Check spelling — title matching is case-insensitive but must match a title in your dataset exactly |
| `user_id X not found in ratings data` | The user_id must exist in the ratings DataFrame/CSV |
