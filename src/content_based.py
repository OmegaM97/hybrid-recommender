"""Content-based recommendation using TF-IDF and cosine similarity."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-based recommender that uses TF-IDF and cosine similarity."""

    def __init__(
        self,
        content_path: str = "data/preprocessed/content_movies.csv",
        ratings_path: str = "data/raw/ratings.csv",
        tfidf_kwargs: Optional[dict] = None,
    ):
        self.content_path = Path(content_path)
        self.ratings_path = Path(ratings_path)
        self.tfidf_kwargs = tfidf_kwargs or {
            "stop_words": "english",
            "ngram_range": (1, 2),
        }
        self.data: pd.DataFrame = pd.DataFrame()
        self.ratings: pd.DataFrame = pd.DataFrame()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.vectorizer: Optional[TfidfVectorizer] = None

    def load_data(self) -> pd.DataFrame:
        """Load the content movie dataset from CSV."""
        if not self.content_path.exists():
            fallback = self.content_path.parent.parent / "content_movies.csv"
            if fallback.exists():
                self.content_path = fallback
            else:
                raise FileNotFoundError(
                    f"Content file not found: {self.content_path}. Run preprocessing first."
                )
        self.data = pd.read_csv(self.content_path)
        if "content" not in self.data.columns or "title" not in self.data.columns:
            raise ValueError(
                f"{self.content_path} must contain movieId, title, and content columns"
            )
        self.data["content"] = self.data["content"].fillna("")
        return self.data

    def load_ratings(self) -> pd.DataFrame:
        """Load the ratings dataset from CSV."""
        if not self.ratings_path.exists():
            raise FileNotFoundError(
                f"Ratings file not found: {self.ratings_path}"
            )
        self.ratings = pd.read_csv(self.ratings_path)
        # Normalize column names
        if "userId" not in self.ratings.columns and "user_id" in self.ratings.columns:
            self.ratings = self.ratings.rename(columns={"user_id": "userId"})
        if "movieId" not in self.ratings.columns and "movie_id" in self.ratings.columns:
            self.ratings = self.ratings.rename(columns={"movie_id": "movieId"})
        return self.ratings

    def fit(self, reload_data: bool = True) -> None:
        """Fit TF-IDF on the content column and build the cosine similarity matrix."""
        if reload_data or self.data.empty:
            self.load_data()
        
        if self.ratings.empty:
            self.load_ratings()

        self.vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data["content"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _get_movie_index(self, movie_title: str) -> int:
        title_lower = movie_title.strip().lower()
        candidate_rows = self.data[self.data["title"].str.lower() == title_lower]
        if not candidate_rows.empty:
            return int(candidate_rows.index[0])

        titles_lower = self.data["title"].str.lower().tolist()
        close = get_close_matches(title_lower, titles_lower, n=1, cutoff=0.6)
        if close:
            match_title = close[0]
            return int(self.data[self.data["title"].str.lower() == match_title].index[0])

        raise ValueError(f"Movie title not found: {movie_title}")

    def recommend_similar_movies(
        self,
        movie_title: str,
        top_n: int = 10,
        include_input: bool = False,
    ) -> pd.DataFrame:
        """Recommend movies similar to a given title."""
        if self.similarity_matrix is None:
            self.fit(reload_data=True)

        movie_idx = self._get_movie_index(movie_title)
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        if not include_input:
            similarity_scores = [item for item in similarity_scores if item[0] != movie_idx]

        top_scores = similarity_scores[:top_n]
        indices = [item[0] for item in top_scores]
        scores = [item[1] for item in top_scores]

        recommendations = self.data.iloc[indices].copy()
        recommendations["score"] = scores
        return recommendations[["movieId", "title", "score"]].reset_index(drop=True)

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        rating_threshold: float = 4.0,
    ) -> pd.DataFrame:
        """
        Recommend movies for a user based on their high-rated movies.
        
        Logic:
        1. Find movies the user rated >= rating_threshold
        2. Use those movies as the user's "content profile"
        3. Get TF-IDF similarities for those movies
        4. Average similarity scores across high-rated movies
        5. Filter out already-seen movies
        6. Return top-N unseen recommendations
        """
        if self.similarity_matrix is None:
            self.fit(reload_data=True)
        
        if self.ratings.empty:
            self.load_ratings()

        # Find movies the user rated highly
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"User {user_id} not found in ratings data")

        high_rated = user_ratings[user_ratings["rating"] >= rating_threshold]
        if high_rated.empty:
            raise ValueError(
                f"User {user_id} has no ratings >= {rating_threshold}. "
                f"Try a lower threshold or check user_id."
            )

        high_rated = high_rated.sort_values(
            by="rating",
            ascending=False
        ).head(5)

        # Map movie IDs to indices in self.data
        movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(self.data["movieId"])}
        
        weighted_similarity = np.zeros(len(self.data))
        total_weight = 0

        for _, row in high_rated.iterrows():
            movie_id = int(row["movieId"])
            rating = float(row["rating"])

            if movie_id not in movie_id_to_idx:
                continue

            idx = movie_id_to_idx[movie_id]

            weighted_similarity += self.similarity_matrix[idx] * rating
            total_weight += rating

        if total_weight == 0:
            raise ValueError(
                f"None of user {user_id}'s high-rated movies found in content data"
            )

        avg_similarity = weighted_similarity / total_weight

        # Get all movies the user has seen (rated)
        seen_movie_ids = set(user_ratings["movieId"].unique())
        seen_indices = {
            idx
            for idx, mid in enumerate(self.data["movieId"])
            if int(mid) in seen_movie_ids
        }

        # Filter to unseen movies and sort
        unseen_scores = [
            (idx, score)
            for idx, score in enumerate(avg_similarity)
            if idx not in seen_indices
        ]
        unseen_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-N
        top_indices = [idx for idx, _ in unseen_scores[:top_n]]
        top_scores = [score for _, score in unseen_scores[:top_n]]

        recommendations = self.data.iloc[top_indices].copy()
        recommendations["score"] = top_scores
        return recommendations[["movieId", "title", "score"]].reset_index(drop=True)


if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.fit()
    
    # Example: Recommend similar movies to a given title
    print("=== Title-based recommendation ===")
    print(recommender.recommend_similar_movies("avengers", top_n=5))
    
    # Example: Recommend movies for a user based on their high-rated movies
    print("\n=== User-based recommendation (content-based) ===")
    try:
        user_recs = recommender.recommend_for_user(user_id=3, top_n=10, rating_threshold=4.5)
        print(f"Recommendations for user 3:")
        print(user_recs.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
