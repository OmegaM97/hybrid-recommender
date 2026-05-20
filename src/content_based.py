"""Content-based recommendation using TF-IDF and cosine similarity."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-based recommender that uses TF-IDF and cosine similarity."""

    def __init__(
        self,
        content_path: str = "data/preprocessed/content_movies.csv",
        tfidf_kwargs: Optional[dict] = None,
    ):
        self.content_path = Path(content_path)
        self.tfidf_kwargs = tfidf_kwargs or {
            "stop_words": "english",
            "ngram_range": (1, 2),
        }
        self.data: pd.DataFrame = pd.DataFrame()
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

    def fit(self, reload_data: bool = True) -> None:
        """Fit TF-IDF on the content column and build the cosine similarity matrix."""
        if reload_data or self.data.empty:
            self.load_data()

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


if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.fit()
    print(recommender.recommend_similar_movies("avengers", top_n=5))
