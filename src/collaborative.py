"""Collaborative filtering recommender using Truncated SVD."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy import sparse


class CollaborativeRecommender:
    """Collaborative filtering recommender trained on interaction matrices."""

    def __init__(
        self,
        data_dir: str = "data/preprocessed",
        n_components: int = 50,
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.n_components = n_components
        self.random_state = random_state

        self.train_matrix: Optional[sparse.csr_matrix] = None
        self.user_map: Dict[int, int] = {}
        self.movie_map: Dict[int, int] = {}
        self.movie_map_inv: Dict[int, int] = {}
        self.svd_model: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.popular_items: Optional[np.ndarray] = None

    def _load_pickle(self, filename: str):
        with open(self.data_dir / filename, "rb") as f:
            return pickle.load(f)

    def load_data(self) -> None:
        """Load training interaction matrix and ID mappings."""
        self.train_matrix = self._load_pickle("interaction_matrix_train.pkl")
        self.user_map = self._load_pickle("user_map.pkl")
        self.movie_map = self._load_pickle("movie_map.pkl")
        self.movie_map_inv = {v: k for k, v in self.movie_map.items()}

        if not sparse.isspmatrix_csr(self.train_matrix):
            self.train_matrix = self.train_matrix.tocsr()

        popularity = np.array(self.train_matrix.sum(axis=0)).ravel()
        self.popular_items = np.argsort(popularity)[::-1]

    def fit(self, save_artifacts: bool = True) -> None:
        """Train Truncated SVD and compute user/item embeddings."""
        if self.train_matrix is None or not self.user_map or not self.movie_map:
            self.load_data()

        self.svd_model = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.user_factors = self.svd_model.fit_transform(self.train_matrix)
        self.item_factors = self.svd_model.components_.T

        if save_artifacts:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "svd_model.pkl", "wb") as f:
                pickle.dump(self.svd_model, f)
            with open(self.data_dir / "user_factors.pkl", "wb") as f:
                pickle.dump(self.user_factors, f)
            with open(self.data_dir / "item_factors.pkl", "wb") as f:
                pickle.dump(self.item_factors, f)

    def _ensure_trained(self) -> None:
        if self.user_factors is None or self.item_factors is None:
            self.fit(save_artifacts=False)

    def _get_user_index(self, user_id: int) -> Optional[int]:
        return self.user_map.get(user_id)

    def _get_movie_index(self, movie_id: int) -> Optional[int]:
        return self.movie_map.get(movie_id)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating score for a given user/movie pair."""
        self._ensure_trained()

        user_idx = self._get_user_index(user_id)
        item_idx = self._get_movie_index(movie_id)
        if user_idx is None or item_idx is None:
            return 0.0

        user_vec = self.user_factors[user_idx]
        item_vec = self.item_factors[item_idx]
        return float(np.dot(user_vec, item_vec))

    def recommend(self, user_id: int, top_n: int = 10) -> np.ndarray:
        """Recommend top-N unseen movies for the given user."""
        self._ensure_trained()

        user_idx = self._get_user_index(user_id)
        if user_idx is None:
            return np.array([self.movie_map_inv[i] for i in self.popular_items[:top_n]], dtype=int)

        user_vector = self.user_factors[user_idx]
        scores = self.item_factors.dot(user_vector)

        user_interactions = self.train_matrix[user_idx].toarray().ravel()
        unseen_mask = user_interactions == 0
        candidate_indices = np.where(unseen_mask)[0]

        sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        top_indices = sorted_indices[:top_n]
        return np.array([self.movie_map_inv[idx] for idx in top_indices], dtype=int)

    def load_artifacts(self) -> None:
        """Load precomputed SVD model and embedding files from disk."""
        self.load_data()
        self.svd_model = self._load_pickle("svd_model.pkl")
        self.user_factors = self._load_pickle("user_factors.pkl")
        self.item_factors = self._load_pickle("item_factors.pkl")

    def recommend_with_titles(
        self,
        user_id: int,
        top_n: int = 10,
        movies_path: str = "data/raw/movies.csv",
    ) -> pd.DataFrame:
        """Recommend top-N movies for user and return movie titles."""
        recommendations = self.recommend(user_id, top_n=top_n)
        titles = self._load_movie_titles(movies_path)
        result = [
            {
                "movieId": int(movie_id),
                "title": titles.get(movie_id, "Unknown"),
            }
            for movie_id in recommendations
        ]
        return pd.DataFrame(result)

    def _load_movie_titles(self, movies_path: str) -> Dict[int, str]:
        movies = pd.read_csv(movies_path)
        mid_col = next((c for c in ("movieId", "movie_id", "id") if c in movies.columns), None)
        title_col = next((c for c in ("title", "movie_title") if c in movies.columns), None)
        if mid_col is None or title_col is None:
            return {}
        return dict(zip(movies[mid_col].astype(int), movies[title_col].astype(str)))


if __name__ == "__main__":
    recommender = CollaborativeRecommender()
    recommender.fit()

    user_id = 4
    recs = recommender.recommend_with_titles(user_id, top_n=10)

    print(f"Recommendations for user {user_id}:")
    print(recs.to_string(index=False))
