"""Hybrid recommender system blending Collaborative and Content-Based filtering."""
from __future__ import annotations

import pandas as pd
from typing import Optional

from collaborative import CollaborativeRecommender
from content_based import ContentBasedRecommender


class HybridRecommender:
    """Combines recommendations from collaborative and content-based models."""

    def __init__(
        self,
        cf_weight: float = 0.5,
        cb_weight: float = 0.5,
        data_dir: str = "data",
    ):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.data_dir = data_dir

        self.cf_recommender = CollaborativeRecommender(
            data_dir=f"{data_dir}/preprocessed"
        )
        self.cb_recommender = ContentBasedRecommender(
            content_path=f"{data_dir}/preprocessed/content_movies.csv",
            ratings_path=f"{data_dir}/raw/ratings.csv"
        )

    def fit(self) -> None:
        """Ensures both underlying recommenders are trained or loaded."""
        # Content model fit
        self.cb_recommender.fit(reload_data=False)
        # Collab model fit
        self.cf_recommender.fit(save_artifacts=False)

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        min_ratings: int = 5,
        movies_path: str = "data/raw/movies.csv"
    ) -> pd.DataFrame:
        """
        Produce a list of top-N movie recommendations by blending CF and CB models.
        """
        # Ensure data is loaded
        if self.cb_recommender.ratings.empty:
            self.cb_recommender.load_ratings()

        # 1. Cold start check
        user_ratings = self.cb_recommender.ratings[self.cb_recommender.ratings["userId"] == user_id]
        if len(user_ratings) < min_ratings:
            # Fall back directly to popular items handled inside collaborative recommender
            return self.cf_recommender.recommend_with_scores(
                user_id=user_id,
                top_n=top_n,
                movies_path=movies_path
            )

        # 2. Get recommendations from both models
        # Fetch a large candidate pool to intersect, 2000 is plenty enough to cover top intersections
        pool_size = 2000
        
        try:
            cb_recs = self.cb_recommender.recommend_for_user(
                user_id=user_id,
                top_n=pool_size,
                rating_threshold=4.0
            ) 
        except ValueError:
            # E.g. No ratings >= 4.0, default to collaborative
            return self.cf_recommender.recommend_with_scores(
                user_id=user_id,
                top_n=top_n,
                movies_path=movies_path
            )
            
        cf_recs = self.cf_recommender.recommend_with_scores(
            user_id=user_id,
            top_n=pool_size,
            movies_path=movies_path
        )

        # 3. Normalize both score sets using Min-Max scaling
        def min_max_scale(series: pd.Series) -> pd.Series:
            s_min = series.min()
            s_max = series.max()
            if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
                return pd.Series(0.0, index=series.index)
            return (series - s_min) / (s_max - s_min)

        if not cb_recs.empty:
            cb_recs["score"] = min_max_scale(cb_recs["score"])
        else:
            cb_recs["score"] = pd.Series(dtype=float)

        if not cf_recs.empty:
            cf_recs["score"] = min_max_scale(cf_recs["score"])
        else:
            cf_recs["score"] = pd.Series(dtype=float)

        # 4. Outer join the items on movieId
        cb_subset = cb_recs[["movieId", "title", "score"]].rename(columns={"score": "cb_score", "title": "cb_title"})
        cf_subset = cf_recs[["movieId", "title", "score"]].rename(columns={"score": "cf_score", "title": "cf_title"})

        hybrid_df = pd.merge(
            cb_subset,
            cf_subset,
            on="movieId",
            how="outer"
        )
        
        hybrid_df["title"] = hybrid_df["cb_title"].combine_first(hybrid_df["cf_title"])
        hybrid_df = hybrid_df.drop(columns=["cb_title", "cf_title"]).fillna(0.0)

        # 5. Combine using weighted sum
        hybrid_df["score"] = (
            hybrid_df["cb_score"] * self.cb_weight +
            hybrid_df["cf_score"] * self.cf_weight
        )

        # 6. Seen movies are naturally filtered out by underlying recommenders

        # 7. Sort and take top N
        hybrid_df = hybrid_df.sort_values(by="score", ascending=False).head(top_n)

        # Return standardized columns
        return hybrid_df[["movieId", "title", "score"]].reset_index(drop=True)


if __name__ == "__main__":
    recommender = HybridRecommender()
    recommender.fit()
    recs = recommender.recommend_for_user(user_id=3, top_n=10)
    print("=== Hybrid Recommendations for User 3 ===")
    print(recs.to_string(index=False))
