"""Preprocessing utilities for the recommender data.

Provides content preprocessing for both content-based and collaborative
recommendation pipelines.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def _detect_movieid_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("movieId", "movie_id", "id"):
        if candidate in df.columns:
            return candidate
    for c in df.columns:
        if c.lower() in ("movieid", "movie_id", "id"):
            return c
    return None


def _detect_userid_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("userId", "user_id", "uid"):
        if candidate in df.columns:
            return candidate
    for c in df.columns:
        if c.lower() in ("userid", "user_id", "uid"):
            return c
    return None


def preprocess_content_movies(
    movies_path: str = "data/raw/movies.csv",
    tags_path: str = "data/raw/tags.csv",
    output_path: str = "data/preprocessed/content_movies.csv",
    extract_year: bool = True,
    nlp_clean: bool = True,
    lemmatize: bool = False,
) -> pd.DataFrame:
    """Build content dataset for content-based recommender and save CSV."""
    movies = pd.read_csv(movies_path)
    mid = _detect_movieid_col(movies) or "movieId"
    if mid != "movieId":
        movies = movies.rename(columns={mid: "movieId"})

    if "title" not in movies.columns:
        raise ValueError("`movies.csv` must contain a 'title' column")

    if "genres" not in movies.columns:
        movies["genres"] = ""

    movies["genres"] = movies["genres"].fillna("")
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    if "movieId" in movies.columns:
        movies = movies.drop_duplicates(subset=["movieId"]).copy()
    else:
        movies = movies.drop_duplicates().copy()

    if extract_year:
        def _extract(t: str):
            if not isinstance(t, str):
                return t, None
            m = re.match(r"^(?P<name>.*)\s+\((?P<year>\d{4})\)\s*$", t.strip())
            if m:
                return m.group("name").strip(), int(m.group("year"))
            return t.strip(), None

        title_year = movies["title"].apply(_extract)
        movies[["title", "year"]] = pd.DataFrame(title_year.tolist(), index=movies.index)

    try:
        tags = pd.read_csv(tags_path)
    except FileNotFoundError:
        tags = pd.DataFrame(columns=["movieId", "tag"])

    tmid = _detect_movieid_col(tags) or "movieId"
    if tmid != "movieId" and tmid in tags.columns:
        tags = tags.rename(columns={tmid: "movieId"})

    tag_col = next((c for c in ("tag", "Tag", "tags") if c in tags.columns), None)
    if tag_col is None:
        tags_agg = pd.DataFrame(columns=["movieId", "tags"]).astype({"movieId": "int64"})
    else:
        tags = tags[["movieId", tag_col]].dropna()
        tags["tag_clean"] = tags[tag_col].astype(str).str.strip()
        tags = tags[tags["tag_clean"] != ""]

        def _join_unique(x):
            seen = []
            for v in x:
                if v not in seen:
                    seen.append(v)
            return " ".join(seen)

        tags_agg = tags.groupby("movieId")["tag_clean"].agg(lambda x: _join_unique(x)).reset_index()
        tags_agg = tags_agg.rename(columns={"tag_clean": "tags"})

    merged = movies.merge(tags_agg, on="movieId", how="left")
    merged["genres_clean"] = merged["genres"].astype(str).str.replace("|", " ", regex=False)
    merged["tags"] = merged.get("tags").fillna("")
    merged["content"] = (merged["genres_clean"].str.strip() + " " + merged["tags"].str.strip()).str.strip()

    def _clean_text(s: str) -> str:
        if s is None:
            return ""
        s = str(s).lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    if nlp_clean:
        merged["content"] = merged["content"].apply(_clean_text)

    if lemmatize:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer

            try:
                nltk.data.find("corpora/wordnet")
            except Exception:
                nltk.download("wordnet")

            lemm = WordNetLemmatizer()

            def _lemmatize_text(text: str) -> str:
                return " ".join(lemm.lemmatize(tok) for tok in str(text).split())

            merged["content"] = merged["content"].apply(_lemmatize_text)
        except Exception:
            pass

    out = merged[["movieId", "title", "content"]].copy()
    out["content"] = out["content"].fillna("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def build_interaction_matrices(
    ratings_path: str = "data/raw/ratings.csv",
    movies_path: str = "data/raw/movies.csv",
    output_dir: str = "data/preprocessed",
    min_user_ratings: int = 5,
    min_movie_ratings: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict[int, int], Dict[int, int]]:
    """Build train/test interaction matrices and save maps to PKL files."""
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    uid_col = _detect_userid_col(ratings) or "userId"
    mid_col = _detect_movieid_col(ratings) or "movieId"
    ratings = ratings.rename(columns={uid_col: "userId", mid_col: "movieId"})

    if "rating" not in ratings.columns:
        raise ValueError("ratings CSV must contain a 'rating' column")

    ratings = ratings.drop_duplicates(subset=["userId", "movieId"]).copy()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()

    ratings = ratings[ratings["rating"].notna()].copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    while True:
        user_counts = ratings["userId"].value_counts()
        movie_counts = ratings["movieId"].value_counts()

        keep_users = user_counts[user_counts >= min_user_ratings].index
        keep_movies = movie_counts[movie_counts >= min_movie_ratings].index

        filtered = ratings[ratings["userId"].isin(keep_users) & ratings["movieId"].isin(keep_movies)]
        if len(filtered) == len(ratings):
            ratings = filtered
            break
        ratings = filtered

    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())
    user_map = {orig: idx for idx, orig in enumerate(user_ids)}
    movie_map = {orig: idx for idx, orig in enumerate(movie_ids)}

    ratings["user_index"] = ratings["userId"].map(user_map)
    ratings["item_index"] = ratings["movieId"].map(movie_map)

    num_users = len(user_map)
    num_items = len(movie_map)

    rows = ratings["user_index"].to_numpy(dtype=int)
    cols = ratings["item_index"].to_numpy(dtype=int)
    values = ratings["rating"].to_numpy(dtype=float)

    order = np.arange(len(ratings))
    rng = np.random.default_rng(random_state)
    rng.shuffle(order)
    split = int(len(order) * (1.0 - test_size))
    train_idx = order[:split]
    test_idx = order[split:]

    train_matrix = sparse.csr_matrix(
        (values[train_idx], (rows[train_idx], cols[train_idx])), shape=(num_users, num_items)
    )
    test_matrix = sparse.csr_matrix(
        (values[test_idx], (rows[test_idx], cols[test_idx])), shape=(num_users, num_items)
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "interaction_matrix_train.pkl", "wb") as f:
        pickle.dump(train_matrix, f)
    with open(output_path / "interaction_matrix_test.pkl", "wb") as f:
        pickle.dump(test_matrix, f)
    with open(output_path / "user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open(output_path / "movie_map.pkl", "wb") as f:
        pickle.dump(movie_map, f)

    return train_matrix, test_matrix, user_map, movie_map


if __name__ == "__main__":
    preprocess_content_movies()
    build_interaction_matrices()

