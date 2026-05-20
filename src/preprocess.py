"""Preprocessing utilities for the recommender data.

Provides `preprocess_content_movies(...)` to build a `data/content_movies.csv`
file from `data/movies.csv` and `data/tags.csv`.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd


def _detect_movieid_col(df: pd.DataFrame) -> Optional[str]:
	for candidate in ("movieId", "movie_id", "id"):
		if candidate in df.columns:
			return candidate
	for c in df.columns:
		if c.lower() in ("movieid", "movie_id", "id"):
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
	"""Build content dataset for content-based recommender and save CSV.

	Output columns: `movieId`, `title`, `content`.

	Parameters:
	- `movies_path`, `tags_path`, `output_path`: file paths
	- `extract_year`: extract year from titles like "Toy Story (1995)"
	- `nlp_clean`: lowercase, remove punctuation, collapse spaces
	- `lemmatize`: optional lemmatization (requires `nltk`)
	"""
	movies = pd.read_csv(movies_path)
	mid = _detect_movieid_col(movies) or "movieId"
	if mid != "movieId":
		movies = movies.rename(columns={mid: "movieId"})

	if "title" not in movies.columns:
		raise ValueError("`movies.csv` must contain a 'title' column")

	# ensure genres column
	if "genres" not in movies.columns:
		movies["genres"] = ""

	# normalize genres
	movies["genres"] = movies["genres"].fillna("")
	movies["genres"] = movies["genres"].replace("(no genres listed)", "")

	# drop exact duplicate movieId rows
	if "movieId" in movies.columns:
		movies = movies.drop_duplicates(subset=["movieId"]).copy()
	else:
		movies = movies.drop_duplicates().copy()

	# optional: extract year from title
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

	# load tags and aggregate
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

	# combine genres and tags into single content field
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
			# if nltk not available or fails, continue without lemmatization
			pass

	out = merged[["movieId", "title", "content"]].copy()
	out["content"] = out["content"].fillna("")
	out.to_csv(output_path, index=False)
	return out


if __name__ == "__main__":
	df = preprocess_content_movies()
	print(f"Wrote {len(df)} rows to data/preprocessed/content_movies.csv")

