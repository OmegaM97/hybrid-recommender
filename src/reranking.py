"""Maximal Marginal Relevance (MMR) reranking for recommendation diversification."""
from __future__ import annotations

import pandas as pd
import numpy as np


def mmr_rerank(
    hybrid_recs: pd.DataFrame,
    similarity_matrix: np.ndarray,
    movie_id_to_idx: dict[int, int],
    top_n: int = 10,
    lambda_param: float = 0.7,
) -> pd.DataFrame:
    """
    Reranks recommendations using Maximal Marginal Relevance (MMR).
    
    Args:
        hybrid_recs: DataFrame with 'movieId', 'title', 'score'.
        similarity_matrix: User/Item similarity or tf-idf cosine similarity matrix.
        movie_id_to_idx: Mapping from movieId to similarity matrix row/col index.
        top_n: Number of final recommendations to return.
        lambda_param: Weighting for relevance vs diversity.
            1.0 = Max relevance (original ranking)
            0.0 = Max diversity
            
    Returns:
        DataFrame containing the top_n items reranked by MMR.
    """
    if hybrid_recs.empty:
        return hybrid_recs

    # Extract required columns
    scores_dict = hybrid_recs.set_index("movieId")["score"].to_dict()
    titles_dict = hybrid_recs.set_index("movieId")["title"].to_dict()
    candidate_ids = list(scores_dict.keys())

    # Normalize scores to 0-1 for stability against similarity bounds
    raw_scores = np.array(list(scores_dict.values()))
    min_s = raw_scores.min()
    max_s = raw_scores.max()
    
    if max_s > min_s:
        def normalized_score(mid):
            return (scores_dict[mid] - min_s) / (max_s - min_s)
    else:
        def normalized_score(mid):
            return 1.0 if max_s > 0 else 0.0

    selected_ids = []
    
    # Fast path if top_n >= candidates length
    limit = min(top_n, len(candidate_ids))

    while len(selected_ids) < limit and candidate_ids:
        best_mmr = -float("inf")
        best_item = None
        
        for cand_id in candidate_ids:
            score_rel = normalized_score(cand_id)
            
            # Compute diversity penalty
            if not selected_ids:
                penalty = 0.0
            else:
                cand_idx = movie_id_to_idx.get(cand_id)
                if cand_idx is not None:
                    similarities = []
                    for sel_id in selected_ids:
                        sel_idx = movie_id_to_idx.get(sel_id)
                        if sel_idx is not None:
                            similarities.append(similarity_matrix[cand_idx, sel_idx])
                        else:
                            similarities.append(0.0)
                    penalty = max(similarities) if similarities else 0.0
                else:
                    penalty = 0.0

            # MMR formula
            mmr_score = (lambda_param * score_rel) - ((1.0 - lambda_param) * penalty)
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_item = cand_id
                
        # Move best_item from candidates to selected
        selected_ids.append(best_item)
        candidate_ids.remove(best_item)

    # Rebuild output DataFrame
    result = []
    for mid in selected_ids:
        result.append({
            "movieId": mid,
            "title": titles_dict[mid],
            "score": scores_dict[mid]
        })
        
    return pd.DataFrame(result)


if __name__ == "__main__":
    from hybrid import HybridRecommender
    
    print("=== Testing MMR Reranking ===")
    recommender = HybridRecommender()
    recommender.fit()
    
    user_id = 3
    
    # 1. Get a larger pool of hybrid recommendations
    initial_recs = recommender.recommend_for_user(user_id=user_id, top_n=30)
    print("\n--- Original Top 10 Hybrid Recommendations ---")
    print(initial_recs.head(10).to_string(index=False))
    
    # 2. Extract similarity components
    sim_matrix = recommender.cb_recommender.similarity_matrix
    movie_id_to_idx = {
        int(mid): idx 
        for idx, mid in enumerate(recommender.cb_recommender.data["movieId"])
    }
    
    # 3. Apply MMR
    diverse_recs = mmr_rerank(
        hybrid_recs=initial_recs,
        similarity_matrix=sim_matrix,
        movie_id_to_idx=movie_id_to_idx,
        top_n=10,
        lambda_param=0.7
    )
    
    print("\n--- MMR Reranked Top 10 Recommendations (lambda=0.7) ---")
    print(diverse_recs.to_string(index=False))
