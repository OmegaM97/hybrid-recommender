## Features

- **Collaborative Filtering (CF):** Leverages Matrix Factorization (Truncated SVD) to map complex underlying user interactions and unearth latent affinities.
- **Content-Based Targeting (CB):** Utilizes pure TF-IDF cosine-similarity against rich item metadata precisely weighted by the highest priority user-ratings via the top 5 bounds.
- **Dynamic Hybrid Blending:** Seamlessly intersects huge hidden candidate pools from both recommendation models, normalizing their disparate dimensions using Min-Max Scaling to output perfect custom-weighted combination results natively.
- **Maximal Marginal Relevance (MMR):** An active diversity pipeline mapped via a mathematically penalized lambda variable ($\lambda=0.7$) to aggressively punish monotonous groupings inside franchises (e.g. recommending 3 adjacent films consequetively diminishes their final ranks relative to fresh independent alternatives).
- **Cold Start Guardian:** Algorithmically safeguards system stability falling-back to popularity-mapped matrices temporarily for users with less than `5` footprint interactions!

## How to Run & Test

Ensure your Python environment acts on identical dependencies (installing basics like `pandas`, `scipy` and `sklearn` from your dependencies tree).

### 1. Test the Pure Hybrid Blending

Run the primary `hybrid.py` execution structure to trigger the standard engine (internally bound natively mapping `user_id=3` metrics).

```bash
uv run python src/hybrid.py
```

### 2. Test MMR Diversification

To process our entire architectural loop simultaneously, run the `reranking` terminal test. This invokes the `HybridRecommender`, pulls a vast subset of intersections, and dynamically mathematically offsets top tier repetitions visually showing structural improvement.

```bash
uv runpython src/reranking.py
```

### 3. Analytics Playground

We generated a Jupyter execution suite directly interacting analyzing all subsets. Launch environments to review `experiments.ipynb`.

```bash
jupyter notebook notebooks/experiments.ipynb
```
