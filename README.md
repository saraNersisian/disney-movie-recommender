
# Disney Movie Recommender (PyTorch)
**What it is**: A neural collaborative filtering recommender trained on MovieLens 100K. Produces top‑N movie recommendations per user and compares against a matrix factorization baseline.
## Demo
- Streamlit app (optional): `streamlit run app/streamlit_app.py`
- Example: Input 3 favorite movies → returns Top‑10 recommendations.
## Dataset
- **MovieLens Latest-small**: users, items, ratings (0–5).  
- Download from: `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`
- **TMDb**: Rich movies metadata with Public API
## Approach
- **Baseline**: Matrix Factorization (implicit feedback w/ negative sampling).  
- **Model**: Neural Collaborative Filtering (user/item embeddings → MLP → dot/prediction).  
- **Loss**: BCE with implicit positives (rated items) + sampled negatives.  
- **Metrics**: Recall@K, NDCG@K, Hit Rate.
_Add training curves and bar charts in `/experiments/results/plots` and embed here._
## Repro
```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 2) Data
bash scripts/download_movielens.sh
python scripts/preprocess.py
# 3) Baseline
jupyter nbconvert --to notebook --execute notebooks/02_baseline_mf.ipynb
# 4) Train NCF
python src/train.py --config experiments/configs/ncf.yaml
python src/evaluate.py --ckpt runs/ncf_best.pt --k 10
# 5) Streamlit 
streamlit run app/streamlit_app.py
<img width="1270" height="646" alt="image" src="https://github.com/user-attachments/assets/d3347966-8bb0-4e05-9a3c-45dd0b42c1d5" />

