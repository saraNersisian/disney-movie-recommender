# 🎬 Disney Movie Recommender (PyTorch)

A simple **movie recommender system** built in Python using the **MovieLens 100K dataset**.  
The model is trained with **Neural Collaborative Filtering (NCF) in PyTorch** and demonstrates personalized recommendations, with a focus on **Disney/Pixar/Marvel titles**.

---

## 🚀 How to Run

### 1. Clone & Setup
```bash
git clone https://github.com/saraNersisian/disney-movie-recommender.git
cd disney-movie-recommender

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
2. Run Jupyter
bash
Copy code
jupyter notebook
3. Notebooks
Run notebooks in order:

01_prepare_data.ipynb → load & preprocess MovieLens data (ratings ≥ 4, reindex IDs, train/val split).

02_train_ncf.ipynb → train a simple PyTorch NCF model and evaluate Recall@10 / NDCG@10.

03_demo_disney.ipynb → generate recommendations for a sample user and highlight Disney movies.

📊 Dataset
MovieLens 100K (latest-small)

100,836 ratings

9,724 movies

610 users

🧠 Approach
Treat ratings ≥ 4 as implicit “likes”.

Model: Neural Collaborative Filtering (user & movie embeddings → MLP → sigmoid).

Metrics: Recall@10, NDCG@10.

📈 Results (placeholder)
Model	Recall@10	NDCG@10
NCF (PyTorch)	0.xx	0.xx

(update after training in 02_train_ncf.ipynb)

🎯 Why Disney?
While trained on MovieLens, the demo emphasizes Disney/Pixar/Marvel/Lucasfilm titles, making the project directly relevant for Disney internship applications.

🛠 Tech Stack
Python 3.10+

PyTorch

Pandas, NumPy, Matplotlib

Jupyter

