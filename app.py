"""
Movie Recommendation System  -  ML Mini Project
Algorithms : 1) KNN (K-Nearest Neighbors, cosine distance)
             2) K-Means Clustering
UI         : Streamlit
Run        : streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

# ---------- 1. Load data ----------
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

# ---------- 2. Feature engineering (text/category -> numbers) ----------
@st.cache_data
def build_features(df):
    genres = MultiLabelBinarizer().fit_transform(df["genres"].str.split("|"))      # one-hot genres
    tags = CountVectorizer(binary=True).fit_transform(df["tags"]).toarray()        # bag of words
    rating = MinMaxScaler().fit_transform(df[["rating"]])                          # scale 0..1
    return np.hstack([genres * 2, tags, rating])                                   # genre weighted x2

# ---------- 3. Algorithm 1: KNN ----------
@st.cache_resource
def train_knn(X):
    return NearestNeighbors(metric="cosine", algorithm="brute").fit(X)

def recommend_knn(idx, X, model, df, n):
    dist, ind = model.kneighbors(X[idx].reshape(1, -1), n_neighbors=n + 1)
    ind, dist = ind[0][1:], dist[0][1:]            # drop the movie itself
    out = df.iloc[ind].copy()
    out["match"] = np.round((1 - dist) * 100, 1)
    return out

# ---------- 4. Algorithm 2: K-Means ----------
@st.cache_resource
def train_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)

def recommend_kmeans(idx, X, model, df, n):
    same = np.where(model.labels_ == model.labels_[idx])[0]
    same = same[same != idx]                        # movies in same cluster
    d = np.linalg.norm(X[same] - X[idx], axis=1)    # euclidean distance
    order = np.argsort(d)[:n]
    out = df.iloc[same[order]].copy()
    out["match"] = np.round(100 / (1 + d[order]), 1)
    return out

# ---------- 5. UI ----------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")

df = load_data()
X = build_features(df)

st.sidebar.header("Settings")
algo = st.sidebar.radio("Choose ML algorithm", ["KNN (K-Nearest Neighbors)", "K-Means Clustering"])
n = st.sidebar.slider("How many recommendations?", 3, 12, 6)
k = 10
if algo.startswith("K-Means"):
    k = st.sidebar.slider("Number of clusters (K)", 4, 15, 10)
st.sidebar.write(f"📊 Total movies: **{len(df)}**")

movie = st.selectbox("Pick a movie you like:", sorted(df["title"]))

if st.button("Recommend 🍿", type="primary"):
    idx = df.index[df["title"] == movie][0]
    if algo.startswith("KNN"):
        recs = recommend_knn(idx, X, train_knn(X), df, n)
    else:
        recs = recommend_kmeans(idx, X, train_kmeans(X, k), df, n)

    st.subheader(f"Because you liked **{movie}**  ({algo.split(' (')[0]})")
    cols = st.columns(3)
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 3].container(border=True):
            st.markdown(f"### {r['title']} ({r['year']})")
            st.write(f"⭐ {r['rating']}  |  🎭 {r['genres'].replace('|', ', ')}")
            st.progress(min(int(r["match"]), 100), text=f"Match: {r['match']}%")

with st.expander("📘 How does this work?"):
    st.write("""
**Features:** genres (one-hot) + tags (bag of words) + rating (scaled 0-1).

**KNN:** finds the K movies closest to your movie using cosine distance.
**K-Means:** groups all movies into K clusters, then recommends movies from the
same cluster as your movie (nearest first).
    """)

with st.expander("🎞️ See movie dataset"):
    st.dataframe(df[["title", "year", "rating", "genres"]], use_container_width=True)
