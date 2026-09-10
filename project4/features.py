"""
Task 1 - Feature representation for the movie utility model U(x) = w^T x.

Design goals (justified in the report):
  1. Low-dimensional & dense: preference elicitation must work from a handful
     of interactions, so x cannot be a huge sparse one-hot vector (e.g. one
     dimension per director/actor) or w would be impossible to estimate from
     few samples.
  2. Taste-relevant: dimensions should track things people actually have
     stable preferences over (genre, tone/maturity, era, pace, foreign vs.
     domestic, and directing/casting "prestige" as a proxy for style/quality
     signal), not incidental identifiers like the movie title itself.
  3. Reusable across the whole catalogue: every feature must be computable
     for *any* movie in the dataset, including ones never chosen before, so
     that a fitted w can score / rank the entire catalogue, not just the
     items seen during elicitation.

Feature vector x is built from:
  - Genre multi-hot (top-K most frequent genres in the corpus)
  - Duration (normalized)
  - Recency (normalized release year)
  - Popularity (log number of votes, normalized)
  - Crowd quality prior (IMDB score, normalized) -- a prior, not the
    personal utility we are trying to estimate
  - Director prestige (leave-one-out average IMDB score of the director's
    other movies in the corpus; a compact target-encoding instead of a
    sparse one-hot over directors)
  - Lead-cast prestige (same idea, averaged over the three credited actors)
  - Maturity level (ordinal encoding of content rating: G<PG<PG-13<R<NC-17)
  - is_english / is_domestic (binary, captures foreign-film taste)
"""
import numpy as np
import pandas as pd

RATING_ORDER = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 4}


def _prestige_encoding(df: pd.DataFrame, name_col: str, global_mean: float) -> pd.Series:
    """Leave-one-out mean IMDB score for a categorical column with many
    rare levels (directors, actors). Falls back to the global mean for
    movies whose director/actor appears only once (avoids leakage/overfitting
    to a single data point)."""
    grp_sum = df.groupby(name_col)["imdb_score"].transform("sum")
    grp_cnt = df.groupby(name_col)["imdb_score"].transform("count")
    loo_mean = (grp_sum - df["imdb_score"]) / (grp_cnt - 1).replace(0, np.nan)
    return loo_mean.fillna(global_mean)


def build_feature_matrix(df: pd.DataFrame, top_k_genres: int = 20):
    """Returns (X, feature_names, movie_ids) where X is an (n_movies, d)
    numpy array, row-aligned with df.index."""
    df = df.reset_index(drop=True).copy()
    global_mean_score = df["imdb_score"].mean()

    # --- genres: multi-hot over the top_k_genres most frequent genres ---
    genre_lists = df["genres"].str.split("|")
    all_genres = pd.Series([g for gl in genre_lists for g in gl])
    top_genres = all_genres.value_counts().head(top_k_genres).index.tolist()
    genre_mat = np.zeros((len(df), len(top_genres)))
    for i, gl in enumerate(genre_lists):
        for g in gl:
            if g in top_genres:
                genre_mat[i, top_genres.index(g)] = 1.0

    # --- numeric, normalized to roughly [0, 1] ---
    duration_n = (df["duration"] - df["duration"].min()) / (df["duration"].max() - df["duration"].min())
    year_n = (df["title_year"] - df["title_year"].min()) / (df["title_year"].max() - df["title_year"].min())
    pop_n = np.log1p(df["num_voted_users"])
    pop_n = (pop_n - pop_n.min()) / (pop_n.max() - pop_n.min())
    score_n = (df["imdb_score"] - df["imdb_score"].min()) / (df["imdb_score"].max() - df["imdb_score"].min())

    # --- prestige (director / cast), then normalized ---
    director_prestige = _prestige_encoding(df, "director_name", global_mean_score)
    df["_actor_stack_score"] = df["imdb_score"]
    actor_scores = []
    for col in ["actor_1_name", "actor_2_name", "actor_3_name"]:
        actor_scores.append(_prestige_encoding(df, col, global_mean_score))
    cast_prestige = pd.concat(actor_scores, axis=1).mean(axis=1)

    dp_n = (director_prestige - director_prestige.min()) / (director_prestige.max() - director_prestige.min() + 1e-9)
    cp_n = (cast_prestige - cast_prestige.min()) / (cast_prestige.max() - cast_prestige.min() + 1e-9)

    # --- maturity, language/country ---
    maturity = df["content_rating"].map(RATING_ORDER).fillna(2) / max(RATING_ORDER.values())
    is_english = (df["language"] == "English").astype(float)
    is_domestic = (df["country"] == "USA").astype(float)

    numeric_block = np.column_stack([
        duration_n, year_n, pop_n, score_n, dp_n, cp_n, maturity, is_english, is_domestic
    ])
    numeric_names = ["duration", "recency", "popularity", "quality_prior",
                      "director_prestige", "cast_prestige", "maturity",
                      "is_english", "is_domestic"]

    X = np.column_stack([genre_mat, numeric_block])
    feature_names = [f"genre::{g}" for g in top_genres] + numeric_names
    return X.astype(float), feature_names, df["movie_title"].tolist()


if __name__ == "__main__":
    df = pd.read_csv("movie_metadata_sample.csv")
    X, names, titles = build_feature_matrix(df)
    print("Feature matrix shape:", X.shape)
    print("Feature names:", names)
    print("\nExample row (", titles[0], "):")
    for n, v in zip(names, X[0]):
        print(f"  {n:22s} {v:.3f}")
    np.save("X_features.npy", X)
    import json
    json.dump({"feature_names": names, "movie_titles": titles}, open("feature_meta.json", "w"), indent=2)
