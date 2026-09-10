import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from train_models import train_all_models

st.set_page_config(page_title="HCAI Project 2", layout="wide")
st.title("🐧 Palmer Penguins — Explainability Dashboard")

# ── Load everything once ─────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    return train_all_models()

with st.spinner("Training models... (~15 seconds)"):
    tree_models, lr_models, X_train, X_test, y_train, y_test, feature_names, species_labels = get_models()

class_names = [species_labels[i] for i in sorted(species_labels)]
num_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
num_feat_idx = [feature_names.index(f) for f in num_features]

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Model Selection")
model_type = st.sidebar.radio("Model Type", ["Decision Tree", "Logistic Regression"])
lam = st.sidebar.slider("λ (regularization penalty)", 0.0, 0.1, 0.01, step=0.001)

# Pick best model based on lambda
if model_type == "Decision Tree":
    best = max(tree_models.values(), key=lambda m: m["acc"] - lam * m["leaves"])
    selected_model = best["model"]
    complexity = best["leaves"]
    complexity_label = "Leaves"
else:
    best = max(lr_models.values(), key=lambda m: m["acc"] - lam * m["complexity"])
    selected_model = best["model"]
    complexity = best["complexity"]
    complexity_label = "Non-zero weights"

st.sidebar.markdown(f"**Test Accuracy:** {best['acc']:.3f}")
st.sidebar.markdown(f"**{complexity_label}:** {complexity}")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌳 Model Visualization", "🔄 Counterfactuals", "📈 Feature Effect Plots"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Model Visualization
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"{model_type} — λ={lam}")
    st.metric("Test Accuracy", f"{best['acc']:.3f}")
    st.metric(complexity_label, complexity)

    if model_type == "Decision Tree":
        fig, ax = plt.subplots(figsize=(22, 8))
        plot_tree(selected_model, ax=ax, feature_names=feature_names,
                  class_names=class_names, filled=True, rounded=True, fontsize=9)
        st.pyplot(fig)
        plt.close()
    else:
        st.subheader("Logistic Regression Coefficients")
        coef_df = pd.DataFrame(selected_model.coef_, columns=feature_names,
                               index=class_names)
        fig, ax = plt.subplots(figsize=(12, 4))
        coef_df.T.plot(kind='bar', ax=ax)
        ax.set_title("Coefficients per class")
        ax.set_ylabel("Coefficient value")
        ax.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Counterfactuals
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Counterfactual Explanations")

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    col1, col2 = st.columns(2)
    with col1:
        example_idx = st.selectbox("Select example (index)", range(len(X_all)))
    with col2:
        target_class = st.selectbox("Target class", list(species_labels.values()))

    x = X_all[example_idx]
    target_code = [k for k, v in species_labels.items() if v == target_class][0]
    current_pred = class_names[selected_model.predict(x.reshape(1, -1))[0]]

    st.markdown(f"**Current prediction:** {current_pred} → **Target:** {target_class}")

    # Identify feature types
    cat_idx = list(range(4, len(feature_names)))  # one-hot columns
    num_idx = list(range(4))                       # numerical columns

    # MAD for numerical features
    MAD = np.median(np.abs(X_train[:, num_idx] - np.median(X_train[:, num_idx], axis=0)), axis=0)
    MAD[MAD == 0] = 1

    def generate_counterfactuals(x, target_code, N=5000, sigma=0.5, k=3):
        samples = []
        for _ in range(N):
            x_new = x.copy().astype(float)
            # Numerical: add Gaussian noise
            x_new[num_idx] += np.random.normal(0, sigma, size=len(num_idx))
            # Categorical: randomly flip with prob 0.3
            for ci in cat_idx:
                if np.random.rand() < 0.3:
                    x_new[ci] = 1 - x_new[ci]
            samples.append(x_new)

        samples = np.array(samples)
        preds = selected_model.predict(samples)
        mask = preds == target_code
        candidates = samples[mask]

        if len(candidates) == 0:
            return None

        # MAD-weighted L1 distance (numerical only)
        dists = np.sum(np.abs(candidates[:, num_idx] - x[num_idx]) / MAD, axis=1)
        top_k = np.argsort(dists)[:k]
        return candidates[top_k], dists[top_k]

    if st.button("Generate Counterfactuals"):
        result = generate_counterfactuals(x, target_code)
        if result is None:
            st.warning("No counterfactuals found. Try a different example or target.")
        else:
            cfs, dists = result
            st.success(f"Found {len(cfs)} counterfactual(s)!")
            for i, (cf, d) in enumerate(zip(cfs, dists)):
                st.markdown(f"**Counterfactual {i+1}** — Distance: {d:.3f}")
                diff_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Original": x,
                    "Counterfactual": cf,
                    "Change": cf - x
                })
                diff_df = diff_df[diff_df["Change"].abs() > 1e-4]
                st.dataframe(diff_df.round(3), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Effect Plots (PDP + ALE)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Feature Effect Plots")
    selected_feature = st.selectbox("Select numerical feature", num_features)
    feat_idx = feature_names.index(selected_feature)

    def compute_pdp(model, X, feat_idx, grid_points=50):
        grid = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), grid_points)
        pdp_vals = []
        for val in grid:
            X_copy = X.copy()
            X_copy[:, feat_idx] = val
            pdp_vals.append(model.predict_proba(X_copy).mean(axis=0))
        return grid, np.array(pdp_vals)

    def compute_ale(model, X, feat_idx, n_bins=20):
        z = X[:, feat_idx]
        quantiles = np.percentile(z, np.linspace(0, 100, n_bins + 1))
        quantiles = np.unique(quantiles)
        n_bins = len(quantiles) - 1
        n_classes = len(class_names)
        ale_vals = np.zeros((n_bins, n_classes))

        for k in range(n_bins):
            mask = (z >= quantiles[k]) & (z <= quantiles[k + 1])
            if mask.sum() == 0:
                continue
            X_low = X[mask].copy(); X_low[:, feat_idx] = quantiles[k]
            X_high = X[mask].copy(); X_high[:, feat_idx] = quantiles[k + 1]
            ale_vals[k] = (model.predict_proba(X_high) - model.predict_proba(X_low)).mean(axis=0)

        ale_accum = np.cumsum(ale_vals, axis=0)
        ale_accum -= ale_accum.mean(axis=0)
        centers = (quantiles[:-1] + quantiles[1:]) / 2
        return centers, ale_accum

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Partial Dependence Plot (PDP)**")
        grid, pdp_vals = compute_pdp(selected_model, X_train, feat_idx)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in enumerate(class_names):
            ax.plot(grid, pdp_vals[:, i], label=name, color=colors[i])
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Average predicted probability")
        ax.set_title(f"PDP — {selected_feature}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Accumulated Local Effects (ALE)**")
        centers, ale_vals = compute_ale(selected_model, X_train, feat_idx)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in enumerate(class_names):
            ax.plot(centers, ale_vals[:, i], label=name, color=colors[i])
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("ALE effect")
        ax.set_title(f"ALE — {selected_feature}")
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
