from __future__ import annotations

import numpy as np
import torch
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _stack_descriptor_rows(x_list, name: str) -> np.ndarray:
    if x_list is None:
        raise ValueError(f"{name} is None.")

    rows = [_to_numpy(x) for x in x_list]

    if len(rows) == 0:
        raise ValueError(f"{name} is empty.")

    for i, x in enumerate(rows):
        if x.ndim != 2:
            raise ValueError(f"{name}[{i}] must be 2D, got shape {x.shape}.")

    X = np.concatenate(rows, axis=0)

    if X.shape[0] == 0:
        raise ValueError(f"{name} contains no descriptor rows.")

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or Inf values.")

    return X


def _subsample(X: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max <= 0:
        raise ValueError("n_max must be positive.")

    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        idx = rng.choice(len(X), n_max, replace=False)
        return X[idx]
    return X


def _embed_2d(X_scaled: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    n_samples, n_features = X_scaled.shape

    if n_samples < 2:
        return np.zeros((n_samples, 2), dtype=float), "constant"

    n_pca = min(30, n_features, n_samples - 1)
    if n_pca < 1:
        X_pca = np.zeros((n_samples, 1), dtype=float)
    else:
        X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X_scaled)

    if n_samples < 4:
        X_emb = np.zeros((n_samples, 2), dtype=float)
        cols = min(2, X_pca.shape[1])
        X_emb[:, :cols] = X_pca[:, :cols]
        return X_emb, "PCA"

    perplexity = min(30, max(2, (n_samples - 1) // 3))
    perplexity = min(perplexity, n_samples - 1)

    X_emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X_pca)

    return X_emb, "t-SNE"


def plot_descriptor_space(
    model,
    train_x,
    valid_x,
    test_x=None,
    n_train_max: int = 1000,
    n_valid_max: int = 500,
    n_test_max: int = 500,
    seed: int = 0,
):
    if getattr(model, "x_M", None) is None:
        raise ValueError("model.x_M is missing.")

    X_M = _to_numpy(model.x_M)
    if X_M.ndim != 2:
        raise ValueError(f"model.x_M must be 2D, got shape {X_M.shape}.")

    X_train = _subsample(_stack_descriptor_rows(train_x, "train_x"), n_train_max, seed)
    X_valid = _subsample(_stack_descriptor_rows(valid_x, "valid_x"), n_valid_max, seed + 1)

    X_parts = [X_train, X_valid]
    label_parts = [
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_valid), dtype=int),
    ]

    trace_specs = [
        (0, "train atoms", 5, "circle", 0.45),
        (1, "valid atoms", 7, "circle", 0.75),
    ]

    if test_x is not None:
        X_test = _subsample(_stack_descriptor_rows(test_x, "test_x"), n_test_max, seed + 2)
        X_parts.append(X_test)
        label_parts.append(2 * np.ones(len(X_test), dtype=int))
        x_m_label = 3
        trace_specs.append((2, "test atoms", 7, "diamond", 0.75))
    else:
        x_m_label = 2

    X_parts.append(X_M)
    label_parts.append(x_m_label * np.ones(len(X_M), dtype=int))
    trace_specs.append((x_m_label, "x_M inducing atoms", 12, "x", 1.0))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)

    X_scaled = StandardScaler().fit_transform(X)
    X_emb, method = _embed_2d(X_scaled, seed=seed)

    fig = go.Figure()

    for label_id, name, size, symbol, opacity in trace_specs:
        mask = labels == label_id
        fig.add_trace(
            go.Scatter(
                x=X_emb[mask, 0],
                y=X_emb[mask, 1],
                mode="markers",
                name=name,
                marker=dict(size=size, symbol=symbol, opacity=opacity),
            )
        )

    title = f"Descriptor space: train / valid / inducing points ({method})"
    if test_x is not None:
        title = f"Descriptor space: train / valid / test / inducing points ({method})"

    fig.update_layout(
        width=800,
        height=650,
        title=title,
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2",
        font=dict(family="Arial", size=18),
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()
    return fig, X_emb, labels
