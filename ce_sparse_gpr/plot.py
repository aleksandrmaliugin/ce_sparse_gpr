from __future__ import annotations

import numpy as np
import torch
import plotly.graph_objects as go


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_1d_numpy(x, name: str) -> np.ndarray:
    arr = to_numpy(x).reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values.")

    return arr


def predict_structures(model, X_structures, return_std: bool = True):
    model.eval()

    with torch.no_grad():
        if return_std:
            try:
                pred, std = model.predict_uncertainty(X_structures)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Could not compute uncertainty. Make sure the model was fitted "
                    "with fit_c(..., build_uncertainty=True), or call plotting with "
                    "return_std=False."
                ) from exc
            return _as_1d_numpy(pred, "prediction"), _as_1d_numpy(std, "std")

        pred = model(X_structures)
        return _as_1d_numpy(pred, "prediction"), None


def mae_metric_np(y_pred, y_true) -> float:
    y_pred = _as_1d_numpy(y_pred, "y_pred")
    y_true = _as_1d_numpy(y_true, "y_true")

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}.")

    return float(np.mean(np.abs(y_pred - y_true)))


def rmse_metric_np(y_pred, y_true) -> float:
    y_pred = _as_1d_numpy(y_pred, "y_pred")
    y_true = _as_1d_numpy(y_true, "y_true")

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}.")

    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _add_prediction_trace(fig, name, y_pred, y_true, y_std, marker_size: int = 12):
    y_pred = _as_1d_numpy(y_pred, f"{name} y_pred")
    y_true = _as_1d_numpy(y_true, f"{name} y_true")

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"{name}: prediction/target shape mismatch: {y_pred.shape} vs {y_true.shape}."
        )

    if y_std is not None:
        y_std = _as_1d_numpy(y_std, f"{name} y_std")
        if y_std.shape != y_pred.shape:
            raise ValueError(
                f"{name}: std/prediction shape mismatch: {y_std.shape} vs {y_pred.shape}."
            )

    rmse = rmse_metric_np(y_pred, y_true)
    mae = mae_metric_np(y_pred, y_true)

    trace_kwargs = dict(
        x=y_pred,
        y=y_true,
        mode="markers",
        name=f"{name}: RMSE = {rmse * 1000:.2f} meV, MAE = {mae * 1000:.2f} meV",
        marker=dict(size=marker_size),
    )

    if y_std is not None:
        trace_kwargs["error_x"] = dict(type="data", array=y_std, visible=True)

    fig.add_trace(go.Scatter(**trace_kwargs))
    return rmse, mae


def plot_results(
    model,
    train_x,
    train_y,
    valid_x,
    valid_y,
    test_x=None,
    test_y=None,
    return_std: bool = True,
    save_plot: bool = False,
    filename: str = "gpr_accuracy.pdf",
):
    model.eval()

    train_y_np = _as_1d_numpy(train_y, "train_y")
    valid_y_np = _as_1d_numpy(valid_y, "valid_y")

    train_pred, train_std = predict_structures(model, train_x, return_std=return_std)
    valid_pred, valid_std = predict_structures(model, valid_x, return_std=return_std)

    arrays_for_limits = [train_pred, valid_pred, train_y_np, valid_y_np]

    test_pred = None
    test_std = None
    test_y_np = None

    if (test_x is None) != (test_y is None):
        raise ValueError("test_x and test_y must either both be provided or both be None.")

    if test_x is not None:
        test_y_np = _as_1d_numpy(test_y, "test_y")
        test_pred, test_std = predict_structures(model, test_x, return_std=return_std)
        arrays_for_limits.extend([test_pred, test_y_np])

    xy_values = np.concatenate([np.asarray(a).reshape(-1) for a in arrays_for_limits])
    xy_min = float(np.min(xy_values))
    xy_max = float(np.max(xy_values))

    if xy_min == xy_max:
        pad = max(1e-6, abs(xy_min) * 1e-3)
        xy_min -= pad
        xy_max += pad

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.linspace(xy_min, xy_max, 10),
            y=np.linspace(xy_min, xy_max, 10),
            mode="lines",
            line=dict(color="grey"),
            showlegend=False,
        )
    )

    rmse_train, mae_train = _add_prediction_trace(
        fig, "Train", train_pred, train_y_np, train_std, marker_size=12
    )
    rmse_valid, mae_valid = _add_prediction_trace(
        fig, "Valid", valid_pred, valid_y_np, valid_std, marker_size=12
    )

    metrics = {
        "rmse_train": rmse_train,
        "rmse_valid": rmse_valid,
        "mae_train": mae_train,
        "mae_valid": mae_valid,
    }

    if test_pred is not None and test_y_np is not None:
        rmse_test, mae_test = _add_prediction_trace(
            fig, "Test", test_pred, test_y_np, test_std, marker_size=12
        )
        metrics["rmse_test"] = rmse_test
        metrics["mae_test"] = mae_test

    fig.update_xaxes(
        title="E<sub>model</sub>, eV",
        title_font=dict(size=25),
        tickfont=dict(size=22),
        automargin=True,
    )
    fig.update_yaxes(
        title="E<sub>DFT</sub>, eV",
        title_font=dict(size=25),
        tickfont=dict(size=22),
        automargin=True,
    )
    fig.update_layout(
        width=750,
        height=750,
        margin=dict(r=50, t=50, pad=4),
        font=dict(family="Arial", size=20),
        legend=dict(x=0.02, y=0.98),
    )

    if save_plot:
        fig.write_image(filename)

    fig.show()
    return fig, metrics
