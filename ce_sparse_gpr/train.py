from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import random_split

from .gpr import inv_softplus


BEST_MODEL_METRICS = ("rmse_valid", "rmse_valid_per_co", "loss")


def stratification_labels_from_n_co(n_co, n_splits: int) -> np.ndarray:

    n_co = np.asarray(n_co).astype(int)
    unique_vals, counts = np.unique(n_co, return_counts=True)
    order = np.argsort(unique_vals)
    unique_vals = unique_vals[order]
    counts = counts[order]

    buckets: list[list[int]] = []
    current_vals: list[int] = []
    current_count = 0
    for val, cnt in zip(unique_vals.tolist(), counts.tolist()):
        current_vals.append(val)
        current_count += cnt
        if current_count >= n_splits:
            buckets.append(current_vals)
            current_vals = []
            current_count = 0
    if current_vals:
        if buckets:
            buckets[-1].extend(current_vals)
        else:

            buckets.append(current_vals)

    val_to_bucket = {v: i for i, vals in enumerate(buckets) for v in vals}
    return np.array([val_to_bucket[v] for v in n_co.tolist()])

_LENGTHSCALE_RATIO = 20.0
_LENGTHSCALE_FALLBACK_SCALE = 1.0  # used only if x_M has a single row (std undefined)

_RAW_OUTPUTSCALE_MIN = float(inv_softplus(torch.tensor(1e-6)))
_RAW_OUTPUTSCALE_MAX = float(inv_softplus(torch.tensor(1e6)))
_RAW_SIGMA2_MIN = float(inv_softplus(torch.tensor(1e-6)))
_RAW_SIGMA2_MAX = float(inv_softplus(torch.tensor(1e6)))


def _clamp_hyperparameters(model) -> None:
    with torch.no_grad():
        x_M = model.x_M
        if x_M.shape[0] > 1:
            scale = x_M.std(dim=0)
        else:
            scale = torch.zeros_like(model.raw_lengthscale)

        degenerate = scale < 1e-6
        scale = torch.where(
            degenerate, torch.full_like(scale, _LENGTHSCALE_FALLBACK_SCALE), scale
        )

        raw_min = inv_softplus(scale / _LENGTHSCALE_RATIO)
        raw_max = inv_softplus(scale * _LENGTHSCALE_RATIO)
        model.raw_lengthscale.clamp_(raw_min, raw_max)

        model.raw_outputscale.clamp_(_RAW_OUTPUTSCALE_MIN, _RAW_OUTPUTSCALE_MAX)
        model.raw_sigma2.clamp_(_RAW_SIGMA2_MIN, _RAW_SIGMA2_MAX)


def split_dataset(
    dataset: torch.utils.data.Dataset,
    train_fraction: float = 0.8,
    seed: int = 42,
):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    if len(dataset) < 2:
        raise ValueError("dataset must contain at least two structures to split.")

    train_size = int(train_fraction * len(dataset))
    train_size = max(1, min(train_size, len(dataset) - 1))
    test_size = len(dataset) - train_size

    return random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )


def get_tensors_from_subset(subset):
    idx = list(subset.indices)
    X = [subset.dataset.X[i] for i in idx]
    y = subset.dataset.y[idx]
    return X, y


def select_by_indices(X, y, indices):
    indices = list(indices)
    X_part = [X[int(i)] for i in indices]
    y_part = y[indices]
    return X_part, y_part


def prepare_xy(X, y, device, dtype):
    if len(X) == 0:
        raise ValueError("X must contain at least one structure.")

    X_prepared = []
    for i, x in enumerate(X):
        x = torch.as_tensor(x, dtype=dtype).to(device)

        if x.ndim != 2:
            raise ValueError(f"X[{i}] must be 2D, got shape {tuple(x.shape)}.")

        if x.shape[1] == 0:
            raise ValueError(f"X[{i}] has zero descriptor columns.")

        if not torch.isfinite(x).all():
            raise ValueError(f"X[{i}] contains NaN or Inf values.")

        X_prepared.append(x)

    y = torch.as_tensor(y, dtype=dtype).to(device)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(1)

    if y.ndim != 1:
        raise ValueError(f"y must be 1D or shape (N, 1), got {tuple(y.shape)}.")

    if y.shape[0] != len(X_prepared):
        raise ValueError(f"len(X)={len(X_prepared)} but len(y)={y.shape[0]}.")

    if not torch.isfinite(y).all():
        raise ValueError("y contains NaN or Inf values.")

    return X_prepared, y


def rmse_metric(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch in RMSE: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}."
        )
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def rmse_metric_per_co(y_pred, y_true, n_co):

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch in RMSE/CO: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}."
        )
    if n_co.shape != y_pred.shape:
        raise ValueError(f"n_co shape {tuple(n_co.shape)} doesn't match y_pred shape {tuple(y_pred.shape)}.")
    if torch.any(n_co <= 0):
        raise ValueError("n_co must be positive (one adsorbed CO minimum).")
    return torch.sqrt(torch.mean(((y_pred - y_true) / n_co) ** 2))


def _save_best_model(
    model, train_x, train_y, model_path: str | Path, skip_validation: bool = False
) -> None:
    model.fit_c(train_x, train_y, build_uncertainty=True, skip_validation=skip_validation)
    model.save(model_path)


def _clone_training_objects(model, optimizer, scheduler):
    # deepcopy keeps optimizer parameter references consistent with the copied
    # model because the whole tuple is copied in a single operation.
    return deepcopy((model, optimizer, scheduler))


def _checkpoint_path(base_path: str | Path, suffix: str) -> str:
    path = Path(base_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))
    return str(path.with_name(f"{path.name}{suffix}.pt"))


def train(
    train_x,
    train_y,
    valid_x,
    valid_y,
    optimizer,
    scheduler,
    model,
    n_epochs: int = 500,
    device=None,
    dtype=torch.float64,
    model_path: str | Path = "best_sparse_atomic_gpr.pt",
    min_lr: float = 1e-4,
    print_every: int = 50,
    restore_best: bool = False,
    initial_best_rmse: float = float("inf"),
):
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")

    if min_lr < 0.0:
        raise ValueError("min_lr must be non-negative.")

    if print_every <= 0:
        raise ValueError("print_every must be positive.")

    # If device is not given, keep the model where it already lives.
    if device is not None:
        model = model.to(device)
    else:
        x_M = getattr(model, "x_M", None)
        device = x_M.device if x_M is not None else torch.device("cpu")

    train_x, train_y = prepare_xy(train_x, train_y, device, dtype)
    valid_x, valid_y = prepare_xy(valid_x, valid_y, device, dtype)

    history = {
        "neg_log_like": [],
        "rmse_train": [],
        "rmse_valid": [],
        "lr": [],
    }

    best_rmse_valid = float(initial_best_rmse)
    best_epoch = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss = model.neg_log_like_loss(train_x, train_y, skip_validation=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became non-finite at epoch {epoch + 1}.")

        loss.backward()
        optimizer.step()
        _clamp_hyperparameters(model)

        model.eval()

        with torch.no_grad():
            model.fit_c(train_x, train_y, build_uncertainty=False, skip_validation=True)
            pred_train = model(train_x, skip_validation=True)
            pred_valid = model(valid_x, skip_validation=True)
            rmse_train = rmse_metric(pred_train, train_y)
            rmse_valid = rmse_metric(pred_valid, valid_y)

        rmse_train_val = float(rmse_train.item())
        rmse_valid_val = float(rmse_valid.item())
        lr_val = float(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(rmse_valid_val)
            else:
                scheduler.step()
            lr_val = float(optimizer.param_groups[0]["lr"])

        history["neg_log_like"].append(float(loss.detach().item()))
        history["rmse_train"].append(rmse_train_val)
        history["rmse_valid"].append(rmse_valid_val)
        history["lr"].append(lr_val)

        if rmse_valid_val < best_rmse_valid:
            best_rmse_valid = rmse_valid_val
            best_epoch = epoch + 1
            with torch.no_grad():
                _save_best_model(model, train_x, train_y, model_path, skip_validation=True)

        if ((epoch + 1) % print_every == 0) or epoch == 0:
            print(
                f"Iter {epoch + 1}/{n_epochs} "
                f"Loss: {loss.detach().item():.6f} "
                f"RMSE train: {rmse_train_val:.6f} "
                f"RMSE valid: {rmse_valid_val:.6f} "
                f"best RMSE valid: {best_rmse_valid:.6f} "
                f"lr: {lr_val:.3e}"
            )

        if lr_val < min_lr:
            break

    history["best_epoch"] = best_epoch

    if restore_best and Path(model_path).exists():
        best_model = type(model)(model_path=model_path, device=device)
        model.load_state_dict(best_model.state_dict(), strict=False)

    return history, best_rmse_valid


def train_with_restarts(
    train_x,
    train_y,
    valid_x,
    valid_y,
    model,
    optimizer_factory,
    scheduler_factory=None,
    n_restarts: int = 3,
    min_improvement: float = 0.0,
    n_epochs: int = 500,
    device=None,
    dtype=torch.float64,
    model_path: str | Path = "best_sparse_atomic_gpr.pt",
    min_lr: float = 1e-4,
    print_every: int = 50,
):

    if n_restarts < 0:
        raise ValueError("n_restarts must be non-negative.")

    # Resolve once up front (like train() does) so load_model_like doesn't
    # silently default a GPU model back to CPU on the first restart.
    if device is None:
        x_M = getattr(model, "x_M", None)
        device = x_M.device if x_M is not None else torch.device("cpu")

    best_rmse = float("inf")
    all_history = []

    for cycle in range(n_restarts + 1):
        if cycle > 0 and Path(model_path).exists():
            best_model = load_model_like(model, model_path, device=device)
            model.load_state_dict(best_model.state_dict(), strict=False)

        optimizer = optimizer_factory(model)
        scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None

        print(f"\n=== Restart cycle {cycle + 1}/{n_restarts + 1} (best RMSE valid so far: {best_rmse:.6f}) ===")

        history, best_rmse_after = train(
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            n_epochs=n_epochs,
            device=device,
            dtype=dtype,
            model_path=model_path,
            min_lr=min_lr,
            print_every=print_every,
            initial_best_rmse=best_rmse,
        )
        all_history.append(history)

        improvement = best_rmse - best_rmse_after
        best_rmse = best_rmse_after

        if cycle > 0 and improvement <= min_improvement:
            print(f"No further improvement (delta={improvement:.3e} <= {min_improvement:.3e}), stopping restarts.")
            break

    return all_history, best_rmse


def train_lbfgs(
    train_x,
    train_y,
    valid_x,
    valid_y,
    model,
    n_steps: int = 20,
    max_iter: int = 20,
    lr: float = 1.0,
    history_size: int = 10,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    line_search_fn: str | None = "strong_wolfe",
    device=None,
    dtype=torch.float64,
    model_path: str | Path = "best_sparse_atomic_gpr.pt",
    print_every: int = 1,
    best_model_metric: str = "rmse_valid",
    loss_plateau_window: int | None = 5,
    loss_plateau_tol: float = 0.5,
    valid_n_co=None,
):

    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    if best_model_metric not in BEST_MODEL_METRICS:
        raise ValueError(f"best_model_metric must be one of {BEST_MODEL_METRICS}, got {best_model_metric!r}.")
    if best_model_metric == "rmse_valid_per_co" and valid_n_co is None:
        raise ValueError("best_model_metric='rmse_valid_per_co' needs valid_n_co.")
    if loss_plateau_window is not None and loss_plateau_window < 1:
        raise ValueError(f"loss_plateau_window must be >= 1 (or None to disable), got {loss_plateau_window}.")
    if loss_plateau_tol < 0:
        raise ValueError(f"loss_plateau_tol must be non-negative, got {loss_plateau_tol}.")
    use_per_co = best_model_metric == "rmse_valid_per_co"

    if device is not None:
        model = model.to(device)
    else:
        x_M = getattr(model, "x_M", None)
        device = x_M.device if x_M is not None else torch.device("cpu")

    train_x, train_y = prepare_xy(train_x, train_y, device, dtype)
    valid_x, valid_y = prepare_xy(valid_x, valid_y, device, dtype)

    if use_per_co:
        valid_n_co = torch.as_tensor(valid_n_co, dtype=dtype).to(device)
        if valid_n_co.shape != valid_y.shape:
            raise ValueError(
                f"valid_n_co shape {tuple(valid_n_co.shape)} doesn't match valid_y shape {tuple(valid_y.shape)}."
            )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn,
    )

    history = {"neg_log_like": [], "rmse_train": [], "rmse_valid": []}
    best_rmse_valid = float("inf")
    best_step = None

    def closure():

        _clamp_hyperparameters(model)
        optimizer.zero_grad(set_to_none=True)
        loss = model.neg_log_like_loss(train_x, train_y, skip_validation=True)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite inside the LBFGS closure.")
        loss.backward()
        return loss

    for step in range(n_steps):
        model.train()
        loss = optimizer.step(closure)
        _clamp_hyperparameters(model)  # defensive; closure already clamps every internal evaluation

        model.eval()
        with torch.no_grad():
            model.fit_c(train_x, train_y, build_uncertainty=False, skip_validation=True)
            pred_train = model(train_x, skip_validation=True)
            pred_valid = model(valid_x, skip_validation=True)
            rmse_train = rmse_metric(pred_train, train_y)
            rmse_valid = rmse_metric(pred_valid, valid_y)

            if use_per_co:
                rmse_valid_per_co = rmse_metric_per_co(pred_valid, valid_y, valid_n_co)
            if best_model_metric == "loss":

                post_step_loss = float(model.neg_log_like_loss(train_x, train_y, skip_validation=True).item())

        rmse_train_val = float(rmse_train.item())
        rmse_valid_val = float(rmse_valid.item())
        loss_val = float(loss.detach().item())
        if best_model_metric == "loss":
            selection_metric_val = post_step_loss
        elif use_per_co:
            selection_metric_val = float(rmse_valid_per_co.item())
        else:
            selection_metric_val = rmse_valid_val
        metric_name = {"loss": "loss", "rmse_valid_per_co": "RMSE valid/CO", "rmse_valid": "RMSE valid"}[best_model_metric]

        history["neg_log_like"].append(loss_val)
        history["rmse_train"].append(rmse_train_val)
        history["rmse_valid"].append(rmse_valid_val)

        if selection_metric_val < best_rmse_valid:
            best_rmse_valid = selection_metric_val
            best_step = step + 1
            with torch.no_grad():
                _save_best_model(model, train_x, train_y, model_path, skip_validation=True)

        if ((step + 1) % print_every == 0) or step == 0:
            print(
                f"LBFGS step {step + 1}/{n_steps} "
                f"Loss: {loss_val:.6f} "
                f"RMSE train: {rmse_train_val:.6f} "
                f"RMSE valid: {rmse_valid_val:.6f} "
                f"best {metric_name}: {best_rmse_valid:.6f}"
            )

        if loss_plateau_window is not None and step >= loss_plateau_window:
            loss_decrease = history["neg_log_like"][step - loss_plateau_window] - loss_val
            if loss_decrease < loss_plateau_tol:
                print(
                    f"Loss dropped by only {loss_decrease:.6g} (< loss_plateau_tol={loss_plateau_tol}) "
                    f"over the last {loss_plateau_window} steps, stopping "
                    f"(best {metric_name}={best_rmse_valid:.6f} at step {best_step})."
                )
                break

    history["best_step"] = best_step

    if best_model_metric == "loss" and best_step is not None:
        return history, history["rmse_valid"][best_step - 1]
    return history, best_rmse_valid


def evaluate(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    device=torch.device("cpu"),
    dtype=torch.float64,
    build_uncertainty: bool = False,
):
    model = model.to(device)
    model.eval()

    train_x, train_y = prepare_xy(train_x, train_y, device, dtype)
    test_x, test_y = prepare_xy(test_x, test_y, device, dtype)

    with torch.no_grad():
        model.fit_c(train_x, train_y, build_uncertainty=build_uncertainty, skip_validation=True)
        pred_test = model(test_x, skip_validation=True)
        rmse_test = rmse_metric(pred_test, test_y)

    return float(rmse_test.item())


def load_model_like(model, model_path, device=torch.device("cpu")):

    return type(model)(model_path=model_path, device=device)


def train_kfold(
    train_x,
    train_y,
    optimizer,
    scheduler,
    model,
    test_x=None,
    test_y=None,
    n_epochs: int = 500,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    print_every: int = 50,
    device=torch.device("cpu"),
    dtype=torch.float64,
    model_path: str | Path = "best_sparse_atomic_gpr.pt",
    min_lr: float = 1e-4,
    evaluate_best_checkpoint: bool = True,
):

    if (test_x is None) != (test_y is None):
        raise ValueError("test_x and test_y must be both given or both omitted.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if n_splits > len(train_x):
        raise ValueError("n_splits cannot exceed the number of training structures.")

    train_y = torch.as_tensor(train_y)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed if shuffle else None,
    )

    fold_results = []
    indices = list(range(len(train_x)))

    for fold, (fold_train_idx, fold_valid_idx) in enumerate(kfold.split(indices)):
        print(f"\nFold {fold + 1}/{n_splits}")

        fold_model, fold_optimizer, fold_scheduler = _clone_training_objects(
            model,
            optimizer,
            scheduler,
        )

        fold_train_x, fold_train_y = select_by_indices(train_x, train_y, fold_train_idx)
        fold_valid_x, fold_valid_y = select_by_indices(train_x, train_y, fold_valid_idx)

        fold_model_path = _checkpoint_path(model_path, f"_fold_{fold + 1}")

        history, best_rmse_valid = train(
            train_x=fold_train_x,
            train_y=fold_train_y,
            valid_x=fold_valid_x,
            valid_y=fold_valid_y,
            optimizer=fold_optimizer,
            scheduler=fold_scheduler,
            model=fold_model,
            n_epochs=n_epochs,
            device=device,
            dtype=dtype,
            model_path=fold_model_path,
            min_lr=min_lr,
            print_every=print_every,
        )

        rmse_test = None
        if test_x is not None:
            eval_model = fold_model
            if evaluate_best_checkpoint and Path(fold_model_path).exists():
                eval_model = load_model_like(fold_model, fold_model_path, device=device)

            rmse_test = evaluate(
                model=eval_model,
                train_x=fold_train_x,
                train_y=fold_train_y,
                test_x=test_x,
                test_y=test_y,
                device=device,
                dtype=dtype,
            )

        test_msg = f"RMSE test = {rmse_test:.6f}" if rmse_test is not None else "(no test set given)"
        print(f"Fold {fold + 1}: best RMSE valid = {best_rmse_valid:.6f} {test_msg}")

        fold_results.append(
            {
                "fold": fold + 1,
                "history": history,
                "best_rmse_valid": best_rmse_valid,
                "rmse_test": rmse_test,
                "train_idx": fold_train_idx,
                "valid_idx": fold_valid_idx,
                "model_path": fold_model_path,
            }
        )

    rmse_valid_values = torch.tensor(
        [r["best_rmse_valid"] for r in fold_results],
        dtype=torch.float64,
    )

    summary = {
        "rmse_valid_mean": rmse_valid_values.mean().item(),
        "rmse_valid_std": rmse_valid_values.std(unbiased=False).item(),
        "fold_results": fold_results,
    }

    summary_msg = (
        "\nKFold summary: "
        f"RMSE valid = {summary['rmse_valid_mean']:.6f} +/- {summary['rmse_valid_std']:.6f}"
    )

    if test_x is not None:
        rmse_test_values = torch.tensor(
            [r["rmse_test"] for r in fold_results],
            dtype=torch.float64,
        )
        summary["rmse_test_mean"] = rmse_test_values.mean().item()
        summary["rmse_test_std"] = rmse_test_values.std(unbiased=False).item()
        summary_msg += (
            f"; RMSE test = {summary['rmse_test_mean']:.6f} +/- {summary['rmse_test_std']:.6f}"
        )

    print(summary_msg)

    return summary


def train_kfold_lbfgs(
    train_x,
    train_y,
    model,
    test_x=None,
    test_y=None,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    n_steps: int = 20,
    max_iter: int = 20,
    lr: float = 1.0,
    print_every: int = 1,
    device=torch.device("cpu"),
    dtype=torch.float64,
    model_path: str | Path = "best_sparse_atomic_gpr.pt",
    evaluate_best_checkpoint: bool = True,
    max_folds: int | None = None,
    n_co=None,
    **lbfgs_kwargs,
):

    if (test_x is None) != (test_y is None):
        raise ValueError("test_x and test_y must be both given or both omitted.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if n_splits > len(train_x):
        raise ValueError("n_splits cannot exceed the number of training structures.")

    train_y = torch.as_tensor(train_y)

    if n_co is not None:
        n_co = torch.as_tensor(n_co, dtype=dtype)
        if n_co.shape != train_y.shape:
            raise ValueError(f"n_co shape {tuple(n_co.shape)} doesn't match train_y shape {tuple(train_y.shape)}.")

    indices = list(range(len(train_x)))

    if n_co is not None:

        strat_labels = stratification_labels_from_n_co(n_co.detach().cpu().numpy(), n_splits)
        kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )
        split_iter = kfold.split(indices, strat_labels)
    else:
        kfold = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )
        split_iter = kfold.split(indices)

    fold_results = []

    for fold, (fold_train_idx, fold_valid_idx) in enumerate(split_iter):
        if max_folds is not None and fold >= max_folds:
            break

        print(f"\nFold {fold + 1}/{n_splits}")

        fold_model = deepcopy(model)

        fold_train_x, fold_train_y = select_by_indices(train_x, train_y, fold_train_idx)
        fold_valid_x, fold_valid_y = select_by_indices(train_x, train_y, fold_valid_idx)
        fold_valid_n_co = n_co[fold_valid_idx] if n_co is not None else None

        fold_model_path = _checkpoint_path(model_path, f"_fold_{fold + 1}")

        history, best_rmse_valid = train_lbfgs(
            train_x=fold_train_x,
            train_y=fold_train_y,
            valid_x=fold_valid_x,
            valid_y=fold_valid_y,
            model=fold_model,
            n_steps=n_steps,
            max_iter=max_iter,
            lr=lr,
            device=device,
            dtype=dtype,
            model_path=fold_model_path,
            print_every=print_every,
            valid_n_co=fold_valid_n_co,
            **lbfgs_kwargs,
        )

        rmse_test = None
        if test_x is not None:
            eval_model = fold_model
            if evaluate_best_checkpoint and Path(fold_model_path).exists():
                eval_model = load_model_like(fold_model, fold_model_path, device=device)

            rmse_test = evaluate(
                model=eval_model,
                train_x=fold_train_x,
                train_y=fold_train_y,
                test_x=test_x,
                test_y=test_y,
                device=device,
                dtype=dtype,
            )

        test_msg = f"RMSE test = {rmse_test:.6f}" if rmse_test is not None else "(no test set given)"
        print(f"Fold {fold + 1}: best RMSE valid = {best_rmse_valid:.6f} {test_msg}")

        fold_results.append(
            {
                "fold": fold + 1,
                "history": history,
                "best_rmse_valid": best_rmse_valid,
                "rmse_test": rmse_test,
                "train_idx": fold_train_idx,
                "valid_idx": fold_valid_idx,
                "model_path": fold_model_path,
            }
        )

    rmse_valid_values = torch.tensor(
        [r["best_rmse_valid"] for r in fold_results],
        dtype=torch.float64,
    )

    summary = {
        "rmse_valid_mean": rmse_valid_values.mean().item(),
        "rmse_valid_std": rmse_valid_values.std(unbiased=False).item(),
        "fold_results": fold_results,
    }

    summary_msg = (
        "\nKFold summary (LBFGS): "
        f"RMSE valid = {summary['rmse_valid_mean']:.6f} +/- {summary['rmse_valid_std']:.6f}"
    )

    if test_x is not None:
        rmse_test_values = torch.tensor(
            [r["rmse_test"] for r in fold_results],
            dtype=torch.float64,
        )
        summary["rmse_test_mean"] = rmse_test_values.mean().item()
        summary["rmse_test_std"] = rmse_test_values.std(unbiased=False).item()
        summary_msg += (
            f"; RMSE test = {summary['rmse_test_mean']:.6f} +/- {summary['rmse_test_std']:.6f}"
        )

    print(summary_msg)

    return summary
