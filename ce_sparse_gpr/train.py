from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from sklearn.model_selection import KFold
from torch.utils.data import random_split


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
        x = torch.as_tensor(x, dtype=dtype, device=device)

        if x.ndim != 2:
            raise ValueError(f"X[{i}] must be 2D, got shape {tuple(x.shape)}.")

        if x.shape[1] == 0:
            raise ValueError(f"X[{i}] has zero descriptor columns.")

        if not torch.isfinite(x).all():
            raise ValueError(f"X[{i}] contains NaN or Inf values.")

        X_prepared.append(x)

    y = torch.as_tensor(y, dtype=dtype, device=device)

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


def _save_best_model(model, train_x, train_y, model_path: str | Path) -> None:
    model.fit_c(train_x, train_y, build_uncertainty=True)
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
    device=torch.device("cpu"),
    dtype=torch.float64,
    model_path: str | Path = "best_sparse_atomic_gpr.pt",
    min_lr: float = 1e-4,
    print_every: int = 50,
    restore_best: bool = False,
):
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")

    if min_lr < 0.0:
        raise ValueError("min_lr must be non-negative.")

    if print_every <= 0:
        raise ValueError("print_every must be positive.")

    model = model.to(device)
    train_x, train_y = prepare_xy(train_x, train_y, device, dtype)
    valid_x, valid_y = prepare_xy(valid_x, valid_y, device, dtype)

    history = {
        "neg_log_like": [],
        "rmse_train": [],
        "rmse_valid": [],
        "lr": [],
    }

    best_rmse_valid = float("inf")
    best_epoch = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss = model.neg_log_like_loss(train_x, train_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became non-finite at epoch {epoch + 1}.")

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            model.fit_c(train_x, train_y, build_uncertainty=False)
            pred_train = model(train_x)
            pred_valid = model(valid_x)
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
                _save_best_model(model, train_x, train_y, model_path)

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
        model.fit_c(train_x, train_y, build_uncertainty=build_uncertainty)
        pred_test = model(test_x)
        rmse_test = rmse_metric(pred_test, test_y)

    return float(rmse_test.item())


def load_model_like(model, model_path, device=torch.device("cpu")):
    """
    Load a checkpoint using the same model class as a template instance.
    """
    return type(model)(model_path=model_path, device=device)


def train_kfold(
    train_x,
    train_y,
    test_x,
    test_y,
    optimizer,
    scheduler,
    model,
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

        print(
            f"Fold {fold + 1}: "
            f"best RMSE valid = {best_rmse_valid:.6f} "
            f"RMSE test = {rmse_test:.6f}"
        )

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
    rmse_test_values = torch.tensor(
        [r["rmse_test"] for r in fold_results],
        dtype=torch.float64,
    )

    summary = {
        "rmse_valid_mean": rmse_valid_values.mean().item(),
        "rmse_valid_std": rmse_valid_values.std(unbiased=False).item(),
        "rmse_test_mean": rmse_test_values.mean().item(),
        "rmse_test_std": rmse_test_values.std(unbiased=False).item(),
        "fold_results": fold_results,
    }

    print(
        "\nKFold summary: "
        f"RMSE valid = {summary['rmse_valid_mean']:.6f} "
        f"+/- {summary['rmse_valid_std']:.6f}; "
        f"RMSE test = {summary['rmse_test_mean']:.6f} "
        f"+/- {summary['rmse_test_std']:.6f}"
    )

    return summary
