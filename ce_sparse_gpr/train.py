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
    """Bucket-label each structure by its (possibly merged) n_co, for
    StratifiedKFold - not the raw n_co itself, because a class with fewer
    than n_splits members (e.g. n_co=9 with a single example in
    ads_unified.db) makes StratifiedKFold.split raise outright ("least
    populated class ... too few members").

    Adjacent n_co values are greedily merged (lowest first) into buckets of
    at least n_splits members each - any leftover too-small remainder at the
    top end joins the last sealed bucket. For ads_unified.db's distribution
    {1:481, 4:87, 5:92, 6:72, 7:41, 8:30, 9:1} with n_splits=5 this yields
    buckets {1},{4},{5},{6},{7},{8,9}: n_co=9's lone example rides along
    with its nearest coverage neighbor instead of crashing the split. This
    only affects which fold a structure lands in - its actual n_co value
    (used elsewhere for the per-CO RMSE metric/target) is untouched.
    """
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
            # Every class pooled together still has fewer than n_splits
            # structures total - StratifiedKFold cannot possibly succeed
            # here regardless of labeling; let it raise its own clear error
            # downstream rather than silently returning something bogus.
            buckets.append(current_vals)

    val_to_bucket = {v: i for i, vals in enumerate(buckets) for v in vals}
    return np.array([val_to_bucket[v] for v in n_co.tolist()])

# How far raw_lengthscale (pre-softplus) is allowed to stray from each
# dimension's own natural scale (std of that descriptor over the inducing
# points), applied after every optimizer step/closure call. A *fixed*
# absolute bound (an earlier version used [1e-3, 1e3] globally, back when
# lengthscale was exp(log_lengthscale)) is wrong for descriptors whose real
# spread is O(1-20): at lengthscale=1000, the RBF kernel value across the
# ENTIRE real data range for such a dimension differs by <2e-4 - the
# dimension is fully collapsed to a constant, and if several dimensions
# collapse at once, K_MM/K_NM become near-rank-1 and no jitter fixes that.
# ratio=20 keeps the kernel's value spread over +/-2 std at roughly
# [0.98, 1.0] - still lets the optimizer down-weight an uninformative
# dimension a lot, without fully collapsing it. (Since gpr.py switched
# lengthscale/sigma2/outputscale from exp(raw) to softplus(raw) - which grows
# only linearly, not exponentially, for large raw - a single bad optimizer
# step can no longer blow these up by many orders of magnitude the way it
# could before; these clamps are now a secondary safety net, not the primary
# defense.)
_LENGTHSCALE_RATIO = 20.0
_LENGTHSCALE_FALLBACK_SCALE = 1.0  # used only if x_M has a single row (std undefined)

# outputscale is the kernel's overall prior-variance multiplier and sigma2 is
# the noise variance - both can still drift a long way in raw-space even
# under softplus, so keep them bounded too. [1e-6, 1e6] is generous either
# for raw unnormalized DFT energies or for standardized targets. sigma2's
# floor is 1e-6, not 1e-8: a near-perfect fold fit pushes sigma2 toward 0,
# and safe_cholesky's jitter cap is itself a fraction of sigma2 - at 1e-8
# that budget becomes razor-thin right when a good fit makes it most likely
# to be needed. 1e-6 keeps a workable jitter budget without forcing a
# noticeably noisier fit than 1e-8 would have.
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

        # A descriptor dimension that is (numerically) constant across the
        # inducing points - e.g. a "center is element X" indicator when every
        # structure is centered on that element, as in the rep pipeline's
        # "single atom: C" feature - has no data-driven scale to normalize
        # by. clamp_min(1e-6) alone treats that 0 as if it were a real, tiny
        # scale and forces the lengthscale itself down near it (~[5e-8, 2e-5]
        # at ratio=20) - which then amplifies ordinary float64 round-off
        # between inducing points that should be exactly equal along that
        # dimension into a large scaled distance, corrupting K_MM (observed:
        # min eigenvalue going slightly negative). Falling back to a large,
        # non-binding scale for degenerate dimensions avoids that instead of
        # causing it; every non-degenerate dimension is completely unaffected.
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
    """RMSE of (pred-true)/n_co per structure, instead of the raw residual -
    used as the checkpoint-selection metric (see train_lbfgs's
    selection_metric) on _datasets whose target spans a wide n_co range (e.g.
    ads_unified's e_ads_total: ~-1.6 eV at n_co=1 up to ~-10 eV at n_co=8).
    A plain whole-structure RMSE is implicitly dominated by the high-coverage
    tail simply because its errors live on a bigger absolute scale, not
    because the model is actually worse there per adsorbed molecule -
    dividing each structure's error by its OWN n_co first puts every CO on
    comparable footing regardless of how many share a structure."""
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
    """Run train() for up to n_restarts+1 cycles, each with a freshly built
    optimizer/scheduler (via the factories), reloading the best checkpoint
    found so far before every cycle after the first.

    This is the "reset optimizer and scheduler" workflow automated: once
    ReduceLROnPlateau bottoms out at min_lr, Adam's accumulated moment
    estimates are usually stale for the local landscape near the current
    optimum, and simply continuing training rarely helps further. Reloading
    the best weights and starting a brand new optimizer/scheduler (fresh
    moments, fresh LR, fresh patience counter) often finds more improvement.

    optimizer_factory: callable(model) -> optimizer, e.g.
        lambda m: torch.optim.Adam(m.parameters(), lr=1e-1)
    scheduler_factory: callable(optimizer) -> scheduler, or None, e.g.
        lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)
    n_restarts: number of *additional* cycles after the first (n_restarts=3 -> up to 4 cycles total)
    min_improvement: stop restarting early once a cycle improves best RMSE
        (valid) by less than this amount
    """
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
    """Fit GP hyperparameters (lengthscale/sigma2/outputscale) with L-BFGS
    instead of Adam + a LR scheduler.

    The NLL surface here is smooth and low-dimensional (a handful to a few
    dozen free parameters, full-batch, no stochasticity) - exactly the regime
    quasi-Newton methods are built for, which is why GPML/GPflow/sklearn all
    default to L-BFGS-B for GP hyperparameter fitting rather than an
    SGD-family optimizer. Each optimizer.step(closure) call already runs its
    own internal line search (up to max_iter evaluations), so this typically
    needs far fewer outer iterations than Adam and no LR schedule or restart
    tricks - a low or oscillating step size from the line search doesn't mean
    the fit is stuck the way a decayed Adam LR does.

    n_steps: number of optimizer.step(closure) calls (outer iterations)
    max_iter: LBFGS's own per-step cap on internal line-search iterations
    loss_plateau_window, loss_plateau_tol: STOP once the LOSS has plateaued -
        it dropped by less than loss_plateau_tol over the last
        loss_plateau_window steps as a whole, i.e. compare loss now against
        loss loss_plateau_window steps ago (not against a running best).
        loss_plateau_window=None disables the check (always runs n_steps).
        Stopping is ALWAYS on the loss, whatever best_model_metric is: the
        loss is what LBFGS actually optimizes and is far smoother than any
        held-out metric (a small valid fold evaluated mid-optimization
        bounced 0.190->0.155->0.194->0.235 step to step while still trending
        down - stopping on that noise risks quitting right before a real
        improvement). A plain "no NEW best in N steps" check almost never
        triggers under LBFGS here, since the loss keeps crawling down by
        vanishingly small amounts long after any practically useful progress
        has stopped (observed: rep_linear_mean's fold 1 still improved loss
        by ~0.01-0.02 per step at step 990/1000). Many tiny "technically an
        improvement" steps each fail to beat the previous best individually,
        yet sum to less than loss_plateau_tol over the window, so it stops.
        loss_plateau_tol is on the loss' own absolute scale (not relative) -
        pick it per problem, the same way div/lr already are.
    best_model_metric: which step's model gets SAVED as the best checkpoint
        (independent of when training stops):
        "rmse_valid"        lowest RMSE on the held-out valid set (default).
        "rmse_valid_per_co" lowest RMSE of (pred-true)/n_co on valid (needs
            valid_n_co; see rmse_metric_per_co). Use on a target whose scale
            grows with n_co (e.g. ads_unified's e_ads_total: ~-1.6 eV at
            n_co=1 vs ~-10 eV at n_co=8) so the "best" checkpoint is the one
            most accurate per adsorbed CO, not just the one that fits the
            high-coverage (large-absolute-error) tail best.
        "loss"              lowest training loss (NLL), evaluated at the
            model's CURRENT parameters after each step. Uses no held-out
            data at all - meant for active learning, where the valid fold is
            tiny/unrepresentative right after a new point is added.
        The returned best_rmse_valid is that metric's best value, except for
        "loss", where it is the raw RMSE valid AT the saved (lowest-loss)
        step - a loss value is not comparable across folds/datasets and
        callers rank folds by held-out error.
    valid_n_co: one entry per valid_x structure; required (and only used)
        when best_model_metric="rmse_valid_per_co".
    """
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
        # LBFGS's own internal line search calls this closure several times
        # per step() at points *it* picks - clamp before every evaluation
        # (not just after step() returns), or the line search can probe a
        # pathological lengthscale and crash neg_log_like_loss's Cholesky
        # before we ever get control back.
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
                # Loss at the parameters we are about to checkpoint - the
                # `loss` returned by optimizer.step is the value at the START
                # of the step (before the parameter update), so it would
                # score the previous step's weights.
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

        # Windowed loss plateau (see loss_plateau_window/loss_plateau_tol).
        # Needs loss_plateau_window+1 loss values on hand.
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
    """
    Load a checkpoint using the same model class as a template instance.
    """
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
    """test_x/test_y are optional: pass both to get an independent RMSE per
    fold (rmse_test_mean/std in the summary); leave both None to skip that and
    rely on the cross-validated rmse_valid_mean/std alone as the generalization
    estimate - a reasonable substitute for a held-out test set when data is too
    scarce to sacrifice a chunk of it permanently."""
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
    """Same as train_kfold, but each fold is fit with train_lbfgs instead of
    Adam + a scheduler. No optimizer/scheduler arguments needed - L-BFGS is
    constructed fresh per fold (bound to that fold's own model copy) inside
    train_lbfgs, so there's no optimizer/scheduler tuple to clone.

    test_x/test_y are optional: pass both for an independent per-fold RMSE
    (rmse_test_mean/std in the summary); leave both None to rely on
    rmse_valid_mean/std alone as the generalization estimate.

    max_folds: train/evaluate only the first this-many folds instead of all
    n_splits - the KFold split itself is unaffected (n_splits still sets the
    train/valid size ratio and which indices land in fold 1, 2, ...), so this
    is for cheaply getting a single representative holdout run (max_folds=1)
    without paying for n_splits full LBFGS fits, e.g. while a dataset is too
    small/imbalanced (a handful of examples in some rare stratum) for a full
    K-fold summary to be a meaningful average in the first place.

    n_co: optional, one entry per train_x structure (same order/length).
    Two effects: (1) the K-fold split is stratified by (bucketed) n_co
    instead of plain-shuffled - see stratification_labels_from_n_co; (2) it
    is sliced per fold and passed to train_lbfgs as valid_n_co, which uses it
    only for best_model_metric="rmse_valid_per_co" (that metric REQUIRES
    n_co here). None (default): ordinary unstratified KFold.

    best_model_metric, loss_plateau_window, loss_plateau_tol: forwarded to
    train_lbfgs via **lbfgs_kwargs - see its docstring."""
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
        # Stratify by (bucketed) n_co instead of a plain shuffle: an
        # unstratified split's fold composition swings a lot with dataset
        # heterogeneity (e.g. this project's ads_unified.db, where growing
        # the dataset by a single active-learning point reshuffles every
        # fold boundary and produced visibly different-looking training
        # trajectories run to run) - stratifying keeps each fold's coverage
        # mix comparable, so fold-to-fold variance reflects the model, not
        # which structures happened to land in which split.
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
