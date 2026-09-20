"""Config-driven training entry point for SparseAtomicGPR models.

Same pipeline as clean.ipynb (load ase.db -> CEConfig -> CEDataset ->
SparseAtomicGPR -> train -> reload best checkpoint -> parity plot), but driven
by a single JSON config instead of notebook cells, in the style of a typical
MLIP training CLI (nequip-train/mace_run_train etc.): one config in, a
self-contained run directory out (checkpoint + resolved config + training log
+ metrics + parity plot), so a run is fully reproducible and comparable
against other runs by just diffing configs.

Usage:
    python -m ce_sparse_gpr.ce_gpr_train config.json

See ce_gpr_train.example.json for a config matching clean.ipynb's current
settings (LBFGS, M=450, div=0.6) as a starting point; ce_gpr_train.ads.example.json
and ce_gpr_train.rep.example.json mirror low_cov.ipynb and rep.ipynb the same
way. Those two need descriptor.atom_indices (see build_atom_indices) to
restrict each structure's descriptor rows to its CO site(s) instead of every
atom - "near_carbon_metals" + aggregate:true for a single site's own energy
(low_cov/ads), "carbon_atoms" + aggregate:false for one row per adsorbed CO,
additively combined into a multi-CO repulsion signal (rep). Leave
descriptor.atom_indices null (or omit it) for the whole-slab case, where every
atom should keep contributing its own row.

No held-out test set is required: K-fold methods (kfold_adam/kfold_lbfgs)
create their own train/valid split per fold internally, so the full dataset
under "dataset" is used for training as-is. "test_dataset" is a stub for a
genuinely separate, never-trained-on test set - leave its db_path null until
one exists; every test_x/test_y downstream (K-fold's rmse_test, the parity
plot's Test trace) is skipped cleanly while it's absent.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from ase.db import connect

from ce_sparse_gpr import CEConfig, CEDataset, SparseAtomicGPR, atoms_near_carbon, calc_mindist, plot_results
from ce_sparse_gpr.plot import mae_metric_np_per_co, rmse_metric_np_per_co
from ce_sparse_gpr.train import (
    select_by_indices,
    train,
    train_kfold,
    train_kfold_lbfgs,
    train_lbfgs,
)


class Tee:
    """Writes to multiple streams at once (used to mirror stdout into a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _json_sanitize(obj):
    """Recursively convert numpy/torch/Path values to plain JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _usable_for_per_co_metric(n_co_values) -> bool:
    """True only if every row carries a genuine adsorbate count (>0).

    n_co is None where the key is simply absent (e.g. rep.db before
    make_rep_anchors.py), but clean/slab _datasets (e.g. clean.db) instead
    store an explicit n_co=0 on every row - not None, but still not a valid
    denominator for a per-CO residual (rmse_metric_np_per_co/_per_co_residual
    require n_co > 0). Both cases must be rejected here.
    """
    return n_co_values is not None and all(n is not None and n > 0 for n in n_co_values)


def _best_model_filename(model_filename: str) -> str:
    """model.pt -> model_best.pt (or model_fold_3.pt -> model_fold_3_best.pt):
    a stable, method-agnostic name for the checkpoint actually used for the
    parity plot/metrics, so users don't have to know which fold "won" or
    whether K-fold was even used to find the model that matters."""
    p = Path(model_filename)
    return f"{p.stem}_best{p.suffix}"


def load_named_dataset(section_cfg: dict | list[dict]):
    """Load (atoms, target_y, path) lists from the ase.db(s) described by a
    config section shaped like {"db_path": ..., "target_y": "dft_energy"}, or
    a list of such dicts to concatenate multiple sources into one training
    set - e.g. real DFT rows plus a separate db of synthetic anchor points
    (see make_rep_anchors.py: n_co=1 structures with e_rep=0, added because
    rep.db itself has no examples below n_co=4 and a linear mean function
    fit only on n_co=4..10 extrapolates to nonsensical negative repulsion
    energies well before reaching n_co=1)."""
    sections = section_cfg if isinstance(section_cfg, list) else [section_cfg]

    atoms_list, y, paths, n_co = [], [], [], []
    for section in sections:
        db_path = section["db_path"]
        target_key = section.get("target_y", "dft_energy")

        db = connect(db_path)
        n_before = len(atoms_list)
        for row in db.select():
            atoms_list.append(row.toatoms())
            y.append(row.key_value_pairs[target_key])
            paths.append(row.key_value_pairs.get("path"))
            n_co.append(row.key_value_pairs.get("n_co"))  # None where absent (e.g. clean/slab _datasets)
        print(f"  {len(atoms_list) - n_before} structures from {db_path!r} (target_y={target_key!r})")

    if len(atoms_list) == 0:
        raise ValueError(f"No structures found in {section_cfg!r}.")

    return atoms_list, y, paths, n_co


def build_atom_indices(atoms_list, atom_indices_cfg: dict | None):
    """descriptor.atom_indices selects which atoms each structure's descriptor
    rows are built from, and whether those rows collapse into one vector.

    None (or no "source"): reproduces the original clean/slab pipeline - every
    atom contributes its own descriptor row, and build_K_NM's own per-structure
    sum gives an additive whole-slab energy model.

    "near_carbon_metals" (aggregate: true): rows are restricted to the metal
    atoms defining a structure's occupied CO site (atoms_near_carbon's first
    return value), then summed into a single vector describing that site as a
    whole - low_cov.ipynb's ads-site pipeline. Required because a site's own
    energy is not an additive sum over its neighbor metals' individual
    descriptors, unlike the whole-slab case above.

    "carbon_atoms" (aggregate: false): rows are restricted to (one row per)
    carbon atom, left unaggregated so build_K_NM's own additive atom-sum
    accumulates one contribution per adsorbed CO in the structure -
    rep.ipynb's repulsion pipeline.
    """
    if not atom_indices_cfg or not atom_indices_cfg.get("source"):
        return None, False

    source = atom_indices_cfg["source"]
    aggregate = bool(atom_indices_cfg.get("aggregate", False))

    atom_indices = []
    for atoms in atoms_list:
        selected_metals, carbon_indices, _ = atoms_near_carbon(atoms)
        if source == "near_carbon_metals":
            atom_indices.append(selected_metals.tolist())
        elif source == "carbon_atoms":
            atom_indices.append(carbon_indices.tolist())
        else:
            raise ValueError(f"Unknown descriptor.atom_indices.source: {source!r}")

    return atom_indices, aggregate


def maybe_aggregate_site_rows(x_list, aggregate: bool):
    if not aggregate:
        return x_list
    return [x.sum(dim=0, keepdim=True) for x in x_list]


def load_optional_test_tensors(
    cfg: dict, ce_config: CEConfig, atom_indices_cfg: dict | None, dtype: torch.dtype = torch.float64
):
    """test_dataset is a stub until a real held-out test set exists: return
    (None, None) while its db_path is unset, otherwise build descriptors for
    it with the SAME (already-fitted) ce_config so dimensions line up with
    train_x."""
    test_cfg = cfg.get("test_dataset")
    if not test_cfg or not test_cfg.get("db_path"):
        return None, None

    atoms_list, y_all, _, _ = load_named_dataset(test_cfg)
    atom_indices, aggregate = build_atom_indices(atoms_list, atom_indices_cfg)
    test_dataset = CEDataset(
        atoms=atoms_list,
        config=ce_config,
        atom_indices=atom_indices,
        target_y=y_all,
        dtype=dtype,
    )
    test_x, test_y = test_dataset.get_all()
    test_x = maybe_aggregate_site_rows(test_x, aggregate)
    return test_x, test_y


def build_descriptor_config(cfg: dict, atoms_list) -> CEConfig:
    d = cfg["descriptor"]
    mindist = d.get("mindist")
    if mindist is None:
        mindist = calc_mindist(atoms_list[0])  # same convention as clean.ipynb

    return CEConfig(
        elements=d["elements"],
        max_order=d.get("max_order", 2),
        mindist=mindist,
        shells=d["shells"],
    )


def resolve_init_lengthscale(model_cfg: dict, train_x):
    """model.init_lengthscale: a number/list (used as-is, current behavior) or
    the string "auto" - per-dimension std of the descriptor over all training
    atoms.

    select_inducing_points() scales distances by the model's *initial*
    lengthscale (it runs once at construction, before any fitting), so a
    single scalar guess applied uniformly across dimensions with very
    different natural scales (e.g. binary "single atom" features vs. pair
    counts up to 20) makes that scaling arbitrary for most dimensions - which
    plausibly contributed to the wild multi-hyperparameter excursions seen
    while debugging this pipeline. Std-per-dimension gives every dimension a
    comparable starting normalization instead.
    """
    value = model_cfg.get("init_lengthscale", 1.0)
    if value != "auto":
        return value

    all_atoms = torch.cat([torch.as_tensor(x) for x in train_x], dim=0)
    std = all_atoms.std(dim=0)
    std = std.clamp_min(1e-3)  # guard against a constant (but non-masked) descriptor dimension
    return std.tolist()


def resolve_warm_start(
    model_cfg: dict,
    checkpoint_path: Path,
    descriptor_dim: int,
    device: str,
    fallback_checkpoint_path: str | Path | None = None,
):
    """model.warm_start: true - if a checkpoint from a PREVIOUS run of this
    same config already sits at its own output path, load its converged
    lengthscale/sigma2/outputscale (and linear_mean, if any) and use them as
    this run's init values instead of the config's init_lengthscale/
    init_sigma2/init_outputscale/"auto".

    Why: this is the exact situation active learning creates - the same
    train_config gets re-run from scratch after every new DFT point, and
    LBFGS spends its first few steps just re-discovering roughly the same
    hyperparameters it already found last cycle (loss dropping from ~1e7 to
    ~1e2 in the first 2-3 steps, every single time - see any AL retrain log).
    Starting from where the previous cycle actually converged skips that
    rediscovery. This does NOT warm-start x_M/c: inducing points are always
    reselected fresh from the current (grown) dataset (see
    SparseAtomicGPR.select_inducing_points) - only the continuous
    hyperparameters, which don't depend on which points got selected.

    fallback_checkpoint_path: used ONLY when `checkpoint_path` doesn't exist
    yet (this train_config's own output dir has never been written to - e.g.
    active learning's very first cycle). Lets ActiveLearningController pass
    the model the running MC evaluator was ACTUALLY deployed with (its
    "models.ads_model"/"models.slab_model" config entry, generally a
    manually-trained baseline living somewhere else entirely, e.g.
    "../training/models/ads/model_best.pt") - without this, cycle 1 would
    cold-start from the config's own init_lengthscale/"auto" even though a
    perfectly good, already-converged model already exists and is right now
    live in the evaluator, wasting exactly the rediscovery this feature
    exists to skip.

    Returns (init_lengthscale, init_sigma2, init_outputscale, linear_mean_or_None).
    Falls back to config defaults (None sentinel for "not overridden") if
    warm_start is off, no previous/fallback checkpoint exists yet (first
    cycle with nothing deployed either), or its descriptor dimension no
    longer matches (e.g. shells/max_order changed).
    """
    if not model_cfg.get("warm_start", False):
        return None, None, None, None

    source_path = checkpoint_path
    if not source_path.exists():
        if fallback_checkpoint_path is not None and Path(fallback_checkpoint_path).exists():
            source_path = Path(fallback_checkpoint_path)
            print(f"warm_start: no checkpoint at {checkpoint_path} yet - bootstrapping from deployed model {source_path} instead.")
        else:
            print(f"warm_start: no previous checkpoint at {checkpoint_path} yet (first cycle) - using config init values.")
            return None, None, None, None

    old_model = SparseAtomicGPR(model_path=str(source_path), device=device)
    if old_model.x_M.shape[1] != descriptor_dim:
        print(
            f"warm_start: {source_path} has descriptor dim {old_model.x_M.shape[1]} "
            f"!= current {descriptor_dim} (descriptor config changed?) - using config init values."
        )
        return None, None, None, None

    init_lengthscale = old_model.lengthscale.detach().cpu().tolist()
    init_sigma2 = float(old_model.sigma2.detach().cpu().item())
    init_outputscale = float(old_model.outputscale.detach().cpu().item())
    linear_mean = old_model.linear_mean.detach().clone() if old_model.linear_mean is not None else None
    print(f"warm_start: reusing converged hyperparameters from {source_path}")
    return init_lengthscale, init_sigma2, init_outputscale, linear_mean


def build_optimizer_scheduler(model, adam_cfg: dict):
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_cfg.get("lr", 1e-1))

    sched_cfg = adam_cfg.get("scheduler")
    if not sched_cfg:
        return optimizer, None

    sched_type = sched_cfg.get("type", "reduce_on_plateau")
    if sched_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 20),
        )
    elif sched_type == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 20),
            T_mult=sched_cfg.get("T_mult", 1),
        )
    else:
        raise ValueError(f"Unknown training.adam.scheduler.type: {sched_type!r}")

    return optimizer, scheduler


_REMOVED_TRAINING_KEYS = {
    "select_by_rmse_per_co": "best_model_metric: 'rmse_valid_per_co' (was true) or 'rmse_valid' (was false)",
}
_REMOVED_LBFGS_KEYS = {
    "early_stop_patience": "loss_plateau_window",
    "loss_tol": "loss_plateau_tol",
    "plateau_metric": "(removed: stopping is always on the loss)",
    "plateau_window": "loss_plateau_window",
    "plateau_tol": "loss_plateau_tol",
}


def selection_and_stop_kwargs(training_cfg: dict) -> dict:
    """The three tags controlling the best checkpoint and when to stop:

    training.best_model_metric       "rmse_valid" | "rmse_valid_per_co" | "loss"
        which step's model is SAVED as best (see train_lbfgs).
    training.lbfgs.loss_plateau_window   steps; null disables the stop rule.
    training.lbfgs.loss_plateau_tol      minimum loss drop over that window.
        Stopping is always on the loss plateau, whatever the metric above.

    Removed keys are rejected outright: an unrecognized key would otherwise
    be silently ignored and the run would quietly use defaults instead of
    what the config says."""
    lbfgs_cfg = training_cfg.get("lbfgs", {})
    stale = [f"{k!r} -> {v}" for k, v in _REMOVED_TRAINING_KEYS.items() if k in training_cfg]
    stale += [f"lbfgs.{k!r} -> {v}" for k, v in _REMOVED_LBFGS_KEYS.items() if k in lbfgs_cfg]
    if stale:
        raise ValueError("training config uses removed key(s): " + "; ".join(stale))
    return {
        "best_model_metric": training_cfg.get("best_model_metric", "rmse_valid"),
        "loss_plateau_window": lbfgs_cfg.get("loss_plateau_window", 5),
        "loss_plateau_tol": lbfgs_cfg.get("loss_plateau_tol", 0.5),
    }


def run_training(
    model, train_x, train_y, test_x, test_y, training_cfg: dict, model_path: str, n_co_all=None
) -> dict:
    method = training_cfg.get("method", "adam")

    if method == "adam":
        # Non-K-fold methods need an explicit valid split for early stopping
        # during training itself (unlike K-fold, which carves its own out of
        # train_x per fold). No held-out set is configured here, so this
        # falls back to validating against the training set itself - fine for
        # a quick fit, but prefer a K-fold method for a real accuracy read.
        adam_cfg = training_cfg.get("adam", {})
        optimizer, scheduler = build_optimizer_scheduler(model, adam_cfg)
        history, best_rmse = train(
            train_x=train_x,
            train_y=train_y,
            valid_x=train_x,
            valid_y=train_y,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            n_epochs=adam_cfg.get("n_epochs", 500),
            model_path=model_path,
            min_lr=adam_cfg.get("min_lr", 1e-4),
            print_every=adam_cfg.get("print_every", 50),
        )
        return {"method": method, "history": history, "best_rmse_valid": best_rmse}

    if method == "lbfgs":
        lbfgs_cfg = training_cfg.get("lbfgs", {})
        history, best_rmse = train_lbfgs(
            train_x=train_x,
            train_y=train_y,
            valid_x=train_x,
            valid_y=train_y,
            model=model,
            n_steps=lbfgs_cfg.get("n_steps", 20),
            max_iter=lbfgs_cfg.get("max_iter", 20),
            lr=lbfgs_cfg.get("lr", 1.0),
            model_path=model_path,
            print_every=lbfgs_cfg.get("print_every", 1),
            **selection_and_stop_kwargs(training_cfg),
        )
        return {"method": method, "history": history, "best_rmse_valid": best_rmse}

    if method == "kfold_adam":
        adam_cfg = training_cfg.get("adam", {})
        kfold_cfg = training_cfg.get("kfold", {})
        optimizer, scheduler = build_optimizer_scheduler(model, adam_cfg)
        summary = train_kfold(
            train_x=train_x,
            train_y=train_y,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            test_x=test_x,
            test_y=test_y,
            n_epochs=adam_cfg.get("n_epochs", 500),
            n_splits=kfold_cfg.get("n_splits", 5),
            shuffle=kfold_cfg.get("shuffle", True),
            seed=kfold_cfg.get("seed", 42),
            model_path=model_path,
            min_lr=adam_cfg.get("min_lr", 1e-4),
        )
        return {"method": method, "summary": summary}

    if method == "kfold_lbfgs":
        lbfgs_cfg = training_cfg.get("lbfgs", {})
        kfold_cfg = training_cfg.get("kfold", {})
        # n_co is passed whenever the dataset has a usable one: it stratifies
        # the K-fold split by coverage, and is the per-fold valid_n_co that
        # best_model_metric="rmse_valid_per_co" needs. Clean/slab datasets
        # (n_co missing, or 0 on every row) fall back to a plain KFold.
        n_co_for_kfold = n_co_all if _usable_for_per_co_metric(n_co_all) else None
        if training_cfg.get("best_model_metric") == "rmse_valid_per_co" and n_co_for_kfold is None:
            raise ValueError(
                "training.best_model_metric='rmse_valid_per_co' needs a positive 'n_co' on every dataset row "
                "(missing, or a clean/slab dataset with n_co=0 everywhere) - use 'rmse_valid' or 'loss'."
            )
        if n_co_for_kfold is not None:
            print("K-fold: stratified by n_co (dataset has n_co on every row).")
        summary = train_kfold_lbfgs(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            model=model,
            n_splits=kfold_cfg.get("n_splits", 5),
            shuffle=kfold_cfg.get("shuffle", True),
            seed=kfold_cfg.get("seed", 42),
            n_steps=lbfgs_cfg.get("n_steps", 20),
            max_iter=lbfgs_cfg.get("max_iter", 20),
            lr=lbfgs_cfg.get("lr", 1.0),
            model_path=model_path,
            **selection_and_stop_kwargs(training_cfg),
            max_folds=kfold_cfg.get("max_folds"),
            n_co=n_co_for_kfold,
        )
        return {"method": method, "summary": summary}

    raise ValueError(f"Unknown training.method: {method!r}")


def run(config_path: str, warm_start_fallback: str | None = None, save_plot: bool = True) -> Path:
    """warm_start_fallback: passed straight through to resolve_warm_start's
    fallback_checkpoint_path - the currently-deployed model to bootstrap
    warm_start from on this config's very first run (before its own
    output.dir has ever been written to). Callers outside active learning

    save_plot: write the parity plot to output.plot_filename via Plotly's
    fig.write_image (needs the kaleido package - a headless-Chromium static
    image renderer). Pass False to skip it: write_image spawns a fresh
    kaleido subprocess EVERY call, and repeated calls within one long-lived
    Python process (e.g. active learning re-invoking run() once per
    retraining cycle) have been observed to eventually hang indefinitely -
    see ActiveLearningController.run_cycle, which always passes False."""
    cfg = load_config(config_path)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = cfg.get("device", "cpu")
    dtype = getattr(torch, cfg.get("dtype", "float64"))

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config immediately, before anything can fail, so a
    # crashed run is still reproducible from the run directory alone.
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log_path = output_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            print(f"Config: {config_path}")
            print(json.dumps(cfg, indent=2))
            print()

            print("Loading dataset:")
            atoms_list, y_all, paths, n_co_all = load_named_dataset(cfg["dataset"])
            print(f"Loaded {len(atoms_list)} structures total")

            ce_config = build_descriptor_config(cfg, atoms_list)
            print(f"mindist = {ce_config.mindist:.4f} A, shells = {ce_config.shells}")

            atom_indices_cfg = cfg["descriptor"].get("atom_indices")
            atom_indices, aggregate_site_rows = build_atom_indices(atoms_list, atom_indices_cfg)
            if atom_indices_cfg and atom_indices_cfg.get("source"):
                print(
                    f"atom_indices: source={atom_indices_cfg['source']!r}, "
                    f"aggregate={aggregate_site_rows}"
                )

            # dtype=dtype: CEDataset defaults to float32 (torch.utils.data.Dataset
            # convention), but every downstream consumer (SparseAtomicGPR,
            # training) assumes float64 - without this, target_y (DFT energies,
            # O(100-1000) eV) gets silently rounded to float32 precision
            # (~1e-4 eV absolute error) before being upcast back to float64 for
            # training, fighting the very sub-0.1 eV accuracy this pipeline is
            # tuned for.
            dataset = CEDataset(
                atoms=atoms_list,
                config=ce_config,
                atom_indices=atom_indices,
                target_y=y_all,
                dtype=dtype,
            )
            print(f"Descriptor dim (after masking): {dataset.X[0].shape[1]}")

            train_x, train_y = dataset.get_all()
            train_x = maybe_aggregate_site_rows(train_x, aggregate_site_rows)
            print(
                f"train: {len(train_x)} structures (full dataset, no split - "
                f"K-fold builds its own train/valid split per fold)"
            )

            test_x, test_y = load_optional_test_tensors(cfg, ce_config, atom_indices_cfg, dtype=dtype)
            if test_x is not None:
                print(f"test: {len(test_x)} structures (from {cfg['test_dataset']['db_path']})")
            else:
                print(
                    "test: none yet (test_dataset.db_path is a stub) - "
                    "K-fold rmse_test and the plot's Test trace will be skipped."
                )
            print()

            model_cfg = cfg["model"]
            # This stays the GEOMETRIC reference for select_inducing_points
            # (see induce_lengthscale below) even when warm-starting - a
            # converged ARD lengthscale is the wrong scale for that (most
            # dimensions pushed to near-irrelevance, collapsing the diversity
            # criterion almost everywhere; see resolve_warm_start/gpr.py).
            auto_lengthscale = resolve_init_lengthscale(model_cfg, train_x)
            if model_cfg.get("init_lengthscale") == "auto":
                print(f"init_lengthscale (auto, per-dim std): {[round(v, 4) for v in auto_lengthscale]}")

            best_filename = _best_model_filename(cfg["output"].get("model_filename", "model.pt"))
            best_model_copy_path = output_dir / best_filename
            warm_lengthscale, warm_sigma2, warm_outputscale, warm_linear_mean = resolve_warm_start(
                model_cfg, best_model_copy_path, dataset.X[0].shape[1], device,
                fallback_checkpoint_path=warm_start_fallback,
            )
            init_lengthscale = warm_lengthscale if warm_lengthscale is not None else auto_lengthscale

            model = SparseAtomicGPR(
                x_train=train_x,
                M=model_cfg.get("M", 100),
                div=model_cfg.get("div", 0.001),
                init_lengthscale=init_lengthscale,
                induce_lengthscale=auto_lengthscale if warm_lengthscale is not None else None,
                init_sigma2=warm_sigma2 if warm_sigma2 is not None else model_cfg.get("init_sigma2", 1e-4),
                init_outputscale=warm_outputscale if warm_outputscale is not None else model_cfg.get("init_outputscale", 1.0),
                config=ce_config,
                jitter=model_cfg.get("jitter", 1e-10),
                device=device,
                dtype=dtype,
                mean_function=model_cfg.get("mean_function", "zero"),
                mean_l2_reg=model_cfg.get("mean_l2_reg", 0.0),
            )
            if warm_linear_mean is not None and model.linear_mean is not None:
                with torch.no_grad():
                    model.linear_mean.copy_(warm_linear_mean.to(device=model.linear_mean.device, dtype=model.linear_mean.dtype))
            print(f"Model: M={model.x_M.shape[0]} inducing points, D={model.x_M.shape[1]}")
            print()

            model_path = str(output_dir / cfg["output"].get("model_filename", "model.pt"))
            result = run_training(
                model, train_x, train_y, test_x, test_y, cfg["training"], model_path, n_co_all=n_co_all
            )

            with open(output_dir / "train_log.json", "w", encoding="utf-8") as f:
                json.dump(_json_sanitize(result), f, indent=2)

            if result["method"] in ("kfold_adam", "kfold_lbfgs"):
                # K-fold trains a separate deep-copied model per fold and
                # checkpoints each one under its own _fold_N path (see
                # _checkpoint_path in train.py); the plain model_path is never
                # written and the original `model` object here was never
                # fitted. Evaluate/plot the fold with the best valid RMSE,
                # using THAT fold's own held-out indices for the "Valid" trace
                # below - it's the only valid split that actually exists now
                # that the top-level dataset isn't split anymore.
                fold_results = result["summary"]["fold_results"]
                best_fold = min(fold_results, key=lambda r: r["best_rmse_valid"])
                best_model_path = best_fold["model_path"]
                print(
                    f"\nBest fold: {best_fold['fold']} "
                    f"(RMSE valid={best_fold['best_rmse_valid']:.6f}) -> {best_model_path}"
                )
                train_y_t = torch.as_tensor(train_y)
                plot_valid_x, plot_valid_y = select_by_indices(train_x, train_y_t, best_fold["valid_idx"])
                plot_valid_n_co = [n_co_all[i] for i in best_fold["valid_idx"]]
            else:
                best_model_path = model_path
                # adam/lbfgs (non-K-fold) were trained validating against the
                # training set itself (see run_training) - keep that for the plot.
                plot_valid_x, plot_valid_y = train_x, train_y
                plot_valid_n_co = n_co_all

            # best_filename/best_model_copy_path were already resolved above
            # (used for warm_start's own checkpoint lookup) - reuse them here
            # rather than recomputing, so both refer to the exact same path.
            wrote_checkpoint = Path(best_model_path).exists()
            if wrote_checkpoint:
                model = SparseAtomicGPR(model_path=best_model_path, device=device)
                print(f"Reloaded best checkpoint from {best_model_path}")

                if Path(best_model_path).resolve() != best_model_copy_path.resolve():
                    shutil.copy2(best_model_path, best_model_copy_path)
                    print(f"Copied best checkpoint -> {best_model_copy_path}")
            else:
                best_model_copy_path = None
                print("\nWarning: no checkpoint was written - evaluating the in-memory model as-is.")

            plot_path = str(output_dir / cfg["output"].get("plot_filename", "parity.pdf"))

            # plot_results() calls fig.show(), which would try to pop open a
            # browser in this non-interactive/headless run - suppress just
            # that call without touching plot.py.
            original_show = go.Figure.show
            go.Figure.show = lambda self, *a, **k: None
            try:
                fig, metrics = plot_results(
                    model,
                    train_x,
                    train_y,
                    plot_valid_x,
                    plot_valid_y,
                    test_x=test_x,
                    test_y=test_y,
                    save_plot=save_plot,
                    filename=plot_path,
                )
            finally:
                go.Figure.show = original_show

            print("Final metrics:", metrics)

            # Per-CO metrics for visibility (checkpoint SELECTION already
            # happened per-CO inside run_training when selection_metric="per_co" -
            # this just reports the same quantity on the reloaded best model).
            if _usable_for_per_co_metric(n_co_all) and _usable_for_per_co_metric(plot_valid_n_co):
                with torch.no_grad():
                    train_pred = model(train_x)
                    valid_pred = model(plot_valid_x)
                train_n_co_t = torch.as_tensor(n_co_all, dtype=torch.float64)
                valid_n_co_t = torch.as_tensor(plot_valid_n_co, dtype=torch.float64)
                metrics["rmse_train_per_co"] = rmse_metric_np_per_co(train_pred, train_y, train_n_co_t)
                metrics["mae_train_per_co"] = mae_metric_np_per_co(train_pred, train_y, train_n_co_t)
                metrics["rmse_valid_per_co"] = rmse_metric_np_per_co(valid_pred, plot_valid_y, valid_n_co_t)
                metrics["mae_valid_per_co"] = mae_metric_np_per_co(valid_pred, plot_valid_y, valid_n_co_t)
                print(
                    "Per-CO metrics: "
                    f"RMSE train/CO={metrics['rmse_train_per_co']:.6f} "
                    f"RMSE valid/CO={metrics['rmse_valid_per_co']:.6f}"
                )

            metrics["best_model_path"] = str(best_model_path)
            if best_model_copy_path is not None:
                metrics["best_model_copy_path"] = str(best_model_copy_path)

            with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    print(f"\nDone. Outputs written to {output_dir}/")
    return output_dir, (best_model_copy_path if best_model_copy_path is not None else Path(best_model_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a SparseAtomicGPR model from a JSON config (see ce_gpr_train.example.json)."
    )
    parser.add_argument("config", help="Path to a JSON config file.")
    args = parser.parse_args()

    run(args.config)


if __name__ == "__main__":
    main()
