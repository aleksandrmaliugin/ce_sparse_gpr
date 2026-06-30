#!/usr/bin/env python3
"""Filter active-learning MC candidates by model-space diversity.

The script reads structures written by the MC active-learning trajectory and
keeps only candidates that are sufficiently different in the descriptor/kernel
space of the loaded SparseAtomicGPR models.

Typical use:

python filter_active_learning_candidates_v1.py \
    --input active_learning_candidates.traj \
    --output active_learning_unique.traj \
    --slab-model slab_model.pt \
    --ads-model ads_model.pt \
    --rep-model rep_model.pt \
    --similarity-threshold 0.995 \
    --min-uncertainty 0.05 \
    --sort-by-uncertainty
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write

from ce_sparse_gpr.ce_extractor import ClusterExpansion
from ce_sparse_gpr.dataset import atoms_near_carbon, aggregate_multi_label_descriptors
from ce_sparse_gpr.gpr import SparseAtomicGPR


@dataclass
class CandidateRecord:
    index: int
    atoms: Atoms
    uncertainty: float
    n_co: int
    exact_hash: str
    feature: np.ndarray
    max_similarity_to_selected: float = float("nan")
    selected: bool = False
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter active-learning candidates by descriptor/kernel-space diversity."
    )

    parser.add_argument("--input", required=True, help="Input ASE trajectory/extxyz with AL candidates.")
    parser.add_argument("--output", required=True, help="Output trajectory/extxyz with selected diverse candidates.")
    parser.add_argument("--summary", default=None, help="Optional CSV summary path.")

    parser.add_argument("--slab-model", required=True, help="SparseAtomicGPR slab model checkpoint.")
    parser.add_argument("--ads-model", required=True, help="SparseAtomicGPR adsorption model checkpoint.")
    parser.add_argument(
        "--rep-model",
        default=None,
        help="Optional SparseAtomicGPR CO-CO repulsion model checkpoint. Used only when N_CO > 1.",
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-unsafe-model-load",
        action="store_true",
        help="Use only for trusted old torch checkpoints if your SparseAtomicGPR supports this flag.",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.995,
        help="Keep candidate if its max cosine similarity to selected structures is below this value.",
    )
    parser.add_argument(
        "--min-uncertainty",
        type=float,
        default=-float("inf"),
        help="Discard candidates with uncertainty below this value.",
    )
    parser.add_argument(
        "--sort-by-uncertainty",
        action="store_true",
        help="Process highest-uncertainty candidates first. Otherwise preserve trajectory order.",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Optional maximum number of selected output structures.",
    )
    parser.add_argument(
        "--disable-exact-hash-filter",
        action="store_true",
        help="Do not remove exact duplicates before descriptor similarity filtering.",
    )
    parser.add_argument(
        "--exact-round-decimals",
        type=int,
        default=8,
        help="Rounding precision for exact hash of slab symbols and C fractional xy positions.",
    )

    parser.add_argument("--weight-slab", type=float, default=1.0)
    parser.add_argument("--weight-ads", type=float, default=1.0)
    parser.add_argument("--weight-rep", type=float, default=1.0)
    parser.add_argument(
        "--weight-n-co",
        type=float,
        default=0.15,
        help="Small extra weight for normalized N_CO, useful to separate different coverages.",
    )

    return parser.parse_args()


def load_model(path: str, device: str, allow_unsafe_model_load: bool) -> SparseAtomicGPR:
    """Load model while tolerating old/new SparseAtomicGPR constructor signatures."""
    try:
        model = SparseAtomicGPR(
            model_path=path,
            device=device,
            allow_unsafe_load=allow_unsafe_model_load,
        )
    except TypeError:
        model = SparseAtomicGPR(model_path=path, device=device)

    model = model.to(device)
    model.eval()
    return model


def finite_float(x, default: float = float("nan")) -> float:
    try:
        value = float(x)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def get_uncertainty(atoms: Atoms) -> float:
    for key in (
        "active_learning_uncertainty",
        "uncertainty",
        "total_uncertainty",
        "std",
        "sigma",
    ):
        if key in atoms.info:
            return finite_float(atoms.info[key])
    return float("nan")


def get_n_co(atoms: Atoms) -> int:
    if "N_CO" in atoms.info:
        value = finite_float(atoms.info["N_CO"])
        if np.isfinite(value):
            return int(round(value))
    return int(sum(1 for atom in atoms if atom.symbol == "C"))


def exact_structure_hash(atoms: Atoms, decimals: int = 8) -> str:
    """Hash discrete slab symbols and rounded C xy positions.

    This removes literal duplicates generated by repeated visits to the same
    composition/occupation state. It intentionally ignores O positions because
    O follows C for upright CO in this MC setup.
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    substrate_symbols = tuple(symbols[(symbols != "C") & (symbols != "O")].tolist())

    c_indices = np.where(symbols == "C")[0]
    if len(c_indices) > 0:
        frac = atoms.cell.scaled_positions(atoms.positions[c_indices])
        c_xy = np.round(frac[:, :2] % 1.0, decimals=decimals)
        c_xy = tuple(map(tuple, c_xy[np.lexsort((c_xy[:, 1], c_xy[:, 0]))]))
    else:
        c_xy = tuple()

    payload = repr((substrate_symbols, c_xy)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return np.zeros_like(v)
    return v / n


@torch.no_grad()
def kernel_sum_feature(model: SparseAtomicGPR, x) -> np.ndarray:
    """Return sum-row kernel feature K_NM for one structure/block."""
    if x is None:
        return np.zeros(int(model.x_M.shape[0]), dtype=np.float64)

    if torch.is_tensor(x):
        if x.numel() == 0 or x.shape[0] == 0:
            return np.zeros(int(model.x_M.shape[0]), dtype=np.float64)
        x_in = x.to(dtype=torch.float64, device=model.x_M.device)
    else:
        arr = np.asarray(x)
        if arr.size == 0 or arr.shape[0] == 0:
            return np.zeros(int(model.x_M.shape[0]), dtype=np.float64)
        x_in = torch.as_tensor(arr, dtype=torch.float64, device=model.x_M.device)

    if x_in.ndim == 1:
        x_in = x_in.unsqueeze(0)

    k_nm = model.build_K_NM([x_in])
    return k_nm.detach().cpu().numpy().reshape(-1).astype(np.float64)


def atoms_near_carbon_top_bridge(atoms: Atoms):
    """Call atoms_near_carbon while staying compatible with older dataset.py."""
    try:
        return atoms_near_carbon(atoms, allowed_site_types=("top", "bridge"))
    except TypeError:
        return atoms_near_carbon(atoms)


def component_features(
    atoms: Atoms,
    slab_model: SparseAtomicGPR,
    ads_model: SparseAtomicGPR,
    rep_model: SparseAtomicGPR | None,
    slab_extractor: ClusterExpansion,
    ads_extractor: ClusterExpansion,
    rep_extractor: ClusterExpansion | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute model-space features for slab, ads and rep blocks."""
    slab_desc = slab_extractor(atoms)
    slab_feat = kernel_sum_feature(slab_model, slab_desc)

    atom_indices, carbon_indices, labels_per_atom = atoms_near_carbon_top_bridge(atoms)
    n_co = int(len(carbon_indices))

    if len(atom_indices) > 0:
        ads_desc = ads_extractor(atoms, atom_indices=atom_indices)
        ads_desc = torch.as_tensor(ads_desc, dtype=torch.float64, device=ads_model.x_M.device)
        x_sites, _ = aggregate_multi_label_descriptors(ads_desc, labels_per_atom)
        ads_feat = kernel_sum_feature(ads_model, x_sites)
    else:
        ads_feat = np.zeros(int(ads_model.x_M.shape[0]), dtype=np.float64)

    if rep_model is not None and rep_extractor is not None and n_co > 1:
        rep_desc = rep_extractor(atoms, atom_indices=carbon_indices)
        rep_feat = kernel_sum_feature(rep_model, rep_desc)
    elif rep_model is not None:
        rep_feat = np.zeros(int(rep_model.x_M.shape[0]), dtype=np.float64)
    else:
        rep_feat = np.zeros(0, dtype=np.float64)

    return slab_feat, ads_feat, rep_feat, n_co


def build_features(
    frames: list[Atoms],
    slab_model: SparseAtomicGPR,
    ads_model: SparseAtomicGPR,
    rep_model: SparseAtomicGPR | None,
    weights: tuple[float, float, float, float],
) -> list[np.ndarray]:
    slab_extractor = ClusterExpansion(slab_model.config)
    ads_extractor = ClusterExpansion(ads_model.config)
    rep_extractor = ClusterExpansion(rep_model.config) if rep_model is not None else None

    components = []
    max_n_co = 1

    for atoms in frames:
        slab_feat, ads_feat, rep_feat, n_co = component_features(
            atoms=atoms,
            slab_model=slab_model,
            ads_model=ads_model,
            rep_model=rep_model,
            slab_extractor=slab_extractor,
            ads_extractor=ads_extractor,
            rep_extractor=rep_extractor,
        )
        components.append((slab_feat, ads_feat, rep_feat, n_co))
        max_n_co = max(max_n_co, int(n_co))

    w_slab, w_ads, w_rep, w_n_co = weights
    features: list[np.ndarray] = []

    for slab_feat, ads_feat, rep_feat, n_co in components:
        parts = [
            w_slab * l2_normalize(slab_feat),
            w_ads * l2_normalize(ads_feat),
            w_rep * l2_normalize(rep_feat),
            np.array([w_n_co * float(n_co) / float(max_n_co)], dtype=np.float64),
        ]
        feature = l2_normalize(np.concatenate(parts))
        features.append(feature)

    return features


def greedy_select(records: list[CandidateRecord], args: argparse.Namespace) -> list[CandidateRecord]:
    if args.sort_by_uncertainty:
        order = sorted(
            range(len(records)),
            key=lambda i: records[i].uncertainty if np.isfinite(records[i].uncertainty) else -np.inf,
            reverse=True,
        )
    else:
        order = list(range(len(records)))

    selected: list[CandidateRecord] = []
    selected_features: list[np.ndarray] = []
    seen_hashes: set[str] = set()

    for i in order:
        rec = records[i]

        if np.isfinite(args.min_uncertainty) and (
            not np.isfinite(rec.uncertainty) or rec.uncertainty < args.min_uncertainty
        ):
            rec.reason = "below_min_uncertainty"
            continue

        if not args.disable_exact_hash_filter and rec.exact_hash in seen_hashes:
            rec.reason = "exact_duplicate"
            continue

        if selected_features:
            sims = [float(np.dot(rec.feature, f)) for f in selected_features]
            max_sim = max(sims)
        else:
            max_sim = -np.inf

        rec.max_similarity_to_selected = max_sim

        if max_sim >= args.similarity_threshold:
            rec.reason = "too_similar"
            continue

        rec.selected = True
        rec.reason = "selected"
        selected.append(rec)
        selected_features.append(rec.feature)
        seen_hashes.add(rec.exact_hash)

        if args.max_structures is not None and len(selected) >= args.max_structures:
            break

    return selected


def write_summary(path: str, records: list[CandidateRecord]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "selected",
                "reason",
                "uncertainty",
                "N_CO",
                "max_similarity_to_selected",
                "exact_hash",
            ],
        )
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r.index):
            writer.writerow(
                {
                    "index": rec.index,
                    "selected": int(rec.selected),
                    "reason": rec.reason,
                    "uncertainty": rec.uncertainty,
                    "N_CO": rec.n_co,
                    "max_similarity_to_selected": rec.max_similarity_to_selected,
                    "exact_hash": rec.exact_hash,
                }
            )


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.similarity_threshold <= 1.0):
        raise ValueError("--similarity-threshold must be in [0, 1].")

    frames = read(args.input, index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    frames = list(frames)

    if len(frames) == 0:
        raise ValueError(f"No frames were read from {args.input!r}.")

    slab_model = load_model(args.slab_model, args.device, args.allow_unsafe_model_load)
    ads_model = load_model(args.ads_model, args.device, args.allow_unsafe_model_load)
    rep_model = (
        load_model(args.rep_model, args.device, args.allow_unsafe_model_load)
        if args.rep_model is not None
        else None
    )

    features = build_features(
        frames=frames,
        slab_model=slab_model,
        ads_model=ads_model,
        rep_model=rep_model,
        weights=(args.weight_slab, args.weight_ads, args.weight_rep, args.weight_n_co),
    )

    records = [
        CandidateRecord(
            index=i,
            atoms=atoms,
            uncertainty=get_uncertainty(atoms),
            n_co=get_n_co(atoms),
            exact_hash=exact_structure_hash(atoms, decimals=args.exact_round_decimals),
            feature=features[i],
        )
        for i, atoms in enumerate(frames)
    ]

    selected = greedy_select(records, args)

    output_parent = Path(args.output).resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)

    if selected:
        write(args.output, [rec.atoms for rec in selected])
    else:
        print("No structures selected. Output file was not written.")

    summary_path = args.summary
    if summary_path is None:
        root, _ = os.path.splitext(args.output)
        summary_path = root + "_summary.csv"
    write_summary(summary_path, records)

    n_below = sum(r.reason == "below_min_uncertainty" for r in records)
    n_exact = sum(r.reason == "exact_duplicate" for r in records)
    n_similar = sum(r.reason == "too_similar" for r in records)

    print(f"Read candidates:          {len(records)}")
    print(f"Selected diverse frames:  {len(selected)}")
    print(f"Discarded below unc.:     {n_below}")
    print(f"Discarded exact dup.:     {n_exact}")
    print(f"Discarded too similar:    {n_similar}")
    print(f"Output:                   {args.output if selected else '(not written)'}")
    print(f"Summary:                  {summary_path}")


if __name__ == "__main__":
    main()
