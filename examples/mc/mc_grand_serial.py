from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ase.geometry import get_distances
from ase.io import read, write


def _find_repo_root(start: Path) -> Path:
    """Walk upward from this file to the repo root (the folder holding
    pyproject.toml next to the ce_sparse_gpr package). Needed only for
    make_database.py, the one module ActiveLearningController imports that
    lives at the repo root rather than inside the package.

    Appended to sys.path (not inserted at the front) so this script's own
    directory always wins name clashes against the repo root."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "ce_sparse_gpr").is_dir():
            return candidate
    return start


_repo_root = str(_find_repo_root(Path(__file__).resolve().parent))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from ce_sparse_gpr import ce_gpr_train
from ce_sparse_gpr.mc_grand_utils import (
    KB_EV,
    AdsorptionSite,
    EnergyComponents,
    MCStats,
    attach_mc_info,
    adsorption_coverage_denominator,
    co_coverage,
    build_adsorbed_structure,
    build_supercell,
    carbon_index_by_site_id,
    find_adsorption_sites,
    finite_float,
    mic_distance,
    min_distance_to_occupied_sites,
    pbc_xy_distance,
    metropolis_hastings_accept,
    nonnegative_float,
    occupation_satisfies_min_distance,
    positive_float,
    propose_pt_pd_swap,
    strip_to_symbols,
    validate_occupation,
    validate_sites,
    assign_substrate_layers,
    frozen_atom_mask_from_layers,
    layer_summary,
)

from ce_sparse_gpr.ce_extractor import ClusterExpansion
from ce_sparse_gpr.gpr import SparseAtomicGPR
from ce_sparse_gpr.calculator import CalculatorCESparseGPR


@dataclass
class LocalMCState:

    slab_atoms: object
    occupation: np.ndarray
    energy: EnergyComponents
    slab_k: dict[int, torch.Tensor]
    ads_k: dict[int, torch.Tensor]
    # Raw per-atom/per-site descriptor rows (NOT kernel-projected against
    # x_M), cached alongside slab_k/ads_k with the same keys. Needed
    # only to compute the exact self-kernel diagonal k(x*,x*) for
    # uncertainty (see _mean_std_from_desc_and_k) - CalculatorEnergyEvaluator
    # never populates these (defaults to {}) since it always has the full
    # descriptor on hand and calls model.predict_uncertainty directly.
    slab_desc: dict[int, torch.Tensor] = field(default_factory=dict)
    ads_desc: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.occupation = np.asarray(self.occupation, dtype=bool).copy()

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def namespace_from_config(cfg: dict) -> argparse.Namespace:
    """Flatten the nested JSON config into the same flat attribute names the
    old argparse.Namespace used to produce, so the rest of this module (main,
    load_or_build_slab, etc.) is reused unchanged. See mc_grand.example.json.
    """
    structure = cfg.get("structure", {})
    models = cfg.get("models", {})
    thermo = cfg.get("thermo", {})
    mc_cfg = cfg.get("mc", {})
    output_cfg = cfg.get("output", {})

    if "slab_model" not in models:
        raise ValueError("config.models must set slab_model.")
    if not models.get("ads_model"):
        raise ValueError("config.models must set ads_model.")

    return argparse.Namespace(
        cell=structure.get("cell"),
        cell_format=structure.get("cell_format"),
        ignore_cell_adsorbates=bool(structure.get("ignore_cell_adsorbates", False)),
        initial_site_match_tol=float(structure.get("initial_site_match_tol", 0.35)),
        supercell=tuple(int(n) for n in structure.get("supercell", (4, 4, 6))),
        lattice_constant=float(structure.get("lattice_constant", 3.98398)),
        pd_fraction=float(structure.get("pd_fraction", 0.5)),
        # Only used when structure.cell is null (synthetic build path) - the
        # vacuum gap added on top of the built slab via ase.Atoms.center().
        # Was hardcoded to 15.0 with no config knob at all; that recipe does
        # NOT match whatever cell/vacuum convention a real DFT dataset used
        # (e.g. this project's own ads_unified.db rows sit in a fixed
        # 20.978 A cell - a synthetic build with vacuum=15.0 on a 4-layer
        # slab lands at ~26 A, a different vacuum thickness entirely).
        vacuum=float(structure.get("vacuum", 15.0)),
        slab_model=models["slab_model"],
        # The single per-CO adsorption model: additive across every occupied
        # site, one descriptor row per carbon atom (the "carbon_atoms" style -
        # see ads_unified's e_ads_total). Covers base adsorption energy AND
        # CO-CO lateral interactions together - see
        # LocalDescriptorEnergyEvaluator/CalculatorEnergyEvaluator.
        ads_model=models["ads_model"],
        freq_model=models.get("freq_model"),
        allow_unsafe_model_load=bool(models.get("allow_unsafe_model_load", False)),
        temperature=float(thermo.get("temperature", 300.0)),
        delta_mu=float(thermo.get("delta_mu", 0.0)),
        nsteps=int(mc_cfg.get("nsteps", 3000)),
        co_height=float(mc_cfg.get("co_height", 1.60)),
        co_bond=float(mc_cfg.get("co_bond", 1.15)),
        min_co_distance=float(mc_cfg.get("min_co_distance", 1.52)),
        z_atol=float(mc_cfg.get("z_atol", 1e-3)),
        bridge_cutoff_factor=float(mc_cfg.get("bridge_cutoff_factor", 1.25)),
        layer_z_tol=float(mc_cfg.get("layer_z_tol", 0.1)),
        freeze_layers=tuple(int(x) for x in mc_cfg.get("freeze_layers", ())),
        freeze_bottom_layers=int(mc_cfg.get("freeze_bottom_layers", 0)),
        local_cutoff_margin=float(mc_cfg.get("local_cutoff_margin", 0.05)),
        seed=int(cfg.get("seed", 1234)),
        device=str(cfg.get("device", "cpu")),
        evaluator_mode=str(cfg.get("evaluator_mode", "local")),
        print_every=int(output_cfg.get("print_every", 100)),
        write_every=int(output_cfg.get("write_every", 1000)),
        trajectory=output_cfg.get("trajectory", "mc_semigrand_traj.xyz"),
        output=output_cfg.get("output", "mc_semigrand_final.xyz"),
    )


def validate_args(args: argparse.Namespace) -> None:
    positive_float(args.temperature, "temperature")
    finite_float(args.delta_mu, "delta_mu")
    positive_float(args.lattice_constant, "lattice_constant")
    nonnegative_float(args.vacuum, "vacuum")
    if any(int(n) <= 0 for n in args.supercell):
        raise ValueError(f"all supercell dimensions must be positive, got {args.supercell}.")
    if not (0.0 <= float(args.pd_fraction) <= 1.0):
        raise ValueError(f"pd_fraction must be between 0 and 1, got {args.pd_fraction}.")

    for name in ("nsteps", "print_every", "write_every"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative, got {getattr(args, name)}.")

    positive_float(args.co_height, "co_height")
    positive_float(args.co_bond, "co_bond")
    nonnegative_float(args.min_co_distance, "min_co_distance")
    nonnegative_float(args.z_atol, "z_atol")
    nonnegative_float(args.layer_z_tol, "layer_z_tol")
    nonnegative_float(args.initial_site_match_tol, "initial_site_match_tol")
    positive_float(args.bridge_cutoff_factor, "bridge_cutoff_factor")
    nonnegative_float(args.local_cutoff_margin, "local_cutoff_margin")
    if int(args.freeze_bottom_layers) < 0:
        raise ValueError(f"freeze_bottom_layers must be non-negative, got {args.freeze_bottom_layers}.")
    if any(int(layer) < 0 for layer in args.freeze_layers):
        raise ValueError(f"freeze_layers must contain non-negative layer ids, got {args.freeze_layers}.")
    if args.evaluator_mode not in ("local", "calculator"):
        raise ValueError(f"evaluator_mode must be 'local' or 'calculator', got {args.evaluator_mode!r}.")
    if args.ads_model is None:
        raise ValueError("config.models must set ads_model - without it nothing predicts any adsorption energy.")


def validate_active_learning_config(al_cfg: dict | None) -> None:
    if not al_cfg or not al_cfg.get("enabled", False):
        return

    if not al_cfg.get("run_script"):
        raise ValueError("active_learning.run_script is required when active_learning.enabled is true.")
    if not os.path.exists(al_cfg["run_script"]):
        raise ValueError(f"active_learning.run_script does not exist: {al_cfg['run_script']!r}.")

    datasets = al_cfg.get("datasets", {})
    train_configs = al_cfg.get("train_configs", {})
    thresholds = al_cfg.get("uncertainty_thresholds", {})
    for component in ActiveLearningController.COMPONENTS:
        if component not in datasets:
            raise ValueError(f"active_learning.datasets.{component} is required.")
        if component not in train_configs:
            raise ValueError(f"active_learning.train_configs.{component} is required.")
        if not os.path.exists(train_configs[component]):
            raise ValueError(f"active_learning.train_configs.{component} does not exist: {train_configs[component]!r}.")
        threshold = float(thresholds.get(component, float("inf")))
        if np.isnan(threshold):
            raise ValueError(f"active_learning.uncertainty_thresholds.{component} must not be NaN.")


def _ensure_parent_dir(path: str | os.PathLike | None) -> None:
    if path in (None, ""):
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _safe_set_torch_threads(n_threads: int = 1) -> None:
    torch.set_num_threads(int(n_threads))
    try:
        torch.set_num_interop_threads(int(n_threads))
    except RuntimeError:
        pass


def load_or_build_slab(args: argparse.Namespace):
    """structure.cell must be set explicitly to load a real structure - no
    implicit filesystem sniffing (this used to silently pick up a same-named
    POSCAR sitting in the current directory if structure.cell was left null,
    which could load an unrelated leftover file with zero indication in the
    config or the log)."""

    input_atoms = None

    if args.cell is not None:
        input_atoms = read(args.cell, format=args.cell_format)
        slab_atoms = strip_to_symbols(input_atoms, keep_symbols=("Pt", "Pd"))
    else:
        slab_atoms = build_supercell(
            structure="fcc",
            lattice_constant=args.lattice_constant,
            composition={"Pd": args.pd_fraction, "Pt": 1.0 - args.pd_fraction},
            supercell=np.diag(args.supercell),
            seed=args.seed,
        )
        slab_atoms.center(vacuum=args.vacuum, axis=2)
        slab_atoms.pbc = (True, True, False)

    if len(slab_atoms) == 0:
        raise ValueError("Slab contains no Pt/Pd atoms.")
    unknown = set(slab_atoms.get_chemical_symbols()) - {"Pt", "Pd"}
    if unknown:
        raise ValueError(f"Slab contains non-substrate elements after filtering: {sorted(unknown)}.")
    return slab_atoms, input_atoms


def initial_occupation_from_input_adsorbates(
    input_atoms,
    slab_atoms,
    sites: list[AdsorptionSite],
    *,
    match_tol: float,
    min_co_distance: float,
    ignore_cell_adsorbates: bool = False,
) -> np.ndarray:

    sites = validate_sites(sites, n_atoms=len(slab_atoms))
    occupation = np.zeros(len(sites), dtype=bool)

    if input_atoms is None or ignore_cell_adsorbates:
        return occupation

    symbols = np.asarray(input_atoms.get_chemical_symbols())
    carbon_indices = np.where(symbols == "C")[0].astype(int)
    oxygen_indices = np.where(symbols == "O")[0].astype(int)

    if len(carbon_indices) == 0:
        return occupation

    if len(oxygen_indices) != len(carbon_indices):
        print(
            "warning: input structure contains "
            f"{len(carbon_indices)} C atoms and {len(oxygen_indices)} O atoms; "
            "initial occupation is inferred from C positions only.",
            flush=True,
        )

    match_tol = nonnegative_float(match_tol, "initial_site_match_tol")
    c_positions = input_atoms.get_positions()[carbon_indices]

    used_sites: dict[int, int] = {}
    for c_idx, c_pos in zip(carbon_indices, c_positions):
        distances = np.array(
            [
                pbc_xy_distance(
                    c_pos,
                    site.position,
                    cell=slab_atoms.cell,
                    pbc=slab_atoms.pbc,
                )
                for site in sites
            ],
            dtype=np.float64,
        )
        best_site = int(np.argmin(distances))
        best_distance = float(distances[best_site])

        if best_distance > match_tol:
            raise ValueError(
                f"Could not map input C atom {int(c_idx)} to an ontop/bridge MC site: "
                f"nearest site is {best_distance:.4f} A away, "
                f"larger than --initial-site-match-tol={match_tol:.4f} A. "
                "This usually means the input CO is on a hollow/off-grid site, "
                "the CO height/site grid settings are inconsistent, or the slab is strongly rumpled."
            )

        if best_site in used_sites:
            raise ValueError(
                f"Two input C atoms map to the same MC site {best_site}: "
                f"C {used_sites[best_site]} and C {int(c_idx)}. "
                "Increase the MC site grid resolution only if this is physically intended; "
                "otherwise check the input structure."
            )

        used_sites[best_site] = int(c_idx)
        occupation[best_site] = True

    if not occupation_satisfies_min_distance(
        sites=sites,
        occupation=occupation,
        min_distance=min_co_distance,
        cell=slab_atoms.cell,
        pbc=slab_atoms.pbc,
    ):
        raise ValueError(
            "CO positions inferred from the input structure violate --min-co-distance. "
            "Use --min-co-distance 0 to disable this hard exclusion, or provide a consistent POSCAR."
        )

    return occupation


def frozen_layers_from_args(args: argparse.Namespace, n_layers: int) -> set[int]:

    frozen = {int(layer) for layer in args.freeze_layers}
    frozen.update(range(int(args.freeze_bottom_layers)))
    if any(layer < 0 for layer in frozen):
        raise ValueError(f"Frozen layer ids must be non-negative, got {sorted(frozen)}.")
    invalid = sorted(layer for layer in frozen if layer >= int(n_layers))
    if invalid:
        raise ValueError(
            f"Frozen layer id(s) {invalid} are out of range. "
            f"Available layer ids are 0..{int(n_layers) - 1}."
        )
    return frozen


def print_layer_report(slab_atoms, layer_ids: np.ndarray, frozen_layers: set[int]) -> None:
    print("Substrate layers are numbered from bottom to top, starting at 0:", flush=True)
    for item in layer_summary(slab_atoms, layer_ids):
        layer = int(item["layer"])
        tag = "frozen" if layer in frozen_layers else "mobile"
        print(
            f"  layer {layer:2d}: n_atoms={int(item['n_atoms']):4d}, "
            f"z_mean={float(item['z_mean']):10.5f} A, "
            f"z_range=[{float(item['z_min']):.5f}, {float(item['z_max']):.5f}] A, "
            f"{tag}",
            flush=True,
        )


def cutoff_from_shells_dict(config, margin: float = 0.05) -> float:
    shells_dict = getattr(config, "shells_dict", None)
    if not shells_dict:
        return float("nan")
    margin = nonnegative_float(margin, "margin")
    upper_bounds = []
    for value in shells_dict.values():
        if value is None or len(value) < 2:
            continue
        upper = float(value[1])
        if np.isfinite(upper):
            upper_bounds.append(upper)
    if not upper_bounds:
        return float("nan")
    return float(max(upper_bounds) + margin)


def _load_sparse_model(model_path: str, device: str, allow_unsafe_load: bool) -> SparseAtomicGPR:
    signature = inspect.signature(SparseAtomicGPR)
    kwargs = {"model_path": model_path, "device": device}
    if "allow_unsafe_load" in signature.parameters:
        kwargs["allow_unsafe_load"] = bool(allow_unsafe_load)
    model = SparseAtomicGPR(**kwargs)
    model = model.to(device)
    model.eval()
    if getattr(model, "c", None) is None:
        raise RuntimeError(f"Model {model_path!r} has no fitted coefficient vector c.")
    if getattr(model, "config", None) is None:
        raise RuntimeError(f"Model {model_path!r} has no CEConfig in checkpoint.")
    return model


class LocalDescriptorEnergyEvaluator:

    def __init__(
        self,
        file_slab_model: str,
        file_ads_model: str,
        sites: Iterable[AdsorptionSite],
        co_bond: float,
        device: str = "cpu",
        allow_unsafe_load: bool = False,
        local_cutoff_margin: float = 0.05,
        file_freq_model: str | None = None,
    ):
        self.device = device
        self.sites = validate_sites(list(sites))
        self.co_bond = positive_float(co_bond, "co_bond")
        self.local_cutoff_margin = nonnegative_float(local_cutoff_margin, "local_cutoff_margin")
        self.allow_unsafe_load = bool(allow_unsafe_load)

        self.file_slab_model = file_slab_model
        self.slab_model = _load_sparse_model(file_slab_model, device, allow_unsafe_load)
        self.slab_extractor = ClusterExpansion(self.slab_model.config)

        # The single per-CO adsorption model: additive across every occupied
        # site, one descriptor row per carbon atom (see ads_unified's
        # e_ads_total) - covers base adsorption energy AND CO-CO lateral
        # interactions together, so it must contribute starting at N_CO=1
        # (there is no separate "first CO free" base-adsorption component
        # anymore).
        self.file_ads_model = file_ads_model
        self.ads_model = _load_sparse_model(file_ads_model, device, allow_unsafe_load)
        ads_elements = set(self.ads_model.config.elements)
        if "C" not in ads_elements:
            raise ValueError(
                "ads model must support carbon-centered descriptors (one row "
                f"per adsorbed CO); got elements {sorted(ads_elements)}."
            )
        self.ads_extractor = ClusterExpansion(self.ads_model.config)

        self.slab_cutoff = cutoff_from_shells_dict(self.slab_model.config, self.local_cutoff_margin)
        self.ads_cutoff = cutoff_from_shells_dict(self.ads_model.config, self.local_cutoff_margin)

        # Stub: see CalculatorEnergyEvaluator's identical freq_model wiring -
        # loaded/validated but not yet used in any energy/uncertainty below.
        self.file_freq_model = file_freq_model
        self.freq_model: SparseAtomicGPR | None = (
            _load_sparse_model(file_freq_model, device, allow_unsafe_load)
            if file_freq_model is not None
            else None
        )
        self.freq_extractor = ClusterExpansion(self.freq_model.config) if self.freq_model is not None else None

        self._validate_model_elements()

    def reload_model(self, component: str, model_path: str) -> None:
        """Hot-swap one component's checkpoint (slab/ads) after an
        active-learning retraining cycle. Any cached kernel rows referencing
        the OLD model (state.slab_k/ads_k) are stale after this and must be
        rebuilt via full_rebuild() - the caller (GrandCO_MC) already does
        that right after a retraining cycle."""
        if component not in ("slab", "ads"):
            raise ValueError(f"Unknown component {component!r}; expected 'slab' or 'ads'.")

        model = _load_sparse_model(model_path, self.device, self.allow_unsafe_load)
        extractor = ClusterExpansion(model.config)
        cutoff = cutoff_from_shells_dict(model.config, self.local_cutoff_margin)

        setattr(self, f"{component}_model", model)
        setattr(self, f"{component}_extractor", extractor)
        setattr(self, f"{component}_cutoff", cutoff)

    def _validate_model_elements(self) -> None:

        slab_elements = set(self.slab_model.config.elements)
        required_metals = {"Pt", "Pd"}

        if not required_metals.issubset(slab_elements):
            raise ValueError(
                f"slab model elements must include Pt and Pd, got {sorted(slab_elements)}."
            )

    def make_atoms(self, slab_atoms, occupation: np.ndarray):
        return build_adsorbed_structure(
            slab_atoms=slab_atoms,
            sites=self.sites,
            occupation=occupation,
            co_bond=self.co_bond,
        )

    def cutoffs(self) -> dict[str, float]:
        return {"slab": self.slab_cutoff, "ads": self.ads_cutoff}

    def initial_state(
        self,
        slab_atoms,
        occupation: np.ndarray | None = None,
        compute_uncertainty: bool = False,
    ) -> LocalMCState:
        if occupation is None:
            occupation = np.zeros(len(self.sites), dtype=bool)
        else:
            occupation = validate_occupation(occupation, len(self.sites))
        return self.full_rebuild(slab_atoms, occupation, compute_uncertainty=compute_uncertainty)

    def full_rebuild(self, slab_atoms, occupation: np.ndarray, compute_uncertainty: bool = False) -> LocalMCState:
        occupation = validate_occupation(occupation, len(self.sites))
        validate_sites(self.sites, n_atoms=len(slab_atoms))
        atoms = self.make_atoms(slab_atoms, occupation)

        slab_k, slab_desc = self._build_all_slab_rows(slab_atoms)
        ads_k, ads_desc = self._build_all_ads_rows(atoms, slab_atoms, occupation)
        energy = self._energy_from_k_maps(
            slab_k, ads_k, slab_desc, ads_desc, compute_uncertainty=compute_uncertainty
        )
        return LocalMCState(
            slab_atoms=slab_atoms.copy(),
            occupation=occupation.copy(),
            energy=energy,
            slab_k=slab_k,
            ads_k=ads_k,
            slab_desc=slab_desc,
            ads_desc=ads_desc,
        )

    def local_update(
        self,
        state: LocalMCState,
        candidate_slab_atoms,
        candidate_occupation: np.ndarray,
        *,
        changed_metal_indices: Iterable[int] = (),
        changed_site_ids: Iterable[int] = (),
        compute_uncertainty: bool = False,
    ) -> LocalMCState:
        candidate_occupation = validate_occupation(candidate_occupation, len(self.sites))
        changed_metal_indices = {int(i) for i in changed_metal_indices}
        changed_site_ids = {int(i) for i in changed_site_ids}
        for site_id in changed_site_ids:
            if site_id < 0 or site_id >= len(self.sites):
                raise IndexError(f"changed site_id {site_id} is out of range.")

        atoms = self.make_atoms(candidate_slab_atoms, candidate_occupation)

        slab_k = dict(state.slab_k)
        slab_desc = dict(state.slab_desc)
        ads_k = {site_id: k for site_id, k in state.ads_k.items() if bool(candidate_occupation[site_id])}
        ads_desc = {site_id: d for site_id, d in state.ads_desc.items() if bool(candidate_occupation[site_id])}

        affected_slab = self._affected_slab_centers(candidate_slab_atoms, changed_metal_indices)
        new_slab_k, new_slab_desc = self._slab_k_rows(candidate_slab_atoms, affected_slab)
        slab_k.update(new_slab_k)
        slab_desc.update(new_slab_desc)

        affected_ads = self._affected_ads_sites(
            candidate_slab_atoms=candidate_slab_atoms,
            old_occupation=state.occupation,
            candidate_occupation=candidate_occupation,
            changed_metal_indices=changed_metal_indices,
            changed_site_ids=changed_site_ids,
        )
        affected_ads_occupied = [
            int(site_id)
            for site_id in sorted(affected_ads)
            if bool(candidate_occupation[int(site_id)])
        ]
        for site_id in affected_ads:
            if not bool(candidate_occupation[int(site_id)]):
                ads_k.pop(int(site_id), None)
                ads_desc.pop(int(site_id), None)
        new_ads_k, new_ads_desc = self._ads_k_rows(
            atoms=atoms,
            slab_atoms=candidate_slab_atoms,
            occupation=candidate_occupation,
            site_ids=affected_ads_occupied,
        )
        ads_k.update(new_ads_k)
        ads_desc.update(new_ads_desc)

        energy = self._energy_from_k_maps(
            slab_k, ads_k, slab_desc, ads_desc, compute_uncertainty=compute_uncertainty
        )
        return LocalMCState(
            slab_atoms=candidate_slab_atoms.copy(),
            occupation=candidate_occupation.copy(),
            energy=energy,
            slab_k=slab_k,
            ads_k=ads_k,
            slab_desc=slab_desc,
            ads_desc=ads_desc,
        )


    def _descriptor_k_rows(self, model: SparseAtomicGPR, descriptor) -> torch.Tensor:
        x = torch.as_tensor(descriptor, dtype=torch.float64, device=model.x_M.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"descriptor must be 2D, got shape {tuple(x.shape)}.")
        if x.shape[1] != model.x_M.shape[1]:
            raise ValueError(
                f"Descriptor dimension mismatch: got {x.shape[1]}, expected {model.x_M.shape[1]}."
            )
        if x.shape[0] == 0:
            return torch.empty((0, model.x_M.shape[0]), dtype=torch.float64, device=model.x_M.device)
        return model.rbf_kernel(x, model.x_M).detach()

    def _slab_k_rows(
        self, slab_atoms, atom_indices: Iterable[int]
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:

        atom_indices = [int(i) for i in atom_indices]
        if not atom_indices:
            return {}, {}
        atom_indices = sorted(set(atom_indices))
        desc = self.slab_extractor(slab_atoms, atom_indices=atom_indices)
        desc_t = torch.as_tensor(desc, dtype=torch.float64, device=self.slab_model.x_M.device)
        k_rows = self._descriptor_k_rows(self.slab_model, desc_t)
        if k_rows.shape[0] != len(atom_indices):
            raise RuntimeError(
                f"Number of slab kernel rows does not match requested centers: "
                f"got {k_rows.shape[0]}, expected {len(atom_indices)}."
            )
        k_map = {int(atom_index): k_rows[pos] for pos, atom_index in enumerate(atom_indices)}
        desc_map = {int(atom_index): desc_t[pos] for pos, atom_index in enumerate(atom_indices)}
        return k_map, desc_map

    def _slab_k_row(self, slab_atoms, atom_index: int) -> torch.Tensor:
        k_map, _ = self._slab_k_rows(slab_atoms, [int(atom_index)])
        return k_map[int(atom_index)]

    def _ads_k_rows(
        self,
        atoms,
        slab_atoms,
        occupation: np.ndarray,
        site_ids: Iterable[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """One descriptor row per occupied CO's carbon atom (additive - see
        the class docstring). Every site_id here must already be occupied in
        `occupation`."""

        site_ids = [int(site_id) for site_id in site_ids]
        if not site_ids:
            return {}, {}
        site_ids = sorted(set(site_ids))

        c_map = carbon_index_by_site_id(len(slab_atoms), occupation)
        carbon_indices = []
        valid_site_ids = []
        for site_id in site_ids:
            if site_id not in c_map:
                raise RuntimeError(f"site {site_id} is not occupied and has no carbon row.")
            valid_site_ids.append(int(site_id))
            carbon_indices.append(int(c_map[int(site_id)]))

        desc = self.ads_extractor(atoms, atom_indices=carbon_indices)
        desc_t = torch.as_tensor(desc, dtype=torch.float64, device=self.ads_model.x_M.device)
        k_rows = self._descriptor_k_rows(self.ads_model, desc_t)
        if k_rows.shape[0] != len(valid_site_ids):
            raise RuntimeError(
                f"Number of ads kernel rows does not match requested sites: "
                f"got {k_rows.shape[0]}, expected {len(valid_site_ids)}."
            )
        k_map = {site_id: k_rows[pos] for pos, site_id in enumerate(valid_site_ids)}
        desc_map = {site_id: desc_t[pos] for pos, site_id in enumerate(valid_site_ids)}
        return k_map, desc_map

    def _ads_k_row(self, atoms, slab_atoms, occupation: np.ndarray, site_id: int) -> torch.Tensor:
        k_map, _ = self._ads_k_rows(atoms, slab_atoms, occupation, [int(site_id)])
        return k_map[int(site_id)]

    def _build_all_slab_rows(self, slab_atoms) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        return self._slab_k_rows(slab_atoms, range(len(slab_atoms)))

    def _build_all_ads_rows(
        self, atoms, slab_atoms, occupation: np.ndarray
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        occupied_site_ids = np.where(occupation)[0].astype(int)
        return self._ads_k_rows(atoms, slab_atoms, occupation, occupied_site_ids)


    def _affected_slab_centers(self, slab_atoms, changed_metal_indices: set[int]) -> set[int]:
        if not changed_metal_indices:
            return set()
        if not np.isfinite(self.slab_cutoff):
            return set(range(len(slab_atoms)))

        positions = slab_atoms.get_positions()
        changed_idx = np.array(sorted(changed_metal_indices), dtype=int)
        pbc = [bool(x) for x in slab_atoms.pbc]

        # One vectorized ASE call instead of a Python double loop over every
        # (center, changed) pair each calling mic_distance -> find_mic
        # individually: find_mic's general path re-runs a Minkowski cell
        # reduction on EVERY call, which is invariant across all of these
        # pairs (the cell doesn't change) - doing that once per changed-atom
        # batch instead of once per pair was the actual bottleneck (87k
        # mic_distance calls / ~20s of a 100-step, 384-atom MC run).
        _, dist = get_distances(positions[changed_idx], positions, cell=slab_atoms.cell, pbc=pbc)
        affected = set(np.where(np.any(dist <= self.slab_cutoff, axis=0))[0].astype(int).tolist())
        affected.update(int(i) for i in changed_metal_indices)
        return affected

    def _changed_co_positions(self, changed_site_ids: set[int]) -> list[np.ndarray]:
        return [np.asarray(self.sites[int(site_id)].position, dtype=np.float64) for site_id in changed_site_ids]

    def _affected_ads_sites(
        self,
        candidate_slab_atoms,
        old_occupation: np.ndarray,
        candidate_occupation: np.ndarray,
        changed_metal_indices: set[int],
        changed_site_ids: set[int],
    ) -> set[int]:
        n_old = int(old_occupation.sum())
        n_new = int(candidate_occupation.sum())
        if n_new <= 1:
            return set(np.where(old_occupation | candidate_occupation)[0].astype(int))
        if n_old <= 1:
            return set(np.where(candidate_occupation)[0].astype(int))

        occupied = sorted(int(i) for i in np.where(candidate_occupation)[0])
        affected = set(changed_site_ids)

        if not np.isfinite(self.ads_cutoff):
            affected.update(occupied)
            return affected

        positions = candidate_slab_atoms.get_positions()
        query_positions = list(self._changed_co_positions(changed_site_ids))
        query_positions.extend(positions[int(i)] for i in changed_metal_indices)
        if not query_positions:
            return affected

        site_positions = np.asarray([self.sites[sid].position for sid in occupied], dtype=np.float64)
        pbc = [bool(x) for x in candidate_slab_atoms.pbc]
        _, dist = get_distances(
            np.asarray(query_positions, dtype=np.float64),
            site_positions,
            cell=candidate_slab_atoms.cell,
            pbc=pbc,
        )
        hit = np.any(dist <= self.ads_cutoff, axis=0)
        affected.update(site_id for site_id, is_hit in zip(occupied, hit) if is_hit)
        return affected


    @staticmethod
    def _sum_k(row_map: dict[int, torch.Tensor], model: SparseAtomicGPR) -> torch.Tensor:
        if not row_map:
            return torch.zeros(model.x_M.shape[0], dtype=torch.float64, device=model.x_M.device)
        rows = [row.to(dtype=torch.float64, device=model.x_M.device) for row in row_map.values()]
        return torch.stack(rows, dim=0).sum(dim=0)

    @staticmethod
    def _mean_from_k(
        model: SparseAtomicGPR,
        k_sum: torch.Tensor,
        desc_map: dict[int, torch.Tensor] | None = None,
    ) -> float:
        value = k_sum.to(model.x_M.device) @ model.c.to(model.x_M.device)
        # The kernel part of forward() only ever predicts the RESIDUAL from
        # the model's mean function (see gpr.py's fit_c/forward) - add that
        # back here too, from the cached raw descriptor rows, or a
        # mean_function="linear" model's energy would silently be missing
        # its (usually dominant, for anything past the training envelope)
        # linear trend term.
        linear_mean = getattr(model, "linear_mean", None)
        if linear_mean is not None and desc_map:
            desc_sum = torch.stack(
                [d.to(dtype=linear_mean.dtype, device=linear_mean.device) for d in desc_map.values()],
                dim=0,
            ).sum(dim=0)
            value = value + desc_sum @ linear_mean
        return finite_float(value.detach().cpu().item(), "component_energy")

    @staticmethod
    def _mean_std_from_desc_and_k(
        model: SparseAtomicGPR, desc_map: dict[int, torch.Tensor], k_sum: torch.Tensor
    ) -> tuple[float, float]:
        """Exact Projected-Process variance (same formula as
        SparseAtomicGPR.predict_uncertainty in gpr.py - see that method's
        docstring), computed from this component's cached kernel-row sum
        (k_sum, equivalent to a row of build_K_NM) plus its cached RAW
        descriptor rows (desc_map - one per cached atom/site, NOT
        kernel-projected). The self-kernel term k(x*,x*) genuinely needs
        those raw rows: it's the only term in the PP formula that isn't
        expressible from k_sum alone, and it's also the term that carries
        "how far this structure is from anything in x_M" - dropping it (the
        previous version of this function did, via an SoR-style
        approximation, back when it also referenced the now-removed
        N x N `L_KSS`) silently throws away exactly the extrapolation
        signal this is meant to catch.
        """
        k_sum = k_sum.to(dtype=torch.float64, device=model.x_M.device)
        mean = LocalDescriptorEnergyEvaluator._mean_from_k(model, k_sum, desc_map)

        if (
            getattr(model, "L_KMM", None) is None
            or getattr(model, "L_A", None) is None
            or not desc_map
        ):
            return mean, float("nan")

        desc_stack = torch.stack(
            [d.to(dtype=torch.float64, device=model.x_M.device) for d in desc_map.values()], dim=0
        )

        K_test = k_sum.reshape(1, -1)  # (1, M)
        K_diag_true = model.rbf_kernel(desc_stack, desc_stack).sum().reshape(1)

        K_MM_inv_k_starM = torch.cholesky_solve(K_test.T, model.L_KMM)  # (M, 1)
        Q_diag = (K_test * K_MM_inv_k_starM.T).sum(dim=1)

        A_inv_k_starM = torch.cholesky_solve(K_test.T, model.L_A)  # (M, 1)
        pp_correction = model.sigma2 * (K_test * A_inv_k_starM.T).sum(dim=1)

        var = torch.clamp(K_diag_true - Q_diag + pp_correction, min=1e-12)
        std = torch.sqrt(var)[0].detach().cpu().item()
        return mean, finite_float(std, "component_uncertainty")

    def _energy_from_k_maps(
        self,
        slab_k: dict[int, torch.Tensor],
        ads_k: dict[int, torch.Tensor],
        slab_desc: dict[int, torch.Tensor] | None = None,
        ads_desc: dict[int, torch.Tensor] | None = None,
        *,
        compute_uncertainty: bool = False,
    ) -> EnergyComponents:
        slab_sum = self._sum_k(slab_k, self.slab_model)
        ads_sum = self._sum_k(ads_k, self.ads_model)

        if compute_uncertainty:
            slab_e, slab_u = self._mean_std_from_desc_and_k(self.slab_model, slab_desc or {}, slab_sum)
            ads_e, ads_u = (
                (0.0, 0.0) if not ads_k else self._mean_std_from_desc_and_k(self.ads_model, ads_desc or {}, ads_sum)
            )
            if np.isfinite(slab_u) and np.isfinite(ads_u):
                unc = float(np.sqrt(slab_u**2 + ads_u**2))
            else:
                unc = float("nan")
        else:
            slab_e = self._mean_from_k(self.slab_model, slab_sum, slab_desc)
            ads_e = 0.0 if not ads_k else self._mean_from_k(self.ads_model, ads_sum, ads_desc)
            unc = float("nan")
            slab_u = float("nan")
            ads_u = float("nan")

        return EnergyComponents(
            total=slab_e + ads_e,
            slab=slab_e,
            ads=ads_e,
            uncertainty=unc,
            uncertainty_slab=slab_u,
            uncertainty_ads=ads_u,
        )






def _scalar_to_float(value, name: str) -> float:
    """Convert a scalar-like tensor/array/object to float."""
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).sum().item()
    else:
        value = np.asarray(value, dtype=float).reshape(-1).sum()
    return finite_float(value, name)


class CalculatorEnergyEvaluator:
    """Full-structure evaluator based on CalculatorCESparseGPR.

    This is intentionally slower than LocalDescriptorEnergyEvaluator: every
    trial move rebuilds the adsorbed ASE structure and asks the calculator to
    recompute all required descriptors.  The MC move set and output format are
    kept identical to the local-descriptor version, which makes this script a
    useful correctness/reference implementation.
    """

    def __init__(
        self,
        file_slab_model: str,
        file_ads_model: str,
        sites: Iterable[AdsorptionSite],
        co_bond: float,
        device: str = "cpu",
        allow_unsafe_load: bool = False,
        local_cutoff_margin: float = 0.05,
        file_freq_model: str | None = None,
    ):
        self.device = device
        self.sites = validate_sites(list(sites))
        self.co_bond = positive_float(co_bond, "co_bond")
        self.allow_unsafe_load = bool(allow_unsafe_load)
        self.local_cutoff_margin = nonnegative_float(local_cutoff_margin, "local_cutoff_margin")
        self.file_slab_model = file_slab_model
        self.file_ads_model = file_ads_model

        signature = inspect.signature(CalculatorCESparseGPR)
        kwargs = {
            "file_slab_model": file_slab_model,
            "file_ads_model": file_ads_model,
            "device": device,
        }
        if "allow_unsafe_load" in signature.parameters:
            kwargs["allow_unsafe_load"] = bool(allow_unsafe_load)

        self.calculator = CalculatorCESparseGPR(**kwargs)

        # Stub: a future CO-vibrational-frequency model. Loaded here so its
        # checkpoint/config are validated early, but it does not yet
        # contribute to any energy/uncertainty below - that needs a defined
        # physical role (e.g. a ZPE/entropy correction to delta_mu) first.
        self.file_freq_model = file_freq_model
        self.freq_model: SparseAtomicGPR | None = (
            _load_sparse_model(file_freq_model, device, allow_unsafe_load)
            if file_freq_model is not None
            else None
        )
        self.freq_extractor = ClusterExpansion(self.freq_model.config) if self.freq_model is not None else None

    def reload_model(self, component: str, model_path: str) -> None:
        """Hot-swap one component's checkpoint (slab/ads) after an
        active-learning retraining cycle, without reconstructing the whole
        evaluator/calculator."""
        if component not in ("slab", "ads"):
            raise ValueError(f"Unknown component {component!r}; expected 'slab' or 'ads'.")

        model = _load_sparse_model(model_path, self.device, self.allow_unsafe_load)
        setattr(self.calculator, f"{component}_model", model)
        setattr(self.calculator, f"{component}_extractor", ClusterExpansion(model.config))

    def make_atoms(self, slab_atoms, occupation: np.ndarray):
        return build_adsorbed_structure(
            slab_atoms=slab_atoms,
            sites=self.sites,
            occupation=occupation,
            co_bond=self.co_bond,
        )

    def cutoffs(self) -> dict[str, float]:
        return {"slab": float("nan"), "ads": float("nan")}

    def initial_state(
        self,
        slab_atoms,
        occupation: np.ndarray | None = None,
        compute_uncertainty: bool = False,
    ) -> LocalMCState:
        if occupation is None:
            occupation = np.zeros(len(self.sites), dtype=bool)
        else:
            occupation = validate_occupation(occupation, len(self.sites))
        return self.full_rebuild(slab_atoms, occupation, compute_uncertainty=compute_uncertainty)

    def full_rebuild(self, slab_atoms, occupation: np.ndarray, compute_uncertainty: bool = False) -> LocalMCState:
        occupation = validate_occupation(occupation, len(self.sites))
        validate_sites(self.sites, n_atoms=len(slab_atoms))
        atoms = self.make_atoms(slab_atoms, occupation)
        energy = self._evaluate_atoms(atoms, compute_uncertainty=compute_uncertainty)
        return LocalMCState(
            slab_atoms=slab_atoms.copy(),
            occupation=occupation.copy(),
            energy=energy,
            slab_k={},
            ads_k={},
        )

    def local_update(
        self,
        state: LocalMCState,
        candidate_slab_atoms,
        candidate_occupation: np.ndarray,
        *,
        changed_metal_indices: Iterable[int] = (),
        changed_site_ids: Iterable[int] = (),
        compute_uncertainty: bool = False,
    ) -> LocalMCState:
        # The Calculator version deliberately ignores the local invalidation
        # hints and evaluates the complete candidate structure.
        return self.full_rebuild(
            candidate_slab_atoms,
            candidate_occupation,
            compute_uncertainty=compute_uncertainty,
        )

    def _evaluate_atoms(self, atoms, compute_uncertainty: bool = False) -> EnergyComponents:
        if compute_uncertainty:
            result = self.calculator.predict_energy_and_uncertainty(atoms)
            slab_energy, total_energy, ads_energy, total_uncertainty, component_uncertainties = result[:5]
            slab_unc = component_uncertainties.get("slab", float("nan"))
            ads_unc = component_uncertainties.get("ads", float("nan"))
            return EnergyComponents(
                total=_scalar_to_float(total_energy, "total_energy"),
                slab=_scalar_to_float(slab_energy, "slab_energy"),
                ads=_scalar_to_float(ads_energy, "ads_energy"),
                uncertainty=_scalar_to_float(total_uncertainty, "total_uncertainty"),
                uncertainty_slab=_scalar_to_float(slab_unc, "slab_uncertainty"),
                uncertainty_ads=_scalar_to_float(ads_unc, "ads_uncertainty"),
            )

        slab_energy, total_energy, ads_energy = self.calculator(atoms)
        return EnergyComponents(
            total=_scalar_to_float(total_energy, "total_energy"),
            slab=_scalar_to_float(slab_energy, "slab_energy"),
            ads=_scalar_to_float(ads_energy, "ads_energy"),
            uncertainty=float("nan"),
            uncertainty_slab=float("nan"),
            uncertainty_ads=float("nan"),
        )


def _fmt_table_float(value: float, width: int = 12, precision: int = 5, scientific: bool = False) -> str:
    value = float(value)
    if np.isnan(value):
        return f"{'nan':>{width}}"
    if scientific:
        return f"{value:{width}.{precision}e}"
    return f"{value:{width}.{precision}f}"


def progress_table_header() -> str:
    return (
        f"{'step':>8} "
        f"{'E_total':>14} {'E_slab':>14} {'E_ads':>14} "
        f"{'unc_tot':>12} {'unc_slab':>12} {'unc_ads':>12} "
        f"{'N_CO':>5} {'theta':>8} "
        f"{'acc_alloy':>10} {'acc_CO':>8} {'acc_ins':>8} {'acc_del':>8} {'acc_mig':>8}"
    )


def progress_table_row(step: int, state: LocalMCState, coverage_denominator: int, stats: MCStats | None = None) -> str:
    n_co = int(state.occupation.sum())
    theta = co_coverage(state.occupation, coverage_denominator)

    if stats is None:
        acc_alloy = acc_co = acc_ins = acc_del = acc_mig = float("nan")
    else:
        acc_alloy = stats.alloy_acceptance
        acc_co = stats.co_acceptance
        acc_ins = stats.co_insert_acceptance
        acc_del = stats.co_delete_acceptance
        acc_mig = stats.co_migration_acceptance

    e = state.energy
    return (
        f"{int(step):8d} "
        f"{_fmt_table_float(e.total, 14, 6)} "
        f"{_fmt_table_float(e.slab, 14, 6)} "
        f"{_fmt_table_float(e.ads, 14, 6)} "
        f"{_fmt_table_float(e.uncertainty, 12, 4, scientific=True)} "
        f"{_fmt_table_float(getattr(e, 'uncertainty_slab', float('nan')), 12, 4, scientific=True)} "
        f"{_fmt_table_float(getattr(e, 'uncertainty_ads', float('nan')), 12, 4, scientific=True)} "
        f"{n_co:5d} {theta:8.4f} "
        f"{_fmt_table_float(acc_alloy, 10, 3)} "
        f"{_fmt_table_float(acc_co, 8, 3)} "
        f"{_fmt_table_float(acc_ins, 8, 3)} "
        f"{_fmt_table_float(acc_del, 8, 3)} "
        f"{_fmt_table_float(acc_mig, 8, 3)}"
    )


class ActiveLearningController:
    """Synchronous active-learning retraining loop.

    When the running MC's slab/ads uncertainty exceeds the threshold
    configured for that component, the whole MC run pauses: a DFT relaxation
    is launched (blocking) via `run_script`, its result is folded into that
    component's own ase.db, the corresponding SparseAtomicGPR is retrained
    via ce_gpr_train.run(), and the freshly retrained checkpoint is hot
    swapped into the running evaluator before MC resumes - see run_cycle.

    An "ads" trigger costs TWO DFT relaxations, not one: the CO-covered
    structure, and this same alloy with every CO removed (the bare-slab
    reference e_ads_total needs). Only the CO-covered one is kept - the
    bare-slab one is used solely as a reference value and is NOT folded
    into the slab dataset (that dataset only grows on a slab-triggered
    cycle, deliberately, to keep the two components' retraining decoupled).
    """

    COMPONENTS = ("slab", "ads")

    def __init__(self, cfg: dict):
        self.thresholds = {
            name: float(cfg.get("uncertainty_thresholds", {}).get(name, float("inf")))
            for name in self.COMPONENTS
        }
        self.run_script = str(cfg["run_script"])
        self.run_dir = Path(cfg.get("run_dir", "active_learning_runs"))
        self.poscar_name = cfg.get("poscar_name", "in.poscar")
        self.finished_marker = cfg.get("finished_marker", "final.traj")
        self.energy_file = cfg.get("energy_file", "final.e")
        self.datasets: dict[str, str] = dict(cfg["datasets"])
        self.train_configs: dict[str, str] = dict(cfg["train_configs"])
        self.co_gas_outcar = cfg.get("co_gas_outcar")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Resume numbering after whatever job_NNNNN dirs already exist (e.g.
        # from a previous run against the same run_dir) instead of always
        # starting at job_00001 and colliding with them.
        self._counter = self._max_existing_job_number()
        self._co_energy: float | None = None

    def triggered_component(self, energy: EnergyComponents, n_co: int) -> str | None:
        """Return the first component whose uncertainty exceeds its own
        threshold, or None. ads needs at least one occupied site."""
        n_co = int(n_co)
        checks = (
            ("slab", energy.uncertainty_slab),
            ("ads", energy.uncertainty_ads if n_co >= 1 else float("nan")),
        )
        for name, value in checks:
            if np.isfinite(value) and value > self.thresholds[name]:
                return name
        return None

    def _co_gas_energy(self) -> float:
        if self._co_energy is None:
            import make_database as mdb

            self._co_energy = (
                mdb.load_co_gas_energy(self.co_gas_outcar)
                if self.co_gas_outcar
                else mdb.load_co_gas_energy()
            )
        return self._co_energy

    def _max_existing_job_number(self) -> int:
        max_n = 0
        for path in self.run_dir.glob("job_*"):
            try:
                n = int(path.name.removeprefix("job_"))
            except ValueError:
                continue
            max_n = max(max_n, n)
        return max_n

    def _run_dft(self, atoms) -> tuple:
        """Write `atoms` as the run script's expected input, launch it
        (blocking) in a fresh job directory, and return (relaxed_atoms,
        energy) once it finishes."""
        self._counter += 1
        job_dir = self.run_dir / f"job_{self._counter:05d}"
        job_dir.mkdir(parents=True, exist_ok=False)

        write(str(job_dir / self.poscar_name), atoms, format="vasp")

        print(f"[active learning] launching {self.run_script} in {job_dir} ...", flush=True)
        result = subprocess.run(
            [sys.executable, str(Path(self.run_script).resolve())],
            cwd=str(job_dir),
            capture_output=True,
            text=True,
        )
        (job_dir / "run_stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (job_dir / "run_stderr.log").write_text(result.stderr or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                f"run.py failed in {job_dir} (exit code {result.returncode}). "
                f"See {job_dir / 'run_stderr.log'}."
            )

        marker_path = job_dir / self.finished_marker
        if not marker_path.exists():
            raise RuntimeError(
                f"run.py exited cleanly but {self.finished_marker!r} is missing in {job_dir}: "
                "the relaxation did not actually finish."
            )

        relaxed_atoms = read(str(marker_path))
        energy_path = job_dir / self.energy_file
        if energy_path.exists():
            energy = float(energy_path.read_text().strip())
        else:
            energy = float(relaxed_atoms.get_potential_energy())

        print(f"[active learning] {job_dir}: DFT energy = {energy:.6f} eV", flush=True)
        return relaxed_atoms, energy

    def run_cycle(
        self,
        component: str,
        evaluator: "CalculatorEnergyEvaluator",
        state: "LocalMCState",
    ) -> None:
        import make_database as mdb

        n_co = int(np.asarray(state.occupation, dtype=bool).sum())

        if component == "slab":
            # The slab model is trained on bare-alloy DFT points only - probe
            # it at zero CO coverage regardless of the MC's current occupation.
            atoms = evaluator.make_atoms(state.slab_atoms, np.zeros_like(state.occupation))
            relaxed_atoms, energy = self._run_dft(atoms)
            mdb.append_row_to_db(self.datasets["slab"], relaxed_atoms, energy)

        else:
            atoms = evaluator.make_atoms(state.slab_atoms, state.occupation)
            relaxed_atoms, energy = self._run_dft(atoms)

            # Real DFT slab-only reference: same formula as
            # compute_e_ads_total_exact, but the bare-slab counterpart (this
            # alloy with every CO removed) is now a SECOND _run_dft call
            # instead of a slab-model prediction. The prediction shortcut let
            # the slab model's own error leak straight into e_ads_total -
            # e.g. two al_test_ads.db rows ended up with e_ads_total > +80 eV
            # (physically should be ~-10 eV at their n_co=8) because the slab
            # model badly mispredicted that alloy composition. This costs one
            # extra DFT relaxation per "ads"-triggered cycle. Used only as a
            # reference value here - NOT folded into the slab dataset (that
            # would grow it on every "ads" trigger too, coupling the two
            # components' retraining, which we don't want).
            bare_atoms = evaluator.make_atoms(state.slab_atoms, np.zeros_like(state.occupation))
            _, slab_energy = self._run_dft(bare_atoms)

            co_energy = self._co_gas_energy()

            e_ads_total = energy - (slab_energy + n_co * co_energy)
            mdb.append_row_to_db(
                self.datasets["ads"], relaxed_atoms, energy, n_co=n_co, e_ads_total=e_ads_total
            )

        print(f"[active learning] retraining {component} model ...", flush=True)
        # On this train_config's very first cycle (its own output.dir has no
        # checkpoint yet), let warm_start bootstrap from whatever model the
        # running MC evaluator was ACTUALLY deployed with, instead of cold-
        # starting from the config's init_lengthscale/"auto" - see
        # resolve_warm_start's fallback_checkpoint_path docstring.
        live_model_path = getattr(evaluator, f"file_{component}_model", None)
        # save_plot=False: skip writing the parity PDF - each call spawns a
        # fresh kaleido (headless-Chromium) subprocess for fig.write_image,
        # and repeated calls across many AL retraining cycles in this same
        # long-lived process have been observed to eventually hang.
        _, new_model_path = ce_gpr_train.run(
            self.train_configs[component], warm_start_fallback=live_model_path, save_plot=False
        )
        print(f"[active learning] retrained {component} model -> {new_model_path}", flush=True)
        evaluator.reload_model(component, str(new_model_path))


class GrandCO_MC:
    def __init__(
        self,
        slab_atoms,
        sites,
        evaluator: LocalDescriptorEnergyEvaluator,
        temperature: float,
        delta_mu: float,
        min_co_distance: float,
        rng: np.random.Generator,
        *,
        active_learning: ActiveLearningController | None = None,
        initial_occupation: np.ndarray | None = None,
        layer_ids: np.ndarray | None = None,
        frozen_atom_mask: np.ndarray | None = None,
        frozen_layers: set[int] | None = None,
    ):
        self.sites = validate_sites(list(sites), n_atoms=len(slab_atoms))
        if len(self.sites) == 0:
            raise ValueError("No adsorption sites were found.")

        self.coverage_denominator = adsorption_coverage_denominator(self.sites)
        self.evaluator = evaluator
        self.temperature = positive_float(temperature, "temperature")
        self.beta = 1.0 / (KB_EV * self.temperature)
        self.delta_mu = finite_float(delta_mu, "delta_mu")
        self.min_co_distance = nonnegative_float(min_co_distance, "min_co_distance")
        self.rng = rng
        self.active_learning = active_learning

        self.slab_atoms = slab_atoms.copy()

        if layer_ids is None:
            self.layer_ids = None
        else:
            self.layer_ids = np.asarray(layer_ids, dtype=int).copy()
            if self.layer_ids.shape != (len(self.slab_atoms),):
                raise ValueError(
                    f"layer_ids must have shape ({len(self.slab_atoms)},), got {self.layer_ids.shape}."
                )

        if frozen_atom_mask is None:
            self.frozen_atom_mask = np.zeros(len(self.slab_atoms), dtype=bool)
        else:
            self.frozen_atom_mask = np.asarray(frozen_atom_mask, dtype=bool).copy()
            if self.frozen_atom_mask.shape != (len(self.slab_atoms),):
                raise ValueError(
                    f"frozen_atom_mask must have shape ({len(self.slab_atoms)},), "
                    f"got {self.frozen_atom_mask.shape}."
                )

        self.mobile_atom_indices = np.where(~self.frozen_atom_mask)[0].astype(int)
        self.frozen_layers = set() if frozen_layers is None else {int(x) for x in frozen_layers}

        self.stats = MCStats()
        self.av_pd_sum = np.zeros(len(self.slab_atoms), dtype=np.float64)
        self.av_comp_count = 0
        if initial_occupation is None:
            self.initial_occupation = np.zeros(len(self.sites), dtype=bool)
        else:
            self.initial_occupation = validate_occupation(initial_occupation, len(self.sites))

    def initial_state(self) -> LocalMCState:
        return self.evaluator.initial_state(
            self.slab_atoms,
            occupation=self.initial_occupation,
            compute_uncertainty=True,
        )

    def make_atoms(self, state: LocalMCState):
        return self.evaluator.make_atoms(state.slab_atoms, state.occupation)

    def _valid_co_insertion(self, occupation: np.ndarray, site_id: int) -> bool:
        if self.min_co_distance <= 0.0:
            return True
        nearest = min_distance_to_occupied_sites(
            sites=self.sites,
            occupation=occupation,
            trial_site_id=int(site_id),
            cell=self.slab_atoms.cell,
            pbc=self.slab_atoms.pbc,
        )
        return nearest >= self.min_co_distance

    def _maybe_retrain(self, state: LocalMCState, candidate: LocalMCState) -> tuple[LocalMCState, LocalMCState]:
        """If `candidate`'s uncertainty trips a configured threshold, pause
        for a full active-learning cycle (DFT -> db -> retrain -> hot swap),
        then re-evaluate BOTH state and candidate from scratch under the
        (possibly now different) models, so the Metropolis test right after
        this compares energies computed with the same model generation."""
        if self.active_learning is None:
            return state, candidate

        n_co = int(np.asarray(candidate.occupation, dtype=bool).sum())
        component = self.active_learning.triggered_component(candidate.energy, n_co=n_co)
        if component is None:
            return state, candidate

        triggering_unc = getattr(candidate.energy, f"uncertainty_{component}")
        threshold = self.active_learning.thresholds[component]
        print(
            f"[active learning] triggered by component={component!r}: "
            f"uncertainty={triggering_unc:.6f} > threshold={threshold:.6f} "
            f"(N_CO={n_co}, on a PROPOSED candidate move - may or may not end up accepted)",
            flush=True,
        )

        self.active_learning.run_cycle(component, self.evaluator, candidate)

        state = self.evaluator.full_rebuild(state.slab_atoms, state.occupation, compute_uncertainty=True)
        candidate = self.evaluator.full_rebuild(
            candidate.slab_atoms, candidate.occupation, compute_uncertainty=True
        )
        return state, candidate

    def attempt_alloy_swap(self, state: LocalMCState) -> LocalMCState:
        self.stats.alloy_attempts += 1
        candidate_slab, pair = propose_pt_pd_swap(
            state.slab_atoms,
            rng=self.rng,
            allowed_indices=self.mobile_atom_indices,
        )
        if candidate_slab is None or pair is None:
            return state

        candidate = self.evaluator.local_update(
            state,
            candidate_slab_atoms=candidate_slab,
            candidate_occupation=state.occupation,
            changed_metal_indices=pair,
            changed_site_ids=(),
            compute_uncertainty=True,
        )
        state, candidate = self._maybe_retrain(state, candidate)

        d_omega = candidate.energy.total - state.energy.total
        if metropolis_hastings_accept(d_omega, beta=self.beta, log_q_reverse_over_forward=0.0, rng=self.rng):
            self.stats.alloy_accepts += 1
            return candidate
        return state

    def _select_co_event(self, occupation: np.ndarray, site_id: int) -> tuple[str, int | None, float, int]:

        n_sites = len(occupation)
        old_occ = np.asarray(occupation, dtype=bool)
        old_value = bool(old_occ[site_id])

        if not old_value:
            n_free_old = int((~old_occ).sum())
            return "insert", None, -np.log(float(n_free_old)), +1

        free_sites = np.where(~old_occ)[0].astype(int)
        events: list[tuple[str, int | None]] = [("delete", None)]
        events.extend(("migrate", int(target)) for target in free_sites)
        event, target = events[int(self.rng.integers(len(events)))]

        if event == "delete":
            log_q = np.log(float(len(events)))
            return "delete", None, log_q, -1

        return "migrate", int(target), 0.0, 0

    def attempt_co_move(self, state: LocalMCState) -> LocalMCState:
        self.stats.co_attempts += 1
        n_sites = len(self.sites)
        site_id = int(self.rng.integers(n_sites))
        old_occ = np.asarray(state.occupation, dtype=bool)
        candidate_occ = old_occ.copy()

        event, target, log_q_reverse_over_forward, delta_n = self._select_co_event(old_occ, site_id)

        if event == "insert":
            self.stats.co_insert_attempts += 1
            if not self._valid_co_insertion(old_occ, site_id):
                return state
            candidate_occ[site_id] = True
            changed_sites = {site_id}

        elif event == "delete":
            self.stats.co_delete_attempts += 1
            candidate_occ[site_id] = False
            changed_sites = {site_id}

        elif event == "migrate":
            self.stats.co_migration_attempts += 1
            if target is None:
                raise RuntimeError("Migration event has no target site.")
            occ_without_origin = old_occ.copy()
            occ_without_origin[site_id] = False
            if not self._valid_co_insertion(occ_without_origin, int(target)):
                return state
            candidate_occ[site_id] = False
            candidate_occ[int(target)] = True
            changed_sites = {site_id, int(target)}

        else:
            raise RuntimeError(f"Unknown CO event: {event}.")

        candidate = self.evaluator.local_update(
            state,
            candidate_slab_atoms=state.slab_atoms,
            candidate_occupation=candidate_occ,
            changed_metal_indices=(),
            changed_site_ids=changed_sites,
            compute_uncertainty=True,
        )
        state, candidate = self._maybe_retrain(state, candidate)

        dE = candidate.energy.total - state.energy.total
        dOmega = dE - self.delta_mu * delta_n
        if metropolis_hastings_accept(
            dOmega,
            beta=self.beta,
            log_q_reverse_over_forward=log_q_reverse_over_forward,
            rng=self.rng,
        ):
            self.stats.co_accepts += 1
            if event == "insert":
                self.stats.co_insert_accepts += 1
            elif event == "delete":
                self.stats.co_delete_accepts += 1
            elif event == "migrate":
                self.stats.co_migration_accepts += 1
            return candidate
        return state

    def accumulate_composition(self, state: LocalMCState) -> None:
        symbols = np.asarray(state.slab_atoms.get_chemical_symbols())
        self.av_pd_sum += (symbols == "Pd").astype(np.float64)
        self.av_comp_count += 1

    def get_av_comp(self) -> np.ndarray:
        if self.av_comp_count == 0:
            symbols = np.asarray(self.slab_atoms.get_chemical_symbols())
            return (symbols == "Pd").astype(np.float64)
        return self.av_pd_sum / float(self.av_comp_count)

    def current_atoms_with_info(self, state: LocalMCState, include_av_comp: bool = False):
        av_comp = self.get_av_comp() if include_av_comp else None
        return attach_mc_info(
            self.make_atoms(state),
            energy=state.energy,
            occupation=state.occupation,
            av_comp=av_comp,
            layer_ids=self.layer_ids,
            frozen_atom_mask=self.frozen_atom_mask,
            coverage_denominator=self.coverage_denominator,
        )

    def run(
        self,
        state: LocalMCState,
        nsteps: int,
        print_every: int,
        write_every: int,
        trajectory: str,
    ) -> LocalMCState:
        if trajectory:
            _ensure_parent_dir(trajectory)
            if os.path.exists(trajectory):
                os.remove(trajectory)

        self.accumulate_composition(state)
        if trajectory:
            write(trajectory, self.current_atoms_with_info(state, include_av_comp=True), format="extxyz", append=False)

        for step in range(1, int(nsteps) + 1):
            state = self.attempt_alloy_swap(state)
            state = self.attempt_co_move(state)

            self.accumulate_composition(state)

            if print_every > 0 and (step == 1 or step % print_every == 0 or step == nsteps):
                print(progress_table_row(step, state, self.coverage_denominator, self.stats), flush=True)

            if trajectory and write_every > 0 and step % write_every == 0:
                write(trajectory, self.current_atoms_with_info(state, include_av_comp=True), format="extxyz", append=True)

        return state


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    cli_parser = argparse.ArgumentParser(
        description=(
            "Serial Metropolis MC for Pt/Pd + semi-grand-canonical CO, driven by a "
            "single JSON config (see mc_grand.example.json). Evaluates trial moves via "
            "an incremental local-descriptor cache by default (evaluator_mode='local'); "
            "pass evaluator_mode='calculator' to instead rebuild the full descriptor "
            "through CalculatorCESparseGPR on every move (slower, useful as a "
            "correctness reference)."
        )
    )
    cli_parser.add_argument("config", help="Path to a JSON config file.")
    cli_args = cli_parser.parse_args()

    cfg = load_config(cli_args.config)
    args = namespace_from_config(cfg)
    validate_args(args)
    al_cfg = cfg.get("active_learning")
    validate_active_learning_config(al_cfg)

    _safe_set_torch_threads(1)
    rng = np.random.default_rng(args.seed)

    _ensure_parent_dir(args.trajectory)
    _ensure_parent_dir(args.output)

    slab_atoms, input_atoms = load_or_build_slab(args)

    layer_ids = assign_substrate_layers(slab_atoms, z_tol=args.layer_z_tol)
    n_layers = int(layer_ids.max()) + 1
    frozen_layers = frozen_layers_from_args(args, n_layers=n_layers)
    frozen_atom_mask = frozen_atom_mask_from_layers(layer_ids, frozen_layers)
    print_layer_report(slab_atoms, layer_ids, frozen_layers)
    print(
        f"Frozen substrate atoms: {int(frozen_atom_mask.sum())}/{len(frozen_atom_mask)}",
        flush=True,
    )

    sites = find_adsorption_sites(
        slab_atoms,
        height=args.co_height,
        z_atol=args.z_atol,
        bridge_cutoff_factor=args.bridge_cutoff_factor,
    )

    n_ontop = sum(site.kind == "ontop" for site in sites)
    n_bridge = sum(site.kind == "bridge" for site in sites)
    coverage_denominator = adsorption_coverage_denominator(sites)
    print(f"Number of adsorption sites: {len(sites)}", flush=True)
    print(f"  ontop sites  = {n_ontop}", flush=True)
    print(f"  bridge sites = {n_bridge}", flush=True)
    print(f"Coverage denominator for theta_CO = {coverage_denominator} surface atoms / ontop sites", flush=True)
    print(f"delta_mu = {args.delta_mu:.8f} eV", flush=True)

    initial_occupation = initial_occupation_from_input_adsorbates(
        input_atoms=input_atoms,
        slab_atoms=slab_atoms,
        sites=sites,
        match_tol=args.initial_site_match_tol,
        min_co_distance=args.min_co_distance,
        ignore_cell_adsorbates=args.ignore_cell_adsorbates,
    )
    initial_n_co = int(initial_occupation.sum())
    if initial_n_co > 0:
        print(
            f"Initial CO occupation inferred from input C atoms: "
            f"N_CO = {initial_n_co}, theta = {co_coverage(initial_occupation, coverage_denominator):.6f}",
            flush=True,
        )
    else:
        print("Initial CO occupation: N_CO = 0", flush=True)

    evaluator_cls = LocalDescriptorEnergyEvaluator if args.evaluator_mode == "local" else CalculatorEnergyEvaluator
    evaluator = evaluator_cls(
        file_slab_model=args.slab_model,
        file_ads_model=args.ads_model,
        file_freq_model=args.freq_model,
        sites=sites,
        co_bond=args.co_bond,
        device=args.device,
        allow_unsafe_load=args.allow_unsafe_model_load,
        local_cutoff_margin=args.local_cutoff_margin,
    )
    cutoffs = evaluator.cutoffs()
    if not any(np.isfinite(value) for value in cutoffs.values()):
        print(
            "energy evaluation mode: CalculatorCESparseGPR full-structure descriptor rebuild",
            flush=True,
        )
    else:
        print(
            "effective local descriptor invalidation cutoffs: "
            f"slab={cutoffs['slab']:.6f} A, "
            f"ads={cutoffs['ads']:.6f} A",
            flush=True,
        )

    active_learning = (
        ActiveLearningController(al_cfg)
        if al_cfg and al_cfg.get("enabled", False)
        else None
    )
    if active_learning is not None:
        print(
            "active learning: enabled, uncertainty thresholds = "
            f"{active_learning.thresholds}",
            flush=True,
        )

    mc = GrandCO_MC(
        slab_atoms=slab_atoms,
        sites=sites,
        evaluator=evaluator,
        temperature=args.temperature,
        delta_mu=args.delta_mu,
        min_co_distance=args.min_co_distance,
        rng=rng,
        active_learning=active_learning,
        initial_occupation=initial_occupation,
        layer_ids=layer_ids,
        frozen_atom_mask=frozen_atom_mask,
        frozen_layers=frozen_layers,
    )

    state = mc.initial_state()
    if not occupation_satisfies_min_distance(sites, state.occupation, args.min_co_distance, slab_atoms.cell, slab_atoms.pbc):
        raise RuntimeError("Initial occupation violates min_co_distance.")

    print("MC progress table:", flush=True)
    print(progress_table_header(), flush=True)
    print(progress_table_row(0, state, mc.coverage_denominator, None), flush=True)

    final_state = mc.run(
        state=state,
        nsteps=args.nsteps,
        print_every=args.print_every,
        write_every=args.write_every,
        trajectory=args.trajectory,
    )

    final_atoms = mc.current_atoms_with_info(final_state, include_av_comp=True)
    write(args.output, final_atoms, format="extxyz")

    print("MC finished cleanly", flush=True)
    print(f"Final N_CO = {int(final_state.occupation.sum())}", flush=True)
    print(f"Final theta_CO = {co_coverage(final_state.occupation, mc.coverage_denominator):.6f}", flush=True)
    print(f"Alloy acceptance = {mc.stats.alloy_acceptance:.6f}", flush=True)
    print(f"CO acceptance = {mc.stats.co_acceptance:.6f}", flush=True)
    print(f"CO insertion acceptance = {mc.stats.co_insert_acceptance:.6f}", flush=True)
    print(f"CO deletion acceptance = {mc.stats.co_delete_acceptance:.6f}", flush=True)
    print(f"CO migration acceptance = {mc.stats.co_migration_acceptance:.6f}", flush=True)
    print(f"Composition samples used for av_comp = {mc.av_comp_count}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
