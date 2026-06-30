from __future__ import annotations

import argparse
import inspect
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ase.io import read, write
from ase.io.trajectory import Trajectory

from mc_grand_utils import (
    KB_EV,
    AdsorptionSite,
    EnergyComponents,
    MCStats,
    attach_mc_info,
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


@dataclass
class LocalMCState:

    slab_atoms: object
    occupation: np.ndarray
    energy: EnergyComponents
    slab_k: dict[int, torch.Tensor]
    ads_k: dict[int, torch.Tensor]
    rep_k: dict[int, torch.Tensor]

    def __post_init__(self) -> None:
        self.occupation = np.asarray(self.occupation, dtype=bool).copy()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serial Metropolis MC for Pt/Pd + semi-grand-canonical CO. "
            "This version uses local descriptor-cache updates and does not call CalculatorCESparseGPR during MC."
        )
    )

    parser.add_argument(
        "--cell",
        default=None,
        help=(
            "Input structure readable by ASE. If it contains C atoms, their xy positions "
            "are mapped onto the generated ontop/bridge site grid and used as the initial CO occupation."
        ),
    )
    parser.add_argument("--cell-format", default=None, help="Optional ASE input format.")
    parser.add_argument(
        "--ignore-cell-adsorbates",
        action="store_true",
        help="Strip C/O from --cell or POSCAR and start from an empty CO occupation.",
    )
    parser.add_argument(
        "--initial-site-match-tol",
        type=float,
        default=0.35,
        help="Maximum xy distance in angstrom for mapping C atoms from the input cell onto MC adsorption sites.",
    )
    parser.add_argument("--supercell", type=int, nargs=3, metavar=("Nx", "Ny", "Nz"), default=(4, 4, 6))
    parser.add_argument("--lattice-constant", type=float, default=3.98398)
    parser.add_argument("--pd-fraction", type=float, default=0.5)

    parser.add_argument("--slab-model", required=True)
    parser.add_argument("--ads-model", required=True)
    parser.add_argument(
        "--rep-model",
        default=None,
        help=(
            "Optional CO-CO repulsion model. It is loaded lazily and used only "
            "when N_CO > 1. If omitted and the MC reaches N_CO > 1, the run stops."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-model-load",
        action="store_true",
        help="Use only for trusted local checkpoints if the installed SparseAtomicGPR supports it.",
    )

    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--delta-mu",
        type=float,
        default=0.0,
        help="CO chemical-potential offset in eV. Larger values favor adsorption.",
    )

    parser.add_argument("--nsteps", type=int, default=3000)
    parser.add_argument("--alloy-attempts-per-step", type=int, default=1)
    parser.add_argument("--co-attempts-per-step", type=int, default=1)

    parser.add_argument("--co-height", type=float, default=1.60)
    parser.add_argument("--co-bond", type=float, default=1.15)
    parser.add_argument(
        "--min-co-distance",
        type=float,
        default=1.52,
        help="Hard C-C xy exclusion distance in angstrom. Use 0 to disable.",
    )
    parser.add_argument("--z-atol", type=float, default=1e-3)
    parser.add_argument("--bridge-cutoff-factor", type=float, default=1.25)
    parser.add_argument(
        "--layer-z-tol",
        type=float,
        default=0.1,
        help=(
            "Tolerance in angstrom for grouping substrate atoms into z-layers. "
            "Layer ids are numbered from 0 at the bottom of the slab."
        ),
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Substrate layer ids to freeze during Pt/Pd swap MC. "
            "Layer numbering starts from 0 at the bottom of the slab, e.g. "
            "--freeze-layers 0 1 freezes the two bottom layers."
        ),
    )
    parser.add_argument(
        "--freeze-bottom-layers",
        type=int,
        default=0,
        help=(
            "Convenience option: freeze this many bottom substrate layers. "
            "For example, --freeze-bottom-layers 2 is equivalent to --freeze-layers 0 1."
        ),
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--write-every", type=int, default=1000)
    parser.add_argument("--trajectory", default="mc_semigrand_traj.xyz")
    parser.add_argument("--output", default="mc_semigrand_final.xyz")

    parser.add_argument(
        "--active-learning",
        action="store_true",
        help="Write trial structures whose cached GP uncertainty exceeds --uncertainty-threshold.",
    )
    parser.add_argument("--uncertainty-threshold", type=float, default=float("inf"))
    parser.add_argument("--active-learning-trajectory", default="active_learning_candidates.traj")

    parser.add_argument(
        "--local-cutoff-margin",
        type=float,
        default=0.05,
        help="Margin added to max(config.shells_dict upper bound) for conservative local invalidation.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_float(args.temperature, "temperature")
    finite_float(args.delta_mu, "delta_mu")
    positive_float(args.lattice_constant, "lattice_constant")
    if any(int(n) <= 0 for n in args.supercell):
        raise ValueError(f"all supercell dimensions must be positive, got {args.supercell}.")
    if not (0.0 <= float(args.pd_fraction) <= 1.0):
        raise ValueError(f"pd_fraction must be between 0 and 1, got {args.pd_fraction}.")

    for name in ("nsteps", "alloy_attempts_per_step", "co_attempts_per_step", "print_every", "write_every"):
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

    if args.active_learning:
        if args.active_learning_trajectory in (None, ""):
            raise ValueError("active_learning_trajectory must be non-empty in active-learning mode.")
        if np.isnan(float(args.uncertainty_threshold)):
            raise ValueError("uncertainty_threshold must not be NaN.")


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

    input_atoms = None

    if args.cell is not None:
        input_atoms = read(args.cell, format=args.cell_format)
        slab_atoms = strip_to_symbols(input_atoms, keep_symbols=("Pt", "Pd"))
    elif os.path.exists("POSCAR"):
        input_atoms = read("POSCAR", format="vasp")
        slab_atoms = strip_to_symbols(input_atoms, keep_symbols=("Pt", "Pd"))
    else:
        slab_atoms = build_supercell(
            structure="fcc",
            lattice_constant=args.lattice_constant,
            composition={"Pd": args.pd_fraction, "Pt": 1.0 - args.pd_fraction},
            supercell=np.diag(args.supercell),
            seed=args.seed,
        )
        slab_atoms.center(vacuum=15.0, axis=2)
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
        file_rep_model: str | None,
        sites: Iterable[AdsorptionSite],
        co_bond: float,
        device: str = "cpu",
        allow_unsafe_load: bool = False,
        local_cutoff_margin: float = 0.05,
    ):
        self.device = device
        self.sites = validate_sites(list(sites))
        self.co_bond = positive_float(co_bond, "co_bond")
        self.local_cutoff_margin = nonnegative_float(local_cutoff_margin, "local_cutoff_margin")
        self.allow_unsafe_load = bool(allow_unsafe_load)

        self.slab_model = _load_sparse_model(file_slab_model, device, allow_unsafe_load)
        self.ads_model = _load_sparse_model(file_ads_model, device, allow_unsafe_load)

        self.file_rep_model = file_rep_model
        self.rep_model: SparseAtomicGPR | None = None
        self.rep_extractor: ClusterExpansion | None = None

        self.slab_extractor = ClusterExpansion(self.slab_model.config)
        self.ads_extractor = ClusterExpansion(self.ads_model.config)

        self.slab_cutoff = cutoff_from_shells_dict(self.slab_model.config, self.local_cutoff_margin)
        self.ads_cutoff = cutoff_from_shells_dict(self.ads_model.config, self.local_cutoff_margin)
        self.rep_cutoff = float("nan")

        self._validate_model_elements()

    def _validate_model_elements(self) -> None:

        slab_elements = set(self.slab_model.config.elements)
        ads_elements = set(self.ads_model.config.elements)

        required_metals = {"Pt", "Pd"}

        if not required_metals.issubset(slab_elements):
            raise ValueError(
                f"slab model elements must include Pt and Pd, got {sorted(slab_elements)}."
            )

        if not required_metals.issubset(ads_elements):
            raise ValueError(
                "ads model is evaluated on metal-centered adsorption-site "
                f"descriptors and must include Pt and Pd; got {sorted(ads_elements)}."
            )

    def _require_rep_model(self) -> SparseAtomicGPR:
        """Load and validate rep_model only when a CO-CO term is actually needed."""
        if self.rep_model is not None:
            return self.rep_model

        if self.file_rep_model is None:
            raise RuntimeError(
                "N_CO > 1 requires a CO-CO repulsion model, but --rep-model was not provided. "
                "For N_CO = 0 or 1 the repulsion term is skipped automatically."
            )

        self.rep_model = _load_sparse_model(
            self.file_rep_model,
            self.device,
            self.allow_unsafe_load,
        )
        rep_elements = set(self.rep_model.config.elements)
        if "C" not in rep_elements:
            raise ValueError(
                "rep model is evaluated only when N_CO > 1 and must support "
                f"carbon-centered descriptors; got elements {sorted(rep_elements)}."
            )
        self.rep_extractor = ClusterExpansion(self.rep_model.config)
        self.rep_cutoff = cutoff_from_shells_dict(self.rep_model.config, self.local_cutoff_margin)
        return self.rep_model

    def make_atoms(self, slab_atoms, occupation: np.ndarray):
        return build_adsorbed_structure(
            slab_atoms=slab_atoms,
            sites=self.sites,
            occupation=occupation,
            co_bond=self.co_bond,
        )

    def cutoffs(self) -> dict[str, float]:
        return {"slab": self.slab_cutoff, "ads": self.ads_cutoff, "rep": self.rep_cutoff}

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

        slab_k = self._build_all_slab_rows(slab_atoms)
        ads_k = self._build_all_ads_rows(atoms, occupation)
        rep_k = self._build_all_rep_rows(atoms, slab_atoms, occupation)
        energy = self._energy_from_k_maps(slab_k, ads_k, rep_k, compute_uncertainty=compute_uncertainty)
        return LocalMCState(
            slab_atoms=slab_atoms.copy(),
            occupation=occupation.copy(),
            energy=energy,
            slab_k=slab_k,
            ads_k=ads_k,
            rep_k=rep_k,
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
        ads_k = {site_id: k for site_id, k in state.ads_k.items() if bool(candidate_occupation[site_id])}
        rep_k = {site_id: k for site_id, k in state.rep_k.items() if bool(candidate_occupation[site_id])}

        affected_slab = self._affected_slab_centers(candidate_slab_atoms, changed_metal_indices)
        slab_k.update(self._slab_k_rows(candidate_slab_atoms, affected_slab))

        affected_ads = self._affected_ads_sites(
            candidate_slab_atoms=candidate_slab_atoms,
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
        ads_k.update(self._ads_k_rows(atoms, affected_ads_occupied))

        affected_rep = self._affected_rep_sites(
            candidate_slab_atoms=candidate_slab_atoms,
            old_occupation=state.occupation,
            candidate_occupation=candidate_occupation,
            changed_metal_indices=changed_metal_indices,
            changed_site_ids=changed_site_ids,
        )
        n_co = int(candidate_occupation.sum())
        if n_co <= 1:
            # There is no CO-CO pair at N_CO <= 1.  The first CO is therefore
            # intentionally absent from rep_k until a second CO appears.
            rep_k = {}
        else:
            if int(state.occupation.sum()) <= 1:
                # Transition 1 -> 2 CO: both the old first CO and the new CO
                # must enter the repulsion cache.  This is the point where the
                # first CO starts contributing to the repulsion model.
                affected_rep = set(np.where(candidate_occupation)[0].astype(int))
            affected_rep_occupied = [
                int(site_id)
                for site_id in sorted(affected_rep)
                if bool(candidate_occupation[int(site_id)])
            ]
            for site_id in affected_rep:
                if not bool(candidate_occupation[int(site_id)]):
                    rep_k.pop(int(site_id), None)
            rep_k.update(
                self._rep_k_rows(
                    atoms=atoms,
                    slab_atoms=candidate_slab_atoms,
                    occupation=candidate_occupation,
                    site_ids=affected_rep_occupied,
                )
            )

        energy = self._energy_from_k_maps(slab_k, ads_k, rep_k, compute_uncertainty=compute_uncertainty)
        return LocalMCState(
            slab_atoms=candidate_slab_atoms.copy(),
            occupation=candidate_occupation.copy(),
            energy=energy,
            slab_k=slab_k,
            ads_k=ads_k,
            rep_k=rep_k,
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

    def _slab_k_rows(self, slab_atoms, atom_indices: Iterable[int]) -> dict[int, torch.Tensor]:

        atom_indices = [int(i) for i in atom_indices]
        if not atom_indices:
            return {}
        atom_indices = sorted(set(atom_indices))
        desc = self.slab_extractor(slab_atoms, atom_indices=atom_indices)
        k_rows = self._descriptor_k_rows(self.slab_model, desc)
        if k_rows.shape[0] != len(atom_indices):
            raise RuntimeError(
                f"Number of slab kernel rows does not match requested centers: "
                f"got {k_rows.shape[0]}, expected {len(atom_indices)}."
            )
        return {int(atom_index): k_rows[pos] for pos, atom_index in enumerate(atom_indices)}

    def _slab_k_row(self, slab_atoms, atom_index: int) -> torch.Tensor:
        rows = self._slab_k_rows(slab_atoms, [int(atom_index)])
        return rows[int(atom_index)]

    def _ads_k_rows(self, atoms, site_ids: Iterable[int]) -> dict[int, torch.Tensor]:

        site_ids = [int(site_id) for site_id in site_ids]
        if not site_ids:
            return {}
        site_ids = sorted(set(site_ids))

        metal_indices: list[int] = []
        for site_id in site_ids:
            if site_id < 0 or site_id >= len(self.sites):
                raise IndexError(f"site_id {site_id} is out of range.")
            metal_indices.extend(int(i) for i in self.sites[site_id].atom_indices)

        metal_indices_unique = sorted(set(metal_indices))
        if not metal_indices_unique:
            raise RuntimeError("Cannot build adsorption descriptors without site-defining metal atoms.")

        desc = self.ads_extractor(atoms, atom_indices=metal_indices_unique)
        desc_t = torch.as_tensor(desc, dtype=torch.float64, device=self.ads_model.x_M.device)
        if desc_t.ndim != 2:
            raise RuntimeError(f"ads descriptor block is not 2D, got shape {tuple(desc_t.shape)}.")
        if desc_t.shape[0] != len(metal_indices_unique):
            raise RuntimeError(
                f"ads descriptor block has {desc_t.shape[0]} rows, "
                f"expected {len(metal_indices_unique)}."
            )

        row_by_metal = {
            int(atom_index): desc_t[pos]
            for pos, atom_index in enumerate(metal_indices_unique)
        }
        site_desc_rows = []
        valid_site_ids = []
        for site_id in site_ids:
            site = self.sites[int(site_id)]
            rows = [row_by_metal[int(atom_index)] for atom_index in site.atom_indices]
            site_desc = torch.stack(rows, dim=0).sum(dim=0, keepdim=False)
            site_desc_rows.append(site_desc)
            valid_site_ids.append(int(site_id))

        site_desc_block = torch.stack(site_desc_rows, dim=0)
        k_rows = self._descriptor_k_rows(self.ads_model, site_desc_block)
        if k_rows.shape[0] != len(valid_site_ids):
            raise RuntimeError(
                f"Number of ads kernel rows does not match requested sites: "
                f"got {k_rows.shape[0]}, expected {len(valid_site_ids)}."
            )
        return {site_id: k_rows[pos] for pos, site_id in enumerate(valid_site_ids)}

    def _ads_site_descriptor(self, atoms, site_id: int):

        site = self.sites[int(site_id)]
        desc = self.ads_extractor(atoms, atom_indices=list(site.atom_indices))
        desc_t = torch.as_tensor(desc, dtype=torch.float64, device=self.ads_model.x_M.device)
        if desc_t.ndim != 2:
            raise RuntimeError(f"ads descriptor for site {site_id} is not 2D.")
        if desc_t.shape[0] != len(site.atom_indices):
            raise RuntimeError(
                f"ads descriptor for site {site_id} has {desc_t.shape[0]} rows, "
                f"expected {len(site.atom_indices)}."
            )
        return desc_t.sum(dim=0, keepdim=True)

    def _ads_k_row(self, atoms, site_id: int) -> torch.Tensor:
        rows = self._ads_k_rows(atoms, [int(site_id)])
        return rows[int(site_id)]

    def _rep_k_rows(
        self,
        atoms,
        slab_atoms,
        occupation: np.ndarray,
        site_ids: Iterable[int],
    ) -> dict[int, torch.Tensor]:

        site_ids = [int(site_id) for site_id in site_ids]
        if not site_ids:
            return {}
        site_ids = sorted(set(site_ids))

        rep_model = self._require_rep_model()
        if self.rep_extractor is None:
            raise RuntimeError("Internal error: rep_extractor was not initialized after loading rep_model.")

        c_map = carbon_index_by_site_id(len(slab_atoms), occupation)
        carbon_indices = []
        valid_site_ids = []
        for site_id in site_ids:
            if site_id not in c_map:
                raise RuntimeError(f"site {site_id} is not occupied and has no carbon row.")
            valid_site_ids.append(int(site_id))
            carbon_indices.append(int(c_map[int(site_id)]))

        desc = self.rep_extractor(atoms, atom_indices=carbon_indices)
        k_rows = self._descriptor_k_rows(rep_model, desc)
        if k_rows.shape[0] != len(valid_site_ids):
            raise RuntimeError(
                f"Number of rep kernel rows does not match requested sites: "
                f"got {k_rows.shape[0]}, expected {len(valid_site_ids)}."
            )
        return {site_id: k_rows[pos] for pos, site_id in enumerate(valid_site_ids)}

    def _rep_k_row(self, atoms, slab_atoms, occupation: np.ndarray, site_id: int) -> torch.Tensor:
        rows = self._rep_k_rows(atoms, slab_atoms, occupation, [int(site_id)])
        return rows[int(site_id)]

    def _build_all_slab_rows(self, slab_atoms) -> dict[int, torch.Tensor]:
        return self._slab_k_rows(slab_atoms, range(len(slab_atoms)))

    def _build_all_ads_rows(self, atoms, occupation: np.ndarray) -> dict[int, torch.Tensor]:
        occupied_site_ids = np.where(occupation)[0].astype(int)
        return self._ads_k_rows(atoms, occupied_site_ids)

    def _build_all_rep_rows(self, atoms, slab_atoms, occupation: np.ndarray) -> dict[int, torch.Tensor]:
        if int(occupation.sum()) <= 1:
            return {}
        occupied_site_ids = np.where(occupation)[0].astype(int)
        return self._rep_k_rows(atoms, slab_atoms, occupation, occupied_site_ids)


    def _affected_slab_centers(self, slab_atoms, changed_metal_indices: set[int]) -> set[int]:
        if not changed_metal_indices:
            return set()
        if not np.isfinite(self.slab_cutoff):
            return set(range(len(slab_atoms)))
        positions = slab_atoms.get_positions()
        affected: set[int] = set(changed_metal_indices)
        for center in range(len(slab_atoms)):
            center_pos = positions[center]
            for changed in changed_metal_indices:
                d = mic_distance(center_pos, positions[int(changed)], slab_atoms.cell, slab_atoms.pbc)
                if d <= self.slab_cutoff:
                    affected.add(int(center))
                    break
        return affected

    def _changed_co_positions(self, changed_site_ids: set[int]) -> list[np.ndarray]:
        return [np.asarray(self.sites[int(site_id)].position, dtype=np.float64) for site_id in changed_site_ids]

    def _affected_ads_sites(
        self,
        candidate_slab_atoms,
        candidate_occupation: np.ndarray,
        changed_metal_indices: set[int],
        changed_site_ids: set[int],
    ) -> set[int]:
        occupied = set(np.where(candidate_occupation)[0].astype(int))
        affected = set(changed_site_ids)
        if not occupied:
            return affected

        positions = candidate_slab_atoms.get_positions()
        changed_co_positions = self._changed_co_positions(changed_site_ids)
        conservative_all = not np.isfinite(self.ads_cutoff)

        for site_id in occupied:
            site = self.sites[int(site_id)]
            if conservative_all:
                affected.add(int(site_id))
                continue

            for center_idx in site.atom_indices:
                center_pos = positions[int(center_idx)]

                for co_pos in changed_co_positions:
                    if mic_distance(center_pos, co_pos, candidate_slab_atoms.cell, candidate_slab_atoms.pbc) <= self.ads_cutoff:
                        affected.add(int(site_id))
                        break
                if int(site_id) in affected:
                    break

                for changed_idx in changed_metal_indices:
                    if mic_distance(center_pos, positions[int(changed_idx)], candidate_slab_atoms.cell, candidate_slab_atoms.pbc) <= self.ads_cutoff:
                        affected.add(int(site_id))
                        break
                if int(site_id) in affected:
                    break

        return affected

    def _affected_rep_sites(
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

        occupied = set(np.where(candidate_occupation)[0].astype(int))
        affected = set(changed_site_ids)
        changed_co_positions = self._changed_co_positions(changed_site_ids)
        positions = candidate_slab_atoms.get_positions()
        conservative_all = not np.isfinite(self.rep_cutoff)

        for site_id in occupied:
            site_pos = np.asarray(self.sites[int(site_id)].position, dtype=np.float64)
            if conservative_all:
                affected.add(int(site_id))
                continue

            for co_pos in changed_co_positions:
                if mic_distance(site_pos, co_pos, candidate_slab_atoms.cell, candidate_slab_atoms.pbc) <= self.rep_cutoff:
                    affected.add(int(site_id))
                    break
            if int(site_id) in affected:
                continue

            for changed_idx in changed_metal_indices:
                if mic_distance(site_pos, positions[int(changed_idx)], candidate_slab_atoms.cell, candidate_slab_atoms.pbc) <= self.rep_cutoff:
                    affected.add(int(site_id))
                    break

        return affected


    @staticmethod
    def _sum_k(row_map: dict[int, torch.Tensor], model: SparseAtomicGPR) -> torch.Tensor:
        if not row_map:
            return torch.zeros(model.x_M.shape[0], dtype=torch.float64, device=model.x_M.device)
        rows = [row.to(dtype=torch.float64, device=model.x_M.device) for row in row_map.values()]
        return torch.stack(rows, dim=0).sum(dim=0)

    @staticmethod
    def _mean_from_k(model: SparseAtomicGPR, k_sum: torch.Tensor) -> float:
        value = (k_sum.to(model.x_M.device) @ model.c.to(model.x_M.device)).detach().cpu().item()
        return finite_float(value, "component_energy")

    @staticmethod
    def _cached_K_MM_inv_K_NM_train_T(model: SparseAtomicGPR) -> torch.Tensor:

        cache_name = "_localdesc_K_MM_inv_K_NM_train_T"
        cached = getattr(model, cache_name, None)
        if cached is not None:
            return cached
        value = torch.cholesky_solve(model.K_NM_train.T, model.L_KMM).detach()
        setattr(model, cache_name, value)
        return value

    @staticmethod
    def _mean_std_from_k(model: SparseAtomicGPR, k_sum: torch.Tensor) -> tuple[float, float]:
        k_sum = k_sum.to(dtype=torch.float64, device=model.x_M.device)
        mean = LocalDescriptorEnergyEvaluator._mean_from_k(model, k_sum)

        if getattr(model, "K_NM_train", None) is None or getattr(model, "L_KMM", None) is None or getattr(model, "L_KSS", None) is None:
            return mean, float("nan")

        K_test = k_sum.reshape(1, -1)
        K_MM_inv_K_NM_train_T = LocalDescriptorEnergyEvaluator._cached_K_MM_inv_K_NM_train_T(model)
        K_star_S = K_test @ K_MM_inv_K_NM_train_T

        K_MM_inv_K_NM_test_T = torch.cholesky_solve(K_test.T, model.L_KMM)
        K_star_star = K_test @ K_MM_inv_K_NM_test_T

        tmp = torch.cholesky_solve(K_star_S.T, model.L_KSS)
        cov = K_star_star - K_star_S @ tmp
        var = torch.clamp(cov.diagonal(), min=1e-12)
        std = torch.sqrt(var)[0].detach().cpu().item()
        return mean, finite_float(std, "component_uncertainty")

    def _energy_from_k_maps(
        self,
        slab_k: dict[int, torch.Tensor],
        ads_k: dict[int, torch.Tensor],
        rep_k: dict[int, torch.Tensor],
        *,
        compute_uncertainty: bool = False,
    ) -> EnergyComponents:
        slab_sum = self._sum_k(slab_k, self.slab_model)
        ads_sum = self._sum_k(ads_k, self.ads_model)

        if rep_k:
            rep_model = self._require_rep_model()
            rep_sum = self._sum_k(rep_k, rep_model)
        else:
            rep_model = None
            rep_sum = None

        if compute_uncertainty:
            slab_e, slab_u = self._mean_std_from_k(self.slab_model, slab_sum)
            ads_e, ads_u = (0.0, 0.0) if not ads_k else self._mean_std_from_k(self.ads_model, ads_sum)
            rep_e, rep_u = (0.0, 0.0) if rep_model is None else self._mean_std_from_k(rep_model, rep_sum)
            if np.isfinite(slab_u) and np.isfinite(ads_u) and np.isfinite(rep_u):
                unc = float(np.sqrt(slab_u**2 + ads_u**2 + rep_u**2))
            else:
                unc = float("nan")
        else:
            slab_e = self._mean_from_k(self.slab_model, slab_sum)
            ads_e = 0.0 if not ads_k else self._mean_from_k(self.ads_model, ads_sum)
            rep_e = 0.0 if rep_model is None else self._mean_from_k(rep_model, rep_sum)
            unc = float("nan")
            slab_u = float("nan")
            ads_u = float("nan")
            rep_u = float("nan")

        return EnergyComponents(
            total=slab_e + ads_e + rep_e,
            slab=slab_e,
            ads=ads_e,
            rep=rep_e,
            uncertainty=unc,
            uncertainty_slab=slab_u,
            uncertainty_ads=ads_u,
            uncertainty_rep=rep_u,
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
        f"{'E_total':>14} {'E_slab':>14} {'E_ads':>14} {'E_rep':>14} "
        f"{'unc_tot':>12} {'unc_slab':>12} {'unc_ads':>12} {'unc_rep':>12} "
        f"{'N_CO':>5} {'theta':>8} "
        f"{'acc_alloy':>10} {'acc_CO':>8} {'acc_ins':>8} {'acc_del':>8} {'acc_mig':>8}"
    )


def progress_table_row(step: int, state: LocalMCState, n_sites: int, stats: MCStats | None = None) -> str:
    n_co = int(state.occupation.sum())
    theta = float(n_co / n_sites) if n_sites > 0 else 0.0

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
        f"{_fmt_table_float(e.rep, 14, 6)} "
        f"{_fmt_table_float(e.uncertainty, 12, 4, scientific=True)} "
        f"{_fmt_table_float(getattr(e, 'uncertainty_slab', float('nan')), 12, 4, scientific=True)} "
        f"{_fmt_table_float(getattr(e, 'uncertainty_ads', float('nan')), 12, 4, scientific=True)} "
        f"{_fmt_table_float(getattr(e, 'uncertainty_rep', float('nan')), 12, 4, scientific=True)} "
        f"{n_co:5d} {theta:8.4f} "
        f"{_fmt_table_float(acc_alloy, 10, 3)} "
        f"{_fmt_table_float(acc_co, 8, 3)} "
        f"{_fmt_table_float(acc_ins, 8, 3)} "
        f"{_fmt_table_float(acc_del, 8, 3)} "
        f"{_fmt_table_float(acc_mig, 8, 3)}"
    )


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
        active_learning: bool = False,
        uncertainty_threshold: float = float("inf"),
        active_learning_trajectory: str | None = None,
        initial_occupation: np.ndarray | None = None,
        layer_ids: np.ndarray | None = None,
        frozen_atom_mask: np.ndarray | None = None,
        frozen_layers: set[int] | None = None,
    ):
        self.sites = validate_sites(list(sites), n_atoms=len(slab_atoms))
        if len(self.sites) == 0:
            raise ValueError("No adsorption sites were found.")

        self.evaluator = evaluator
        self.temperature = positive_float(temperature, "temperature")
        self.beta = 1.0 / (KB_EV * self.temperature)
        self.delta_mu = finite_float(delta_mu, "delta_mu")
        self.min_co_distance = nonnegative_float(min_co_distance, "min_co_distance")
        self.rng = rng
        self.active_learning = bool(active_learning)
        self.uncertainty_threshold = float(uncertainty_threshold)
        if np.isnan(self.uncertainty_threshold):
            raise ValueError("uncertainty_threshold must not be NaN.")
        self.active_learning_trajectory = active_learning_trajectory
        self.active_learning_count = 0

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
            compute_uncertainty=self.active_learning,
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

    def maybe_export_active_learning_candidate(self, state: LocalMCState) -> None:
        if not self.active_learning:
            return
        uncertainty = float(getattr(state.energy, "uncertainty", float("nan")))
        if not np.isfinite(uncertainty) or uncertainty < self.uncertainty_threshold:
            return
        if self.active_learning_trajectory in (None, ""):
            raise RuntimeError("active_learning_trajectory is empty.")

        atoms_out = attach_mc_info(
            self.make_atoms(state),
            energy=state.energy,
            occupation=state.occupation,
            av_comp=None,
            layer_ids=self.layer_ids,
            frozen_atom_mask=self.frozen_atom_mask,
        )
        atoms_out.info["active_learning_uncertainty"] = uncertainty
        atoms_out.info["active_learning_uncertainty_slab"] = float(getattr(state.energy, "uncertainty_slab", float("nan")))
        atoms_out.info["active_learning_uncertainty_ads"] = float(getattr(state.energy, "uncertainty_ads", float("nan")))
        atoms_out.info["active_learning_uncertainty_rep"] = float(getattr(state.energy, "uncertainty_rep", float("nan")))
        atoms_out.info["active_learning_index"] = self.active_learning_count
        _ensure_parent_dir(self.active_learning_trajectory)
        mode = "a" if self.active_learning_count > 0 else "w"
        traj = Trajectory(self.active_learning_trajectory, mode)
        try:
            traj.write(atoms_out)
        finally:
            traj.close()
        self.active_learning_count += 1

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
            compute_uncertainty=self.active_learning,
        )
        self.maybe_export_active_learning_candidate(candidate)

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
            compute_uncertainty=self.active_learning,
        )
        self.maybe_export_active_learning_candidate(candidate)

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
        )

    def run(
        self,
        state: LocalMCState,
        nsteps: int,
        alloy_attempts_per_step: int,
        co_attempts_per_step: int,
        print_every: int,
        write_every: int,
        trajectory: str,
    ) -> LocalMCState:
        if trajectory:
            _ensure_parent_dir(trajectory)
            if os.path.exists(trajectory):
                os.remove(trajectory)
        if self.active_learning and self.active_learning_trajectory and os.path.exists(self.active_learning_trajectory):
            os.remove(self.active_learning_trajectory)

        self.accumulate_composition(state)
        if trajectory:
            write(trajectory, self.current_atoms_with_info(state, include_av_comp=True), format="extxyz", append=False)

        for step in range(1, int(nsteps) + 1):
            for _ in range(int(alloy_attempts_per_step)):
                state = self.attempt_alloy_swap(state)
            for _ in range(int(co_attempts_per_step)):
                state = self.attempt_co_move(state)

            self.accumulate_composition(state)

            if print_every > 0 and (step == 1 or step % print_every == 0 or step == nsteps):
                print(progress_table_row(step, state, len(state.occupation), self.stats), flush=True)

            if trajectory and write_every > 0 and step % write_every == 0:
                write(trajectory, self.current_atoms_with_info(state, include_av_comp=True), format="extxyz", append=True)

        return state


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    validate_args(args)
    _safe_set_torch_threads(1)
    rng = np.random.default_rng(args.seed)

    _ensure_parent_dir(args.trajectory)
    _ensure_parent_dir(args.output)
    if args.active_learning:
        _ensure_parent_dir(args.active_learning_trajectory)

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
    print(f"Number of adsorption sites: {len(sites)}", flush=True)
    print(f"  ontop sites  = {n_ontop}", flush=True)
    print(f"  bridge sites = {n_bridge}", flush=True)
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
            f"N_CO = {initial_n_co}, theta = {initial_occupation.mean():.6f}",
            flush=True,
        )
    else:
        print("Initial CO occupation: N_CO = 0", flush=True)

    evaluator = LocalDescriptorEnergyEvaluator(
        file_slab_model=args.slab_model,
        file_ads_model=args.ads_model,
        file_rep_model=args.rep_model,
        sites=sites,
        co_bond=args.co_bond,
        device=args.device,
        allow_unsafe_load=args.allow_unsafe_model_load,
        local_cutoff_margin=args.local_cutoff_margin,
    )
    cutoffs = evaluator.cutoffs()
    rep_cutoff_text = "lazy/not loaded" if not np.isfinite(cutoffs["rep"]) else f"{cutoffs['rep']:.6f} A"
    print(
        "effective local descriptor invalidation cutoffs: "
        f"slab={cutoffs['slab']:.6f} A, "
        f"ads={cutoffs['ads']:.6f} A, "
        f"rep={rep_cutoff_text}",
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
        active_learning=args.active_learning,
        uncertainty_threshold=args.uncertainty_threshold,
        active_learning_trajectory=args.active_learning_trajectory,
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
    print(progress_table_row(0, state, len(state.occupation), None), flush=True)

    final_state = mc.run(
        state=state,
        nsteps=args.nsteps,
        alloy_attempts_per_step=args.alloy_attempts_per_step,
        co_attempts_per_step=args.co_attempts_per_step,
        print_every=args.print_every,
        write_every=args.write_every,
        trajectory=args.trajectory,
    )

    final_atoms = mc.current_atoms_with_info(final_state, include_av_comp=True)
    write(args.output, final_atoms, format="extxyz")

    print("MC finished cleanly", flush=True)
    print(f"Final N_CO = {int(final_state.occupation.sum())}", flush=True)
    print(f"Final theta_CO = {final_state.occupation.mean():.6f}", flush=True)
    print(f"Alloy acceptance = {mc.stats.alloy_acceptance:.6f}", flush=True)
    print(f"CO acceptance = {mc.stats.co_acceptance:.6f}", flush=True)
    print(f"CO insertion acceptance = {mc.stats.co_insert_acceptance:.6f}", flush=True)
    print(f"CO deletion acceptance = {mc.stats.co_delete_acceptance:.6f}", flush=True)
    print(f"CO migration acceptance = {mc.stats.co_migration_acceptance:.6f}", flush=True)
    print(f"Composition samples used for av_comp = {mc.av_comp_count}", flush=True)
    if args.active_learning:
        print(f"Active-learning structures written = {mc.active_learning_count}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
