"""Standardize noisy VASP OUTCAR geometries onto an ideal fcc lattice and
collect them into ase.db databases.

The standardization pipeline (unwrap_slab_z ... process_outcar) is the same
logic developed and explained step by step in dataset_outcar_pipeline.ipynb —
this module is that pipeline extracted into an importable/runnable form, plus
the directory layout used by make_database.ipynb (_datasets/clean,
_datasets/low_cov, _datasets/rep) and the ase.db writing step.

Run as a script to (re)build all three databases:

    python make_database.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.db import connect
from ase.io import read
from tqdm import tqdm

from ce_sparse_gpr.calculator import CalculatorCESparseGPR
from ce_sparse_gpr.ce_extractor import ClusterExpansion
from ce_sparse_gpr.gpr import SparseAtomicGPR
from ce_sparse_gpr.mc_grand_utils import (
    find_adsorption_sites,
    build_adsorbed_structure,
    strip_to_symbols,
    pbc_xy_distance,
    mic_distance,
    assign_substrate_layers,
)

DATASETS_ROOT = Path("examples/training/datasets")


# =========================================================================
# Geometry standardization pipeline (see dataset_outcar_pipeline.ipynb)
# =========================================================================

def unwrap_slab_z(atoms: Atoms) -> Atoms:
    """Roll z so the vacuum gap sits at the cell seam, undoing VASP-side
    wraparound of atoms across the periodic z boundary."""
    atoms = atoms.copy()
    atoms.set_constraint(None)  # else set_positions silently no-ops on FixAtoms atoms
    cell_z = float(atoms.cell[2, 2])
    positions = atoms.get_positions()
    z = positions[:, 2] % cell_z

    order = np.argsort(z)
    z_sorted = z[order]
    gaps = np.diff(z_sorted, append=z_sorted[0] + cell_z)
    boundary = z_sorted[int(np.argmax(gaps))] + gaps.max() / 2

    positions[:, 2] = (positions[:, 2] - boundary) % cell_z
    atoms.set_positions(positions)
    atoms.pbc = (atoms.pbc[0], atoms.pbc[1], False)
    return atoms


def normalize_slab_z(noisy_atoms: Atoms, substrate_symbols: tuple[str, ...]) -> Atoms:
    noisy_atoms = unwrap_slab_z(noisy_atoms)
    noisy_substrate = strip_to_symbols(noisy_atoms, keep_symbols=substrate_symbols)
    z0 = float(noisy_substrate.get_positions()[:, 2].min())

    noisy_atoms = noisy_atoms.copy()
    positions = noisy_atoms.get_positions()
    positions[:, 2] -= z0
    noisy_atoms.set_positions(positions)
    return noisy_atoms


def infer_slab_shape(noisy_substrate: Atoms, lattice_constant: float, layer_z_tol: float = 0.6):
    cell = noisy_substrate.cell
    nx = int(round(cell[0, 0] / lattice_constant))
    ny = int(round(cell[1, 1] / lattice_constant))
    if nx < 1 or ny < 1:
        raise ValueError(
            f"Could not infer an in-plane supercell size from cell x,y = "
            f"{cell[0, 0]:.3f}, {cell[1, 1]:.3f} A and lattice_constant = {lattice_constant:.3f} A."
        )

    layer_ids = assign_substrate_layers(noisy_substrate, z_tol=layer_z_tol)
    n_layers = int(layer_ids.max()) + 1

    return nx, ny, n_layers


def build_ideal_reference_slab(nx: int, ny: int, n_layers: int, a: float = 2.818 * np.sqrt(2), vacuum: float = 15.0) -> Atoms:
    repeat_z = (n_layers + 1) // 2  # one cubic-cell repeat = 2 atomic (100) layers
    slab = bulk("Pt", "fcc", a=a, cubic=True).repeat((nx, ny, repeat_z))

    layer_ids = assign_substrate_layers(slab, z_tol=1e-3)
    n_generated = int(layer_ids.max()) + 1
    n_drop = n_generated - n_layers
    if n_drop < 0:
        raise RuntimeError("Internal error: generated fewer layers than requested.")
    if n_drop > 0:
        slab = slab[layer_ids < n_layers]  # drop the topmost layer(s), keep the bottom-anchored registry

    positions = slab.get_positions()
    positions[:, 2] -= positions[:, 2].min()
    slab.set_positions(positions)

    cell = slab.get_cell().copy()
    cell[2, 2] = positions[:, 2].max() + vacuum
    slab.set_cell(cell, scale_atoms=False)
    slab.pbc = (True, True, False)

    return slab


def transfer_substrate_occupation(
    ideal_slab: Atoms, noisy_atoms: Atoms, match_cutoff: float = 0.4, substrate_symbols: tuple[str, ...] = ("Pt", "Pd")
) -> Atoms:
    noisy_substrate = strip_to_symbols(noisy_atoms, keep_symbols=substrate_symbols)

    ideal_slab = ideal_slab.copy()
    ideal_pos = ideal_slab.get_positions()
    symbols = np.array(ideal_slab.get_chemical_symbols(), dtype=object)

    noisy_pos = noisy_substrate.get_positions()
    noisy_sym = np.array(noisy_substrate.get_chemical_symbols(), dtype=object)

    if len(noisy_pos) != len(ideal_pos):
        raise ValueError(
            f"Substrate atom count mismatch: ideal slab has {len(ideal_pos)}, "
            f"relaxed structure has {len(noisy_pos)}."
        )

    used = set()
    for i in range(len(ideal_pos)):
        best_j, best_d = None, np.inf
        for j in range(len(noisy_pos)):
            if j in used:
                continue
            d = mic_distance(ideal_pos[i], noisy_pos[j], cell=ideal_slab.cell, pbc=ideal_slab.pbc)
            if d < best_d:
                best_d, best_j = d, j

        if best_j is None or best_d > match_cutoff:
            raise ValueError(
                f"No matching substrate atom for ideal lattice site {i}; "
                f"nearest distance = {best_d:.3f} A (cutoff = {match_cutoff:.3f} A)."
            )

        symbols[i] = noisy_sym[best_j]
        used.add(best_j)

    ideal_slab.set_chemical_symbols(symbols.tolist())
    return ideal_slab


def transfer_co_occupation(
    ideal_slab: Atoms, noisy_atoms: Atoms, ads_height: float = 1.60, co_bond: float = 1.15, site_match_cutoff: float = 0.8
):
    sites = find_adsorption_sites(ideal_slab, height=ads_height)

    symbols = np.array(noisy_atoms.get_chemical_symbols())
    carbon_indices = np.where(symbols == "C")[0]
    carbon_pos = noisy_atoms.get_positions()[carbon_indices]

    occupation = np.zeros(len(sites), dtype=bool)
    used_sites = set()

    for c_idx, c_pos in zip(carbon_indices, carbon_pos):
        distances = np.array([
            pbc_xy_distance(c_pos, site.position, cell=ideal_slab.cell, pbc=ideal_slab.pbc)
            for site in sites
        ])
        best_site = int(np.argmin(distances))
        best_dist = float(distances[best_site])

        if best_dist > site_match_cutoff:
            raise ValueError(
                f"No matching adsorption site for C atom {int(c_idx)}: "
                f"nearest site is {best_dist:.3f} A away (cutoff = {site_match_cutoff:.3f} A)."
            )
        if best_site in used_sites:
            raise ValueError(f"Two C atoms map to the same adsorption site {best_site}.")

        used_sites.add(best_site)
        occupation[best_site] = True

    return build_adsorbed_structure(ideal_slab, sites, occupation, co_bond=co_bond), occupation, sites


def site_label(site, ideal_slab: Atoms) -> str:
    neighbor_symbols = sorted(ideal_slab[i].symbol for i in site.atom_indices)
    return f"{site.kind}_{''.join(neighbor_symbols)}"


def summarize_co_occupation(ideal_slab: Atoms, sites, occupation: np.ndarray) -> dict:
    occupied_sites = [site for site, occ in zip(sites, occupation) if occ]

    flags = {
        "n_co": len(occupied_sites),
        "n_ontop": sum(1 for site in occupied_sites if site.kind == "ontop"),
        "n_bridge": sum(1 for site in occupied_sites if site.kind == "bridge"),
    }

    label_counts = Counter(site_label(site, ideal_slab) for site in occupied_sites)
    flags.update({label: int(count) for label, count in label_counts.items()})

    return flags


def standardize_relaxed_atoms(
    noisy_atoms: Atoms,
    *,
    lattice_constant: float = 2.818 * np.sqrt(2),
    vacuum: float = 15.0,
    ads_height: float = 1.60,
    co_bond: float = 1.15,
    slab_match_cutoff: float = 0.4,
    site_match_cutoff: float = 0.8,
    layer_z_tol: float = 0.6,
    substrate_symbols: tuple[str, ...] = ("Pt", "Pd"),
):
    noisy_atoms = normalize_slab_z(noisy_atoms, substrate_symbols)
    noisy_substrate = strip_to_symbols(noisy_atoms, keep_symbols=substrate_symbols)

    nx, ny, n_layers = infer_slab_shape(noisy_substrate, lattice_constant, layer_z_tol=layer_z_tol)
    ideal_slab = build_ideal_reference_slab(nx, ny, n_layers, a=lattice_constant, vacuum=vacuum)

    ideal_slab = transfer_substrate_occupation(
        ideal_slab,
        noisy_atoms,
        match_cutoff=slab_match_cutoff,
        substrate_symbols=substrate_symbols,
    )

    geometry, occupation, sites = transfer_co_occupation(
        ideal_slab,
        noisy_atoms,
        ads_height=ads_height,
        co_bond=co_bond,
        site_match_cutoff=site_match_cutoff,
    )

    flags = summarize_co_occupation(ideal_slab, sites, occupation)
    flags["nx"] = nx
    flags["ny"] = ny
    flags["n_layers"] = n_layers

    return geometry, flags


def process_outcar(outcar_path, **standardize_kwargs) -> dict:
    """OUTCAR -> {"geometry", "energy", "path", "nx", "ny", "n_layers", "n_co", "n_ontop", "n_bridge", <site labels>}."""
    outcar_path = Path(outcar_path)

    relaxed_atoms = read(outcar_path)
    energy = float(relaxed_atoms.get_potential_energy())

    geometry, flags = standardize_relaxed_atoms(relaxed_atoms, **standardize_kwargs)

    return {
        "geometry": geometry,
        "energy": energy,
        "path": str(outcar_path),
        **flags,
    }


# =========================================================================
# _datasets/ directory layout -> lists of standardized {geometry, energy, ...}
# =========================================================================

def collect_outcars(paths: Iterable[Path], require_finished: bool = True, **standardize_kwargs) -> list[dict]:
    """Run process_outcar over a list of OUTCAR paths, skipping (and reporting)
    any that don't exist or fail to standardize.

    VASP writes a valid OUTCAR for whatever ionic step the job reached, even if
    the relaxation was killed (walltime, crash, ...) before converging - reading
    such an OUTCAR "succeeds" but silently returns a high-force, non-equilibrium
    geometry/energy. The run.py scripts here only write final.traj (and final.e)
    after ase's relaxation loop actually converges, so by default we require
    final.traj to sit next to OUTCAR as proof the relaxation finished; pass
    require_finished=False to fall back to trusting OUTCAR alone.
    """
    results = []
    for outcar in tqdm(list(paths), desc="Standardizing OUTCARs"):
        outcar = Path(outcar)
        if not outcar.exists():
            tqdm.write(f"Skip {outcar}: does not exist")
            continue
        if require_finished and not (outcar.parent / "final.traj").exists():
            tqdm.write(f"Skip {outcar}: relaxation did not finish (no final.traj next to OUTCAR)")
            continue
        try:
            results.append(process_outcar(outcar, **standardize_kwargs))
        except Exception as e:
            tqdm.write(f"Skip {outcar}: {e}")
    return results


def build_clean_9layer_dataset(root=DATASETS_ROOT / "clean", **standardize_kwargs) -> list[dict]:
    """Bare Pt/Pd alloy slabs, no CO, 9 atomic layers (thick-slab convergence set)."""
    root = Path(root)
    dirlist = sorted(p.name for p in root.iterdir() if p.name.startswith("alloy"))
    paths = [root / name / "OUTCAR" for name in dirlist]
    return collect_outcars(paths, **standardize_kwargs)


def build_clean_4layer_dataset(
    low_cov_root=DATASETS_ROOT / "low_cov", rep_root=DATASETS_ROOT / "rep", **standardize_kwargs
) -> list[dict]:
    """Bare Pt/Pd alloy slabs, no CO, 4 atomic layers - the geometry actually used by
    low_cov/rep CO-adsorption structures, so this is the right reference for E_slab
    there (unlike the 9-layer set: its atoms never see two surfaces within the
    descriptor cutoff at once, so a model trained only on it extrapolates badly on
    these thin slabs - see the clean.db vs low_cov geometry investigation)."""
    low_cov_root = Path(low_cov_root)
    rep_root = Path(rep_root)

    # Paired with the low_cov ontop/bridge CO structures (same alloy_XXX naming,
    # see build_low_cov_dataset) - the bare-alloy structures they were built from.
    clean_top_dirlist = sorted(
        p.name for p in (low_cov_root / "Run_1_250_seed13_clean_TOP").iterdir() if p.name.startswith("alloy")
    )
    clean_top_paths = [low_cov_root / "Run_1_250_seed13_clean_TOP" / name / "OUTCAR" for name in clean_top_dirlist]

    seed13_dirlist = sorted(
        p.name for p in (low_cov_root / "Run_1_250_seed13").iterdir() if p.name.startswith("alloy")
    )
    seed13_paths = [low_cov_root / "Run_1_250_seed13" / name / "OUTCAR" for name in seed13_dirlist]

    rep_pure_paths = sorted((rep_root).glob("alloy_*/alloy_pure/OUTCAR"))

    results = collect_outcars(clean_top_paths, **standardize_kwargs)

    # Run_1_250_seed13 was saved without run.py's final.traj/final.e markers (only
    # OUTCAR is present) - verified separately that these are single-ionic-step
    # static runs on an already-relaxed geometry, not truncated relaxations, so the
    # require_finished gate is disabled here rather than dropping 250 good structures.
    results += collect_outcars(seed13_paths, require_finished=False, **standardize_kwargs)

    results += collect_outcars(rep_pure_paths, **standardize_kwargs)

    return results


def build_low_cov_dataset(root=DATASETS_ROOT / "low_cov", **standardize_kwargs) -> list[dict]:
    """Single-CO ontop + single-CO bridge structures."""
    root = Path(root)

    dirlist_ontop = sorted(
        p.name for p in (root / "Run_1_250_seed13_clean_TOP").iterdir() if p.name.startswith("alloy")
    )
    ontop_paths = [root / "Run_1_250_CO_same_place_TOP" / f"{name}_CO" / "OUTCAR" for name in dirlist_ontop]

    dirlist_bridge = sorted(
        p.name for p in (root / "Run_1_250_seed13").iterdir() if p.name.startswith("alloy")
    )
    bridge_paths = [root / "Run_1_250_CO_bridge" / f"{name}_CO" / "OUTCAR" for name in dirlist_bridge]

    return collect_outcars(ontop_paths, **standardize_kwargs) + collect_outcars(bridge_paths, **standardize_kwargs)


def _read_outcar_energy(outcar_path: Path, require_finished: bool = True) -> float | None:
    """Return an OUTCAR's DFT total energy, or None if it doesn't exist (or,
    when require_finished, has no final.traj sibling proving the relaxation
    actually finished - see collect_outcars). Used for exact-pairing energy
    references, which don't need standardize_relaxed_atoms since only the
    scalar energy is needed, not the geometry."""
    outcar_path = Path(outcar_path)
    if not outcar_path.exists():
        return None
    if require_finished and not (outcar_path.parent / "final.traj").exists():
        return None
    try:
        return float(read(outcar_path).get_potential_energy())
    except Exception:
        return None


def build_low_cov_dataset_exact(root=DATASETS_ROOT / "low_cov", **standardize_kwargs) -> list[dict]:
    """Single-CO ontop + single-CO bridge structures, each paired 1:1 with
    its OWN clean-alloy DFT reference at the exact same Pt/Pd composition -
    verified via atom-by-atom, position-matched symbol comparison (every
    metal atom's nearest counterpart across all 234 ontop + 250 bridge pairs
    carries the same element; 0 mismatches, 0 missing pairs).

    Only pairs geometry with its exact clean-alloy reference energy here
    (stores "clean_reference_energy"/"clean_reference_path", same contract as
    build_rep_dataset_exact_slab) - apply compute_e_ads_total_exact
    afterward to get the actual target, so both the n_co=1 (here) and
    n_co=4..8 (build_rep_dataset_exact_slab) halves of the combined dataset
    go through the exact same formula, in one place."""
    root = Path(root)

    def build_pairs(clean_dir: str, clean_require_finished: bool, co_dir: str, co_suffix: str = "_CO"):
        clean_root = root / clean_dir
        if not clean_root.exists():
            return []
        clean_names = sorted(p.name for p in clean_root.iterdir() if p.name.startswith("alloy"))
        results = []
        n_skipped = 0
        for name in clean_names:
            clean_path = clean_root / name / "OUTCAR"
            co_path = root / co_dir / f"{name}{co_suffix}" / "OUTCAR"

            clean_energy = _read_outcar_energy(clean_path, require_finished=clean_require_finished)
            if clean_energy is None:
                n_skipped += 1
                continue
            if not co_path.exists() or not (co_path.parent / "final.traj").exists():
                n_skipped += 1
                continue

            try:
                r = process_outcar(co_path, **standardize_kwargs)
            except Exception as e:
                tqdm.write(f"Skip {co_path}: {e}")
                n_skipped += 1
                continue

            r["clean_reference_energy"] = clean_energy
            r["clean_reference_path"] = str(clean_path)
            results.append(r)

        print(f"  {clean_dir} -> {co_dir}: {len(results)} exact pairs, {n_skipped} skipped")
        return results

    ontop = build_pairs("Run_1_250_seed13_clean_TOP", True, "Run_1_250_CO_same_place_TOP")
    bridge = build_pairs("Run_1_250_seed13", False, "Run_1_250_CO_bridge")
    return ontop + bridge


def build_rep_dataset_exact_slab(root=DATASETS_ROOT / "rep", **standardize_kwargs) -> list[dict]:
    """rep structures (4-8 CO) paired with their own alloy_pure clean DFT
    reference where one exists (verified: 19/21 alloys have alloy_pure, all
    174 checked pairs position-matched with 0 composition mismatches).

    alloy_004 and alloy_006 have no alloy_pure and are skipped entirely (per
    instruction). Adds "clean_reference_energy" (exact DFT) per row; combine
    with compute_e_ads_total_exact to get the training target.

    rep/new's 9-10 CO tail (pure-Pt/pure-Pd endpoints + one mixed structure)
    has no alloy_pure of its own, but an exhaustive scan of all 1858 OUTCARs
    under _datasets/ found that rep/new/02 (Pd19Pt13, n_co=9) is an EXACT
    atom-by-atom composition match (position-matched, 0 mismatches) for
    alloy_000 - verified against three independent sources
    (rep/alloy_000/alloy_pure, low_cov's two alloy_000 clean references),
    which agree with each other to <1 microeV. So new/02 gets
    alloy_000/alloy_pure as its clean reference too. rep/new/01 (pure Pt) and
    rep/new/03 (pure Pd) have NO match anywhere in the dataset - a bare
    pure-Pt/pure-Pd 4-layer slab was simply never calculated - and are
    skipped; getting them would need a new DFT run, not something derivable
    from existing data."""
    root = Path(root)
    results = []
    n_skipped_alloys = 0
    for alloy_dir in sorted(root.glob("alloy_*")):
        pure_path = alloy_dir / "alloy_pure" / "OUTCAR"
        clean_energy = _read_outcar_energy(pure_path, require_finished=False)
        if clean_energy is None:
            n_skipped_alloys += 1
            continue

        for n in (4, 5, 6, 7, 8):
            for sub in (f"{n}_CO_bridge", f"TOP/{n}_CO_top"):
                outcar_path = alloy_dir / sub / "OUTCAR"
                if not outcar_path.exists() or not (outcar_path.parent / "final.traj").exists():
                    continue
                try:
                    r = process_outcar(outcar_path, **standardize_kwargs)
                except Exception as e:
                    tqdm.write(f"Skip {outcar_path}: {e}")
                    continue
                r["clean_reference_energy"] = clean_energy
                r["clean_reference_path"] = str(pure_path)
                results.append(r)

    # rep/new/02 borrows alloy_000's clean reference (see docstring) - 01/03
    # have no reference anywhere in the dataset and are left out entirely.
    borrowed_pure_path = root / "alloy_000" / "alloy_pure" / "OUTCAR"
    borrowed_clean_energy = _read_outcar_energy(borrowed_pure_path, require_finished=True)
    n_new_included = 0
    if borrowed_clean_energy is not None:
        new_outcar = root / "new" / "02" / "OUTCAR"
        if new_outcar.exists() and (new_outcar.parent / "final.traj").exists():
            new_kwargs = {"slab_match_cutoff": 0.5, **standardize_kwargs}
            try:
                r = process_outcar(new_outcar, **new_kwargs)
                r["clean_reference_energy"] = borrowed_clean_energy
                r["clean_reference_path"] = str(borrowed_pure_path) + " (borrowed: composition-matches alloy_000)"
                results.append(r)
                n_new_included = 1
            except Exception as e:
                tqdm.write(f"Skip {new_outcar}: {e}")

    print(
        f"rep exact-slab: {len(results)} structures from alloys with alloy_pure "
        f"({n_skipped_alloys} alloys skipped for missing alloy_pure) "
        f"+ {n_new_included} from rep/new (02 only, via borrowed alloy_000 reference; "
        "01/03 have no clean reference anywhere in the dataset)"
    )
    return results


def build_rep_mixed_dataset_exact_slab(root=DATASETS_ROOT / "rep", **standardize_kwargs) -> list[dict]:
    """Mixed top+bridge occupancy structures (rep/mixed/alloy_XXX/<Nt><Mb>,
    e.g. "2t_3b" = 2 ontop + 3 bridge CO simultaneously, n_co=4-7 across the
    11 Nt_Mb combinations present) - previously entirely unused by any
    dataset builder here. These fill a real gap: every other rep/low_cov
    structure is pure-ontop-only or pure-bridge-only, but the MC itself
    allows mixed occupancy (min_co_distance only forbids CO's that are close
    together, regardless of site kind) - a model that's never seen a mixed
    structure is extrapolating every time the MC produces one.

    Paired with the SAME alloy_XXX/alloy_pure reference as the rest of
    rep/alloy_XXX (rep/mixed has no alloy_pure of its own - it doesn't need
    one, the alloy composition is identical, just the adsorbate placement
    differs) - verified via the same position-matched symbol check as
    build_rep_dataset_exact_slab: 175/175 checked pairs match exactly, 0
    mismatches. alloy_004 and alloy_006 have no alloy_pure (same as the main
    rep tree) and are skipped for the same reason."""
    root = Path(root)
    mixed_root = root / "mixed"
    if not mixed_root.exists():
        return []

    results = []
    n_skipped_alloys = 0
    for alloy_dir in sorted(mixed_root.glob("alloy_*")):
        alloy_name = alloy_dir.name
        pure_path = root / alloy_name / "alloy_pure" / "OUTCAR"
        clean_energy = _read_outcar_energy(pure_path, require_finished=False)
        if clean_energy is None:
            n_skipped_alloys += 1
            continue

        for sub_dir in sorted(alloy_dir.iterdir()):
            outcar_path = sub_dir / "OUTCAR"
            if not outcar_path.exists() or not (outcar_path.parent / "final.traj").exists():
                continue
            try:
                r = process_outcar(outcar_path, **standardize_kwargs)
            except Exception as e:
                tqdm.write(f"Skip {outcar_path}: {e}")
                continue
            r["clean_reference_energy"] = clean_energy
            r["clean_reference_path"] = str(pure_path)
            r["mixed_occupancy"] = sub_dir.name  # e.g. "2t_3b", for traceability
            results.append(r)

    print(
        f"rep mixed exact-slab: {len(results)} structures "
        f"({n_skipped_alloys} alloys skipped for missing alloy_pure)"
    )
    return results


def compute_e_ads_total_exact(results: list[dict], co_energy: float) -> list[dict]:
    """Total n_co-CO adsorption energy, purely from exact DFT subtraction -
    no GPR model of any kind involved:

        e_ads_total = E_total(n_co CO adsorbed) - E_slab_exact(same alloy,
                      no CO) - n_co * E_CO_gas

    For n_co=1 (low_cov) this is the traditional single-site e_ads; for
    n_co=4..8 (rep) it is the FULL multi-CO system energy relative to
    gas-phase CO references, deliberately NOT decomposed into an
    independent-site part (predicted by an ads model) plus a repulsion
    residual - unlike the old compute_e_rep, which subtracted an ads model's
    own prediction and so propagated that model's error into every e_rep
    target. One model (additive per-adsorbed-CO descriptor, same
    architecture as the old rep pipeline) is meant to be trained directly on
    e_ads_total across the WHOLE n_co range at once, with low_cov's n_co=1
    rows acting as real-DFT low-coverage anchors (see
    build_low_cov_dataset_exact) instead of the earlier synthetic
    zero-anchor structures.

    Requires each result to carry "clean_reference_energy" and "n_co" (see
    build_rep_dataset_exact_slab / build_low_cov_dataset_exact)."""
    for r in results:
        r["e_ads_total"] = r["energy"] - (r["clean_reference_energy"] + r["n_co"] * co_energy)
    return results


def build_rep_dataset(root=DATASETS_ROOT / "rep", **standardize_kwargs) -> list[dict]:
    """Higher-coverage (4-8 CO) bridge + top structures used for the repulsion dataset,
    plus the extra structures under rep/new (pure-Pt/pure-Pd endpoints at 9-10 CO)."""
    root = Path(root)
    paths = []
    for alloy_dir in sorted(root.glob("alloy_*")):
        for n in (4, 5, 6, 7, 8):
            paths.append(alloy_dir / f"{n}_CO_bridge" / "OUTCAR")
            paths.append(alloy_dir / "TOP" / f"{n}_CO_top" / "OUTCAR")
    results = collect_outcars(paths, **standardize_kwargs)

    # Near-saturation coverage (9-10 CO on 32 substrate atoms) relaxes the substrate a
    # bit more than the 4-8 CO structures above - 1-2 atoms per structure land at
    # 0.44-0.49 A from their ideal site, just over the default 0.4 A slab_match_cutoff
    # (verified: not a registry/stacking issue, all other atoms match at <0.26 A, and
    # loosening the cutoff to 0.5 A doesn't change which atoms match or affect any of
    # the structures above - it only accepts these two that were previously borderline).
    new_kwargs = {"slab_match_cutoff": 0.5, **standardize_kwargs}
    new_paths = sorted((root / "new").glob("*/OUTCAR"))
    results += collect_outcars(new_paths, **new_kwargs)

    return results


# =========================================================================
# ase.db writing
# =========================================================================

def write_db(results: list[dict], db_path) -> Path:
    """Write standardized results to a fresh ase.db (any existing rows at
    db_path are cleared, not appended to).

    Clears via db.delete(...) on an open connection rather than removing
    the file first (append=False / manual unlink): a sqlite connection to
    db_path opened elsewhere in the same process (e.g. an earlier EDA cell
    that read this same db) keeps the file locked on Windows, so
    os.remove/unlink on it raises PermissionError even though writing
    through a fresh connection is fine."""
    db_path = Path(db_path)
    db = connect(db_path)
    existing_ids = [row.id for row in db.select()]
    if existing_ids:
        db.delete(existing_ids)
    for r in results:
        extra = {k: v for k, v in r.items() if k not in ("geometry", "energy")}
        db.write(r["geometry"], dft_energy=r["energy"], **extra)
    return db_path


def append_row_to_db(db_path, geometry, energy: float, **extra) -> int:
    """Add a single row to an existing ase.db without touching any row
    already there - unlike write_db (which clears the whole table first),
    this is for incrementally growing a dataset one new DFT point at a time,
    e.g. from the MC active-learning retraining loop.

    Returns the new row's id."""
    db_path = Path(db_path)
    db = connect(db_path)
    row_id = db.write(geometry, dft_energy=float(energy), **extra)
    return int(row_id)


# =========================================================================
# E_ads = E_total(DFT) - (E_slab(predicted) + n_co * E_CO_gas)
# =========================================================================

def load_co_gas_energy(outcar_path=DATASETS_ROOT / "low_cov" / "CO_gase" / "OUTCAR") -> float:
    """Reference total energy of an isolated CO molecule (gas phase)."""
    return float(read(outcar_path).get_potential_energy())


def load_clean_slab_model(model_path: str = "sparse_atomic_gpr_slab___.pt"):
    """Load the trained clean (bare Pt/Pd slab, no adsorbate) energy model,
    plus a descriptor extractor built from that model's own CEConfig - reused
    below so E_slab predictions use exactly the descriptor space the model
    was trained on."""
    model_clean = SparseAtomicGPR(model_path=model_path)
    clean_extractor = ClusterExpansion(model_clean.config)
    return model_clean, clean_extractor


def predict_slab_energy(model_clean, clean_extractor, atoms, substrate_symbols=("Pt", "Pd")) -> float:
    """E_slab predicted by model_clean, with any adsorbate atoms stripped
    from `atoms` first (filter_atoms=True in ClusterExpansion would do this
    too, but stripping explicitly keeps this correct even if model_clean's
    own config.elements ever includes C/O)."""
    slab_atoms = strip_to_symbols(atoms, keep_symbols=substrate_symbols)
    desc = clean_extractor(slab_atoms)
    desc_t = torch.as_tensor(desc, dtype=model_clean.x_M.dtype)
    with torch.no_grad():
        return float(model_clean([desc_t]).item())


def compute_e_ads(
    results: list[dict],
    co_energy: float,
    model_clean,
    clean_extractor,
    substrate_symbols=("Pt", "Pd"),
) -> list[dict]:
    """Add an "e_ads" key to each result dict in place (also returned for
    convenience): E_ads = E_total - (E_slab_predicted + n_co * E_CO_gas).
    Requires each result to already carry "geometry", "energy" and "n_co"
    (all produced by process_outcar/build_*_dataset)."""
    for r in results:
        e_slab = predict_slab_energy(model_clean, clean_extractor, r["geometry"], substrate_symbols)
        r["e_ads"] = r["energy"] - (e_slab + r["n_co"] * co_energy)

    return results


# =========================================================================
# E_rep = E_total(DFT) - (E_slab(predicted) + E_ads(predicted, summed over
#         every CO site in the structure))
# =========================================================================

def load_rep_calculator(
    slab_model_path: str = "sparse_atomic_gpr_slab___.pt",
    ads_model_path: str = "sparse_atomic_gpr_ads___.pt",
) -> CalculatorCESparseGPR:
    """CalculatorCESparseGPR already implements E_total = E_slab + E_ads +
    E_rep (see ce_sparse_gpr/calculator.py): with no rep model loaded, its
    rep_energy term is always zero, so calling it gives exactly
    E_slab(predicted) + E_ads(predicted) - E_ads itself is the SUM of
    model_ads' per-site prediction over every CO adsorbed in the structure
    (aggregate_multi_label_descriptors groups each site's neighbor-metal rows,
    and summing those site rows into one "structure" before calling model_ads
    makes the model's own per-structure sum add up the sites' individual
    predictions - see build_K_NM). rep.db structures carry several adsorbed
    CO's at once (n_co 4-8), so this sum-of-independent-site-energies is the
    natural quantity to subtract off DFT's total to isolate the leftover
    repulsion between them."""
    return CalculatorCESparseGPR(file_slab_model=slab_model_path, file_ads_model=ads_model_path)


def backfill_e_ads(
    db_path,
    model_clean,
    clean_extractor,
    substrate_symbols=("Pt", "Pd"),
) -> int:
    """Compute e_ads for every row already in db_path and write it in place
    (db.update, geometry/energy untouched) via compute_e_ads's own formula.

    Why this exists: write_db (called from main()) only ever writes
    geometry/dft_energy/n_co/... - never e_ads/e_rep, because those need an
    already-TRAINED clean/ads model that doesn't exist yet the first time
    low_cov.db/rep.db are built (chicken-and-egg). Skipping this step is easy
    to miss silently: db_path ends up with every OTHER column populated and
    training against "e_ads"/"e_rep" simply KeyErrors - it doesn't look like
    a data problem until you trace it back. Run this once the clean model is
    trained, before training the ads model; see backfill_e_rep for rep.db."""
    db = connect(db_path)
    co_energy = load_co_gas_energy()
    rows = list(db.select())
    results = [
        {
            "geometry": row.toatoms(),
            "energy": row.key_value_pairs["dft_energy"],
            "n_co": row.key_value_pairs["n_co"],
        }
        for row in rows
    ]
    compute_e_ads(results, co_energy, model_clean, clean_extractor, substrate_symbols)
    for row, r in zip(rows, results):
        db.update(row.id, e_ads=r["e_ads"])
    return len(rows)


def backfill_e_rep(db_path, calc: CalculatorCESparseGPR) -> int:
    """Compute e_rep for every row already in db_path and write it in place
    (db.update, geometry/energy untouched) via compute_e_rep's own formula.
    Run this once the ads model is trained - see backfill_e_ads."""
    db = connect(db_path)
    co_energy = load_co_gas_energy()
    rows = list(db.select())
    results = [
        {
            "geometry": row.toatoms(),
            "energy": row.key_value_pairs["dft_energy"],
            "n_co": row.key_value_pairs["n_co"],
        }
        for row in rows
    ]
    compute_e_rep(results, co_energy, calc)
    for row, r in zip(rows, results):
        db.update(row.id, e_rep=r["e_rep"])
    return len(rows)


def compute_e_rep(results: list[dict], co_energy: float, calc: CalculatorCESparseGPR) -> list[dict]:
    """Add an "e_rep" key to each result dict in place (also returned for
    convenience):

        E_rep = E_total - (E_slab_predicted + n_co * E_CO_gas + E_ads_predicted)

    calc's own ads_energy is model_ads' raw prediction, summed over sites -
    i.e. it predicts each site's e_ads (see compute_e_ads), which is itself
    ALREADY net of one E_CO_gas per site. So E_slab + ads_energy alone is
    missing n_co * E_CO_gas entirely (leaving a huge, mostly-gas-reference
    residual, not a repulsion signal) - it has to be added back explicitly
    here, exactly as in compute_e_ads, before what's left over is genuinely
    the (small) inter-adsorbate repulsion this field is meant to capture.

    Requires each result to already carry "geometry", "energy" and "n_co"
    (all produced by process_outcar/build_rep_dataset)."""
    for r in results:
        _, total_energy, _ = calc(r["geometry"])
        r["e_rep"] = r["energy"] - (float(total_energy.item()) + r["n_co"] * co_energy)

    return results


def main() -> None:
    """Rebuilds geometry/dft_energy/n_co/... for all four databases from raw
    OUTCARs. This alone is NOT enough to train the ads/rep models: low_cov.db
    needs "e_ads" and rep.db needs "e_rep", and both require an already-
    trained model (clean, then ads) to derive - a chicken-and-egg this
    function can't resolve on its own. After running this:
        1. train the clean model on clean_4layer.db
        2. backfill_e_ads(low_cov.db, <trained clean model>, ...), then train ads
        3. backfill_e_rep(rep.db, <calc with trained clean+ads>), then train rep
    See backfill_e_ads/backfill_e_rep."""
    clean_9layer_results = build_clean_9layer_dataset()
    print(f"clean_9layer: {len(clean_9layer_results)} structures")
    write_db(clean_9layer_results, DATASETS_ROOT / "clean_9layer.db")

    clean_4layer_results = build_clean_4layer_dataset()
    print(f"clean_4layer: {len(clean_4layer_results)} structures")
    write_db(clean_4layer_results, DATASETS_ROOT / "clean_4layer.db")

    low_cov_results = build_low_cov_dataset()
    print(f"low_cov: {len(low_cov_results)} structures")
    write_db(low_cov_results, DATASETS_ROOT / "low_cov.db")

    rep_results = build_rep_dataset()
    print(f"rep: {len(rep_results)} structures")
    write_db(rep_results, DATASETS_ROOT / "rep.db")


if __name__ == "__main__":
    main()
