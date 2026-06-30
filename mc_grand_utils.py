from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ase import Atoms, Atom
from ase.build import bulk, make_supercell
from ase.geometry import find_mic, wrap_positions


KB_EV = 8.617333262145e-5  # eV/K


@dataclass(frozen=True)
class AdsorptionSite:

    position: np.ndarray
    kind: str
    atom_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        pos = np.asarray(self.position, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"AdsorptionSite.position must have shape (3,), got {pos.shape}.")
        if not np.isfinite(pos).all():
            raise ValueError("AdsorptionSite.position contains NaN or Inf.")
        object.__setattr__(self, "position", pos)

        kind = str(self.kind)
        if kind not in {"ontop", "bridge"}:
            raise ValueError(f"Unsupported adsorption-site kind: {kind!r}.")
        object.__setattr__(self, "kind", kind)

        atom_indices = tuple(int(i) for i in self.atom_indices)
        expected = 1 if kind == "ontop" else 2
        if len(atom_indices) != expected:
            raise ValueError(
                f"Site kind {kind!r} expects {expected} atom index/indices, got {len(atom_indices)}."
            )
        if any(i < 0 for i in atom_indices):
            raise ValueError("Adsorption-site atom indices must be non-negative.")
        object.__setattr__(self, "atom_indices", atom_indices)


@dataclass
class EnergyComponents:
    total: float
    slab: float
    ads: float
    rep: float
    uncertainty: float = float("nan")
    uncertainty_slab: float = float("nan")
    uncertainty_ads: float = float("nan")
    uncertainty_rep: float = float("nan")

    def __post_init__(self) -> None:
        self.total = finite_float(self.total, "total")
        self.slab = finite_float(self.slab, "slab")
        self.ads = finite_float(self.ads, "ads")
        self.rep = finite_float(self.rep, "rep")

        for name in ("uncertainty", "uncertainty_slab", "uncertainty_ads", "uncertainty_rep"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) or np.isnan(value)):
                raise ValueError(f"{name} must be finite or NaN, got {value}.")
            setattr(self, name, value)


@dataclass
class MCStats:
    alloy_attempts: int = 0
    alloy_accepts: int = 0
    co_attempts: int = 0
    co_accepts: int = 0

    co_insert_attempts: int = 0
    co_insert_accepts: int = 0
    co_delete_attempts: int = 0
    co_delete_accepts: int = 0
    co_migration_attempts: int = 0
    co_migration_accepts: int = 0

    @property
    def alloy_acceptance(self) -> float:
        return self.alloy_accepts / self.alloy_attempts if self.alloy_attempts else 0.0

    @property
    def co_acceptance(self) -> float:
        return self.co_accepts / self.co_attempts if self.co_attempts else 0.0

    @property
    def co_insert_acceptance(self) -> float:
        return self.co_insert_accepts / self.co_insert_attempts if self.co_insert_attempts else 0.0

    @property
    def co_delete_acceptance(self) -> float:
        return self.co_delete_accepts / self.co_delete_attempts if self.co_delete_attempts else 0.0

    @property
    def co_migration_acceptance(self) -> float:
        return self.co_migration_accepts / self.co_migration_attempts if self.co_migration_attempts else 0.0


def finite_float(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def positive_float(value, name: str) -> float:
    value = finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def nonnegative_float(value, name: str) -> float:
    value = finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def validate_occupation(occupation: np.ndarray, n_sites: int, name: str = "occupation") -> np.ndarray:
    occupation = np.asarray(occupation, dtype=bool)
    if occupation.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean array.")
    if len(occupation) != int(n_sites):
        raise ValueError(f"{name} must have length {n_sites}, got {len(occupation)}.")
    return occupation.copy()


def validate_sites(sites: Sequence[AdsorptionSite], n_atoms: int | None = None) -> list[AdsorptionSite]:
    sites = list(sites)
    for site_id, site in enumerate(sites):
        if not isinstance(site, AdsorptionSite):
            site = AdsorptionSite(
                position=np.asarray(site.position, dtype=np.float64),
                kind=str(site.kind),
                atom_indices=tuple(site.atom_indices),
            )
            sites[site_id] = site

        if n_atoms is not None:
            for idx in site.atom_indices:
                if idx >= n_atoms:
                    raise ValueError(
                        f"Adsorption site {site_id} references atom index {idx}, "
                        f"but slab has only {n_atoms} atoms."
                    )
    return sites


def assign_substrate_layers(atoms: Atoms, z_tol: float = 0.1) -> np.ndarray:

    z_tol = nonnegative_float(z_tol, "z_tol")
    if len(atoms) == 0:
        raise ValueError("Cannot assign layers for an empty structure.")

    positions = atoms.get_positions()
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("atoms positions must be finite and have shape (n_atoms, 3).")

    z = np.asarray(positions[:, 2], dtype=np.float64)
    order = np.argsort(z)
    layer_ids = np.empty(len(atoms), dtype=int)

    current_layer = -1
    current_values: list[float] = []
    current_mean = None

    for idx in order:
        zi = float(z[int(idx)])
        if current_mean is None or abs(zi - current_mean) > z_tol:
            current_layer += 1
            current_values = [zi]
            current_mean = zi
        else:
            current_values.append(zi)
            current_mean = float(np.mean(current_values))
        layer_ids[int(idx)] = current_layer

    return layer_ids


def layer_summary(atoms: Atoms, layer_ids: np.ndarray) -> list[dict[str, float | int]]:

    layer_ids = np.asarray(layer_ids, dtype=int)
    if layer_ids.shape != (len(atoms),):
        raise ValueError(f"layer_ids must have shape ({len(atoms)},), got {layer_ids.shape}.")
    if len(layer_ids) == 0:
        return []
    if np.any(layer_ids < 0):
        raise ValueError("layer_ids must be non-negative for substrate atoms.")

    z = atoms.get_positions()[:, 2]
    summary: list[dict[str, float | int]] = []
    for layer in sorted(set(int(x) for x in layer_ids)):
        mask = layer_ids == layer
        z_layer = z[mask]
        summary.append(
            {
                "layer": int(layer),
                "n_atoms": int(mask.sum()),
                "z_mean": float(np.mean(z_layer)),
                "z_min": float(np.min(z_layer)),
                "z_max": float(np.max(z_layer)),
            }
        )
    return summary


def frozen_atom_mask_from_layers(layer_ids: np.ndarray, frozen_layers: Iterable[int]) -> np.ndarray:

    layer_ids = np.asarray(layer_ids, dtype=int)
    frozen = {int(layer) for layer in frozen_layers}
    if any(layer < 0 for layer in frozen):
        raise ValueError(f"Frozen layer ids must be non-negative, got {sorted(frozen)}.")
    if len(layer_ids) == 0:
        raise ValueError("layer_ids must not be empty.")

    n_layers = int(layer_ids.max()) + 1
    invalid = sorted(layer for layer in frozen if layer >= n_layers)
    if invalid:
        raise ValueError(
            f"Frozen layer id(s) {invalid} are out of range. "
            f"Available layer ids are 0..{n_layers - 1}."
        )
    return np.isin(layer_ids, np.array(sorted(frozen), dtype=int))


def build_supercell(
    structure: str,
    lattice_constant: float,
    composition: dict[str, float],
    supercell: np.ndarray,
    seed: int | None = None,
) -> Atoms:

    if not composition:
        raise ValueError("composition must not be empty.")

    lattice_constant = positive_float(lattice_constant, "lattice_constant")
    supercell_arr = np.asarray(supercell)
    if supercell_arr.shape != (3, 3):
        raise ValueError(f"supercell must have shape (3, 3), got {supercell_arr.shape}.")
    if not np.all(np.isfinite(supercell_arr)):
        raise ValueError("supercell contains NaN or Inf.")
    if not np.allclose(supercell_arr, np.rint(supercell_arr)):
        raise ValueError("supercell must contain integer entries.")
    supercell_int = np.asarray(np.rint(supercell_arr), dtype=int)
    if round(np.linalg.det(supercell_int)) == 0:
        raise ValueError("supercell matrix must have non-zero determinant.")

    elements = [str(x) for x in composition.keys()]
    fractions = np.asarray(list(composition.values()), dtype=np.float64)
    if not np.isfinite(fractions).all():
        raise ValueError("composition contains NaN or Inf.")
    if np.any(fractions < 0.0):
        raise ValueError("composition fractions must be non-negative.")
    if fractions.sum() <= 0.0:
        raise ValueError("composition fractions must sum to a positive value.")
    fractions = fractions / fractions.sum()

    atoms = bulk(elements[0], crystalstructure=structure, a=lattice_constant, cubic=True)
    atoms = make_supercell(atoms, supercell_int)
    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ValueError("Generated supercell contains no atoms.")

    counts = np.rint(fractions * n_atoms).astype(int)
    counts[-1] += n_atoms - counts.sum()
    if np.any(counts < 0) or counts.sum() != n_atoms:
        raise RuntimeError("Failed to convert composition fractions to atom counts.")

    symbols: list[str] = []
    for element, count in zip(elements, counts):
        symbols.extend([element] * int(count))

    rng = np.random.default_rng(seed)
    rng.shuffle(symbols)
    atoms.set_chemical_symbols(symbols)
    return atoms


def strip_to_symbols(atoms: Atoms, keep_symbols: Iterable[str] = ("Pt", "Pd")) -> Atoms:

    keep = {str(symbol) for symbol in keep_symbols}
    if not keep:
        raise ValueError("keep_symbols must not be empty.")
    indices = [i for i, atom in enumerate(atoms) if atom.symbol in keep]
    if not indices:
        raise ValueError(f"No atoms with symbols {sorted(keep)} were found.")
    stripped = atoms[indices]
    if len(stripped) == 0:
        raise RuntimeError("Internal error: empty structure after symbol filtering.")
    return stripped


def mic_distance(pos_i, pos_j, cell, pbc) -> float:

    pos_i = np.asarray(pos_i, dtype=np.float64)
    pos_j = np.asarray(pos_j, dtype=np.float64)
    if pos_i.shape != (3,) or pos_j.shape != (3,):
        raise ValueError("positions must have shape (3,).")
    dr = pos_j - pos_i
    dr_mic, _ = find_mic(dr, cell=cell, pbc=[bool(x) for x in pbc])
    return float(np.linalg.norm(dr_mic))


def pbc_xy_distance(pos_i, pos_j, cell, pbc) -> float:

    pos_i = np.asarray(pos_i, dtype=np.float64)
    pos_j = np.asarray(pos_j, dtype=np.float64)
    if pos_i.shape != (3,) or pos_j.shape != (3,):
        raise ValueError("positions must have shape (3,).")
    dr = pos_j - pos_i
    dr_mic, _ = find_mic(dr, cell=cell, pbc=[bool(pbc[0]), bool(pbc[1]), False])
    return float(np.linalg.norm(dr_mic[:2]))


def get_top_indices(atoms: Atoms, z_atol: float = 1e-3) -> np.ndarray:
    z_atol = nonnegative_float(z_atol, "z_atol")
    if len(atoms) == 0:
        raise ValueError("Cannot find top layer in an empty structure.")
    positions = atoms.get_positions()
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("atoms positions must be finite and have shape (n_atoms, 3).")
    z = positions[:, 2]
    zmax = float(z.max())
    top_indices = np.where(np.isclose(z, zmax, atol=z_atol))[0]
    if len(top_indices) == 0:
        raise RuntimeError("No top-layer atoms found. Increase z_atol if the surface is rumpled.")
    return top_indices.astype(int)


def _unique_sites(
    atoms: Atoms,
    sites: list[AdsorptionSite],
    frac_tol: float = 1e-5,
    z_tol: float = 1e-4,
) -> list[AdsorptionSite]:
    unique: list[AdsorptionSite] = []
    seen: set[tuple] = set()
    for site in sites:
        scaled = atoms.cell.scaled_positions(np.asarray([site.position], dtype=np.float64))[0]
        scaled[:2] = scaled[:2] % 1.0
        key = (
            site.kind,
            int(round(float(scaled[0]) / frac_tol)),
            int(round(float(scaled[1]) / frac_tol)),
            int(round(float(site.position[2]) / z_tol)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(site)
    return unique


def find_adsorption_sites(
    atoms: Atoms,
    height: float = 1.60,
    z_atol: float = 1e-3,
    bridge_cutoff_factor: float = 1.25,
    include_ontop: bool = True,
    include_bridge: bool = True,
) -> list[AdsorptionSite]:

    height = positive_float(height, "height")
    z_atol = nonnegative_float(z_atol, "z_atol")
    bridge_cutoff_factor = positive_float(bridge_cutoff_factor, "bridge_cutoff_factor")
    if not include_ontop and not include_bridge:
        raise ValueError("At least one of include_ontop/include_bridge must be True.")

    positions = atoms.get_positions()
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("atoms positions must be finite and have shape (n_atoms, 3).")

    cell = atoms.cell
    pbc = atoms.pbc
    zmax = float(positions[:, 2].max())
    top_indices = get_top_indices(atoms, z_atol=z_atol)
    sites: list[AdsorptionSite] = []

    if include_ontop:
        for idx in top_indices:
            idx = int(idx)
            sites.append(
                AdsorptionSite(
                    position=np.array([positions[idx, 0], positions[idx, 1], zmax + height], dtype=np.float64),
                    kind="ontop",
                    atom_indices=(idx,),
                )
            )

    if include_bridge and len(top_indices) >= 2:
        pair_data: list[tuple[int, int, float]] = []
        distances: list[float] = []
        for a, i in enumerate(top_indices):
            for j in top_indices[a + 1 :]:
                d_xy = pbc_xy_distance(positions[int(i)], positions[int(j)], cell=cell, pbc=pbc)
                if d_xy > 1e-6:
                    distances.append(d_xy)
                    pair_data.append((int(i), int(j), d_xy))

        if distances:
            bridge_cutoff = bridge_cutoff_factor * float(min(distances))
            for i, j, d_xy in pair_data:
                if d_xy > bridge_cutoff:
                    continue
                dr = positions[j] - positions[i]
                dr_mic, _ = find_mic(dr, cell=cell, pbc=[bool(pbc[0]), bool(pbc[1]), False])
                midpoint = positions[i] + 0.5 * dr_mic
                bridge_position = np.array([midpoint[0], midpoint[1], zmax + height], dtype=np.float64)
                bridge_position = wrap_positions(
                    np.asarray([bridge_position], dtype=np.float64),
                    cell=cell,
                    pbc=[bool(pbc[0]), bool(pbc[1]), False],
                    eps=1e-9,
                )[0]
                sites.append(AdsorptionSite(position=bridge_position, kind="bridge", atom_indices=(i, j)))

    sites = _unique_sites(atoms, sites)
    if not sites:
        raise RuntimeError("No adsorption sites were found.")
    return sites


def build_adsorbed_structure(
    slab_atoms: Atoms,
    sites: Sequence[AdsorptionSite],
    occupation: np.ndarray,
    co_bond: float = 1.15,
) -> Atoms:

    sites = validate_sites(sites, n_atoms=len(slab_atoms))
    occupation = validate_occupation(occupation, len(sites))
    co_bond = positive_float(co_bond, "co_bond")

    atoms = slab_atoms.copy()
    for occupied, site in zip(occupation, sites):
        if not bool(occupied):
            continue
        c_pos = np.asarray(site.position, dtype=np.float64).copy()
        o_pos = c_pos.copy()
        o_pos[2] += co_bond
        atoms.append(Atom("C", c_pos))
        atoms.append(Atom("O", o_pos))
    return atoms


def carbon_index_by_site_id(n_slab_atoms: int, occupation: np.ndarray) -> dict[int, int]:

    occupation = np.asarray(occupation, dtype=bool)
    mapping: dict[int, int] = {}
    n_slab_atoms = int(n_slab_atoms)
    rank = 0
    for site_id, occupied in enumerate(occupation):
        if bool(occupied):
            mapping[int(site_id)] = n_slab_atoms + 2 * rank
            rank += 1
    return mapping


def min_distance_to_occupied_sites(
    sites: Sequence[AdsorptionSite],
    occupation: np.ndarray,
    trial_site_id: int,
    cell,
    pbc,
) -> float:

    sites = validate_sites(sites)
    occupation = validate_occupation(occupation, len(sites))
    trial_site_id = int(trial_site_id)
    if trial_site_id < 0 or trial_site_id >= len(sites):
        raise IndexError("trial_site_id is out of range.")

    occupied_ids = np.where(occupation)[0]
    occupied_ids = occupied_ids[occupied_ids != trial_site_id]
    if len(occupied_ids) == 0:
        return float("inf")

    trial_pos = sites[trial_site_id].position
    return float(
        min(
            pbc_xy_distance(trial_pos, sites[int(j)].position, cell=cell, pbc=pbc)
            for j in occupied_ids
        )
    )


def occupation_satisfies_min_distance(
    sites: Sequence[AdsorptionSite],
    occupation: np.ndarray,
    min_distance: float,
    cell,
    pbc,
) -> bool:

    min_distance = nonnegative_float(min_distance, "min_distance")
    if min_distance <= 0.0:
        return True
    sites = validate_sites(sites)
    occupation = validate_occupation(occupation, len(sites))
    occupied = np.where(occupation)[0]
    for pos, i in enumerate(occupied):
        for j in occupied[pos + 1 :]:
            d = pbc_xy_distance(sites[int(i)].position, sites[int(j)].position, cell=cell, pbc=pbc)
            if d < min_distance:
                return False
    return True


def metropolis_hastings_accept(
    delta_omega: float,
    beta: float,
    log_q_reverse_over_forward: float,
    rng: np.random.Generator,
) -> bool:

    delta_omega = finite_float(delta_omega, "delta_omega")
    beta = positive_float(beta, "beta")
    log_q_reverse_over_forward = finite_float(log_q_reverse_over_forward, "log_q_reverse_over_forward")
    log_alpha = -beta * delta_omega + log_q_reverse_over_forward
    if log_alpha >= 0.0:
        return True
    return bool(np.log(rng.random()) < log_alpha)


def propose_pt_pd_swap(
    slab_atoms: Atoms,
    rng: np.random.Generator,
    allowed_symbols: tuple[str, str] = ("Pt", "Pd"),
    allowed_indices: Iterable[int] | None = None,
) -> tuple[Atoms | None, tuple[int, int] | None]:

    if len(allowed_symbols) != 2:
        raise ValueError("allowed_symbols must contain exactly two symbols.")

    symbols = np.asarray(slab_atoms.get_chemical_symbols())
    species0, species1 = allowed_symbols

    if allowed_indices is None:
        mobile_mask = np.ones(len(slab_atoms), dtype=bool)
    else:
        allowed_indices_arr = np.asarray(list(allowed_indices), dtype=int).ravel()
        mobile_mask = np.zeros(len(slab_atoms), dtype=bool)
        if allowed_indices_arr.size > 0:
            if np.any(allowed_indices_arr < 0) or np.any(allowed_indices_arr >= len(slab_atoms)):
                raise IndexError("allowed_indices contains atom index out of range.")
            mobile_mask[allowed_indices_arr] = True

    idx0 = np.where((symbols == species0) & mobile_mask)[0]
    idx1 = np.where((symbols == species1) & mobile_mask)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return None, None

    i = int(rng.choice(idx0))
    j = int(rng.choice(idx1))
    candidate = slab_atoms.copy()
    candidate_symbols = candidate.get_chemical_symbols()
    candidate_symbols[i], candidate_symbols[j] = candidate_symbols[j], candidate_symbols[i]
    candidate.set_chemical_symbols(candidate_symbols)
    return candidate, (i, j)


def attach_mc_info(
    atoms: Atoms,
    energy: EnergyComponents,
    occupation: np.ndarray,
    av_comp: np.ndarray | None = None,
    layer_ids: np.ndarray | None = None,
    frozen_atom_mask: np.ndarray | None = None,
) -> Atoms:

    atoms = atoms.copy()
    occupation = np.asarray(occupation, dtype=bool)
    n_co = int(occupation.sum())
    n_sites = int(len(occupation))

    atoms.info["E_total"] = float(energy.total)
    atoms.info["E_slab"] = float(energy.slab)
    atoms.info["E_ads"] = float(energy.ads)
    atoms.info["E_rep"] = float(energy.rep)
    atoms.info["uncertainty"] = float(getattr(energy, "uncertainty", float("nan")))
    atoms.info["uncertainty_slab"] = float(getattr(energy, "uncertainty_slab", float("nan")))
    atoms.info["uncertainty_ads"] = float(getattr(energy, "uncertainty_ads", float("nan")))
    atoms.info["uncertainty_rep"] = float(getattr(energy, "uncertainty_rep", float("nan")))
    atoms.info["N_CO"] = n_co
    atoms.info["theta_CO"] = float(n_co / n_sites) if n_sites > 0 else 0.0

    is_adsorbate = np.array([1.0 if atom.symbol in {"C", "O"} else 0.0 for atom in atoms], dtype=np.float64)
    atoms.set_array("is_adsorbate", is_adsorbate)

    n_substrate = len(atoms) - 2 * n_co

    if layer_ids is not None:
        layer_ids = np.asarray(layer_ids, dtype=int)
        if len(layer_ids) != n_substrate:
            raise ValueError(
                f"layer_ids must have length {n_substrate} for substrate atoms, got {len(layer_ids)}."
            )
        layer_full = np.full(len(atoms), -1, dtype=int)
        layer_full[:n_substrate] = layer_ids
        atoms.set_array("layer_id", layer_full)

    if frozen_atom_mask is not None:
        frozen_atom_mask = np.asarray(frozen_atom_mask, dtype=bool)
        if len(frozen_atom_mask) != n_substrate:
            raise ValueError(
                f"frozen_atom_mask must have length {n_substrate} for substrate atoms, "
                f"got {len(frozen_atom_mask)}."
            )
        frozen_full = np.zeros(len(atoms), dtype=np.float64)
        frozen_full[:n_substrate] = frozen_atom_mask.astype(np.float64)
        atoms.set_array("is_frozen_layer", frozen_full)

    if av_comp is not None:
        av_comp = np.asarray(av_comp, dtype=np.float64)
        if len(av_comp) != n_substrate:
            raise ValueError(
                f"av_comp must have length {n_substrate} for substrate atoms, got {len(av_comp)}."
            )
        av_comp_full = np.full(len(atoms), np.nan, dtype=np.float64)
        av_comp_full[:n_substrate] = av_comp
        atoms.set_array("av_comp", av_comp_full)
        atoms.set_array("av_pd", av_comp_full.copy())

        av_pt_full = np.full(len(atoms), np.nan, dtype=np.float64)
        av_pt_full[:n_substrate] = 1.0 - av_comp
        atoms.set_array("av_pt", av_pt_full)

    return atoms