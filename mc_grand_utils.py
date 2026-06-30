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
    """One lattice-gas CO adsorption site.

    position is the carbon position of an upright CO molecule.
    atom_indices are substrate atoms defining the site:
    one atom for ontop, two atoms for bridge.
    """

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


def build_supercell(
    structure: str,
    lattice_constant: float,
    composition: dict[str, float],
    supercell: np.ndarray,
    seed: int | None = None,
) -> Atoms:
    """Build a random alloy supercell with a fixed integer composition."""
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
    """Return only substrate atoms, removing adsorbates from an input structure."""
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
    """Full 3D minimum-image distance with the supplied PBC mask."""
    pos_i = np.asarray(pos_i, dtype=np.float64)
    pos_j = np.asarray(pos_j, dtype=np.float64)
    if pos_i.shape != (3,) or pos_j.shape != (3,):
        raise ValueError("positions must have shape (3,).")
    dr = pos_j - pos_i
    dr_mic, _ = find_mic(dr, cell=cell, pbc=[bool(x) for x in pbc])
    return float(np.linalg.norm(dr_mic))


def pbc_xy_distance(pos_i, pos_j, cell, pbc) -> float:
    """Minimum-image distance projected onto xy."""
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
    """Find ontop and bridge adsorption sites on the top substrate layer."""
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
    """Create slab + upright CO molecules from a fixed site-occupation vector."""
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
    """Map occupied site_id -> carbon atom index in build_adsorbed_structure output."""
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
    """Return minimum C-C xy distance from a trial site to already occupied sites."""
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
    """Check pairwise occupied-site C-C xy distances."""
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
    """Accept with min(1, exp(-beta*dOmega) * q_reverse/q_forward)."""
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
) -> tuple[Atoms | None, tuple[int, int] | None]:
    """Swap two substrate atoms of different species.

    Returns (None, None) if the structure is pure Pt or pure Pd; the caller can
    treat this as a skipped/rejected alloy move instead of crashing the MC run.
    """
    if len(allowed_symbols) != 2:
        raise ValueError("allowed_symbols must contain exactly two symbols.")

    symbols = np.asarray(slab_atoms.get_chemical_symbols())
    species0, species1 = allowed_symbols
    idx0 = np.where(symbols == species0)[0]
    idx1 = np.where(symbols == species1)[0]
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
) -> Atoms:
    """Attach scalar MC metadata and optional composition averages."""
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

    if av_comp is not None:
        av_comp = np.asarray(av_comp, dtype=np.float64)
        n_substrate = len(atoms) - 2 * n_co
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
