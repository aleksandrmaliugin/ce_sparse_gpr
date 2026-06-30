from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset
from tqdm import tqdm

from .ce_config import CEConfig, normalize_mask
from .ce_extractor import ClusterExpansion


class CEDataset(Dataset):
    def __init__(
        self,
        atoms: list[ase.Atoms],
        config: CEConfig,
        atom_indices: list[list[int]] | None = None,
        target_y: np.ndarray | torch.Tensor | Sequence | None = None,
        dtype: torch.dtype = torch.float32,
        fit_descriptor_mask: bool = True,
        refit_descriptor_mask: bool = False,
        descriptor_mask_atol: float = 0.0,
        filter_atoms: bool = True,
        show_progress: bool = True,
    ):
        if target_y is None:
            raise ValueError("target_y must be provided.")

        if len(atoms) == 0:
            raise ValueError("atoms must contain at least one structure.")

        if atom_indices is not None and len(atom_indices) != len(atoms):
            raise ValueError("atom_indices and atoms must have the same length.")

        self.atoms = list(atoms)
        self.config = config
        self.atom_indices = atom_indices
        self.target_y = target_y
        self.dtype = dtype
        self.fit_descriptor_mask = bool(fit_descriptor_mask)
        self.refit_descriptor_mask = bool(refit_descriptor_mask)
        self.descriptor_mask_atol = float(descriptor_mask_atol)
        self.filter_atoms = bool(filter_atoms)
        self.show_progress = bool(show_progress)

        if self.descriptor_mask_atol < 0.0:
            raise ValueError("descriptor_mask_atol must be non-negative.")

        self.extractor = ClusterExpansion(self.config)
        self.X, self.y = self.build_dataset()

        if len(self.X) != self.y.shape[0]:
            raise ValueError("X and y must have the same number of structures.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def get_all(self):
        return self.X, self.y

    def build_dataset(self):
        x_dataset_full: list[np.ndarray] = []

        iterator = range(len(self.atoms))
        if self.show_progress:
            iterator = tqdm(iterator)

        for i in iterator:
            atom_indices = None if self.atom_indices is None else self.atom_indices[i]

            descriptors = self.extractor(
                atoms=self.atoms[i],
                atom_indices=atom_indices,
                filter_atoms=self.filter_atoms,
                apply_mask=False,
            )

            if descriptors.ndim != 2:
                raise ValueError(
                    f"Descriptor for structure {i} is not 2D: shape={descriptors.shape}."
                )

            x_dataset_full.append(descriptors)

        need_fit_mask = (
            self.fit_descriptor_mask
            and (self.refit_descriptor_mask or self.config.descriptor_mask is None)
        )

        if need_fit_mask:
            descriptor_mask = self._make_nonzero_descriptor_mask(x_dataset_full)
            self.config.descriptor_mask = normalize_mask(descriptor_mask)

        x_dataset = [
            self.extractor.apply_descriptor_mask(
                x,
                descriptor_names=self.extractor.full_descriptor_names,
                descriptor_keys=self.extractor.full_descriptor_keys,
            )
            for x in x_dataset_full
        ]
        x_dataset = [torch.as_tensor(x, dtype=self.dtype) for x in x_dataset]

        y_dataset = torch.as_tensor(self.target_y, dtype=self.dtype)

        if y_dataset.ndim == 2 and y_dataset.shape[1] == 1:
            y_dataset = y_dataset.squeeze(1)

        if y_dataset.ndim != 1:
            raise ValueError(
                f"target_y must be one-dimensional or shape (N, 1), got {tuple(y_dataset.shape)}."
            )

        if y_dataset.shape[0] != len(self.atoms):
            raise ValueError(
                f"target_y has length {y_dataset.shape[0]}, but atoms has length {len(self.atoms)}."
            )

        if not torch.isfinite(y_dataset).all():
            raise ValueError("target_y contains NaN or Inf values.")

        return x_dataset, y_dataset

    def _make_nonzero_descriptor_mask(self, x_dataset: list[np.ndarray]) -> np.ndarray:
        if len(x_dataset) == 0:
            raise ValueError("Cannot build descriptor mask from an empty dataset.")

        X_all = np.concatenate(x_dataset, axis=0)

        if X_all.ndim != 2:
            raise ValueError("Descriptor dataset must be a list of 2D arrays.")

        if X_all.shape[1] == 0:
            raise ValueError("Descriptor matrix has zero columns.")

        mask = np.any(np.abs(X_all) > self.descriptor_mask_atol, axis=0)

        if not np.any(mask):
            raise ValueError("Descriptor mask removes all descriptors.")

        n_total = int(mask.size)
        n_kept = int(mask.sum())
        n_removed = n_total - n_kept

        print(
            f"Descriptor mask fitted: kept {n_kept}/{n_total} features, "
            f"removed {n_removed} always-zero features."
        )

        return mask


def classify_carbon_sites(
    atoms: ase.Atoms,
    radius: float = 2.4,
    carbon_symbol: str = "C",
    oxygen_symbol: str = "O",
    distance_tol: float = 0.15,
):
    """Classify each carbon atom as top, bridge, hollow or unknown.

    Returns
    -------
    carbon_site_ids
        carbon_index -> integer site_id
    carbon_site_metals
        carbon_index -> list of metal atom indices defining the site
    site_info
        site_id -> dict with type, carbon_index, metal_indices, metal_distances
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if distance_tol < 0.0:
        raise ValueError("distance_tol must be non-negative.")

    symbols = np.asarray(atoms.get_chemical_symbols())
    carbon_indices = np.where(symbols == carbon_symbol)[0].astype(int)

    i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, radius)

    carbon_site_ids: dict[int, int] = {}
    carbon_site_metals: dict[int, list[int]] = {}
    site_info: dict[int, dict] = {}

    for site_id, c_idx in enumerate(carbon_indices):
        mask = i_arr == c_idx
        neigh = j_arr[mask]
        dist = d_arr[mask]
        neigh_symbols = symbols[neigh]

        metal_mask = (neigh_symbols != carbon_symbol) & (neigh_symbols != oxygen_symbol)
        metal_indices = neigh[metal_mask].astype(int)
        metal_distances = dist[metal_mask].astype(float)

        if len(metal_indices) == 0:
            site_type = "unknown"
            site_metals: list[int] = []
            site_distances: list[float] = []
        else:
            order = np.argsort(metal_distances)
            metal_indices = metal_indices[order]
            metal_distances = metal_distances[order]

            min_dist = float(metal_distances[0])
            close_mask = metal_distances <= min_dist + distance_tol
            close_metals = metal_indices[close_mask]
            close_distances = metal_distances[close_mask]

            if len(close_metals) == 1:
                site_type = "top"
                site_metals = [int(close_metals[0])]
                site_distances = [float(close_distances[0])]
            elif len(close_metals) == 2:
                site_type = "bridge"
                order2 = np.argsort(close_metals)
                site_metals = [int(x) for x in close_metals[order2]]
                site_distances = [float(x) for x in close_distances[order2]]
            else:
                site_type = "hollow"
                order3 = np.argsort(close_metals[:3])
                site_metals = [int(x) for x in close_metals[:3][order3]]
                site_distances = [float(x) for x in close_distances[:3][order3]]

        carbon_site_ids[int(c_idx)] = int(site_id)
        carbon_site_metals[int(c_idx)] = site_metals
        site_info[int(site_id)] = {
            "type": site_type,
            "carbon_index": int(c_idx),
            "metal_indices": site_metals,
            "metal_distances": site_distances,
        }

    return carbon_site_ids, carbon_site_metals, site_info


def atoms_near_carbon(
    atoms: ase.Atoms,
    radius: float = 2.8,
    carbon_symbol: str = "C",
    oxygen_symbol: str = "O",
    distance_tol: float = 0.5,
    allowed_site_types: tuple[str, ...] | None = ("top", "bridge", "hollow"),
):
    """
    Return metal atoms defining occupied CO adsorption sites.
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    carbon_indices = np.where(symbols == carbon_symbol)[0].astype(int)

    if len(carbon_indices) == 0:
        return np.array([], dtype=int), carbon_indices, []

    carbon_site_ids, carbon_site_metals, site_info = classify_carbon_sites(
        atoms=atoms,
        radius=radius,
        carbon_symbol=carbon_symbol,
        oxygen_symbol=oxygen_symbol,
        distance_tol=distance_tol,
    )

    if allowed_site_types is not None:
        allowed = set(allowed_site_types)
        bad_sites = {
            site_id: info
            for site_id, info in site_info.items()
            if info["type"] not in allowed
        }
        if bad_sites:
            raise ValueError(
                "Unsupported CO adsorption site(s) encountered. "
                f"Allowed site types: {sorted(allowed)}; bad sites: {bad_sites}."
            )

    selected_set: set[int] = set()
    atom_to_site_ids: dict[int, list[int]] = defaultdict(list)

    for c_idx in carbon_indices:
        c_idx = int(c_idx)
        site_id = carbon_site_ids[c_idx]
        metal_indices = carbon_site_metals[c_idx]

        if len(metal_indices) == 0:
            raise ValueError(
                f"Carbon atom {c_idx} has no metal atoms defining an adsorption site."
            )

        for m_idx in metal_indices:
            m_idx = int(m_idx)
            selected_set.add(m_idx)
            atom_to_site_ids[m_idx].append(site_id)

    selected = np.array(sorted(selected_set), dtype=int)
    labels_per_atom = [atom_to_site_ids[int(atom_idx)] for atom_idx in selected]

    return selected, carbon_indices, labels_per_atom


def aggregate_multi_label_descriptors(x, labels_per_atom):

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.ndim != 2:
        raise ValueError(f"x must be a 2D tensor, got shape {tuple(x.shape)}.")

    if len(labels_per_atom) != x.shape[0]:
        raise ValueError(
            f"len(labels_per_atom)={len(labels_per_atom)} but x.shape[0]={x.shape[0]}."
        )

    site_to_rows: dict[int, list[int]] = defaultdict(list)

    for row_idx, labels in enumerate(labels_per_atom):
        for label in labels:
            site_to_rows[int(label)].append(int(row_idx))

    if len(site_to_rows) == 0:
        return (
            torch.empty((0, x.shape[1]), dtype=x.dtype, device=x.device),
            np.array([], dtype=int),
        )

    site_labels = np.array(sorted(site_to_rows.keys()), dtype=int)
    x_sites = []

    for site_id in site_labels:
        rows = torch.tensor(site_to_rows[int(site_id)], dtype=torch.long, device=x.device)
        x_sites.append(x[rows].sum(dim=0, keepdim=True))

    return torch.cat(x_sites, dim=0), site_labels


def calc_mindist(atoms: ase.Atoms) -> float:
    if len(atoms) < 2:
        raise ValueError("calc_mindist requires at least two atoms.")

    D = atoms.get_all_distances(mic=True)
    np.fill_diagonal(D, np.inf)
    mindist = float(D.min())

    if not np.isfinite(mindist):
        raise ValueError("Could not compute a finite minimum distance.")

    return mindist
