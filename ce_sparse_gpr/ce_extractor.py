from __future__ import annotations

from collections import defaultdict
from itertools import combinations, combinations_with_replacement
from typing import Iterable
import warnings

import ase
import numpy as np
from ase.neighborlist import NeighborList, neighbor_list

from .ce_config import CEConfig


class ClusterExpansion:

    def __init__(self, config: CEConfig):
        if config is None:
            raise ValueError("ClusterExpansion requires a non-empty CEConfig.")

        self.config = config
        self.shells = config.shells_dict
        self.max_order = config.max_order
        self.elements = tuple(config.elements)
        self.element_rank = {
            element: rank for rank, element in enumerate(self.elements)
        }

        self.descriptor_names: list[str] | None = None
        self.full_descriptor_names: list[str] | None = None
        self.descriptor_keys: list[str] | None = None
        self.full_descriptor_keys: list[str] | None = None

        self._validate_config()

    @property
    def names(self) -> list[str] | None:
        return self.descriptor_names

    @names.setter
    def names(self, value: list[str] | None) -> None:
        self.descriptor_names = None if value is None else list(value)

    @property
    def full_names(self) -> list[str] | None:
        return self.full_descriptor_names

    @full_names.setter
    def full_names(self, value: list[str] | None) -> None:
        self.full_descriptor_names = None if value is None else list(value)

    @property
    def descriptor_mask(self) -> np.ndarray | None:
        if self.config.descriptor_mask is None:
            return None
        return np.asarray(self.config.descriptor_mask, dtype=bool)

    @staticmethod
    def _as_index_list(atom_indices: Iterable[int] | np.ndarray | None) -> list[int] | None:
        if atom_indices is None:
            return None
        return [int(i) for i in np.asarray(atom_indices, dtype=int).ravel()]

    def _shell_bounds(self, shell_name: str) -> tuple[float, float]:
        try:
            rmin, rmax = self.shells[shell_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown shell '{shell_name}'. Available shells: {list(self.shells)}."
            ) from exc
        return float(rmin), float(rmax)

    @staticmethod
    def _format_window(rmin: float, rmax: float) -> str:
        return f"[{rmin:.2f}, {rmax:.2f}) Å"

    @staticmethod
    def _format_pair_label(a: str, b: str) -> str:
        return f"{a}-{b}"

    @staticmethod
    def _format_triplet_label(center: str, a: str, b: str) -> str:
        return f"center={center}; neighbors={a}-{b}"

    def chemical_labels_atomic(self, order: int) -> list[str]:
        
        if order == 1:
            return list(self.elements)

        if order == 2:
            return [a + b for a, b in combinations_with_replacement(self.elements, 2)]

        if order == 3:
            labels: list[str] = []
            for center in self.elements:
                for a, b in combinations_with_replacement(self.elements, 2):
                    labels.append(center + a + b)
            return labels

        raise ValueError("order must be 1, 2, or 3.")

    def chemical_labels_atomic_display(self, order: int) -> list[str]:

        if order == 1:
            return list(self.elements)

        if order == 2:
            return [
                self._format_pair_label(a, b)
                for a, b in combinations_with_replacement(self.elements, 2)
            ]

        if order == 3:
            labels: list[str] = []
            for center in self.elements:
                for a, b in combinations_with_replacement(self.elements, 2):
                    labels.append(self._format_triplet_label(center, a, b))
            return labels

        raise ValueError("order must be 1, 2, or 3.")

    def _descriptor_key(self, geom_type: str, compact_label: str) -> str:

        return f"{geom_type}:{compact_label}"

    def _descriptor_display_name(self, geom_type: str, display_label: str) -> str:

        if geom_type == "singles":
            return f"single atom: {display_label}"

        if geom_type.startswith("trip_hips_"):
            body = geom_type.removeprefix("trip_hips_")
            try:
                hips_name, base_name = body.split("_base_", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed triplet geometry type: {geom_type!r}.") from exc

            hips_rmin, hips_rmax = self._shell_bounds(hips_name)
            base_rmin, base_rmax = self._shell_bounds(base_name)

            return (
                "triplet: "
                f"center-neighbor r={self._format_window(hips_rmin, hips_rmax)}, "
                f"neighbor-neighbor r={self._format_window(base_rmin, base_rmax)}; "
                f"{display_label}"
            )

        rmin, rmax = self._shell_bounds(geom_type)
        return f"pair r={self._format_window(rmin, rmax)}: {display_label}"

    def apply_descriptor_mask(
        self,
        descriptor,
        descriptor_names: list[str] | None = None,
        descriptor_keys: list[str] | None = None,
        *,
        names: list[str] | None = None,
    ):
        
        if descriptor_names is None and names is not None:
            descriptor_names = names

        descriptor = np.asarray(descriptor, dtype=float)

        if descriptor.ndim != 2:
            raise ValueError(
                f"descriptor must be a 2D array, got shape {descriptor.shape}."
            )

        mask = self.descriptor_mask

        if mask is None:
            if descriptor_names is not None:
                self.full_descriptor_names = list(descriptor_names)
                self.descriptor_names = list(descriptor_names)
            if descriptor_keys is not None:
                self.full_descriptor_keys = list(descriptor_keys)
                self.descriptor_keys = list(descriptor_keys)
            return descriptor

        if descriptor.shape[1] != len(mask):
            raise ValueError(
                f"Descriptor mask length mismatch: mask has {len(mask)} entries, "
                f"descriptor has {descriptor.shape[1]} columns."
            )

        if descriptor_names is not None:
            if len(descriptor_names) != len(mask):
                raise ValueError(
                    f"descriptor_names has length {len(descriptor_names)}, "
                    f"but mask has length {len(mask)}."
                )
            self.full_descriptor_names = list(descriptor_names)
            self.descriptor_names = [
                name for name, keep in zip(descriptor_names, mask) if keep
            ]

        if descriptor_keys is not None:
            if len(descriptor_keys) != len(mask):
                raise ValueError(
                    f"descriptor_keys has length {len(descriptor_keys)}, "
                    f"but mask has length {len(mask)}."
                )
            self.full_descriptor_keys = list(descriptor_keys)
            self.descriptor_keys = [
                key for key, keep in zip(descriptor_keys, mask) if keep
            ]

        return descriptor[:, mask]

    def _filter_atoms_by_elements(
        self,
        atoms: ase.Atoms,
        atom_indices: list[int] | None = None,
    ) -> tuple[ase.Atoms, list[int] | None]:
        symbols = atoms.get_chemical_symbols()
        allowed = set(self.elements)
        keep_old = [i for i, symbol in enumerate(symbols) if symbol in allowed]

        if len(keep_old) == len(atoms):
            return atoms, atom_indices

        if len(keep_old) == 0:
            raise ValueError(
                f"No atoms remain after filtering by allowed elements: {self.elements}."
            )

        old_to_new = {old: new for new, old in enumerate(keep_old)}
        atoms_new = atoms[keep_old]

        if atom_indices is None:
            return atoms_new, None

        atom_indices_new = [old_to_new[idx] for idx in atom_indices if idx in old_to_new]

        if len(atom_indices) > 0 and len(atom_indices_new) == 0:
            raise ValueError(
                "No atom_indices remain after filtering atoms by config.elements. "
                f"Requested indices: {atom_indices}; allowed elements: {self.elements}."
            )

        if len(atom_indices_new) != len(atom_indices):
            warnings.warn(
                "Some requested atom_indices were removed by element filtering. "
                "Check config.elements if this was not intended.",
                RuntimeWarning,
                stacklevel=2,
            )

        return atoms_new, atom_indices_new

    def _validate_config(self) -> None:
        if self.max_order not in (1, 2, 3):
            raise ValueError("max_order must be 1, 2, or 3.")

        if self.max_order >= 2 and not self.shells:
            raise ValueError("shells must be provided when max_order >= 2.")

        for shell_name, bounds in self.shells.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(
                    f"Shell '{shell_name}' must be a tuple/list (rmin, rmax)."
                )

            rmin, rmax = float(bounds[0]), float(bounds[1])

            if not np.isfinite([rmin, rmax]).all():
                raise ValueError(f"Shell '{shell_name}' has non-finite bounds: {bounds}.")

            if rmin < 0.0:
                raise ValueError(f"Shell '{shell_name}' has negative rmin: {rmin}.")

            if rmax <= rmin:
                raise ValueError(
                    f"Shell '{shell_name}' must satisfy rmax > rmin, got {(rmin, rmax)}."
                )

    def _validate_atoms(
        self,
        atoms: ase.Atoms,
        atom_indices: list[int] | None = None,
    ) -> list[str]:
        if not isinstance(atoms, ase.Atoms):
            raise TypeError(f"atoms must be ase.Atoms, got {type(atoms)!r}.")

        elements_list = atoms.get_chemical_symbols()
        unknown = sorted(set(elements_list) - set(self.elements))

        if unknown:
            raise ValueError(
                f"Unknown elements in atoms: {unknown}. Allowed elements: {self.elements}."
            )

        if atom_indices is not None:
            n_atoms = len(atoms)
            for idx in atom_indices:
                if idx < 0 or idx >= n_atoms:
                    raise ValueError(
                        f"Atom index {idx} is out of bounds for {n_atoms} atoms."
                    )

        return elements_list

    def canonical_pair_elements(self, a: str, b: str) -> tuple[str, str]:
        if a not in self.element_rank or b not in self.element_rank:
            raise ValueError(
                f"Unknown pair elements ({a}, {b}); allowed: {self.elements}."
            )

        if self.element_rank[a] <= self.element_rank[b]:
            return a, b
        return b, a

    def _build_single_clusters(self, atoms: ase.Atoms) -> dict[str, list[list[int]]]:
        return {"singles": [[i] for i in range(len(atoms))]}

    def _build_pair_clusters(self, atoms: ase.Atoms) -> dict[str, list[tuple[int, int, tuple[int, int, int], float]]]:
        pair_clusters = {shell_name: [] for shell_name in self.shells}

        if not self.shells:
            return pair_clusters

        max_rmax = max(rmax for _, rmax in self.shells.values())
        i_arr, j_arr, S_arr, d_arr = neighbor_list("ijSd", atoms, max_rmax)

        for i, j, S, d in zip(i_arr, j_arr, S_arr, d_arr):
            i = int(i)
            j = int(j)
            d = float(d)

            if i == j:
                continue

            S_tuple = tuple(int(x) for x in S)

            for shell_name, (rmin, rmax) in self.shells.items():
                if rmin <= d < rmax:
                    pair_clusters[shell_name].append((i, j, S_tuple, d))
                    break

        for shell_name in pair_clusters:
            pair_clusters[shell_name].sort(key=lambda x: (x[0], x[1], x[2]))

        return pair_clusters

    @staticmethod
    def pairlist_to_center_dict(pair_list):
        neigh = defaultdict(list)
        for i, j, Sj, d in pair_list:
            neigh[int(i)].append((int(j), tuple(Sj), float(d)))
        return neigh

    @staticmethod
    def image_distance(
        atoms: ase.Atoms,
        j: int,
        Sj: tuple[int, int, int],
        k: int,
        Sk: tuple[int, int, int],
    ) -> float:
        positions = atoms.get_positions()
        cell = np.asarray(atoms.get_cell())

        rj = positions[j] + np.asarray(Sj, dtype=float) @ cell
        rk = positions[k] + np.asarray(Sk, dtype=float) @ cell

        return float(np.linalg.norm(rj - rk))

    def _build_triplet_clusters(self, atoms: ase.Atoms, pair_clusters: dict):
        triplet_clusters = {}
        shell_names = list(self.shells.keys())

        for hips_name in shell_names:
            hips_pairs = pair_clusters.get(hips_name, [])
            neigh = self.pairlist_to_center_dict(hips_pairs)

            for base_name in shell_names:
                base_rmin, base_rmax = self.shells[base_name]
                triplet_name = f"trip_hips_{hips_name}_base_{base_name}"
                triplets = []

                for center, nbrs in neigh.items():
                    for (j, Sj, _), (k, Sk, _) in combinations(nbrs, 2):
                        d_jk = self.image_distance(atoms, j, Sj, k, Sk)

                        if base_rmin <= d_jk < base_rmax:
                            if (j, Sj) <= (k, Sk):
                                triplets.append((center, j, Sj, k, Sk))
                            else:
                                triplets.append((center, k, Sk, j, Sj))

                triplets.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
                triplet_clusters[triplet_name] = triplets

        return triplet_clusters

    def build_clusters(self, atoms: ase.Atoms):
        clusters = {}

        if self.max_order >= 1:
            clusters.update(self._build_single_clusters(atoms))

        pair_clusters = {}
        if self.max_order >= 2:
            pair_clusters = self._build_pair_clusters(atoms)
            clusters.update(pair_clusters)

        if self.max_order >= 3:
            clusters.update(self._build_triplet_clusters(atoms, pair_clusters))

        return clusters

    def ordered_geom_types(self) -> list[str]:
        geom_types: list[str] = []

        if self.max_order >= 1:
            geom_types.append("singles")

        if self.max_order >= 2:
            geom_types.extend(self.shells.keys())

        if self.max_order >= 3:
            for hips_name in self.shells.keys():
                for base_name in self.shells.keys():
                    geom_types.append(f"trip_hips_{hips_name}_base_{base_name}")

        return geom_types

    def count_descriptors_atomic(
        self,
        elements_list: list[str],
        clusters: dict,
        atom_indices: list[int] | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        n_atoms = len(elements_list)
        selected_centers = set(atom_indices) if atom_indices is not None else None

        descriptor_names: list[str] = []
        descriptor_keys: list[str] = []
        blocks: list[np.ndarray] = []

        for geom_type in self.ordered_geom_types():
            if geom_type == "singles":
                order = 1
            elif geom_type.startswith("trip"):
                order = 3
            else:
                order = 2

            compact_labels = self.chemical_labels_atomic(order)
            display_labels = self.chemical_labels_atomic_display(order)
            label_to_col = {label: i for i, label in enumerate(compact_labels)}
            block = np.zeros((n_atoms, len(compact_labels)), dtype=float)

            for cluster in clusters.get(geom_type, []):
                if geom_type == "singles":
                    center = int(cluster[0])
                    key = elements_list[center]

                elif geom_type.startswith("trip"):
                    center = int(cluster[0])
                    j = int(cluster[1])
                    k = int(cluster[3])
                    a, b = self.canonical_pair_elements(elements_list[j], elements_list[k])
                    key = elements_list[center] + a + b

                else:
                    center = int(cluster[0])
                    j = int(cluster[1])
                    a, b = self.canonical_pair_elements(elements_list[center], elements_list[j])
                    key = a + b

                if selected_centers is not None and center not in selected_centers:
                    continue

                try:
                    col = label_to_col[key]
                except KeyError as exc:
                    raise ValueError(
                        f"Descriptor key '{key}' is not in label list for geometry "
                        f"type '{geom_type}'. Allowed labels: {compact_labels}."
                    ) from exc

                block[center, col] += 1.0

            blocks.append(block)
            descriptor_keys.extend(
                self._descriptor_key(geom_type, label) for label in compact_labels
            )
            descriptor_names.extend(
                self._descriptor_display_name(geom_type, label)
                for label in display_labels
            )

        descriptor = np.concatenate(blocks, axis=1) if blocks else np.empty((n_atoms, 0))
        return descriptor, descriptor_names, descriptor_keys

    def build_clusters_local(self, atoms: ase.Atoms, atom_indices: list[int]):
        unique_indices = sorted(set(int(i) for i in atom_indices))
        clusters = {}

        if self.max_order >= 1:
            clusters["singles"] = [[i] for i in unique_indices]

        pair_clusters = {}
        if self.max_order >= 2:
            pair_clusters = self._build_pair_clusters_local(atoms, unique_indices)
            clusters.update(pair_clusters)

        if self.max_order >= 3:
            clusters.update(self._build_triplet_clusters(atoms, pair_clusters))

        return clusters

    def _build_pair_clusters_local(self, atoms: ase.Atoms, atom_indices: list[int]):
        pair_clusters = {shell_name: [] for shell_name in self.shells}

        if len(atom_indices) == 0 or not self.shells:
            return pair_clusters

        max_rmax = max(rmax for _, rmax in self.shells.values())
        cutoffs = [0.5 * max_rmax] * len(atoms)

        nl = NeighborList(
            cutoffs=cutoffs,
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms)

        positions = atoms.get_positions()
        cell = np.asarray(atoms.get_cell())

        for i in atom_indices:
            neigh_indices, offsets = nl.get_neighbors(int(i))
            ri = positions[int(i)]

            for j, S in zip(neigh_indices, offsets):
                j = int(j)
                S_tuple = tuple(int(x) for x in S)
                rj = positions[j] + np.asarray(S_tuple) @ cell
                d = float(np.linalg.norm(rj - ri))

                for shell_name, (rmin, rmax) in self.shells.items():
                    if rmin <= d < rmax:
                        pair_clusters[shell_name].append((int(i), j, S_tuple, d))
                        break

        for shell_name in pair_clusters:
            pair_clusters[shell_name].sort(key=lambda x: (x[0], x[1], x[2]))

        return pair_clusters

    def generate_all_descriptors(
        self,
        atoms: ase.Atoms,
        atom_indices: list[int] | None = None,
        apply_mask: bool = True,
    ) -> np.ndarray:
        atom_indices = self._as_index_list(atom_indices)
        elements_list = self._validate_atoms(atoms, atom_indices)

        if atom_indices is None:
            clusters = self.build_clusters(atoms)
        else:
            clusters = self.build_clusters_local(atoms, atom_indices)

        descriptor, descriptor_names, descriptor_keys = self.count_descriptors_atomic(
            elements_list=elements_list,
            clusters=clusters,
            atom_indices=atom_indices,
        )

        self.full_descriptor_names = list(descriptor_names)
        self.full_descriptor_keys = list(descriptor_keys)

        if atom_indices is not None:
            descriptor = descriptor[atom_indices]

        if apply_mask:
            descriptor = self.apply_descriptor_mask(
                descriptor,
                descriptor_names=descriptor_names,
                descriptor_keys=descriptor_keys,
            )
        else:
            self.descriptor_names = list(descriptor_names)
            self.descriptor_keys = list(descriptor_keys)

        return descriptor

    def __call__(
        self,
        atoms: ase.Atoms,
        atom_indices: list[int] | np.ndarray | None = None,
        filter_atoms: bool = True,
        apply_mask: bool = True,
    ) -> np.ndarray:
        atom_indices = self._as_index_list(atom_indices)

        if filter_atoms:
            atoms, atom_indices = self._filter_atoms_by_elements(
                atoms=atoms,
                atom_indices=atom_indices,
            )

        return self.generate_all_descriptors(
            atoms=atoms,
            atom_indices=atom_indices,
            apply_mask=apply_mask,
        )
