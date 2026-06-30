from __future__ import annotations

from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch

from .ce_extractor import ClusterExpansion
from .gpr import SparseAtomicGPR
from .dataset import atoms_near_carbon
from .dataset import aggregate_multi_label_descriptors


class CalculatorCESparseGPR:
    """
    Energy and uncertainty calculator for CE-descriptor sparse GPR models.

    E_total = E_slab + E_ads + E_rep
    """

    def __init__(
        self,
        file_slab_model: str | Path | None = None,
        file_ads_model: str | Path | None = None,
        file_rep_model: str | Path | None = None,
        device=None,
        dtype: torch.dtype = torch.float64,
        allow_unsafe_load: bool = False,
    ):
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.allow_unsafe_load = bool(allow_unsafe_load)

        load_kwargs = {"allow_unsafe_load": self.allow_unsafe_load}

        self.slab_model = self._load_optional_model(file_slab_model, **load_kwargs)
        self.ads_model = self._load_optional_model(file_ads_model, **load_kwargs)
        self.rep_model = self._load_optional_model(file_rep_model, **load_kwargs)

        self.slab_extractor = self._make_optional_extractor(self.slab_model)
        self.ads_extractor = self._make_optional_extractor(self.ads_model)
        self.rep_extractor = self._make_optional_extractor(self.rep_model)

    def _load_optional_model(self, model_path, **load_kwargs) -> SparseAtomicGPR | None:
        if model_path is None:
            return None

        model = SparseAtomicGPR(model_path=model_path, **load_kwargs)

        if model.config is None:
            raise ValueError(
                f"Model checkpoint {model_path!r} has no CEConfig. Re-save the model with config."
            )

        if self.device is not None:
            model = model.to(self.device)

        model.eval()
        return model

    @staticmethod
    def _make_optional_extractor(model: SparseAtomicGPR | None) -> ClusterExpansion | None:
        if model is None:
            return None
        return ClusterExpansion(model.config)

    def _zero_energy(self) -> torch.Tensor:
        device = self.device if self.device is not None else torch.device("cpu")
        return torch.zeros((), dtype=self.dtype, device=device)

    def _as_tensor(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.dtype)
        else:
            x = x.to(dtype=self.dtype)

        if self.device is not None:
            x = x.to(self.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise ValueError(f"Descriptor must be 2D, got shape {tuple(x.shape)}.")

        if x.shape[0] == 0:
            raise ValueError("Descriptor has zero rows.")

        if not torch.isfinite(x).all():
            raise ValueError("Descriptor contains NaN or Inf values.")

        return x

    @staticmethod
    def _symbols(atoms: ase.Atoms) -> np.ndarray:
        return np.asarray(atoms.get_chemical_symbols())

    def _selected_indices_compatible_with_model(
        self,
        atoms: ase.Atoms,
        atom_indices,
        model: SparseAtomicGPR | None,
    ) -> np.ndarray:
        if model is None or model.config is None:
            return np.array([], dtype=int)

        atom_indices = np.asarray(atom_indices, dtype=int).ravel()
        if atom_indices.size == 0:
            return np.array([], dtype=int)

        symbols = self._symbols(atoms)
        allowed = set(model.config.elements)
        keep = [int(i) for i in atom_indices if 0 <= int(i) < len(symbols) and symbols[int(i)] in allowed]
        return np.array(keep, dtype=int)

    def _model_has_any_atoms(self, atoms: ase.Atoms, model: SparseAtomicGPR | None) -> bool:
        if model is None or model.config is None:
            return False
        symbols = set(atoms.get_chemical_symbols())
        return bool(symbols.intersection(set(model.config.elements)))

    def _predict_scalar(self, model: SparseAtomicGPR, desc) -> torch.Tensor:
        desc = self._as_tensor(desc)
        y = model([desc])
        return y.reshape(-1).sum()

    def _predict_mean_std_scalar(
        self,
        model: SparseAtomicGPR,
        desc,
        uncertainty_mode: str = "quadrature",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        desc = self._as_tensor(desc)
        mean, std = model.predict_uncertainty([desc])

        mean = torch.as_tensor(mean, dtype=self.dtype, device=desc.device)
        std = torch.as_tensor(std, dtype=self.dtype, device=desc.device)

        energy = mean.reshape(-1).sum()
        std = std.reshape(-1)

        if uncertainty_mode == "quadrature":
            uncertainty = torch.sqrt(torch.sum(std ** 2))
        elif uncertainty_mode == "sum":
            uncertainty = torch.sum(std)
        elif uncertainty_mode == "max":
            uncertainty = torch.max(std)
        else:
            raise ValueError(
                "Unknown uncertainty_mode. Expected 'quadrature', 'sum' or 'max', "
                f"got {uncertainty_mode!r}."
            )

        return energy, uncertainty

    def _component_result(
        self,
        model: SparseAtomicGPR | None,
        desc,
        compute_uncertainty: bool,
        uncertainty_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Return energy, uncertainty, applied flag for one model component."""
        if model is None or desc is None:
            zero = self._zero_energy()
            return zero, zero, False

        desc = np.asarray(desc, dtype=float)
        if desc.ndim != 2 or desc.shape[0] == 0:
            zero = self._zero_energy()
            return zero, zero, False

        if compute_uncertainty:
            energy, unc = self._predict_mean_std_scalar(
                model=model,
                desc=desc,
                uncertainty_mode=uncertainty_mode,
            )
        else:
            energy = self._predict_scalar(model, desc)
            unc = self._zero_energy()

        return energy, unc, True

    def _slab_descriptor(self, atoms: ase.Atoms):
        if self.slab_model is None or self.slab_extractor is None:
            return None
        if not self._model_has_any_atoms(atoms, self.slab_model):
            return None
        return self.slab_extractor(atoms)

    def _co_indices(self, atoms: ase.Atoms):
        atom_indices, carbon_indices, labels_per_atom = atoms_near_carbon(atoms)

        if len(atom_indices) != len(labels_per_atom):
            raise RuntimeError(
                "atoms_near_carbon returned inconsistent atom_indices and labels_per_atom."
            )

        return np.asarray(atom_indices, dtype=int), np.asarray(carbon_indices, dtype=int), labels_per_atom

    def _adsorption_site_descriptors(self, atoms: ase.Atoms):
        if self.ads_model is None or self.ads_extractor is None:
            return None, np.array([], dtype=int), np.array([], dtype=int)

        atom_indices, carbon_indices, labels_per_atom = self._co_indices(atoms)
        if len(carbon_indices) == 0 or len(atom_indices) == 0:
            return None, carbon_indices, np.array([], dtype=int)

        compatible = self._selected_indices_compatible_with_model(atoms, atom_indices, self.ads_model)
        if len(compatible) == 0:
            return None, carbon_indices, np.array([], dtype=int)

        compatible_set = set(int(i) for i in compatible)
        filtered_labels = [
            labels for idx, labels in zip(atom_indices, labels_per_atom) if int(idx) in compatible_set
        ]

        ads_desc = self.ads_extractor(atoms, atom_indices=compatible)
        ads_desc = self._as_tensor(ads_desc)

        x_site, site_labels = aggregate_multi_label_descriptors(
            x=ads_desc,
            labels_per_atom=filtered_labels,
        )

        if x_site.shape[0] == 0:
            return None, carbon_indices, site_labels

        return x_site.detach().cpu().numpy(), carbon_indices, site_labels

    def _repulsion_descriptors(self, atoms: ase.Atoms, carbon_indices: np.ndarray):
        if self.rep_model is None or self.rep_extractor is None:
            return None
        if len(carbon_indices) <= 1:
            return None

        compatible = self._selected_indices_compatible_with_model(atoms, carbon_indices, self.rep_model)
        if len(compatible) == 0:
            return None

        return self.rep_extractor(atoms, atom_indices=compatible)

    def _evaluate_components(
        self,
        atoms: ase.Atoms,
        compute_uncertainty: bool,
        uncertainty_mode: str = "quadrature",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, bool]]:
        with torch.no_grad():
            slab_desc = self._slab_descriptor(atoms)
            slab_energy, slab_unc, slab_applied = self._component_result(
                self.slab_model,
                slab_desc,
                compute_uncertainty=compute_uncertainty,
                uncertainty_mode=uncertainty_mode,
            )

            ads_desc, carbon_indices, _ = self._adsorption_site_descriptors(atoms)
            ads_energy, ads_unc, ads_applied = self._component_result(
                self.ads_model,
                ads_desc,
                compute_uncertainty=compute_uncertainty,
                uncertainty_mode=uncertainty_mode,
            )

            rep_desc = self._repulsion_descriptors(atoms, carbon_indices)
            rep_energy, rep_unc, rep_applied = self._component_result(
                self.rep_model,
                rep_desc,
                compute_uncertainty=compute_uncertainty,
                uncertainty_mode=uncertainty_mode,
            )

            total_energy = slab_energy + ads_energy + rep_energy
            total_uncertainty = torch.sqrt(slab_unc ** 2 + ads_unc ** 2 + rep_unc ** 2)

            component_uncertainties = {
                "slab": slab_unc,
                "ads": ads_unc,
                "rep": rep_unc,
                "total": total_uncertainty,
            }
            component_applied = {
                "slab": slab_applied,
                "ads": ads_applied,
                "rep": rep_applied,
            }

        return (
            slab_energy,
            total_energy,
            ads_energy,
            rep_energy,
            total_uncertainty,
            component_uncertainties,
            component_applied,
        )

    def __call__(self, atoms: ase.Atoms):
        """Return slab_energy, total_energy, ads_energy, rep_energy."""
        result = self._evaluate_components(
            atoms=atoms,
            compute_uncertainty=False,
        )
        slab_energy, total_energy, ads_energy, rep_energy = result[:4]
        return slab_energy, total_energy, ads_energy, rep_energy

    def predict_energy_and_uncertainty(
        self,
        atoms: ase.Atoms,
        uncertainty_mode: str = "quadrature",
        return_applied: bool = False,
    ):

        result = self._evaluate_components(
            atoms=atoms,
            compute_uncertainty=True,
            uncertainty_mode=uncertainty_mode,
        )

        if return_applied:
            return result

        return result[:6]

    def uncertainty(self, atoms: ase.Atoms, uncertainty_mode: str = "quadrature"):
        result = self.predict_energy_and_uncertainty(
            atoms=atoms,
            uncertainty_mode=uncertainty_mode,
        )
        return result[4], result[5]

    def predict_uncertainty(self, atoms: ase.Atoms, uncertainty_mode: str = "quadrature"):
        total_uncertainty, _ = self.uncertainty(
            atoms=atoms,
            uncertainty_mode=uncertainty_mode,
        )
        return total_uncertainty
