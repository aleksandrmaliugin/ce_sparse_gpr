from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from tqdm import tqdm

from .ce_config import CEConfig


class SparseAtomicGPR(nn.Module):

    def __init__(
        self,
        x_train=None,
        model_path=None,
        config: CEConfig | None = None,
        M: int = 100,
        div: float = 0.001,
        init_lengthscale=1.0,
        init_sigma2: float = 1e-4,
        init_outputscale: float = 1.0,
        jitter: float = 1e-10,
        device=None,
        allow_unsafe_load: bool = False,
    ):
        super().__init__()

        self.device = torch.device(device or "cpu")
        self.jitter = float(jitter)
        self.config = config
        self.allow_unsafe_load = bool(allow_unsafe_load)

        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")

        self.register_buffer("y_train", None)
        self.register_buffer("x_M", None)
        self.register_buffer("K_NM_train", None)
        self.register_buffer("c", None)
        self.register_buffer("L_KMM", None)
        self.register_buffer("L_KSS", None)

        if model_path is not None:
            self._load_model(model_path, allow_unsafe_load=allow_unsafe_load)
            return

        if x_train is None:
            raise ValueError("Provide x_train if model_path is not given.")

        x_list = self._as_structure_list(x_train)
        if len(x_list) == 0:
            raise ValueError("x_train must contain at least one structure.")

        D = x_list[0].shape[1]
        self.log_lengthscale = nn.Parameter(
            self._init_positive_vector(init_lengthscale, D, "init_lengthscale").log()
        )
        self.log_sigma2 = nn.Parameter(
            self._init_positive_scalar(init_sigma2, "init_sigma2").log()
        )
        self.log_outputscale = nn.Parameter(
            self._init_positive_scalar(init_outputscale, "init_outputscale").log()
        )

        self.x_M = self.select_inducing_points(x_list, M=M, div=div).to(self.device)

    def _init_positive_scalar(self, value, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float64, device=self.device)

        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar, got shape {tuple(value.shape)}.")

        if not torch.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")

        return value

    def _init_positive_vector(self, value, size: int, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float64, device=self.device)

        if value.ndim == 0:
            value = value.repeat(size)
        elif value.ndim != 1 or value.shape[0] != size:
            raise ValueError(
                f"{name} has shape {tuple(value.shape)}, expected scalar or ({size},)."
            )

        if not torch.isfinite(value).all():
            raise ValueError(f"All {name} values must be finite.")

        if torch.any(value <= 0.0):
            raise ValueError(f"All {name} values must be positive.")

        return value

    def _safe_torch_load(self, model_path, allow_unsafe_load: bool):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        try:
            return torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as safe_error:
            if not allow_unsafe_load:
                raise RuntimeError(
                    "Could not load checkpoint with weights_only=True. If this is an "
                    "old trusted checkpoint created by you, reload with "
                    "allow_unsafe_load=True. Do not use allow_unsafe_load=True for "
                    "untrusted model files."
                ) from safe_error

            return torch.load(model_path, map_location=self.device, weights_only=False)

    def _load_model(self, model_path, allow_unsafe_load: bool = False) -> None:
        state = self._safe_torch_load(model_path, allow_unsafe_load=allow_unsafe_load)

        required = ["x_M", "log_sigma2"]
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"Checkpoint is missing required keys: {missing}.")

        self.y_train = state.get("y_train")
        self.y_train = self.y_train.to(self.device) if self.y_train is not None else None

        self.x_M = state["x_M"].to(self.device)
        self.jitter = float(state.get("jitter", self.jitter))

        self.K_NM_train = state.get("K_NM_train")
        self.K_NM_train = (
            self.K_NM_train.to(self.device) if self.K_NM_train is not None else None
        )

        self.c = state.get("c")
        self.c = self.c.to(self.device) if self.c is not None else None

        self.L_KMM = state.get("L_KMM")
        self.L_KMM = self.L_KMM.to(self.device) if self.L_KMM is not None else None

        self.L_KSS = state.get("L_KSS")
        self.L_KSS = self.L_KSS.to(self.device) if self.L_KSS is not None else None

        D = self.x_M.shape[1]
        default_lengthscale = torch.ones(D, dtype=torch.float64, device=self.device).log()
        default_outputscale = torch.tensor(1.0, dtype=torch.float64, device=self.device).log()

        self.log_lengthscale = nn.Parameter(
            state.get("log_lengthscale", default_lengthscale).to(self.device)
        )
        self.log_sigma2 = nn.Parameter(state["log_sigma2"].to(self.device))
        self.log_outputscale = nn.Parameter(
            state.get("log_outputscale", default_outputscale).to(self.device)
        )

        cfg = state.get("config")
        self.config = CEConfig.from_dict(cfg) if cfg is not None else None

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale) + 1e-12

    @property
    def sigma2(self) -> torch.Tensor:
        return torch.exp(self.log_sigma2) + 1e-12

    @property
    def outputscale(self) -> torch.Tensor:
        return torch.exp(self.log_outputscale) + 1e-12

    def _as_2d_tensor(self, x) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise ValueError(f"Each structure descriptor must be 2D, got {tuple(x.shape)}.")

        if x.shape[1] == 0:
            raise ValueError("Descriptor dimension is zero.")

        if not torch.isfinite(x).all():
            raise ValueError("Descriptor contains NaN or Inf values.")

        return x

    def _as_structure_list(self, x) -> list[torch.Tensor]:
        if torch.is_tensor(x):
            return [self._as_2d_tensor(x)]

        if isinstance(x, (list, tuple)):
            return [self._as_2d_tensor(item) for item in x]

        return [self._as_2d_tensor(x)]

    def rbf_kernel(self, x1, x2) -> torch.Tensor:
        x1 = self._as_2d_tensor(x1)
        x2 = self._as_2d_tensor(x2)

        if x1.shape[1] != self.lengthscale.shape[0] or x2.shape[1] != self.lengthscale.shape[0]:
            raise ValueError(
                f"Descriptor dimension mismatch in kernel: x1={x1.shape[1]}, "
                f"x2={x2.shape[1]}, lengthscale={self.lengthscale.shape[0]}."
            )

        diff = (x1[:, None, :] - x2[None, :, :]) / self.lengthscale[None, None, :]
        dist2 = (diff ** 2).sum(dim=-1)
        K = self.outputscale * torch.exp(-0.5 * dist2)

        if not torch.isfinite(K).all():
            raise RuntimeError("RBF kernel contains NaN or Inf values.")

        return K

    @torch.no_grad()
    def select_inducing_points(self, x_train, M: int = 100, div: float = 0.001) -> torch.Tensor:
        if M <= 0:
            raise ValueError("M must be positive.")

        if div < 0.0:
            raise ValueError("div must be non-negative.")

        x_list = self._as_structure_list(x_train)
        all_atoms = torch.cat(x_list, dim=0)

        if all_atoms.shape[0] == 0:
            raise ValueError("Cannot select inducing points from an empty training set.")

        x_M = [all_atoms[0]]

        for atom in tqdm(all_atoms[1:]):
            xm = torch.stack(x_M, dim=0)
            k = self.rbf_kernel(atom[None, :], xm).squeeze(0)
            k_xx = self.rbf_kernel(atom[None, :], atom[None, :]).squeeze()
            k_mm = self.rbf_kernel(xm, xm).diagonal()

            denom = torch.sqrt(k_xx * k_mm).clamp_min(1e-300)
            sims = k / denom

            if sims.max() < div:
                x_M.append(atom)

            if len(x_M) >= M:
                break

        print(f"Selected {len(x_M)} inducing points out of {M}")
        return torch.stack(x_M, dim=0)

    def build_K_NM(self, x_list) -> torch.Tensor:
        x_list = self._as_structure_list(x_list)

        if self.x_M is None:
            raise RuntimeError("Inducing points x_M are not initialized.")

        if len(x_list) == 0:
            raise ValueError("x_list must contain at least one structure.")

        M = self.x_M.shape[0]
        K_NM = torch.empty(len(x_list), M, dtype=torch.float64, device=self.x_M.device)

        for s, x in enumerate(x_list):
            if x.shape[1] != self.x_M.shape[1]:
                raise ValueError(
                    f"Descriptor dimension mismatch for structure {s}: "
                    f"got {x.shape[1]}, expected {self.x_M.shape[1]}."
                )
            K_NM[s] = self.rbf_kernel(x, self.x_M).sum(dim=0)

        return K_NM

    def safe_cholesky(self, A: torch.Tensor, jitter=None, max_tries: int = 5, name: str = "matrix"):
        if jitter is None:
            jitter = self.jitter

        if jitter <= 0.0:
            raise ValueError("jitter must be positive.")

        A = torch.as_tensor(A, dtype=torch.float64, device=self.device)
        A = 0.5 * (A + A.T)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got {tuple(A.shape)}.")

        if not torch.isfinite(A).all():
            raise RuntimeError(f"Cholesky failed for {name}: matrix contains NaN or Inf.")

        eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        current_jitter = float(jitter)

        for _ in range(max_tries):
            try:
                return torch.linalg.cholesky(A + current_jitter * eye)
            except torch.linalg.LinAlgError:
                current_jitter *= 10.0

        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A.detach())
            eig_min = eigvals.min().item()
            eig_max = eigvals.max().item()

        raise RuntimeError(
            f"Cholesky failed for {name}: "
            f"final_jitter={current_jitter:.2e}, "
            f"eig_min={eig_min:.3e}, eig_max={eig_max:.3e}, "
            f"sigma2={self.sigma2.detach().item():.3e}, "
            f"outputscale={self.outputscale.detach().item():.3e}, "
            f"lengthscale_min={self.lengthscale.detach().min().item():.3e}, "
            f"lengthscale_max={self.lengthscale.detach().max().item():.3e}"
        )

    def solve_c(self, K_NM: torch.Tensor, y) -> torch.Tensor:
        y = torch.as_tensor(y, dtype=torch.float64, device=self.x_M.device)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {tuple(y.shape)}.")

        if K_NM.shape[0] != y.shape[0]:
            raise ValueError(
                f"K_NM has {K_NM.shape[0]} rows, but y has length {y.shape[0]}."
            )

        K_MM = self.rbf_kernel(self.x_M, self.x_M)

        # Algebraically equivalent to
        #   K_MM + K_NM.T @ K_NM / sigma2
        # but avoids explicit division by small sigma2 and is much better
        # conditioned numerically.
        A = self.sigma2 * K_MM + K_NM.T @ K_NM
        b = K_NM.T @ y

        L = self.safe_cholesky(A, name="sigma2*K_MM + K_NM.T@K_NM")
        return torch.cholesky_solve(b[:, None], L).squeeze(-1)

    def rmse_loss(self, train_x, train_y) -> torch.Tensor:
        y = torch.as_tensor(train_y, dtype=torch.float64, device=self.x_M.device)
        K_NM = self.build_K_NM(train_x)
        c = self.solve_c(K_NM, y)
        y_pred = K_NM @ c
        return ((y_pred - y) ** 2).mean()

    def neg_log_like_loss(self, train_x, train_y) -> torch.Tensor:
        y = torch.as_tensor(train_y, dtype=torch.float64, device=self.x_M.device)

        if y.ndim != 1:
            raise ValueError(f"train_y must be 1D, got shape {tuple(y.shape)}.")

        K_NM = self.build_K_NM(train_x)
        K_MM = self.rbf_kernel(self.x_M, self.x_M)

        L_MM = self.safe_cholesky(K_MM, name="K_MM")
        K_MM_inv_K_MN = torch.cholesky_solve(K_NM.T, L_MM)
        K_NN = K_NM @ K_MM_inv_K_MN

        N = y.shape[0]
        A = K_NN + self.sigma2 * torch.eye(N, dtype=torch.float64, device=self.x_M.device)
        L = self.safe_cholesky(A, name="K_NN + sigma2*I")
        alpha = torch.cholesky_solve(y[:, None], L).squeeze(-1)

        loss = 0.5 * (y @ alpha)
        loss = loss + torch.log(torch.diagonal(L)).sum()

        if not torch.isfinite(loss):
            raise RuntimeError("Negative log-likelihood became NaN or Inf.")

        return loss

    def fit_c(self, train_x, train_y, build_uncertainty: bool = False) -> torch.Tensor:
        y = torch.as_tensor(train_y, dtype=torch.float64, device=self.x_M.device)

        if y.ndim != 1:
            raise ValueError(f"train_y must be 1D, got shape {tuple(y.shape)}.")

        K_NM = self.build_K_NM(train_x)
        c = self.solve_c(K_NM, y)

        self.c = c.detach()
        self.K_NM_train = K_NM.detach()
        self.y_train = y.detach()

        K_MM = self.rbf_kernel(self.x_M, self.x_M).detach()
        self.L_KMM = self.safe_cholesky(K_MM, name="K_MM")

        if build_uncertainty:
            K_MM_inv_K_NM_T = torch.cholesky_solve(self.K_NM_train.T, self.L_KMM)
            K_SS = self.K_NM_train @ K_MM_inv_K_NM_T
            K_SS = K_SS + self.sigma2.detach() * torch.eye(
                K_SS.shape[0], dtype=torch.float64, device=self.x_M.device
            )
            self.L_KSS = self.safe_cholesky(K_SS, name="K_SS")

        return self.c

    def fit_c_no_grad(self, train_x, train_y, build_uncertainty: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.fit_c(train_x, train_y, build_uncertainty=build_uncertainty)

    def check_descriptor_dim(self, x) -> None:
        x_list = self._as_structure_list(x)
        expected = self.x_M.shape[1]

        for idx, desc in enumerate(x_list):
            if desc.shape[1] != expected:
                raise ValueError(
                    f"Descriptor dimension mismatch for structure {idx}: "
                    f"got {desc.shape[1]}, expected {expected}."
                )

    def forward(self, x) -> torch.Tensor:
        self.check_descriptor_dim(x)

        if self.c is None:
            raise RuntimeError("Call fit_c(x, y) before prediction.")

        K_NM = self.build_K_NM(x)
        return K_NM @ self.c

    def predict_uncertainty(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        if self.c is None:
            raise RuntimeError("Call fit_c(train_x, train_y) first.")

        if self.K_NM_train is None or self.L_KMM is None or self.L_KSS is None:
            raise RuntimeError(
                "Uncertainty matrices are missing. Re-run fit_c(..., build_uncertainty=True)."
            )

        K_NM_test = self.build_K_NM(x)
        mean = K_NM_test @ self.c

        K_MM_inv_K_NM_train_T = torch.cholesky_solve(self.K_NM_train.T, self.L_KMM)
        K_star_S = K_NM_test @ K_MM_inv_K_NM_train_T

        K_MM_inv_K_NM_test_T = torch.cholesky_solve(K_NM_test.T, self.L_KMM)
        K_star_star = K_NM_test @ K_MM_inv_K_NM_test_T

        tmp = torch.cholesky_solve(K_star_S.T, self.L_KSS)
        cov = K_star_star - K_star_S @ tmp
        cov = 0.5 * (cov + cov.T)

        var = torch.clamp(cov.diagonal(), min=1e-12)
        std = torch.sqrt(var)

        return mean, std

    @torch.no_grad()
    def diagnose_system(self, x, y=None) -> dict[str, float | int]:
        K_NM = self.build_K_NM(x)
        K_MM = self.rbf_kernel(self.x_M, self.x_M)
        A = self.sigma2 * K_MM + K_NM.T @ K_NM
        A = 0.5 * (A + A.T)

        eig_A = torch.linalg.eigvalsh(A)
        s_K = torch.linalg.svdvals(K_NM)

        info = {
            "sigma2": self.sigma2.item(),
            "outputscale": self.outputscale.item(),
            "lengthscale_min": self.lengthscale.min().item(),
            "lengthscale_max": self.lengthscale.max().item(),
            "K_NM_min": K_NM.min().item(),
            "K_NM_max": K_NM.max().item(),
            "K_NM_norm_mean": K_NM.norm(dim=1).mean().item(),
            "K_NM_singular_min": s_K.min().item(),
            "K_NM_singular_max": s_K.max().item(),
            "K_NM_rank_1e-10": int((s_K > 1e-10 * s_K.max()).sum().item()),
            "A_eig_min": eig_A.min().item(),
            "A_eig_max": eig_A.max().item(),
        }

        if y is not None:
            y = torch.as_tensor(y, dtype=torch.float64, device=self.x_M.device)
            info.update(
                {
                    "y_mean": y.mean().item(),
                    "y_std": y.std(unbiased=False).item(),
                    "y_min": y.min().item(),
                    "y_max": y.max().item(),
                }
            )

        return info

    def save(self, path) -> None:
        torch.save(
            {
                "x_M": self.x_M.detach(),
                "jitter": self.jitter,
                "log_lengthscale": self.log_lengthscale.detach(),
                "log_sigma2": self.log_sigma2.detach(),
                "log_outputscale": self.log_outputscale.detach(),
                "c": self.c.detach() if self.c is not None else None,
                "K_NM_train": self.K_NM_train.detach() if self.K_NM_train is not None else None,
                "y_train": self.y_train.detach() if self.y_train is not None else None,
                "L_KMM": self.L_KMM.detach() if self.L_KMM is not None else None,
                "L_KSS": self.L_KSS.detach() if self.L_KSS is not None else None,
                "config": self.config.to_dict() if self.config is not None else None,
            },
            path,
        )
