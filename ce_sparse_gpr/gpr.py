from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .ce_config import CEConfig


@torch.jit.script
def _rbf_kernel_core(
    x1: torch.Tensor,
    x2: torch.Tensor,
    ls: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Compiled inner loop of the RBF kernel — called from every hot path."""
    dist2 = torch.cdist(x1 / ls, x2 / ls).pow(2)
    return scale * torch.exp(-0.5 * dist2)


def inv_softplus(value: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: raw such that softplus(raw) == value, for value > 0.

    Stable form (log(exp(v)-1) = v + log(1 - exp(-v))) - avoids overflowing
    exp(v) for large v.
    """
    return value + torch.log(-torch.expm1(-value))


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
        induce_lengthscale=None,
        jitter: float = 1e-10,
        device=None,
        dtype: torch.dtype = torch.float64,
        allow_unsafe_load: bool = False,
        mean_function: str = "zero",
        mean_l2_reg: float = 0.0,
    ):
        super().__init__()

        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self.jitter = float(jitter)
        self.config = config
        self.allow_unsafe_load = bool(allow_unsafe_load)

        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")

        if mean_l2_reg < 0.0:
            raise ValueError("mean_l2_reg must be non-negative.")
        self.mean_l2_reg = float(mean_l2_reg)

        self.register_buffer("y_train", None)
        self.register_buffer("x_M", None)
        self.register_buffer("K_NM_train", None)
        self.register_buffer("c", None)
        self.register_buffer("L_KMM", None)
        self.register_buffer("L_A", None)

        if model_path is not None:
            self._load_model(model_path, allow_unsafe_load=allow_unsafe_load)
            return

        if x_train is None:
            raise ValueError("Provide x_train if model_path is not given.")

        x_list = self._as_structure_list(x_train)
        if len(x_list) == 0:
            raise ValueError("x_train must contain at least one structure.")

        D = x_list[0].shape[1]
        self.raw_lengthscale = nn.Parameter(
            inv_softplus(self._init_positive_vector(init_lengthscale, D, "init_lengthscale"))
        )
        self.raw_sigma2 = nn.Parameter(
            inv_softplus(self._init_positive_scalar(init_sigma2, "init_sigma2"))
        )
        self.raw_outputscale = nn.Parameter(
            inv_softplus(self._init_positive_scalar(init_outputscale, "init_outputscale"))
        )

        self.mean_function = self._validate_mean_function(mean_function)
        if self.mean_function == "linear":
            self.linear_mean = nn.Parameter(torch.zeros(D, dtype=self.dtype, device=self.device))
        else:
            self.linear_mean = None

        selection_lengthscale = (
            self._init_positive_vector(induce_lengthscale, D, "induce_lengthscale")
            if induce_lengthscale is not None
            else None
        )
        self.x_M = self.select_inducing_points(x_list, M=M, div=div, lengthscale=selection_lengthscale).to(self.device)

    @staticmethod
    def _validate_mean_function(mean_function: str) -> str:
        if mean_function not in ("zero", "linear"):
            raise ValueError(f"mean_function must be 'zero' or 'linear', got {mean_function!r}.")
        return mean_function

    def mean_function_values(self, x, skip_validation: bool = False) -> torch.Tensor:
        """m(x) for each structure in x: 0 for mean_function="zero" (the GP's
        usual default, unchanged), or an additive linear-in-descriptor trend
        (sum over atoms of atom_descriptor @ linear_mean) for "linear".

        Why: an RBF kernel is a strictly LOCAL interpolator - far from every
        inducing point its prediction reverts to the (zero, by default) prior
        mean, not to whatever real trend the data has. For a quantity like
        CO-CO repulsion that grows smoothly and monotonically with coverage,
        that means the model doesn't just get uncertain past its training
        envelope, it silently collapses toward 0 (verified: sparse_atomic_gpr
        _rep___.pt predicts ~0 for coverages a bit past its densest training
        points, while a plain linear fit on the same descriptor tracks the
        true DFT values to a few % even there). A linear mean function fixes
        this at the root: the GP now only has to learn the SMALL local
        correction on top of an explicit linear trend that keeps extrapolating
        correctly by construction, while keeping the RBF kernel for
        everything the trend alone can't capture (and calibrated uncertainty
        via the same machinery as the zero-mean case).
        """
        x_list = self._as_structure_list(x, skip_validation=skip_validation)
        device = self.x_M.device if self.x_M is not None else self.device

        if self.linear_mean is None:
            return torch.zeros(len(x_list), dtype=self.dtype, device=device)

        sums = torch.stack([xs.sum(dim=0) for xs in x_list], dim=0)  # (n_structures, D)
        return sums.to(dtype=self.linear_mean.dtype, device=self.linear_mean.device) @ self.linear_mean

    def _init_positive_scalar(self, value, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=self.dtype).to(self.device)

        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar, got shape {tuple(value.shape)}.")

        if not torch.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")

        return value

    def _init_positive_vector(self, value, size: int, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=self.dtype).to(self.device)

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

        required = ["x_M", "raw_sigma2"]
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"Checkpoint is missing required keys: {missing}.")

        self.y_train = state.get("y_train")
        self.y_train = self.y_train.to(self.device) if self.y_train is not None else None

        self.x_M = state["x_M"].to(self.device)
        # Restore dtype from checkpoint; overrides constructor argument so the
        # loaded model uses exactly the dtype it was saved with.
        dtype_str = state.get("dtype", "float64")
        self.dtype = getattr(torch, dtype_str, torch.float64)
        self.jitter = float(state.get("jitter", self.jitter))

        self.K_NM_train = state.get("K_NM_train")
        self.K_NM_train = (
            self.K_NM_train.to(self.device) if self.K_NM_train is not None else None
        )

        self.c = state.get("c")
        self.c = self.c.to(self.device) if self.c is not None else None

        self.L_KMM = state.get("L_KMM")
        self.L_KMM = self.L_KMM.to(self.device) if self.L_KMM is not None else None

        # Renamed from the old N_train x N_train "L_KSS" (DTC route) to the
        # M x M "L_A" (Projected Process route, numerically identical
        # predictions - see fit_c/predict_uncertainty) - old checkpoints
        # simply won't have this key, so predict_uncertainty will correctly
        # ask for a re-fit instead of loading a stale, wrong-shaped matrix.
        self.L_A = state.get("L_A")
        self.L_A = self.L_A.to(self.device) if self.L_A is not None else None

        D = self.x_M.shape[1]
        default_lengthscale = inv_softplus(torch.ones(D, dtype=self.dtype, device=self.device))
        default_outputscale = inv_softplus(torch.tensor(1.0, dtype=self.dtype, device=self.device))

        self.raw_lengthscale = nn.Parameter(
            state.get("raw_lengthscale", default_lengthscale).to(self.device)
        )
        self.raw_sigma2 = nn.Parameter(state["raw_sigma2"].to(self.device))
        self.raw_outputscale = nn.Parameter(
            state.get("raw_outputscale", default_outputscale).to(self.device)
        )

        # Old checkpoints simply won't have these keys - they default to the
        # zero mean function, identical to their original (pre-mean-function)
        # behavior.
        self.mean_function = self._validate_mean_function(state.get("mean_function", "zero"))
        linear_mean = state.get("linear_mean")
        self.linear_mean = (
            nn.Parameter(linear_mean.to(self.device)) if linear_mean is not None else None
        )
        # Old checkpoints won't have this key - 0.0 reproduces the old,
        # unregularized behavior exactly (this only affects neg_log_like_loss
        # during further training, not inference/forward).
        self.mean_l2_reg = float(state.get("mean_l2_reg", 0.0))

        cfg = state.get("config")
        self.config = CEConfig.from_dict(cfg) if cfg is not None else None

    @property
    def lengthscale(self) -> torch.Tensor:
        return F.softplus(self.raw_lengthscale) + 1e-12

    @property
    def sigma2(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma2) + 1e-12

    @property
    def outputscale(self) -> torch.Tensor:
        return F.softplus(self.raw_outputscale) + 1e-12

    def _as_2d_tensor(self, x, skip_validation: bool = False) -> torch.Tensor:
        # Use x_M.device as the ground truth when available: it updates correctly
        # after model.to(device) calls, unlike self.device which is set at init time.
        device = self.x_M.device if self.x_M is not None else self.device

        # skip_validation=True is an explicit opt-in from the training hot
        # loop (see train.py's prepare_xy + neg_log_like_loss/fit_c/forward
        # calls there), never a default: train_x/valid_x are converted and
        # isfinite-checked ONCE by prepare_xy before training starts, then
        # this same object is passed back into build_K_NM on every single
        # closure evaluation (hundreds+ times per LBFGS outer step). Redoing
        # torch.as_tensor + torch.isfinite(...).all() on every one of those
        # calls measured at ~50% of total training step time, checking data
        # that cannot have changed since the last check (it's not a live
        # tensor being mutated - see prepare_xy). Every other caller
        # (inference via forward()/predict_uncertainty(), or training calls
        # that don't pass the flag) keeps the full check on every call.
        if (
            skip_validation
            and torch.is_tensor(x)
            and x.dtype == self.dtype
            and x.device == device
            and x.ndim == 2
            and x.shape[1] > 0
        ):
            return x

        x = torch.as_tensor(x, dtype=self.dtype).to(device)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise ValueError(f"Each structure descriptor must be 2D, got {tuple(x.shape)}.")

        if x.shape[1] == 0:
            raise ValueError("Descriptor dimension is zero.")

        if not torch.isfinite(x).all():
            raise ValueError("Descriptor contains NaN or Inf values.")

        return x

    def _as_structure_list(self, x, skip_validation: bool = False) -> list[torch.Tensor]:
        if torch.is_tensor(x):
            return [self._as_2d_tensor(x, skip_validation=skip_validation)]

        if isinstance(x, (list, tuple)):
            return [self._as_2d_tensor(item, skip_validation=skip_validation) for item in x]

        return [self._as_2d_tensor(x, skip_validation=skip_validation)]

    def rbf_kernel(self, x1, x2) -> torch.Tensor:
        x1 = self._as_2d_tensor(x1)
        x2 = self._as_2d_tensor(x2)

        ls = self.lengthscale
        if x1.shape[1] != ls.shape[0] or x2.shape[1] != ls.shape[0]:
            raise ValueError(
                f"Descriptor dimension mismatch in kernel: x1={x1.shape[1]}, "
                f"x2={x2.shape[1]}, lengthscale={ls.shape[0]}."
            )

        K = _rbf_kernel_core(x1, x2, ls, self.outputscale)

        if not torch.isfinite(K).all():
            raise RuntimeError("RBF kernel contains NaN or Inf values.")

        return K

    @torch.no_grad()
    def select_inducing_points(self, x_train, M: int = 100, div: float = 0.001, lengthscale=None) -> torch.Tensor:
        if M <= 0:
            raise ValueError("M must be positive.")

        if div < 0.0:
            raise ValueError("div must be non-negative.")

        x_list = self._as_structure_list(x_train)
        all_atoms = torch.cat(x_list, dim=0)
        N = all_atoms.shape[0]

        if N == 0:
            raise ValueError("Cannot select inducing points from an empty training set.")

        # Defaults to self.lengthscale (unchanged behavior), but callers that
        # warm-start raw_lengthscale from a PREVIOUS run's converged value
        # (see ce_gpr_train.resolve_warm_start) must pass a fresh, neutral
        # `lengthscale` here instead: a converged ARD lengthscale routinely
        # has most dimensions pushed to near-irrelevance (lengthscale >> data
        # spread) with only one or two still small/discriminative - scaling
        # candidate distances by THAT collapses almost every point into
        # "duplicate" under a fixed div threshold (verified: M=450 requested,
        # only 2 points survived), because select_inducing_points is a
        # GEOMETRIC diversity criterion, not an optimization starting point.
        ls = self.lengthscale if lengthscale is None else lengthscale
        scaled_atoms = all_atoms / ls

        # Process candidates in chunks: one batched cdist against the current
        # x_M covers a whole chunk (instead of one cdist launch per atom), and
        # within a chunk each acceptance only has to update distances for the
        # rest of that chunk (instead of the full remaining array). This keeps
        # both the sparse-acceptance case (few, cheap chunks) and the
        # dense-scan case (many duplicate atoms, most of x_train gets visited)
        # fast, unlike a pure per-atom or pure full-suffix update.
        #
        # The sequential accept/reject decision itself runs on CPU/NumPy: each
        # `if` on a CUDA tensor forces a device-to-host sync, so doing that
        # per atom (as a naive port of this loop would) serializes on GPU
        # round-trips and can make GPU slower than CPU here. Pulling one chunk
        # over at a time bounds the syncs to N / chunk_size instead of N.
        chunk_size = 256
        selected_idx = [0]
        i = 1

        with tqdm(total=N - 1) as pbar:
            while i < N and len(selected_idx) < M:
                chunk_end = min(i + chunk_size, N)
                chunk = scaled_atoms[i:chunk_end]
                x_M_scaled = scaled_atoms[selected_idx]
                min_dist2_gpu = torch.cdist(chunk, x_M_scaled).pow(2).min(dim=1).values

                chunk_np = chunk.detach().cpu().numpy()
                min_dist2 = min_dist2_gpu.detach().cpu().numpy()

                for offset in range(chunk_end - i):
                    if np.exp(-0.5 * min_dist2[offset]) < div:
                        selected_idx.append(i + offset)
                        if len(selected_idx) >= M:
                            break
                        if offset + 1 < (chunk_end - i):
                            diff = chunk_np[offset + 1 :] - chunk_np[offset]
                            new_dist2 = (diff * diff).sum(axis=1)
                            np.minimum(min_dist2[offset + 1 :], new_dist2, out=min_dist2[offset + 1 :])

                pbar.update(chunk_end - i)
                i = chunk_end

        x_M = all_atoms[selected_idx]

        print(f"Selected {x_M.shape[0]} inducing points out of {M}")
        return x_M

    def build_K_NM(self, x_list, skip_validation: bool = False) -> torch.Tensor:
        x_list = self._as_structure_list(x_list, skip_validation=skip_validation)

        if self.x_M is None:
            raise RuntimeError("Inducing points x_M are not initialized.")

        if len(x_list) == 0:
            raise ValueError("x_list must contain at least one structure.")

        device = self.x_M.device
        M = self.x_M.shape[0]
        D = self.x_M.shape[1]

        for s, x in enumerate(x_list):
            if x.shape[1] != D:
                raise ValueError(
                    f"Descriptor dimension mismatch for structure {s}: "
                    f"got {x.shape[1]}, expected {D}."
                )

        # Single batched kernel call: (total_atoms, M) computed all at once.
        # On GPU this fully utilises parallelism; on CPU it reduces Python overhead.
        sizes = [x.shape[0] for x in x_list]
        all_atoms = torch.cat(x_list, dim=0)  # (total_atoms, D)

        K_all = _rbf_kernel_core(
            all_atoms, self.x_M, self.lengthscale, self.outputscale
        )  # (total_atoms, M)

        if len(x_list) == 1:
            # k_M(x*)^T: a single structure's row of K_NM is just the sum of
            # its atoms' kernel rows — skip the scatter machinery below.
            return K_all.sum(dim=0, keepdim=True)

        # Scatter-sum each atom's contribution back to its structure.
        struct_ids = torch.repeat_interleave(
            torch.arange(len(x_list), device=device),
            torch.tensor(sizes, dtype=torch.long, device=device),
        )
        K_NM = torch.zeros(len(x_list), M, dtype=K_all.dtype, device=device)
        K_NM.scatter_add_(0, struct_ids.unsqueeze(1).expand_as(K_all), K_all)

        return K_NM

    def safe_cholesky(self, A: torch.Tensor, jitter=None, max_tries: int = 10, name: str = "matrix"):
        if jitter is None:
            jitter = self.jitter

        if jitter <= 0.0:
            raise ValueError("jitter must be positive.")

        A = 0.5 * (A + A.T)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got {tuple(A.shape)}.")

        if not torch.isfinite(A).all():
            raise RuntimeError(f"Cholesky failed for {name}: matrix contains NaN or Inf.")

        eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        current_jitter = float(jitter)

        # jitter is meant to be a tiny numerical nudge, not a substitute for
        # correct hyperparameters - cap how far it's allowed to escalate.
        # A cap relative to sigma2 (an earlier version used 1%, then 5%) falls
        # apart exactly when sigma2 is itself small (a near-perfect fold fit
        # pushes it toward its floor), leaving a razor-thin budget right when
        # a real defect is most likely to show up. A fixed absolute ceiling
        # (matching e.g. gpytorch's own default cholesky_jitter, which is
        # likewise a plain constant, not scaled by any model parameter) is
        # simpler and doesn't have that failure mode.
        max_jitter = max(current_jitter, 1e-6)

        for tries in range(max_tries):
            try:
                return torch.linalg.cholesky(A + current_jitter * eye)
            except torch.linalg.LinAlgError:
                if current_jitter >= max_jitter:
                    break
                # Escalate x10, but land exactly on max_jitter instead of
                # overshooting past it - otherwise a value that would have
                # fixed a small eigenvalue defect (e.g. cap=4.2e-6, needed
                # ~1.2e-6) can get skipped entirely by the x10 jump.
                current_jitter = min(current_jitter * 10.0, max_jitter)

        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A.detach())
            eig_min = eigvals.min().item()
            eig_max = eigvals.max().item()

        raise RuntimeError(
            f"Cholesky failed for {name}: "
            f"final_jitter={current_jitter:.2e} (capped at {max_jitter:.2e}), "
            f"eig_min={eig_min:.3e}, eig_max={eig_max:.3e}, "
            f"sigma2={self.sigma2.detach().item():.3e}, "
            f"outputscale={self.outputscale.detach().item():.3e}, "
            f"lengthscale_min={self.lengthscale.detach().min().item():.3e}, "
            f"lengthscale_max={self.lengthscale.detach().max().item():.3e}"
        )

    def _stable_L_A(self, K_MM: torch.Tensor, K_NM: torch.Tensor, L_MM: torch.Tensor) -> torch.Tensor:
        """Numerically stable Cholesky factor of A = sigma2*K_MM + K_NM.T @ K_NM.

        Forming A as that direct sum and factoring it can lose enough
        precision to look non-PSD (a small *negative* "eigenvalue") whenever
        K_MM is ill-conditioned - e.g. two nearly-duplicate inducing points -
        even though A is guaranteed PSD in exact arithmetic. Verified in
        practice: with two inducing points 1e-5 apart, the naive A had
        min eigenvalue 7e-8 (a hair from flipping negative); the reformulation
        below had min eigenvalue ~11, nowhere near singular, same inputs.

        gpytorch's InducingPointKernel avoids the same problem the same way
        (see _inducing_inv_root/solve_triangular in inducing_point_kernel.py):
        never build sigma2*K_MM + K_NM.T@K_NM as one dense sum. Instead,
        "whiten" K_NM by K_MM's own (separately jittered) Cholesky factor
        first, then add sigma2*I to *that* - a matrix whose eigenvalues are
        provably >= sigma2 regardless of how degenerate K_MM/K_NM are, so its
        own Cholesky essentially never needs jitter to rescue it:

            A = L_MM (sigma2*I + B B^T) L_MM^T,   B = L_MM^{-1} K_MN
            L_A = L_MM @ L_C,   where sigma2*I + B B^T = L_C L_C^T
        """
        M = K_MM.shape[0]
        B = torch.linalg.solve_triangular(L_MM, K_NM.T, upper=False)  # (M, N)
        C = self.sigma2 * torch.eye(M, dtype=K_MM.dtype, device=K_MM.device) + B @ B.T
        L_C = self.safe_cholesky(C, name="sigma2*I + B@B.T (whitened A)")
        return L_MM @ L_C

    def solve_c(self, K_NM: torch.Tensor, y) -> torch.Tensor:
        y = torch.as_tensor(y, dtype=self.x_M.dtype).to(self.x_M.device)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {tuple(y.shape)}.")

        if K_NM.shape[0] != y.shape[0]:
            raise ValueError(
                f"K_NM has {K_NM.shape[0]} rows, but y has length {y.shape[0]}."
            )

        K_MM = self.rbf_kernel(self.x_M, self.x_M)
        L_MM = self.safe_cholesky(K_MM, name="K_MM")
        L_A = self._stable_L_A(K_MM, K_NM, L_MM)
        b = K_NM.T @ y

        return torch.cholesky_solve(b[:, None], L_A).squeeze(-1)

    def rmse_loss(self, train_x, train_y, skip_validation: bool = False) -> torch.Tensor:
        y = torch.as_tensor(train_y, dtype=self.x_M.dtype).to(self.x_M.device)
        mean_vals = self.mean_function_values(train_x, skip_validation=skip_validation)
        K_NM = self.build_K_NM(train_x, skip_validation=skip_validation)
        c = self.solve_c(K_NM, y - mean_vals)
        y_pred = K_NM @ c + mean_vals
        return ((y_pred - y) ** 2).mean()

    def neg_log_like_loss(self, train_x, train_y, skip_validation: bool = False) -> torch.Tensor:
        y = torch.as_tensor(train_y, dtype=self.x_M.dtype).to(self.x_M.device)

        if y.ndim != 1:
            raise ValueError(f"train_y must be 1D, got shape {tuple(y.shape)}.")

        # The GP (kernel part) always models the RESIDUAL from the mean
        # function - y_resid == y when mean_function="zero" (unchanged
        # behavior), so this only has an effect when a mean function is set.
        mean_vals = self.mean_function_values(train_x, skip_validation=skip_validation)
        y_resid = y - mean_vals

        K_NM = self.build_K_NM(train_x, skip_validation=skip_validation)
        K_MM = self.rbf_kernel(self.x_M, self.x_M)

        # A = σ²K_MM + K_NM.T @ K_NM  (M×M instead of N×N)
        # Avoids forming the N×N Nyström matrix and O(N³) Cholesky.
        # (Cholesky of A is computed via _stable_L_A, not by forming A
        # directly - see that method's docstring for why.)
        L_MM = self.safe_cholesky(K_MM, name="K_MM")
        L_A = self._stable_L_A(K_MM, K_NM, L_MM)

        # Quadratic form via Woodbury:
        # y_resid^T (Q_NN + σ²I)^{-1} y_resid = (||y_resid||² - Kmy^T A^{-1} Kmy) / σ²
        Kmy = K_NM.T @ y_resid
        alpha = torch.cholesky_solve(Kmy[:, None], L_A).squeeze(-1)
        quad = (y_resid @ y_resid - Kmy @ alpha) / self.sigma2

        # Log-determinant via matrix determinant lemma:
        # log|Q_NN + σ²I| = log|A| - log|K_MM| + (N-M)·log(σ²)
        N = y_resid.shape[0]
        M = self.x_M.shape[0]
        log_det = (
            2.0 * torch.log(torch.diagonal(L_A)).sum()
            - 2.0 * torch.log(torch.diagonal(L_MM)).sum()
            + (N - M) * torch.log(self.sigma2)
        )

        loss = 0.5 * (quad + log_det)

        # Gaussian prior on the (otherwise completely unregularized) linear
        # mean weights: -log p(w) = 0.5*mean_l2_reg*||w||^2 + const, added
        # straight onto the NLL like any other Bayesian prior term. Without
        # this, nothing stops L-BFGS from continuing to trade the kernel's
        # local, data-driven signal (outputscale) for a larger global linear
        # term as it keeps shrinking the training NLL - observed directly on
        # the rep model: outputscale collapsed ~20x (0.073 -> 0.0034) while
        # ||linear_mean|| grew (0.72 -> 0.88) well past the step that actually
        # minimized RMSE valid, i.e. the extra NLL improvement in that regime
        # was pure hyperparameter overfitting, not a better fit. mean_l2_reg=0
        # (default) reproduces the old, unregularized behavior exactly.
        if self.linear_mean is not None and self.mean_l2_reg > 0.0:
            loss = loss + 0.5 * self.mean_l2_reg * (self.linear_mean**2).sum()

        if not torch.isfinite(loss):
            raise RuntimeError("Negative log-likelihood became NaN or Inf.")

        return loss

    def fit_c(self, train_x, train_y, build_uncertainty: bool = False, skip_validation: bool = False) -> torch.Tensor:
        y = torch.as_tensor(train_y, dtype=self.x_M.dtype).to(self.x_M.device)

        if y.ndim != 1:
            raise ValueError(f"train_y must be 1D, got shape {tuple(y.shape)}.")

        mean_vals = self.mean_function_values(train_x, skip_validation=skip_validation)
        K_NM = self.build_K_NM(train_x, skip_validation=skip_validation)
        c = self.solve_c(K_NM, y - mean_vals)

        self.c = c.detach()
        self.K_NM_train = K_NM.detach()
        self.y_train = y.detach()  # original (non-residual) targets, for reference

        K_MM = self.rbf_kernel(self.x_M, self.x_M).detach()
        self.L_KMM = self.safe_cholesky(K_MM, name="K_MM")

        if build_uncertainty:
            # Projected Process variance (Rasmussen & Williams, GPML S8.3.3):
            # numerically identical to the DTC variance this used to compute via
            # K_SS = Q_train,train + sigma^2 I (N_train x N_train), but needs only
            # A = sigma^2 K_MM + K_NM.T @ K_NM (M x M) - see predict_uncertainty.
            # (via _stable_L_A, not by forming A directly.)
            self.L_A = self._stable_L_A(K_MM, self.K_NM_train, self.L_KMM)

        return self.c

    def fit_c_no_grad(
        self, train_x, train_y, build_uncertainty: bool = False, skip_validation: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.fit_c(train_x, train_y, build_uncertainty=build_uncertainty, skip_validation=skip_validation)

    def check_descriptor_dim(self, x, skip_validation: bool = False) -> None:
        x_list = self._as_structure_list(x, skip_validation=skip_validation)
        expected = self.x_M.shape[1]

        for idx, desc in enumerate(x_list):
            if desc.shape[1] != expected:
                raise ValueError(
                    f"Descriptor dimension mismatch for structure {idx}: "
                    f"got {desc.shape[1]}, expected {expected}."
                )

    def forward(self, x, skip_validation: bool = False) -> torch.Tensor:
        self.check_descriptor_dim(x, skip_validation=skip_validation)

        if self.c is None:
            raise RuntimeError("Call fit_c(x, y) before prediction.")

        K_NM = self.build_K_NM(x, skip_validation=skip_validation)
        return K_NM @ self.c + self.mean_function_values(x, skip_validation=skip_validation)

    def predict_uncertainty(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        if self.c is None:
            raise RuntimeError("Call fit_c(train_x, train_y) first.")

        if self.L_KMM is None or self.L_A is None:
            raise RuntimeError(
                "Uncertainty matrices are missing. Re-run fit_c(..., build_uncertainty=True)."
            )

        x_list = self._as_structure_list(x)
        K_NM_test = self.build_K_NM(x_list)  # (N_test, M)
        # The kernel/GP part only ever modeled the residual from the mean
        # function (see fit_c) - add it back for the actual prediction. The
        # variance below is untouched: it's the variance of that same
        # residual process, and a (fixed, or jointly-fit-but-still-a-mean)
        # mean function doesn't change the GP's own predictive covariance.
        mean = K_NM_test @ self.c + self.mean_function_values(x_list)

        # Projected Process variance (Rasmussen & Williams, GPML eq. 8.27):
        #   var(x*) = k(x*,x*) - k_M(x*)^T K_MM^-1 k_M(x*)
        #             + sigma^2 k_M(x*)^T (sigma^2 K_MM + K_MN K_NM)^-1 k_M(x*)
        # Numerically identical to the DTC variance this used to compute via an
        # N_train x N_train K_SS (same predictive distribution, see GPML Table 8.1),
        # but only needs the M x M K_MM/A factors already built in fit_c.
        K_diag_true = torch.stack([self.rbf_kernel(xs, xs).sum() for xs in x_list])

        K_MM_inv_k_starM = torch.cholesky_solve(K_NM_test.T, self.L_KMM)  # (M, N_test)
        Q_diag = (K_NM_test * K_MM_inv_k_starM.T).sum(dim=1)  # diag(k_*M K_MM^-1 k_M*)

        A_inv_k_starM = torch.cholesky_solve(K_NM_test.T, self.L_A)  # (M, N_test)
        pp_correction = self.sigma2 * (K_NM_test * A_inv_k_starM.T).sum(dim=1)  # diag(sigma^2 k_*M A^-1 k_M*)

        var = torch.clamp(K_diag_true - Q_diag + pp_correction, min=1e-12)
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
            y = torch.as_tensor(y, dtype=self.x_M.dtype).to(self.x_M.device)
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
                "dtype": str(self.dtype).replace("torch.", ""),
                "jitter": self.jitter,
                "raw_lengthscale": self.raw_lengthscale.detach(),
                "raw_sigma2": self.raw_sigma2.detach(),
                "raw_outputscale": self.raw_outputscale.detach(),
                "c": self.c.detach() if self.c is not None else None,
                "K_NM_train": self.K_NM_train.detach() if self.K_NM_train is not None else None,
                "y_train": self.y_train.detach() if self.y_train is not None else None,
                "L_KMM": self.L_KMM.detach() if self.L_KMM is not None else None,
                "L_A": self.L_A.detach() if self.L_A is not None else None,
                "mean_function": self.mean_function,
                "linear_mean": self.linear_mean.detach() if self.linear_mean is not None else None,
                "mean_l2_reg": self.mean_l2_reg,
                "config": self.config.to_dict() if self.config is not None else None,
            },
            path,
        )
