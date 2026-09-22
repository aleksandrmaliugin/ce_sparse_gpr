"""Fake DFT "run script" for testing the MC active-learning loop end to end
without a real VASP install.

ActiveLearningController._run_dft (see mc_grand_serial.py) invokes
this script with no arguments, cwd set to a fresh job directory that already
contains the structure to "relax" (poscar_name, default "in.poscar"). This
stub:

  1. reads that POSCAR,
  2. sleeps a random duration to stand in for a real relaxation's wall time,
  3. predicts its energy from the CURRENT slab+ads models (the same
     checkpoints the running MC uses), plus n_co * E_CO_gas (a fixed,
     genuine DFT reference value, not a prediction - see CO_GAS_OUTCAR) so
     the result looks like a real DFT total energy of a CO-covered
     structure rather than just the ads model's own (already CO-gas- and
     slab-referenced) e_ads_total - optionally jittered by a small Gaussian
     to avoid every "DFT" point landing exactly on the model's own mean
     (which would teach a retraining cycle nothing new),
  4. writes the outputs active_learning's _run_dft expects: final_marker
     (default "final.traj", just the unrelaxed input structure - this stub
     does not actually relax anything) and energy_file (default "final.e"),
     plus a minimal OUTCAR-formatted text file for anyone who wants to eyeball
     the job directory the way a real VASP run would look.

This is ONLY for exercising the retraining plumbing (uncertainty trigger ->
DFT call -> dataset append -> ce_gpr_train.run -> hot-swap). The "DFT" energy
is not physically meaningful - it is the model's own prediction, so do not
use this to draw any conclusions about model accuracy.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))  # repo root: ce_sparse_gpr package, ce_gpr_train.py

from ce_sparse_gpr.calculator import CalculatorCESparseGPR  # noqa: E402
import make_database as mdb  # noqa: E402 - repo root, same sys.path entry as above

# Must match mc_grand.example.json's "models" section (paths are resolved
# relative to THIS file, not to the job directory this script runs in).
SLAB_MODEL = SCRIPT_DIR / "../training/models/clean/model_best.pt"
ADS_MODEL = SCRIPT_DIR / "../training/models/ads/model_best.pt"
# Must match mc_grand.test.json's active_learning.co_gas_outcar.
CO_GAS_OUTCAR = SCRIPT_DIR / "../../_datasets/low_cov/CO_gase/OUTCAR"

POSCAR_NAME = "in.poscar"
FINAL_TRAJ_NAME = "final.traj"
FINAL_ENERGY_NAME = "final.e"
OUTCAR_NAME = "OUTCAR"

SLEEP_RANGE_SECONDS = (2.0, 8.0)
NOISE_STD_EV = 0.001  # set to 0.0 for the model's exact, un-jittered prediction

_co_gas_energy_cache: float | None = None


def _co_gas_energy() -> float:
    global _co_gas_energy_cache
    if _co_gas_energy_cache is None:
        _co_gas_energy_cache = mdb.load_co_gas_energy(CO_GAS_OUTCAR)
    return _co_gas_energy_cache


def fake_relax(atoms):
    """No actual relaxation - the stub returns the input structure unchanged,
    only "predicting" its energy. Swap this out for a real VASP/ASE-calculator
    call to turn this stub into a genuine active-learning DFT driver."""
    calculator = CalculatorCESparseGPR(
        file_slab_model=str(SLAB_MODEL),
        file_ads_model=str(ADS_MODEL),
    )
    _, total_energy, _ = calculator(atoms)
    energy = float(total_energy.item())

    # CalculatorCESparseGPR's total_energy is E_slab + E_ads, where E_ads IS
    # the ads model's own target (e_ads_total = E_total(DFT) - E_slab -
    # n_co*E_CO_gas) - it does NOT add the n_co*E_CO_gas term back in, so on
    # its own it is not a plausible raw DFT total energy for a CO-covered
    # structure (a real VASP total energy includes the atoms that make up
    # the adsorbed CO). Without this term, active_learning.run_cycle's own
    # e_ads_total = energy - (slab_energy + n_co*co_energy) recovers a value
    # short by n_co*E_CO_gas - e.g. ~+96.7 eV too high at n_co=8 (8 * -12.09
    # eV), which is exactly the sign and rough size of the nonsense
    # e_ads_total values seen in al_test_ads.db before this fix.
    n_co = atoms.get_chemical_symbols().count("C")
    if n_co > 0:
        energy += n_co * _co_gas_energy()

    if NOISE_STD_EV > 0.0:
        energy += random.gauss(0.0, NOISE_STD_EV)
    return atoms, energy


def write_fake_outcar(path: Path, energy: float, n_atoms: int) -> None:
    """Minimal, not-remotely-complete OUTCAR text - just enough that
    make_database.py's _read_outcar_energy() (or a human skimming the job
    directory) finds a familiar-looking TOTEN line."""
    path.write_text(
        "vasp.6.x.x (synthetic - written by examples/mc/run.py, NOT a real DFT run)\n"
        f"number of ions     NIONS =  {n_atoms}\n"
        "\n"
        "----------------------------------------- Iteration    1(   1)  ---------------------------------------\n"
        f"  free  energy   TOTEN  =      {energy:.8f} eV\n"
        "\n"
        f"  energy  without entropy =      {energy:.8f}  energy(sigma->0) =      {energy:.8f}\n",
        encoding="utf-8",
    )


def main() -> None:
    poscar_path = Path(POSCAR_NAME)
    if not poscar_path.exists():
        raise FileNotFoundError(f"{POSCAR_NAME!r} not found in {Path.cwd()} - expected the MC's structure here.")

    atoms = read(str(poscar_path), format="vasp")

    sleep_s = random.uniform(*SLEEP_RANGE_SECONDS)
    print(f"[run.py] simulating a DFT relaxation for {sleep_s:.1f}s ...", flush=True)
    time.sleep(sleep_s)

    relaxed_atoms, energy = fake_relax(atoms)
    print(f"[run.py] fake DFT energy = {energy:.6f} eV", flush=True)

    write(FINAL_TRAJ_NAME, relaxed_atoms)
    Path(FINAL_ENERGY_NAME).write_text(f"{energy:.8f}\n", encoding="utf-8")
    write_fake_outcar(Path(OUTCAR_NAME), energy, len(relaxed_atoms))

    print("[run.py] done.", flush=True)


if __name__ == "__main__":
    main()
