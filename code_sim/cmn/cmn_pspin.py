"""
Fully-connected Ising p-spin utilities (no external field).

Hamiltonian
-----------
    H(σ) = Σ_{i1<…<ip} J_{i1…ip} σ_{i1}⋯σ_{ip},
    with   J ~ 𝒩(0, p! / (2 N^{p-1})).

All public functions parallel the cmn_sk API so existing
analysis scripts (curate_sigma_list, etc.) stay unchanged.
"""

from __future__ import annotations
import itertools, math
import numpy as np

# --------------------------------------------------------------------------- #
# Couplings
# --------------------------------------------------------------------------- #

def init_J_pspin(N: int, p: int, *, rng=None) -> dict[tuple[int, ...], float]:
    """
    Return a dict  {index-tuple: J}  containing **all**
    ( N choose p ) couplings for the fully connected model.
    """
    if rng is None:
        rng = np.random.default_rng()
    var_J = math.factorial(p) / (2.0 * N**(p - 1))
    σ_J   = math.sqrt(var_J)

    Jdict = {}
    for comb in itertools.combinations(range(N), p):
        Jdict[comb] = rng.normal(0.0, σ_J)
    return Jdict

# --------------------------------------------------------------------------- #
# Local fields and DFEs
# --------------------------------------------------------------------------- #

def compute_lfs(σ: np.ndarray,
                Jdict: dict[tuple[int, ...], float]) -> np.ndarray:
    """
    Local fields h_i = ∂H/∂σ_i  (no external field part).
    """
    N   = len(σ)
    lfs = np.zeros(N, dtype=np.float32)

    for idxs, J in Jdict.items():
        prod = np.prod(σ[list(idxs)])
        for i in idxs:
            lfs[i] += J * prod / σ[i]         # σ_i = ±1 ⇒ divide by σ_i
    return lfs


def compute_dfe(σ: np.ndarray,
                Jdict: dict[tuple[int, ...], float]) -> np.ndarray:
    """
    ΔE_i = −2 σ_i h_i   (Ising single-spin flip cost).
    """
    return -2.0 * σ * compute_lfs(σ, Jdict)


def compute_bdfe(σ, Jdict):
    dfe  = compute_dfe(σ, Jdict)
    mask = dfe > 0
    return dfe[mask], np.where(mask)[0]

# --------------------------------------------------------------------------- #
# Greedy SSWM walk to a local maximum
# --------------------------------------------------------------------------- #

def _sswm_pick_beneficial(σ, Jdict, rng):
    bdfe, idx = compute_bdfe(σ, Jdict)
    if idx.size == 0:
        return None
    return rng.choice(idx, p=bdfe / bdfe.sum())


def relax_pspin(σ0, Jdict, *, rng):
    """
    Greedy walk (SSWM) until no beneficial flips remain.
    Returns a local maximum configuration σ.
    """
    σ = σ0.copy()
    while True:
        i = _sswm_pick_beneficial(σ, Jdict, rng)
        if i is None:
            break
        σ[i] *= -1
    return σ
