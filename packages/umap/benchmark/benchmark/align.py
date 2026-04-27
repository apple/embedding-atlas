"""Embedding alignment utilities (Kabsch / Procrustes)."""

import numpy as np


def _kabsch(emb, ref):
    """Optimal rotation + translation (Kabsch). Returns (transformed, error)."""
    import warnings

    valid = np.isfinite(emb).all(axis=1) & np.isfinite(ref).all(axis=1)
    if valid.sum() < 3:
        return emb.astype(np.float32), np.inf

    e_valid = emb[valid]
    r_valid = ref[valid]

    c_emb = e_valid.mean(axis=0)
    c_ref = r_valid.mean(axis=0)
    E = e_valid - c_emb
    R = r_valid - c_ref

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = E.T @ R
        U, S, Vt = np.linalg.svd(H)
    if np.any(~np.isfinite(U)) or np.any(~np.isfinite(Vt)):
        return emb.astype(np.float32), np.inf

    d = np.linalg.det(Vt.T @ U.T)
    sign = np.eye(2)
    if d < 0:
        sign[1, 1] = -1
    rot = Vt.T @ sign @ U.T

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (emb - c_emb) @ rot.T + c_ref

    err = np.sum((result[valid] - r_valid) ** 2)
    return result.astype(np.float32), err


def align_to_reference(emb, ref_emb):
    """Align emb to ref_emb using rotation + translation, allowing a flip.

    Tries the embedding as-is and with x-axis flipped, picks whichever
    gives a lower residual after optimal rotation.
    """
    emb = emb.astype(np.float64)
    ref = ref_emb.astype(np.float64)

    result_normal, err_normal = _kabsch(emb, ref)

    flipped = emb.copy()
    flipped[:, 0] = -flipped[:, 0]
    result_flipped, err_flipped = _kabsch(flipped, ref)

    if err_flipped < err_normal:
        return result_flipped
    return result_normal
