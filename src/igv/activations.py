"""Activation capture for Boltz-2 trunk representations.

Records single (s) and pair (z) representations at chosen Pairformer layer
indices during a no-grad forward through the same code path the scans use.
The pair representation is compressed via N-mode SVD (PairSAE recipe) to
avoid storing the full L x L x 128 tensor.

The SVD pair-summary recipe is taken from the PairSAE abstract (Migliorini
et al., arXiv 2606.27440). Details beyond the abstract are unverified:
the mode-1/mode-2 unfolding choice and the sign-determinism convention are
our best guess at what PairSAE does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class CapturedActivations:
    """Activations captured at specific (recycle, layer) indices."""

    s: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    pair_summary: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


def _sign_canonical(U, S):
    """Make SVD sign-deterministic: for each column of U, ensure the
    entry with the largest absolute value is positive."""
    import torch

    max_abs_idx = U.abs().argmax(dim=0)
    signs = torch.sign(U[max_abs_idx, torch.arange(U.shape[1], device=U.device)])
    signs[signs == 0] = 1.0
    return U * signs.unsqueeze(0), S


def pair_summary(z, k=64):
    """PairSAE-style N-mode SVD summary of z.

    Parameters
    ----------
    z : Tensor, shape (L, L, C) or (1, L, L, C)
        Pair representation. Batch dim is squeezed if present.
    k : int
        Number of singular vectors per mode.

    Returns
    -------
    Tensor, shape (L, 2*k), dtype fp32
        Concatenation of mode-1 and mode-2 top-k left singular vectors
        scaled by singular values. Zero-padded when L < k.

    Sign ambiguity is resolved by making the largest-magnitude entry
    of each singular vector positive.

    Computed in fp32 on device. Never stores raw z.
    """
    import torch

    if z.dim() == 4:
        z = z.squeeze(0)
    L, _, C = z.shape
    z = z.float()

    actual_k = min(k, L)

    # Mode-1: unfold along first spatial axis -> (L, L*C)
    mode1 = z.reshape(L, L * C)
    U1, S1, _ = torch.linalg.svd(mode1, full_matrices=False)
    U1 = U1[:, :actual_k]
    S1 = S1[:actual_k]
    U1, S1 = _sign_canonical(U1, S1)
    feat1 = U1 * S1.unsqueeze(0)

    # Mode-2: unfold along second spatial axis (transpose i,j) -> (L, L*C)
    mode2 = z.permute(1, 0, 2).reshape(L, L * C)
    U2, S2, _ = torch.linalg.svd(mode2, full_matrices=False)
    U2 = U2[:, :actual_k]
    S2 = S2[:actual_k]
    U2, S2 = _sign_canonical(U2, S2)
    feat2 = U2 * S2.unsqueeze(0)

    if actual_k < k:
        pad = k - actual_k
        feat1 = torch.nn.functional.pad(feat1, (0, pad))
        feat2 = torch.nn.functional.pad(feat2, (0, pad))

    return torch.cat([feat1, feat2], dim=1)


def make_capture_callback(
    layers: list[int],
    target_recycle: int,
) -> tuple[Callable, CapturedActivations]:
    """Create a callback for use with confidence_forward's activation_callback.

    Parameters
    ----------
    layers : list[int]
        Pairformer layer indices to capture (0-based).
    target_recycle : int
        Recycle iteration to capture at (0-based; the final iteration of a
        recycling_steps=1 run is index 1).

    Returns
    -------
    callback : callable
        Signature: callback(recycle_idx, layer_idx, s, z) -> None.
        Designed to be passed as ``activation_callback`` to confidence_forward.
    captured : CapturedActivations
        The container that will be populated during the forward pass.
    """
    import torch

    captured = CapturedActivations()
    layer_set = set(layers)
    seen: set[tuple[int, int]] = set()

    def callback(recycle_idx: int, layer_idx: int, s, z):
        if recycle_idx != target_recycle:
            return
        if layer_idx not in layer_set:
            return
        key = (recycle_idx, layer_idx)
        if key in seen:
            return
        seen.add(key)
        with torch.no_grad():
            captured.s[key] = s.detach().squeeze(0).float().cpu().numpy()
            captured.pair_summary[key] = (
                pair_summary(z.detach()).cpu().numpy()
            )

    return callback, captured


def extract_token_metadata(feats) -> dict[str, np.ndarray]:
    """Extract per-token metadata arrays from feats dict.

    Returns arrays all of length L (non-padding tokens):
      - chain_id: int array (asym_id)
      - residue_index: int array
      - mol_type: int array (0=protein, 3=nonpolymer/ligand, etc.)

    is_interface is not included: computing it reliably requires
    inter-chain distance calculations on predicted coordinates, and those
    may be all-zero for sequence-only inputs.
    """
    pad = feats["token_pad_mask"][0].bool().cpu()

    meta = {}
    meta["chain_id"] = feats["asym_id"][0].cpu()[pad].numpy().astype(np.int32)
    meta["residue_index"] = feats["residue_index"][0].cpu()[pad].numpy().astype(np.int32)

    if "mol_type" in feats:
        meta["mol_type"] = feats["mol_type"][0].cpu()[pad].numpy().astype(np.int32)
    else:
        meta["mol_type"] = np.zeros(int(pad.sum()), dtype=np.int32)

    return meta


WRITE_SHARD_VERSION = 2


def write_shard(
    path,
    captured: CapturedActivations,
    metadata: dict[str, np.ndarray],
):
    """Write a single complex's activations to an npz file atomically.

    Writes to a temporary file then renames, so a partial write never
    produces a valid-looking shard.

    The npz contains per captured (recycle, layer):
      - s_r{r}_l{l}: shape (L, 384), float16
      - pair_r{r}_l{l}: shape (L, 2*k), float32 (v2: U*S products can
        exceed the fp16 representable range; fp32 is used since the pair
        summary is small -- L x 128 -- and the overhead is negligible)
    Plus metadata arrays (chain_id, residue_index, mol_type).
    """
    import logging
    import os
    import tempfile
    from pathlib import Path

    _log = logging.getLogger(__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for (r, l), s_arr in captured.s.items():
        s_fp16 = s_arr.astype(np.float16)
        if not np.all(np.isfinite(s_fp16)):
            raise ValueError(
                f"s_r{r}_l{l} contains non-finite values after fp16 cast"
            )
        max_abs = np.abs(s_arr).max()
        if max_abs > 6e4:
            _log.warning(
                "s_r%d_l%d: |s|_max = %.1f > 6e4; fp16 precision is degraded",
                r, l, max_abs,
            )
        arrays[f"s_r{r}_l{l}"] = s_fp16
    for (r, l), p_arr in captured.pair_summary.items():
        arrays[f"pair_r{r}_l{l}"] = p_arr.astype(np.float32)
    for name, arr in metadata.items():
        arrays[name] = arr
    arrays["_shard_version"] = np.array(WRITE_SHARD_VERSION, dtype=np.int32)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    try:
        os.close(fd)
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
