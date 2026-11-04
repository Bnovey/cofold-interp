"""Sparse autoencoder for Boltz-2 cached activations.

Implements Matryoshka BatchTopK (and plain TopK) with AuxK dead-latent loss,
input normalisation, and a random-SAE baseline constructor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Key names for cached .npz files — isolated here so they're easy to adjust
# ---------------------------------------------------------------------------

# Must match igv.activations.write_shard. Default recycle 1 = final iteration
# at recycling_steps=1, the setting the scans use.
SINGLE_KEY = "s_r{recycle}_l{layer}"     # (n_tokens, 384) fp16
PAIR_KEY = "pair_r{recycle}_l{layer}"    # (n_tokens, 128) fp16
META_KEYS = ("chain_id", "residue_index", "mol_type")
DEFAULT_RECYCLE = 1

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tokens_from_npz(
    path: str | Path,
    layer: int,
    input_mode: Literal["s", "s+pair"] = "s",
    recycle: int = DEFAULT_RECYCLE,
) -> torch.Tensor:
    """Load per-token vectors from one cached .npz file.

    Returns float32 tensor of shape (n_tokens, d).
    """
    import numpy as np

    data = np.load(path, allow_pickle=True)
    s = data[SINGLE_KEY.format(recycle=recycle, layer=layer)].astype(np.float32)
    if input_mode == "s+pair":
        p = data[PAIR_KEY.format(recycle=recycle, layer=layer)].astype(np.float32)
        s = np.concatenate([s, p], axis=1)
    return torch.from_numpy(s)


@dataclass
class TokenDataset:
    """Flat dataset of activation vectors, split by complex."""

    train: torch.Tensor  # (N_train, d)
    val: torch.Tensor    # (N_val, d)
    train_complex_ids: list[str] = field(default_factory=list)
    val_complex_ids: list[str] = field(default_factory=list)


def build_token_dataset(
    cache_dir: str | Path,
    layer: int,
    input_mode: Literal["s", "s+pair"] = "s",
    val_fraction: float = 0.15,
    seed: int = 42,
    recycle: int = DEFAULT_RECYCLE,
) -> TokenDataset:
    """Build a train/val token dataset split BY COMPLEX (never by token).

    All .npz files in ``cache_dir`` are loaded. A held-out fraction of
    complexes (by count) are used for validation.
    """
    cache_dir = Path(cache_dir)
    npz_files = sorted(cache_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {cache_dir}")

    rng = torch.Generator().manual_seed(seed)
    n_val = max(1, int(len(npz_files) * val_fraction))
    perm = torch.randperm(len(npz_files), generator=rng).tolist()
    val_indices = set(perm[:n_val])

    train_tensors, val_tensors = [], []
    train_ids, val_ids = [], []

    for i, p in enumerate(npz_files):
        tokens = load_tokens_from_npz(p, layer=layer, input_mode=input_mode, recycle=recycle)
        cid = p.stem
        if i in val_indices:
            val_tensors.append(tokens)
            val_ids.append(cid)
        else:
            train_tensors.append(tokens)
            train_ids.append(cid)

    return TokenDataset(
        train=torch.cat(train_tensors, dim=0),
        val=torch.cat(val_tensors, dim=0),
        train_complex_ids=train_ids,
        val_complex_ids=val_ids,
    )


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


@dataclass
class NormStats:
    """Per-dimension mean and a single global scale so E||x||² = d."""

    mean: torch.Tensor   # (d,)
    scale: float

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) * self.scale

    def invert(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm / self.scale + self.mean.to(x_norm.device)

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "scale": self.scale}

    @classmethod
    def from_state_dict(cls, sd: dict[str, object]) -> NormStats:
        return cls(mean=sd["mean"], scale=sd["scale"])


def compute_norm_stats(x: torch.Tensor) -> NormStats:
    """Compute normalisation stats from a training set.

    Mean-subtracts per dimension, then scales so E||x||² = d.
    """
    mean = x.mean(dim=0)
    centered = x - mean
    avg_sq_norm = (centered ** 2).sum(dim=1).mean().item()
    d = x.shape[1]
    scale = math.sqrt(d / max(avg_sq_norm, 1e-12))
    return NormStats(mean=mean, scale=scale)


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


@dataclass
class SAEConfig:
    d_input: int = 384
    n_latents: int = 4096
    k: int = 64
    mode: Literal["topk", "batchtopk"] = "batchtopk"
    matryoshka_sizes: tuple[int, ...] = (256, 1024, 4096)
    auxk_coeff: float = 1.0 / 32
    auxk_k: int = 256
    dead_threshold_steps: int = 10000


class SAEOutput:
    """Output of SparseAutoencoder.forward.

    Spaces:
      x_hat     -- reconstruction in *normalised* space (zero-mean, unit-variance).
      error     -- residual in *normalised* space: x_norm - x_hat.
      x_hat_orig -- reconstruction in *original* (denormalised) space.
      error_orig -- residual in *original* space: x - x_hat_orig.
                    Downstream attribution uses error_orig.
    """

    __slots__ = ("x_hat", "f", "error", "auxk_loss", "losses",
                 "x_hat_orig", "error_orig")

    def __init__(
        self,
        x_hat: torch.Tensor,
        f: torch.Tensor,
        error: torch.Tensor,
        auxk_loss: torch.Tensor | None = None,
        losses: dict[str, torch.Tensor] | None = None,
        x_hat_orig: torch.Tensor | None = None,
        error_orig: torch.Tensor | None = None,
    ):
        self.x_hat = x_hat
        self.f = f
        self.error = error
        self.auxk_loss = auxk_loss
        self.losses = losses or {}
        self.x_hat_orig = x_hat_orig
        self.error_orig = error_orig


class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig, norm_stats: NormStats | None = None):
        super().__init__()
        self.cfg = cfg
        self.norm_stats = norm_stats

        self.encoder = nn.Linear(cfg.d_input, cfg.n_latents, bias=True)
        self.decoder = nn.Linear(cfg.n_latents, cfg.d_input, bias=True)

        self._init_weights()

        self.register_buffer(
            "steps_since_fired",
            torch.zeros(cfg.n_latents, dtype=torch.long),
        )
        self._inference_threshold: torch.Tensor | None = None

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            )
            self.encoder.weight.copy_(self.decoder.weight.T)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    # ---- normalisation wrappers ----

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_stats is not None:
            return self.norm_stats.apply(x)
        return x

    def denormalise(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.norm_stats is not None:
            return self.norm_stats.invert(x_norm)
        return x_norm

    # ---- sparsity ----

    def _topk_mask(self, pre_acts: torch.Tensor, k: int) -> torch.Tensor:
        topk_vals = pre_acts.topk(k, dim=-1).values
        threshold = topk_vals[..., -1:]
        return pre_acts >= threshold

    def _apply_topk(self, pre_acts: torch.Tensor) -> torch.Tensor:
        mask = self._topk_mask(pre_acts, self.cfg.k)
        return pre_acts * mask

    def _apply_batchtopk(self, pre_acts: torch.Tensor) -> torch.Tensor:
        B, N = pre_acts.shape
        total_k = self.cfg.k * B
        flat = pre_acts.reshape(-1)
        topk_vals = flat.topk(total_k).values
        threshold = topk_vals[-1]
        mask = (flat >= threshold).reshape(B, N)
        return pre_acts * mask

    # ---- encode / decode ----

    def encode(self, x: torch.Tensor, use_inference_threshold: bool = False) -> torch.Tensor:
        """Encode input to sparse latent activations.

        During inference, if ``use_inference_threshold`` is True and a threshold
        has been estimated, applies per-token thresholding instead of batch-level.
        """
        x_norm = self.normalise(x)
        pre_acts = F.relu(self.encoder(x_norm))

        if use_inference_threshold and self._inference_threshold is not None:
            thresh = self._inference_threshold.to(pre_acts.device)
            return pre_acts * (pre_acts >= thresh)

        if self.cfg.mode == "topk":
            return self._apply_topk(pre_acts)
        return self._apply_batchtopk(pre_acts)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f)

    def forward(
        self, x: torch.Tensor, *, return_aux: bool = True
    ) -> SAEOutput:
        x_norm = self.normalise(x)
        pre_acts = F.relu(self.encoder(x_norm))

        if self.cfg.mode == "topk":
            f = self._apply_topk(pre_acts)
        else:
            f = self._apply_batchtopk(pre_acts)

        x_hat_norm = self.decode(f)
        error = x_norm - x_hat_norm

        active_mask = f > 0
        if self.training:
            fired = active_mask.any(dim=0)
            self.steps_since_fired[fired] = 0
            self.steps_since_fired[~fired] += 1

        losses: dict[str, torch.Tensor] = {}

        if self.cfg.matryoshka_sizes:
            total_recon = torch.zeros(1, device=x.device)
            for prefix_size in self.cfg.matryoshka_sizes:
                f_prefix = f[:, :prefix_size]
                x_hat_prefix = self.decoder.bias + F.linear(
                    f_prefix, self.decoder.weight[:, :prefix_size]
                )
                loss_prefix = (x_norm - x_hat_prefix).pow(2).mean()
                losses[f"recon_{prefix_size}"] = loss_prefix
                total_recon = total_recon + loss_prefix
            losses["recon"] = total_recon
        else:
            losses["recon"] = error.pow(2).mean()

        auxk_loss = None
        if return_aux and self.cfg.auxk_coeff > 0:
            auxk_loss = self._auxk_loss(pre_acts, x_norm, x_hat_norm)
            if auxk_loss is not None:
                losses["auxk"] = auxk_loss

        x_hat_orig = self.denormalise(x_hat_norm)
        error_orig = x - x_hat_orig

        return SAEOutput(
            x_hat=x_hat_norm,
            f=f,
            error=error,
            auxk_loss=auxk_loss,
            losses=losses,
            x_hat_orig=x_hat_orig,
            error_orig=error_orig,
        )

    # ---- AuxK dead-latent loss ----

    def _auxk_loss(
        self, pre_acts: torch.Tensor, x_norm: torch.Tensor, x_hat_norm: torch.Tensor,
    ) -> torch.Tensor | None:
        dead_mask = self.steps_since_fired >= self.cfg.dead_threshold_steps
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return None

        k_aux = min(self.cfg.auxk_k, n_dead)
        dead_pre = pre_acts[:, dead_mask]
        topk_vals, topk_idx = dead_pre.topk(k_aux, dim=-1)

        dead_f = torch.zeros_like(dead_pre)
        dead_f.scatter_(1, topk_idx, topk_vals)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        dead_decoder_weight = self.decoder.weight[:, dead_indices]
        residual = x_norm - x_hat_norm.detach()
        aux_recon = F.linear(dead_f, dead_decoder_weight)
        return (residual - aux_recon).pow(2).mean()

    # ---- decoder column normalisation ----

    @torch.no_grad()
    def normalise_decoder(self):
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def project_decoder_grads(self):
        """Remove the gradient component parallel to each decoder column."""
        if self.decoder.weight.grad is None:
            return
        W = self.decoder.weight
        g = self.decoder.weight.grad
        proj = (g * W).sum(dim=0, keepdim=True) * W
        g.sub_(proj)

    # ---- inference threshold estimation ----

    @torch.no_grad()
    def estimate_inference_threshold(self, data: torch.Tensor, n_samples: int = 8192):
        """Estimate per-latent thresholds from training activations.

        For each latent, the threshold is set to the minimum nonzero activation
        observed in a sample of training data, so that per-token TopK at
        inference approximates the batch-level sparsity pattern.
        """
        idx = torch.randperm(len(data))[:n_samples]
        sample = data[idx]
        f = self.encode(sample)
        active = f > 0
        min_active = torch.where(active, f, torch.full_like(f, float("inf")))
        thresholds = min_active.min(dim=0).values
        self._inference_threshold = thresholds

    # ---- checkpoint ----

    def full_state_dict(self) -> dict[str, object]:
        sd: dict[str, object] = {
            "model": self.state_dict(),
            "cfg": {
                "d_input": self.cfg.d_input,
                "n_latents": self.cfg.n_latents,
                "k": self.cfg.k,
                "mode": self.cfg.mode,
                "matryoshka_sizes": list(self.cfg.matryoshka_sizes),
                "auxk_coeff": self.cfg.auxk_coeff,
                "auxk_k": self.cfg.auxk_k,
                "dead_threshold_steps": self.cfg.dead_threshold_steps,
            },
        }
        if self.norm_stats is not None:
            sd["norm_stats"] = self.norm_stats.state_dict()
        if self._inference_threshold is not None:
            sd["inference_threshold"] = self._inference_threshold
        return sd

    @classmethod
    def from_checkpoint(cls, sd: dict[str, object]) -> SparseAutoencoder:
        cfg_d = sd["cfg"]
        cfg = SAEConfig(
            d_input=cfg_d["d_input"],
            n_latents=cfg_d["n_latents"],
            k=cfg_d["k"],
            mode=cfg_d["mode"],
            matryoshka_sizes=tuple(cfg_d["matryoshka_sizes"]),
            auxk_coeff=cfg_d["auxk_coeff"],
            auxk_k=cfg_d["auxk_k"],
            dead_threshold_steps=cfg_d["dead_threshold_steps"],
        )
        norm_stats = None
        if "norm_stats" in sd:
            norm_stats = NormStats.from_state_dict(sd["norm_stats"])
        sae = cls(cfg, norm_stats=norm_stats)
        sae.load_state_dict(sd["model"])
        if "inference_threshold" in sd:
            sae._inference_threshold = sd["inference_threshold"]
        return sae


# ---------------------------------------------------------------------------
# Random SAE baseline (§5)
# ---------------------------------------------------------------------------


def random_sae(cfg: SAEConfig, norm_stats: NormStats | None = None, seed: int = 0) -> SparseAutoencoder:
    """SAE with a random unit-norm decoder, untrained. For baseline comparisons."""
    gen = torch.Generator().manual_seed(seed)
    sae = SparseAutoencoder(cfg, norm_stats=norm_stats)
    with torch.no_grad():
        sae.decoder.weight.normal_(generator=gen)
        norms = sae.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        sae.decoder.weight.div_(norms)
        sae.encoder.weight.copy_(sae.decoder.weight.T)
        sae.encoder.bias.zero_()
        sae.decoder.bias.zero_()
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def normalised_mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    mse = (x - x_hat).pow(2).sum(dim=-1).mean()
    var = (x - x.mean(dim=0)).pow(2).sum(dim=-1).mean()
    return (mse / var.clamp(min=1e-12)).item()


def fraction_variance_explained(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return 1.0 - normalised_mse(x, x_hat)


def l0(f: torch.Tensor) -> float:
    return (f > 0).float().sum(dim=-1).mean().item()


def dead_fraction(steps_since_fired: torch.Tensor, threshold: int) -> float:
    return (steps_since_fired >= threshold).float().mean().item()


def cross_seed_overlap(
    decoder_a: torch.Tensor, decoder_b: torch.Tensor, threshold: float = 0.7,
) -> float:
    """Fraction of latents in A with max cosine sim to B > threshold.

    Decoder weights are (d_input, n_latents), columns are atoms.
    """
    a = F.normalize(decoder_a, dim=0)
    b = F.normalize(decoder_b, dim=0)
    cos = (a.T @ b).abs()
    max_sim = cos.max(dim=1).values
    return (max_sim > threshold).float().mean().item()
