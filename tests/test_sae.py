"""Tests for igv.sae — all CPU, tiny dims."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from igv.sae import (
    SAEConfig,
    SAEOutput,
    SparseAutoencoder,
    NormStats,
    TokenDataset,
    build_token_dataset,
    compute_norm_stats,
    cross_seed_overlap,
    dead_fraction,
    fraction_variance_explained,
    l0,
    normalised_mse,
    random_sae,
    SINGLE_KEY,
    PAIR_KEY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 32
N_LATENTS = 64
K = 3


def _cfg(**kw) -> SAEConfig:
    defaults = dict(
        d_input=D,
        n_latents=N_LATENTS,
        k=K,
        mode="topk",
        matryoshka_sizes=(),
        auxk_coeff=1.0 / 32,
        auxk_k=8,
        dead_threshold_steps=5,
    )
    defaults.update(kw)
    return SAEConfig(**defaults)


def _make_sae(**kw):
    cfg = _cfg(**kw)
    return SparseAutoencoder(cfg)


# ---------------------------------------------------------------------------
# 1. Shapes
# ---------------------------------------------------------------------------


class TestShapes:
    def test_encode_shape(self):
        sae = _make_sae()
        x = torch.randn(16, D)
        f = sae.encode(x)
        assert f.shape == (16, N_LATENTS)

    def test_decode_shape(self):
        sae = _make_sae()
        f = torch.randn(16, N_LATENTS)
        x_hat = sae.decode(f)
        assert x_hat.shape == (16, D)

    def test_forward_output_types(self):
        sae = _make_sae()
        x = torch.randn(16, D)
        out = sae(x)
        assert out.x_hat.shape == (16, D)
        assert out.f.shape == (16, N_LATENTS)
        assert out.error.shape == (16, D)

    def test_matryoshka_shapes(self):
        sae = _make_sae(matryoshka_sizes=(16, 32, 64))
        x = torch.randn(8, D)
        out = sae(x)
        assert "recon_16" in out.losses
        assert "recon_32" in out.losses
        assert "recon_64" in out.losses


# ---------------------------------------------------------------------------
# 2. TopK: exactly k active per token
# ---------------------------------------------------------------------------


class TestTopK:
    def test_exactly_k_active_topk(self):
        sae = _make_sae(mode="topk")
        x = torch.randn(32, D)
        f = sae.encode(x)
        active_per_token = (f > 0).sum(dim=1)
        assert (active_per_token == K).all()


# ---------------------------------------------------------------------------
# 3. BatchTopK: total actives = k * B
# ---------------------------------------------------------------------------


class TestBatchTopK:
    def test_total_actives_batchtopk(self):
        sae = _make_sae(mode="batchtopk")
        B = 32
        x = torch.randn(B, D)
        f = sae.encode(x)
        total_active = (f > 0).sum().item()
        assert total_active == K * B

    def test_ties_and_zeros(self):
        """When pre_acts contain ties and zeros, BatchTopK must still
        produce a finite number of active latents without crashing.
        With ties the >= threshold may select more than k*B when many
        pre_acts are exactly equal to the threshold; that is an inherent
        property of BatchTopK. The test verifies the output is well-formed."""
        sae = _make_sae(mode="batchtopk")
        B = 16
        x = torch.zeros(B, D)
        f = sae.encode(x)
        assert f.shape == (B, N_LATENTS)
        assert torch.isfinite(f).all()

        with torch.no_grad():
            sae.encoder.weight.fill_(1.0 / D)
            sae.encoder.bias.fill_(0.5)
        x_tied = torch.ones(B, D)
        f_tied = sae.encode(x_tied)
        assert f_tied.shape == (B, N_LATENTS)
        assert torch.isfinite(f_tied).all()
        assert (f_tied >= 0).all()


# ---------------------------------------------------------------------------
# 4. Decoder unit-norm preserved after steps
# ---------------------------------------------------------------------------


class TestDecoderNorm:
    def test_unit_norm_after_init(self):
        sae = _make_sae()
        norms = sae.decoder.weight.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_unit_norm_after_step(self):
        sae = _make_sae()
        sae.train()
        opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
        x = torch.randn(16, D)
        out = sae(x)
        loss = out.losses.get("recon", out.error.pow(2).mean())
        loss.backward()
        sae.project_decoder_grads()
        opt.step()
        sae.normalise_decoder()
        norms = sae.decoder.weight.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Normalisation round-trips
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_roundtrip(self):
        x = torch.randn(100, D) * 5 + 3
        stats = compute_norm_stats(x)
        x_norm = stats.apply(x)
        x_back = stats.invert(x_norm)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_mean_subtracted(self):
        x = torch.randn(1000, D) + 10.0
        stats = compute_norm_stats(x)
        x_norm = stats.apply(x)
        assert x_norm.mean(dim=0).abs().max().item() < 0.1

    def test_scale_correct(self):
        x = torch.randn(5000, D) * 3 + 2
        stats = compute_norm_stats(x)
        x_norm = stats.apply(x)
        avg_sq = x_norm.pow(2).sum(dim=1).mean().item()
        assert abs(avg_sq - D) / D < 0.05


# ---------------------------------------------------------------------------
# 6. Synthetic sparse dictionary recovery
# ---------------------------------------------------------------------------


class TestDictionaryRecovery:
    def test_recovery(self):
        torch.manual_seed(42)
        d = 32
        n_atoms = 64
        k_true = 3
        n_samples = 8000

        true_dict = torch.randn(d, n_atoms)
        true_dict = true_dict / true_dict.norm(dim=0, keepdim=True)

        codes = torch.zeros(n_samples, n_atoms)
        for i in range(n_samples):
            active = torch.randperm(n_atoms)[:k_true]
            codes[i, active] = torch.randn(k_true).abs() + 0.5
        data = codes @ true_dict.T

        cfg = _cfg(d_input=d, n_latents=n_atoms, k=k_true, mode="topk")
        norm_stats = compute_norm_stats(data)
        sae = SparseAutoencoder(cfg, norm_stats=norm_stats)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

        sae.train()
        for step in range(3000):
            idx = torch.randint(0, n_samples, (256,))
            batch = data[idx]
            out = sae(batch, return_aux=False)
            loss = out.error.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            sae.project_decoder_grads()
            opt.step()
            sae.normalise_decoder()

        sae.eval()
        with torch.no_grad():
            out = sae(data[:1000])
            fve = fraction_variance_explained(sae.normalise(data[:1000]), out.x_hat)
            assert fve > 0.9, f"FVE={fve}"

        learned = sae.decoder.weight.data
        cos = (torch.nn.functional.normalize(true_dict, dim=0).T
               @ torch.nn.functional.normalize(learned, dim=0)).abs()
        max_sim = cos.max(dim=1).values
        recovered = (max_sim > 0.9).float().mean().item()
        assert recovered > 0.5, f"Only recovered {recovered:.2%} of atoms"


# ---------------------------------------------------------------------------
# 7. AuxK revives artificially killed latents
# ---------------------------------------------------------------------------


class TestAuxK:
    def test_auxk_revives_dead(self):
        torch.manual_seed(0)
        cfg = _cfg(
            d_input=D,
            n_latents=N_LATENTS,
            k=K,
            auxk_coeff=1.0 / 32,
            auxk_k=8,
            dead_threshold_steps=5,
        )
        sae = SparseAutoencoder(cfg)
        sae.steps_since_fired[:32] = 100
        initial_dead = (sae.steps_since_fired >= cfg.dead_threshold_steps).sum().item()
        assert initial_dead == 32

        sae.train()
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
        x = torch.randn(256, D)
        for _ in range(200):
            out = sae(x)
            loss = out.losses["recon"]
            if out.auxk_loss is not None:
                loss = loss + cfg.auxk_coeff * out.auxk_loss
            opt.zero_grad()
            loss.backward()
            sae.project_decoder_grads()
            opt.step()
            sae.normalise_decoder()

        final_dead = (sae.steps_since_fired >= cfg.dead_threshold_steps).sum().item()
        assert final_dead < initial_dead, (
            f"Dead latents did not decrease: {initial_dead} -> {final_dead}"
        )

    def test_auxk_gives_dead_latents_nonzero_gradient(self):
        """The AuxK loss must route additional gradient to dead decoder
        columns compared to training without it. Dead latents have small
        but nonzero pre-activations (via encoder bias) so they do not win
        the main topk but AuxK can select them."""
        torch.manual_seed(1)
        dead_indices = list(range(32))

        grads = {}
        for auxk_coeff in [0.0, 1.0 / 32]:
            cfg = _cfg(
                auxk_coeff=auxk_coeff,
                auxk_k=8,
                dead_threshold_steps=5,
            )
            sae = SparseAutoencoder(cfg)
            sae.train()
            sae.steps_since_fired[dead_indices] = 100
            with torch.no_grad():
                sae.encoder.weight[dead_indices] = 0
                sae.encoder.bias[dead_indices] = 0.01

            x = torch.randn(64, D)
            out = sae(x)

            dead_fired = (out.f[:, dead_indices] > 0).any().item()
            assert not dead_fired, "dead latents should not win main topk"

            loss = out.losses["recon"]
            if out.auxk_loss is not None:
                loss = loss + cfg.auxk_coeff * out.auxk_loss
            loss.backward()

            dead_grad = sae.decoder.weight.grad[:, dead_indices].clone()
            grads[auxk_coeff] = dead_grad.norm().item()

        assert grads[0.0] < 1e-8, (
            f"Dead latents should get no gradient without auxk: {grads[0.0]}"
        )
        assert grads[1.0 / 32] > 1e-6, (
            f"Dead latents should get gradient via auxk: {grads[1.0/32]}"
        )


# ---------------------------------------------------------------------------
# 8. Held-out split is by complex
# ---------------------------------------------------------------------------


class TestDataSplit:
    def test_split_by_complex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i in range(20):
                cid = f"complex_{i:03d}"
                n_tokens = np.random.randint(10, 50)
                np.savez(
                    tmpdir / f"{cid}.npz",
                    **{
                        SINGLE_KEY.format(recycle=1, layer=0): np.random.randn(n_tokens, 384).astype(np.float16),
                        PAIR_KEY.format(recycle=1, layer=0): np.random.randn(n_tokens, 128).astype(np.float16),
                        "chain": np.array(["A"] * n_tokens),
                        "resnum": np.arange(n_tokens),
                        "resname": np.array(["ALA"] * n_tokens),
                        "token_type": np.array(["protein"] * n_tokens),
                    },
                )
            ds = build_token_dataset(tmpdir, layer=0, input_mode="s")
            assert len(ds.train_complex_ids) > 0
            assert len(ds.val_complex_ids) > 0
            assert set(ds.train_complex_ids) & set(ds.val_complex_ids) == set()
            assert len(ds.train) + len(ds.val) > 0

    def test_s_plus_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i in range(5):
                n_tokens = 20
                np.savez(
                    tmpdir / f"c{i}.npz",
                    **{
                        SINGLE_KEY.format(recycle=1, layer=1): np.random.randn(n_tokens, 384).astype(np.float16),
                        PAIR_KEY.format(recycle=1, layer=1): np.random.randn(n_tokens, 128).astype(np.float16),
                    },
                )
            ds = build_token_dataset(tmpdir, layer=1, input_mode="s+pair")
            assert ds.train.shape[1] == 512, f"train dim={ds.train.shape[1]}, expected 512"
            assert ds.val.shape[1] == 512, f"val dim={ds.val.shape[1]}, expected 512"
            assert len(ds.train) > 0
            assert len(ds.val) > 0


# ---------------------------------------------------------------------------
# 9. Checkpoint resume reproduces the same state
# ---------------------------------------------------------------------------


class TestSAEOutputOrigSpace:
    def test_x_hat_orig_and_error_orig(self):
        """x_hat_orig is in original space, error_orig = x - x_hat_orig."""
        x = torch.randn(32, D) * 5 + 3
        norm_stats = compute_norm_stats(x)
        sae = SparseAutoencoder(_cfg(), norm_stats=norm_stats)
        sae.eval()
        with torch.no_grad():
            out = sae(x)
        assert out.x_hat_orig is not None
        assert out.error_orig is not None
        assert out.x_hat_orig.shape == x.shape
        assert out.error_orig.shape == x.shape
        torch.testing.assert_close(
            out.error_orig, x - out.x_hat_orig, atol=1e-5, rtol=1e-5,
        )

    def test_x_hat_orig_is_denormalised(self):
        x = torch.randn(32, D) * 5 + 3
        norm_stats = compute_norm_stats(x)
        sae = SparseAutoencoder(_cfg(), norm_stats=norm_stats)
        sae.eval()
        with torch.no_grad():
            out = sae(x)
        x_hat_from_norm = norm_stats.invert(out.x_hat)
        torch.testing.assert_close(out.x_hat_orig, x_hat_from_norm, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 9. Checkpoint resume reproduces the same state
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    def test_roundtrip(self):
        torch.manual_seed(7)
        cfg = _cfg()
        norm_stats = compute_norm_stats(torch.randn(100, D))
        sae = SparseAutoencoder(cfg, norm_stats=norm_stats)
        sae.steps_since_fired[0] = 42
        sae.estimate_inference_threshold(torch.randn(64, D))

        sd = sae.full_state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(sd, f.name)
            loaded_sd = torch.load(f.name, map_location="cpu", weights_only=False)

        sae2 = SparseAutoencoder.from_checkpoint(loaded_sd)
        assert sae2.cfg.d_input == cfg.d_input
        assert sae2.cfg.n_latents == cfg.n_latents
        x = torch.randn(8, D)
        with torch.no_grad():
            f1 = sae.encode(x)
            f2 = sae2.encode(x)
        assert torch.allclose(f1, f2)


# ---------------------------------------------------------------------------
# 10. Random SAE baseline
# ---------------------------------------------------------------------------


class TestRandomSAE:
    def test_decoder_unit_norm(self):
        cfg = _cfg()
        sae = random_sae(cfg, seed=0)
        norms = sae.decoder.weight.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_deterministic(self):
        cfg = _cfg()
        a = random_sae(cfg, seed=0)
        b = random_sae(cfg, seed=0)
        assert torch.allclose(a.decoder.weight, b.decoder.weight)


# ---------------------------------------------------------------------------
# 11. Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_cross_seed_overlap_self(self):
        W = torch.randn(D, N_LATENTS)
        W = W / W.norm(dim=0, keepdim=True)
        assert cross_seed_overlap(W, W) == 1.0

    def test_cross_seed_overlap_random(self):
        Wa = torch.randn(D, N_LATENTS)
        Wb = torch.randn(D, N_LATENTS)
        Wa = Wa / Wa.norm(dim=0, keepdim=True)
        Wb = Wb / Wb.norm(dim=0, keepdim=True)
        assert cross_seed_overlap(Wa, Wb) < 0.5

    def test_normalised_mse_zero(self):
        x = torch.randn(100, D)
        assert normalised_mse(x, x) < 1e-6

    def test_fve_perfect(self):
        x = torch.randn(100, D)
        assert fraction_variance_explained(x, x) > 0.999

    def test_l0_topk(self):
        sae = _make_sae(mode="topk")
        x = torch.randn(32, D)
        f = sae.encode(x)
        assert abs(l0(f) - K) < 0.01
