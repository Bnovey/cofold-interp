"""Tests for scripts/29_splice_gate.py -- CPU, fake model, no Boltz."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

torch = pytest.importorskip("torch")

from igv.sae import (
    SAEConfig,
    SparseAutoencoder,
    NormStats,
    compute_norm_stats,
    random_sae as make_random_sae,
)


# ---------------------------------------------------------------------------
# Helpers: fake model with pairformer layers
# ---------------------------------------------------------------------------

D_S, D_Z, N_LAYERS = 16, 8, 3


class FakeLayer(torch.nn.Module):
    def __init__(self, idx, d_s, d_z):
        super().__init__()
        self.offset_s = torch.nn.Parameter(
            torch.full((d_s,), float(idx + 1) * 0.1)
        )
        self.offset_z = torch.nn.Parameter(
            torch.full((d_z,), float(idx + 1) * 0.01)
        )

    def forward(self, s, z, mask, pair_mask, pf_chunk,
                use_kernels, use_cuequiv_mul, use_cuequiv_attn):
        return s + self.offset_s, z + self.offset_z


class FakePairformerModule(torch.nn.Module):
    def __init__(self, n_layers, d_s, d_z):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [FakeLayer(i, d_s, d_z) for i in range(n_layers)]
        )

    def forward(self, s, z, mask=None, pair_mask=None, use_kernels=False):
        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, 128, False, False, False)
        return s, z


def _build_fake_model():
    class FM:
        pass
    model = FM()
    model.pairformer_module = FakePairformerModule(N_LAYERS, D_S, D_Z)
    return model


def _run_with_callback(model, s, z, callback, n_layers):
    _recycle_counter = [0]
    mask = torch.ones(s.shape[0], s.shape[1])
    pair_mask = torch.ones(s.shape[0], s.shape[1], s.shape[1])

    def _pairformer_group_fn(s_, z_, mask_, pair_mask_, span):
        start, stop = int(span[0].item()), int(span[1].item())
        for i in range(start, stop):
            s_, z_ = model.pairformer_module.layers[i](
                s_, z_, mask_, pair_mask_, 128, False, False, False,
            )
            if callback is not None:
                cb_result = callback(_recycle_counter[0], i, s_, z_)
                if cb_result is not None:
                    s_, z_ = cb_result
        return s_, z_

    span = torch.tensor([0, n_layers])
    return _pairformer_group_fn(s, z, mask, pair_mask, span)


# ---------------------------------------------------------------------------
# Tests: callback replacement
# ---------------------------------------------------------------------------


class TestCallbackReplacement:

    def test_replacement_changes_output(self):
        model = _build_fake_model()
        s = torch.randn(1, 4, D_S)
        z = torch.randn(1, 4, 4, D_Z)

        s_base, _ = _run_with_callback(model, s.clone(), z.clone(), None, N_LAYERS)

        def replace_layer_1(recycle_idx, layer_idx, s_, z_):
            if layer_idx == 1:
                return torch.zeros_like(s_), z_
            return None

        s_spliced, _ = _run_with_callback(
            model, s.clone(), z.clone(), replace_layer_1, N_LAYERS,
        )
        assert not torch.allclose(s_base, s_spliced)

    def test_none_return_is_noop(self):
        model = _build_fake_model()
        s = torch.randn(1, 4, D_S)
        z = torch.randn(1, 4, 4, D_Z)

        s_base, z_base = _run_with_callback(
            model, s.clone(), z.clone(), None, N_LAYERS,
        )

        def noop(recycle_idx, layer_idx, s_, z_):
            return None

        s_noop, z_noop = _run_with_callback(
            model, s.clone(), z.clone(), noop, N_LAYERS,
        )
        torch.testing.assert_close(s_noop, s_base, atol=1e-6, rtol=0)
        torch.testing.assert_close(z_noop, z_base, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Tests: identity and eps-kept arms reproduce baseline
# ---------------------------------------------------------------------------


class TestIdentityAndEpsKept:

    def test_identity_matches_baseline(self):
        model = _build_fake_model()
        s = torch.randn(1, 4, D_S)
        z = torch.randn(1, 4, 4, D_Z)

        s_base, z_base = _run_with_callback(
            model, s.clone(), z.clone(), None, N_LAYERS,
        )

        def identity(recycle_idx, layer_idx, s_, z_):
            return s_, z_

        s_id, z_id = _run_with_callback(
            model, s.clone(), z.clone(), identity, N_LAYERS,
        )
        torch.testing.assert_close(s_id, s_base, atol=1e-6, rtol=0)
        torch.testing.assert_close(z_id, z_base, atol=1e-6, rtol=0)

    def test_eps_kept_matches_baseline(self):
        """Splicing x_hat_orig + error_orig = x should reproduce baseline."""
        model = _build_fake_model()
        s = torch.randn(1, 4, D_S)
        z = torch.randn(1, 4, 4, D_Z)

        norm_stats = compute_norm_stats(torch.randn(100, D_S))
        cfg = SAEConfig(d_input=D_S, n_latents=32, k=3, mode="topk",
                        matryoshka_sizes=())
        sae = SparseAutoencoder(cfg, norm_stats=norm_stats)
        sae.eval()

        s_base, z_base = _run_with_callback(
            model, s.clone(), z.clone(), None, N_LAYERS,
        )

        target_layer = 1

        def eps_kept(recycle_idx, layer_idx, s_, z_):
            if layer_idx != target_layer:
                return None
            s_squeezed = s_.squeeze(0) if s_.dim() == 3 else s_
            with torch.no_grad():
                out = sae(s_squeezed)
            s_new = out.x_hat_orig + out.error_orig
            if s_.dim() == 3:
                s_new = s_new.unsqueeze(0)
            return s_new, z_

        s_eps, z_eps = _run_with_callback(
            model, s.clone(), z.clone(), eps_kept, N_LAYERS,
        )
        torch.testing.assert_close(s_eps, s_base, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: gate verdict logic
# ---------------------------------------------------------------------------


class TestGateVerdict:

    def test_verdict_basic(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

        # Import directly via exec to avoid module issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "splice_gate",
            str(Path(__file__).resolve().parents[1] / "scripts" / "29_splice_gate.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        passed, reason = mod.gate_verdict(
            delta_splice=0.001,
            typical_effect=0.01,
            identity_pass=True,
            threshold_frac=0.5,
        )
        assert passed
        assert "ratio=" in reason

        passed, reason = mod.gate_verdict(
            delta_splice=0.008,
            typical_effect=0.01,
            identity_pass=True,
            threshold_frac=0.5,
        )
        assert not passed

        passed, reason = mod.gate_verdict(
            delta_splice=0.001,
            typical_effect=0.01,
            identity_pass=False,
            threshold_frac=0.5,
        )
        assert not passed
        assert "identity" in reason

        passed, reason = mod.gate_verdict(
            delta_splice=0.001,
            typical_effect=None,
            identity_pass=True,
        )
        assert not passed
        assert "no scan" in reason


# ---------------------------------------------------------------------------
# Tests: typical effect from scan CSV
# ---------------------------------------------------------------------------


class TestTypicalEffect:

    def test_from_synthetic_csv(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "splice_gate",
            str(Path(__file__).resolve().parents[1] / "scripts" / "29_splice_gate.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        scan_csv = tmp_path / "results" / "FAKE_complex_pde_scan.csv"
        scan_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(scan_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mutation_id", "delta"])
            writer.writerow(["WT", "0.0"])
            for i, d in enumerate([0.01, -0.02, 0.03, -0.04, 0.05]):
                writer.writerow([f"mut{i}", str(d)])

        result = mod.load_typical_effect("FAKE", tmp_path, "complex_pde")
        assert result is not None
        assert abs(result - 0.03) < 1e-8

    def test_missing_csv_returns_none(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "splice_gate",
            str(Path(__file__).resolve().parents[1] / "scripts" / "29_splice_gate.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.load_typical_effect("NONEXIST", tmp_path, "complex_pde")
        assert result is None

    def test_affinity_csv_delta_score(self, tmp_path):
        """Affinity scans use 'delta_score' column."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "splice_gate",
            str(Path(__file__).resolve().parents[1] / "scripts" / "29_splice_gate.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        scan_csv = tmp_path / "results" / "affinity_scan_FAKE.csv"
        scan_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(scan_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tki", "mutation", "delta_score"])
            for i, d in enumerate([0.1, -0.2, 0.15]):
                writer.writerow(["drug", f"mut{i}", str(d)])

        result = mod.load_typical_effect("FAKE", tmp_path, "affinity")
        assert result is not None
        assert abs(result - 0.15) < 1e-8
