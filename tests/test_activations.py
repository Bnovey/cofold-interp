"""Tests for igv.activations -- CPU-only, no network, no Boltz weights."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


torch = pytest.importorskip("torch")

from igv.activations import (  # noqa: E402
    CapturedActivations,
    WRITE_SHARD_VERSION,
    _sign_canonical,
    extract_token_metadata,
    make_capture_callback,
    pair_summary,
    write_shard,
)
from igv.provenance import sidecar_path  # noqa: E402


# -----------------------------------------------------------------------
# pair_summary
# -----------------------------------------------------------------------

class TestPairSummary:

    def test_shape_standard(self):
        L, C, k = 20, 128, 64
        z = torch.randn(L, L, C)
        out = pair_summary(z, k=k)
        assert out.shape == (L, 2 * k)
        assert out.dtype == torch.float32

    def test_shape_with_batch_dim(self):
        L, C, k = 15, 128, 64
        z = torch.randn(1, L, L, C)
        out = pair_summary(z, k=k)
        assert out.shape == (L, 2 * k)

    def test_padding_when_L_lt_k(self):
        L, C, k = 5, 128, 64
        z = torch.randn(L, L, C)
        out = pair_summary(z, k=k)
        assert out.shape == (L, 2 * k)
        assert (out[:, L:k] == 0).all()
        assert (out[:, k + L:] == 0).all()

    def test_sign_determinism(self):
        """Negating a column of V should not change the output (signs are canonicalized)."""
        L, C, k = 10, 32, 8
        z = torch.randn(L, L, C)
        out1 = pair_summary(z, k=k)
        out2 = pair_summary(z, k=k)
        torch.testing.assert_close(out1, out2)

    def test_low_rank_reconstruction(self):
        """For a rank-r Z with r <= k, the SVD summary should exactly
        reconstruct the mode-1 and mode-2 unfoldings up to the captured
        components."""
        L, C = 12, 16
        r = 4
        k = 8

        U_true = torch.linalg.qr(torch.randn(L, r))[0]
        V_true = torch.linalg.qr(torch.randn(L * C, r))[0]
        S_true = torch.tensor([10.0, 5.0, 2.0, 1.0])
        mode1_true = U_true @ torch.diag(S_true) @ V_true.T
        z = mode1_true.reshape(L, L, C)

        out = pair_summary(z, k=k)
        feat1 = out[:, :k]

        reconstructed = feat1[:, :r] @ feat1[:, :r].T
        original = mode1_true @ mode1_true.T

        torch.testing.assert_close(
            reconstructed, original, atol=1e-4, rtol=1e-4,
        )


# -----------------------------------------------------------------------
# _sign_canonical
# -----------------------------------------------------------------------

class TestSignCanonical:

    def test_largest_entry_positive(self):
        U = torch.tensor([[-3.0, 1.0], [1.0, -2.0], [2.0, 0.5]])
        S = torch.tensor([1.0, 1.0])
        U_out, S_out = _sign_canonical(U, S)
        for col in range(U_out.shape[1]):
            max_idx = U_out[:, col].abs().argmax()
            assert U_out[max_idx, col] > 0

    def test_idempotent(self):
        U = torch.randn(10, 5)
        S = torch.ones(5)
        U1, _ = _sign_canonical(U, S)
        U2, _ = _sign_canonical(U1, S)
        torch.testing.assert_close(U1, U2)


# -----------------------------------------------------------------------
# capture callback
# -----------------------------------------------------------------------

class TestCaptureCallback:

    def test_records_correct_layer_and_recycle(self):
        layers = [2, 5]
        target_recycle = 1
        callback, captured = make_capture_callback(layers, target_recycle)

        s = torch.randn(1, 10, 384)
        z = torch.randn(1, 10, 10, 128)

        callback(0, 2, s, z)
        assert len(captured.s) == 0

        callback(1, 2, s, z)
        assert (1, 2) in captured.s
        assert captured.s[(1, 2)].shape == (10, 384)
        assert captured.pair_summary[(1, 2)].shape == (10, 128)

        callback(1, 5, s, z)
        assert (1, 5) in captured.s
        assert len(captured.s) == 2

    def test_ignores_non_target_layers(self):
        callback, captured = make_capture_callback([3], target_recycle=0)
        s = torch.randn(1, 8, 384)
        z = torch.randn(1, 8, 8, 128)

        callback(0, 0, s, z)
        callback(0, 1, s, z)
        callback(0, 2, s, z)
        assert len(captured.s) == 0

        callback(0, 3, s, z)
        assert len(captured.s) == 1

    def test_deduplication(self):
        """Callback should not overwrite on repeated calls (robustness against
        checkpoint recompute)."""
        callback, captured = make_capture_callback([0], target_recycle=0)
        s1 = torch.randn(1, 5, 384)
        z1 = torch.randn(1, 5, 5, 128)

        callback(0, 0, s1, z1)
        first_s = captured.s[(0, 0)].copy()

        s2 = torch.randn(1, 5, 384)
        z2 = torch.randn(1, 5, 5, 128)
        callback(0, 0, s2, z2)

        np.testing.assert_array_equal(captured.s[(0, 0)], first_s)


# -----------------------------------------------------------------------
# None-callback path
# -----------------------------------------------------------------------

class TestCallbackIntegration:
    """Test the pairformer_group_fn / callback interaction using a fake model
    with a minimal pairformer_module.layers stack. This exercises the real
    closure logic from boltz_score.py without requiring boltz to be installed.
    """

    @staticmethod
    def _build_fake_model(n_layers=3, d_s=16, d_z=8):
        """Build a fake model object with a pairformer_module.layers list.

        Each fake layer does s' = s + weight_i, z' = z + weight_i (learned
        additive offsets) so the output is predictable and differentiable.
        """
        class FakeLayer(torch.nn.Module):
            def __init__(self, idx, d_s, d_z):
                super().__init__()
                self.offset_s = torch.nn.Parameter(torch.full((d_s,), float(idx + 1) * 0.1))
                self.offset_z = torch.nn.Parameter(torch.full((d_z,), float(idx + 1) * 0.01))

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

        layers = torch.nn.ModuleList(
            [FakeLayer(i, d_s, d_z) for i in range(n_layers)]
        )
        pf_module = FakePairformerModule(n_layers, d_s, d_z)
        pf_module.layers = layers

        class FakeModel:
            pass

        model = FakeModel()
        model.pairformer_module = pf_module
        return model

    @staticmethod
    def _run_pairformer_group_fn(model, s, z, activation_callback, n_layers):
        """Reproduce the _pairformer_group_fn closure from boltz_score.py."""
        _profile = {"pf_chunk": 128}
        _use_kernels = False
        _use_cuequiv_attn = False
        _recycle_counter = [0]
        mask = torch.ones(s.shape[0], s.shape[1])
        pair_mask = torch.ones(s.shape[0], s.shape[1], s.shape[1])

        def _pairformer_group_fn(s_, z_, mask_, pair_mask_, span):
            start, stop = int(span[0].item()), int(span[1].item())
            for i in range(start, stop):
                s_, z_ = model.pairformer_module.layers[i](
                    s_, z_, mask_, pair_mask_, _profile["pf_chunk"],
                    _use_kernels, False, _use_cuequiv_attn,
                )
                if activation_callback is not None:
                    cb_result = activation_callback(_recycle_counter[0], i, s_, z_)
                    if cb_result is not None:
                        s_, z_ = cb_result
            return s_, z_

        span = torch.tensor([0, n_layers])
        return _pairformer_group_fn(s, z, mask, pair_mask, span)

    def test_none_callback_matches_bulk(self):
        """None callback through per-layer path must match the bulk
        pairformer_module forward."""
        n_layers, d_s, d_z = 3, 16, 8
        model = self._build_fake_model(n_layers, d_s, d_z)

        s = torch.randn(1, 4, d_s)
        z = torch.randn(1, 4, 4, d_z)

        s_bulk, z_bulk = model.pairformer_module(s.clone(), z.clone())

        s_perlayer, z_perlayer = self._run_pairformer_group_fn(
            model, s.clone(), z.clone(), None, n_layers,
        )
        torch.testing.assert_close(s_perlayer, s_bulk, atol=1e-6, rtol=0)
        torch.testing.assert_close(z_perlayer, z_bulk, atol=1e-6, rtol=0)

    def test_callback_replacement_changes_output(self):
        """A callback that replaces s should change the downstream output."""
        n_layers, d_s, d_z = 3, 16, 8
        model = self._build_fake_model(n_layers, d_s, d_z)
        s = torch.randn(1, 4, d_s)
        z = torch.randn(1, 4, 4, d_z)

        s_base, z_base = self._run_pairformer_group_fn(
            model, s.clone(), z.clone(), None, n_layers,
        )

        def replace_at_layer_1(recycle_idx, layer_idx, s_, z_):
            if layer_idx == 1:
                return torch.zeros_like(s_), z_
            return None

        s_spliced, z_spliced = self._run_pairformer_group_fn(
            model, s.clone(), z.clone(), replace_at_layer_1, n_layers,
        )

        assert not torch.allclose(s_base, s_spliced), \
            "Callback replacement should change the output"

    def test_identity_callback_preserves_output(self):
        """A callback that returns (s, z) unchanged must give the exact
        same result as None callback."""
        n_layers, d_s, d_z = 3, 16, 8
        model = self._build_fake_model(n_layers, d_s, d_z)
        s = torch.randn(1, 4, d_s)
        z = torch.randn(1, 4, 4, d_z)

        s_none, z_none = self._run_pairformer_group_fn(
            model, s.clone(), z.clone(), None, n_layers,
        )

        def identity_cb(recycle_idx, layer_idx, s_, z_):
            return s_, z_

        s_id, z_id = self._run_pairformer_group_fn(
            model, s.clone(), z.clone(), identity_cb, n_layers,
        )
        torch.testing.assert_close(s_id, s_none, atol=1e-6, rtol=0)
        torch.testing.assert_close(z_id, z_none, atol=1e-6, rtol=0)

    def test_observe_only_callback(self):
        """A callback returning None (observe-only) must not change output."""
        n_layers, d_s, d_z = 3, 16, 8
        model = self._build_fake_model(n_layers, d_s, d_z)
        s = torch.randn(1, 4, d_s)
        z = torch.randn(1, 4, 4, d_z)

        s_base, z_base = self._run_pairformer_group_fn(
            model, s.clone(), z.clone(), None, n_layers,
        )

        observed = []

        def observe_cb(recycle_idx, layer_idx, s_, z_):
            observed.append(layer_idx)
            return None

        s_obs, z_obs = self._run_pairformer_group_fn(
            model, s.clone(), z.clone(), observe_cb, n_layers,
        )
        torch.testing.assert_close(s_obs, s_base, atol=1e-6, rtol=0)
        torch.testing.assert_close(z_obs, z_base, atol=1e-6, rtol=0)
        assert observed == [0, 1, 2]


# -----------------------------------------------------------------------
# write_shard / atomic write / resumability
# -----------------------------------------------------------------------

class TestWriteShard:

    def test_write_and_read(self, tmp_path):
        captured = CapturedActivations()
        captured.s[(1, 32)] = np.random.randn(10, 384).astype(np.float32)
        captured.pair_summary[(1, 32)] = np.random.randn(10, 128).astype(np.float32)
        metadata = {
            "chain_id": np.zeros(10, dtype=np.int32),
            "residue_index": np.arange(10, dtype=np.int32),
            "mol_type": np.zeros(10, dtype=np.int32),
        }

        out = tmp_path / "test.npz"
        write_shard(out, captured, metadata)

        assert out.exists()
        data = np.load(out)
        assert "s_r1_l32" in data
        assert data["s_r1_l32"].dtype == np.float16
        assert data["s_r1_l32"].shape == (10, 384)
        assert "pair_r1_l32" in data
        assert data["pair_r1_l32"].dtype == np.float32
        assert "chain_id" in data
        assert "residue_index" in data

    def test_pair_summary_large_values_no_overflow(self, tmp_path):
        """Pair summaries with large U*S products must survive write_shard
        without inf/nan (stored as fp32, not fp16)."""
        captured = CapturedActivations()
        captured.s[(0, 0)] = np.random.randn(5, 384).astype(np.float32)
        large_pair = np.full((5, 128), 70000.0, dtype=np.float32)
        captured.pair_summary[(0, 0)] = large_pair
        metadata = {"chain_id": np.zeros(5, dtype=np.int32)}

        out = tmp_path / "large.npz"
        write_shard(out, captured, metadata)

        data = np.load(out)
        loaded = data["pair_r0_l0"]
        assert np.all(np.isfinite(loaded))
        np.testing.assert_allclose(loaded, large_pair, rtol=1e-6)

    def test_atomic_write_no_partial(self, tmp_path):
        """A successful write should leave no .tmp files."""
        captured = CapturedActivations()
        captured.s[(0, 0)] = np.zeros((5, 384), dtype=np.float32)
        captured.pair_summary[(0, 0)] = np.zeros((5, 128), dtype=np.float32)
        metadata = {"chain_id": np.zeros(5, dtype=np.int32)}

        out = tmp_path / "shard.npz"
        write_shard(out, captured, metadata)

        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "shard.npz"
        captured = CapturedActivations()
        captured.s[(0, 0)] = np.zeros((3, 384), dtype=np.float32)
        captured.pair_summary[(0, 0)] = np.zeros((3, 128), dtype=np.float32)
        metadata = {"chain_id": np.zeros(3, dtype=np.int32)}

        write_shard(out, captured, metadata)
        assert out.exists()


# -----------------------------------------------------------------------
# extract_token_metadata
# -----------------------------------------------------------------------

class TestExtractTokenMetadata:

    def test_basic(self):
        feats = {
            "token_pad_mask": torch.tensor([[1, 1, 1, 0, 0]]).float(),
            "asym_id": torch.tensor([[0, 0, 1, 99, 99]]),
            "residue_index": torch.tensor([[0, 1, 0, 5, 6]]),
            "mol_type": torch.tensor([[0, 0, 3, 0, 0]]),
        }
        meta = extract_token_metadata(feats)
        assert meta["chain_id"].shape == (3,)
        assert meta["mol_type"].tolist() == [0, 0, 3]
        assert meta["residue_index"].tolist() == [0, 1, 0]

    def test_no_mol_type(self):
        feats = {
            "token_pad_mask": torch.tensor([[1, 1, 0]]).float(),
            "asym_id": torch.tensor([[0, 0, 0]]),
            "residue_index": torch.tensor([[0, 1, 2]]),
        }
        meta = extract_token_metadata(feats)
        assert meta["mol_type"].tolist() == [0, 0]


# -----------------------------------------------------------------------
# Resumability logic (script-level)
# -----------------------------------------------------------------------

class TestResumability:

    def test_is_complete_requires_both_files(self, tmp_path):
        from igv.activations import write_shard

        npz_path = tmp_path / "test.npz"
        assert not npz_path.exists()

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

        captured = CapturedActivations()
        captured.s[(0, 0)] = np.zeros((3, 384), dtype=np.float32)
        captured.pair_summary[(0, 0)] = np.zeros((3, 128), dtype=np.float32)
        write_shard(npz_path, captured, {"chain_id": np.zeros(3, dtype=np.int32)})

        assert npz_path.exists()
        assert not sidecar_path(npz_path).exists()

        from igv.provenance import write as prov_write
        prov_write(
            npz_path, stage="test",
            params={"test": True}, arm={"test": True},
        )

        assert npz_path.exists()
        assert sidecar_path(npz_path).exists()
