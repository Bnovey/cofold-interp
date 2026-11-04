"""Tests for igv.protenix_score.

CPU-runnable where possible; GPU-only tests are marked and skipped when
CUDA is unavailable.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.protenix_score import (
    PROTENIX_SCORES,
    _NUM_RESTYPE_CLASSES,
    _UNK_INDEX,
    _build_token_map,
    _chain_cache_key,
    _compute_score_from_logits,
    _oop_msa_inference_forward,
    _prediction_cache_key,
    build_mean_aa_res_type_baseline,
    build_shuffled_res_type_baseline,
    build_unk_res_type_baseline,
    canonical_aa_token_indices,
    coord_fingerprint,
    numerics_arm,
    predict_structure_coords,
)


# ---------------------------------------------------------------------------
# Canonical AA indices
# ---------------------------------------------------------------------------


class TestCanonicalAATokenIndices:
    def test_shape_and_range(self):
        idx = canonical_aa_token_indices()
        assert idx.shape == (20,)
        assert idx.dtype == np.int64
        assert idx.min() == 0
        assert idx.max() == 19

    def test_contiguous(self):
        idx = canonical_aa_token_indices()
        np.testing.assert_array_equal(idx, np.arange(20, dtype=np.int64))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    def _make_restype(self, L=10):
        rt = torch.zeros(1, L, _NUM_RESTYPE_CLASSES)
        for i in range(L):
            rt[0, i, i % 20] = 1.0
        return rt

    def _make_ifd(self, L=10):
        return {
            "asym_id": torch.zeros(L, dtype=torch.long),
            "restype": self._make_restype(L),
        }

    def test_mean_aa_shape(self):
        L = 10
        rt = self._make_restype(L)
        ifd = self._make_ifd(L)
        baseline = build_mean_aa_res_type_baseline(rt, ifd)
        assert baseline.shape == rt.shape
        assert baseline.dtype == torch.float32

    def test_mean_aa_values(self):
        L = 5
        rt = self._make_restype(L)
        ifd = self._make_ifd(L)
        baseline = build_mean_aa_res_type_baseline(rt, ifd)
        for i in range(L):
            row = baseline[0, i]
            assert abs(row[:20].sum().item() - 1.0) < 1e-5
            for j in range(20):
                assert abs(row[j].item() - 0.05) < 1e-5
            assert row[20:].sum().item() == 0.0

    def test_unk_baseline_values(self):
        L = 5
        rt = self._make_restype(L)
        ifd = self._make_ifd(L)
        baseline = build_unk_res_type_baseline(rt, ifd)
        for i in range(L):
            row = baseline[0, i]
            assert row[_UNK_INDEX].item() == 1.0
            assert abs(row.sum().item() - 1.0) < 1e-5

    def test_shuffled_preserves_composition(self):
        L = 20
        rt = self._make_restype(L)
        ifd = self._make_ifd(L)
        baseline = build_shuffled_res_type_baseline(rt, ifd, seed=42)
        orig_argmax = rt[0].argmax(dim=-1).sort().values
        shuf_argmax = baseline[0].argmax(dim=-1).sort().values
        assert torch.equal(orig_argmax, shuf_argmax)

    def test_shuffled_actually_shuffles(self):
        L = 20
        rt = self._make_restype(L)
        ifd = self._make_ifd(L)
        baseline = build_shuffled_res_type_baseline(rt, ifd, seed=42)
        assert not torch.equal(rt, baseline)


# ---------------------------------------------------------------------------
# Token map
# ---------------------------------------------------------------------------


class TestTokenMap:
    def test_basic_mapping(self):
        chains = {"A": "ACDE", "B": "FGH"}
        ifd = {
            "asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1]),
            "residue_index": torch.tensor([0, 1, 2, 3, 0, 1, 2]),
        }
        tm = _build_token_map(ifd, chains)
        assert tm[("A", 0)] == 0
        assert tm[("A", 3)] == 3
        assert tm[("B", 0)] == 4
        assert tm[("B", 2)] == 6

    def test_all_residues_present(self):
        chains = {"X": "ACDEFGHIKLMNPQRSTVWY"}
        L = 20
        ifd = {
            "asym_id": torch.zeros(L, dtype=torch.long),
            "residue_index": torch.arange(L),
        }
        tm = _build_token_map(ifd, chains)
        for i in range(L):
            assert ("X", i) in tm

    def test_one_based_residue_index_is_normalised(self):
        """Protenix emits residue_index 1-based, not 0-based.

        Verified on 1JTG: chain A 1..262, chain B 1..165 against sequences of
        exactly those lengths. Callers index 0-based into the reference
        sequence, so the map must start at 0 for every chain.
        """
        chains = {"A": "ACDE", "B": "FGH"}
        ifd = {
            "asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1]),
            "residue_index": torch.tensor([1, 2, 3, 4, 1, 2, 3]),
        }
        tm = _build_token_map(ifd, chains)
        assert tm[("A", 0)] == 0
        assert tm[("A", 3)] == 3
        assert tm[("B", 0)] == 4
        assert tm[("B", 2)] == 6
        for ch, seq in chains.items():
            for i in range(len(seq)):
                assert (ch, i) in tm, f"missing {(ch, i)}"

    def test_offset_pdb_numbering_is_normalised(self):
        """A chain numbered from an arbitrary offset still maps from 0."""
        chains = {"A": "ACDE"}
        ifd = {
            "asym_id": torch.zeros(4, dtype=torch.long),
            "residue_index": torch.tensor([101, 102, 103, 104]),
        }
        tm = _build_token_map(ifd, chains)
        assert sorted(tm.keys()) == [("A", 0), ("A", 1), ("A", 2), ("A", 3)]


# ---------------------------------------------------------------------------
# Score computation from logits (P4 test)
# ---------------------------------------------------------------------------


class TestComputeScoreFromLogits:
    def _make_logits(self, N=10, n_atoms=20, n_bins=64):
        pde = torch.randn(1, N, N, n_bins)
        pae = torch.randn(1, N, N, n_bins)
        plddt = torch.randn(1, n_atoms, 50)
        return plddt, pae, pde

    def _make_ifd(self, N=10):
        return {
            "asym_id": torch.cat([torch.zeros(N // 2, dtype=torch.long),
                                  torch.ones(N - N // 2, dtype=torch.long)]),
            "has_frame": torch.ones(N, dtype=torch.bool),
        }

    def test_complex_pde_is_scalar(self):
        plddt, pae, pde = self._make_logits()
        ifd = self._make_ifd()
        score = _compute_score_from_logits("complex_pde", plddt, pae, pde, ifd)
        assert score.dim() == 0
        assert torch.isfinite(score)

    def test_complex_plddt_is_scalar(self):
        plddt, pae, pde = self._make_logits()
        ifd = self._make_ifd()
        score = _compute_score_from_logits("complex_plddt", plddt, pae, pde, ifd)
        assert score.dim() == 0

    def test_iptm_is_scalar(self):
        plddt, pae, pde = self._make_logits()
        ifd = self._make_ifd()
        score = _compute_score_from_logits("iptm", plddt, pae, pde, ifd)
        assert score.dim() == 0

    def test_ptm_is_scalar(self):
        plddt, pae, pde = self._make_logits()
        ifd = self._make_ifd()
        score = _compute_score_from_logits("ptm", plddt, pae, pde, ifd)
        assert score.dim() == 0

    def test_pae_interface_mean_is_scalar(self):
        plddt, pae, pde = self._make_logits()
        ifd = self._make_ifd()
        score = _compute_score_from_logits("pae_interface_mean", plddt, pae, pde, ifd)
        assert score.dim() == 0

    def test_unknown_score_raises(self):
        plddt, pae, pde = self._make_logits()
        ifd = self._make_ifd()
        with pytest.raises(ValueError, match="Unknown score"):
            _compute_score_from_logits("nonexistent", plddt, pae, pde, ifd)


# ---------------------------------------------------------------------------
# Gradient reachability through logits (P4 gradient test)
# ---------------------------------------------------------------------------


class TestGradientReachability:
    """Test that gradients flow from each score back through the logits."""

    def _make_logits_with_grad(self, N=10, n_atoms=20):
        pde = torch.randn(1, N, N, 64, requires_grad=True)
        pae = torch.randn(1, N, N, 64, requires_grad=True)
        plddt = torch.randn(1, n_atoms, 50, requires_grad=True)
        return plddt, pae, pde

    def _make_ifd(self, N=10):
        return {
            "asym_id": torch.cat([torch.zeros(N // 2, dtype=torch.long),
                                  torch.ones(N - N // 2, dtype=torch.long)]),
            "has_frame": torch.ones(N, dtype=torch.bool),
        }

    @pytest.mark.parametrize("score_name", sorted(PROTENIX_SCORES.keys()))
    def test_grad_reaches_logits(self, score_name):
        plddt, pae, pde = self._make_logits_with_grad()
        ifd = self._make_ifd()
        score = _compute_score_from_logits(score_name, plddt, pae, pde, ifd)
        score.backward()

        if score_name in ("complex_pde",):
            assert pde.grad is not None
            assert pde.grad.abs().sum() > 0
        elif score_name in ("complex_plddt",):
            assert plddt.grad is not None
            assert plddt.grad.abs().sum() > 0
        elif score_name in ("iptm", "ptm", "pae_interface_mean"):
            assert pae.grad is not None
            assert pae.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# P2: Out-of-place MSA inference_forward
# ---------------------------------------------------------------------------


class TestOOPMSAForward:
    def test_oop_no_inplace_mutation(self):
        """The OOP version must not modify the input tensor in-place."""
        N_msa, N_tok, c_m, c_z = 1, 8, 64, 128
        m = torch.randn(N_msa, N_tok, c_m)
        z = torch.randn(N_tok, N_tok, c_z)
        m_orig = m.clone()

        mock_self = MagicMock()
        mock_self.msa_pair_weighted_averaging = lambda mc, zz: torch.zeros_like(mc)
        mock_self.transition_m = lambda mc: torch.zeros_like(mc)

        result = _oop_msa_inference_forward(mock_self, m, z, chunk_size=4)
        assert torch.equal(m, m_orig), "Input tensor was modified in-place"
        assert result.shape == m.shape

    def test_oop_gradient_flows(self):
        """Gradient must flow through the OOP path."""
        N_msa, N_tok, c_m, c_z = 1, 4, 8, 16
        m = torch.randn(N_msa, N_tok, c_m, requires_grad=True)
        z = torch.randn(N_tok, N_tok, c_z)

        mock_self = MagicMock()
        mock_self.msa_pair_weighted_averaging = lambda mc, zz: mc * 0.1
        mock_self.transition_m = lambda mc: mc * 0.1

        result = _oop_msa_inference_forward(mock_self, m, z, chunk_size=2)
        result.sum().backward()
        assert m.grad is not None
        assert m.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Numerics arm
# ---------------------------------------------------------------------------


class TestNumericsArm:
    def test_keys(self):
        arm = numerics_arm()
        assert arm["model"] == "protenix"
        assert arm["model_name"] == "protenix_base_default_v1.0.0"
        assert "dtype" in arm
        assert "autocast" in arm


# ---------------------------------------------------------------------------
# Score registry
# ---------------------------------------------------------------------------


class TestScoreRegistry:
    def test_expected_scores(self):
        expected = {"complex_pde", "complex_plddt", "pae_interface_mean", "iptm", "ptm"}
        assert set(PROTENIX_SCORES.keys()) == expected

    def test_hard_max_annotation(self):
        assert PROTENIX_SCORES["iptm"] == "hard_max"
        assert PROTENIX_SCORES["ptm"] == "hard_max"
        assert PROTENIX_SCORES["complex_pde"] == "dense"


# ---------------------------------------------------------------------------
# Weight guard
# ---------------------------------------------------------------------------

from igv.protenix_score import (  # noqa: E402
    _ALLOWED_WEIGHT_NAMES,
    _REJECTED_WEIGHT_PREFIX,
)


class TestWeightGuard:
    def test_default_is_apache(self):
        from igv.protenix_score import _DEFAULT_MODEL_NAME

        assert _DEFAULT_MODEL_NAME in _ALLOWED_WEIGHT_NAMES

    def test_v2_prefix_rejected(self):
        assert _REJECTED_WEIGHT_PREFIX == "protenix-v2"

    def test_v2_would_raise_in_load(self):
        with pytest.raises(ValueError, match="proprietary"):
            from igv.protenix_score import load_model

            load_model("cpu", model_name="protenix-v2")

    def test_allowed_names_all_v1_or_v05(self):
        for name in _ALLOWED_WEIGHT_NAMES:
            assert "v1" in name or "v0.5" in name, f"{name} not v0.5 or v1"


# ---------------------------------------------------------------------------
# MSA resolution
# ---------------------------------------------------------------------------

from igv.protenix_score import resolve_msa_paths  # noqa: E402


class TestResolveMSAPaths:
    def test_single_seq_returns_none(self, tmp_path):
        result = resolve_msa_paths({"A": "ACDE"}, str(tmp_path), single_seq=True)
        assert result is None

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_msa_paths({"A": "ACDE"}, "/nonexistent/path")

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .a3m"):
            resolve_msa_paths({"A": "ACDE"}, str(tmp_path))

    def test_flat_layout(self, tmp_path):
        (tmp_path / "A.a3m").write_text(">query\nACDE\n")
        result = resolve_msa_paths({"A": "ACDE"}, str(tmp_path))
        assert result is not None
        assert "A" in result
        assert result["A"].name == "A.a3m"

    def test_boltz_layout(self, tmp_path):
        msa_dir = tmp_path / "processed" / "msa" / "0" / "a3m"
        msa_dir.mkdir(parents=True)
        (msa_dir / "hits.a3m").write_text(">query\nACDE\n")
        result = resolve_msa_paths({"A": "ACDE"}, str(tmp_path))
        assert result is not None
        assert "A" in result


# ---------------------------------------------------------------------------
# GPU-only tests (skipped when CUDA is unavailable)
# ---------------------------------------------------------------------------

_requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


@_requires_gpu
class TestFiniteDifference:
    """Finite-difference check: perturb one restype row by epsilon and compare
    with grad * epsilon."""

    def test_fd_complex_pde(self):
        pytest.skip("Requires full Protenix model on GPU")


@_requires_gpu
class TestScoreParityWithProtenix:
    """Parity test: _compute_score_from_logits must match the original
    Protenix ipTM/pTM/pLDDT functions under no_grad on the same logits,
    within 1e-5."""

    def test_iptm_parity(self):
        try:
            from protenix.model.sample_confidence import (
                calculate_iptm as ptx_iptm,
            )
        except ImportError:
            pytest.skip("protenix not installed")

        N = 20
        pae_logits = torch.randn(1, N, N, 64, device="cuda")
        plddt_logits = torch.randn(1, N, 50, device="cuda")
        pde_logits = torch.randn(1, N, N, 64, device="cuda")
        asym_id = torch.cat([torch.zeros(10, dtype=torch.long),
                             torch.ones(10, dtype=torch.long)]).cuda()
        has_frame = torch.ones(N, dtype=torch.bool, device="cuda")
        ifd = {"asym_id": asym_id, "has_frame": has_frame}

        our = _compute_score_from_logits("iptm", plddt_logits, pae_logits, pde_logits, ifd)

        pae_prob = torch.softmax(pae_logits.squeeze().float(), dim=-1)
        ref = ptx_iptm(
            pae_prob=pae_prob, has_frame=has_frame, asym_id=asym_id,
            min_bin=0.0, max_bin=32.0, no_bins=64,
        )
        assert abs(our.item() - ref.item()) < 1e-5, (
            f"ipTM parity: ours={our.item()}, ref={ref.item()}"
        )

    def test_ptm_parity(self):
        try:
            from protenix.model.sample_confidence import calculate_ptm as ptx_ptm
        except ImportError:
            pytest.skip("protenix not installed")

        N = 20
        pae_logits = torch.randn(1, N, N, 64, device="cuda")
        plddt_logits = torch.randn(1, N, 50, device="cuda")
        pde_logits = torch.randn(1, N, N, 64, device="cuda")
        has_frame = torch.ones(N, dtype=torch.bool, device="cuda")
        asym_id = torch.cat([torch.zeros(10, dtype=torch.long),
                             torch.ones(10, dtype=torch.long)]).cuda()
        ifd = {"asym_id": asym_id, "has_frame": has_frame}

        our = _compute_score_from_logits("ptm", plddt_logits, pae_logits, pde_logits, ifd)

        pae_prob = torch.softmax(pae_logits.squeeze().float(), dim=-1)
        ref = ptx_ptm(
            pae_prob=pae_prob, has_frame=has_frame,
            min_bin=0.0, max_bin=32.0, no_bins=64,
        )
        assert abs(our.item() - ref.item()) < 1e-5, (
            f"pTM parity: ours={our.item()}, ref={ref.item()}"
        )

    def test_plddt_parity(self):
        try:
            from protenix.model.sample_confidence import logits_to_score as ptx_lts
        except ImportError:
            pytest.skip("protenix not installed")

        N, N_atoms = 20, 40
        plddt_logits = torch.randn(1, N_atoms, 50, device="cuda")
        pae_logits = torch.randn(1, N, N, 64, device="cuda")
        pde_logits = torch.randn(1, N, N, 64, device="cuda")
        asym_id = torch.zeros(N, dtype=torch.long, device="cuda")
        has_frame = torch.ones(N, dtype=torch.bool, device="cuda")
        ifd = {"asym_id": asym_id, "has_frame": has_frame}

        our = _compute_score_from_logits(
            "complex_plddt", plddt_logits, pae_logits, pde_logits, ifd,
        )

        ref_scores = ptx_lts(plddt_logits.squeeze(), min_bin=0.0, max_bin=1.0, no_bins=50)
        ref = ref_scores.mean() * 100.0
        assert abs(our.item() - ref.item()) < 1e-5, (
            f"pLDDT parity: ours={our.item()}, ref={ref.item()}"
        )


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------


class TestPredictionCacheKey:
    def test_stable_for_same_inputs(self):
        chains = {"A": "ACDE", "B": "FGH"}
        k1 = _prediction_cache_key(chains, "1JTG")
        k2 = _prediction_cache_key(chains, "1JTG")
        assert k1 == k2

    def test_differs_for_different_dataset(self):
        chains = {"A": "ACDE", "B": "FGH"}
        k1 = _prediction_cache_key(chains, "1JTG")
        k2 = _prediction_cache_key(chains, "2XYZ")
        assert k1 != k2

    def test_differs_for_different_chains(self):
        k1 = _prediction_cache_key({"A": "ACDE", "B": "FGH"}, "1JTG")
        k2 = _prediction_cache_key({"A": "ACDE", "B": "FGI"}, "1JTG")
        assert k1 != k2

    def test_differs_when_sequences_swapped(self):
        k1 = _prediction_cache_key({"A": "ACDE", "B": "FGH"}, "ds")
        k2 = _prediction_cache_key({"A": "FGH", "B": "ACDE"}, "ds")
        assert k1 != k2

    def test_chain_cache_key_stable(self):
        chains = {"A": "ACDE", "B": "FGH"}
        assert _chain_cache_key(chains) == _chain_cache_key(chains)


class TestPredictStructureCoordsCache:
    """Cache behaviour of predict_structure_coords, using a fake model."""

    @staticmethod
    def _make_fake_model(n_atoms=20, call_counter=None):
        model = MagicMock()
        model.configs.model.N_cycle = 1

        def _get_pairformer_output(ifd, N_cycle=1, inplace_safe=False):
            s_in = torch.randn(1, 10, 64)
            s = torch.randn(1, 10, 64)
            z = torch.randn(1, 10, 10, 128)
            return s_in, s, z

        model.get_pairformer_output = _get_pairformer_output
        model.inference_noise_scheduler = MagicMock(
            return_value=torch.linspace(1, 0, 10),
        )

        def _sample_diffusion(**kwargs):
            if call_counter is not None:
                call_counter.append(1)
            return torch.randn(1, 1, n_atoms, 3)

        model.sample_diffusion = _sample_diffusion
        model.diffusion_module = MagicMock()
        return model

    @staticmethod
    def _make_ifd(n_atoms=20):
        return {
            "ref_pos": torch.randn(1, n_atoms, 3),
            "restype": torch.randn(1, 10, 32),
        }

    def test_round_trip_save_load(self, tmp_path):
        counter = []
        model = self._make_fake_model(call_counter=counter)
        ifd = self._make_ifd()

        x1 = predict_structure_coords(
            model, ifd, None, "cpu",
            cache_dir=tmp_path, chains={"A": "ACDE"}, dataset="test",
        )
        assert len(counter) == 1
        cached_path = tmp_path / "predicted_coords"
        assert cached_path.exists()

        x2 = predict_structure_coords(
            model, ifd, None, "cpu",
            cache_dir=tmp_path, chains={"A": "ACDE"}, dataset="test",
        )
        assert len(counter) == 1
        assert torch.equal(x1, x2)

    def test_wrong_atom_count_triggers_repredict(self, tmp_path):
        counter = []
        model = self._make_fake_model(n_atoms=20, call_counter=counter)
        ifd_20 = self._make_ifd(n_atoms=20)

        predict_structure_coords(
            model, ifd_20, None, "cpu",
            cache_dir=tmp_path, chains={"A": "ACDE"}, dataset="test",
        )
        assert len(counter) == 1

        ifd_30 = self._make_ifd(n_atoms=30)
        x2 = predict_structure_coords(
            model, ifd_30, None, "cpu",
            cache_dir=tmp_path, chains={"A": "ACDE"}, dataset="test",
        )
        assert len(counter) == 2

    def test_no_cache_without_args(self, tmp_path):
        counter = []
        model = self._make_fake_model(call_counter=counter)
        ifd = self._make_ifd()

        predict_structure_coords(model, ifd, None, "cpu")
        predict_structure_coords(model, ifd, None, "cpu")
        assert len(counter) == 2
        assert not (tmp_path / "predicted_coords").exists()

    def test_atomic_write_no_partial_file(self, tmp_path):
        """Simulate a crash during torch.save -- no partial file should remain."""
        from unittest.mock import patch

        counter = []
        model = self._make_fake_model(call_counter=counter)
        ifd = self._make_ifd()

        with patch("torch.save", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                predict_structure_coords(
                    model, ifd, None, "cpu",
                    cache_dir=tmp_path, chains={"A": "ACDE"}, dataset="test",
                )

        pred_dir = tmp_path / "predicted_coords"
        if pred_dir.exists():
            remaining = list(pred_dir.rglob("*"))
            pt_files = [f for f in remaining if f.suffix in (".pt", ".tmp")]
            assert len(pt_files) == 0, f"Partial files left: {pt_files}"


# ---------------------------------------------------------------------------
# Coordinate fingerprint
# ---------------------------------------------------------------------------


class TestCoordFingerprint:
    def test_identical_tensors(self):
        t = torch.randn(1, 1, 20, 3)
        fp1 = coord_fingerprint(t)
        fp2 = coord_fingerprint(t.clone())
        assert fp1["sha256"] == fp2["sha256"]
        assert fp1["shape"] == fp2["shape"]
        assert fp1["abs_max"] == fp2["abs_max"]

    def test_different_tensors(self):
        t1 = torch.randn(1, 1, 20, 3)
        t2 = torch.randn(1, 1, 20, 3)
        fp1 = coord_fingerprint(t1)
        fp2 = coord_fingerprint(t2)
        assert fp1["sha256"] != fp2["sha256"]

    def test_keys(self):
        fp = coord_fingerprint(torch.randn(1, 1, 10, 3))
        assert "sha256" in fp
        assert "shape" in fp
        assert "abs_max" in fp


@_requires_gpu
class TestMemoryLadder:
    """Memory measurement at different token counts and recycle depths.
    Skipped in CI; run explicitly on the GPU VM."""

    def test_memory_ladder(self):
        pytest.skip("Run scripts/08_memscale_protenix.py explicitly on GPU")
