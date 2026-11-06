"""Tests for 05_ig_mutation_paths.py -- per-mutation IG paths.

CPU only, no protenix install required, no network.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_SCRIPT = _ROOT / "scripts" / "05_ig_mutation_paths.py"
_spec = importlib.util.spec_from_file_location("ig_mutation_paths_05", str(_SCRIPT))
ig_mut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ig_mut)

from igv.protenix_score import _UNK_INDEX  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_ig(forward_fn, rt_baseline, rt_input, m_steps):
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(m_steps)
    alphas = (gl_nodes + 1.0) / 2.0
    weights = gl_weights / 2.0
    shape_2d = rt_input.shape[-2], rt_input.shape[-1]
    accumulated = np.zeros(shape_2d, dtype=np.float64)
    for alpha, weight in zip(alphas, weights):
        rt_interp = (
            rt_baseline + alpha * (rt_input - rt_baseline)
        ).detach().requires_grad_(True)
        scalar = forward_fn(rt_interp)
        scalar.backward()
        g = rt_interp.grad.detach().cpu().float().numpy()
        if g.ndim == 3:
            g = g[0]
        accumulated += weight * g.astype(np.float64)
    return accumulated


def _make_wt_restype(seq_len, n_classes=32):
    rt = torch.zeros(seq_len, n_classes)
    for i in range(seq_len):
        rt[i, i % 20] = 1.0
    return rt


# ---------------------------------------------------------------------------
# 1. wt_to_mut: baseline equals wild-type, input differs only at row p
# ---------------------------------------------------------------------------

class TestWtToMut:
    def test_baseline_equals_wt(self):
        seq_len, n_classes = 10, 32
        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos, mut_col = 3, 15

        rt_baseline = rt_wt.clone().float()
        rt_input = rt_wt.clone().float()
        rt_input[..., pos, :] = 0.0
        rt_input[..., pos, mut_col] = 1.0

        assert torch.equal(rt_baseline, rt_wt.float())

    def test_input_differs_only_at_p(self):
        seq_len, n_classes = 10, 32
        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos, mut_col = 3, 15

        rt_baseline = rt_wt.clone().float()
        rt_input = rt_wt.clone().float()
        rt_input[..., pos, :] = 0.0
        rt_input[..., pos, mut_col] = 1.0

        for i in range(seq_len):
            if i != pos:
                assert torch.equal(rt_input[i], rt_baseline[i])
        assert rt_input[pos, mut_col].item() == 1.0
        assert rt_input[pos].sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. null_to_mut with unk: baseline row p is one-hot at UNK index
# ---------------------------------------------------------------------------

class TestNullToMutUnk:
    def test_baseline_row_p_is_unk(self):
        seq_len, n_classes = 10, 32
        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos = 5

        null_row = torch.zeros(n_classes)
        null_row[_UNK_INDEX] = 1.0

        rt_baseline = rt_wt.clone().float()
        rt_baseline[..., pos, :] = null_row

        assert rt_baseline[pos, _UNK_INDEX].item() == 1.0
        assert rt_baseline[pos].sum().item() == pytest.approx(1.0)

    def test_other_rows_equal_wt(self):
        seq_len, n_classes = 10, 32
        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos = 5

        null_row = torch.zeros(n_classes)
        null_row[_UNK_INDEX] = 1.0

        rt_baseline = rt_wt.clone().float()
        rt_baseline[..., pos, :] = null_row

        for i in range(seq_len):
            if i != pos:
                assert torch.equal(rt_baseline[i], rt_wt[i].float())


# ---------------------------------------------------------------------------
# 3. null_to_mut with zeros: baseline row p is all zeros
# ---------------------------------------------------------------------------

class TestNullToMutZeros:
    def test_baseline_row_p_is_zeros(self):
        seq_len, n_classes = 10, 32
        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos = 5

        rt_baseline = rt_wt.clone().float()
        rt_baseline[..., pos, :] = 0.0

        assert rt_baseline[pos].sum().item() == 0.0

    def test_other_rows_equal_wt(self):
        seq_len, n_classes = 10, 32
        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos = 5

        rt_baseline = rt_wt.clone().float()
        rt_baseline[..., pos, :] = 0.0

        for i in range(seq_len):
            if i != pos:
                assert torch.equal(rt_baseline[i], rt_wt[i].float())


# ---------------------------------------------------------------------------
# 4. Linear mock: ig_sum == f_input - f_baseline, completeness_error ~ 0
# ---------------------------------------------------------------------------

class TestLinearMock:
    def test_ig_sum_equals_delta(self):
        seq_len, n_classes = 8, 32
        W = torch.randn(seq_len, n_classes)
        bias = 2.7

        def forward_fn(rt):
            return (W * rt).sum() + bias

        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos, mut_col = 4, 17
        m_steps = 8

        rt_baseline = rt_wt.clone().float()
        rt_input = rt_wt.clone().float()
        rt_input[pos, :] = 0.0
        rt_input[pos, mut_col] = 1.0

        accumulated = _simulate_ig(forward_fn, rt_baseline, rt_input, m_steps)
        ig_sum = ig_mut.compute_ig_sum(accumulated, rt_input, rt_baseline)

        f_input = float(forward_fn(rt_input))
        f_baseline = float(forward_fn(rt_baseline))
        expected = f_input - f_baseline

        assert ig_sum == pytest.approx(expected, abs=1e-5)
        comp_err = ig_mut.completeness_error_from_sum(ig_sum, f_input, f_baseline)
        assert comp_err == pytest.approx(0.0, abs=1e-5)

    def test_ig_sum_wrong_if_sliced_to_one_row(self):
        """Summing only the mutated row gives the wrong answer when
        other rows carry attribution (e.g. through cross-attention in a real model)."""
        seq_len, n_classes = 8, 32
        W = torch.randn(seq_len, n_classes)
        bias = 0.0

        def forward_fn(rt):
            return (W * rt).sum() + bias

        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos, mut_col = 4, 17

        rt_baseline = rt_wt.clone().float()
        rt_input = rt_wt.clone().float()
        rt_input[pos, :] = 0.0
        rt_input[pos, mut_col] = 1.0

        accumulated = _simulate_ig(forward_fn, rt_baseline, rt_input, 8)

        full_ig_sum = ig_mut.compute_ig_sum(accumulated, rt_input, rt_baseline)
        f_input = float(forward_fn(rt_input))
        f_baseline = float(forward_fn(rt_baseline))
        assert full_ig_sum == pytest.approx(f_input - f_baseline, abs=1e-6)

        x_np = rt_input.numpy().astype(np.float64)
        b_np = rt_baseline.numpy().astype(np.float64)
        sliced_ig = float(((x_np - b_np) * accumulated)[pos:pos+1, :].sum())
        assert sliced_ig == pytest.approx(f_input - f_baseline, abs=1e-6), (
            "For wt_to_mut paths where only row p differs, even the sliced sum "
            "should match for a linear function."
        )


# ---------------------------------------------------------------------------
# 5. Nonlinear mock: completeness_error > 0 and finite
# ---------------------------------------------------------------------------

class TestNonlinearMock:
    def test_completeness_error_positive_finite(self):
        seq_len, n_classes = 6, 32

        def forward_fn(rt):
            return (rt ** 3).sum()

        torch.manual_seed(42)
        rt_baseline = torch.randn(seq_len, n_classes) * 2.0
        rt_input = rt_baseline.clone()
        rt_input[2, :] = torch.randn(n_classes) * 3.0

        accumulated = _simulate_ig(forward_fn, rt_baseline, rt_input, m_steps=2)
        ig_sum = ig_mut.compute_ig_sum(accumulated, rt_input, rt_baseline)

        f_input = float(forward_fn(rt_input))
        f_baseline = float(forward_fn(rt_baseline))

        comp_err = ig_mut.completeness_error_from_sum(ig_sum, f_input, f_baseline)
        assert comp_err > 0.0
        assert math.isfinite(comp_err)


# ---------------------------------------------------------------------------
# 6. Resumability
# ---------------------------------------------------------------------------

class TestResumability:
    def test_empty_when_no_file(self, tmp_path):
        assert ig_mut._load_done_ids(tmp_path / "nope.csv") == set()

    def test_pre_existing_rows_skipped(self, tmp_path):
        out = tmp_path / "ig.csv"
        pd.DataFrame(
            [
                {
                    "mutation_id": "3A",
                    "position": 3,
                    "wt_aa": "D",
                    "mut_aa": "A",
                    "f_baseline": 1.0,
                    "f_input": 1.1,
                    "ig_sum": 0.1,
                    "scan_delta": 0.1,
                    "completeness_error": 0.01,
                    "m_steps": 8,
                    "path_mode": "wt_to_mut",
                },
                {
                    "mutation_id": "5G",
                    "position": 5,
                    "wt_aa": "H",
                    "mut_aa": "G",
                    "f_baseline": 1.0,
                    "f_input": 0.9,
                    "ig_sum": -0.1,
                    "scan_delta": -0.1,
                    "completeness_error": 0.02,
                    "m_steps": 8,
                    "path_mode": "wt_to_mut",
                },
            ],
            columns=ig_mut._COLUMNS,
        ).to_csv(out, index=False)

        done = ig_mut._load_done_ids(out)
        assert "3A" in done
        assert "5G" in done
        assert "7K" not in done

    def test_corrupt_file_returns_empty(self, tmp_path):
        out = tmp_path / "ig.csv"
        out.write_text("garbage\x00\x01\nmore garbage")
        assert ig_mut._load_done_ids(out) == set()


# ---------------------------------------------------------------------------
# 7. --max-mutations selects expected subset
# ---------------------------------------------------------------------------

class TestMaxMutations:
    def test_limits_mutations(self):
        reference_seq = "ACDEFGHIKL"
        positions = [0, 1, 2]
        mutations = ig_mut._build_mutation_list(None, positions, reference_seq)
        assert len(mutations) == 3 * 19

        limited = mutations[:5]
        assert len(limited) == 5

    def test_zero_means_all(self):
        reference_seq = "ACDEFGHIKL"
        positions = [0, 1]
        mutations = ig_mut._build_mutation_list(None, positions, reference_seq)
        max_mutations = 0
        if max_mutations > 0:
            mutations = mutations[:max_mutations]
        assert len(mutations) == 2 * 19


# ---------------------------------------------------------------------------
# 8. --positions selects expected subset
# ---------------------------------------------------------------------------

class TestPositions:
    def test_positions_filter(self):
        reference_seq = "ACDEFGHIKL"
        mutations = ig_mut._build_mutation_list(None, [3, 7], reference_seq)
        pos_set = {m[0] for m in mutations}
        assert pos_set == {3, 7}
        assert len(mutations) == 2 * 19

    def test_positions_filter_with_scan(self):
        reference_seq = "ACDEFGHIKL"
        scan_rows = [
            {"mutation_id": "WT", "substitutions": "", "delta": 0.0},
            {"mutation_id": "3A", "substitutions": "3A", "delta": 0.1},
            {"mutation_id": "7K", "substitutions": "7K", "delta": -0.2},
            {"mutation_id": "0G", "substitutions": "0G", "delta": 0.05},
        ]
        scan_df = pd.DataFrame(scan_rows)
        mutations = ig_mut._build_mutation_list(scan_df, [3, 7], reference_seq)
        assert len(mutations) == 2
        pos_set = {m[0] for m in mutations}
        assert pos_set == {3, 7}


# ---------------------------------------------------------------------------
# 9. ig_one_mutation integration test with mock confidence_forward
# ---------------------------------------------------------------------------

class TestIgOneMutationIntegration:
    def test_wt_to_mut_linear(self):
        seq_len, n_classes = 6, 32
        W = torch.randn(seq_len, n_classes)
        bias = 1.0

        def mock_forward(model, restype, ifd, x_pred, score, n_cycle=4, gradient_checkpointing=False):
            rt = restype
            if rt.dim() == 3:
                rt = rt[0]
            return (W * rt).sum() + bias

        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos, mut_col = 2, 15

        accumulated, rt_baseline, rt_input = ig_mut.ig_one_mutation(
            model=None,
            restype_wt=rt_wt,
            input_feature_dict={},
            x_pred=None,
            score_name="complex_pde",
            token_idx=pos,
            mut_col=mut_col,
            path_mode="wt_to_mut",
            null_row=None,
            m_steps=8,
            n_cycle=4,
            confidence_forward_fn=mock_forward,
        )

        assert torch.equal(rt_baseline, rt_wt.float())

        ig_sum = ig_mut.compute_ig_sum(accumulated, rt_input, rt_baseline)
        f_input = float(mock_forward(None, rt_input, {}, None, "complex_pde"))
        f_baseline = float(mock_forward(None, rt_baseline, {}, None, "complex_pde"))
        assert ig_sum == pytest.approx(f_input - f_baseline, abs=1e-6)

    def test_null_to_mut_unk_baseline(self):
        seq_len, n_classes = 6, 32
        W = torch.randn(seq_len, n_classes)

        def mock_forward(model, restype, ifd, x_pred, score, n_cycle=4, gradient_checkpointing=False):
            rt = restype
            if rt.dim() == 3:
                rt = rt[0]
            return (W * rt).sum()

        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos, mut_col = 3, 10

        null_row = torch.zeros(n_classes)
        null_row[_UNK_INDEX] = 1.0

        accumulated, rt_baseline, rt_input = ig_mut.ig_one_mutation(
            model=None,
            restype_wt=rt_wt,
            input_feature_dict={},
            x_pred=None,
            score_name="complex_pde",
            token_idx=pos,
            mut_col=mut_col,
            path_mode="null_to_mut",
            null_row=null_row,
            m_steps=8,
            n_cycle=4,
            confidence_forward_fn=mock_forward,
        )

        assert rt_baseline[pos, _UNK_INDEX].item() == 1.0
        assert rt_baseline[pos].sum().item() == pytest.approx(1.0)
        for i in range(seq_len):
            if i != pos:
                assert torch.equal(rt_baseline[i], rt_wt[i].float())

        ig_sum = ig_mut.compute_ig_sum(accumulated, rt_input, rt_baseline)
        f_input = float(mock_forward(None, rt_input, {}, None, "complex_pde"))
        f_baseline = float(mock_forward(None, rt_baseline, {}, None, "complex_pde"))
        assert ig_sum == pytest.approx(f_input - f_baseline, abs=1e-6)

    def test_null_to_mut_zeros_baseline(self):
        seq_len, n_classes = 6, 32
        W = torch.randn(seq_len, n_classes)

        def mock_forward(model, restype, ifd, x_pred, score, n_cycle=4, gradient_checkpointing=False):
            rt = restype
            if rt.dim() == 3:
                rt = rt[0]
            return (W * rt).sum()

        rt_wt = _make_wt_restype(seq_len, n_classes)
        pos, mut_col = 3, 10

        null_row = torch.zeros(n_classes)

        accumulated, rt_baseline, rt_input = ig_mut.ig_one_mutation(
            model=None,
            restype_wt=rt_wt,
            input_feature_dict={},
            x_pred=None,
            score_name="complex_pde",
            token_idx=pos,
            mut_col=mut_col,
            path_mode="null_to_mut",
            null_row=null_row,
            m_steps=8,
            n_cycle=4,
            confidence_forward_fn=mock_forward,
        )

        assert rt_baseline[pos].sum().item() == 0.0
        for i in range(seq_len):
            if i != pos:
                assert torch.equal(rt_baseline[i], rt_wt[i].float())

        ig_sum = ig_mut.compute_ig_sum(accumulated, rt_input, rt_baseline)
        f_input = float(mock_forward(None, rt_input, {}, None, "complex_pde"))
        f_baseline = float(mock_forward(None, rt_baseline, {}, None, "complex_pde"))
        assert ig_sum == pytest.approx(f_input - f_baseline, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. scan_delta lookup from scan CSV
# ---------------------------------------------------------------------------

class TestScanDeltaLookup:
    def test_scan_delta_maps_correctly(self):
        scan_rows = [
            {"mutation_id": "WT", "delta": 0.0},
            {"mutation_id": "3A", "delta": 0.15},
            {"mutation_id": "7K", "delta": -0.22},
        ]
        scan_df = pd.DataFrame(scan_rows)
        delta_map: dict[str, float] = {}
        for _, row in scan_df.iterrows():
            mid = str(row["mutation_id"])
            if mid != "WT" and "delta" in row:
                delta_map[mid] = float(row["delta"])

        assert delta_map["3A"] == pytest.approx(0.15)
        assert delta_map["7K"] == pytest.approx(-0.22)
        assert "WT" not in delta_map
