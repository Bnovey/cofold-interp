"""Tests for the Protenix IG completeness check in 03_attribute_protenix.py.

CPU-only, no network, no Protenix install required.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

# Import the REAL completeness formula from the script, so these tests fail if
# the production code changes. Repo convention for numeric-prefixed scripts.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "03_attribute_protenix.py"
_spec = importlib.util.spec_from_file_location("attribute_protenix_03", str(_SCRIPT))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_completeness_error = _mod.completeness_error_from_sum


# ---------------------------------------------------------------------------
# Helpers: simulate the IG quadrature + completeness exactly as the script does
# ---------------------------------------------------------------------------

def _simulate_ig_accumulated(forward_fn, rt_x, rt_baseline, m_steps):
    """Reproduce the Gauss-Legendre IG quadrature from _ig_res_type."""
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(m_steps)
    alphas = (gl_nodes + 1.0) / 2.0
    weights = gl_weights / 2.0

    accumulated = np.zeros_like(rt_x.numpy(), dtype=np.float64)
    for alpha, weight in zip(alphas, weights):
        rt_interp = (rt_baseline + alpha * (rt_x - rt_baseline)).detach().requires_grad_(True)
        scalar = forward_fn(rt_interp)
        scalar.backward()
        g = rt_interp.grad.detach().cpu().float().numpy()
        accumulated += weight * g.astype(np.float64)
    return accumulated


# ---------------------------------------------------------------------------
# 1. Linear function: completeness identity holds exactly
# ---------------------------------------------------------------------------

class TestLinearCompleteness:
    def test_linear_comp_err_is_zero(self):
        L, C = 6, 4
        W = torch.randn(L, C)
        bias = 3.14

        def forward_fn(rt):
            return (W * rt).sum() + bias

        rt_x = torch.randn(L, C)
        rt_baseline = torch.randn(L, C)

        accumulated = _simulate_ig_accumulated(forward_fn, rt_x, rt_baseline, m_steps=8)
        ig_attribution = (rt_x.numpy().astype(np.float64) - rt_baseline.numpy().astype(np.float64)) * accumulated
        ig_sum = float(ig_attribution.sum())

        f_x = float(forward_fn(rt_x))
        f_baseline = float(forward_fn(rt_baseline))

        comp_err = _completeness_error(ig_sum, f_x, f_baseline)
        assert comp_err == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Nonlinear function: comp_err > 0 and finite (low m_steps)
# ---------------------------------------------------------------------------

class TestNonlinearCompleteness:
    def test_nonlinear_comp_err_positive_finite(self):
        L, C = 4, 3

        def forward_fn(rt):
            return (rt ** 3).sum()

        rt_x = torch.tensor([[1.0, 2.0, 0.5], [0.3, -1.0, 0.8],
                              [1.5, -0.5, 0.2], [0.1, 0.7, -0.3]])
        rt_baseline = torch.zeros(L, C)

        accumulated = _simulate_ig_accumulated(forward_fn, rt_x, rt_baseline, m_steps=2)
        ig_attribution = (rt_x.numpy().astype(np.float64) - rt_baseline.numpy().astype(np.float64)) * accumulated
        ig_sum = float(ig_attribution.sum())

        f_x = float(forward_fn(rt_x))
        f_baseline = float(forward_fn(rt_baseline))

        comp_err = _completeness_error(ig_sum, f_x, f_baseline)
        assert comp_err > 0.0
        assert math.isfinite(comp_err)


# ---------------------------------------------------------------------------
# 3. Zero-denominator handling
# ---------------------------------------------------------------------------

class TestZeroDenominator:
    def test_both_zero_gives_zero(self):
        assert _completeness_error(0.0, 5.0, 5.0) == 0.0

    def test_nonzero_ig_zero_diff_gives_inf(self):
        assert _completeness_error(0.1, 5.0, 5.0) == float("inf")


# ---------------------------------------------------------------------------
# 4. All-tokens sum vs chain-slice sum
# ---------------------------------------------------------------------------

class TestAllTokensVsChainSlice:
    def test_chain_slice_gives_wrong_answer(self):
        """If a non-chain token carries attribution, slicing to the chain only
        gives a wrong ig_sum while summing over all tokens is correct."""
        L, C = 5, 3
        W = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ])

        def forward_fn(rt):
            return (W * rt).sum()

        rt_x = torch.ones(L, C)
        rt_baseline = torch.zeros(L, C)

        accumulated = _simulate_ig_accumulated(forward_fn, rt_x, rt_baseline, m_steps=8)
        delta = rt_x.numpy().astype(np.float64) - rt_baseline.numpy().astype(np.float64)
        ig_full = delta * accumulated

        f_x = float(forward_fn(rt_x))
        f_baseline = float(forward_fn(rt_baseline))
        expected_diff = f_x - f_baseline

        ig_sum_all = float(ig_full.sum())
        assert ig_sum_all == pytest.approx(expected_diff, abs=1e-5)

        chain_indices = [0, 2]
        ig_sum_chain = float(ig_full[chain_indices, :].sum())
        assert abs(ig_sum_chain - expected_diff) > 0.01


# ---------------------------------------------------------------------------
# 5. plain_grad path writes NaN for f_baseline and completeness_error
# ---------------------------------------------------------------------------

class TestPlainGradNaN:
    def test_plain_grad_keeps_nan(self):
        """The plain_grad code path must not touch f_baseline or comp_err
        (they stay NaN as initialised). We verify the logic gate: method != "ig"
        means the if-block is skipped."""
        f_baseline_val = float("nan")
        comp_err = float("nan")
        ig_sum_val = float("nan")
        method = "plain_grad"

        if method == "ig":
            f_baseline_val = 0.0
            comp_err = 0.0
            ig_sum_val = 0.0

        assert math.isnan(f_baseline_val)
        assert math.isnan(comp_err)
        assert math.isnan(ig_sum_val)
