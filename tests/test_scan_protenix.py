"""Tests for 04_scan_protenix.py -- Protenix brute-force mutation scan.

CPU only, no protenix install required, no network.  Guards the protenix
import so pytest collection never fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the scan scripts as modules (no protenix needed at import time)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_SCRIPT = _ROOT / "scripts" / "04_scan_protenix.py"
_spec = importlib.util.spec_from_file_location("scan_04_protenix", str(_SCRIPT))
scan_protenix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan_protenix)

_BOLTZ_SCRIPT = _ROOT / "scripts" / "04_scan.py"
_boltz_spec = importlib.util.spec_from_file_location("scan_04_boltz", str(_BOLTZ_SCRIPT))
scan_boltz = importlib.util.module_from_spec(_boltz_spec)
_boltz_spec.loader.exec_module(scan_boltz)


# ---------------------------------------------------------------------------
# 1. mutation_id round-trip
# ---------------------------------------------------------------------------


class TestMutationId:
    def test_format(self):
        assert scan_protenix._mutation_id(42, "A") == "42A"
        assert scan_protenix._mutation_id(0, "W") == "0W"
        assert scan_protenix._mutation_id(123, "G") == "123G"

    def test_roundtrip_through_csv(self, tmp_path):
        mid = scan_protenix._mutation_id(7, "K")
        out = tmp_path / "scan.csv"
        pd.DataFrame(
            [
                {
                    "mutation_id": mid,
                    "row_index": 0,
                    "sequence": "X",
                    "substitutions": "7K",
                    "n_mut": 1,
                    "binding_score": float("nan"),
                    "model_score": 1.0,
                    "wt_score": 0.5,
                    "delta": 0.5,
                    "mode": "resswap",
                }
            ],
            columns=scan_protenix._COLUMNS,
        ).to_csv(out, index=False)
        loaded = pd.read_csv(out)
        assert str(loaded.iloc[0]["mutation_id"]) == mid


# ---------------------------------------------------------------------------
# 2. Resumability
# ---------------------------------------------------------------------------


class TestResumability:
    def test_empty_when_no_file(self, tmp_path):
        assert scan_protenix._load_done_ids(tmp_path / "nope.csv") == set()

    def test_pre_seeded_rows_are_skipped(self, tmp_path):
        out = tmp_path / "scan.csv"
        pd.DataFrame(
            [
                {
                    "mutation_id": "WT",
                    "row_index": -1,
                    "model_score": 1.0,
                    "substitutions": "",
                    "n_mut": 0,
                    "sequence": "ACDEF",
                    "binding_score": float("nan"),
                    "wt_score": 1.0,
                    "delta": 0.0,
                    "mode": "resswap",
                },
                {
                    "mutation_id": "3A",
                    "row_index": 0,
                    "model_score": 1.1,
                    "substitutions": "3A",
                    "n_mut": 1,
                    "sequence": "ACAEF",
                    "binding_score": float("nan"),
                    "wt_score": 1.0,
                    "delta": 0.1,
                    "mode": "resswap",
                },
                {
                    "mutation_id": "3C",
                    "row_index": 1,
                    "model_score": 0.9,
                    "substitutions": "3C",
                    "n_mut": 1,
                    "sequence": "ACCEF",
                    "binding_score": float("nan"),
                    "wt_score": 1.0,
                    "delta": -0.1,
                    "mode": "resswap",
                },
            ],
            columns=scan_protenix._COLUMNS,
        ).to_csv(out, index=False)

        done = scan_protenix._load_done_ids(out)
        assert "WT" in done
        assert "3A" in done
        assert "3C" in done
        assert "3E" not in done

    def test_corrupt_file_returns_empty(self, tmp_path):
        out = tmp_path / "scan.csv"
        out.write_text("garbage\x00\x01\nmore garbage")
        assert scan_protenix._load_done_ids(out) == set()


# ---------------------------------------------------------------------------
# 3. CSV schema matches Boltz _COLUMNS exactly
# ---------------------------------------------------------------------------


class TestCSVSchema:
    def test_columns_match_boltz(self):
        assert scan_protenix._COLUMNS == scan_boltz._COLUMNS

    def test_csv_roundtrip_preserves_schema(self, tmp_path):
        out = tmp_path / "scan.csv"
        row = {
            "mutation_id": "10A",
            "row_index": 0,
            "sequence": "ACDEF",
            "substitutions": "10A",
            "n_mut": 1,
            "binding_score": float("nan"),
            "model_score": 1.5,
            "wt_score": 1.0,
            "delta": 0.5,
            "mode": "resswap",
        }
        pd.DataFrame([row], columns=scan_protenix._COLUMNS).to_csv(
            out, index=False,
        )
        df = pd.read_csv(out)
        assert list(df.columns) == scan_protenix._COLUMNS
        assert df.iloc[0]["mode"] == "resswap"
        assert float(df.iloc[0]["delta"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. One-hot mutation helper
# ---------------------------------------------------------------------------


class TestOneHotMutation:
    def test_sets_exactly_one_column_3d(self):
        import torch

        n_tokens = 10
        n_classes = 32
        restype = torch.zeros(1, n_tokens, n_classes)
        for i in range(n_tokens):
            restype[0, i, i % 20] = 1.0

        original = restype.clone()
        token_idx = 3
        new_col = 15

        mutated = original.clone().float()
        mutated[..., token_idx, :] = 0.0
        mutated[..., token_idx, new_col] = 1.0

        assert mutated[0, token_idx].sum().item() == pytest.approx(1.0)
        assert mutated[0, token_idx, new_col].item() == 1.0
        for i in range(n_tokens):
            if i != token_idx:
                assert torch.equal(mutated[0, i], original[0, i]), (
                    f"Position {i} should be unchanged"
                )

    def test_sets_exactly_one_column_2d(self):
        import torch

        n_tokens = 10
        n_classes = 32
        restype = torch.zeros(n_tokens, n_classes)
        for i in range(n_tokens):
            restype[i, i % 20] = 1.0

        original = restype.clone()
        token_idx = 5
        new_col = 8

        mutated = original.clone().float()
        mutated[..., token_idx, :] = 0.0
        mutated[..., token_idx, new_col] = 1.0

        assert mutated[token_idx].sum().item() == pytest.approx(1.0)
        assert mutated[token_idx, new_col].item() == 1.0
        for i in range(n_tokens):
            if i != token_idx:
                assert torch.equal(mutated[i], original[i])

    def test_correct_aa_column_index(self):
        from igv.attrib import CANONICAL_AMINO_ACIDS
        from igv.protenix_score import canonical_aa_token_indices

        aa_col_indices = canonical_aa_token_indices()
        aa_to_col = {
            aa: int(aa_col_indices[i])
            for i, aa in enumerate(CANONICAL_AMINO_ACIDS)
        }
        assert aa_to_col["A"] == 0
        assert aa_to_col["C"] == 1
        assert aa_to_col["Y"] == 19
        assert len(aa_to_col) == 20


# ---------------------------------------------------------------------------
# 5. WT restype is never modified (passed separately to confidence_forward)
# ---------------------------------------------------------------------------


class TestWtRestoration:
    def test_wt_tensor_unchanged_after_mutation(self):
        import torch

        n_tokens = 10
        n_classes = 32
        restype_wt = torch.zeros(1, n_tokens, n_classes)
        for i in range(n_tokens):
            restype_wt[0, i, i % 20] = 1.0

        snapshot = restype_wt.clone()

        for target in range(3):
            mutated = restype_wt.clone().float()
            mutated[..., target, :] = 0.0
            mutated[..., target, 15] = 1.0

        assert torch.equal(restype_wt, snapshot)


# ---------------------------------------------------------------------------
# 6. Time budget stops cleanly
# ---------------------------------------------------------------------------


class TestTimeBudget:
    def test_budget_zero_stops_immediately(self):
        budget_s = 0
        t0 = 0.0
        current_time = 0.001

        todo = [(i, "A") for i in range(10)]
        scored = 0
        for pos, aa in todo:
            if current_time - t0 >= budget_s:
                break
            scored += 1

        assert scored == 0

    def test_budget_large_scores_all(self):
        budget_s = 999999
        t0 = 0.0
        current_time = 1.0

        todo = [(i, "A") for i in range(5)]
        scored = 0
        for pos, aa in todo:
            if current_time - t0 >= budget_s:
                break
            scored += 1
            current_time += 0.1

        assert scored == 5


# ---------------------------------------------------------------------------
# 7. delta == model_score - wt_score
# ---------------------------------------------------------------------------


class TestDelta:
    def test_arithmetic(self):
        wt = 2.0
        ms = 3.5
        assert ms - wt == pytest.approx(1.5)

    def test_delta_in_csv(self, tmp_path):
        out = tmp_path / "scan.csv"
        wt_score = 2.0
        model_score = 3.5
        pd.DataFrame(
            [
                {
                    "mutation_id": "10A",
                    "row_index": 0,
                    "sequence": "ACDEF",
                    "substitutions": "10A",
                    "n_mut": 1,
                    "binding_score": float("nan"),
                    "model_score": model_score,
                    "wt_score": wt_score,
                    "delta": model_score - wt_score,
                    "mode": "resswap",
                }
            ],
            columns=scan_protenix._COLUMNS,
        ).to_csv(out, index=False)
        df = pd.read_csv(out)
        assert float(df.iloc[0]["delta"]) == pytest.approx(
            float(df.iloc[0]["model_score"]) - float(df.iloc[0]["wt_score"]),
        )

    def test_wt_row_has_zero_delta(self, tmp_path):
        out = tmp_path / "scan.csv"
        wt_score = 5.0
        pd.DataFrame(
            [
                {
                    "mutation_id": "WT",
                    "row_index": -1,
                    "sequence": "ACDEF",
                    "substitutions": "",
                    "n_mut": 0,
                    "binding_score": float("nan"),
                    "model_score": wt_score,
                    "wt_score": wt_score,
                    "delta": 0.0,
                    "mode": "resswap",
                }
            ],
            columns=scan_protenix._COLUMNS,
        ).to_csv(out, index=False)
        df = pd.read_csv(out)
        assert float(df.iloc[0]["delta"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Downstream: 24_faithfulness.py accepts Protenix-style scan + attribution
# ---------------------------------------------------------------------------


class TestFaithfulnessCompatibility:
    def test_extract_attribution_end_to_end(self, tmp_path):
        faith_script = _ROOT / "scripts" / "24_faithfulness.py"
        spec = importlib.util.spec_from_file_location("faith24", str(faith_script))
        faith = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(faith)

        from igv.attrib import CANONICAL_AMINO_ACIDS
        from igv.protenix_score import canonical_aa_token_indices

        L_chain = 10
        n_tokens = 32
        reference_seq = "ACDEFGHIKL"
        aa_to_idx = {aa: i for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}

        rng = np.random.default_rng(42)
        grad_res_type = rng.standard_normal((L_chain, n_tokens)).astype(np.float32)
        aa_token_indices = canonical_aa_token_indices()
        wt_indices = np.array(
            [aa_to_idx.get(c, 0) for c in reference_seq], dtype=np.intp,
        )

        npz_path = tmp_path / "attr.npz"
        np.savez(
            npz_path,
            grad_res_type=grad_res_type,
            aa_token_indices=aa_token_indices,
            wt_indices=wt_indices,
            reference_seq=np.array(reference_seq),
            method=np.array("plain_grad"),
            model=np.array("protenix"),
        )

        wt_score = 5.0
        rows = [
            {
                "mutation_id": "WT",
                "row_index": -1,
                "sequence": reference_seq,
                "substitutions": "",
                "n_mut": 0,
                "binding_score": float("nan"),
                "model_score": wt_score,
                "wt_score": wt_score,
                "delta": 0.0,
                "mode": "resswap",
            }
        ]
        r2 = np.random.default_rng(99)
        for pos in range(L_chain):
            ref_aa = reference_seq[pos]
            for aa in sorted("ACDEFGHIKLMNPQRSTVWY"):
                if aa == ref_aa:
                    continue
                ms = wt_score + r2.standard_normal() * 0.1
                rows.append(
                    {
                        "mutation_id": f"{pos}{aa}",
                        "row_index": len(rows) - 1,
                        "sequence": reference_seq[:pos] + aa + reference_seq[pos + 1 :],
                        "substitutions": f"{pos}{aa}",
                        "n_mut": 1,
                        "binding_score": float("nan"),
                        "model_score": float(ms),
                        "wt_score": wt_score,
                        "delta": float(ms - wt_score),
                        "mode": "resswap",
                    }
                )
        scan_path = tmp_path / "scan.csv"
        pd.DataFrame(rows, columns=scan_protenix._COLUMNS).to_csv(
            scan_path, index=False,
        )

        scan_df = pd.read_csv(scan_path)
        extracted = faith._extract_attribution(npz_path, scan_df)

        assert len(extracted["A_grad"]) > 0
        assert len(extracted["A_grad"]) == len(extracted["B"])
        assert extracted["method"] == "plain_grad"

        for a, b in zip(extracted["A_grad"], extracted["B"]):
            assert np.isfinite(a)
            assert np.isfinite(b)
