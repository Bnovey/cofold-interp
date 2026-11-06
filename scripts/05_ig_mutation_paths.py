#!/usr/bin/env python3
"""Stage 05 -- per-mutation IG paths for Protenix.

For each mutation, runs a SEPARATE short IG path that changes only the mutated
position. The IG sum is directly comparable to the scan's ``delta`` because
the path endpoints are exactly wild-type and mutant.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.dms import resolve_pdb_complex  # noqa: E402
from igv.provenance import write as prov_write  # noqa: E402
from igv.skempi import skempi_positions  # noqa: E402

log = logging.getLogger("ig_mutation_paths")

AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")

_COLUMNS = [
    "mutation_id",
    "position",
    "wt_aa",
    "mut_aa",
    "f_baseline",
    "f_input",
    "ig_sum",
    "scan_delta",
    "completeness_error",
    "m_steps",
    "path_mode",
]


def _mutation_id(pos: int, aa: str) -> str:
    return f"{pos}{aa}"


def _load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    try:
        prev = pd.read_csv(out_path)
    except Exception:
        return done
    if "mutation_id" in prev.columns:
        done = set(prev["mutation_id"].astype(str))
    return done


def _load_completeness_error():
    script = Path(__file__).resolve().parent / "03_attribute_protenix.py"
    spec = importlib.util.spec_from_file_location("attribute_protenix_03", str(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.completeness_error_from_sum


completeness_error_from_sum = _load_completeness_error()


def _build_mutation_list(
    scan_df: pd.DataFrame | None,
    positions: list[int],
    reference_seq: str,
) -> list[tuple[int, str, str]]:
    """Return (position, wt_aa, mut_aa) triples."""
    if scan_df is not None:
        mutations = []
        for _, row in scan_df.iterrows():
            mid = str(row["mutation_id"])
            if mid == "WT":
                continue
            subs = str(row.get("substitutions", mid))
            pos = int("".join(c for c in subs if c.isdigit()))
            mut_aa = subs[-1]
            if pos < 0 or pos >= len(reference_seq):
                continue
            if positions and pos not in positions:
                continue
            wt_aa = reference_seq[pos]
            mutations.append((pos, wt_aa, mut_aa))
        return mutations

    pairs = []
    for pos in positions:
        wt_aa = reference_seq[pos]
        for aa in AMINO_ACIDS:
            if aa != wt_aa:
                pairs.append((pos, wt_aa, aa))
    return pairs


def ig_one_mutation(
    model,
    restype_wt,
    input_feature_dict,
    x_pred,
    score_name: str,
    token_idx: int,
    mut_col: int,
    path_mode: str,
    null_row,
    m_steps: int,
    n_cycle: int,
    confidence_forward_fn,
):
    """Run IG for a single mutation. Returns (accumulated_grad, rt_baseline, rt_input)."""
    import torch

    rt_baseline = restype_wt.clone().float()
    rt_input = restype_wt.clone().float()

    rt_input[..., token_idx, :] = 0.0
    rt_input[..., token_idx, mut_col] = 1.0

    if path_mode == "null_to_mut":
        rt_baseline[..., token_idx, :] = null_row.clone()

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(m_steps)
    alphas = (gl_nodes + 1.0) / 2.0
    weights = gl_weights / 2.0

    shape_2d = rt_input.shape[-2], rt_input.shape[-1]
    accumulated = np.zeros(shape_2d, dtype=np.float64)

    for alpha, weight in zip(alphas, weights):
        rt_interp = (
            rt_baseline + alpha * (rt_input - rt_baseline)
        ).detach().requires_grad_(True)

        scalar = confidence_forward_fn(
            model, rt_interp, input_feature_dict, x_pred, score_name,
            n_cycle=n_cycle, gradient_checkpointing=False,
        )
        scalar.backward()

        grad = rt_interp.grad
        if grad is None:
            raise RuntimeError("No gradient on restype at IG step.")
        g = grad.detach().cpu().float().numpy()
        if g.ndim == 3:
            g = g[0]
        accumulated += weight * g.astype(np.float64)
        del rt_interp, scalar

    return accumulated, rt_baseline, rt_input


def compute_ig_sum(accumulated, rt_input, rt_baseline):
    """Compute ig_sum over ALL tokens and ALL restype columns."""
    x_np = rt_input.detach().cpu().float().numpy()
    b_np = rt_baseline.detach().cpu().float().numpy()
    if x_np.ndim == 3:
        x_np = x_np[0]
    if b_np.ndim == 3:
        b_np = b_np[0]
    ig_attribution = (x_np.astype(np.float64) - b_np.astype(np.float64)) * accumulated
    return float(ig_attribution.sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--chain", default="H")
    p.add_argument("--score", default="complex_pde")
    p.add_argument("--n-cycle", type=int, default=4)
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument(
        "--out", default="results/{dataset}_{score}_ig_mutation_paths.csv",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights-dir", default=None)
    p.add_argument(
        "--model-name", default="protenix_base_default_v1.0.0",
    )
    p.add_argument("--m-steps", type=int, default=8)
    p.add_argument(
        "--path-mode", default="wt_to_mut",
        choices=["wt_to_mut", "null_to_mut"],
    )
    p.add_argument(
        "--null-mode", default="unk", choices=["unk", "zeros"],
    )
    p.add_argument("--scan", default=None, help="Scan CSV to read delta from")
    p.add_argument(
        "--max-mutations", type=int, default=0,
        help="Take the first N mutations (0 = all).",
    )
    p.add_argument(
        "--positions", default="skempi",
        help=(
            "Positions to scan. "
            "'skempi' (default) = SKEMPI-measured positions; "
            "or comma-separated 0-based indices."
        ),
    )
    p.add_argument(
        "--single-seq", action="store_true",
    )
    p.add_argument("--msa-dir", default=None)
    p.add_argument("--msa-depth", type=int, default=None)
    p.add_argument(
        "--x-pred-cache", action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument(
        "--feature-cache", action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument(
        "--time-budget-s", type=int, default=6600,
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    from igv.protenix_score import PROTENIX_SCORES

    if args.score not in PROTENIX_SCORES:
        raise SystemExit(
            f"--score {args.score!r} is not valid. "
            f"Available: {sorted(PROTENIX_SCORES)}"
        )

    out = Path(args.out.format(dataset=args.dataset, score=args.score))
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    resolved = resolve_pdb_complex(args.dataset, args.chain, cache_dir)
    pdb_path = resolved.pdb_path
    reference_seq = resolved.reference_seq
    struct_chains = resolved.chains
    struct_name = resolved.struct_name

    log.info(
        "Complex: %s  chains: %s",
        struct_name, {c: len(s) for c, s in struct_chains.items()},
    )

    if args.positions == "skempi":
        positions = skempi_positions(args.dataset, args.chain, cache_dir)
    else:
        positions = sorted(int(x) for x in args.positions.split(","))

    for pos in positions:
        if pos < 0 or pos >= len(reference_seq):
            raise ValueError(
                f"Position {pos} out of range for chain {args.chain} "
                f"(length {len(reference_seq)})"
            )

    scan_df = None
    scan_delta_map: dict[str, float] = {}
    if args.scan:
        scan_df = pd.read_csv(args.scan)
        for _, row in scan_df.iterrows():
            mid = str(row["mutation_id"])
            if mid != "WT" and "delta" in row:
                scan_delta_map[mid] = float(row["delta"])

    mutations = _build_mutation_list(scan_df, positions, reference_seq)

    if args.max_mutations > 0:
        mutations = mutations[: args.max_mutations]

    log.info(
        "Per-mutation IG: %d mutations, m_steps=%d, path_mode=%s",
        len(mutations), args.m_steps, args.path_mode,
    )

    done_ids = _load_done_ids(out)
    if done_ids:
        log.info("Resuming: %d mutations already in %s", len(done_ids), out)
    todo = [
        (pos, wt, mut) for pos, wt, mut in mutations
        if _mutation_id(pos, mut) not in done_ids
    ]

    if not todo:
        log.info("Nothing to do; all %d mutations already scored.", len(mutations))
        return

    if args.dry_run:
        log.info(
            "--dry-run: would score %d mutations (%d remaining)",
            len(mutations), len(todo),
        )
        return

    import torch

    from igv.attrib import CANONICAL_AMINO_ACIDS, free_cuda_memory
    from igv.protenix_score import (
        _UNK_INDEX,
        build_features,
        canonical_aa_token_indices,
        confidence_forward,
        coord_fingerprint,
        load_model,
        numerics_arm,
        predict_structure_coords,
        resolve_msa_paths,
    )

    log.info("Loading Protenix model (%s)", args.model_name)
    model, configs = load_model(
        args.device,
        model_name=args.model_name,
        weights_dir=args.weights_dir,
        n_cycle=args.n_cycle,
    )

    msa_dir_arg = args.msa_dir
    if msa_dir_arg is None and not args.single_seq:
        default_msa = cache_dir / "boltz_attr" / f"{args.dataset}_{args.chain}"
        if default_msa.is_dir():
            msa_dir_arg = str(default_msa)
            log.info("Using default Boltz MSA dir: %s", msa_dir_arg)
        else:
            raise FileNotFoundError(
                f"No --msa-dir supplied and default Boltz MSA dir {default_msa} "
                "does not exist. Either run the Boltz pipeline first, supply "
                "--msa-dir explicitly, or pass --single-seq to run without MSA."
            )
    msa_paths = resolve_msa_paths(
        struct_chains, msa_dir_arg, single_seq=args.single_seq,
    )
    msa_source = "single_seq" if msa_paths is None else str(msa_dir_arg)
    log.info("MSA source: %s (depth=%s)", msa_source, args.msa_depth or "default")

    feat_cache = cache_dir / "protenix_ig_mut" / f"{args.dataset}_{args.chain}"
    log.info("Building Protenix features (cache=%s, feature_cache=%s)", feat_cache, args.feature_cache)
    input_feature_dict, token_map = build_features(
        struct_chains, feat_cache, args.device, configs, model=model,
        msa_paths=msa_paths, msa_depth=args.msa_depth,
        use_feature_cache=args.feature_cache, model_name=args.model_name,
        feature_cache_dir=cache_dir / "protenix_features",
    )
    _ref_pos_fp = coord_fingerprint(input_feature_dict["ref_pos"])
    log.info("ref_pos fingerprint: %s", _ref_pos_fp)

    for i in range(len(reference_seq)):
        if (args.chain, i) not in token_map:
            raise RuntimeError(
                f"token_map missing ({args.chain}, {i}); the featurisation "
                f"does not cover the full reference sequence "
                f"(length {len(reference_seq)})."
            )

    log.info("Predicting wild-type structure coordinates ...")
    cache_kw = (
        dict(cache_dir=cache_dir, chains=struct_chains, dataset=args.dataset)
        if args.x_pred_cache else {}
    )
    x_pred = predict_structure_coords(
        model, input_feature_dict, configs, args.device, **cache_kw,
    )
    _x_pred_fp = coord_fingerprint(x_pred)
    log.info("x_pred fingerprint: %s", _x_pred_fp)

    aa_col_indices = canonical_aa_token_indices()
    aa_to_col = {
        aa: int(aa_col_indices[i])
        for i, aa in enumerate(CANONICAL_AMINO_ACIDS)
    }

    restype_wt = input_feature_dict["restype"].detach().clone()
    n_classes = restype_wt.shape[-1]

    if args.path_mode == "null_to_mut":
        if args.null_mode == "unk":
            null_row = torch.zeros(n_classes, device=restype_wt.device, dtype=torch.float32)
            null_row[_UNK_INDEX] = 1.0
        else:
            null_row = torch.zeros(n_classes, device=restype_wt.device, dtype=torch.float32)
    else:
        null_row = None

    done_ids = _load_done_ids(out)
    todo = [
        (pos, wt, mut) for pos, wt, mut in mutations
        if _mutation_id(pos, mut) not in done_ids
    ]

    if not todo:
        log.info("Nothing to do; all mutations already scored.")
    else:
        write_header = not out.exists()
        t0 = time.time()
        n_scored = 0
        n_skipped = len(done_ids)

        for pos, wt_aa, mut_aa in todo:
            elapsed = time.time() - t0
            if elapsed >= args.time_budget_s:
                remaining = len(todo) - n_scored
                log.info(
                    "Time budget exhausted (%.0f s >= %d s). "
                    "Stopping cleanly; %d mutations remain.",
                    elapsed, args.time_budget_s, remaining,
                )
                break

            mid = _mutation_id(pos, mut_aa)
            token_idx = token_map.get((args.chain, pos))
            if token_idx is None:
                log.warning("Position %d not in token_map; skipping %s", pos, mid)
                continue

            accumulated, rt_baseline, rt_input = ig_one_mutation(
                model, restype_wt, input_feature_dict, x_pred,
                args.score, token_idx, aa_to_col[mut_aa],
                args.path_mode, null_row,
                args.m_steps, args.n_cycle,
                confidence_forward,
            )

            with torch.no_grad():
                f_baseline = float(confidence_forward(
                    model, rt_baseline.detach().float(),
                    input_feature_dict, x_pred, args.score,
                    n_cycle=args.n_cycle, gradient_checkpointing=False,
                ))
                f_input = float(confidence_forward(
                    model, rt_input.detach().float(),
                    input_feature_dict, x_pred, args.score,
                    n_cycle=args.n_cycle, gradient_checkpointing=False,
                ))

            ig_sum = compute_ig_sum(accumulated, rt_input, rt_baseline)
            comp_err = completeness_error_from_sum(ig_sum, f_input, f_baseline)

            scan_delta = scan_delta_map.get(mid, float("nan"))

            if args.path_mode == "wt_to_mut" and not np.isnan(scan_delta):
                actual_delta = f_input - f_baseline
                if abs(actual_delta - scan_delta) > 0.01:
                    log.warning(
                        "Mutation %s: f_input - f_baseline = %.6f but "
                        "scan delta = %.6f (diff %.6f). Possible featurisation "
                        "or structure mismatch.",
                        mid, actual_delta, scan_delta, actual_delta - scan_delta,
                    )

            pd.DataFrame(
                [
                    {
                        "mutation_id": mid,
                        "position": pos,
                        "wt_aa": wt_aa,
                        "mut_aa": mut_aa,
                        "f_baseline": f_baseline,
                        "f_input": f_input,
                        "ig_sum": ig_sum,
                        "scan_delta": scan_delta,
                        "completeness_error": comp_err,
                        "m_steps": args.m_steps,
                        "path_mode": args.path_mode,
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)
            write_header = False

            free_cuda_memory()
            n_scored += 1
            rate = (time.time() - t0) / n_scored
            log.info(
                "[%d/%d] pos=%d %s->%s ig_sum=%.6f comp_err=%.4f "
                "(%.1fs/mut, eta %.0f min)",
                n_scored + n_skipped, len(mutations), pos, wt_aa, mut_aa,
                ig_sum, comp_err,
                rate, rate * (len(todo) - n_scored) / 60,
            )

    prov_write(
        out,
        stage="05_ig_mutation_paths",
        inputs={"dataset": args.dataset, "structure": struct_name},
        params={
            "n_mutations": len(mutations),
            "chain": args.chain,
            "path_mode": args.path_mode,
            "null_mode": args.null_mode,
            "m_steps": args.m_steps,
            "n_cycle": args.n_cycle,
            "msa_source": msa_source,
            "msa_depth": args.msa_depth,
            "model_name": args.model_name,
            "time_budget_s": args.time_budget_s,
            "feature_cache": args.feature_cache,
            "x_pred_cache": args.x_pred_cache,
        },
        arm={
            "score": args.score,
            "method": "ig_mutation_paths",
            "path_mode": args.path_mode,
            "null_mode": args.null_mode,
            "m_steps": args.m_steps,
            "dataset": args.dataset,
            "chain": args.chain,
            "x_pred_fingerprint": _x_pred_fp,
            "ref_pos_fingerprint": _ref_pos_fp,
            **numerics_arm(),
        },
        notes=(
            f"Per-mutation IG paths. "
            f"path_mode={args.path_mode}, m_steps={args.m_steps}."
        ),
    )
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
