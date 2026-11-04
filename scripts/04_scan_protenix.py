#!/usr/bin/env python3
"""Stage 04 -- brute-force mutation scan for Protenix (resswap only).

Companion to ``04_scan.py`` (Boltz-2). Produces an identical CSV schema so
``24_faithfulness.py`` and ``26_vs_experiment.py`` consume the output unchanged.

Only ``resswap`` mode is implemented: for each mutant, only the restype one-hot
at the target position is edited; coordinates, atom features and MSA stay at
wild-type values.  This is the ground truth for a first-order Taylor / TISM
prediction, matching the semantics of Protenix gradient attribution in stage 03.

Fixed geometry.  Wild-type coordinates from ``predict_structure_coords`` are
obtained once and reused for every mutant, so the scan and the gradient measure
sensitivity at identical geometry.
"""

from __future__ import annotations

import argparse
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

log = logging.getLogger("scan_protenix")

AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")

_COLUMNS = [
    "mutation_id",
    "row_index",
    "sequence",
    "substitutions",
    "n_mut",
    "binding_score",
    "model_score",
    "wt_score",
    "delta",
    "mode",
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--chain", default="H")
    p.add_argument("--score", default="complex_pde")
    p.add_argument("--n-cycle", type=int, default=4)
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--structure", default=None)
    p.add_argument(
        "--out", default="results/{dataset}_{score}_protenix_scan.csv",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights-dir", default=None)
    p.add_argument(
        "--model-name", default="protenix_base_default_v1.0.0",
        help="Protenix model/weight name (Apache-2.0 only).",
    )
    p.add_argument(
        "--x-pred", default="predicted", choices=["zeros", "predicted"],
    )
    p.add_argument(
        "--positions", default="skempi",
        help=(
            "Positions to scan (full 19-aa grid per position). "
            "'skempi' (default) = SKEMPI-measured positions; "
            "or comma-separated 0-based indices."
        ),
    )
    p.add_argument(
        "--msa-dir", default=None,
        help=(
            "Directory with per-chain .a3m files. Accepts the Boltz layout "
            "(data/raw/boltz_attr/<DATASET>_<CHAIN>/). Required unless --single-seq."
        ),
    )
    p.add_argument("--msa-depth", type=int, default=None)
    p.add_argument(
        "--single-seq", action="store_true",
        help="Run without MSA (single-sequence mode). Must be explicit.",
    )
    p.add_argument(
        "--time-budget-s", type=int, default=6600,
        help="Stop cleanly after this many seconds (default 6600, fits a 7200 s VM).",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only, no model")
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

    resolved = resolve_pdb_complex(
        args.dataset, args.chain, cache_dir,
        structure_override=args.structure,
    )
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

    mutant_pairs: list[tuple[int, str]] = []
    for pos in positions:
        ref_aa = reference_seq[pos]
        for aa in AMINO_ACIDS:
            if aa != ref_aa:
                mutant_pairs.append((pos, aa))

    log.info(
        "Protenix resswap scan: %d positions, %d mutant pairs",
        len(positions), len(mutant_pairs),
    )

    boltz_scan_path = Path(f"results/{args.dataset}_{args.score}_scan.csv")
    if boltz_scan_path.exists():
        try:
            boltz_df = pd.read_csv(boltz_scan_path)
            boltz_mids = set(boltz_df["mutation_id"].astype(str))
            our_mids = {_mutation_id(p, a) for p, a in mutant_pairs} | {"WT"}
            overlap = boltz_mids & our_mids
            log.info(
                "Overlap with Boltz scan (%s): %d/%d mutations",
                boltz_scan_path, len(overlap), len(our_mids),
            )
        except Exception:
            pass

    done_ids = _load_done_ids(out)
    if done_ids:
        log.info(
            "Resuming: %d mutations already scored in %s", len(done_ids), out,
        )
    todo = [(p, a) for p, a in mutant_pairs if _mutation_id(p, a) not in done_ids]
    wt_already = "WT" in done_ids

    if not todo and wt_already:
        log.info(
            "Nothing to do; all %d mutant pairs already scored.", len(mutant_pairs),
        )
        return

    if args.dry_run:
        log.info(
            "--dry-run: would score %d mutants (%d remaining) with score=%s",
            len(mutant_pairs), len(todo), args.score,
        )
        return

    import torch

    from igv.attrib import CANONICAL_AMINO_ACIDS, free_cuda_memory
    from igv.protenix_score import (
        build_features,
        canonical_aa_token_indices,
        confidence_forward,
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

    feat_cache = cache_dir / "protenix_scan" / f"{args.dataset}_{args.chain}"
    log.info("Building Protenix features (cache=%s)", feat_cache)
    input_feature_dict, token_map = build_features(
        struct_chains, feat_cache, args.device, configs, model=model,
        msa_paths=msa_paths, msa_depth=args.msa_depth,
    )

    for i in range(len(reference_seq)):
        if (args.chain, i) not in token_map:
            raise RuntimeError(
                f"token_map missing ({args.chain}, {i}); the featurisation "
                f"does not cover the full reference sequence "
                f"(length {len(reference_seq)}). "
                "Check that the input JSON matches the PDB."
            )

    if args.x_pred == "predicted":
        log.info("Predicting wild-type structure coordinates ...")
        x_pred = predict_structure_coords(
            model, input_feature_dict, configs, args.device,
        )
    else:
        n_atoms = input_feature_dict["ref_pos"].shape[-2]
        x_pred = torch.zeros(1, 1, n_atoms, 3, device=args.device)

    _coords_max = float(x_pred.abs().max())
    log.info(
        "x_pred (mode=%s, shape %s), coords.abs().max()=%g",
        args.x_pred, list(x_pred.shape), _coords_max,
    )
    if args.x_pred == "predicted" and _coords_max == 0.0:
        raise RuntimeError(
            "x_pred mode is 'predicted' but predicted coordinates are all zeros."
        )

    wt_score: float
    if wt_already:
        prev = pd.read_csv(out)
        wt_row = prev[prev["mutation_id"] == "WT"]
        wt_score = float(wt_row.iloc[0]["model_score"])
        log.info("Wild-type score (from resume): %.6f", wt_score)
    else:
        restype_wt = input_feature_dict["restype"].detach().float()
        with torch.no_grad():
            scalar = confidence_forward(
                model, restype_wt, input_feature_dict, x_pred, args.score,
                n_cycle=args.n_cycle, gradient_checkpointing=False,
            )
            wt_score = float(scalar)
        log.info("Wild-type %s = %.6f", args.score, wt_score)

        write_header = not out.exists()
        pd.DataFrame(
            [
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
            ],
            columns=_COLUMNS,
        ).to_csv(out, mode="a", header=write_header, index=False)

    aa_col_indices = canonical_aa_token_indices()
    aa_to_col = {
        aa: int(aa_col_indices[i])
        for i, aa in enumerate(CANONICAL_AMINO_ACIDS)
    }

    restype_wt = input_feature_dict["restype"].detach().clone()

    done_ids = _load_done_ids(out)
    todo = [(p, a) for p, a in mutant_pairs if _mutation_id(p, a) not in done_ids]

    if not todo:
        log.info("Nothing to do; all mutant pairs already scored.")
    else:
        write_header = not out.exists()
        t0 = time.time()
        n_scored = 0
        n_skipped = len(done_ids) - 1  # exclude WT

        for pos, aa in todo:
            elapsed = time.time() - t0
            if elapsed >= args.time_budget_s:
                remaining = len(todo) - n_scored
                log.info(
                    "Time budget exhausted (%.0f s >= %d s). "
                    "Stopping cleanly; %d mutants remain.",
                    elapsed, args.time_budget_s, remaining,
                )
                break

            mid = _mutation_id(pos, aa)
            token_idx = token_map.get((args.chain, pos))
            if token_idx is None:
                log.warning(
                    "Position %d not in token_map; skipping %s", pos, mid,
                )
                continue

            mutated_rt = restype_wt.clone().float()
            mutated_rt[..., token_idx, :] = 0.0
            mutated_rt[..., token_idx, aa_to_col[aa]] = 1.0

            with torch.no_grad():
                scalar = confidence_forward(
                    model, mutated_rt, input_feature_dict, x_pred, args.score,
                    n_cycle=args.n_cycle, gradient_checkpointing=False,
                )
                model_score = float(scalar)

            mut_seq = reference_seq[:pos] + aa + reference_seq[pos + 1:]
            pd.DataFrame(
                [
                    {
                        "mutation_id": mid,
                        "row_index": n_scored,
                        "sequence": mut_seq,
                        "substitutions": f"{pos}{aa}",
                        "n_mut": 1,
                        "binding_score": float("nan"),
                        "model_score": model_score,
                        "wt_score": wt_score,
                        "delta": model_score - wt_score,
                        "mode": "resswap",
                    }
                ],
                columns=_COLUMNS,
            ).to_csv(out, mode="a", header=write_header, index=False)
            write_header = False

            free_cuda_memory()
            n_scored += 1
            rate = (time.time() - t0) / n_scored
            log.info(
                "[%d/%d] pos=%d aa=%s score=%.6f delta=%.6f "
                "(%.1fs/mutant, eta %.0f min)",
                n_scored + n_skipped, len(mutant_pairs), pos, aa,
                model_score, model_score - wt_score,
                rate, rate * (len(todo) - n_scored) / 60,
            )

    prov_write(
        out,
        stage="04_scan_protenix",
        inputs={"dataset": args.dataset, "structure": struct_name},
        params={
            "n_positions": len(positions),
            "n_mutants": len(mutant_pairs),
            "chain": args.chain,
            "mode": "resswap",
            "x_pred": args.x_pred,
            "n_cycle": args.n_cycle,
            "msa_source": msa_source,
            "msa_depth": args.msa_depth,
            "model_name": args.model_name,
            "time_budget_s": args.time_budget_s,
        },
        arm={
            "score": args.score,
            "method": "scan",
            "trunk": "forward_only",
            "geometry": f"x_pred_{args.x_pred}",
            "dataset": args.dataset,
            "chain": args.chain,
            "mode": "resswap",
            "x_pred_coords_frozen": True,
            "wt_score": wt_score,
            **numerics_arm(),
        },
        notes=(
            f"Protenix brute-force resswap scan. "
            f"WT coords frozen ({args.x_pred})."
        ),
    )
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
