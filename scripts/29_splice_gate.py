#!/usr/bin/env python3
"""Stage 29 -- splice-gate: verify SAE reconstruction is faithful enough for attribution.

For each evaluation complex and SAE checkpoint, splice x_hat_orig in place of s
at a target layer and report the score change relative to the typical mutation
effect from the brute-force scans. The gate PASSES only if the splice
distortion is below a configurable fraction of the mutation effect.

Arms:
  1. baseline       -- no callback, bulk pairformer path
  2. identity       -- callback returns s unchanged (GPU check: must match baseline)
  3. sae_splice     -- s -> x_hat_orig from the SAE
  4. eps_kept       -- s -> x_hat_orig + error_orig (should equal identity)
  5. random_sae     -- s -> x_hat_orig from a random SAE with same norm stats
  6. mean_ablation  -- s -> per-dim training mean (scale for layer importance)

Resumable per (complex, arm, seed). Stops before the VM's 7200 s hard limit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.provenance import write as prov_write

log = logging.getLogger("splice_gate")

SKEMPI_COMPLEXES = ["1JTG", "3S9D", "3HFM", "2JEL"]

SCAN_PATTERNS = [
    "results/{complex}_complex_pde_scan.csv",
    "results/{complex}_complex_pde_scan_refeaturise.csv",
]

AFFINITY_SCAN_PATTERNS = [
    "results/affinity_scan_{complex}.csv",
    "results/affinity_scan_{complex}_fp32.csv",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_typical_effect(complex_id: str, repo_root: Path, mode: str = "complex_pde") -> float | None:
    """Compute median |delta| from the scan CSV for a given complex.

    Returns None if no scan file is found.
    """
    patterns = SCAN_PATTERNS if mode == "complex_pde" else AFFINITY_SCAN_PATTERNS

    scan_path = None
    for pat in patterns:
        candidate = repo_root / pat.format(complex=complex_id)
        if candidate.exists():
            scan_path = candidate
            break

    if scan_path is None:
        return None

    deltas = []
    with open(scan_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mutation_id", "") == "WT":
                continue
            d = row.get("delta") or row.get("delta_score")
            if d is not None:
                try:
                    deltas.append(abs(float(d)))
                except ValueError:
                    continue

    if not deltas:
        return None
    return float(np.median(deltas))


def gate_verdict(
    delta_splice: float,
    typical_effect: float | None,
    identity_pass: bool,
    threshold_frac: float = 0.5,
) -> tuple[bool, str]:
    if not identity_pass:
        return False, "identity arm failed"
    if typical_effect is None:
        return False, "no scan data to compute typical effect"
    if typical_effect == 0:
        return False, "typical effect is zero"
    ratio = abs(delta_splice) / typical_effect
    if ratio < threshold_frac:
        return True, f"ratio={ratio:.4f} < {threshold_frac}"
    return False, f"ratio={ratio:.4f} >= {threshold_frac}"


def _result_key(complex_id: str, arm: str, seed: int | None) -> str:
    if seed is not None:
        return f"{complex_id}_{arm}_seed{seed}"
    return f"{complex_id}_{arm}"


def _result_path(out_dir: Path, sae_name: str, layer: int) -> Path:
    return out_dir / f"{sae_name}_L{layer}.json"


def _is_arm_done(results: dict, key: str) -> bool:
    return key in results


def _make_splice_callback(
    sae, layer: int, target_recycle: int, mode: str = "sae",
    train_mean=None,
):
    """Build a callback that splices activations at a specific layer.

    mode:
      "identity"     -- return (s, z) unchanged
      "sae"          -- replace s with x_hat_orig
      "eps_kept"     -- replace s with x_hat_orig + error_orig
      "random_sae"   -- replace s with random SAE x_hat_orig
      "mean_ablation"-- replace s with per-dim training mean
    """
    import torch

    def callback(recycle_idx, layer_idx, s, z):
        if recycle_idx != target_recycle:
            return None
        if layer_idx != layer:
            return None

        if mode == "identity":
            return s, z

        if mode in ("sae", "random_sae"):
            s_squeezed = s.squeeze(0) if s.dim() == 3 else s
            with torch.no_grad():
                out = sae(s_squeezed)
            s_new = out.x_hat_orig
            if s.dim() == 3:
                s_new = s_new.unsqueeze(0)
            return s_new, z

        if mode == "eps_kept":
            s_squeezed = s.squeeze(0) if s.dim() == 3 else s
            with torch.no_grad():
                out = sae(s_squeezed)
            s_new = out.x_hat_orig + out.error_orig
            if s.dim() == 3:
                s_new = s_new.unsqueeze(0)
            return s_new, z

        if mode == "mean_ablation":
            s_new = train_mean.expand_as(s).clone()
            return s_new, z

        return None

    return callback


def _run_forward(model, s_inputs, feats, x_pred, score_name, recycling_steps,
                 activation_callback, forward_fn):
    import torch
    with torch.no_grad():
        score = forward_fn(
            model, s_inputs, feats, x_pred,
            score_name=score_name,
            gradient_checkpointing=False,
            recycling_steps=recycling_steps,
            activation_callback=activation_callback,
        )
    return score.item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sae-paths", nargs="+", required=True,
                    help="Paths to SAE checkpoint .pt files")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--recycle", type=int, default=1,
                    help="Target recycle index for splicing (default: 1 = final at recycling_steps=1)")
    p.add_argument("--recycling-steps", type=int, default=1)
    p.add_argument("--set", default="skempi", choices=["skempi", "ligand"],
                    help="Evaluation set (default: skempi)")
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--checkpoint-dir",
                    default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-msa-server", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Gate threshold: max |Δsplice| / typical_effect (default 0.5)")
    p.add_argument("--out-dir", default="results/sae_gate")
    p.add_argument("--time-budget-s", type=int, default=6600)
    p.add_argument("--identity-tol", type=float, default=1e-5,
                    help="Relative tolerance for identity arm vs baseline (default 1e-5)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    import torch
    from igv.sae import SparseAutoencoder, random_sae as make_random_sae

    saes = {}
    for sae_path_str in args.sae_paths:
        sae_path = Path(sae_path_str)
        sd = torch.load(sae_path, map_location=args.device, weights_only=False)
        sae = SparseAutoencoder.from_checkpoint(sd)
        sae.eval()
        sae.to(args.device)
        sae_name = sae_path.stem
        saes[sae_name] = (sae, sae_path)

    if args.set == "skempi":
        complexes = SKEMPI_COMPLEXES
        score_name = "complex_pde"
        scan_mode = "complex_pde"
    else:
        from igv.ligand_targets import TARGETS
        complexes = list(TARGETS.keys())[:4]
        score_name = "affinity_pred_value"
        scan_mode = "affinity"

    from igv.boltz_score import (
        affinity_forward,
        confidence_forward,
        embedder_only,
        load_affinity_model,
        load_model,
        numerics_arm,
    )

    if args.set == "skempi":
        model, _ = load_model(args.checkpoint_dir, args.device)
        forward_fn = confidence_forward
    else:
        model, _ = load_affinity_model(args.checkpoint_dir, args.device)
        forward_fn = affinity_forward

    t_start = time.time()

    for sae_name, (sae, sae_path) in saes.items():
        result_file = _result_path(out_dir, sae_name, args.layer)
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)
        else:
            results = {}

        rsae = make_random_sae(sae.cfg, norm_stats=sae.norm_stats, seed=0)
        rsae.eval()
        rsae.to(args.device)

        all_pass = True

        for complex_id in complexes:
            if time.time() - t_start > args.time_budget_s:
                log.info("Time budget exhausted.")
                break

            typical_effect = load_typical_effect(complex_id, repo_root, scan_mode)
            if typical_effect is None:
                log.warning("No scan data for %s; skipping ratio computation.", complex_id)

            log.info("Evaluating %s / %s / layer=%d", sae_name, complex_id, args.layer)

            try:
                from scripts_27_helpers import featurize_and_embed
            except ImportError:
                pass

            baseline_key = _result_key(complex_id, "baseline", None)
            identity_key = _result_key(complex_id, "identity", None)
            splice_key = _result_key(complex_id, "sae_splice", None)
            eps_key = _result_key(complex_id, "eps_kept", None)
            random_key = _result_key(complex_id, "random_sae", None)
            mean_key = _result_key(complex_id, "mean_ablation", None)

            from igv.boltz_score import build_complex_feats

            try:
                from igv.dms import resolve_pdb_complex
                from igv.skempi import get_complex as skempi_get_complex
                cx = skempi_get_complex(complex_id)
                chain = cx.partner1[0]
                resolved = resolve_pdb_complex(complex_id, chain, cache_dir)
                feats, _ = build_complex_feats(
                    resolved.chains, resolved.pdb_path,
                    cache_dir / f"boltz_sae_gate/{complex_id}",
                    args.device,
                    use_msa_server=not args.no_msa_server,
                )
            except Exception:
                log.exception("Failed to featurize %s; skipping.", complex_id)
                continue

            s_inputs = embedder_only(model, feats)
            x_pred = feats["coords"].detach()

            if not _is_arm_done(results, baseline_key):
                score_0 = _run_forward(
                    model, s_inputs, feats, x_pred,
                    score_name, args.recycling_steps, None, forward_fn,
                )
                results[baseline_key] = {"score": score_0}
                log.info("  baseline: %.6f", score_0)
            score_0 = results[baseline_key]["score"]

            if not _is_arm_done(results, identity_key):
                id_cb = _make_splice_callback(None, args.layer, args.recycle, mode="identity")
                score_id = _run_forward(
                    model, s_inputs, feats, x_pred,
                    score_name, args.recycling_steps, id_cb, forward_fn,
                )
                rel_diff = abs(score_id - score_0) / max(abs(score_0), 1e-12)
                id_pass = rel_diff < args.identity_tol
                results[identity_key] = {
                    "score": score_id,
                    "delta": score_id - score_0,
                    "rel_diff": rel_diff,
                    "pass": id_pass,
                }
                if not id_pass:
                    log.error(
                        "  IDENTITY FAIL: |%.6f - %.6f| / |%.6f| = %.2e > %.2e",
                        score_id, score_0, score_0, rel_diff, args.identity_tol,
                    )
                else:
                    log.info("  identity: %.6f (rel_diff=%.2e, PASS)", score_id, rel_diff)
            id_pass = results[identity_key].get("pass", False)

            if not _is_arm_done(results, splice_key):
                cb = _make_splice_callback(sae, args.layer, args.recycle, mode="sae")
                score_splice = _run_forward(
                    model, s_inputs, feats, x_pred,
                    score_name, args.recycling_steps, cb, forward_fn,
                )
                delta = score_splice - score_0
                results[splice_key] = {
                    "score": score_splice,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "typical_effect": typical_effect,
                    "ratio": abs(delta) / typical_effect if typical_effect else None,
                }
                log.info("  sae_splice: %.6f (delta=%.6f)", score_splice, delta)

            if not _is_arm_done(results, eps_key):
                cb = _make_splice_callback(sae, args.layer, args.recycle, mode="eps_kept")
                score_eps = _run_forward(
                    model, s_inputs, feats, x_pred,
                    score_name, args.recycling_steps, cb, forward_fn,
                )
                results[eps_key] = {
                    "score": score_eps,
                    "delta": score_eps - score_0,
                    "rel_diff": abs(score_eps - score_0) / max(abs(score_0), 1e-12),
                }
                log.info("  eps_kept: %.6f (delta=%.6f)", score_eps, score_eps - score_0)

            if not _is_arm_done(results, random_key):
                cb = _make_splice_callback(rsae, args.layer, args.recycle, mode="random_sae")
                score_rand = _run_forward(
                    model, s_inputs, feats, x_pred,
                    score_name, args.recycling_steps, cb, forward_fn,
                )
                results[random_key] = {
                    "score": score_rand,
                    "delta": score_rand - score_0,
                    "abs_delta": abs(score_rand - score_0),
                }
                log.info("  random_sae: %.6f (delta=%.6f)", score_rand, score_rand - score_0)

            if not _is_arm_done(results, mean_key):
                train_mean_val = sae.norm_stats.mean.to(args.device) if sae.norm_stats else torch.zeros(sae.cfg.d_input, device=args.device)
                if train_mean_val.dim() == 1:
                    train_mean_val = train_mean_val.unsqueeze(0).unsqueeze(0)
                cb = _make_splice_callback(
                    None, args.layer, args.recycle, mode="mean_ablation",
                    train_mean=train_mean_val,
                )
                score_mean = _run_forward(
                    model, s_inputs, feats, x_pred,
                    score_name, args.recycling_steps, cb, forward_fn,
                )
                results[mean_key] = {
                    "score": score_mean,
                    "delta": score_mean - score_0,
                    "abs_delta": abs(score_mean - score_0),
                }
                log.info("  mean_ablation: %.6f (delta=%.6f)", score_mean, score_mean - score_0)

            splice_delta = results.get(splice_key, {}).get("abs_delta", float("inf"))
            passed, reason = gate_verdict(
                splice_delta, typical_effect, id_pass, args.threshold,
            )
            results[_result_key(complex_id, "verdict", None)] = {
                "pass": passed,
                "reason": reason,
                "typical_effect": typical_effect,
            }
            if not passed:
                all_pass = False
            log.info("  verdict: %s (%s)", "PASS" if passed else "FAIL", reason)

            with open(result_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
                f.write("\n")

        results["gate_pass"] = all_pass

        with open(result_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
            f.write("\n")

        prov_write(
            result_file,
            stage="29_splice_gate",
            inputs={
                "sae_path": str(sae_path),
                "sae_sha256": _file_sha256(sae_path),
            },
            params={
                "layer": args.layer,
                "recycle": args.recycle,
                "recycling_steps": args.recycling_steps,
                "threshold": args.threshold,
                "identity_tol": args.identity_tol,
                "complexes": complexes,
                "score_name": score_name,
            },
            arm={
                "stage": "29_splice_gate",
                "sae_name": sae_name,
                "layer": args.layer,
                "recycle": args.recycle,
                "feat_seed": 42,
                **numerics_arm(),
            },
            notes=f"gate_pass={all_pass}",
        )

        log.info("SAE %s gate: %s", sae_name, "PASS" if all_pass else "FAIL")

    elapsed = time.time() - t_start
    log.info("Done in %.0f s.", elapsed)


if __name__ == "__main__":
    main()
