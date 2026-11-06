#!/usr/bin/env python3
"""Gradient attribution of a Protenix confidence score w.r.t. the input restype one-hot.

Sibling of ``03_attribute.py`` (Boltz-2). Produces the same npz/json artifacts
with ``.prov.json`` sidecar so downstream metrics scripts consume them unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.attrib import CANONICAL_AMINO_ACIDS
from igv.data import build_library, download, read_pdb_chains
from igv.dms import resolve_pdb_complex
from igv.provenance import write as prov_write

log = logging.getLogger(__name__)

STRUCTURE_FOR_DATASET = {
    "4fqi_h1": "4fqi_hlab",
    "4fqi_h3": "4fqi_hlab",
}

IPTM_SCORES = {"iptm", "ptm"}


def completeness_error_from_sum(
    ig_sum: float, f_x: float, f_baseline: float,
) -> float:
    """Relative error of the completeness identity: sum(IG) vs f(x) - f(baseline).

    Mirrors :func:`igv.attrib.completeness_error`, which cannot be reused here
    because it takes an ``AttribResult`` and the Protenix path carries numpy
    arrays. Returns 0.0 when the denominator and numerator are both zero, and
    inf when only the denominator is.
    """
    diff = f_x - f_baseline
    if abs(diff) == 0.0:
        return 0.0 if abs(ig_sum) == 0.0 else float("inf")
    return abs(ig_sum - diff) / abs(diff)


def _ig_res_type(
    model, input_feature_dict, score_name, x_pred,
    baseline_name: str,
    m_steps: int,
    n_cycle: int,
    quadrature: str = "gausslegendre",
) -> tuple[np.ndarray, "torch.Tensor"]:
    """Integrated gradient of the score w.r.t. restype.

    Returns (accumulated, rt_baseline) where accumulated is a 2-D numpy array
    of shape (L_full, num_tokens) and rt_baseline is the baseline tensor used
    for the path integral.
    """
    import torch
    from igv.protenix_score import (
        build_mean_aa_res_type_baseline,
        build_shuffled_res_type_baseline,
        build_unk_res_type_baseline,
        confidence_forward,
    )

    restype_orig = input_feature_dict["restype"]
    shape = restype_orig.shape
    num_tokens = shape[-1]
    L_full = shape[-2] if restype_orig.dim() >= 2 else shape[0]

    if baseline_name == "mean_aa":
        rt_baseline = build_mean_aa_res_type_baseline(restype_orig, input_feature_dict)
    elif baseline_name == "unk":
        rt_baseline = build_unk_res_type_baseline(restype_orig, input_feature_dict)
    elif baseline_name == "shuffled":
        rt_baseline = build_shuffled_res_type_baseline(restype_orig, input_feature_dict)
    else:
        rt_baseline = torch.zeros_like(restype_orig).float()

    rt_x = restype_orig.detach().float()

    if quadrature == "gausslegendre":
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(m_steps)
        alphas_np = (gl_nodes + 1.0) / 2.0
        weights_np = gl_weights / 2.0
    else:
        alphas_np = np.linspace(0.0, 1.0, m_steps + 1)
        weights_np = np.ones(len(alphas_np)) / len(alphas_np)

    accumulated = np.zeros((L_full, num_tokens), dtype=np.float64)

    for step_i, (alpha, weight) in enumerate(zip(alphas_np, weights_np)):
        log.info("restype IG step %d/%d", step_i + 1, len(alphas_np))
        rt_interp = (rt_baseline + alpha * (rt_x - rt_baseline)).detach().requires_grad_(True)

        try:
            scalar = confidence_forward(
                model, rt_interp, input_feature_dict, x_pred, score_name,
                n_cycle=n_cycle, gradient_checkpointing=True,
            )
            scalar.backward()
        finally:
            pass

        grad = rt_interp.grad
        if grad is None:
            raise RuntimeError(
                "No gradient on restype at IG step %d/%d." % (step_i + 1, len(alphas_np))
            )
        g = grad.detach().cpu().float().numpy()
        if g.ndim == 3:
            g = g[0]
        accumulated += weight * g.astype(np.float64)
        del rt_interp, scalar

    return accumulated, rt_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradient attribution of Protenix confidence score w.r.t. restype"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--chain", default="H")
    parser.add_argument(
        "--score", required=True,
        choices=sorted(
            ["complex_pde", "complex_plddt", "pae_interface_mean", "iptm", "ptm"]
        ),
    )
    parser.add_argument("--method", required=True, choices=["plain_grad", "ig"])
    parser.add_argument("--m-steps", type=int, default=15)
    parser.add_argument("--n-cycle", type=int, default=4)
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument(
        "--out", default="results/{dataset}_{score}_{method}_protenix_grad.npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--structure", default=None)
    parser.add_argument(
        "--baseline", default="zeros",
        choices=["zeros", "mean_aa", "unk", "shuffled"],
    )
    parser.add_argument(
        "--x-pred", default="predicted", choices=["zeros", "predicted"],
    )
    parser.add_argument("--no-res-type-grad", action="store_true")
    parser.add_argument(
        "--x-pred-cache", action=argparse.BooleanOptionalAction, default=True,
        help="Cache predicted coordinates on disk (default: on).",
    )
    parser.add_argument(
        "--feature-cache", action=argparse.BooleanOptionalAction, default=True,
        help="Cache feature dict on disk for cross-process reproducibility (default: on).",
    )
    parser.add_argument(
        "--msa-dir", default=None,
        help=(
            "Directory with per-chain .a3m files. Accepts the Boltz layout "
            "(data/raw/boltz_attr/<DATASET>_<CHAIN>/). Required unless --single-seq."
        ),
    )
    parser.add_argument(
        "--msa-depth", type=int, default=None,
        help="Override MSA sampling cutoff (number of rows). Default: Protenix default (16384).",
    )
    parser.add_argument(
        "--single-seq", action="store_true",
        help="Run without MSA (single-sequence mode). Must be explicit.",
    )
    parser.add_argument(
        "--model-name", default="protenix_base_default_v1.0.0",
        help="Protenix model/weight name (default: protenix_base_default_v1.0.0, Apache-2.0).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch
    from igv.protenix_score import (
        build_features,
        canonical_aa_token_indices,
        confidence_forward,
        coord_fingerprint,
        load_model,
        numerics_arm,
        predict_structure_coords,
        record_iptm_argmax,
        res_type_gradient_forward,
        resolve_msa_paths,
    )

    cache_dir = Path(args.cache_dir)
    dataset = args.dataset
    chain = args.chain
    score = args.score
    method = args.method
    out_path = Path(args.out.format(dataset=dataset, score=score, method=method))

    # --- 1. Load library and structure ---
    try:
        resolved = resolve_pdb_complex(
            dataset, chain, cache_dir, structure_override=args.structure,
        )
    except KeyError:
        resolved = None

    if resolved is not None:
        data_source = resolved.data_source
        log.info("%s complex %s (chain=%s)", data_source.upper(), resolved.struct_name, chain)
        pdb_path = resolved.pdb_path
        reference_seq = resolved.reference_seq
        struct_name = resolved.struct_name
        chains = resolved.chains
    else:
        data_source = "abbibench"
        log.info("Building library for %s (chain=%s)", dataset, chain)
        lib = build_library(dataset, cache_dir, chain=chain)
        reference_seq = lib.reference_seq
        struct_name = args.structure or STRUCTURE_FOR_DATASET.get(dataset)
        if struct_name is None:
            raise ValueError(f"No default structure for {dataset!r}. Use --structure.")
        pdb_path = download(struct_name, "structure", cache_dir)
        pdb_chains = read_pdb_chains(pdb_path)
        chains = {}
        for ch_id, seq in pdb_chains.items():
            if ch_id == chain:
                chains[ch_id] = reference_seq
            else:
                chains[ch_id] = seq

    log.info("Complex chains: %s", {c: len(s) for c, s in chains.items()})

    # --- 2. Load model and featurise ---
    model_name = args.model_name
    log.info("Loading Protenix model (%s)", model_name)
    model, configs = load_model(
        args.device, model_name=model_name,
        weights_dir=args.weights_dir, n_cycle=args.n_cycle,
    )

    # --- 2b. Resolve MSA ---
    msa_dir_arg = args.msa_dir
    if msa_dir_arg is None and not args.single_seq:
        default_msa = cache_dir / "boltz_attr" / f"{dataset}_{chain}"
        if default_msa.is_dir():
            msa_dir_arg = str(default_msa)
            log.info("Using default Boltz MSA dir: %s", msa_dir_arg)
        else:
            raise FileNotFoundError(
                f"No --msa-dir supplied and default Boltz MSA dir {default_msa} "
                "does not exist. Either run the Boltz pipeline first, supply "
                "--msa-dir explicitly, or pass --single-seq to run without MSA."
            )
    msa_paths = resolve_msa_paths(chains, msa_dir_arg, single_seq=args.single_seq)
    msa_source = "single_seq" if msa_paths is None else str(msa_dir_arg)
    log.info("MSA source: %s (depth=%s)", msa_source, args.msa_depth or "default")

    feat_cache = cache_dir / "protenix_attr" / f"{dataset}_{chain}"
    log.info("Building Protenix features (cache=%s, feature_cache=%s)", feat_cache, args.feature_cache)
    input_feature_dict, token_map = build_features(
        chains, feat_cache, args.device, configs, model=model,
        msa_paths=msa_paths, msa_depth=args.msa_depth,
        use_feature_cache=args.feature_cache, model_name=model_name,
        feature_cache_dir=cache_dir / "protenix_features",
    )
    _ref_pos_fp = coord_fingerprint(input_feature_dict["ref_pos"])
    log.info("ref_pos fingerprint: %s", _ref_pos_fp)

    # --- 3. Obtain x_pred ---
    if args.x_pred == "predicted":
        cache_kw = (
            dict(cache_dir=cache_dir, chains=chains, dataset=dataset)
            if args.x_pred_cache else {}
        )
        x_pred = predict_structure_coords(
            model, input_feature_dict, configs, args.device, **cache_kw,
        )
    else:
        n_atoms = input_feature_dict["ref_pos"].shape[-2]
        x_pred = torch.zeros(1, 1, n_atoms, 3, device=args.device)

    _coords_max = float(x_pred.abs().max())
    _x_pred_fp = coord_fingerprint(x_pred)
    log.info("x_pred (mode=%s), coords.abs().max()=%g", args.x_pred, _coords_max)
    log.info("x_pred fingerprint: %s", _x_pred_fp)

    # --- 4. Res-type gradient ---
    token_indices = [token_map[(chain, i)] for i in range(len(reference_seq))]
    res_type_extras: dict = {}
    res_type_grad_enabled = not args.no_res_type_grad

    if res_type_grad_enabled:
        log.info("Computing restype gradient (method=%s)", method)
        t0_rt = time.time()
        aa_token_indices = canonical_aa_token_indices()
        aa_to_idx = {aa: i for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}
        wt_indices = np.array(
            [aa_to_idx.get(c, 0) for c in reference_seq], dtype=np.intp,
        )

        if method == "plain_grad":
            rt_result = res_type_gradient_forward(
                model, input_feature_dict, score, x_pred=x_pred,
                n_cycle=args.n_cycle, gradient_checkpointing=True,
            )
            grad_res_type_full = rt_result["grad_res_type"]
        elif method == "ig":
            grad_res_type_full, rt_baseline_tensor = _ig_res_type(
                model, input_feature_dict, score, x_pred,
                baseline_name=args.baseline, m_steps=args.m_steps,
                n_cycle=args.n_cycle,
            )

        grad_res_type_chain = grad_res_type_full[token_indices, :]

        res_type_extras = {
            "grad_res_type": grad_res_type_chain.astype(np.float32),
            "aa_token_indices": aa_token_indices,
            "wt_indices": wt_indices,
        }
        elapsed_rt = time.time() - t0_rt
        log.info("restype gradient: shape %s in %.1f s", grad_res_type_chain.shape, elapsed_rt)

    # --- 5. f_x and completeness ---
    with torch.no_grad():
        f_x_rt = input_feature_dict["restype"].detach().float()
        f_x_val = float(confidence_forward(
            model, f_x_rt, input_feature_dict, x_pred, score,
            n_cycle=args.n_cycle, gradient_checkpointing=False,
        ))
    log.info("f(x) = %.6f", f_x_val)

    f_baseline_val = float("nan")
    comp_err = float("nan")
    ig_sum_val = float("nan")

    if method == "ig" and res_type_grad_enabled:
        with torch.no_grad():
            f_baseline_val = float(confidence_forward(
                model, rt_baseline_tensor.detach().float(),
                input_feature_dict, x_pred, score,
                n_cycle=args.n_cycle, gradient_checkpointing=False,
            ))
        log.info("f(baseline) = %.6f", f_baseline_val)

        rt_x_np = input_feature_dict["restype"].detach().cpu().float().numpy()
        if rt_x_np.ndim == 3:
            rt_x_np = rt_x_np[0]
        rt_bl_np = rt_baseline_tensor.detach().cpu().float().numpy()
        if rt_bl_np.ndim == 3:
            rt_bl_np = rt_bl_np[0]
        ig_attribution = (rt_x_np.astype(np.float64) - rt_bl_np.astype(np.float64)) * grad_res_type_full
        ig_sum_val = float(ig_attribution.sum())

        comp_err = completeness_error_from_sum(
            ig_sum_val, f_x_val, f_baseline_val,
        )

        log.info(
            "Completeness error: %.4f (%.2f%%) — %s 5%% criterion",
            comp_err, comp_err * 100,
            "PASSES" if comp_err < 0.05 else "FAILS",
        )

    # --- 6. ipTM argmax ---
    argmax_record: dict = {}
    if score in IPTM_SCORES:
        token_to_chain: dict[int, str] = {}
        for (ch_id, resi), tok_idx in token_map.items():
            token_to_chain[tok_idx] = ch_id
        with torch.no_grad():
            s_inputs = model.input_embedder(input_feature_dict, inplace_safe=False)
            s, z = _run_trunk_nograd(model, input_feature_dict, args.n_cycle)
            plddt_l, pae_l, pde_l, _ = model.confidence_head(
                input_feature_dict, s_inputs=s_inputs, s_trunk=s, z_trunk=z,
                pair_mask=None, x_pred_coords=x_pred,
                use_embedding=True, inplace_safe=False,
            )
        argmax_record = record_iptm_argmax(pae_l, input_feature_dict, token_to_chain)
        log.info("ipTM argmax: token=%s chain=%s", argmax_record.get("argmax_token"),
                 argmax_record.get("argmax_chain"))
        del s, z, plddt_l, pae_l, pde_l

    # --- 7. Save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "token_indices": np.array(token_indices, dtype=np.int64),
        "f_x": np.float64(f_x_val),
        "f_baseline": np.float64(f_baseline_val),
        "completeness_error": np.float64(comp_err),
        "ig_sum": np.float64(ig_sum_val),
        "reference_seq": np.array(reference_seq),
        "chain": np.array(chain),
        "score": np.array(score),
        "method": np.array(method),
        "m_steps": np.int64(args.m_steps),
        "x_pred_mode": np.array(args.x_pred),
        "model": np.array("protenix"),
    }
    if argmax_record:
        for k, v in argmax_record.items():
            save_dict[f"argmax_{k}" if not k.startswith("argmax_") else k] = (
                np.array(v) if v is not None else np.array(None)
            )
    save_dict.update(res_type_extras)
    np.savez(out_path, **save_dict)
    log.info("Wrote %s", out_path)

    # --- 8. Provenance ---
    prov_write(
        out_path,
        stage="03_attribute_protenix",
        inputs={
            "dataset": dataset,
            "structure": struct_name,
            "cache_dir": str(cache_dir),
            "data_source": data_source,
        },
        params={
            "chain": chain,
            "score": score,
            "method": method,
            "m_steps": args.m_steps,
            "n_cycle": args.n_cycle,
            "baseline": args.baseline,
            "res_type_grad": res_type_grad_enabled,
            "x_pred_mode": args.x_pred,
            "x_pred_cache": args.x_pred_cache,
            "feature_cache": args.feature_cache,
            "msa_source": msa_source,
            "msa_depth": args.msa_depth,
            "single_seq": args.single_seq,
        },
        arm={
            "model": "protenix",
            "model_name": model_name,
            "score": score,
            "method": method,
            "trunk": "full",
            "m_steps": args.m_steps if method == "ig" else 1,
            "n_cycle": args.n_cycle,
            "dataset": dataset,
            "chain": chain,
            "x_pred_mode": args.x_pred,
            "msa_source": msa_source,
            "msa_depth": args.msa_depth,
            **numerics_arm(),
            "x_pred_cache": args.x_pred_cache,
            "feature_cache": args.feature_cache,
            "x_pred_fingerprint": _x_pred_fp,
            "ref_pos_fingerprint": _ref_pos_fp,
            "coords_abs_max": _coords_max,
            "f_x": f_x_val,
            "f_baseline": f_baseline_val,
            "completeness_error": comp_err,
            "ig_sum": ig_sum_val,
        },
    )
    log.info("Done.")


def _run_trunk_nograd(model, input_feature_dict, n_cycle):
    """Run the trunk under no_grad for diagnostics."""
    import torch

    with torch.no_grad():
        _, s, z = model.get_pairformer_output(
            input_feature_dict, N_cycle=n_cycle, inplace_safe=False,
        )
    return s.detach(), z.detach()


if __name__ == "__main__":
    main()
