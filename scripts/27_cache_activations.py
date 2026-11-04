#!/usr/bin/env python3
"""Stage 27 -- cache Boltz-2 trunk activations for SAE training.

Runs the trunk under torch.no_grad() through the same code path the scans
use and writes per-complex shards containing:
  - s (L, 384) fp16 at each requested layer
  - PairSAE-style SVD pair summary (L, 2*k) fp16
  - per-token metadata (chain_id, residue_index, mol_type)

Resumable: skips complexes whose npz + prov already exist.
Stops cleanly before the VM's hard 7200 s auto-stop via --time-budget-s.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.provenance import write as prov_write  # noqa: E402
from igv.provenance import sidecar_path  # noqa: E402

log = logging.getLogger("cache_activations")

SKEMPI_COMPLEXES = ["1JTG", "3S9D", "3HFM", "2JEL"]

LIGAND_TARGETS = [
    "ABL1-imatinib",
    "Mpro-nirmatrelvir",
    "EGFR-erlotinib",
    "PimA-GDD",
]

PAIR_SUMMARY_VERSION = "nmode_svd_v1"


def _build_skempi_manifest(cache_dir: Path):
    """Build manifest entries for the 4 SKEMPI complexes (WT only).

    Mutant activations from library parquets would require featurising each
    mutant. For the initial cache, we capture WT only; mutants can be added
    by extending this function to read the scan CSVs.
    """
    from igv.data import read_pdb_chains
    from igv.dms import resolve_pdb_complex

    entries = []
    for dataset in SKEMPI_COMPLEXES:
        chain = _default_chain_for_skempi(dataset)
        try:
            resolved = resolve_pdb_complex(dataset, chain, cache_dir)
        except (KeyError, FileNotFoundError) as exc:
            log.warning("Skipping %s: %s", dataset, exc)
            continue
        entries.append({
            "complex_id": f"{dataset}_WT",
            "kind": "skempi",
            "featurize": "complex",
            "chains": resolved.chains,
            "pdb_path": resolved.pdb_path,
            "dataset": dataset,
            "chain": chain,
        })
    return entries


def _default_chain_for_skempi(dataset: str) -> str:
    from igv.skempi import get_complex as skempi_get_complex
    cx = skempi_get_complex(dataset)
    return cx.partner1[0]


def _build_ligand_manifest(cache_dir: Path):
    """Build manifest entries for protein-ligand targets."""
    from igv.ligand_targets import TARGETS

    entries = []
    for name in LIGAND_TARGETS:
        if name not in TARGETS:
            log.warning("Ligand target %s not in registry; skipping", name)
            continue
        target = TARGETS[name]
        entries.append({
            "complex_id": name,
            "kind": "ligand",
            "featurize": "affinity",
            "target": target,
        })
    return entries


def _featurize_complex(entry, cache_dir, device, use_msa_server, feat_seed=None):
    """Featurize a single complex and return (feats, model_loader_name).

    Returns (feats, "complex") or (feats, "affinity").
    """
    from igv.boltz_score import build_complex_feats, build_affinity_feats

    if entry["featurize"] == "complex":
        msa_by_chain = _resolve_msa(
            cache_dir, entry["dataset"], entry["chain"], entry["chains"],
        )
        feats, _ = build_complex_feats(
            entry["chains"],
            entry["pdb_path"],
            cache_dir / f"boltz_sae_cache/{entry['complex_id']}",
            device,
            use_msa_server=use_msa_server and msa_by_chain is None,
            msa=msa_by_chain,
            feat_seed=feat_seed,
        )
        return feats, "complex"
    else:
        target = entry["target"]
        from igv.data import read_pdb_chains, download_rcsb
        pdb_path = download_rcsb(target.pdb_id, cache_dir)
        pdb_chains = read_pdb_chains(pdb_path)
        protein_chains = {
            c: pdb_chains[c] for c in target.chains if c in pdb_chains
        }
        feats, _ = build_affinity_feats(
            protein_chains,
            target.ligand_ccd,
            cache_dir / f"boltz_sae_cache/{entry['complex_id']}",
            device,
            ligand_ccd=target.ligand_ccd,
            use_msa_server=use_msa_server,
            structure_pdb=pdb_path,
            resnum_offset=target.resnum_offset,
            feat_seed=feat_seed,
        )
        return feats, "affinity"


def _resolve_msa(cache_dir, dataset, chain, chains):
    """Try to find cached MSA from stage 03."""
    msa_dir = cache_dir / "boltz_attr" / f"{dataset}_{chain}" / "msa"
    if not msa_dir.is_dir():
        return None

    msa_files = sorted(msa_dir.glob("input_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    chain_ids = list(chains.keys())
    if len(msa_files) != len(chain_ids):
        return None
    return dict(zip(chain_ids, msa_files))


def _is_complete(out_dir: Path, complex_id: str) -> bool:
    """Check if a shard and its provenance sidecar already exist."""
    npz = out_dir / f"{complex_id}.npz"
    prov = sidecar_path(npz)
    return npz.exists() and prov.exists()


def _peak_vram_mib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _reset_vram_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--set", default="skempi", choices=["skempi", "ligand", "all"],
        help="Which complex set to cache (default: skempi).",
    )
    p.add_argument(
        "--layers", nargs="+", type=int, default=[32, 63],
        help="Pairformer layer indices to capture (0-based, default: 32 63).",
    )
    p.add_argument(
        "--recycle", default="final",
        help="Recycle iteration to capture ('final' or integer, default: final).",
    )
    p.add_argument("--recycling-steps", type=int, default=1)
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/sae_cache/{set}")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    p.add_argument("--no-msa-server", action="store_true")
    p.add_argument(
        "--time-budget-s", type=int, default=6600,
        help="Stop cleanly after this many seconds (default 6600, before 7200 s VM auto-stop).",
    )
    p.add_argument("--feat-seed", type=int, default=42,
                    help="RNG seed for deterministic featurisation (default 42).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir.format(**{"set": args.set}))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recycle == "final":
        target_recycle = args.recycling_steps
    else:
        target_recycle = int(args.recycle)

    manifest = []
    if args.set in ("skempi", "all"):
        manifest.extend(_build_skempi_manifest(cache_dir))
    if args.set in ("ligand", "all"):
        manifest.extend(_build_ligand_manifest(cache_dir))

    todo = [e for e in manifest if not _is_complete(out_dir, e["complex_id"])]
    done_count = len(manifest) - len(todo)
    log.info(
        "Manifest: %d complexes, %d already cached, %d remaining.",
        len(manifest), done_count, len(todo),
    )

    if args.dry_run:
        for e in todo:
            log.info("  would cache: %s (%s)", e["complex_id"], e["kind"])
        return

    import torch
    torch.manual_seed(args.feat_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.feat_seed)

    from igv.activations import (
        CapturedActivations,
        extract_token_metadata,
        make_capture_callback,
        write_shard,
    )
    from igv.boltz_score import (
        affinity_forward,
        confidence_forward,
        embedder_only,
        load_affinity_model,
        load_model,
        numerics_arm,
        predict_structure_coords,
    )

    need_confidence = any(e["featurize"] == "complex" for e in todo)
    need_affinity = any(e["featurize"] == "affinity" for e in todo)

    model_conf = None
    model_aff = None
    if need_confidence:
        model_conf, _ = load_model(args.checkpoint_dir, args.device)
    if need_affinity:
        model_aff, _ = load_affinity_model(args.checkpoint_dir, args.device)

    t_start = time.time()
    n_cached = 0

    for entry in todo:
        elapsed = time.time() - t_start
        if elapsed > args.time_budget_s:
            remaining = len(todo) - n_cached
            log.info(
                "Time budget exhausted (%.0f s > %d s). %d complexes remain.",
                elapsed, args.time_budget_s, remaining,
            )
            break

        complex_id = entry["complex_id"]
        log.info("Caching %s ...", complex_id)
        t0 = time.time()
        _reset_vram_peak()

        try:
            feats, feat_kind = _featurize_complex(
                entry, cache_dir, args.device, not args.no_msa_server,
                feat_seed=args.feat_seed,
            )
        except Exception:
            log.exception("Failed to featurize %s; skipping.", complex_id)
            continue

        model = model_aff if feat_kind == "affinity" else model_conf

        callback, captured = make_capture_callback(
            layers=args.layers,
            target_recycle=target_recycle,
        )

        with torch.no_grad():
            s_inputs = embedder_only(model, feats)
            x_pred = feats["coords"].detach()
            if feat_kind == "affinity":
                affinity_forward(
                    model, s_inputs, feats, x_pred,
                    score_name="affinity_pred_value",
                    gradient_checkpointing=False,
                    recycling_steps=args.recycling_steps,
                    activation_callback=callback,
                )
            else:
                confidence_forward(
                    model, s_inputs, feats, x_pred,
                    score_name="complex_pde",
                    gradient_checkpointing=False,
                    recycling_steps=args.recycling_steps,
                    activation_callback=callback,
                )

        metadata = extract_token_metadata(feats)
        wall_time = time.time() - t0
        peak_vram = _peak_vram_mib()

        npz_path = out_dir / f"{complex_id}.npz"
        write_shard(npz_path, captured, metadata)

        prov_write(
            npz_path,
            stage="27_cache_activations",
            inputs={"complex_id": complex_id, "kind": entry["kind"]},
            params={
                "layers": args.layers,
                "recycle": target_recycle,
                "recycling_steps": args.recycling_steps,
                "pair_summary_recipe": PAIR_SUMMARY_VERSION,
            },
            arm={
                "stage": "27_cache_activations",
                "featurize": feat_kind,
                "feat_seed": args.feat_seed,
                **numerics_arm(),
            },
            notes=(
                f"wall_time_s={wall_time:.1f}, "
                f"peak_vram_mib={peak_vram:.0f}"
            ),
        )

        n_cached += 1
        log.info(
            "  %s: %.1f s, %.0f MiB VRAM peak, %d layers captured.",
            complex_id, wall_time, peak_vram, len(captured.s),
        )

    remaining = len(todo) - n_cached
    elapsed = time.time() - t_start
    log.info(
        "Done. Cached %d complexes in %.0f s. %d remain.",
        n_cached, elapsed, remaining,
    )


if __name__ == "__main__":
    main()
