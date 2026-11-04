#!/usr/bin/env python3
"""Train a Matryoshka BatchTopK SAE on cached Boltz-2 activations (E0)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from igv.provenance import write as prov_write
from igv.sae import (
    SAEConfig,
    SparseAutoencoder,
    TokenDataset,
    build_token_dataset,
    compute_norm_stats,
    cross_seed_overlap,
    dead_fraction,
    fraction_variance_explained,
    l0,
    normalised_mse,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_seed(
    dataset: TokenDataset,
    cfg: SAEConfig,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    out_path: Path,
    eval_every: int = 500,
    save_every: int = 1000,
    dead_token_threshold: int = 10_000_000,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_stats = compute_norm_stats(dataset.train)
    sae = SparseAutoencoder(cfg, norm_stats=norm_stats).to(device)

    with torch.no_grad():
        sae.decoder.bias.copy_(
            norm_stats.apply(dataset.train.mean(dim=0)).to(device)
        )

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    start_step = 0
    ckpt_path = out_path.with_suffix(".ckpt.pt")
    if ckpt_path.exists():
        log.info("Resuming from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        sae.steps_since_fired.copy_(ckpt["steps_since_fired"])
        torch.set_rng_state(ckpt["rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])

    train_data = dataset.train.to(device)
    val_data = dataset.val.to(device)
    n_train = len(train_data)

    eval_log: list[dict] = []
    t0 = time.time()

    for step in range(start_step, steps):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        batch = train_data[idx]

        out = sae(batch)
        loss = out.losses["recon"]
        if out.auxk_loss is not None:
            loss = loss + cfg.auxk_coeff * out.auxk_loss

        optimizer.zero_grad()
        loss.backward()
        sae.project_decoder_grads()
        optimizer.step()
        sae.normalise_decoder()

        if (step + 1) % eval_every == 0 or step == steps - 1:
            sae.eval()
            with torch.no_grad():
                val_out = sae(val_data, return_aux=False)
                val_x_hat = val_out.x_hat
                val_x_norm = sae.normalise(val_data)
                entry = {
                    "step": step + 1,
                    "train_loss": loss.item(),
                    "val_nmse": normalised_mse(val_x_norm, val_x_hat),
                    "val_fve": fraction_variance_explained(val_x_norm, val_x_hat),
                    "val_l0": l0(val_out.f),
                    "dead_frac": dead_fraction(
                        sae.steps_since_fired, cfg.dead_threshold_steps
                    ),
                    "elapsed_s": time.time() - t0,
                }
            eval_log.append(entry)
            log.info(
                "step=%d loss=%.4f val_nmse=%.4f val_fve=%.4f l0=%.1f dead=%.3f",
                entry["step"],
                entry["train_loss"],
                entry["val_nmse"],
                entry["val_fve"],
                entry["val_l0"],
                entry["dead_frac"],
            )
            sae.train()

        if (step + 1) % save_every == 0:
            _save_checkpoint(sae, optimizer, step + 1, ckpt_path)

    sae.eval()
    sae.estimate_inference_threshold(train_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae.full_state_dict(), out_path)
    if ckpt_path.exists():
        ckpt_path.unlink()
    log.info("Saved %s", out_path)

    return {
        "seed": seed,
        "final": eval_log[-1] if eval_log else {},
        "eval_log": eval_log,
    }


def _save_checkpoint(sae, optimizer, step, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": sae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "steps_since_fired": sae.steps_since_fired.clone(),
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--input", choices=["s", "s+pair"], default="s")
    parser.add_argument("--n-latents", type=int, default=4096)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--mode", choices=["topk", "batchtopk"], default="batchtopk")
    parser.add_argument(
        "--matryoshka-sizes",
        type=int,
        nargs="*",
        default=[256, 1024, 4096],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--auxk-coeff", type=float, default=1.0 / 32)
    parser.add_argument("--auxk-k", type=int, default=256)
    parser.add_argument("--dead-threshold-steps", type=int, default=10000)
    parser.add_argument(
        "--out", type=str, default="results/sae/pilot",
        help="Output directory; checkpoints saved as <out>/seed<k>.pt",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    d_input = 384 if args.input == "s" else 512
    cfg = SAEConfig(
        d_input=d_input,
        n_latents=args.n_latents,
        k=args.k,
        mode=args.mode,
        matryoshka_sizes=tuple(args.matryoshka_sizes),
        auxk_coeff=args.auxk_coeff,
        auxk_k=args.auxk_k,
        dead_threshold_steps=args.dead_threshold_steps,
    )

    log.info("Loading dataset from %s layer=%d input=%s", args.cache_dir, args.layer, args.input)
    dataset = build_token_dataset(
        args.cache_dir, layer=args.layer, input_mode=args.input,
    )
    log.info(
        "Train tokens: %d (%d complexes), Val tokens: %d (%d complexes)",
        len(dataset.train),
        len(dataset.train_complex_ids),
        len(dataset.val),
        len(dataset.val_complex_ids),
    )

    out_dir = Path(args.out)
    all_results = {}
    checkpoint_paths = []

    for seed in args.seeds:
        out_path = out_dir / f"seed{seed}.pt"
        result = train_one_seed(
            dataset,
            cfg,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            out_path=out_path,
            eval_every=args.eval_every,
            save_every=args.save_every,
        )
        all_results[seed] = result
        checkpoint_paths.append(out_path)

    overlap = {}
    if len(checkpoint_paths) >= 2:
        decoders = {}
        for p in checkpoint_paths:
            sd = torch.load(p, map_location="cpu", weights_only=False)
            sae_tmp = SparseAutoencoder.from_checkpoint(sd)
            seed_label = p.stem
            decoders[seed_label] = sae_tmp.decoder.weight.data
        keys = sorted(decoders)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                frac = cross_seed_overlap(decoders[a], decoders[b])
                overlap[f"{a}_vs_{b}"] = round(frac, 4)
                log.info("Overlap %s vs %s: %.4f", a, b, frac)

    metrics = {
        "cfg": {
            "d_input": cfg.d_input,
            "n_latents": cfg.n_latents,
            "k": cfg.k,
            "mode": cfg.mode,
            "matryoshka_sizes": list(cfg.matryoshka_sizes),
            "auxk_coeff": cfg.auxk_coeff,
        },
        "n_train_tokens": len(dataset.train),
        "n_val_tokens": len(dataset.val),
        "n_train_complexes": len(dataset.train_complex_ids),
        "n_val_complexes": len(dataset.val_complex_ids),
        "per_seed": {
            str(s): r.get("final", {}) for s, r in all_results.items()
        },
        "cross_seed_overlap_gt07": overlap,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
        f.write("\n")

    prov_write(
        metrics_path,
        stage="28_train_sae",
        inputs={"cache_dir": str(args.cache_dir), "layer": args.layer},
        params={
            "input": args.input,
            "n_latents": args.n_latents,
            "k": args.k,
            "mode": args.mode,
            "matryoshka_sizes": list(cfg.matryoshka_sizes),
            "seeds": args.seeds,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "auxk_coeff": args.auxk_coeff,
        },
        arm={"stage": "E0_sae", "input": args.input, "layer": args.layer},
        notes="Pilot SAE training (E0)",
    )

    for ckpt_p in checkpoint_paths:
        prov_write(
            ckpt_p,
            stage="28_train_sae",
            inputs={"cache_dir": str(args.cache_dir), "layer": args.layer},
            params={
                "input": args.input,
                "n_latents": args.n_latents,
                "k": args.k,
                "seed": int(ckpt_p.stem.replace("seed", "")),
                "steps": args.steps,
                "lr": args.lr,
            },
            arm={"stage": "E0_sae", "input": args.input, "layer": args.layer},
        )

    log.info("Done. Metrics at %s", metrics_path)


if __name__ == "__main__":
    main()
