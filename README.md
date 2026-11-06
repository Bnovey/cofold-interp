# cofold-interp — can one gradient replace a mutation scan?

Antibodies bind to targets, and drug designers want to know which mutations at
the binding interface make that binding stronger or weaker.

The usual way to answer this with a model like Boltz-2 is brute force: change
one amino acid, run the model, write down the score, repeat a few thousand
times. That costs roughly **615 GPU-hours** for the datasets here.

This repo tests a shortcut. Run the model **once**, and instead of just reading
the score, ask the model how the score would change if you nudged each part of
the input. That is a gradient, and in principle it predicts every mutation at
once from a single pass. The technique is
[Integrated Gradients](https://arxiv.org/abs/1703.01365).

**Does the shortcut actually work?** That is the whole question. This is a
validation project, not a tool — the answer may well be no.

## The two validation arms

Ground truth comes from two sources:

- **Alanine scanning (SKEMPI arm).**
  [SKEMPI 2.0](https://life.bsc.es/pid/skempi2/) provides 7,085 measured ddG
  values across 348 complexes. The primary target is **1JTG** (TEM-1 / BLIP,
  28 measured positions on chain B, 11 hot spots); **3HFM** (HyHEL-10 /
  lysozyme) is the second complex, with 1VFB, 1JRH, and 2JEL available for
  pooling. This arm asks which *positions* matter for binding.

- **Saturation (DMS arm).**
  [Starr et al. 2020](https://doi.org/10.1016/j.cell.2020.08.012) measured
  binding of the SARS-CoV-2 RBD to ACE2 for all 20 amino acids at essentially
  every position (3,802 single mutants on 6M0J). This arm asks how each of the
  20 *amino acids* performs at each position, using within-position comparisons
  that eliminate the burial and residue-volume confounds that plague alanine
  scanning. The matched expression readout provides a folding-vs-binding
  control.

### The three comparisons

| | What we compare | What it tells us |
|---|---|---|
| **shortcut vs slow way** | gradient shortcut vs the model's own brute-force scan | Does the shortcut match the model it came from? |
| **model vs lab** | the model's scan vs real lab measurements | Is the model right about biology at all? |
| **shortcut vs lab** | gradient shortcut vs real lab measurements | What would someone actually get from this? |

The first is the honest test of the shortcut. The second is a property of
Boltz-2 and is already known to be weak (Spearman ~0.13, below ProteinMPNN's
0.30). If it is near zero, the third cannot be good no matter how well the
shortcut works — so the three have to be read together.

## Status

**The memory problem is solved.** Getting the gradient requires a backward pass
through the whole model, which did not fit on an 80 GB GPU at the size we need
(730 tokens). Two settings fix it:

```bash
IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16
```

That brings the requirement from 93.7 GB down to a measured **55.2 GB**, and a
single backward pass at 730 tokens takes 104 seconds. Details and all
measurements: [docs/MEMSCALE_RESULTS.md](docs/MEMSCALE_RESULTS.md).

**The correctness problem is resolved.** The Integrated Gradients completeness
check (attributions must sum to the total score change) failed at m=16
integration steps. The cause was quadrature resolution, not half precision: fp32
fails *worse* than bf16 at the same step count (rel err 1.7205 vs 1.1709). At
m=32, completeness passes with rel err **0.0442** on L=554. bf16 is exonerated.

**The `random_weights` sanity check passes.** Randomising all 5,035 parameter
tensors destroys the attribution (Spearman 0.077 against a 0.3 threshold), so
the gradient depends on what Boltz-2 learned rather than on input geometry.

**Preliminary result on 1JTG chain B (SKEMPI arm).** One measured run at
427 tokens, fp32, m=32, `mean_aa` baseline:

| run | Spearman vs ddG | null p95 | above null | partial (vs burial) | AUROC |
|---|---|---|---|---|---|
| plain_grad (zeros) | 0.306 | 0.318 | no | 0.180 | 0.685 |
| IG m=32 (zeros) | 0.238 | 0.336 | no | 0.010 | 0.583 |
| **IG m=32 (mean_aa)** | **0.357** | 0.313 | **yes** | 0.211 | **0.695** |

The `mean_aa` baseline was the cause of the early nulls — the all-zeros
baseline scores 11.43 where the real complex scores 2.62, pushing the integral
off-manifold. Do not oversell this: CI is [-0.002, 0.656], excluding zero by
two thousandths. Partial correlation is 0.211 against burial's 0.436. This is
one unreplicated run on 28 effective positions.

**No end-to-end result yet.** The pipeline has never produced a shortcut-vs-scan
number: the brute-force scan has not been run, so "shortcut vs slow way" — the
headline claim — has no number. The saturation arm (Starr 2020 / 6M0J) is wired
and tested but has not run on GPU. The next steps are:

1. Pool 3HFM / 1VFB / 1JRH / 2JEL to test whether the 0.357 replicates at
   larger n (~$12, ~2.5 h).
2. Run the 1JTG brute-force scan (532 mutants, ~$13) to get the first
   shortcut-vs-scan number.
3. Run the saturation arm on 6M0J (one IG run covers all positions; 399
   forward passes for the brute-force comparison on the 21 interface positions).

## Repo layout

```
src/igv/              Core library
  attrib.py             Integrated Gradients and plain-gradient attribution
  boltz_score.py        Boltz-2 scoring wrapper with memory settings
  data.py               Dataset loading (AbBiBench, PDB)
  dms.py                DMS / saturation dataset loading (Starr 2020)
  skempi.py             SKEMPI 2.0 loading and hot-spot extraction
  deterministic.py      Patches for featurisation determinism
  gpu.py                VRAM queries and CPU-degradation seam
  metrics.py            Spearman, AUROC, bootstrap CIs
  provenance.py         Artifact provenance sidecars
scripts/
  00_fetch_data.py      Download raw data from AbBiBench / HuggingFace
  01_build_library.py   Parse mutant libraries
  02_embed_deltas.py    Compute embedding deltas E_mut - E_ref  [GPU]
  03_attribute.py       Gradient attribution (IG or plain_grad)  [GPU]
  04_scan.py            Brute-force mutation scan  [GPU]
  05_predict.py         Predict scores from gradient * delta
  06_metrics.py         Evaluation metrics
  07_sanity.py          Sanity checks (completeness, random_weights)  [GPU]
  08_memscale.py        VRAM measurement sweep  [GPU, diagnostic]
  09_path_profile.py    Gradient fidelity along the IG path  [GPU, diagnostic]
  10_skempi_hotspots.py SKEMPI hot-spot ground truth + confound panel
  11_within_position.py Within-position scoring for the DMS arm
  12_compare.py         Three-way comparison (shortcut vs scan vs lab)
  run_all.sh            Full pipeline runner (AbBiBench datasets)
  run_complex.sh        Runner for SKEMPI and DMS arms
  cloud/                GCP provisioning and sync scripts
tests/                549 tests, no GPU needed
docker/Dockerfile     Reproducible GPU environment
docs/                 Pipeline reference, memory results, cloud how-to
```

## Setup

CPU, on a laptop:

```bash
pip install -e ".[dev]"    # or: make setup
make test                  # 549 tests, no GPU needed
make fetch library         # download the data, build the mutant list
```

GPU (stages 02, 03, 04, 07 and the `08_memscale` diagnostic):

```bash
docker build -t igv -f docker/Dockerfile .
docker run --gpus all --shm-size=32g --ipc=host \
  -e IGV_TRI_ATTN_CKPT=1 -e IGV_AUTOCAST=bf16 \
  -v $HOME/boltz_cache:/root/.boltz -v $(pwd):/app -w /app igv \
  bash scripts/run_all.sh
```

`make help` lists every stage. `--dry-run` works on most of them.

### Running the SKEMPI and DMS arms

`run_all.sh` covers the AbBiBench datasets. For SKEMPI and DMS complexes, use
`run_complex.sh`:

```bash
# SKEMPI arm
DATASET=1JTG CHAIN=B POSITIONS=skempi bash scripts/run_complex.sh

# DMS / saturation arm
DATASET=spike_rbd CHAIN=E POSITIONS=dms_interface bash scripts/run_complex.sh
```

Add `--dry-run` to print every command without executing. Environment overrides:
`SCORE` (default `complex_pde`), `METHOD` (default `ig`), `M_STEPS` (32),
`BASELINE` (`mean_aa`), `POSITIONS`, `INTERFACE_CUTOFF` (5.0).

### GPU requirements

One 80 GB GPU is enough with `IGV_TRI_ATTN_CKPT=1 IGV_AUTOCAST=bf16`. Without
them you need about 94 GB, which no single card of that class has. See
[docs/CLOUD.md](docs/CLOUD.md) for how to rent one on GCP.

## Two warnings

**Do not add packages.** Only `torch==2.7.1+cu126` and `boltz==2.2.1` work.
Installing `boltzgen`, `protenix`, `chai_lab`, `gnina`, or `cuequivariance`
quietly replaces the pinned CUDA build of torch and then segfaults on A100.
See [constraints.txt](constraints.txt).

**Scores from `IGV_AUTOCAST=bf16` runs cannot be compared with older ones.**
Half precision shifts the numbers slightly, so any reference score has to be
recomputed under the same setting. Every artifact records which settings
produced it.

## Docs

- [docs/PIPELINE.md](docs/PIPELINE.md) — the stages, what each one writes, and how provenance works
- [docs/MEMSCALE_RESULTS.md](docs/MEMSCALE_RESULTS.md) — measured memory requirements, correctness checks, and the completeness failure
- [docs/CLOUD.md](docs/CLOUD.md) — renting a suitable GPU on GCP
