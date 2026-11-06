# Overview — what this project is, and how to run it without repeating our mistakes

Written 2026-10-15 for a collaborator joining the project. Everything here is
measured unless marked `[UNVERIFIED]`. Detail lives in `notes/logging.md`,
which is the research log and the authority; this document is the map.

---

## 1. What this is, and what it has found

Boltz-2 predicts protein structure and binding. If you want to know which
residues matter for an interface, the honest method is brute force: mutate one
residue, re-run the model, record the change, repeat a few hundred times. The
shortcut is **gradient attribution** — run the model once and ask its own
derivatives which inputs the score was sensitive to. This repo asks whether the
shortcut tells the truth.

Three comparisons, defined in `notes/logging.md` §1 and used throughout:

| | compares | asks | result (4 complexes) | clears own null |
|---|---|---|---|---|
| **A–B** shortcut vs slow way | attribution vs the model's own scan | is attribution faithful to the model? | **+0.518** | **4 / 4** |
| **B–C** model vs lab | the model's scan vs SKEMPI ddG | is the model right about biology? | +0.235, p=0.19 | 2 / 4 |
| **A–C** shortcut vs lab | attribution vs SKEMPI ddG | what does a practitioner get? | +0.265, p=0.17 | 2 / 4 |

**The shortcut reproduces the model far better than the model reproduces
nature.** Attribution is not the weak link.

Per complex (resswap scan, IG at the best available rung):

| complex | chain | tokens | SKEMPI pos | n (A–B) | A–B | partial vs distance | n (B–C) | B–C | A–C (IG) |
|---|---|---|---|---|---|---|---|---|---|
| 1JTG | B | 427 | 28 | 532 | **+0.6678** (m=64) | +0.6653 | 30 | **+0.4376** | **+0.5591** |
| 3S9D | A | 307 | 50 | 950 | **+0.5937** (m=64) | +0.5722 | 54 | **+0.4205** | **+0.4678** |
| 3HFM | Y | 558 | 15 | 285 | +0.4054 (m=32) | +0.4043 | 42 | +0.2530 | −0.0049 |
| 2JEL | P | 520 | 32 | 608 | +0.4042 (m=32) | **+0.2629** | 44 | −0.1693 | +0.0385 |

### Settled

- A–B is real: mean 0.518, sd 0.134, every CI excludes zero, weakest complex
  ~6× its own permutation null. Compare TISM in genomics (Sasse et al.,
  *iScience* 2024, PMC11404212), which reports ~0.70 gradient-vs-ISM.
- IG beats plain gradient everywhere, and by most where the signal is weakest:
  +0.2235 on 3HFM (0.4054 vs 0.1819, where plain_grad barely clears its null).
- Against experiment the margin is larger still: IG +0.4678 / +0.5591 on the
  two working complexes, plain_grad +0.1196 / +0.2618 and **0 of 4 clearing
  its null**.
- **IG does not converge on the affinity head.** On ABL1/imatinib absolute
  completeness error is ~1–3 against endpoint spans of 0.085 (`mean_aa`) and
  0.429 (`unk`), and `ig_sum` flips sign between consecutive m under both
  baselines. A 5.1× larger span left absolute error unchanged. No baseline and
  no step count fixes this (§20–22). The confidence head converges cleanly —
  3S9D 8.51% at m=32 → **0.24% at m=64** — so this is the affinity module, not
  the trunk.

### Not settled

- **Four complexes, p=0.19 on B–C.** §14's lesson — "1 of 8 clearing its null
  is what noise looks like at eight looks" — is improved on, not overturned.
- **The 36× resswap/refeaturise gap is confounded** (§2 below).
- **2JEL and 3HFM have no m=64 rung.** 3HFM fails the 5% completeness gate at
  m=32 (11.14%); 2JEL passes (2.37%). The complex whose attribution is least
  arithmetically accountable is also where IG most outperforms plain_grad.
  Unexplained, and the obvious referee question.

---

## 2. The scientific traps

These are the expensive ones. Each has already cost this project real money or
a retracted claim.

### 2.1 The geometry confound — always report the partial

Interface residues are close to the partner; so are important ones. A map that
looks like it recovers hot spots may just be reporting distance.

- **Phase 5** produced 80 SKEMPI cycles at the 90.7th percentile, 10.4× the
  median. It was geometry. Only the distance control separated the readings.
- **2JEL refeaturise plain_grad vs ddG reads +0.7234** (n=18, p=0.0010). Also
  geometry: distance correlates **−0.938** with ddG on those rows, the partial
  falls to +0.607 with a CI straddling zero, and A–B there is −0.08. Recorded
  as a trap in §26, never quoted as a result.

**Rule: never report a raw correlation without the partial against minimum
heavy-atom distance** (to partner chain for PPI, to ligand for protein–ligand).
`24_faithfulness.py` and `26_vs_experiment.py` both compute it. On the four
complexes the partial costs 0.4% / 0.3% / 3.6% / **35.0%** — it only bites on
2JEL, but you cannot know that in advance.

### 2.2 Sign orientation — derive it, then verify on named anchors

- `complex_pde` is a predicted **distance error**: lower is better. It is in
  `12_compare.LOWER_IS_BETTER`.
- Scan `delta = f(mutant) − f(wild type)`, so **positive = model says worse**.
- SKEMPI `ddG = RT·ln(Kd_mut/Kd_wt)`, so **positive = destabilising**.
- Same axis. **No negation.** Agreement is a positive correlation.

A sign flip inverts every conclusion and is silent. The repo has been bitten
(`f1a0a0c Orient the affinity score before it silently flips a correlation`).
Verify on mutations experiment is confident about before quoting anything:

- **1JTG / BLIP**: the five largest measured ddGs are W150A (+4.34), K74A
  (+3.69), H41A (+3.44), F36A (+3.20), Y53A (+3.18) — canonical BLIP hot
  spots. All five must carry positive scan deltas. They do.
- **3HFM / HyHEL-10**: the two largest are K96M and K97D, the canonical
  epitope hot spots. Both positive.

`26_vs_experiment.py` logs `ORIENTATION ANCHOR INVERTED` and records the
verdict in JSON rather than raising — an inverted anchor is a result (2JEL
inverts at p=0.145, i.e. no signal, not a sign error), but it must be visible.

### 2.3 resswap vs refeaturise — know which function you are measuring

Boltz-2 receives residue identity through two partly redundant channels: a
categorical one-hot `res_type`, and atom-level features plus coordinates built
from that residue's atom table. **Gradients only flow through the one-hot.**

| mode | what changes | what it measures |
|---|---|---|
| `resswap` (default) | `res_type[pos]` only; coords, atom features, MSA stay wild type | **the exact function the gradient differentiates** — the faithfulness ground truth |
| `refeaturise` | mutant atom features rebuilt, **wild-type coordinates reused** | the model's response to a rebuilt mutant |

Measured on 3S9D: median |delta| 0.0120 resswap vs **0.4331** refeaturise, a
**36×** gap, and the two scans are **uncorrelated** (Spearman −0.0744).
Attribution is null against refeaturise everywhere (3S9D IG −0.0123 p=0.90).

**The confound, unresolved:** refeaturise places mutant atom features at
wild-type coordinates — an input no real protein presents. Part of that 36×
may be the model reacting to a sequence–structure inconsistency rather than to
the mutation. A wild-type control rules out a featurisation bug (`wt_ref` and
`wt_fresh` agree to the last digit, offset exactly 0.0), but not this.
**Running the coordinate-consistent control is the top open item (§7).**

Two further cautions on refeaturise:

- The `token_to_rep_atom` guard skips most mutations — 847/950 on 3S9D (89%),
  500/532 on 1JTG (94%), all on `TOKEN_TO_REP_ATOM CONTENT DIFFERS`, none on
  shape. Survivors are the heavy-atom-count-preserving substitutions, which is
  a selection effect, not a random sample.
- **Refeaturise is untestable against experiment**: zero SKEMPI mutations
  survive the guard on 3S9D and 1JTG (18 and 3 on the others).

The guard exists because it must. `featurizer.py:726` builds
`token_to_rep_atom` from per-token `disto_idx`, and `loss/confidence.py:188-189`
uses it to select one representative atom per token. Gly→Trp (4 vs 14 heavy
atoms) shifts the table, so reusing wild-type coordinates reads the wrong atom
for every token after the mutation.

### 2.4 Targets are not interchangeable — never pool them

Minimum heavy-atom distance from each measured position to the ligand/partner:

| target | within 6 Å | beyond 6 Å | range | what it can test |
|---|---|---|---|---|
| **Mpro** | **12 / 12** (11 within 4 Å) | **0** | 1.9–9.1 Å | within-position ranking only — **cannot test geometry at all** |
| **EGFR** | 3 / 20 | **17 / 20** | 3.3–21.3 Å | the distal regime |
| PimA | 5 / 13 | 8 / 13 | 2.7–16.9 Å | both |
| ABL1 | 6 / 19 | 13 / 19 | 3.0–27.2 Å | both |

- **Mpro's distance partial is unidentifiable by construction.** Every measured
  position is a drug-site residue. Its value is within-position depth (18
  substitutions at Q192, 17 at H172, 16 at S144) — the only dataset that
  restores within-position ranking. Say so before quoting any Mpro enrichment.
- **EGFR carries the opposite risk.** 17 of 20 positions are distal, out to
  21.3 Å. Distal resistance mutations act through conformational equilibrium,
  which a co-folding model scoring **pinned coordinates** cannot represent even
  in principle. A null on EGFR is not evidence about attribution.
- **3HFM cannot test the geometry confound either** — all 42 SKEMPI mutations
  lie within 2.3–3.9 Å.

**Report per target, and state which question each can answer.**

### 2.5 Underpowered configurations — check df before buying GPU

- **3HFM has 15 positions against 8 confounds.** §14 spent ~100 min of A100 to
  learn that this gives a partial-correlation CI of [−1.0, +1.0], i.e.
  unidentifiable. At the mutation level (n=285) it is fine; never quote a
  position-level partial for 3HFM without the caveat.
- **n = 30–54 against experiment.** Four complexes, t=+1.67, p=0.19.
- **The two complexes that work are 80–87% alanine** (2JEL is 14%). They mostly
  answer "which positions matter", not the per-amino-acid claim the project set
  out to make. That confound cannot be controlled with this data.
- **2JEL's failure is its panel, not the method**: A–B collapses to 0.1574 on
  its 44 SKEMPI rows against 0.4042 on the full 608-row scan. Faithfulness
  fails on exactly the rows experiment chose.
- CIs and permutation nulls **assume independent mutations**, but rows cluster
  in positions (19 share each). Both are optimistic; the per-position
  aggregates are the honest unit.

---

## 3. Running a GPU job

### Environment

VM `igv-gpu`, `a2-ultragpu-1g` (1× A100-80GB, 12 vCPU, 170 GB RAM),
zone `us-central1-a`, project `agrosbio`. **Always pass `--project=agrosbio`**
— a missing project flag fails with a confusing `PERMISSION_DENIED`, and even
`gcloud storage ls` needs it.

Work runs inside the `igv:latest` docker image:

```bash
docker run --rm --gpus all --shm-size=32g --ipc=host \
  -v $HOME/IG:/app -w /app -v $HOME/boltz_cache:/root/.boltz \
  igv:latest python3 scripts/<stage>.py <flags>
```

Setup from scratch: `scripts/cloud/gcp_launch.sh`, then
`scripts/cloud/bootstrap.sh`, then `scripts/cloud/fetch_weights.sh`. See
`docs/CLOUD.md`.

### The stages that matter

| script | what it does | GPU |
|---|---|---|
| `03_attribute.py` | gradient / IG attribution of a Boltz-2 confidence score | yes |
| `04_scan.py` | brute-force mutation scan (the A–B ground truth) | yes |
| `24_faithfulness.py` | A–B: attribution vs scan | no (CPU, ~30 s) |
| `26_vs_experiment.py` | B–C and A–C: scan and attribution vs SKEMPI ddG | no |
| `18_gate_affinity.py` | IG on the affinity head (protein–ligand) | yes |
| `23_affinity_path_profile.py` | diagnostic: f(α) and directional derivative along the IG path, both heads | yes |
| `03_attribute_protenix.py` | same as 03 but Protenix — **never run on GPU** | yes |

### A complete worked run (what actually produced §25)

```bash
# 1. attribution, both rungs. NOT resumable -- must fit the window.
python3 scripts/03_attribute.py --dataset 1JTG --chain B --score complex_pde \
  --method ig --m-steps 32 --baseline mean_aa --x-pred predicted \
  --out results/1JTG_complex_pde_ig_grad_m32.npz
python3 scripts/03_attribute.py --dataset 1JTG --chain B --score complex_pde \
  --method ig --m-steps 64 --baseline mean_aa --x-pred predicted \
  --out results/1JTG_complex_pde_ig_grad.npz
python3 scripts/03_attribute.py --dataset 1JTG --chain B --score complex_pde \
  --method plain_grad --x-pred predicted \
  --out results/1JTG_complex_pde_plain_grad_grad.npz

# 2. scans. Resumable (keyed on mutation_id); WT is written as row 0.
python3 scripts/04_scan.py --dataset 1JTG --chain B --score complex_pde \
  --mode resswap --x-pred predicted --positions skempi \
  --out results/1JTG_complex_pde_scan.csv
python3 scripts/04_scan.py --dataset 1JTG --chain B --score complex_pde \
  --mode refeaturise --x-pred predicted --positions skempi \
  --out results/1JTG_complex_pde_scan_refeaturise.csv

# 3. analysis, CPU only
python3 scripts/24_faithfulness.py \
  --attribution results/1JTG_complex_pde_ig_grad.npz \
  --scan results/1JTG_complex_pde_scan.csv \
  --dataset 1JTG --chain B --score complex_pde \
  --out results/1JTG_faith_ig64_resswap.json
python3 scripts/26_vs_experiment.py --dataset 1JTG --chain B \
  --scan results/1JTG_complex_pde_scan.csv --mode resswap \
  --attribution ig64=results/1JTG_complex_pde_ig_grad.npz \
  --attribution plaingrad=results/1JTG_complex_pde_plain_grad_grad.npz \
  --out results/1JTG_complex_pde_vs_experiment_resswap.json
```

Notes on flags that bite:

- `03_attribute.py --baseline` accepts only `{zeros, none, mean_aa}`. The
  `unk` and `shuffled` baselines exist **only** in `18_gate_affinity.py`,
  `23_affinity_path_profile.py` and `03_attribute_protenix.py`.
- `--x-pred predicted` is the default and is what you want. `zeros` puts every
  atom at the origin; §14 found stage 03 silently doing this for months.
- `--positions skempi` is required for SKEMPI datasets. `--dry-run` plans
  without touching the GPU.

### Measured cost and memory

Wall times from `notes/vm_logs_sec25/driver.log`, one A100-80GB, fp32:

| stage | 3S9D (307 tok) | 1JTG (427) | 2JEL (520) | 3HFM (558) |
|---|---|---|---|---|
| 03 IG m=32 | ~20 s/step | **3322 s** (55 min) | reused from §14 | reused from §14 |
| 03 IG m=64 | **2662 s** (44 min) | **6562 s** (109 min) | ~234 min (not run) | ~259 min (not run) |
| 03 plain_grad | 113 s | ~116 s `[UNVERIFIED]` | **245 s** | **271 s** |
| 04 resswap | **4477 s** (75 min, 951 fwd) | **3489 s** (58 min) | **11008 s** (183 min) | **6006 s** (100 min) |
| 04 refeaturise | 1110 s | 1355 s | 3131 s | 1558 s |
| 24 / 26 | ~30 s, CPU | ~30 s | ~30 s | ~30 s |

IG step time scales worse than token count: **50.75 s/step on 1JTG** (427
tokens) against ~20 s/step on 3S9D (307) — 2.54× the time for 1.39× the
tokens.

**Peak VRAM — the 80 GB card IS required:**

| job | peak | note |
|---|---|---|
| 3HFM plain_grad | **71828 MiB (70.1 GiB)** | the binding constraint, ~10 GiB headroom |
| 2JEL plain_grad | 62708 MiB (61.2 GiB) | |
| 1JTG IG m=32 / m=64 | 43608 MiB (42.6 GiB) | |
| 3S9D IG m=64 | 20648 MiB (20.2 GiB) | smallest complex |
| any 04 scan | 11498–16740 MiB | forward only, far cheaper |

Three caveats on those numbers:

1. These are `nvidia-smi memory.used` — **driver-reserved**, including the
   PyTorch caching-allocator pool, not live tensor footprint.
   `torch.cuda.max_memory_allocated` was never logged. Read as "80 GB
   suffices, 24 GB does not", not "70 GiB of tensors exist".
2. Sampled at 10 s. The plain_grad figures rest on 25–28 samples over ~4 min
   jobs and could understate a brief true peak.
3. **plain_grad apparently peaks higher than IG m=64**, which is
   counterintuitive (61–70 GiB vs 42.6 on 1JTG). Likely a sampling or
   allocator-pool artifact rather than a real inversion. `[UNVERIFIED]` —
   worth one clean measurement with `max_memory_allocated` before anyone sizes
   hardware on it.

`docs/CLOUD.md` records that a 48 GB card (L40S/A40/A6000) is a documented
stage-03 OOM, and 40 GB likewise.

The 1JTG plain_grad time is the one figure here not recoverable from
`driver.log` (that stage ran in an earlier window and the log records only
`NOTSELECTED` for later passes); it is carried over from a run report. Every
other cell is from `driver.log` or `notes/draft_section_24_faithfulness.md`.

> **Stale claim warning.** `notes/draft_section_24_faithfulness.md` still says
> *"the 80 GB card is not needed for this arm"*, based on 3S9D's 20.2 GiB.
> That was measured on the smallest complex and **does not generalise**;
> `notes/logging.md` §25 (COMPLETION) formally withdraws it. The draft has not
> been updated. Trust the table above.

---

## 4. Operational hazards

Each of these has already cost this project time. Each has a fix.

### 4.1 The VM stops itself after exactly 2 hours

`scheduling.maxRunDuration = 7200s` with `instanceTerminationAction: STOP`.
GCP issues `compute.instances.deferredStop` as user `system` **exactly two
hours after every start**, to within three seconds. On 2026-10-13 this killed
two in-flight m=128 attributions before anyone identified it; the first guess
("an external watchdog") was wrong.

It is not in `scripts/cloud/gcp_launch.sh` — it was set directly on the
instance, so a freshly launched VM may not inherit it.

To raise it (**the instance must be STOPPED**):

```bash
gcloud compute instances set-scheduling igv-gpu --zone=us-central1-a \
  --project=agrosbio --max-run-duration=12h --instance-termination-action=STOP
```

**Restore it to `2h` afterwards.** It is a genuine safety net against a hung
job burning an A100 unattended. Verify with:

```bash
gcloud compute instances describe igv-gpu --zone=us-central1-a \
  --project=agrosbio --format="value(status,scheduling.maxRunDuration.seconds)"
```

Size every job to finish inside the window, or make it resumable first.

### 4.2 `us-central1-a` returns STOCKOUT

Starting the VM can fail with *"does not have enough resources available ...
(state:STOCKOUT)"*. This happened on 2026-10-14 and stranded finished results
on the disk for hours. The boot disk is **zonal**, so you cannot simply start
elsewhere — you would need a snapshot restored into another zone.

Fix: retry on a timer (shortages usually clear within hours). Do not assume a
start failure means something is broken.

### 4.3 The SSH key has a passphrase

`gcloud compute ssh` fails with `Permission denied (publickey)` if the agent is
empty — `~/.ssh/google_compute_engine` is passphrase-protected and is the only
key the VM accepts. This cost one session its entire GPU slot.

```bash
ssh-add --apple-use-keychain ~/.ssh/google_compute_engine   # run in Terminal.app
```

Do **not** work around it by adding a different key to instance metadata.
There is also a second local account, `bnovey@berkeley.edu`, with **no roles on
this project** — switching to it produces confusing permission errors rather
than a clear message. Check `gcloud config get-value account`.

### 4.4 Stage 03 is not resumable; the scans are

`04_scan.py` resumes on `mutation_id` and reuses the wild-type score
(`Resuming: N mutations already scored`). **Stage 03 has no resume** — an
interrupted attribution loses everything. Plan accordingly against §4.1.

### 4.5 Never run two GPU jobs at once

Contended-GPU results have been invalid before — see
`results/INVALID_contended_gpu_sanity_bf16.json`. Serialise everything. If you
build a driver, have it check for a running job before starting another.

### 4.6 Sync every artifact the moment it lands

Not at the end. On 2026-10-13 the VM stopped mid-run three times; the first
stop nearly lost a completed m=64 attribution that existed only on the VM.
After switching to sync-on-completion, the next two stops cost nothing.

Every artifact has a `.prov.json` sidecar. **Sync both.**

### 4.7 The `.DONE`-marker bug class

A driver stamped `*.DONE` whenever a stage returned 0 — **including stages
skipped because they were not selected**. 2JEL and 3HFM acquired zero-byte
`scan.csv.DONE` markers for scans that never ran, and the driver logged
`COMPLEX 2JEL COMPLETE`. Every later window would have skipped those scans and
reported finished complexes with no data.

Caught by listing artifacts, not by reading the log. **Distinguish exit codes**
(ran-ok / already-done / not-selected / deferred) and stamp only on ran-ok.
More generally: verify state, do not trust a success line — a local watcher in
this project printed `STOPPED_ON_COMPLETE` while its stop command had errored.

---

## 5. Storage and cost

### Where things live

| what | where | backed up? |
|---|---|---|
| code, notes, research log | git → `github.com/Bnovey/cofold-interp`, branch `protein-ligand-pivot` | yes |
| **results/ and data/** | local disk **and** `gs://agrosbio-ig-artifacts` | yes, since 2026-10-15 |
| model weights | `~/boltz_cache` on the VM | **no, deliberately** |
| VM environment | snapshot `igv-gpu-final-20261015` | yes |

**`results/` and `data/` are gitignored.** The bucket is their only backup. It
mirrors both (208 files / 115 MB and 86 files / 85 MB at time of writing), has
**versioning enabled**, and sits in `us-central1` — the same region as the VM,
so VM↔bucket transfer is free.

```bash
gcloud storage rsync results/ gs://agrosbio-ig-artifacts/results/ --recursive
gcloud storage rsync data/    gs://agrosbio-ig-artifacts/data/    --recursive
```

**Do not back up the model weights.** They are public and re-downloadable via
`scripts/cloud/fetch_weights.sh`; paying to store several GB protects against
nothing. Reproducibility comes from pinning the version, which the `.prov.json`
sidecars already record.

### Restoring the VM

The 500 GB pd-ssd was snapshotted and the instance deleted. The snapshot holds
the full environment: docker image `igv:latest`, the Python env, `boltz_cache`,
and the repo checkout. Actual used space was **58.4 GB** of the 500 GB
provisioned, which is why the snapshot is cheap.

```bash
gcloud compute disks create igv-gpu \
  --source-snapshot=igv-gpu-final-20261015 \
  --zone=us-central1-a --type=pd-ssd --project=agrosbio
# then attach to a new a2-ultragpu-1g instance; see docs/CLOUD.md
```

### Rough costs

Verify before relying on these; rates change.

| item | cost |
|---|---|
| A100-80GB (`a2-ultragpu-1g`), on demand | **~$5.07/hr** (~$122/day) |
| 500 GB pd-ssd, billed even while stopped | ~$2.83/day (~$85/month) |
| snapshot, billed on **used** bytes (~58 GB) | ~$1.50/month `[UNVERIFIED rate]` |
| bucket, standard class | ~$0.02/GB/month → ~$0.004/month for 200 MB |

Use **on-demand, not spot**, for GPU stages: stage 03 is not resumable and a
preemption loses the run.

---

## 6. The honest cost–benefit — IG is barely a shortcut

This changes what is worth running, so it belongs up front in any plan.

| complex | brute force (resswap) | IG | speedup |
|---|---|---|---|
| 3S9D (950 mutants) | 75 min | 44 min (m=64) | **1.7× faster** |
| 1JTG (532 mutants) | 58 min | 109 min (m=64) | **1.9× SLOWER** |

TISM reported ~160× in genomics. Here IG ranges from 1.7× faster to nearly 2×
slower than simply running every mutation.

The reason is structural: **brute force costs one forward pass per mutation and
scales with how many mutations you want; IG costs a fixed 2m forward-and-
backward passes**, and a backward pass through this trunk with checkpointing is
far more expensive than a forward one. Crossover is around 1000 mutations
(~50 positions at 19 substitutions). Below that, brute force wins outright.

**The plain gradient is the only genuine shortcut**: 116–271 s against 58–183
min, i.e. **22–45×** — but it clears its null against experiment on **0 of 4**
complexes (mean A–C +0.1214 vs IG's +0.2651).

Practical guidance:

- Want a full saturation scan of a large interface? IG is marginally worth it.
- Want a handful of positions? Run the mutations — you get the model's exact
  answer faster than approximating it.
- Want a cheap screen, and only ranking matters? Plain gradient, with the
  caveat above.
- **At m=32 vs m=64 the rung barely matters for A–C**: 1JTG gives +0.5613 and
  +0.5591. Pay for m=64 only when you need the completeness ladder.

---

## 7. What is open, in priority order

### 7.1 The coordinate-consistent control — decides the paper's thesis

Re-predict each mutant's geometry rather than reusing the wild type's, then
rerun the scan and the A–B comparison. Everything about the "attribution reads
a channel that carries little of a real mutation's effect" reading rests on a
36× gap measured with mutant atom features at wild-type coordinates.

- If the gap **survives**: attribution is faithful to a function that is not
  the one practitioners care about. The strong, interesting paper.
- If it **collapses**: we were measuring the model's reaction to a malformed
  input. Still publishable, but a different and smaller claim.

Not yet implemented. Needs `04_scan.py` to call `predict_structure_coords` per
mutant, which makes the scan far more expensive — budget accordingly.

### 7.2 Protenix — built, never run on GPU

Turns "Boltz-2 does this" into "co-folding models do this".

- `src/igv/protenix_score.py` + `scripts/03_attribute_protenix.py`, with all
  four gradient blockers patched at runtime (no fork): the recycling loop under
  `set_grad_enabled(self.training)`; an in-place MSA slice update;
  `ConfidenceHead.stop_gradient`; and a `@torch.no_grad()` summary-score
  function.
- **Confidence scores only — Protenix has no affinity head.** The ligand arm
  stays Boltz-only.
- Separate image `docker/Dockerfile.protenix` (protenix conflicts with
  `boltz==2.2.1` and pins cuequivariance 0.8.0, a known A100 crash). Apache-2.0
  `protenix_base_default_v1.0.0` weights only; v2 is proprietary and
  `load_model` refuses it.
- **Never built, never run.** First session must: build the image, run
  `scripts/08_memscale_protenix.py`, run the GPU-only ipTM/pTM/pLDDT parity
  test in `tests/test_protenix_score.py`, then `complex_pde` on 1JTG/3S9D.
- Both models apply a hard max inside pTM and ipTM — `complex_pde` avoids it
  and is the score that transfers. (Answers the open question in `PLAN.md` §10.)
- Chai-1 was rejected: compiled trunk blocks per-block checkpointing, and
  full-trunk backward already OOM'd on a 141 GB H200.

### 7.3 The missing m=64 rungs for 2JEL and 3HFM

~234 and ~259 min respectively. A referee will ask why the two weakest
complexes have the least accountable attributions.

### 7.4 More complexes

Four is defensible; eight is comfortable. The pipeline is built and
`notes/draft_ligand_targets.md` has vetted protein–ligand candidates (EGFR,
Mpro, PimA — all with ground truth fetched and structural guards passing).

### 7.5 Verify the prior-art citations before using them

A literature search concluded that **no one has compared gradient attribution
against a brute-force mutation scan on a folding model**, which would make the
A–B number the first of its kind. That conclusion is plausible but its
citations were **never written to disk** — `notes/draft_prior_art_channels.md`
does not exist, despite being referenced. Several cited works post-date
available training data and none were independently confirmed.

**Treat the whole novelty claim as `[UNVERIFIED]`.** Check every DOI resolves
and says what is claimed before it goes near an abstract. A missed prior paper
is the one failure mode that sinks a benchmark paper.

---

## 8. Conventions

- **`notes/logging.md` is the research log.** Numbered sections, append-only in
  spirit; corrections are written as explicit retractions rather than edits
  (see §21, and §25's retraction of its own first pass). Read the last section
  first.
- **`notes/PLAN.md` is immutable.** It is the original proposal. Record what
  changed in `logging.md` instead.
- **`notes/ERRORS_LOG.md`** holds numbered defects and their resolutions.
- **Every artifact gets a `.prov.json` sidecar** — git SHA, dirty flag,
  parameters, input checksums. Untracked files make `git_dirty: true` and
  render a run unreproducible from its commit alone, which is why
  `probe_mem.py` and friends are gitignored.
- **Register predictions before the run**, in a file, committed. See
  `notes/_sec25_predictions_registered.md`. A prediction only counts if its
  failure is written down too — and when it is wrong, say so (§25 retracted a
  premature "FALSIFIED" call made on the single most favourable complex).
- **Name the stage before reporting it.** "Gate A" is an index into a long
  document, not a name. Open any status report with one plain sentence on what
  was tested and what a failure would mean.
- **Commits** are written as a claim plus its evidence and its limits — a
  headline sentence, the numbers, then what the result does *not* establish.
  Run the full suite (`python3 -m pytest -q`, currently 1375 passing) and
  `python3 -m ruff check .` (currently 1 pre-existing error in
  `scripts/msa_sensitivity.py`) before committing; the ruff count must not
  increase.
