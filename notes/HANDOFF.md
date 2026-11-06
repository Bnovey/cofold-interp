# Handoff — operational runbook

How to start the VM, run the pipeline, and avoid known traps.

For the research plan and science context, see `logging.md`. For the original
proposal, see `PLAN.md` (must not be modified).

---

## 0. Where we are -- 2026-10-01. THE POSITION-LEVEL METHOD IS ALSO A NEGATIVE.

**VM TERMINATED. Local `main` has today's six commits, 852 tests, ruff clean.**

**The simplex reduction fix is done, executed, and makes no difference.** Four
complexes, paired against the old L2 norm on the same artifacts:

| complex | pos | simplex | L2 | cleared own null? |
|---|---|---|---|---|
| 3S9D A | 50 | +0.4683 | +0.0356 | simplex only |
| 1JTG B | 28 | +0.1714 | +0.2946 | neither |
| 2JEL P | 32 | -0.3396 | +0.0031 | neither |
| 3HFM Y | 15 | +0.1573 | +0.4218 | neither |

simplex mean +0.1143 (p=0.54); L2 mean +0.1888 (p=0.16); paired difference
-0.0744 (p=0.70). Simplex won 1 of 4. **1 of 8 measurements cleared its own
shuffled null.** Full detail in `logging.md` section 14 "VERDICT".

### Read this before proposing anything

1. **Do not re-run the simplex fix expecting a different answer.** It is wired
   end to end and correct; `ERRORS_LOG.md` entry 31's diagnosis was right and
   fixing it changed nothing measurable. Three defects had to be closed first
   (entries 37, 39, 40 plus the producer gap) -- all closed.
2. **Do not quote the 3S9D +0.4683 alone.** One favourable draw in four; the
   sign flips across complexes. Between-complex sd ~0.33 swamps the
   between-method difference of -0.07.
3. **Boltz-2 has no PPI affinity score** (entry 36). The affinity head raises
   `ValueError("Affinity is currently only supported for ligands.")` at
   `schema.py:1066-1070`. There is nothing better in this model to aim at.
4. **Real RSA absorbs the position-level signal** (entry 38). On 1JTG the
   partial fell +0.211 -> -0.005 once Shrake-Rupley RSA entered the panel.
   `burial` alone beats the attribution against ddG on every complex measured.
5. **m=32 is now too few steps.** Real geometry tripled the
   `f(x) - f(baseline)` span; completeness failed on 3 of 4 runs (8.51%,
   11.14%, 5.99%; only 2JEL passed at 2.37%). Anything quoted as final needs
   m=64.
6. **Check degrees of freedom before buying GPU time.** 3HFM has 15 positions
   and the panel has 8 confounds, so its partial CI came back [-1.0, +1.0] --
   unidentifiable. That cost ~100 min of A100 to learn nothing.
7. **Check the NVIDIA driver before every long run** (entry 39). An unattended
   upgrade broke the GPU mid-session; the already-running container kept
   working for 25 minutes on a GPU the host could no longer describe, so
   "still producing output" is not a health check.
8. **The VM now has `~/autostop.sh`.** It halts the instance when
   `~/replicate.log` reports ALL DONE, or if the job dies, with a 6h cap.
   Results live on the persistent boot disk (`/dev/root`) and survive. Re-arm
   it for any long detached run -- the instance does not stop itself otherwise.

**Spend 2026-09-30/10-01: ~8 h of A100, ~$40.** Cumulative project spend is
still far under the original Phase 5 budget.

---

## 0. Where we are — 2026-09-30. PHASE 5 IS CLOSED AS A NEGATIVE.

**Local: `main` at `529b14d`, 796 tests, ruff clean. VM: TERMINATED. Do not
start it for Phase 5 -- there is nothing left in that plan worth buying.**

**Gate C failed on 1JTG.** `Partial(A[i,j], coupling | A[i], A[j]) = +0.0019`
against a kill condition written before the data existed. The pair map encodes
**contact geometry, not binding energetics**: `Spearman(|A[i,j]|, centroid
distance) = -0.527` while the raw correlation against measured coupling is
-0.0335 at permutation p=0.78. Section 12 predicted exactly this failure
signature. Full detail in `ERRORS_LOG.md` entry 35 and `logging.md` section 13
"STAGE 3 RESULT".

**Artifacts (all local; `results/` is gitignored):**

| file | what |
|---|---|
| `results/1JTG_complex_pde_pair_ig.npz` | 93 MB, the (427, 427) map + all four m rungs + token map |
| `results/1JTG_complex_pde_pair_capture.json` | Gate B statistics, completeness ladder |
| `results/1JTG_coupling.json` | the four controls, Gate C |
| `results/gate_a_1VFB_complex_pde.json` | Gate A, zeros geometry |
| `results/gate_a_1VFB_xpred_compare.json` | Gate A + the x_pred sensitivity measurement |

**Total spend: ~$4 across four A100 sessions**, against ~$60-75 budgeted to
reach this point. Stage 4 (1BRS plus four complexes) is correctly unspent.

### Read this before proposing the next thing

1. **The target was never an affinity score.** All six registered scores are
   structure-confidence metrics; `complex_pde` is predicted distance error.
   Boltz-2's affinity head `boltz2_aff.ckpt` is in the checkpoint directory and
   **deliberately not loaded** (`boltz_score.py:475-493`, `affinity=False` at
   :891). Section 12's claim says "binding-affinity score". Either correct the
   claim or change the target; do not restate it as written.
2. **A convincing false positive was available and we nearly had it.** The 80
   SKEMPI cycles sit at the 90.7th percentile of the map by `|A|`, 10.4x its
   median, half in the top decile. That reads as "the attribution recovers
   known hot spots". It is geometry -- interface residues are close, the map is
   high on close pairs. Only Control 1 separates the two readings. Any future
   claim of hot-spot recovery must report the distance partial beside it.
3. **Do not quote enrichment-over-uniform across different L.** Gate B's
   "top-10 = 59.75x uniform" is inflated by 90,951 cells versus entry 31's 165
   positions. The denominator-free measure: 50% of `|A|` mass needs 8.4% of
   cells, i.e. ~6x, not 60x. Use the mass-fraction curve.
4. **The ground truth was wrong and is now fixed** (`ERRORS_LOG.md` entry 33).
   Coupling std 0.92 not 1.84, 33/80 pairs above 0.5 not 53, and the
   cross-chain physics check is null (p=0.75) rather than confirmatory. Use
   `igv.coupling` with within-reference matching; never a fixed RT.
5. **What is NOT established:** that gradient attribution on pair
   representations fails. Only that attributing a structure-confidence score
   yields a map of structure. Whether the affinity head carries PPI energetic
   signal is open, and the prior is poor -- trained predominantly on
   protein-ligand data, and King et al. (arXiv:2512.06592) find Boltz-2
   fine-tuned for PPI affinity underperforms sequence baselines. Settle that
   cheaply before building.

---

## 0.05 Gate A — passed 2026-09-29

**Local: `main` at `62889c6`, 750 tests, ruff clean. VM: TERMINATED, synced to
`62889c6`. `origin/main` in sync.**

**GATE A PASSED 2026-09-29, ~$1.70, ~20 min of A100.** The z seam works:
gradient reaches an externally supplied pair tensor through the real
checkpointed confidence head. Artifact
`results/gate_a_1VFB_complex_pde.json`, full detail in `logging.md`
section 13 Stage 1. Headline: at L=352 on full 1VFB, `z.grad` is
`(1, 352, 352, 128)`, fp32, zero-fraction 0.0000, all finite, and
**completeness absolute error 0.003627 — 0.64% relative at m=5**. Peak VRAM
8.68 GiB, wall time 5.0 s for five IG steps. Phase 5's method is viable.

**What Phase 5 now rests on, stated plainly:** the *ground truth* was found to
be wrong in the same session (`ERRORS_LOG.md` entry 33). Corrected coupling std
is 0.92 not 1.84, 33 of 80 pairs exceed 0.5 kcal/mol not 53, and the
cross-chain physics sanity check collapses to a 0.070 kcal/mol gap at
Mann-Whitney p=0.75 — no evidence. So the gradient machinery is confirmed
working, and the quantity it will be validated against is harder and less
reassuring than planned. Both facts are needed together.

### Two blockers before Stage 2 — free, local, and they corrupt rather than crash

1. **`x_pred` is zeros.** Gate A logged `feats['coords'].abs().max() = 0`. The
   sequence-only YAML sets every atom to (0,0,0), so the confidence head sees
   no geometry. Fine for a gradient-flow test; wrong for a real capture. Stage
   2 must run an actual structure prediction (~45 s) and pass real `x_pred`.
2. **The featurisation cache silently substitutes** (`ERRORS_LOG.md` entry 34).
   `data/raw/boltz_homopolymer/<AA>` is not keyed on dataset, so it returned
   1JTG's 427-token features for a 352-token 1VFB request. Entry 24 fixed this
   for the main path only. Only `_build_token_map`'s run-length guard caught
   it; otherwise the baseline would have been built from the wrong protein and
   Gate A would plausibly have passed anyway.

### Syncing the VM — `git pull` does NOT work, use a bundle

Discovered 2026-09-29. The VM's remote is `git@github.com:Bnovey/cofold-interp.git`
but `~/.ssh` holds only `authorized_keys` — **no GitHub private key**, so
`git pull` fails with `Permission denied (publickey)` after an initial
`Host key verification failed`. Adding the host key via `ssh-keyscan` fixes
only the first error, not the second. Section 2's "Option A" below is
therefore not currently available.

What works, and keeps provenance honest (a real commit, not a dirty tree):

```bash
# on the laptop
git bundle create /tmp/ig.bundle main          # ~486 KB for this repo
gcloud compute scp /tmp/ig.bundle igv-gpu:/tmp/ig.bundle \
  --zone=us-central1-a --project=agrosbio

# on the VM
cd ~/IG && git fetch /tmp/ig.bundle main:refs/bundle-main \
  && git reset --hard refs/bundle-main && git log --oneline -1
```

**Also: `data/raw/` is gitignored, so PDB files are NOT on the VM.** Gate A
failed its first run with `PDB not found: data/raw/1vfb.pdb`. Fetch before
running:

```bash
curl -sfL -o data/raw/1vfb.pdb https://files.rcsb.org/download/1VFB.pdb
```

**Corrected 2026-10-01: `notes/` is no longer gitignored** -- `logging.md`,
`ERRORS_LOG.md`, `MEMORY.md`, `HANDOFF.md` and `PLAN.md` are now tracked and
published, so a `git pull` or bundle fetch DOES carry them to the VM. The
`results/*.npz` artifacts (96 MB) are still ignored and still do not travel.
The original warning below held while `notes/` was excluded and is kept because
any bundle built from a commit before `b06e4d1` still lacks them:
~~`notes/` is gitignored too, so the corrected plan documents never reach the
VM. The code syncs; the reasoning does not.~~

---

## 0.1 Previous status — 2026-09-27

> Historical section, kept for the pipeline defects in 2.1 which are still
> worth reading. Its status line said "`main` at `4962fbc`, tree clean, 549
> tests" and was stale by several commits when found. Test counts here go out
> of date silently -- **print the number, do not trust the doc.**

**No GPU result was newer than 2026-09-24 as of that date** (superseded: Gate A
ran 2026-09-29)**.** The 2026-09-27 session never ran a
model. The VM was started once for a health check (A100-SXM4-80GB confirmed,
429 GB free) and stopped again about 12 minutes later, roughly $1. The only
measured science in the project remains the 1JTG chain B table in `logging.md`
section 8 — headline **IG m=32 with `mean_aa` = 0.357**, above a null p95 of
0.313, with a CI of [-0.002, 0.656] that excludes zero by two thousandths. Do
not describe that as replicated; it has not been.

### What the 2026-09-27 session actually did

1. **Rejected GB1, adopted Starr 2020 / 6M0J as the saturation arm.** GB1's
   floor lands on the hot spots (position 27 has all 19 substitutions pinned),
   it couples folding to binding by its own definition, and 1FCC is the wrong
   protein G paralogue. Entries 25-27, `logging.md` sections 9-10.
2. **Verified the within-position design removes the volume confound** —
   +0.44 pooled to -0.034 within position on Starr, with pooled-on-Starr at
   +0.027, which shows the +0.44 is a property of the alanine-scan *design*.
3. **Established that the folding confound misses the interface.** Binding and
   expression correlate +0.64 pooled but only +0.074 over interface mutants,
   where residualising keeps 100% of the binding variance. Entry 28.
4. **Built the saturation arm:** `src/igv/dms.py`,
   `scripts/11_within_position.py`, `scripts/12_compare.py`.
5. **Fixed four pipeline defects, each of which would have wasted paid GPU
   time.** See section 2.1 — this is the part most worth reading.

### Two numbers corrected this session

- The within-position volume residual **on the 21 interface positions we will
  actually analyse is +0.092 mean / +0.245 median**, not the -0.034 measured
  over all 194. Quote the interface figure next to any interface result.
- **A perfect binding predictor scores +0.355 against expression**, because the
  two readouts are themselves correlated at the interface. So a gradient at
  ~0.35 there is not contamination; the diagnostic is the *gap* between
  prediction-vs-binding and prediction-vs-expression.

### Sync to GitHub — resolved 2026-09-28

`git push origin main` was chosen and done. `origin/main` is now `2b2c01f`, in
sync with local, so the work finally has a copy off the laptop.

Worth recording: GitHub had been **27 commits behind**, further back than the
VM. The old remote head was `fb0fc1e`, so the VM's `b3d3763` was itself ahead of
the remote. Anyone reasoning about "how far behind is X" should check against
both, not assume the remote is current.

**The VM is still at `b3d3763`, 10 commits behind, and still TERMINATED.** It
needs a `git pull` in its checkout before any GPU run.

---

## 1. Infrastructure

### GCP

| | |
|---|---|
| Project | `agrosbio` |
| Billing account | see `gcloud billing accounts list` (there are two; the second is **closed**, ignore it) |
| Instance | `igv-gpu`, `a2-ultragpu-1g`, `us-central1-a` |
| Status | **TERMINATED** (= stopped, not deleted) |
| Boot disk | 500 GB pd-ssd, `READY`, **preserved** |
| GPU verified | `NVIDIA A100-SXM4-80GB`, 81920 MiB, driver `580.173.02` |
| Host | 12 vCPU, 167 GB RAM, 468 GB free on `/` |

Credits pay for this. Credits apply before the card automatically, but they
**expire by date** and when they run out billing silently rolls to the card.
Budget alerts only notify; they do not cap.

### Quota (approved 2026-09-02, region `us-central1`)

```
NVIDIA_A100_80GB_GPUS              limit=1   usage=0
A2_CPUS                            limit=12  usage=0
NVIDIA_A100_GPUS       (40GB)      limit=1
PREEMPTIBLE_NVIDIA_A100_GPUS       limit=16
PREEMPTIBLE_NVIDIA_A100_80GB_GPUS  limit=0
```

`A2_CPUS=12` is exactly what `a2-ultragpu-1g` consumes, so **only one A2 machine
can exist at a time.** A second one — including a Colab Enterprise runtime —
fails with what looks like a capacity error.

### Cost

| Item | Rate |
|---|---|
| `a2-ultragpu-1g` running | **~$5.07/hr** (~$122/day) |
| Stopped, disk only | ~$0.12/hr (~$3/day) |

### Start / stop

```bash
gcloud compute instances start igv-gpu --zone=us-central1-a --project=agrosbio
# ~5-6 min in practice, not ~30 s. `start` returns BEFORE sshd is listening --
# the next ssh gets "Connection refused". Poll with the until-loop in section 2.
# External IP changes on every start; connect by name, not IP.

gcloud compute instances stop igv-gpu --zone=us-central1-a --project=agrosbio \
  --discard-local-ssd=true
# --discard-local-ssd is REQUIRED on a2-ultragpu (it has Local SSD attached).
# Safe: our work lives on the boot disk, the Local SSD is unmounted scratch.
```

### SSH

The GCE key `~/.ssh/google_compute_engine` is **passphrase-protected**. Agent
keys are shared via the macOS launchd socket, so if the agent is empty, SSH from
an automated session fails with `Permission denied (publickey)` *after* logging
`Server accepts key` — that specific pattern means "can't sign", not "wrong key".

Fix, run by the user in **Terminal.app** (the prompt cannot be answered from
inside Claude Code):

```bash
ssh-add --apple-use-keychain ~/.ssh/google_compute_engine
```

Verify with `ssh-add -l`.

---

## 2. Resume — exact steps

```bash
cd /Users/bowmannovey/Agros/IG

# 0. Load the SSH key (Terminal.app, once, if `ssh-add -l` is empty).
#    The trap: ssh reports "Server accepts key" AND "agent contains no
#    identities" -- that is the passphrase not being in the agent, not a
#    permissions problem.
ssh-add --apple-use-keychain ~/.ssh/google_compute_engine

# 1. Start the VM. ~5-6 min in practice, not the ~30 s this file used to claim.
#    Billing starts here at ~$5.07/hr. sshd is NOT up when `start` returns --
#    the first ssh gets "Connection refused". Wait for it, do not sleep blindly:
gcloud compute instances start igv-gpu --zone=us-central1-a --project=agrosbio
until gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
        --command=true >/dev/null 2>&1; do :; done; echo SSH_READY

# 2. Get the code onto the box. DECIDE FIRST -- see section 0.
#    Option A (publishes to GitHub):
#      git push origin main
#      gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
#        --command='cd ~/IG && git pull --ff-only'
#    Option B (nothing leaves): scp the changed files, as the old flow did.
#    Verify either way -- the VM must report the same commit as local HEAD:
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='cd ~/IG && git log --oneline -1'

# 3. Bootstrap (detached; torch/boltz layers are already cached)
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='cd ~/IG && nohup bash scripts/cloud/bootstrap.sh > ~/bootstrap.log 2>&1 & echo started'

# 4. Watch it
gcloud compute ssh igv-gpu --zone=us-central1-a --project=agrosbio \
  --command='tail -40 ~/bootstrap.log'
```

**Gate:** bootstrap's verification table must show `torch 2.7.1+cu126`,
`cuda_available True`, `gpu_0_vram_gib >= 80`, `boltz 2.2.1`, and every
`forbidden_*` row as "not found". If torch is any other version, the container
lost its fight with the base image — stop and fix that before running science.

### Running a job

`run_all.sh` is **AbBiBench-only** — stages 00 and 01 do not apply to a SKEMPI
or DMS complex, which is why the 1JTG session was driven stage by stage by hand.
Use `run_complex.sh`, which covers both other arms, skips 00/01, and fails fast
on an unknown dataset name **before** any GPU work:

```bash
# SKEMPI arm: 07 -> 02 -> 03 -> 04 -> 05 -> 12 -> 10
DATASET=1JTG CHAIN=B POSITIONS=skempi bash scripts/run_complex.sh

# DMS/saturation arm: 07 -> 02 -> 03 -> 04 -> 05 -> 12 -> 11
DATASET=spike_rbd CHAIN=E POSITIONS=dms_interface bash scripts/run_complex.sh
```

Add `--dry-run` to print every command without executing. Do that first; it is
free and it is how three of the four defects below were caught.

Env overrides: `SCORE` (default `complex_pde`), `METHOD` (default `ig`),
`M_STEPS` (32), `BASELINE` (`mean_aa`), `POSITIONS`, `INTERFACE_CUTOFF` (5.0).

### Planned job sizes, measured not estimated

| Job | Scope | Note |
|---|---|---|
| Pool 3HFM, 1VFB, 1JRH, 2JEL | 4 complexes | ~$12, ~2.5 h. Takes effective n from 28 to ~128. Do this **before** the scan: if 0.357 does not reproduce, it was noise and that is cheap to learn. |
| 1JTG brute force | 28 positions x 19 = **532 mutants** | ~$13. `logging.md` said 931; the real number comes from `04_scan.py --dry-run`. |
| 6M0J saturation | 791 tokens | ~1.5-3 h per 32-step IG run. One IG run covers **all 194 positions x 20**; only the scan scales, at 399 forward passes for the 5 A set. |

**Stop the VM when done. It does not stop itself.**

```bash
gcloud compute instances stop igv-gpu --zone=us-central1-a --project=agrosbio \
  --discard-local-ssd=true
```

## 2.1 The four defects fixed on 2026-09-27 — read before trusting the pipeline

Every one of these would have burned paid A100 time, and none of them would have
raised an error at the point of the mistake.

1. **`04_scan.py` had no SKEMPI branch.** `--dataset 1JTG` fell through to the
   AbBiBench path and 404'd on a benchmarking CSV that does not exist. The 404
   was the *lucky* case: a name that happens to exist in AbBiBench would have
   downloaded a different protein and scanned it silently — the same shape as
   the boltz cache trap in entry 24. Resolution now goes registry-first through
   `igv.dms.resolve_pdb_complex`, so fallthrough is impossible rather than
   merely unlikely.
2. **The runner ended on `06_metrics.py`**, which requires columns
   `pred`/`binding_score`/`n_mut` while `05_predict.py` writes
   `position`/`mut_aa`/`score_delta` on these paths. It would have aborted
   *after* stages 02-04 had already spent the hours. Stage 06 is AbBiBench-only.
3. **`07_sanity.py` had no DMS branch** — it stopped at "No structure known for
   spike_rbd", and it is the first stage in the runner, so the saturation arm
   could not start at all.
4. **Stage 04's scan CSV was written and read by nothing.** The most expensive
   GPU stage produced an artifact that was never compared to anything, and
   Phase 4 had no implementation. Now `scripts/12_compare.py`. Entry 29.

**The lesson, which is more reusable than the fixes:** defect 4 survived a
validation pass that walked the stage list confirming *every input is produced
by an earlier stage*. That was true and was never the question. The check that
finds this class of bug is the reverse — **is every output consumed** — and a
produced-but-unread file is invisible to any test that runs a stage in
isolation. Check both directions of the file graph when adding a stage.

---

## 3. Environment facts worth not rediscovering

- **Verification table (passing):** python 3.11.15, torch 2.7.1+cu126,
  cuda_available True, gpu_0_vram_gib 79.2, NVIDIA A100-SXM4-80GB, gpu_count 1,
  boltz 2.2.1, and `boltzgen` / `protenix` / `chai_lab` / `gnina` /
  `cuequivariance` all "not found".
- **The VM's `~/IG` is a real checkout**, synced by shipping a tarball that
  includes `.git` (only 488 KB) and running `git reset --hard`, so provenance
  records a genuine commit. Build the tarball with `COPYFILE_DISABLE=1` on macOS
  or it carries `._*` AppleDouble files that land untracked and flip `git_dirty`.
- **Container runs as root**, so artifacts under `results/` and `data/` come out
  root-owned on the host. `sudo chown` before `gcloud compute scp`.
- **Timings on A100:** model load ~45 s; one Boltz-2 structure prediction
  (600 steps) ~45 s; one scored mutant in `signal_control` ~40 s; a full
  `07_sanity` run ~25 min; one backward pass at L=730 ~104 s; Docker rebuild
  ~30 s when only the `pip install -e` layer is invalidated.
- **Rebuild cheaply:** put new `apt`/`pip` layers *after* the torch and boltz
  installs, or you re-download multiple GB.

### `bootstrap.sh` gotchas

1. **`docker/Dockerfile` — `COPY . /app`**. Must be built from the repo root
   with `-f docker/Dockerfile`. An earlier `COPY .. /app` reached outside the
   build context and silently copied nothing.
2. **`get_device_properties(0).total_memory`** (not `.total_mem`). The wrong
   attribute name produces `AttributeError` inside the container, so the
   verification JSON never prints and bootstrap dies with the misleading
   "Verification script produced no parseable output".
3. **Bare `docker` calls fail** because `usermod -aG docker` does not take
   effect in the session that runs it. `bootstrap.sh` probes for this and
   falls back to `sudo docker`.

### Diagnostics on the VM

`~/IG/probe_mem.py` (VRAM at each stage + OOM traceback) and
`~/IG/probe_snap2.py` (allocation-trace replay attributing live bytes at peak by
call site). Both gitignored. `probe_snap2.py` is the one that turned the memory
investigation from guesswork into measurement.

---

## 4. Traps

- **Only one A2 machine at a time** (`A2_CPUS=12`). Never run the VM and a Colab
  runtime together.
- **Stage 03 is not resumable.** Do not use spot/preemptible; a preemption kills
  a 20–40 min run with no recovery.
- **Use the path-averaged gradient, never the `(x − baseline) × grad` product**,
  for per-substitution predictions. `attrib.py:206-219` documents why. The
  `(x − baseline)` factor belongs only to the completeness identity.
- **`plain_gradient` is not `integrated_gradient(m_steps=1, "uniform")`** — the
  latter averages alpha=0 and alpha=1.
- **`compute_ptms` swallows exceptions** and returns zeros with only a bare
  `print`. `boltz_score.py` asserts non-zero before backward; keep that.
- **Never write to `/Users/bowmannovey/Agros/IG_Agros`** — read-only predecessor
  reference.
- **`PLAN.md` must not be modified.**
- **`signal_control` is the load-bearing sanity check.** Boltz-2 gives uniformly
  high confidence regardless of biological relevance. If the score itself doesn't
  move under mutation, its gradient can't either, and a null result is
  uninterpretable. The check passes (std=1.492e-02, n=30).
- Trust `assert_provenance` on the recorded `arm`, not on the flags you passed.
  The predecessor's two worst bugs were silent no-ops found only by auditing
  artifacts.
- **Do not switch the outer checkpoint to `use_reentrant=False`.** It is not a
  style choice. Reentrant mode is load-bearing for memory — the non-reentrant
  path builds the full forward graph and OOMs immediately. See `ERRORS_LOG.md`
  entry 9.
- **Always pass `--project=agrosbio` to every `gcloud` command.** The local
  default config points at `g-prs-478707`, which has the Compute Engine API
  disabled. A command missing `--project` fails with a confusing
  `PERMISSION_DENIED` and then *offers to enable Compute Engine on that other
  project*. Do not accept that prompt. The project id is recorded nowhere in
  `gcloud config` — only in this file and `docs/CLOUD.md`.
- **`run_all.sh` is AbBiBench-only.** For a SKEMPI or DMS complex use
  `run_complex.sh`. Running `run_all.sh` on either will fail in stage 00 or 01.
- **The VM cannot `git pull`** — no GitHub key in `~/.ssh`. Use the `git bundle`
  recipe in section 0. Symptom order is misleading: `Host key verification
  failed` first, then `Permission denied (publickey)` once the host key is
  added. Fixing the first does not fix the second.
- **`data/raw/` and `notes/` are gitignored**, so PDB files and every planning
  document are absent from the VM. Fetch structures with `curl` from RCSB
  before a run; never expect to read `logging.md` on the box.
- **The boltz featurisation cache substitutes silently across datasets, and has
  actually done so** (`ERRORS_LOG.md` entry 34). `process_inputs` skips any
  input whose YAML stem is already processed, the stem is always `input`, and
  `data/raw/boltz_homopolymer/<AA>` does not encode the dataset — so a stale
  `processed/` returns another protein's tensors while the freshly written
  `input.yaml` looks correct. **If you change complex, delete
  `data/raw/boltz_*` for the paths you are about to use.** `_build_token_map`'s
  run-length check is the only thing standing between this and a silently
  wrong attribution.
- **Dry-run first, every time.** `bash scripts/run_complex.sh --dry-run` is free
  and caught three of the four defects in section 2.1.
- **`complex_pde` is lower-is-better; the other five scores are not.**
  `12_compare.py` derives orientation from the score name
  (`LOWER_IS_BETTER = {"complex_pde"}`). Hardcoding a sign anywhere else will
  invert a conclusion while still producing a plausible number.
- **Starr `bind_avg` is positive for *tighter* binding; SKEMPI ddG is positive
  for *weaker*.** Everything must go through `igv.dms.binding_ddg`, which
  negates onto SKEMPI's convention.
- **Known minor test gap:** nothing pins `interface_positions`' cutoff boundary
  at exactly 5.000 A — a `<=` to `<` mutation survives the suite. Measure-zero
  in practice, recorded rather than fixed.
- **Every claim in this repo's docs should be traceable to a command.** The
  Phase 3 estimate of 931 mutants sat in `logging.md` for days and was wrong;
  `04_scan.py --dry-run` says 532. Prefer printing the number to estimating it.
