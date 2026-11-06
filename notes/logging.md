# Logging — gradient attribution on Boltz-2, measured against experiment

Two validation arms. **Alanine scanning** (SKEMPI, sections 1-8) asks which
*positions* matter and is confounded by side-chain size; **saturation**
(Starr 2020 on 6M0J, section 10) asks how each of the 20 *amino acids*
performs and is not. Section 9 is the evidence for moving to the second.

Supersedes the target selection in `HANDOFF.md`. `PLAN.md` is the original
proposal and must not be modified; this document records what changed and why.

---

## 1. The claim

> ~32 runs of Boltz-2 can predict how every possible mutation changes binding at
> a protein-protein interface — closely enough to replace the ~2,400 full runs it
> would otherwise take, and closely enough to match measured binding energies.

This is a validation project. A clean negative is a result.

### The three comparisons, in plain terms

The repo previously called these T1/T2/T3. Use the plain names.

| Name | Compares | Answers |
|---|---|---|
| **shortcut vs slow way** | attribution vs Boltz-2's own brute-force scan | Does the shortcut reproduce the model? |
| **model vs lab** | Boltz-2's scan vs measured ddG | Is the model right about biology? |
| **shortcut vs lab** | attribution vs measured ddG | What does a practitioner get? |

The first is the honest test of the shortcut: it holds whether or not Boltz-2
knows any biology. The third is bounded above by the second.

---

## 2. Why the target changed

### AbBiBench's `4fqi_h1` is the wrong shape

Characterised directly from `data/raw/4fqi_h1_benchmarking_data.csv`:

- 65,094 variants over exactly **16 positions** of 121, one alternative amino
  acid each
- modal variant carries **8 simultaneous mutations**; only 16 single mutants
  exist in the whole file
- scores are **left-censored**, 1,675 variants piled at the assay floor of 7.000

It cannot validate a per-amino-acid prediction, because it never varies the
amino acids. Spearman(mutation count, score) = **-0.22**, so Boltz-2's score
there is genuine per-position signal rather than a mutation-count artifact --
the dataset is sound, just narrow.

### A number in `HANDOFF.md` was wrong

`HANDOFF.md:252` claims Boltz-2 scores 0.71 on `4fqi_h1`. From the published
leaderboard (`github.com/MSBMI-SAFE/AbBiBench`, README):

| Dataset | Boltz-2 | ProteinMPNN | ESM-IF1 | FoldX |
|---|---|---|---|---|
| `3gbn_h1` | **0.71** | 0.59 | 0.59 | 0.59 |
| `4fqi_h1` | **0.40** | 0.61 | 0.65 | 0.64 |

The 0.71 belongs to `3gbn_h1`. On `4fqi_h1` Boltz-2 is beaten by every cheap
alternative. We had picked as go/no-go a dataset where the model under study
loses.

Also worth recording: AbBiBench scores every model by **log-likelihood of the
sequence given the structure**, and there is no Boltz-2 code in their public
repo (14 models have `get_model_log_likelihood.py`; Boltz-2 and AF3 do not). So
what quantity they extracted from Boltz-2 is not verifiable, and our
`complex_pde` may not be comparable to their 0.13. Do not claim comparability
without checking.

### SKEMPI 2.0 has what we need

7,085 entries over 348 complexes, **5,112 single-point**, **2,961 to alanine**,
with wild-type and mutant Kd. Gives per-position *and* per-amino-acid
resolution, continuous ddG in kcal/mol, and it is not blocked by the `ref_pos`
defect (see section 4).

---

## 3. Target: 1JTG, with 3HFM as the second complex

Ranked by usable depth, all SKEMPI single-point mutations with computable ddG:

| Complex | Positions | Mutations | Hot spots | aa/position | Partners |
|---|---|---|---|---|---|
| **1JTG** | **49** | 138 | **13** | 2.8 | TEM-1 beta-lactamase / BLIP |
| 3S9D | 75 | 156 | 10 | 2.1 | Interferon alpha-2 / receptor |
| 1A22 | 133 | 212 | 7 | 1.6 | Human growth hormone / receptor |
| 1AO7 | 60 | 136 | 11 | 2.3 | TCR / pMHC |
| 3HFM | 27 | 96 | 13 | 3.6 | HyHEL-10 / HEW lysozyme |

**Primary: `1JTG`.** Nearly double 3HFM's positions, the same 13 hot spots,
almost the same per-amino-acid depth, and *smaller* once the duplicate chains
are dropped.

**Secondary: `3HFM`**, plus `1VFB` / `1JRH` / `2JEL` if the pooled hot-spot arm
needs more positions (27 alone is thin).

### 1JTG setup facts, verified

- **Four chains: A(262) B(165) C(261) D(165), total 853 tokens.** A/B and C/D
  are two copies of the complex in the asymmetric unit. **Use A+B only = 427
  tokens**, below 3HFM's 558 and well below `4fqi_hlab`'s 730 (which needs
  55.2 GiB in bf16). fp32 may fit.
- Mutation split: chain A (beta-lactamase) 42 muts / 21 positions / 2 hot spots;
  **chain B (BLIP, the inhibitor) 96 muts / 28 positions / 11 hot spots**. Chain
  B is where the signal is.
- **138/138 mutations map to residue indices with zero wild-type mismatches**,
  using the correct numbering column.

### The numbering bug -- fix before anything else

`scripts/10_skempi_hotspots.py:97` reads `Mutation(s)_cleaned`. That is
*sequential* numbering. `Mutation(s)_PDB` is what matches a deposited structure.

| Complex | Rows where the two columns agree |
|---|---|
| 3HFM | 96/96 |
| 1VFB | 56/56 |
| 1JTG | 96/138 |
| 1A22 | 54/212 |
| **3S9D** | **0/156** |

On 1JTG, `cleaned` gives 41 wild-type mismatches; `PDB` gives zero. 3HFM passed
only because its two schemes coincide -- that clean 96/96 was luck, not
validation. The wild-type guard caught this, which is the whole reason it exists
(cf. `ERRORS_LOG.md` entry 17).

---

## 4. Blockers

### `ref_pos` nondeterminism -- RESOLVED and VERIFIED 2026-09-24

Was: boltz featurisation differed by up to **10.7 Angstrom** in `ref_pos`
between two runs of the identical input, contaminating every `s_mut - s_ref`
delta and the brute-force scan alike.

Root cause was **not** the RDKit conformer generation the original entry blamed.
It is `featurizerv2.py:1467-1473`, which applies a random rotation and
translation per residue via `torch.randn`, while `inferencev2.py:270` seeds numpy
only and never calls `torch.manual_seed`. `src/igv/deterministic.py` patches
`center_random_augmentation` to `augmentation=False` -- keeping the centring,
dropping the roto-translation, which is training-time augmentation with no place
at inference.

**Verified on the VM**, not inferred: 78/78 feature tensors byte-identical across
two runs, and reference-vs-mutant `ref_pos` changes confined to the mutated token
with all 421 others identical. `ERRORS_LOG.md` entries 20 and 18.

### Substitution enumeration -- RESOLVED

`--positions skempi|<list>` enumerates all 19 substitutions at chosen positions;
the `assert n_deltas < 500` is now `--max-substitutions` (default 5000). 1JTG
chain B gives 532 substitutions at 28 positions.

### Stages 02 and 03 featurisation -- PARTIALLY ADDRESSED

Both now key their boltz cache on dataset (entry 24), so neither can silently
receive another complex's features. But stage 03 still featurises with the MSA
server while stage 02's reference path differs, so confirm they expand around the
same point before trusting a delta-based prediction.

### The 2.0 tolerance in `02_embed_deltas.py` -- can now be tightened

It was set against a measured ~1.5 noise scale that no longer exists. With
featurisation deterministic, this should return to something strict.

---

## 5. Plan

### Phase 0 -- no GPU -- ALL DONE except item 7

1. ~~Switch to `Mutation(s)_PDB`~~ done, with a regression test.
2. ~~Seed conformer generation~~ done and verified on the VM (section 4).
3. ~~Enumerate all 19 substitutions~~ done, `--positions` + `--max-substitutions`.
4. ~~Chain subsetting for 1JTG~~ done, 427 tokens, C/D excluded.
5. ~~Confound controls~~ done, and they set a bar the attribution has not cleared
   (section 6).
6. ~~Fix `HANDOFF.md` / `README.md`~~ done.
7. Characterise the remaining 16 AbBiBench datasets -- **in progress**.

### Phase 1 -- precheck (~$1)

One `plain_grad` backward pass on 1JTG chain B. Cheap look for any signal before
committing to longer runs. Go/no-go.

### Phase 2 -- IG (~$10)

`m=32` with a convergence check at 16 and 64, so "32 was enough" is demonstrated
rather than asserted. Produces the per-amino-acid grid.

Completeness is **load-bearing here**: `integrated_gradient` returns
`grad = accumulated_grads`, the path-averaged gradient, and `score_deltas` uses
exactly that array (`src/igv/attrib.py:270-277`, `:393-404`). If the quadrature
under-resolves, every prediction is wrong. m=32 passed at L=554 (0.0442);
1JTG A+B is 427 tokens, so it should hold comfortably.

### Phase 3 -- brute force (~$13)

Scan **all 19 substitutions at the 28 measured positions = 532 mutants**. The
"shortcut vs slow way" comparison needs no lab data, only the model, so scan as
widely as affordable. A full saturation scan of chain B (165 x 19 = 3,135) is
the eventual ideal at roughly another 20 GPU-hours -- hold it until Phase 1
shows something.

Corrected 2026-09-27: this section previously said 49 positions and 931
mutants. `skempi_positions("1JTG", "B")` returns **28**, so the real figure is
532 and the phase is ~43% cheaper than first estimated. Measured, not
re-estimated -- `python scripts/04_scan.py --dataset 1JTG --chain B --positions
skempi --dry-run` prints it.

### Phase 4 -- compare -- IMPLEMENTED 2026-09-27

`scripts/12_compare.py`. Until today this phase had no implementation and
stage 04's scan CSV was written and read by nothing, so the most expensive GPU
stage in the pipeline produced an artifact that was never compared to anything.
See entry 29.

It reports the triple that is the actual argument -- attribution vs experiment,
scan vs experiment, attribution vs scan -- plus the within-position split on a
DMS dataset and the forward-versus-backward pass counts, which is the practical
case for replacement. Score orientation is derived from the score name, since
`complex_pde` is lower-is-better and the other five are not; verified by feeding
a scan that agrees perfectly with experiment under each score's own semantics
and confirming both report positive.

All on the same mutations: shortcut vs slow way; shortcut vs lab; model vs lab;
and IG vs the single-pass precheck. That last one is a finding either way -- if
they agree the expensive version is unnecessary, and if they diverge the
function is strongly non-linear between baseline and input, which the alpha
profile already suggests (53% of the score change in the first 5% of the path).

### Cost

Anchored on the one measured number, 104.2 s per backward at 730 tokens. 1JTG
A+B is 427 tokens, so passes are cheaper than the 3HFM estimates. Roughly 12
GPU-hours core, ~18 with realistic slack for the OOMs and reruns this project
has hit in every prior session. At $5.07/hr that is **$60-90**.

The saturation arm (section 10) is on top of that. 6M0J is ~791 tokens
against 1JTG's 427, so budget ~1.5-3 h per 32-step IG run rather than ~40
min, and size the position set to the budget rather than running all ~200.

---

## 6. Controls we owe

From Adebayo (*Sanity Checks for Saliency Maps*, NeurIPS 2018) and from
arXiv:2606.22181, whose failure mode is the one most likely to bite us.

**Already passing** (at L=554): `random_weights` rho=0.077 -- randomising 5,035
parameter tensors destroys the attribution, so the signal depends on learned
weights. `dead_target` 0.0.

**The confound panel -- built, and it has already set the bar.** The
allergenicity paper found their classifiers relied on *"physicochemical and
compositional sequence features rather than epitope-specific mechanisms."*
Measured on **1JTG chain B**, 28 positions, from the PDB alone with no model
involved (`ERRORS_LOG.md` entry 19):

| Confound | Spearman vs ddG |
|---|---|
| **burial** (heavy atoms within 10 A) | **+0.54** |
| **residue volume** | **+0.48** |
| distance to binding partner | -0.32 |
| hydrophobicity | -0.14 |
| normalised position in chain | +0.11 |

**Counting nearby heavy atoms predicts binding energy at rho ~0.5.** An
attribution scoring 0.5 against ddG tells us nothing a distance calculation
would not. So the **partial correlation** -- does the gradient predict ddG after
burial and residue size are regressed out? -- is the load-bearing number, not
the raw Spearman. Report both.

Two caveats on that bar. It is sensitive to aggregation (+0.54 max signed, +0.59
max |ddG|, +0.47 mean), so always state which. And its bootstrap 95% CI is
**[+0.21, +0.78]** at n=28 -- wide enough that pooling 3HFM, 1VFB, 1JRH and 2JEL
moves from optional to close to mandatory before any claim is made. Distance to
partner looks weak only because SKEMPI measured interface residues exclusively,
so the variable cannot discriminate; that is range restriction, not evidence the
interface is unimportant.

`scripts/10_skempi_hotspots.py` now reports the panel, the partial correlation,
AUROC / AUPRC alongside precision@k, a shuffled-ranking null, and bootstrap CIs.
Matching arXiv:2606.22181's metrics makes the comparison to them direct.

---

## 7. Positioning

Do **not** claim first-to-validate, and do not aim at the leaderboard. IG
approximates Boltz-2's own scan, so "shortcut vs lab" is bounded by Boltz-2's
0.13 average -- ProteinMPNN's 0.30 is out of reach by construction, and
ProteinMPNN is also already fast, so the speed argument does not beat it either.
What the shortcut is 77x faster than is brute-force scanning *with a co-folding
model*.

**arXiv:2606.22181** (ICML 2026 workshop) is commonly miscited, including in
this repo, as "gradients on proteins, negative". Read it properly: they
benchmarked attribution on protein *allergenicity* classifiers against annotated
epitopes, and found

> *"Integrated Gradients identified residues that were functionally important to
> the model, but not overlapping annotated epitopes"*

That is the attribution **working** and the model relying on compositional
shortcuts -- the "shortcut is honest, model is wrong" quadrant. It supports this
work rather than contradicting it. Their ground truth was binary epitope
annotations and in-silico mutagenesis; ours is measured ddG in kcal/mol. Frame
as: *attribution on sequence-only models recovered model-internal features but
not biology; does a model that explicitly builds the interface do better, judged
against binding energies rather than annotations?*

Others to engage: **TISM** (iScience 2024, the method, genomics only, never
compared to experiment, Pearson 0.70 is the number to beat), **ProtDBench**,
**BindEnergyCraft**, **GearBind**, **Majdandzic** (Genome Biology 2023,
off-simplex contamination), the **AF3/SKEMPI** paper (NeurIPS 2024).

Venue: MLSB, 5 pages excluding references, welcomes work in progress.

---

## 8. Status

Committed through `4962fbc`. **549 tests, ruff 11.** Working tree clean on
`main`. VM `igv-gpu` TERMINATED, 500 GB pd-ssd retained at ~$2.83/day.

**No GPU result is newer than 2026-09-24.** The 2026-09-27 session was entirely
local: it built the saturation arm, fixed four pipeline defects, and never ran
a model. The table below is unchanged and is still the only measured science in
the project.

### Measured on 1JTG chain B, 2026-09-24 (427 tokens, fp32, ~$24 of A100)

| run | Spearman | null p95 | above null | partial | AUROC |
|---|---|---|---|---|---|
| plain_grad (zeros) | 0.306 | 0.318 | no | 0.180 | 0.685 |
| IG m=32 (zeros) | 0.238 | 0.336 | no | 0.010 | 0.583 |
| **IG m=32 (mean_aa)** | **0.357** | 0.313 | **YES** | 0.211 | **0.695** |

**The baseline was the cause of the early nulls.** `complex_pde` is a predicted
distance error; the all-zeros baseline is not a protein and scores 11.4310 where
the real complex scores 2.6200, so the integral was dominated by off-manifold
behaviour. `--baseline mean_aa` drops f(baseline) to 3.2491 and cuts the
integral's **absolute** error 30x (1.5222 -> 0.0504). See `ERRORS_LOG.md`
entry 21.

**Do not oversell the positive.** CI [-0.002, 0.656], excluding zero by two
thousandths. Partial correlation 0.211 (CI [-0.34, +0.56]) against burial's
0.436. Gradient-vs-hydrophobicity is -0.527 with CI [-0.75, -0.18], which
excludes zero while gradient-vs-ddG barely does.

**Determinism is proven, not inferred.** `verify_deterministic_feats.py` exits 0
on the VM: 78/78 tensors byte-identical across two runs, and reference-vs-mutant
`ref_pos` changes confined to the mutated token with all 421 others identical.
Entry 18 is closed (entry 20).

**Also settled:** path integration is not a rounding error on the plain
gradient — Spearman +0.58 between the two per-residue rankings, only 3 of the
top 10 residues shared, IG norms 3.6x larger. `plain_grad` is not a cheap
substitute for `ig`.

**Still not done:** the brute-force scan has never run, so "shortcut vs slow
way" — the headline claim — has no number. Completeness fails the 5% *relative*
gate at 8.01%, though see entry 21 on why that gate is miscalibrated for a tight
baseline.

VM `igv-gpu` is TERMINATED; the 500 GB pd-ssd is retained at ~$2.83/day.

---

## 9. The data problem, and where it points

**Alanine scanning cannot validate the project's actual claim.** SKEMPI is ~90%
X->Alanine, so it tests *which positions matter*, not *how each of the 20 amino
acids performs*. Worse, the confound is mechanistic rather than statistical:
1JTG chain B's 11 hot spots are W, K, H, F, H, Y, W, R, F, D, W — all large
aromatic or charged residues — and X->A *deletes a large side chain*. Residue
volume correlating with ddG at +0.44 is therefore what the assay measures, not
noise to regress away. Any method that learns "large aromatic at an interface
matters" scores respectably without representing binding. See entry 22.

**SKEMPI's counts are replicates, not breadth.** 1JTG chain B's 96 mutations are
28 distinct mutations measured repeatedly; W150's five W->A values span
+3.50..+4.81, putting experimental noise at ~±0.6 kcal/mol. Effective n is 28,
which is why every CI above is so wide. Entry 23.

**Saturation data is the fix, and it is measured, not argued.** With all 20
substitutions at one position, the comparison becomes *within-position*, so
burial, exposure and distance-to-partner are constant across the row and cancel
identically. Verified on real saturation data (GB1, below):

| Design | Residue volume vs binding effect |
|---|---|
| Pooled (what we did on SKEMPI) | **+0.44** |
| Within-position | **+0.135** mean, **+0.197** median |

And the within-position correlation *changes sign* position to position — 20 of
54 positions negative, range -0.92..+0.71 — so on pooling it cancels rather
than accumulating. The confound panel is not something to regress away; it is
something the experimental design removes. This is the single most important
methodological result in the project so far.

### GB1/IgG-Fc — assessed and rejected

Olson et al. 2014 (Current Biology). Verified locally: 1045 singles = exactly
55 positions x 19 substitutions, every position saturated. It is cheap (56
residues) and it is what produced the within-position numbers above. Rejected
as the primary saturation target on four counts:

1. **The hot spots are censored out.** The floor is exactly `ln(0.01)`, with 87
   values pinned there. Position 27 has **std 0.000 — all 19 substitutions at
   the floor**; position 43 (Trp) has 17/19. 13 of 55 positions are flat
   (std < 0.3). Within-position ranking is impossible at precisely the residues
   alanine scanning calls hottest, which are the ones we most need to rank.
2. **Folding and binding are coupled.** The authors define `W = f_N * W_N`, so a
   destabilised variant reads as a non-binder. For a *structure* model like
   Boltz-2, "the gradient tracks foldability" is an a priori likely confound,
   which makes a stability control mandatory rather than optional. GB1 needs a
   cross-paper merge (Nisthal 2019 PNAS folding ddG) to approximate one.
3. **The structure is the wrong paralogue.** 1FCC is protein G **C2** domain
   (P19909), not B1 (P06654); 3 of 56 positions differ, giving 57 wild-type
   mismatches, and 988/1045 mutants map cleanly. No better structure exists in
   the PDB. Numbering offset is exactly -226.
4. **`lnW` is not ddG**, so GB1 results cannot be pooled with, or plotted on the
   same axis as, anything from SKEMPI.

Structural trap recorded in case GB1 is ever revisited: in 1FCC chains A and B
are an **obligate Fc homodimer 2.34 A apart**, not redundant copies like 1JTG's
C/D. Dropping B to save tokens models a half-molecule that does not exist in
solution. Use A+B+C = 468 tokens, drop only D.

### AbBiBench saturation scans — why not

`2fjg` and `g6_LC` *are* complete saturation scans — `2fjg` verified at 2223
rows = 117 positions x 19, all singles, consensus == PDB wild type. The problem
is the other side of the ledger: Boltz-2's own per-dataset score on `2fjg` is
**0.08**. Across AbBiBench, Boltz-2's score correlates *inversely* with how much
per-amino-acid structure a dataset has — 0.71 on `3gbn_h1`, which has 11
positions x 1 alternative, no true singles, and an 11% floor. The datasets
Boltz-2 looks good on are the ones that cannot test our claim. Recorded so the
0.71 is never cited as encouragement.

## 10. The saturation arm: SARS-CoV-2 RBD / ACE2

**Decision (2026-09-27): go straight to Starr et al. 2020 on 6M0J. Skip GB1
entirely.** The within-position machinery has to be written either way, and
writing it against the dataset that can actually carry the claim avoids
building it twice. GB1's cheapness does not offset a censored hot-spot set, a
wrong-paralogue structure, and a readout that cannot be put on the same axis as
our SKEMPI results.

`SPIKE_SARS2_Starr_2020_binding` — Tite-seq on the RBD, all 20 amino acids at
essentially every position.

| | GB1 | **SPIKE/6M0J** |
|---|---|---|
| Singles | 1045 | **~3802** (~99.6% saturation) |
| Readout | enrichment ratio (`lnW`) | **delta log10 KD — real affinity** |
| Convertible to ddG? | no | **yes**, ~1.36 kcal/mol per log10 |
| Stability control | cross-paper merge needed | **same paper, same assay, same variants** |
| Structure | wrong paralogue, 3.2 A | 6M0J, 2.45 A, single copy |
| Tokens | 468 | ~791 |

The matched **expression** readout is the decisive advantage. It is the
folding-vs-binding decomposition GB1 needs a separate publication to
approximate, available for the identical variants in the identical assay. Since
the likeliest failure mode for a structure model's gradient is that it tracks
foldability, having that control in-file converts a fatal ambiguity into a
measurable one.

Cost is the only objection: 791 tokens against 1JTG's 427, so roughly 1.5-3 h
per 32-step IG run rather than ~40 min.

### Setup facts — VERIFIED 2026-09-27, independently of the loader

- Dataset: 4221 rows, of which **3802 singles** over 201 sites (331-531). 198 of
  201 sites carry all 19 substitutions; the other three carry 15, 14 and 11.
- The `raw.githubusercontent.com` path returns a **Git LFS pointer, not the CSV**.
  Use the `media.githubusercontent.com/media/...` URL.
- 6M0J chains: **A = ACE2, 597 residues; E = RBD, 194 residues**, author numbering
  333-526 and contiguous. Total **791 tokens**. No other protein chains. HETATM
  entities our parser drops: Zn, Cl, NAG, HOH.
- **Zero wild-type mismatches** across all 194 mapped sites. Contrast 1JTG, where
  the wrong numbering column gave 41 silent mismatches (entry 20).
- Chain E stops at 526, so sites 331, 332 and 527-531 have no structure.
  **133 singles are unusable; n = 3669 over 194 positions.**
- Sign convention confirmed in code: `binding_ddg(-1.0) = +1.36` kcal/mol, i.e.
  weaker binding maps to positive ddG, matching SKEMPI.
- **Censoring is not a problem.** `bind_avg` has one row at its minimum and 146
  within 0.1 of it, spread over 40 positions; only site 497 has as many as 10 of
  19 near the floor. Nothing resembling GB1's position 27 (19/19 pinned).
  `expr_avg` has exactly one row within 0.1 of its minimum.

### The within-position design, measured on this dataset

| Confound, within-position | mean | median | negative |
|---|---|---|---|
| Residue volume vs binding | **-0.034** | -0.045 | 109/201 |
| Hydrophobicity vs binding | **-0.006** | -0.084 | 108/201 |

Pooled on this dataset, volume vs binding is **+0.027** and volume vs |binding|
is **-0.020**. So the +0.44 measured on SKEMPI is not a property of proteins; it
is a property of the *alanine-scan design*, which confirms entry 22 by
construction rather than by argument.

### Folding vs binding: the confound is real, and it is confined to where it does not matter

This is the reason to use this dataset, and the global number is misleading.
Pooled, binding and expression correlate at **+0.64** (r2 0.40), and
within-position the median is **+0.75** — a structure model whose gradient
tracks foldability would score well on binding for the wrong reason. But split
by distance to ACE2:

| | n positions | bind~expr median | bind std | expr std |
|---|---|---|---|---|
| **Interface (<=5 A)** | 21 | **+0.38** | **0.695** | 0.275 |
| Non-interface | 173 | +0.752 | 0.338 | 0.481 |

Pooled over interface mutants only, binding and expression correlate at
**+0.074**, and residualising binding on expression keeps **100%** of the
binding variance (1.224 -> 1.221). At the interface the two readouts are
effectively orthogonal: binding varies most and expression varies least, while
away from the interface the reverse holds. The folding confound is therefore
not something to fight — it lives almost entirely outside the region the claim
is about, and the expression column lets us demonstrate that rather than assume
it.

**Consequence: the primary analysis is interface positions only, within
position, with expression reported as a control.** Also note 80 of 194 positions
have binding std < 0.3 and carry no rankable signal; the interface median is
0.695.

### Two calibration numbers, measured through stage 11 on 2026-09-27

**The volume confound is only *mostly* cancelled on the 21 positions we will
actually analyse.** The headline -0.034 is the mean over all 194 positions. On
the interface subset alone it is **+0.092 mean, +0.245 median**, CI
[-0.082, +0.266]. The CI includes zero and n is 21, so this is not a finding —
but it is not the -0.034 either, and the interface number is the one that
belongs next to any interface result. Report it.

**A *perfect* binding predictor scores +0.355 against expression.** Feeding
stage 11 the observed `bind_avg` as its own prediction gives
`pred_vs_bind = +1.000` by construction and `pred_vs_expr = **+0.355**`, because
binding and expression are themselves correlated at +0.38 within interface
positions. So a gradient correlating ~0.35 with expression is **not** evidence
of a folding confound — it is exactly what a pure binding predictor does here.
The diagnostic is the *gap*: `pred_vs_bind` clearly above `pred_vs_expr` is the
good case; the two roughly equal is the bad case. Without this reference value
a perfectly clean result could be read as contaminated.

### Position sets, and why only brute force pays for them

| Cutoff to ACE2 | Positions | Mutants |
|---|---|---|
| <= 4 A | 17 | 323 |
| **<= 5 A** | **21** | **399** |
| <= 6 A | 30 | 570 |
| <= 8 A | 46 | 874 |
| <= 10 A | 60 | 1140 |

All 17 literature ACE2 contact residues fall inside the 5 A set, including the
variant-of-concern positions **K417, E484, N501**, which is independent evidence
the cutoff is picking out the real interface rather than an arbitrary shell.

One IG run yields attributions for **every** position at once, so the position
count is free on the gradient side — 791 tokens is the whole cost. Only the
brute-force arm scales with positions, at 19 forward passes each: **399 mutant
predictions for the 5 A set**. Start there; widen to 8 A only if 21 positions
proves too few to separate the methods.

### Next, in priority order

1. **Pool 3HFM, 1VFB, 1JRH, 2JEL** (~$12, ~2.5 h). All registered and verified.
   Takes n from 28 to ~128 and directly tests whether 0.357 survives
   replication. Do this **before** the scan: if the effect does not reproduce,
   0.357 was n=28 noise and we learn it cheaply.
2. **The brute-force scan on 1JTG** (~$22, ~4 h). Mandatory — the headline claim
   is unpublishable without it, and it separates "the method fails" from
   "Boltz-2 has no binding signal here". Only one of those is fixable.
3. **The saturation arm on 6M0J.** Dataset and structure are verified (above);
   `src/igv/dms.py` loads them. Remaining local work: wire the 21 interface
   positions into stages 02/03 and add the within-position scoring to
   `06_metrics.py`. Then one IG run at 791 tokens gives all 194 positions x 20,
   and the brute-force comparison is 399 forward passes.
4. **Re-gate completeness** on absolute error -- **DONE 2026-09-29**, see
   `ERRORS_LOG.md` entry 32. Gates on `COMPLETENESS_ABS_THRESHOLD = 0.10`; the
   8.01% relative "failure" was the gate punishing the better baseline, and it
   blocked nothing.

Not yet justified: a different attribution method. Item 2 is what establishes
whether the method is at fault.

**Superseded in part by section 11**, which revises this list against the
literature and against a re-analysis of the saved 1JTG artifact.

---

## 11. What the field already knows, and what our own data says

Surveyed 2026-09-29, no GPU. Two things came out of it: the bar is better
defined than we thought, and our own saved artifact says something we had not
looked for.

### The baselines we are actually measured against

**RSALOR** -- Tsishyn, Hermans, Rooman & Pucci, *Bioinformatics* 41(6):btaf322
(2025), "Residue conservation and solvent accessibility are (almost) all you
need". `(1 - RSA) x LOR`, two features, **zero trainable parameters**. Average
Spearman **0.473** across 217 ProteinGym DMS datasets, matching or beating 27
deep predictors including ESM-2, SaProt, EVE and GEMME. **RSA alone scores
0.356.** Our confound panel (`src/igv/skempi.py:519`) has burial, distance,
hydrophobicity, volume and normalised position -- but **no conservation term**.

**Corrected 2026-09-29: we hold neither RSALOR factor, not "half".** The
panel's `burial` is not RSA. `compute_burial` (`src/igv/skempi.py:450-482`)
counts neighbouring heavy atoms within 10 A -- a spatial density, as its own
docstring says. RSALOR is `(1 - RSA) x LOR`, so a real comparison needs **two**
new terms: a genuine relative solvent accessibility (DSSP or FreeSASA) and the
log-odds-ratio conservation term. Costing this as "one MSA fetch" understates
it. Also note **no MSA is on disk** and the only implemented fetch route runs
inside boltz (`src/igv/boltz_score.py:845`, via `build_complex_feats(
use_msa_server=True)`), which is not installed locally -- so this is not a
local CPU-only change either. It needs a VM session or new fetch code.

**SKEMPI per-interface Spearman**, the protocol that matters. Pooled numbers run
~0.3 higher because between-complex variance dominates, so any comparison must
state its protocol:

| Method | Per-interface rho |
|---|---|
| FoldX | 0.37 (name-split) / 0.48 (cluster-bootstrapped) |
| Flex ddG | 0.42 |
| Best learned, leakage-controlled (ProSST+ProtBFF) | 0.48 |
| Best learned, name-split (BA-DDG) | 0.51 |
| B-factor baseline | 0.169 |
| ESM-1v | **-0.012** |

Two further floors: a trivial predictor returning the mean ddG **for the
mutation type alone**, no structure at all, reaches Pearson **0.46**; and the
experimental noise ceiling is Pearson **~0.89**, so published 0.91s are overfit
by construction. Sequence-only pLMs are at zero per-structure -- that is the
company we are *not* in, and worth saying explicitly.

### The negative result closest to ours

Yao, Song, Baerenfaller & Zhakparov, arXiv:2606.22181 -- already cited in
section 6 as the failure mode most likely to bite us, and it is. Six attribution
signals including IG against IEDB epitopes: **no model-derived attribution
exceeded its random baseline**; IG scored AUROC **0.476 against random 0.501**.
But IG *passed* their faithfulness test -- masking top-IG residues moved the
prediction, p<0.001. **Model faithfulness and biological faithfulness fully
dissociate.** Our `random_weights` control (entry 15) tests the first; SKEMPI
tests the second. Report them separately and name them as such.

Also relevant: King et al., arXiv:2512.06592 (MLSB 2025) fine-tuned Boltz-2 for
protein-protein affinity and found it "underperforms relative to sequence-based
alternatives in both small- and larger-scale data regimes". Combined with the
affinity head having been trained predominantly on protein-ligand data, whether
it carries PPI energetic signal at all is a prior question we should state up
front rather than have a reviewer raise it.

### What the niche looks like

Three independent searches found **no published work applying gradient
attribution to a co-folding model's affinity head** -- not Boltz-2, not
AlphaFold3, not Chai-1. ExplainableFold (KDD 2023) is the nearest neighbour and
is counterfactual, on structure rather than affinity. PairSAE (arXiv:2606.27440)
is on Boltz-2 but uses sparse autoencoders and no gradients. The genomics
community solved the discrete-input attribution problem years ago -- shuffled
references, the simplex correction, ISM validation -- and none of it has been
carried into structure prediction. **Carrying it over carefully is itself the
contribution.**

### Measured on our own artifact, 2026-09-29

From `results/1JTG_hotspots_ig_meanaa.json`, n=28 positions, all against
**|ddG|** unless stated. Recorded because two earlier readings of these numbers
were wrong in opposite directions.

- **Burial does not significantly beat IG.** Burial 0.436, IG 0.357. Paired
  bootstrap of the difference over 20,000 resamples: **+0.199, CI95
  [-0.193, +0.600], P(burial > IG) = 0.84.** The burial figure of +0.54 quoted
  in section 8 is against *signed* ddG; against |ddG| it is 0.436. Compare
  like with like.
- **They are not measuring the same thing.** Spearman(IG, burial) = **+0.315**.
- **Their partial correlations are near-identical:** IG controlling for burial
  **+0.245**; burial controlling for IG **+0.256**. Neither is redundant.
- **Combined they reach +0.508** (rank sum), which sits at FoldX's
  cluster-bootstrapped per-interface 0.48 and the best leakage-controlled
  learned model's 0.477. Gain from adding IG to burial: **+0.071, CI95
  [-0.182, +0.304]**, not significant at n=28.

**This reframes the question.** It was "can the attribution beat a ruler", and
the answer looked like no. The better question is "does the attribution carry
energetic information that geometry does not", and the partial correlations say
plausibly yes. It also matches the strongest result in the structure-model
literature: ProtBFF's win comes from *injecting* burial and interface features
into a learned model, not from either alone -- its ablation shows interface and
burial are the two largest single contributors.

**Every number in this subsection has a CI spanning zero or nearly so.** This is
a hypothesis worth testing, not a result. The test is n, not method.

### Revised priorities

1. **Pool 3HFM, 1VFB, 1JRH, 2JEL** (~$12, ~2.5 h). Unchanged as item 1, and
   now better motivated: it tests the complementarity finding above as well as
   whether 0.357 replicates.
2. **Add a conservation term to the confound panel.** CPU only, one MSA fetch,
   no A100. Without it we cannot state how we do against RSALOR, which is the
   comparison the field will make first.
3. **Wire the simplex path** (entry 31) on the VM -- it needs boltz installed to
   backpropagate to `res_type`. Until then stage 10 still uses the L2 norm and
   the fix is not live.
4. **The brute-force scan on 1JTG** (~$22, ~4 h) -- unchanged, still the only
   thing that answers the headline claim.
5. **The alpha-profile diagnostic** (`--baseline` now exists, entry 32): plot
   `F(x' + a(x - x'))` for both baselines and report `F(baseline)`. Turns "we
   tried two baselines and one worked" into a mechanistic explanation. Cheap,
   but it does need GPU -- the only saved profile data is 4fqi at L=554.

Still not justified: a different attribution *method*. Items 1 and 4 establish
whether the method is at fault, and the reduction fix in entry 31 has not yet
been measured.

**Superseded as the focus by section 12 (Phase 5), decided 2026-09-29.** The
list above is all position-level work validated at n=28, where nothing survives
its own CI. Phase 5 asks a pair-level question at n=76 on data already on disk,
and rides the same backward pass as item 1. Items 1 and 2 remain worth doing and
are unchanged in cost; they are no longer the front of the queue.

### Co-folding / pairformer interpretability — extended survey, 2026-10-01

Seven papers added. Where a paper is already cited in this document, only the
new detail is recorded here.

**PairSAE**, Migliorini et al.,
[arXiv:2606.27440](https://arxiv.org/abs/2606.27440). Already cited in
section 12. New finding: SAEs do not transfer cleanly to pairformer
architectures (quadratic feature blow-up; concepts distributed jointly across s
and z). Uses N-mode SVD into token-wise interaction roles. Evaluated on Boltz-2
over PLINDER. **Its affinity validation is against Boltz-2's own predicted
affinity, not experimental Kd** — so it does not touch experimental ground
truth, which widens rather than narrows the slot described at `PLAN.md:74-78`.

**Two Stages of Folding**, Lu, Brinkmann, Huber, Mueller, Belinkov, Bau,
Wendler, NeurIPS 2026,
[arXiv:2602.06020](https://arxiv.org/abs/2602.06020). Causal interventions on
ESMFold/OpenFold/Boltz-1. Charge linearly decodable from s at ROC-AUC ~1.0 by
early blocks; distance probes on z reach R^2 ~0.9 in late blocks; seq2pair
dominates blocks 0-10, pair2seq 25-48; in Boltz-1 the seq2pair write-in is
confined to the MSA module (blocks 0-3). Cross-architecture transplant via
whitened Procrustes: Boltz-1->ESMFold R^2 ~0.75. Cost ~1,000-1,500 GPU-hours.
No PAE/ipTM analysis, no MSA ablation.

**AlphaInterp**, Feldman & Skolnick,
[PMC13131572](https://pmc.ncbi.nlm.nih.gov/articles/PMC13131572/). AF3
checkpoint probing. pTM linearly decodable from the pair channel at R^2=0.86 at
the final Pairformer pass vs 0.34 post-init; patching pair-representation PCs
causally moves distogram entropy; transplanting a high-confidence representation
drops entropy by -0.268. MSA ablation: TM 0.9363 -> 0.5388, and
pair-representation effective dimensionality **rises** ~17 -> ~24. Mutational
drift confined to Pairformer layers. **Most relevant published work to this
project; nobody has done the equivalent for PDE.**

**Jedryszek et al.**,
[arXiv:2608.11475](https://arxiv.org/abs/2608.11475). Already cited in
section 12. New detail: probes + SAEs + causal interventions across Boltz-1's
trunk/diffusion boundary. Sequence chemistry strongly attenuated inside the
diffusion module. Cautionary: a beta-strand-direction probe at F1=0.82 whose
steering produced **no measurable structural change**. Decodable != causal.

**Yao et al.**,
[arXiv:2606.22181](https://arxiv.org/abs/2606.22181). Already cited in
sections 7 and 11. New detail: IG on ESM-2 classifiers; residue-level overlap
with annotated epitopes **no better than random** despite strong AUC; reliance
on generic physicochemical/compositional features. Note the match to this
repo's own 2026-09-24 numbers: gradient-vs-hydrophobicity -0.527
(CI [-0.75,-0.18], excludes zero) against gradient-vs-ddG 0.357
(CI [-0.002, 0.656], barely does).

**Wan, Zhang, Xue, Coveney**,
[arXiv:2603.05532](https://arxiv.org/html/2603.05532v1). Boltz-2 binding
probability correlates only r=0.21-0.25 with its *own* predicted affinity;
confidence saturated at mean >=0.944, ~90% above 0.90. **Caveat: these are
protein-ligand affinity/binding-probability figures, not PPI `complex_pde`. Do
not inherit them directly.**

**Masters, Mahmoud et al.**, *Nat Commun* 2025,
[s41467-025-63947-5](https://www.nature.com/articles/s41467-025-63947-5).
Co-folding models are insensitive to significant binding-site modifications and
largely memorise training poses.

**Gap, as a targeted negative finding.** No systematic benchmark of
IG/saliency attribution against masked-marginal or brute-force mutation scans,
on PLMs or on folding models; and no attribution of a *confidence* head on any
folding model.

**UNVERIFIED, do not cite:** AF3 dipTM-vs-SKEMPI-ddG "Pearson R=0.102"; Mac1
eLife reviewed-preprint claims; PairSAE's MLSB 2025 venue line; AbBiBench
per-model Spearmans; ESMC-6B SAE release details; PLM-SAE "+0.138 mean".

---

## 12. Phase 5 — attribute to the pair representation, and predict residue coupling

Proposed 2026-09-29. **This is the new focus.** Everything in sections 8 and 11
attributes one scalar per residue and validates it against single-mutant ddG at
n=28, where nothing survives its own confidence interval. This phase asks a
different question, on a sample nearly three times larger, using data already
on disk and a gradient that falls out of a backward pass we already run.

### The idea

Binding is not a property of residues. It is a property of residue **pairs**,
and Boltz-2's central object is the pair tensor `z`, shape `(1, L, L, 128)`.
Every attribution in this project so far -- and, as far as three independent
literature sweeps found, every attribution published on any protein model --
collapses to one number per position. The gradient with respect to `z` gives an
**L x L interaction map** instead: for each pair of residues, how much that
specific contact contributes to predicted affinity.

Alternatives cannot reach this. Burial and the rest of the confound panel give
one number per position. Brute-force ISM would need **double** mutants, an L^2
scan -- the exact combinatorial wall this project exists to avoid.

### The ground truth -- CORRECTED 2026-09-29 (second pass). Read this before quoting any coupling number.

> **The figures first recorded in this section were wrong, and wrong in the
> direction that flattered the phase.** They came from an undocumented
> row-order dependence -- keeping the *last* duplicate measurement per single
> mutant -- combined with computing ddG at a fixed 298 K. Reference
> `17430899` is a Van't Hoff temperature series measuring 1JTG at **279, 281,
> 283, 285, 288, 293, 298, 301, 303 K** while every other reference sits at
> 296-299 K, and the last-row rule systematically selects those rows. Their
> ddG was then computed as if measured at room temperature. The apparent
> coupling signal was substantially temperature-mismatch error plus
> between-lab offset. The old recipe reproduces every original figure to the
> digit (std 1.84 at ddof=0, range -4.96 to +7.40, 53/82, cross 1.374, same
> 0.888), which is how it was identified; that exactness is a symptom, not a
> validation.

A double-mutant cycle measures whether two residues interact:
`coupling = ddG(AB) - ddG(A) - ddG(B)`. Zero if they act independently. It is
a **difference of differences**, so it only means anything when all three ddGs
share an experimental context -- that is the only way the lab's wild-type Kd,
buffer, method and temperature cancel. Mixing references injects between-lab
offset directly into the coupling term, which is the very quantity we predict.
**Match within `Reference`**, breaking ties on temperature. Not by averaging
across measurements (also wrong, if less so: std 1.30), and never by row order.

Computed from `data/raw/skempi_v2.csv` on 1JTG under within-reference matching:

| | corrected | as first recorded (wrong) |
|---|---|---|
| Complete cycles | **80** | 82 |
| Distinct position pairs | **74** | 76 |
| Distinct positions involved | **31** | 31 |
| Cross-chain cycles (spanning the interface) | **66 / 80** | 66 (of 82; a *cycle* count, not pairs) |
| Cycles with abs(coupling) > 0.5 kcal/mol | **33 / 80** | 53 / 82 |
| Coupling range | **-3.53 to +1.96** kcal/mol, std **0.92** | -4.96 to +7.40, std 1.84 |
| Replicate spread on the 6 repeated pairs | median **0.42** kcal/mol, max 2.15 | median 0.21, max 2.35 |

**The target's dynamic range roughly halved and the pairs clearly above noise
fell from 53 to 33.** Sample size survives; difficulty does not. Note the
`abs(c) > 0.5` counts throughout this document and section 13's table were
computed under the wrong rule and are all inflated.

**1JTG has 83 double mutants, the most of any complex in SKEMPI** (next is
3S9D at 59, then 1BRS at 45). Our target was chosen for other reasons and
happens to be the best-suited complex in the canonical dataset for this.

Two facts matter for inference. Unlike entry 23 -- where 96 mutations collapsed
to 28 positions, 3.4x redundancy -- **80 cycles are 74 distinct pairs, 1.08x**.
This is a near-independent sample. But the 74 pairs draw on only 31 positions,
so the same residue recurs across pairs: **cluster-bootstrap by position, not by
pair**, or significance will be overstated.

**The noise floor is not 0.21 kcal/mol.** That figure was the coupling-level
replicate spread over 6 repeated pairs under the broken rule, and it was quoted
as "the experimental noise floor and therefore the ceiling any predictor can
hit". Corrected, the coupling replicate spread is **median 0.42, max 2.15**.
More importantly, the spread of the *same single mutation* measured by
different labs is **median 1.16, 90th percentile 1.63, max 5.37 kcal/mol** on
1JTG (33 of 58 singles are measured more than once, across 10 references; the
worst case differs by 5.37 kcal/mol between two papers). Within-reference
matching is what keeps that from propagating into the coupling term -- it is
not a refinement, it is the thing that makes the target measurable at all.

**Physics sanity check -- it does NOT pass. This was the artifact, not
evidence.** As first recorded: cross-chain mean abs(coupling) 1.374 against
0.889 same-chain, read as confirmation that "residues coupling across the
interface -- which is what binding is -- are more strongly coupled". Under
within-reference matching the two are **0.639 (n=66) and 0.569 (n=14)**: a gap
of **0.070 kcal/mol at Mann-Whitney p=0.75, Welch p=0.65**, against a coupling
noise floor of 0.42. There is **no evidence** that cross-interface pairs couple
more strongly than within-chain pairs in this dataset.

This one matters beyond its own line. It was the only independent check that
the ground truth reflected interface biology rather than assay noise, and it
was produced by the same bug as the inflated coupling values. Phase 5 now rests
on Gate C alone, with no prior reassurance that the target is physical. That is
a materially weaker position than this section originally described, and it
should be stated to referees rather than discovered by them.

### Method

**Layer IG on `z`, not input IG.** `BoltzScorer._checkpointed_forward(self, s,
z, mask, pair_mask)` (`src/igv/boltz_score.py:1335`) already takes `z` as an
argument and returns `(s, z)`, so `z` can be interpolated directly between the
`mean_aa` baseline's pair tensor and the wild-type's, and the model re-entered
from there. This makes completeness well-defined at that layer and avoids
retaining grad on an intermediate inside reentrant checkpointing.

    A[i,j] = (z_x[i,j] - z_b[i,j]) . integral_0^1 dF/dz[i,j] dalpha

contracted over the 128 channels, then symmetrised as `A[i,j] + A[j,i]` since
the pair representation is not guaranteed symmetric.

**Contract with a dot product, not an L2 norm.** This is entry 31 arriving in
advance: an L2 norm over 128 channels is non-negative, discards sign, and by
concentration of measure flattens the map whatever the model does. The dot
product preserves sign and satisfies completeness. Do not repeat that mistake
one dimension up.

### Controls — the second one decides whether this is interesting

1. **Inter-residue distance.** Nearby pairs couple more, so distance will
   predict coupling on its own. Report partial correlation controlling for
   Cbeta-Cbeta distance. This is the burial lesson (section 11) applied before
   rather than after.
2. **The two single-residue attributions.** Does the off-diagonal carry
   information beyond the diagonal? Partial correlation of `A[i,j]` against
   coupling, controlling for `A[i]` and `A[j]`. **If the pair term adds nothing
   over the two single terms, the pair tensor is not telling us anything new
   and this phase ends here.** Exactly the IG-vs-burial test from section 11,
   one dimension up.
3. **Permutation null** over pair labels, matching the shuffled-null convention
   already used in stage 10.
4. **Cluster bootstrap by position** (31 clusters), not by pair.

### Cost

No GPU beyond a run already required. The pair gradient is ~93 MB fp32 at
1JTG's 427 tokens (427^2 x 128 x 4). Capture it during the Phase 2 IG run.

### What would kill it, stated in advance

- Control 2 fails: the pair map is just the outer product of the singles.
- The map tracks contact geometry rather than energetics -- plausible, since
  the pair representation is what the model uses to predict structure. Control
  1 is the test, and a strong distance correlation with a null partial is the
  failure signature.
- ~~`z` interpolation turns out not to be re-enterable cleanly~~ --
  **ASSESSED 2026-09-29, GREEN.** See below.

### Feasibility of the z seam -- assessed 2026-09-29

**Verdict: green.** `z` enters `_checkpointed_forward` as a parameter and
leaves as a return value with no side effects and no other uses
(`src/igv/boltz_score.py:1335-1349`), so it can be supplied externally without
touching any existing path.

**Seam: `src/igv/boltz_score.py:1780`**, immediately before the existing
`model.confidence_module(...)` call. Nothing upstream -- `z_init`, the
recycling loop, the MSA blocks -- needs to change; `s`, `x_pred` and `feats`
are fixed across interpolation steps.

**The checkpointing fear does not apply here.** The inner per-layer checkpoints
are already `use_reentrant=False` (line 1347). The reentrant one that caused
entries 8-11 is the *outer* wrapper at line 1832, and calling the confidence
head directly bypasses it: grad is enabled throughout, `_ckpt_mha`
(lines 1428-1429) takes its checkpointed branch, and gradients accumulate on
`z.grad` normally. Entry 11's triangle-attention checkpointing already uses
this exact pattern.

**Recycling resolves itself.** Only the final `z_` after all recycling
iterations reaches the confidence head (line 1777 onward); intermediate values
never do. So "the pair tensor" is unambiguous -- attribute to the final one.
Per-iteration attribution would need new code and is not needed here.

**Memory: ~5.5-8 GiB against the 78 GiB that full-trunk backward needs**
(entry 12, all nine measured configurations). A 10-14x reduction, because the
confidence head is 8 checkpointed layers rather than 64 pairformer blocks plus
4 MSA blocks. **This is the same frozen-trunk escape entry 12 already chose**,
so Phase 5 does not need the full-trunk problem solved -- it rides the decision
we already made.

Carry the matching caveat: attributing through the confidence head with `s` and
the trunk fixed answers "how does the confidence head use `z`", not "how does
the whole model use `z`". That is the limitation frozen-trunk already accepts,
and it should be stated rather than glossed.

### The de-risking test -- and what it must NOT be

Run at L ~ 100, `m_steps=5`, a few minutes on the A100, before anything is
costed.

**The test must call the real confidence head.** The version first proposed
scored a synthetic random `z` through a mock `(z**2).sum()*0.01` and checked
completeness. That is a tautology -- IG on a quadratic satisfies completeness
by construction, and it exercises none of the risk. The whole question is
whether gradient flows through the **real checkpointed confidence pairformer**
to an externally supplied `z`. A mock cannot answer it and would give false
assurance, which is exactly the failure mode of the `dead_target` check
(section 4) that passes vacuously at 0.000e+00 every run.

Pass criteria: `z.grad` is not None; not uniformly zero; shape
`(1, L, L, 128)`; all finite; and completeness relative error under ~20% at
m=5 against a real `f(z_x) - f(z_baseline)` computed from the same head.

Three runtime unknowns, none blocking: whether model parameters are frozen
(`MEMORY.md` section 6 says they are not -- check
`sum(p.numel() for p in model.parameters() if p.requires_grad)`), the exact `s`
dimension, and whether `z.grad` stays fp32 under autocast (`MEMORY.md` section
2.2 says `z` is fp32 in the forward even under bf16 -- verify).

### Status of the novelty claim

Three literature sweeps found no gradient attribution on a co-folding model's
affinity head. Pairwise readouts from protein models exist but all target
**contacts and structure** on sequence-only models -- Rao et al. (ICLR 2021,
attention-head logistic regression), Vig et al. (ICLR 2021), Zhang et al.
(*PNAS* 2024, the categorical Jacobian, ~19L forward passes), Thorstenson
(arXiv:2606.21876). None targets binding energetics, and none is a gradient on
an internal pair tensor.

**Confirmed 2026-09-29 by a dedicated search: no one has produced a pairwise
attribution map for binding affinity or ddG.** The strongest evidence is the
incumbent naming the gap itself. BALM-PPI (Singh et al., bioRxiv
10.64898/2026.03.30.715237) is the current state of the art in *explainable*
PPI affinity -- >12,000 complexes, residue-level IG maps that recover known
hotspots -- and its own text concedes that its Integrated Gradients
attributions "reflect mean-pooled embeddings and do not yet provide pairwise
residue-residue contact predictions."

Every ingredient exists separately; nobody has assembled them:

| Nearest neighbour | L x L | Attribution | Target | Gap |
|---|---|---|---|---|
| Categorical Jacobian (Zhang, *PNAS* 2024) | yes | finite difference | MLM logits | wrong target |
| Top-K attention heads (Thorstenson) | yes | no, a readout | contacts | wrong target |
| Nambiar et al. 2025 pLM epistasis matrix | **yes** | finite difference | DMS fitness epistasis | closest on form, wrong target |
| DFIM (Greenside, *Bioinformatics* 2018) | **yes** | **yes** | TF binding | **right method, wrong molecule -- DNA** |
| Integrated Hessians (Janizek, Sturmfels & Lee, *JMLR* 22(104):1-54, 2021) | **yes** | **yes**, axiomatic | anything differentiable | **never applied to any protein model** |
| BALM-PPI (2026) | no | yes, IG | **binding affinity** | first-order only, and says so |

Note DFIM: genomics solved pairwise attribution in 2018 and it never crossed
into proteins. That is the same pattern as the Majdandzic simplex correction in
entry 31 -- **the genomics community's answers keep not being carried over**,
and carrying them is repeatedly where the contribution turns out to be.

### Calibration -- what counts as a good coupling correlation

**Do not benchmark against SKEMPI multi-point numbers.** IGMI reports Pearson
0.77 on M1707, DDMut-PPI 0.83 on SM1124, MutPPI+ 0.880. Those are correlations
against **total** multi-point ddG, which is dominated by the additive sum of
the singles. A model can score 0.8 there with **zero** signal on the
non-additive part. The search found no paper reporting a SKEMPI-wide
correlation on the coupling term alone -- which is both why this is open and
why the comparison must not be made.

The honest benchmarks for a coupling-term prediction:

| Source | Target | Number |
|---|---|---|
| Nambiar et al. 2025 (bioRxiv 10.1101/2025.09.14.676130), ESM-2 | TEM-1 epistasis, ~12k doubles | **r = 0.37** |
| same, YAP1-WW, ~8,670 pairs | fitness epistasis | **r = 0.34** |
| same, Pab1-RRM2, ~36,500 pairs | fitness epistasis | **r = 0.26** |
| FEP (Schrodinger), 45 staph nuclease doubles | **non-additivity** | **r = 0.79** -- UNVERIFIED, snippet-sourced, verify before citing |

So **r in the 0.26-0.37 band is what a pLM currently achieves on epistasis**,
and that is against fitness, not binding. Corroborating the difficulty:
Kolchina et al. (bioRxiv 10.64898/2026.02.17.706292) benchmark 95 zero-shot VEP
models and find they "perform well for single mutations and non-epistatic
combinations ... [but] fail to predict the effect of strongly epistatic
combinations."

**Added 2026-09-29 with the ground-truth correction: our own attainable ceiling
is lower than those numbers, for a reason that has nothing to do with method.**
The pLM epistasis figures come from 8,600-36,500 doubles per dataset with
correspondingly averaged-down noise. We have 74 pairs against a coupling
replicate spread of 0.42 kcal/mol on a target whose corrected std is 0.92. That
is a signal-to-noise ratio that caps achievable r well below what a clean
36,500-pair fitness dataset supports, before any question of whether the
gradient carries information. **Compute the attenuation bound explicitly from
the replicate spread and report r alongside it**, rather than comparing a
74-pair correlation to a 36,500-pair one as though they were commensurable. A
result of r = 0.3 here and r = 0.3 on TEM-1 are not the same claim.

### The cost argument is not the contribution

Thorstenson gets the *contact* signal in one forward pass, so "cheaper than the
categorical Jacobian's ~19L" is not by itself a result. **The contribution is
the target quantity (ddG coupling) and the attribution semantics, not the
FLOPs.** State it that way.

### Two things to close before claiming priority

1. **IGMI -- READ IN FULL 2026-09-29. It does not pre-empt us.** Wu, Xu, Xin,
   Zhang, Liu, Zhu, Wei, Zhao, Yu & Feng, *Bioinformatics* 42(4):btag150, 2026,
   PMC13070472. Its only pairwise object is Figure 8c, whose in-image title is
   "Attention Mechanism Head 1 Weight Matrix Heatmap": **one head, one layer,
   one complex, 128x128 rather than L x L** (the model only ever sees a fixed
   128-residue mutation-centred subgraph), non-negative, unitless, computed in
   the forward pass and **never connected to the predicted ddG by any
   computation**. The predicted ddG for that complex is not even reported. The
   word "attribution" appears nowhere in the paper; neither do gradient,
   Shapley, saliency, or completeness. The deciding sentence is the first line
   of their section 3.4: *"Since attention weights can reflect residue-level
   contributions in PPIs, we analyzed IGMI's attention distributions"* --
   attention weights, justified by citation rather than axiom, and described as
   **residue-level**. Its pairwise object is validated against nothing: one
   qualitative case study, no coupling energies, no contact AUC. It also never
   reports the coupling term, only total multi-point ddG on M1707.

   **But phrase our novelty carefully.** IGMI contains an exact
   completeness-satisfying *per-residue* decomposition it never uses
   interpretively -- Eq. 15/17 give `ddG = sum_i ddG_i` by construction. So
   claim **pairwise** attribution, not "first attribution of any kind for
   ddG"; a referee who knows Eq. 17, or the per-residue decomposition
   literature generally, would push back on the broader phrasing.

   Three framings that survive, in increasing strength: (i) **attribution vs
   attention** -- theirs is an architectural internal with no output
   dependence, ours is a gradient of the score with a completeness story;
   (ii) **sign** -- attention is non-negative and cannot say whether a pair
   helps or hurts binding, which for ddG is the entire point; (iii)
   **validation** -- they validate their pairwise object against nothing, so
   validating against double-mutant cycles makes us first to show one means
   something.
2. **MLSB and OpenReview -- SWEPT 2026-09-29. No pre-emption.** All 225 MLSB
   titles across 2023/2024/2025; OpenReview via `api2.openreview.net` with
   `source=forum` across ICLR 2024-26, ICML 2025-26, NeurIPS 2023-25 mains and
   the MLSB / GEM / GenBio / LMRL / AI4Science / MLGenX / Mech-Interp
   workshops, including under-review, withdrawn and rejected; arXiv and Europe
   PMC with a preprint filter. `"pairwise attribution" AND protein` returns
   zero on arXiv; `abs:"integrated gradients" AND abs:"binding affinity"`
   returns zero.

   **Three papers must be cited and explicitly distinguished:**

   - **PairSAE** (Migliorini et al., MLSB 2025, arXiv:2606.27440) is the
     closest thing that exists -- it overlaps on pair representation,
     co-folding model, and Boltz-2 affinity simultaneously. It does not
     pre-empt us for a specific reason worth stating in the paper: its stated
     contribution is summarising the pair tensor "into token-wise interaction
     roles" precisely to avoid "a quadratic blow-up of features". **The L x L
     object is what they engineer around; it is what we produce.** It is also
     an SAE rather than a gradient, affinity enters as a probe *target* rather
     than the quantity attributed, and it is protein-ligand on PLINDER.
   - **TopoScorer** (ICLR 2026 submission, OpenReview QNcrdCKNa5) states our
     motivating gap almost verbatim -- deep affinity models "generally lack
     interpretable attributions" -- but answers it by building a new
     interpretable architecture rather than attributing an existing model.
     **This is the paper most likely to be raised in review.** Pre-empt it by
     framing ours as post-hoc attribution of a frozen co-folding score.
   - **AF2BIND** (MLSB 2023) probes AF2's pair representation for binding
     sites. So do not claim "first interpretability on pair representations",
     and do not claim "first interpretability of Boltz affinity" either --
     PairSAE has a colourable claim to the latter.

### The claim, phrased to survive review

**CORRECTED 2026-09-30 -- the original wording was infeasible.** It read
"...of a co-folding model's *binding-affinity* score...". Boltz-2 cannot
produce a binding-affinity score for a protein-protein complex: the parser
raises `ValueError("Affinity is currently only supported for ligands.")` at
`schema.py:1066-1070`, and the docs restrict the affinity binder to a ligand
chain of at most 128 heavy atoms. See `ERRORS_LOG.md` entry 36. Do not restore
the old phrasing; there is no such score to attribute.

**"First gradient-based, pair-resolved attribution of a co-folding model's
structure-confidence score onto the L x L pair representation."** Every element is
load-bearing: PairSAE has pair-representation + co-folding + affinity but is
SAE-based and deliberately token-level; TopoScorer has affinity + attribution
but is a bespoke architecture; the Boltz probing cluster has pair
representation + causal intervention but targets structure; IGMI has an L x L
object and a ddG target but the object is attention, not attribution. Claim the
intersection, never the parts.

Supporting argument worth citing: Jedryszek et al. (arXiv:2608.11475) showed a
beta-strand direction in Boltz-1 that was highly linearly decodable (F1 0.82)
yet produced **no** structural change when steered. **Linear decodability does
not imply causal influence** -- which is a direct argument for gradient-based
attribution over the probing and SAE work that currently dominates this space.

### Residual risk, stated honestly

MLSB 2020-2022 was not scanned (predates co-folding affinity heads).
Non-archival workshop posters with no OpenReview record are invisible to every
index. ICLR/ICML 2026 submissions still under embargo would not surface. And
Europe PMC search is title/abstract-weighted, so this done as an unadvertised
side analysis inside a broader affinity paper would be missed. Confidence is
high for the specific claim above, moderate-high for any looser phrasing.

---

## 13. Execution plan for Phase 5

Written 2026-09-29. Five stages, three decision gates, and an explicit early
exit at each. Total exposure if it runs to completion is roughly **$60-75 of
A100**; total exposure if it dies at the first gate is **about $2**. The
ordering is chosen so the cheapest disconfirming evidence arrives first.

### Which complexes can replicate this -- not the ones we planned

The Phase 1 pooling set (3HFM, 1VFB, 1JRH, 2JEL) was chosen for **single**
mutants. Phase 5 needs **doubles with both singles measured**, and that is a
different set. Computed from `data/raw/skempi_v2.csv`, complexes with at least
8 complete cycles:

**This table was computed under the broken row-order rule and is superseded.**
The `distinct pairs` column below is corrected to within-reference matching;
the `abs(c) > 0.5` and `std` columns are NOT corrected per complex -- they were
inflated by the same mechanism, so treat them as unreliable and recompute
before using them to prioritise. Only 1JTG has been recomputed in full.

| Complex | pairs (within-ref) | pairs (as first recorded) | abs(c) > 0.5 -- INFLATED | std -- INFLATED |
|---|---|---|---|---|
| **1JTG_A_B** | **74** | 76 | 33 / 80 (corrected) | **0.92** (corrected) |
| 3S9D_A_B | 53 | 58 | 6 | 0.49 |
| **1BRS_A_D** | **33** | 37 | 25 | 1.35 |
| 4G0N_A_B | 32 | 32 | 18 | 0.73 |
| 1LFD_A_B | 26 | 26 | 22 | 0.94 |
| 1AO7_ABC_DE | 21 | 24 | 16 | 0.81 |
| 1DAN_HL_UT | **4** | 14 | 9 | 0.93 |
| 1VFB_AB_C | 14 | 14 | 5 | 1.05 |
| 1DQJ_AB_C | 13 | 13 | 9 | 1.68 |
| 1DVF_AB_CD | 13 | 13 | 11 | 1.00 |
| 3HFM_HL_Y | 11 | 11 | 14 | **2.12** |
| **Total** | **294** | **375** | | |

**`1DAN_HL_UT` collapses from 14 pairs to 4** -- its cycles were almost
entirely assembled across labs. Drop it from the replication set; at n=4 it
cannot contribute either way.

**Read the `abs(c) > 0.5` column, not the cycle count.** 3S9D has the second
most cycles and almost no coupling -- 6 of 59 above 0.5 kcal/mol, std 0.49. It
is a near-additive system and would dilute the signal while looking like a
large sample. Do not use it early. (This conclusion survives the correction:
3S9D's coupling was the *least* inflated of the set, so if anything it is more
clearly near-additive than the table suggested.)

**1BRS is barnase-barstar**, the system on which double-mutant cycles were
established as a method (Schreiber & Fersht). 33 pairs on the canonical
coupling system is the single most persuasive replication available, and it
should be the first complex after 1JTG. But its "25 strongly-coupled pairs" is
an uncorrected figure -- recompute it before leaning on that phrasing, because
1JTG's equivalent count fell by 38% under correction.

**And check 1BRS's cluster count before trusting a CI from it.** Measured
2026-09-29 once it was registered: 35 cycles, **33 distinct pairs on only 12
distinct positions** -- 2.75x redundancy, against 1JTG's 74 pairs on 31
positions (2.4x). Section 12 correctly requires cluster-bootstrapping by
position rather than pair, but **12 clusters is a small bootstrap**, and a CI
computed from it will be wide and unstable regardless of the effect. Barnase-
barstar is the right system for persuasiveness and the wrong one for a tight
interval; plan to report it as a directional replication, not as independent
confirmation with its own significance claim. All 33 of its pairs are
cross-chain, which is at least the geometry we want.

Ceiling if everything runs: **294 distinct pairs** (290 excluding 1DAN, which
falls below the 8-cycle threshold entirely, and before any decision on 3S9D),
against n=28 today. Still an order of magnitude more than the position-level
work, which remains the point.

**Two things found when the implementation was verified 2026-09-29. Both
change what Stage 4 has to do.**

**1. Quote the right denominator -- 294 and 310 are both correct.** 294 is the
within-reference total over the 11 complexes this table lists. Running
`extract_all_complexes(min_cycles=8)` over all of SKEMPI gives **310**, because
it surfaces four complexes never listed here: `5XCO_A_B` (9 pairs), `5M2O_A_B`
(7), `1KNE_A_P` (3), `3MZG_A_B` (1). 5XCO is worth adding to the queue. But
note **`1KNE_A_P` has 33 cycles collapsing to 3 distinct pairs -- 11x
redundancy**, which is entry 23's problem in a new place: it would look like a
large sample and contribute almost nothing independent. Screen candidates on
distinct pairs, never on cycle count.

**2. Seven of the ten Stage 4 complexes are not registered, including 1BRS.**
`SKEMPI_COMPLEXES` (`src/igv/skempi.py:62`) holds only 1JTG, 3HFM, 1VFB, 1JRH
and 2JEL -- the Phase 1 *single*-mutant set. Missing: **1BRS**, 4G0N, 1LFD,
1AO7, 1DQJ, 1DVF, 3S9D. `scripts/13_coupling.py` calls `get_complex` before
doing any work and exits on `KeyError`, so **Stage 4a as written fails on its
first line** -- on barnase-barstar, the complex this plan calls the single most
persuasive replication available. Cheap to fix, and it had to be found before
the VM was billing rather than after. The general point: the replication set
was chosen from the CSV, and nobody checked it against the structural registry
the pipeline actually resolves through.

---

### Stage 0 -- local, free, no GPU

Can all be done now. None of it needs the VM.
**Executed 2026-09-29. Status against each item below.**

0a. ~~**Resolve the uncommitted 2026-09-29 09:49 changeset**~~ -- **DONE**
    before this session; tree clean at `33c545d`, "Clean up the repo for
    outside readers".
0b. ~~**Implement `confidence_head_forward`**~~ -- **DONE**, commit `6f41295`.
    `confidence_head_forward` in `src/igv/boltz_score.py`, `pair_layer_ig` +
    `PairAttribResult` in `src/igv/attrib.py`, 17 boltz-free plumbing tests in
    `tests/test_pair_attrib.py`. Two deviations from this section, both
    deliberate: symmetrisation is **`(A + A^T)/2`**, since the literal
    `A[i,j] + A[j,i]` doubles the total and breaks completeness; and the
    distogram is recomputed from the interpolated `z` rather than frozen,
    because it is a direct function of `z`. Section 12's line numbers were
    verified accurate at `33c545d`.
0c. **Implement `src/igv/coupling.py`** -- **DONE, then corrected.** Note the
    spec above said "compute ddG from `Affinity_*_parsed` at RT = 0.001987 *
    298.15". **That instruction was wrong and is the origin of the bug in the
    ground truth.** 1JTG spans 279-303 K (reference `17430899` is a Van't Hoff
    series); a fixed temperature is not available. Use per-row temperature via
    the existing `add_ddg` (`src/igv/skempi.py:219`), and match cycles
    **within `Reference`**. See the corrected ground-truth block in section 12.
0d. **Add a conservation term to the confound panel** (section 11 item 2).
    **NOT DONE -- deferred, and mis-scoped here.** It is not cheap and not
    local: no MSA is on disk, the only implemented fetch runs inside boltz
    which is not installed locally, and `burial` is not RSA so RSALOR needs
    two new terms rather than one. See the correction in section 11. It is
    confirmed off Phase 5's critical path -- nothing in the pair-attribution
    plan or the four controls consumes a conservation feature -- so deferring
    costs Phase 5 nothing.

### Stage 1 -- sync and de-risk -- **DONE 2026-09-29. GATE A: PASS. ~$1.70, ~20 min.**

1a. ~~Start the VM, `git pull`.~~ Done, synced to `62889c6`. **`git pull` does
    not work on the VM** -- see the sync note in `HANDOFF.md`; it has no GitHub
    credentials. A `git bundle` was used instead.
1b. ~~Run the real, non-mocked z-gradient completeness test.~~ Done at
    **L = 352** (full 1VFB complex, chains A/B/C), not L ~ 100 -- a single
    chain scored with `complex_pde` risks a degenerate score and therefore a
    spurious FAIL, and the cost difference was pennies. Real confidence head,
    `mean_aa` z baseline, MSA server. Artifact:
    `results/gate_a_1VFB_complex_pde.json`.

> **GATE A: PASS**, all five criteria.
>
> | criterion | result |
> |---|---|
> | `z.grad` not None | yes |
> | not uniformly zero | max abs **4.652914e-04**, zero fraction **0.0000** |
> | shape `(1, L, L, 128)` | `(1, 352, 352, 128)` |
> | all finite | NaN 0, Inf 0 |
> | completeness | **abs err 0.003627**, rel err 0.0064 |
>
> Completeness is the result worth keeping: `ig_sum = -0.571934` against a true
> `f(z_x) - f(z_b) = -0.568307`, i.e. **0.64% relative error at m=5**. The seam
> is not marginally usable, it is clean, and the L x L map is a genuine
> decomposition of the score. **Wall time 5.0 s** for all five IG steps.

**Two things in this section's own pass criteria were wrong, and would have
mattered:**

1. **"completeness relative error under ~20%" is the wrong gate.** That is
   entry 32's bug restated -- relative error divides by the span, so a *good*
   baseline sits close to the input, the span collapses, and the ratio
   inflates. Combined with the `mean_aa` baseline this section correctly
   specifies, gating on relative error can fail the better setup. Gate A gates
   on **absolute** error against `07_sanity.py`'s `COMPLETENESS_ABS_THRESHOLD`
   and reports relative as information. Here both were comfortable, so the
   distinction did not bite -- but it was luck, not design.
2. **There was no INCONCLUSIVE outcome.** A single FAIL code conflates
   "gradient does not reach `z`" with "the setup was degenerate", and those
   have opposite consequences: the first ends Phase 5, the second means fix the
   harness and rerun. `14_gate_a.py` now exits 2 for a collapsed span, a
   non-finite or zero score, a constant path, or degenerate `z_x`, checked
   *before* the five criteria.

**The three runtime unknowns are now measured, not assumed** (all three were
listed in `MEMORY.md` section 6 as open; all three are now settled there):

| unknown | measured |
|---|---|
| (a) parameters frozen? | **25,026,048 trainable / 481,698,944 frozen** -- 95% frozen already. MEMORY.md's "never frozen" was wrong. |
| (b) `s` dimension | `s.shape = (1, 352, 384)` |
| (c) `z.grad` fp32 under autocast? | **yes**, `torch.float32` |

**Memory: peak 8.68 GiB** (8.55 GiB for the single backward). Section 12
predicted 5.5-8 GiB, so slightly over the top of the range but ~9x below the
78 GiB a full-trunk backward needs. The frozen-trunk escape holds.

**Settled bonus:** `score.backward()` parks only **18.1 MiB across 315
tensors** in parameter `.grad`. So switching `pair_layer_ig` to
`torch.autograd.grad` would save ~18 MiB against an 8.68 GiB peak and is not
worth splitting the convention with `integrated_gradient`. Recorded because
the pre-run guess was that this was a significant VRAM cost; it is not.

### Two blockers found by Gate A that must be fixed before Stage 2

Both are free, local, and would corrupt the Stage 2 capture rather than crash
it -- which is the dangerous kind.

1. **`x_pred` is zeros.** Gate A logged `feats['coords'].abs().max() = 0`,
   settling the "GEOMETRY IS FIXED" question in `MEMORY.md` section 6: the
   sequence-only YAML gives every atom (0,0,0), so the confidence head sees no
   structure. Harmless for a gradient-flow test. **Not harmless for Stage 2**,
   where the pair map would describe a structureless complex. Stage 2 must run
   a real structure prediction (~45 s) and feed genuine `x_pred`.
2. **The featurisation cache substitutes silently.** `ERRORS_LOG.md` entry 34:
   the homopolymer path returned 1JTG's 427-token features for a 352-token
   1VFB request, because `data/raw/boltz_homopolymer/<AA>` is not keyed on
   dataset. Entry 24 fixed this for the main path only. Caught solely by
   `_build_token_map`'s run-length guard; without it the `mean_aa` baseline
   would have come from the wrong protein and **Gate A would plausibly have
   passed anyway**. Key every featurisation directory on dataset and audit for
   other instances.

### Stage 2 -- capture on 1JTG (~$10, ~2 h)

2a. One IG run on 1JTG capturing **both** the pair gradient (L, L) and the
    (L, 20) simplex map from entry 31. Same backward passes; take both.
2b. Also capture `F(baseline)` and `F(input)` for the alpha-profile
    (section 11 item 5) -- `09_path_profile.py` already supports `--baseline`.

> **GATE B, sanity only.** Is the pair map non-degenerate -- not near-constant,
> not concentrated on one row, not tracking the diagonal alone? Report the same
> concentration statistics entry 31 used, because a flat map here means the
> contraction is wrong, not that biology is absent.

### Stage 3 -- analysis, local, free

3a. Join to the 82 cycles / 76 pairs. Run all four controls from section 12:
    distance partial, **singles partial**, permutation null, cluster bootstrap
    by position (31 clusters, not 76 pairs).

### STAGE 3 RESULT -- GATE C FAILED, 2026-09-30. Phase 5 is closed as a negative.

Run: `scripts/13_coupling.py --complex 1JTG --pair-map
results/1JTG_complex_pde_pair_ig.npz`. Full detail in `ERRORS_LOG.md` entry 35.

| control | result |
|---|---|
| **Gate C -- partial(A[i,j], coupling \| A[i], A[j])** | **+0.0019** |
| raw Spearman(A[i,j], coupling) | -0.0335 |
| permutation null, n=1000 | null mean +0.0029, **p = 0.78** |
| cluster bootstrap, 31 positions | 95% CI **[-0.156, +0.062]** |
| Spearman(\|A[i,j]\|, \|coupling\|) | +0.072, p = 0.53 |
| distance ↔ coupling | **+0.4005** |
| partial(A[i,j], coupling \| distance) | +0.1764 |
| **Spearman(\|A[i,j]\|, centroid distance)** | **-0.527, p ~ 0** |

**The failure signature is the one this section predicted.** It listed "the map
tracks contact geometry rather than energetics" as a kill condition and said
"a strong distance correlation with a null partial is the failure signature".
Measured: -0.527 against distance, +0.0019 partial. Exactly that.

**The near-miss worth remembering.** The 80 SKEMPI cycles sit at the 90.7th
percentile of the map by \|A\|, 10.4x the map's median, with 50% inside the top
decile. In isolation that reads as the attribution finding the known hot pairs.
It is geometry: experimentalists mutate interface residues, which are close,
and the map is high on close pairs. That claim would have looked good and been
wrong, and only Control 1 separates the readings.

**Why this negative is trustworthy:** completeness exact (abs err 0.000000 at
m=64), ranking converged (rho = +1.0000 across m=8/16/32/64), map
non-degenerate on every Gate B statistic, and the target demonstrably
predictable -- distance alone scores +0.40 on it. A predictor can score here;
ours cannot.

**Second, independent diagnosis: the target was never an affinity score.** All
six registered scores are structure-confidence metrics, and Boltz-2's affinity
head `boltz2_aff.ckpt` sits in the checkpoint directory deliberately unloaded
(`boltz_score.py:475-493`, `affinity=False` at :891). The claim below says
"binding-affinity score"; `complex_pde` is predicted distance error. Correct
the claim, or change the target -- but do not restate it as written.

**Cost: ~$4 across four A100 sessions**, against the ~$60-75 budgeted to reach
this point. Stage 4's 1BRS replication and the remaining four complexes are
correctly unspent. The ordering this plan chose -- cheapest disconfirming
evidence first, analysis written before any GPU time -- is what bought that.

**Do not read this as "gradient attribution on pair representations does not
work."** It is narrower: attributing a *structure-confidence* score yields a
map of *geometry*, which does not predict energetic coupling. Whether the
affinity head carries PPI energetic signal is an open and separate question,
and section 11 already flagged the prior -- it is trained predominantly on
protein-ligand data, and King et al. (arXiv:2512.06592) find Boltz-2 fine-tuned
for PPI affinity "underperforms relative to sequence-based alternatives". Settle
that cheaply before building anything on it.

---

> **GATE C -- the kill condition, stated before the data exists.** If
> `A[i,j]` adds nothing over `A[i]` and `A[j]`, the pair tensor carries nothing
> the diagonal did not, and **Phase 5 ends here.** Write the result up as a
> negative and return to section 11's queue. Benchmark for a positive:
> r in 0.26-0.37 is what pLM epistasis achieves (section 12); do **not**
> compare against the 0.77-0.88 multi-point numbers, which are dominated by
> additivity.

### Stage 4 -- replication (~$10 per complex)

Only if Gate C passes. In order:

4a. **1BRS** (~$10) -- barnase-barstar, 37 pairs, the canonical system. If the
    effect does not appear here it is not real.
4b. **4G0N, 1LFD, 1AO7, 3HFM** (~$40) -- takes the pooled total toward ~250
    pairs. 3HFM is small at 11 pairs but has the strongest coupling in the set
    (std 2.12), so it is a high-information cheap add.
4c. 3S9D only as a **negative control**: a near-additive system where the
    predictor should score near zero. If it scores well there, the signal is
    additive contamination.

### Stage 5 -- write-up

Claim exactly the sentence in section 12, cite and distinguish PairSAE,
TopoScorer, AF2BIND and IGMI, report model-faithfulness and
biological-faithfulness separately per Yao et al., and state the frozen-trunk
limitation rather than glossing it.

### What this plan deliberately defers

- **The brute-force scan** (~$22). Still the only thing that answers the
  original headline claim, and still unpublishable without it -- but it is a
  claim about the *position-level* method, and Phase 5 is now the lead.
- **The 6M0J saturation arm.** Built and tested, never run. The (L, 20) map
  from entry 31 is its natural input, so revisit after Gate C.
- **Pooling 3HFM/1VFB/1JRH/2JEL for single mutants.** Note 2JEL has **zero**
  doubles and 1JRH six, so that set does almost nothing for Phase 5; it stays
  in the queue for the position-level question only.

---

## 14. The simplex reduction, measured at last -- and it makes no difference

Run 2026-09-30 on the A100, first execution of the `(L, 20)` path. Entry 31
opened 2026-09-24 and had never been tested end to end; three defects sat
between it and a trustworthy number (`ERRORS_LOG.md` entries 39, 40, plus the
consumer/producer gap and the hardcoded zero geometry). All four are closed.

### What was actually fixed before any number was believed

1. **Real geometry.** `03_attribute.py` took `x_pred = feats["coords"]`
   unconditionally, and the sequence-only YAML puts every atom at the origin --
   confirmed live: `feats['coords'].abs().max() = 0`. Now `--x-pred predicted`
   by default. Measured effect on 1JTG: `coords.abs().max()` 0 -> 31.5654.
2. **The producer was never wired.** `10_skempi_hotspots.py` read a
   `grad_res_type` key that nothing wrote, so the "fixed" path silently took
   the L2 fallback. Only the consumer half of entry 31 had ever existed.
3. **The homopolymer cache was keyed on the chains it did not featurise**
   (entry 40). Caught by the guard, which aborted rather than building the
   baseline from 262 alanines standing in for the partner chain.
4. **The chance baseline was a count, not a fraction** (entry 37), so every
   top-k comparison in the project had read "never beats chance" by
   construction.

### VERDICT after all four complexes -- 2026-10-01. The fix does not help.

**Written after 3HFM landed. The 3S9D-only reading below was recorded while two
of four complexes were still running and is kept for provenance, not as a
result. Read this block first.**

| complex | positions | simplex | L2 | diff | cleared own null? |
|---|---|---|---|---|---|
| 3S9D A | 50 | **+0.4683** | +0.0356 | +0.4327 | simplex only |
| 1JTG B | 28 | +0.1714 | +0.2946 | -0.1232 | neither |
| 2JEL P | 32 | **-0.3396** | +0.0031 | -0.3427 | neither |
| 3HFM Y | 15 | +0.1573 | +0.4218 | -0.2645 | neither |

- **simplex** mean +0.1143, sd 0.3349, t=+0.68, **p=0.54**
- **L2** mean +0.1888, sd 0.2028, t=+1.86, **p=0.16**
- **paired difference** mean -0.0744, t=-0.43, **p=0.70**
- simplex beat L2 in **1 of 4** complexes
- **1 of 8** measurements cleared its own shuffled null -- what noise gives at
  eight looks

**Conclusion: the L2 reduction was never what held the method back.** Entry 31's
diagnosis of the reduction was theoretically correct and is still worth keeping
(sign-discarding, concentration-of-measure flat), but fixing it changes nothing
measurable. On 2JEL the corrected map is actively anti-predictive -- AUROC
0.365, precision@5 = 0.00 against a 0.25 base rate, Spearman CI
[-0.6273, -0.0019] excluding zero on the wrong side.

**Do not quote the 3S9D number alone.** It is one favourable draw in four, and
the sign flips across complexes. The between-complex variance (sd ~0.33)
swamps the between-method difference (-0.07).

**Two caveats that weaken even these numbers:**
1. **Completeness failed on 3 of 4 runs** -- 8.51% (3S9D), 11.14% (3HFM), 5.99%
   (1JTG) against the 5% criterion; only 2JEL passed at 2.37%. Real geometry
   roughly tripled the `f(x) - f(baseline)` span, so m=32 is now too few
   quadrature steps. The one run that passed is the strongest negative.
2. **3HFM was underpowered by construction** -- 15 positions against 8
   confounds gives a partial-correlation CI of [-1.0, +1.0], i.e.
   unidentifiable. Check degrees of freedom before buying GPU time; this cost
   ~100 minutes of A100 to learn nothing.

**Predictions registered before the data and falsified by it** -- recorded
because a prediction only counts if its failure is written down:
- "global correlation will stay at or below chance" -- false on 3S9D (+0.4683
  against a null of +0.2441).
- "base rate and positional coverage explain the spread, so 2JEL and 3HFM will
  land between 3S9D and 1JTG" -- false; 2JEL (25%, intermediate) produced the
  most extreme negative. There is no working account of the between-complex
  variation. Do not invent one.

---

### Superseded: the 3S9D-only reading, recorded 2026-09-30 before the tie-break

### 3S9D chain A -- 50 positions, 6 hot spots, 12% base rate

The best-powered position-level test this project has run: nearly twice 1JTG's
position count, at a quarter of its hot-spot prior, on a shorter sequence
(L=307 vs 427).

| metric | simplex (L,20) | L2 norm | shuffled null p95 |
|---|---|---|---|
| Spearman vs \|ddG\| | **+0.4683** CI [+0.2060, +0.6734] | +0.0356 | +0.2441 |
| partial, all confounds | **+0.3632** CI [-0.0109, +0.7036] | -0.1367 | -- |
| AUROC | **0.8674** CI [0.6792, 0.9956] | 0.5644 | 0.7197 |
| AUPRC | **0.6335** | -- | 0.3390 |
| hot-spot precision@5 | **0.60** (chance 0.12) | **0.00** | -- |

Spearman clears the null and its CI excludes zero. AUROC and AUPRC both clear
their nulls, and AUPRC 0.63 against a 0.12 base rate is the strongest
needle-finding number in the project. **The reduction is what produced it:**
0.0356 -> 0.4683, and 0.00 -> 0.60 at the top of the list. On this complex the
L2 norm was indistinguishable from noise, exactly as entry 31's
concentration-of-measure argument predicts.

### But 1JTG says the opposite, and that is unresolved

Same artifact, both reductions, so the comparison is clean:

| complex | positions | base rate | L2 | simplex |
|---|---|---|---|---|
| 3S9D A | 50 | 12% | +0.0356 | **+0.4683** |
| 1JTG B | 28 | 43% | +0.2946 | **+0.1714** |

The reduction helps enormously on one complex and hurts on the other. Do not
report either in isolation. Candidate explanations, none yet tested: 1JTG's 43%
hot-spot prior compresses the dynamic range; 3S9D has far denser positional
coverage; 1JTG is an enzyme-inhibitor pair and 3S9D a different interface
class. 2JEL (P, 32 positions, 25%) and 3HFM (Y, 15 positions, 33%) were queued
the same session to break the tie.

### Honest caveats, recorded before the tie-break landed

- **The partial's CI still grazes zero** (-0.0109). "Adds signal beyond
  geometry" is suggestive, not established.
- **Confounds still win head-to-head on 3S9D**: `burial` +0.6411 vs |ddG|
  against the attribution's +0.4683; `distance_to_partner` -0.5769;
  `rsa_complex` -0.5698; `delta_rsa` +0.5620.
- **Completeness is out of tolerance**: 8.51% relative on 3S9D, 5.99% on 1JTG.
  Real geometry roughly tripled the `f(x) - f(baseline)` span, so m=32 is now
  too few quadrature steps for the absolute 0.10 gate. Re-run the survivors at
  m=64 before quoting anything as final.
- **A prediction registered before the run failed.** It was predicted that the
  global correlation would stay at or below chance and that the gradient would
  track `residue_volume` most strongly. Both wrong on 3S9D: the correlation
  clears the null, and the strongest associations are `rsa_complex` (-0.3815)
  and `burial` (+0.3322). Recorded because a prediction only counts if its
  failure is written down too.
- **1JTG was the weakest available test and was chosen for cache convenience.**
  At a 43% base rate a perfect top-5 is a 1-in-125 event; on 3S9D it is
  1-in-300,000. Complex selection must be justified on positional coverage and
  base rate from here on, never on what is already featurised.

---

## 15. Boltz-2 affinity head cannot be the attribution target for PPI — 2026-10-01

**Verdict: will not run as shipped on a protein-protein complex, and is not
differentiable to `res_type` even if patched.** Upstream at
`jwohlwend/boltz@b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. Extends entry 36,
which established the parser gate; this section traces the full architecture.

### The parser gate and its neighbours

`src/boltz/data/parse/schema.py:1065-1070` raises
`ValueError("Chain {binder} is not a ligand! Affinity is currently only
supported for ligands.")`. Further guards: `:1075-1077` (one affinity ligand
max), `:1103` (no multi-copy), `:1191-1192` (no multi-residue binder),
`:1206-1211` (<=128 heavy atoms, warns >56).

The token mask itself is `asym_id`-based only
(`src/boltz/data/tokenize/boltz2.py:172-173`), and `AffinityInfo`
(`src/boltz/data/types.py:565-569`) is just `(chain_id, mw)`, so pointing it at
a protein chain is *structurally* legal — the schema guard is the only
mol-type gate on the binder. But
`src/boltz/data/feature/featurizerv2.py:2307-2308` reads
`data.record.affinity.mw`, and `mw` is computed only in the RDKit ligand
branches (`schema.py:1202`, `:1274`), so a MW would have to be injected
manually.

### Gradient is identically zero

`src/boltz/model/models/boltz2.py:704-711` (and `:631-652` in the ensemble
branch) calls `self.affinity_module(s_inputs=s_inputs.detach(),
z=z_affinity.detach(), x_pred=coords_affinity, ...)`. `coords_affinity`
(`:624`) is `dict_out["sample_atom_coords"].detach()[best_idx]`, a diffusion
sample selected by `torch.argsort(dict_out["iptm"])[0].item()` (`:622-623`).

Separately, `src/boltz/model/modules/affinity.py:105-108` bucketises distances
(`cdist` -> `(d.unsqueeze(-1) > self.boundaries).sum(-1).long()` ->
`nn.Embedding`), so `x_pred` is never differentiable — though under this
repo's fixed-WT-geometry approach that is a constant, as in the confidence path.

A differentiable entry point does exist: `boltz2.py:627`
`s_inputs = self.input_embedder(feats, affinity=True)`, and
`InputEmbedder.forward` (`trunkv2.py:173, 192-196`) does
`res_type = feats["res_type"].float()` -> `self.res_type_encoding(res_type)` (a
Linear). **But all three tensors passed to `affinity_module` carry `.detach()`,
severing the graph between embedder and head.**

### Crop discards most of the interface

`src/boltz/data/crop/affinity.py` (`AffinityCropper`, `neighborhood_size=10`,
`max_tokens_protein=200`), invoked at
`src/boltz/data/module/inferencev2.py:238-244` with `max_tokens=256,
max_atoms=2048`. Crop centre is `affinity.py:71`, keyed on `affinity_mask` not
mol_type. `:89-93` uses `mol_type == NONPOLYMER` to define `ligand_ids`, which
is exempted from the 200-protein-token budget — that set is empty for a
protein binder, so the binder competes for the protein budget. 1JTG is 427
tokens against a 256 cap.

### Featurisation regime differs from confidence

Affinity featurisation uses **MSA depth 1**
(`featurizerv2.py:2294-2302`, `max_seqs=1`) and disables templates (`:2312`).

### Outputs

Both heads in one `AffinityHeadsTransformer`, `affinity.py:214-222`:
`affinity_pred_value` = log10(IC50), IC50 in uM, lower = stronger
(`docs/prediction.md:244`); `affinity_logits_binary` -> sigmoid ->
`affinity_probability_binary` (`docs:242`). `boltz2.py:688-698` applies an MW
correction `1.03526*v - 0.59993*mw**0.3 + 2.83288`, fit on small molecules and
meaningless at protein mass.

### Upstream confirms

Maintainer, issue [#301](https://github.com/jwohlwend/boltz/issues/301): "Not
at the moment! We only predict affinity for protein - small molecule ligands."
Same in [#336](https://github.com/jwohlwend/boltz/issues/336).
`docs/prediction.md:104` "must be a ligand chain (not a protein, DNA or RNA)";
`:312` table marks Affinity protein-ligand only.

### Fine-tuning does not help

[arXiv:2512.06592](https://arxiv.org/abs/2512.06592) froze the trunk,
**re-initialised and retrained the affinity module from scratch**, and widened
its inputs to all inter/intra-chain representations. Results: TCR3d
r=0.153/rho=0.091; PPB-Affinity r=0.338/rho=0.357 — below ESM2-650M (r=0.239)
and ProtT5 (r=0.48/rho=0.51). No zero-shot numbers reported. Their conclusion:
"current structure-based representations are not primed for performant affinity
prediction."

**UNVERIFIED:** `boltz2_aff.ckpt` hyperparameters (ensemble flag, module dims)
— checkpoint not downloadable during the check.

### Consequence

`PLAN.md:68-72` pre-registered: "attribution on ipTM should fail, and
attribution on Boltz-2's dense affinity head should work." The first half is
confirmed by the measured 2026-09-24 run. The second half is **untestable in
this model on PPI for architectural reasons** — not budget-limited. This
completes the negative rather than leaving it open, and it reinforces (does not
contradict) the Phase-5 post-mortem at section 14, lines 1411-1413.

---

## 16. MSA sensitivity — pin the wild-type MSA in stage 04, 2026-10-01

### MSA re-query under point mutation

New diagnostic `scripts/msa_sensitivity.py`, output
`results/msa_sensitivity.json`. 1JTG chain B, WT plus 5 alanine mutants
(E30A, Y52A, N88A, W111A, W149A), ColabFold API queried directly because
`boltz` is not installed locally.

Accession-level overlap (Jaccard) and profile L1 per position:

| Mutation | %shared | L1/pos | L1@mut | L1@other | ratio |
|---|---|---|---|---|---|
| Y52A | 98.6% | 0.0147 | 0.0274 | 0.0147 | 1.9 |
| N88A | 90.4% | 0.0689 | 0.0804 | 0.0689 | 1.2 |
| W111A | 67.1% | 0.1213 | 0.1485 | 0.1211 | 1.2 |
| W149A | 64.6% | 0.1681 | 0.1646 | 0.1682 | 1.0 |
| E30A | 50.0% | 0.4540 | 0.8378 | 0.4517 | 1.9 |

L1 rank-orders **perfectly** with homolog-set overlap. The L1@mut/L1@other
ratio averages 1.4 (range 1.0-1.9), so **the profile shift is uniform across
columns, not localised at the mutated position**. Mechanism: the profile moves
because alignment membership changes (MMseqs2 scores against the query, so
borderline hits fall below threshold), not because the mutated column changes.

**The WT replicate control is INVALID.** Three repeat WT fetches returned 100%
overlap, L1=0.000 — but the server reused the same ticket ID, so this measured
*cache* determinism, not *search* determinism. It cannot separate mutation
effect from server variance. Do not cite the "zero noise floor".

**Decision is unaffected either way.** IG holds `profile` fixed by
construction, so this variation is unrepresentable by the gradient under either
explanation, and it carries no position-specific information IG could use. **Pin
the WT MSA in stage 04** by passing `msa=msa_by_chain` as stage 02 already does
(`scripts/02_embed_deltas.py:285,335`). `--no-msa-server` alone is NOT the fix
— it supplies no MSA at all, a third condition.

**Correction.** The first version of this diagnostic reported "92.6% homolog set
difference, server nondeterminism dominates". That was an artifact of comparing
full a3m header lines, which carry per-query bit scores and E-values (same
accession `UniRef100_UPI001E2E0A3B` appears with scores 250/259/260 across
queries). Corrected by comparing accessions only.

### ColabFold `mode="all"` may cost two-thirds of MSA diversity

`mode="all"` returns UniRef hits only: **71 homologs** for 1JTG chain B.
`mode="env"` additionally returns ~137 BFD/Mgnify/MetaEuk/SMAG environmental
hits in `bfd.mgnify30.metaeuk30.smag30.a3m`. The repo uses `mode="all"`.

Relevance: AlphaInterp (Feldman & Skolnick, bioRxiv 2026,
[PMC13131572](https://pmc.ncbi.nlm.nih.gov/articles/PMC13131572/)) finds MSA
**diversity, not depth**, drives performance: >=80% identity homologs give
near-zero benefit even at n=10; <=30% identity gives TM 0.88-0.92; 5-10
appropriately diverged homologs approach full performance; fake MSAs give no
benefit at any depth. Environmental databases are where diverged homologs live.

**Unresolved.** This affects every Boltz-2 number produced so far, not only the
pinning question. One run at `mode="env"` would settle it.

---

## 17. Pivot to protein-ligand -- opened 2026-10-02

Motivation: sections 13-15 closed PPI as a negative because every target was a
structure-confidence score; Boltz-2's affinity head exists only for
protein-small-molecule. Protein-ligand is where the pre-registered claim
("attribution on the affinity head should work", `PLAN.md:68-72`) is testable.

### Priority order -- decided by the user 2026-10-02

**Model vs lab comes first; shortcut vs brute-force scan is conditional on it.**
If Boltz-2's affinity output already correlates poorly with measured data on
the chosen ground truth, do NOT spend GPU on the shortcut-vs-scan comparison --
a faithful shortcut to a model that is wrong about biology is not a useful
result for this project. Run the model-vs-lab check (forward passes only,
cheap at <=256 tokens) before any attribution-vs-scan work, and state the
threshold that counts as "sucks" before the number exists.

This reverses the README's framing, which calls shortcut-vs-scan "the honest
test of the shortcut". It still is the honest test of faithfulness; it is just
no longer worth buying unless the model clears the lab bar.

### Two cheap gates before any spend

1. Gradient reachability: patch out the three `.detach()` calls feeding
   `affinity_module` (`boltz2.py:704-711`, ensemble branch `:631-652`) and
   confirm a finite non-zero gradient reaches `res_type`, with IG completeness.
2. Ground-truth inventory: which mutation-vs-ligand or ligand-series datasets
   have complexes that fit the 256-token affinity crop, with how many
   measurements each -- checked against the confound panel's degrees of
   freedom (the 3HFM lesson, HANDOFF item 6).

### Metrology of the A-vs-C comparison, researched 2026-10-02

A = `grad[i] . (E_mut[i] - E_wt[i])`, the one-pass prediction (`attrib.py:393`).
B = `affinity_pred_value(mut) - affinity_pred_value(wt)` by actual re-run.
C = the lab measurement.

**A is exactly TISM** (Taylor in-silico mutagenesis, iScience PMC11404212).
Because `res_type` is one-hot, `grad . (e_mut - e_wt) = grad[mut] - grad[wt]`,
which is the published estimator. Two consequences:

* Free correction. The one-hot *difference* form cancels the off-simplex
  constant gradient component that Majdandzic & Koo (Genome Biology 2023,
  10.1186/s13059-023-02956-3) show corrupts raw gradient maps. Raw
  `grad[mut]` alone would not get this. Keep the difference form.
* Expected ceiling. On genomic CNNs TISM-vs-true-ISM is mean Pearson 0.70
  (median 0.73, n=741,798 maps), degrading to 0.61 on ReLU nets, and
  concordance FALLS as a model trains (0.85 early -> 0.71 at best predictive
  performance) as nonlinearity accumulates. **No protein-structure-model
  equivalent exists (UNVERIFIED -- none found).** Boltz-2 is deeper, the
  substitution is a 20-way discrete jump rather than a base swap, so expect
  the low end. This is why A-vs-B must be measured and never assumed.

**Hauser 2018 metrology.** 144 mutations, **8** TKIs (axitinib, bosutinib,
dasatinib, imatinib, nilotinib, ponatinib, erlotinib, gefitinib) -- not 6.
Tabulates dpIC50, reports ddG via `ddG = -RT ln(IC50_wt/IC50_mut)` at
**T = 300 K, RT = 0.596** -- so 1 log10 unit = **1.373** kcal/mol here, not the
298 K value of 1.364. Safer: convert Hauser back to dlog10(IC50) and compare in
log units, using no RT at all. 131 of 144 come from **cell-viability** IC50,
not biochemical; 13 (erlotinib/gefitinib) required docking.

**Assay noise is inside C and is measured.** Hauser's own ddG-from-dpIC50 vs
ddG-from-dKd check: **RMSE 0.81 [0.59-1.04] kcal/mol at n=27** (~0.6 log10),
against inter-lab repeatability of 0.32 [0.28-0.36]. Mechanism is real for
ABL1: T315I raises ATP affinity ~2x (PNAS 1919221117), which moves IC50 by up
to ~0.3 log10 with Ki unchanged. This is a hard floor on B-vs-C -- noise, not
invalidity.

**Boltz-2 has NO published mutation-effect evaluation.** Every reported number
is per-assay ligand-series Pearson: FEP+ 4-target r=0.66, CASP16 r~0.66, and
r>0.55 on only 3 of 8 blinded Recursion assays. The paper's stated reason for
pooling Ki/Kd/IC50 labels is that the Cheng-Prusoff term cancels "when
comparing affinity values within the same assay" -- i.e. fixed protein, varying
ligand. **A protein mutation changes the protein, so that cancellation argument
does not transfer to our comparison.** BoltzOmics (PMC13401010) r=0.63-0.71 is
across drugs within a fixed variant, NOT mutation deltas, and does not support
mutation-delta skill. This is the honest structural cap on B-vs-C.

**MW correction.** `boltz2.py:688-698` applies `1.03526*v - 0.59993*mw^0.3 +
2.83288`. For a fixed ligand the offset cancels but the **1.03526 slope does
not**, so a gradient on the raw head output is 3.5% small against the reported
value. Rank correlations are unaffected; a calibration slope is not. Pick one
surface and state which.

### Reference scale -- the honest bar, PROPOSED not yet ratified

| method | set | r |
|---|---|---|
| FEP+ alchemical (Hauser, the ceiling) | ABL1/TKI n=142 | RMSE 1.07, MUE 0.79 kcal/mol; accuracy 0.88 but **sensitivity 0.47** |
| flexddg | external TKI n=116 | 0.49 |
| mCSM-lig | external TKI n=116 | 0.43 |
| PremPLI | external TKI n=116 | 0.43 |
| mCSM-lig / PremPLI | Platinum n=586 | 0.76 / 0.75 -- **both trained on it; leaked, not a target** |

Source: Brief Bioinform 2026, PMC12874888. **Pre-register against the 0.43-0.49
external band, never against Platinum's 0.75.** Near 0.4 on B-vs-C matches the
ML field; near 0.2 fires the stop condition above.

### Invalidity flags -- distinct from noise

1. **Distal / conformational-equilibrium mutations are unmodellable by
   construction**, not merely noisy. Hauser notes mutations can shift kinase
   conformational populations and modulate ATP affinity. This repo scores at
   fixed WT geometry on a single argmax-ipTM diffusion sample (section 15), so
   conformational-selection resistance cannot be represented. Exclude or report
   separately; do not average them in.
2. **A-vs-C alone is unfalsifiable.** It is the product of surrogate fidelity
   (A-vs-B) and model biology (B-vs-C). A null in A-vs-C cannot be apportioned
   between them. So B must be computed even though section 17 deprioritises the
   scan comparison -- see the ordering note below.

### Corrected ordering -- cheap-forward-first, and it preserves the priority

The user's instruction (do not buy shortcut-vs-scan if the model already fails
against lab data) survives intact, because the cost structure inverts here:

1. **Crop determinism and coverage first, and it gates everything.** If
   `AffinityCropper` keeps a different residue set for WT than for a mutant,
   then B is a difference between scores computed on two different protein
   subsets, which corrupts B as well as the embedding deltas. Free to check.
2. **B vs C, forward passes only.** ~144 forwards at <=256 tokens. No backward
   pass, no IG, no 80 GB requirement. This is "is the model right about
   biology" and it is the gate.
3. **Only if B-vs-C clears the bar: A.** IG with a differentiable trunk is the
   expensive part. Gating it on step 2 means the cheap measurement decides
   whether the expensive one is ever bought.

Computing B is technically "the brute-force scan", but it is the prerequisite
for the model-vs-lab number, not the shortcut-vs-scan comparison that was
deprioritised. Both A-vs-B and A-vs-C stay behind gate 2.

### ABL1 ground truth, counts verified independently 2026-10-02

Source found: `Supplementary Data 3.xlsx` in the authors' repo
`github.com/kehauser/Predicting-resistance-of-clinical-Abl-mutations-to-targeted-kinase-inhibitors-using-FEP`.
Loader `src/igv/abl.py`, fetch `scripts/17_fetch_abl.py`. I re-derived every
count from the downloaded CSV rather than trusting the agent's report; all
matched exactly.

| quantity | verified |
|---|---|
| total rows | 279 |
| measured (IC50 and DDG_exp both present) | 144 |
| ABL TKI measurements | 131 |
| TKIs in file | 8 -- 6 ABL plus erlotinib/gefitinib as EGFR negative controls |
| per-TKI | 21 each, axitinib 26 |
| **distinct mutations** | **31** |
| **distinct positions** | **20** |
| **positions with >1 distinct mutant AA** | **6 of 20** |
| T315I / imatinib DDG_exp | +2.32 kcal/mol |

**`DDG_exp` is ddG in kcal/mol, NOT dpIC50.** Computed as
`RT ln(IC50_mut / IC50_wt)`. Confirmed by back-computing the implied WT IC50
per TKI: constant to +-0.005 in log10 across all mutations of a given TKI
(imatinib WT 199.2 nM, dasatinib 2.0 nM). **Positive = resistance**, which is
already the SKEMPI convention, so no negation is needed anywhere.

**Numbering is ABL1 isoform 1b** (clinical standard, T315 the gatekeeper;
isoform 1a would make it T333, a +18 offset). Verified by mapping all 31
mutations onto PDB 4WA9 chain B: 31/31 wild-type residues match, zero
mismatches. Note 1OPJ is *mouse* ABL1, adjusted by the authors to the human
reference.

**Corrected PDB/TKI mapping** (my brief to the agent had this wrong):
3UE4 = bosutinib, 3CS9 = nilotinib, 4XEY = dasatinib.

### The 144 is inflated by the inhibitor axis -- Hauser cannot carry the headline

The mutation axis has only **31 variants across 20 positions**. The panel is
8 covariates without an MSA, 10 with (`skempi.py:860-900`). 20 positions
against 8-10 covariates is the **3HFM trap** verbatim (HANDOFF item 6: 15
positions, 8 confounds, partial CI [-1.0, +1.0], unidentifiable, ~100 min of
A100 to learn nothing). **Hauser cannot support a confound-controlled partial
correlation** -- and controlling for burial/RSA is precisely what killed the
protein-protein arm (entry 38), so it is not optional.

Worse for the DMS-style route: only **6 of 20 positions have more than one
mutant amino acid**, so `within_position_spearman` (`metrics.py:408-440`) has
nothing to rank at 14 of 20 positions. The within-position trick that made the
Starr arm clean does **not** transfer along the mutation axis here.

### But the inhibitor axis is the clean design, and it is free

Hauser is a 2D panel: ~31 mutations x 6 inhibitors. Flip the comparison to
**within-mutation, across-inhibitor** -- for each mutation, rank the 6 TKIs by
measured ddG and compare against the model's ranking. Two properties:

1. **Burial, RSA, residue volume, conservation and position are all constant
   within a mutation**, so every confound that absorbed the position-level
   signal in entry 38 is cancelled *structurally*, not statistically. No
   partial correlation needed, so the degrees-of-freedom problem disappears.
2. **It is the one regime Boltz-2's own label-pooling argument covers.** The
   paper justifies mixing Ki/Kd/IC50 by Cheng-Prusoff cancellation "when
   comparing affinity values within the same assay" -- fixed protein, varying
   ligand. Within-mutation across-inhibitor is exactly fixed protein, varying
   ligand. The mutation axis is the direction the justification does NOT cover.

It also has a strong qualitative anchor: ponatinib was designed to retain
potency against T315I while imatinib fails on it. A model that cannot reproduce
that ordering has no business being attributed.

**Caveat before anyone gets excited:** this measures sensitivity to the
*ligand*, while our attribution target is the *protein* `res_type` gradient.
A good within-mutation number would validate the model's affinity output, not
the protein-side attribution. It gates B-vs-C honestly; it does not stand in
for the mutation-axis result we actually want.

### Recommendation, pending ratification

Promote the **imatinib DMS** (4,922 variants, ~270 positions, 19 AAs per
position -- `pritchardlabatpsu/abl_dms`) from scale-up to **primary** for the
mutation axis: it has the degrees of freedom and restores within-position
ranking. Keep Hauser for (a) the within-mutation across-inhibitor gate above
and (b) the T315I/ponatinib qualitative check. Unresolved for the DMS: it reads
out cell growth under drug, so loss-of-function is confounded with resistance
and it needs a folding-vs-binding control like the Starr arm's expression
readout.

### AffinityCropper is mutation-invariant -- VERIFIED INDEPENDENTLY 2026-10-02

This was the open question that could have invalidated the whole affinity arm:
if the crop kept a different residue set for wild-type than for a mutant, then
`score(mut) - score(wt)` would be a difference across two different protein
subsets, and `E_mut - E_ref` would be computed in two different index spaces.
Not noisy -- invalid.

**It is safe.** Verified by reading the actual pinned source, not a report:
downloaded the `boltz-2.2.1` sdist from PyPI
(`files.pythonhosted.org/.../boltz-2.2.1.tar.gz`) and read
`src/boltz/data/crop/affinity.py` directly.

1. The crop distance is computed from `valid_tokens["center_coords"]` -- the
   **tokenised structure's reference coordinates**, i.e. the input PDB -- and
   NOT from `x_pred`. So the diffusion sample cannot perturb the crop.
2. Selection is `indices = np.argsort(dists)` followed by a greedy
   neighbourhood expansion. Purely geometric.
3. **No RNG anywhere.** The only occurrences of "random" in the file are in the
   docstring, which still documents a `random : np.random.RandomState`
   parameter that is *not in the signature* -- a stale doc, not a live call.
   `grep -n "random\|rng\|seed\|shuffle\|permut"` returns only lines 47-48,
   both docstring.

**Therefore:** a point mutation changes only `res_type`, while `center_coords`
come from the same PDB for WT and mutant, so `dists` is identical and the crop
selects the identical residue set. The embedding delta is computed in one index
space. Safe.

**The condition this rests on, which must not be quietly dropped.** The
invariance holds *because* we hold geometry fixed at one real structure. If
anyone later switches to co-folding each mutant separately, `center_coords`
becomes per-mutant, the crop can shift, and this guarantee is void. Any move to
per-mutant structure prediction re-opens this question.

Also confirmed: boltz 2.2.1 **is** on PyPI (wheel + sdist) so
`docker/Dockerfile:22` is fine. An earlier `pip download --no-binary :all:`
failure that appeared to show 2.0.3 as latest was a local requires-python
filter, not a broken pin. Do not "fix" the Dockerfile on that basis.

### CORRECTION -- crop size is not the degrees-of-freedom constraint

The agent's report concluded "at 200 sites, the analysis is comfortably
identifiable (200 >> 40)". **That is wrong and must not be read as settled.**
It counts *residues surviving the crop*, but identifiability is bounded by the
number of **measured ground-truth positions**, which is 20 for Hauser (section
above) against a panel of 8-10 covariates. A 200-residue crop containing only
20 measured positions is exactly as unidentifiable as 3HFM's 15. The crop
matters for a different reason -- whether each measured site is visible to the
model at all -- and `check_crop_coverage` now answers that. Keep the two
questions separate.

### Run parameters for the first protein-ligand IG, decided 2026-10-02

**Baseline: `mean_aa`. Decided by the user, and it is the one parameter that
must not be wrong before spending.** `scripts/18_gate_affinity.py` shipped with
a hardcoded zeros baseline (`torch.zeros_like(res_type_orig)`), and
`build_mean_aa_baseline` (`attrib.py:52`) was never wired into the affinity
path. The precedent is in the README: the all-zeros baseline scores 11.43 where
the real complex scores 2.62, pushing the integral off-manifold, and on 1JTG
chain B the measured Spearman against ddG was 0.238 with zeros against **0.357
with `mean_aa`** -- the difference between this project's only positive result
and a null. A zeros-baseline affinity run would have reproduced a null for a
cause already diagnosed and already fixed once.

**One run yields every mutation, so there is no affinity stage-02.** The
artifact is the `(1, L, vocab)` `res_type` attribution. Because `res_type` is
one-hot, the prediction for `w -> m` at position `i` is `grad[i][m] -
grad[i][w]` (the TISM identity), so the single gradient already contains all 20
substitutions at every position. The earlier audit listed "build a new stage 02
for affinity embedding deltas" as a blocker; it is not one. Consequence for
cost: **the GPU produces one artifact and every correlation is computed locally
for free**, so a disappointing number never requires re-renting a card.

**Defaults chosen, with reasons:**

| parameter | value | why |
|---|---|---|
| baseline | `mean_aa` | above |
| m (completeness ladder) | up to **64** | HANDOFF item 5: real geometry tripled the `f(x) - f(baseline)` span and m=32 failed completeness on 3 of 4 runs. Anything final needs 64. |
| geometry | **real structure, not zeros** | Gate A logged `coords.abs().max() = 0` from sequence-only YAML. Fine for a flow test, wrong for a capture. |
| structure for imatinib | **2HYY** (proposed) | Hauser used 1OPJ, which is *mouse* ABL1 adjusted to the human reference. We hold geometry fixed, so a species mismatch at the pocket is a live confound. Departs from the paper's setup -- recorded as a deliberate deviation, not an oversight. |
| first target | imatinib alone | ~21 mutations, shakes out a path that has never executed. All six TKIs unlocks the within-mutation across-inhibitor design but is six runs. |

**Infrastructure facts verified 2026-10-02, against stale documentation:**
`igv-gpu` already exists in `us-central1-a` as `a2-ultragpu-1g` (A100-80GB),
status TERMINATED -- so this is a restart, not a provision, and the boot disk
already carries the Boltz-2 weights. Regional `NVIDIA_A100_80GB_GPUS` quota is
now **1**, not the 0 that `docs/CLOUD.md` recorded (table corrected). Also
available and unused: `NVIDIA_L4_GPUS` = 8, `NVIDIA_A100_GPUS` (40 GB) = 1.

**The 80 GB card is probably unnecessary for this arm.** The measured 55.2 GiB
was the confidence path at L=730 with a full MSA. The affinity path caps at
L=256 with MSA depth 1 and templates off, which extrapolates to ~7 GiB on
pair-tensor L^2 scaling. If that holds, subsequent runs belong on an L4 at a
fraction of $5.07/hr. The extrapolation is crude, so the first run stays on the
already-provisioned A100 rather than debugging a new instance type while also
debugging code that has never run.

### The ABL1 + imatinib run, fully specified from real geometry -- 2026-10-02

All of the following was derived locally and free, from the 2HYY PDB and
UniProt. No GPU involved.

**Numbering frame, and a label correction that matters.** UniProt **P00519 is
1130 residues**, which is isoform **1a**, and 2HYY chain A's author numbering
matches it with shift **exactly 0** (verified: the only shift in [-40, +40] for
which UniProt agrees with all 263 resolved 2HYY residues is 0). The gatekeeper
is T at 315 in that frame, which is the clinical T315I convention.

So `src/igv/abl.py`'s docstring calling the frame "isoform 1b" is
**mislabelled** -- the numbers are right and were empirically verified 31/31
against 4WA9, but the isoform NAME is wrong. 1b is the 1149-residue form, +19.
This matters because a future reader who "corrects" for an 18 or 19 offset
would silently misalign every mutation. Fix the label, keep the numbers.

**Ground truth sits entirely inside 2HYY.** All 31 measured mutations match
their wild-type residue at a resolved position in chain A, and all 20 distinct
positions are present and resolved -- `31 of 31` and `20 of 20`, zero
exceptions. Chain A spans 235-498 with **one gap, residue 275**.

**Crop survival, predicted from real geometry: 19 of 20.** Minimum heavy-atom
distance from each residue to imatinib (`STI`, 37 heavy atoms in chain A), rank
each residue, keep the nearest 200 as an approximation of the cropper's budget:

| surviving | lost |
|---|---|
| **19 of 20** measured positions | **E459 only**, at 27.2 A, rank 254 of 263 |

The pocket positions are unambiguous: T315 at 3.0 A (rank 5), V299 3.3, F317
3.3, L248 3.6, Y253 3.7, F359 4.2. This is the number that gates whether the
run is worth buying, and it clears. (Approximation caveat: the real cropper
does a greedy +-10 neighbourhood expansion rather than a pure distance rank, so
treat 19 as indicative, not exact. `check_crop_coverage` reports the truth at
runtime.)

**The additive offset was broken by the gap, and the fix is a continuous
construct.** `--resnum-offset 229` was wrong twice over: 2HYY chain A starts at
**235**, not 229; and an additive index->resnum offset is only valid if the
construct is contiguous. With residue 275 absent, everything from 276 onward
shifts by one -- which is **14 of the 20 measured positions** (276, 279, 292,
299, 315, 317, 343, 351, 355, 359, 384, 396, 459, 486). That misalignment is
silent and would still produce a correlation, against the wrong residues.

Fix: pass a **continuous** 235-498 construct, 264 residues, with position 275
filled from UniProt (it is **E**). Then `index + 235 == residue number` holds
exactly and `--resnum-offset 235` is correct. Verified: index 0 -> W235,
index 80 -> T315, index 251 -> F486.

Construct (264 aa, ABL1 235-498, P00519 frame):

```
WEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQ
```

**Known landmine:** residue 275 (E) has no coordinates in 2HYY, so coordinate
injection leaves it at the origin. It will read as very distant from the ligand
and be cropped, which is harmless since 275 is not a measured position -- but
the injection must REPORT unmatched residues rather than pass silently, because
a larger unmatched set would corrupt the crop geometry without any visible
symptom.

**Also settled:** 2HYY is human wild-type ABL1 kinase domain with imatinib
(RCSB: "Human Abl kinase domain in complex with imatinib (STI571, Glivec)",
*Homo sapiens*). It is NOT the T315I mutant -- an agent claimed that and it is
false. 1IEP, proposed as a human alternative, is *Mus musculus*; so is 1OPJ,
which is what Hauser used. Choosing 2HYY is therefore a deliberate upgrade to
human wild-type geometry and a deliberate deviation from the paper.

Diffusion sampling in `predict_structure_coords` is **not seeded**, so
`--x-pred predicted` is not reproducible across fresh runs. `--x-pred pdb` is
deterministic and is the right mode here.

## 18. FIRST PROTEIN-LIGAND RESULT -- the model-vs-lab gate FAILS, 2026-10-02

**B-vs-C on ABL1 + imatinib. The pre-registered sign-flip kill condition fired.
Do not buy the IG run on this system.**

Run: `scripts/20_affinity_scan.py`, fp32, ABL1 kinase domain 235-498 (264 aa,
contiguous, P00519 frame), ligand `STI` by CCD code, geometry fixed at **2HYY**
chain A with experimental coordinates injected, single-sequence pinned MSA,
forward passes only. **0.6 min of A100.**

### Plumbing all verified good

| check | result |
|---|---|
| coordinate injection | **2110 / 2189 atoms (96.4%)**, 1 unmatched residue: A:275 -- the crystal gap, exactly as predicted |
| ligand coordinates | injected via CCD; the zero-coordinate guard did NOT fire |
| tokens after crop | 237 non-padding, token map 201 entries |
| crop coverage | **10 of 13** imatinib positions; lost **351, 355, 459** |
| wild-type score | `affinity_pred_value` = **0.579723** = log10(IC50/uM) |

Two corrections to my own earlier predictions:
* I predicted 19 of 20 positions surviving from a nearest-200 distance rank.
  The real cropper did worse: it dropped **M351 (8.5 A, rank 71)** and
  **E355 (9.4 A, rank 82)**, which a pure distance rank says should survive
  comfortably. The greedy +-10 neighbourhood expansion does not behave like a
  distance rank. **Do not predict crop coverage by distance ranking.**
* Imatinib covers only **13** distinct positions, not 20 -- the 20 is the union
  across all six TKIs.

### The result

n = 18 scored mutations, one inhibitor.

| statistic | value |
|---|---|
| **Spearman(delta_score, DDG_exp)** | **-0.1879** (p = 0.455) |
| Pearson | -0.4295 |
| permutation null (10k) | p5 -0.409, p95 +0.396 -- **does not clear its own null** |
| two-sided permutation p | 0.4498 |
| bootstrap 95% CI | **[-0.684, +0.359]** |
| model dynamic range | 0.367 log10 against the lab's 1.574 log10 -- **4.3x compressed** |

**The sign is negative, i.e. pointing the wrong way, and it is indistinguishable
from noise.** But the decisive part is not the correlation, it is the
pre-registered qualitative condition:

| strongest measured resistance | DDG_exp | model delta | |
|---|---|---|---|
| L248R | +2.33 | -0.0006 | WRONG SIGN |
| T315I | +2.32 | -0.0991 | WRONG SIGN |
| E255V | +2.22 | -0.1772 | WRONG SIGN |
| Y253F | +1.90 | -0.0672 | WRONG SIGN |
| E255K | +1.50 | -0.0716 | WRONG SIGN |

**All five strongest resistance mutations are predicted as STRONGER binders.**
The stop condition named **T315I, E255K and Y253H** in advance, before any data
existed, as mutations "every method in the literature gets right". Three of the
five here are from that named list. E255V -- the second strongest measured
resistance -- carries the single most negative delta in the entire set.

**bf16 is exonerated as the cause.** A bf16 run gave T315I = -0.097656 against
fp32's -0.099106, and the signs agree throughout on the five above. But bf16
error of ~0.01-0.03 is a large fraction of a typical delta of ~0.1, so **fp32 is
mandatory for any quoted number** -- bf16 deltas land on a 1/128 grid at the
same magnitude as the effect being measured.

**Absolute calibration is also off.** WT predicted IC50 = 10^0.5797 = **3.8 uM**
against Hauser's inferred wild-type of **~0.20 uM** -- roughly 19x weak, about
1.3 log10 units.

### What this does and does NOT establish

It does NOT cleanly establish "Boltz-2's affinity head cannot rank mutations",
because the test confounds two things:

1. Boltz-2's affinity skill, and
2. **our perturbation model** -- `res_type` flipped while geometry stays fixed
   at the wild-type crystal.

T315I's resistance is steric: Ile's beta-branched side chain clashes with
imatinib and removes the Thr315 hydrogen bond. Holding coordinates at a
Thr-shaped pocket and flipping only the identity token may make that mechanism
**structurally unrepresentable**, which is the same invalidity class as the
conformational-resistance flag recorded earlier. A clean test of the model's
mutation skill would let geometry relax, which this repo deliberately does not
do -- fixed geometry is what makes the IG attribution well-posed in the first
place. **That tension is now a measured finding, not a hypothesis.**

### Consequence, per the standing priority

The user's instruction was to not buy the shortcut-vs-scan comparison if the
model already fails against real data. It fails. **The IG run on ABL1 +
imatinib is therefore not worth buying**, and was not run.

### Extending to six TKIs is NOT just compute

Each inhibitor needs its OWN co-crystal, its own verified construct range, gap
list, contiguity check and CCD code -- the full 2HYY workup repeated five times.
It cannot reuse 2HYY: imatinib binds the **DFG-out inactive** conformation
while dasatinib binds the **active** conformation, so scoring dasatinib in the
imatinib-bound structure would be meaningless. I briefly started this and
stopped; recording it so nobody mistakes it for a cheap win.

### Spend

~0.6 min of compute for the gate itself. Session total ~$10-12, nearly all of
it SSH recovery and five rounds of first-contact debugging, not science.

## 19. GOAL REFRAMED -- interpretability, not prediction. 2026-10-04

**The user's stated primary goal is now: look inside the model and see which
residues it deems most important.** This is an interpretability objective, not
a prediction objective, and it changes what gates what.

### What this changes

| | old framing | new framing |
|---|---|---|
| primary question | does the shortcut predict lab ddG? | what has the model learned to treat as important? |
| core validation | A vs C (shortcut vs lab) | **A vs B (is the attribution faithful to the model?)** |
| role of B vs C | **the gate** -- a failure stops the project | **context** -- a caveat on what the model's "importance" means |
| section 18's negative | a stop condition | a FINDING about the model's internals |

Section 18 stands unchanged as a measurement. Its interpretation changes: "the
model's notion of importance does not track measured binding energetics" is an
interpretability result about a model the field is actively building on, not a
reason to halt.

**The standing priority from 2026-10-02 (do not buy shortcut-vs-scan if the
model fails against lab data) is hereby SUPERSEDED for this arm.** It was the
right rule while the goal was prediction. Under an interpretability goal the
model being wrong about biology does not make its internals uninteresting --
arguably the opposite. [[model-vs-lab-before-scan]] needs updating.

### B is already in hand, so faithfulness is free

Section 18 produced 18 forward-pass deltas on ABL1 + imatinib. Those ARE the
ground truth for A-vs-B. Once the IG run yields the `(L, vocab)` attribution,
faithfulness is a local, zero-cost computation: compare `grad[mut] - grad[wt]`
against the measured `delta_score` for the same 18 mutations. No extra GPU.

### The trap this arm must not fall into

An interpretability claim with no ground truth is very hard to falsify, and
this project has already nearly been fooled once. Phase 5's pair map looked
like it recovered known hot spots -- 80 SKEMPI cycles at the 90.7th percentile,
10.4x the median -- and it was **geometry**: interface residues are close, and
the map was simply high on close pairs. Only the distance control separated the
two readings.

So "which residues does the model deem important" must be reported against:

1. **Faithfulness (A vs B).** Does the attribution predict what the model
   actually does when the residue is changed? This is the primary check and it
   is now free.
2. **Distance to the ligand, as a partial.** If high-attribution residues are
   simply the pocket residues, the map encodes geometry, not learned chemistry.
   Report the partial correlation against minimum heavy-atom distance to
   imatinib ALONGSIDE any enrichment claim. Never the enrichment alone.
3. **The random-weights control**, which already passes on the confidence path
   (Spearman 0.077 against a 0.3 threshold), so attribution depends on what the
   model learned rather than on input geometry. Re-run it on the affinity head.
4. **IG completeness**, so the attribution is arithmetically accountable to
   `f(x) - f(baseline)`.

A pretty heat map over the binding site, with none of the above, is the
convincing false positive this arm is most likely to produce.

### Novelty note

This strengthens rather than weakens the paper's position. The earlier claim
was attribution of a *structure-confidence* score. Attributing the actual
**affinity** output is the more valuable object, and PairSAE -- the closest
prior work, which has Boltz-2 + affinity + protein-ligand -- is SAE-based and
deliberately token-level, engineering around the quadratic object rather than
producing it. Section 18's failed correlation is also a usable result for that
paper: it is evidence that affinity-head representations are not aligned with
binding energetics, which bears directly on the probing and SAE literature
built on those representations.

## 20. IG ON THE AFFINITY HEAD -- the gradient works, the function does not. 2026-10-04

Run: `scripts/18_gate_affinity.py`, full-trunk path, fp32, `mean_aa` baseline,
2HYY geometry injected, ligand `STI` by CCD, m = {16, 32, 64}. ~19 min of A100,
**peak VRAM 16.80 GiB** (so the 80 GB card is unnecessary -- an L4 would do, and
quota for 8 is already held). Artifacts
`results/gate_affinity_abl1_imatinib.{json,npz}`, both local.

### Mechanically the seam works

| check | result |
|---|---|
| `res_type.grad` not None | PASS |
| not uniformly zero | PASS, max abs 1.10, zero fraction 0.0000 |
| shape | PASS, `(1, 237, 33)` |
| all finite | PASS, 0 NaN, 0 Inf |
| **trunk / s_inputs norm ratio** | **7.12x** (5.641 vs 0.792) |

The 7.12x ratio is the check I rejected the first implementation over, and it
clears comfortably: the trunk carries most of the gradient, so this is not the
shallow embedder-only path that would have passed a reachability gate while
meaning nothing.

### But completeness fails catastrophically, and non-monotonically

| m | ig_sum | f(x) - f(b) | rel err | wall |
|---|---|---|---|---|
| 16 | **-2.234** | -0.0847 | 25.38 | 152 s |
| 32 | **+2.994** | -0.0847 | 36.35 | 304 s |
| 64 | **+1.184** | -0.0847 | 14.98 | 607 s |

This is not "needs more steps". **The sum flips sign between m=16 and m=32 and
does not converge**, and it overshoots the quantity it must equal by 15-36x.
Contrast the confidence path, which reached rel err 0.0442 at m=32 and 0.0036
at m=5 on the pair layer. So `res_type_ig` is NOT arithmetically accountable
here and should not be quoted; the raw gradient is the only defensible object,
which is also what the TISM difference form uses.

### THE HEADLINE: the affinity head barely uses the protein sequence

| input | `affinity_pred_value` | implied IC50 |
|---|---|---|
| real ABL1 kinase domain | 0.5797 | **3.80 uM** |
| `mean_aa` -- uniform 1/20 at EVERY position | 0.6644 | 4.62 uM |
| all zeros | 0.8172 | 6.56 uM |

**Replacing the entire 264-residue protein sequence with a uniform amino-acid
distribution changes the predicted affinity by 0.085 log10 -- a factor of
1.2.** Zeroing it entirely moves it 1.7x. The model's affinity output for this
complex is essentially independent of the protein sequence; it is reading the
structure (which we hold fixed) and the ligand.

**And single point mutations move the score MORE than erasing the whole
sequence.** From section 18, E255V gives -0.1772 and T315A +0.1899, both larger
in magnitude than the 0.0847 full-sequence-erasure effect. A model with a
coherent sequence-to-affinity mapping cannot behave this way. The per-mutation
deltas are therefore better read as numerical roughness than as the model
"believing" anything about those residues.

This one fact explains everything else: if the function barely depends on the
variable being differentiated, the gradient is dominated by noise -- hence the
non-convergent completeness, and hence the faithfulness result below.

### Faithfulness (A vs B) is ANTI-correlated

Primary validation under section 19's reframe, computed locally at zero cost
against section 18's 18 forward-pass deltas:

| | value |
|---|---|
| Spearman(A, B) | **-0.4964** (p = 0.036) |
| Pearson | -0.5384 |
| permutation null p5 / p95 | -0.399 / +0.408 |
| two-sided permutation p | 0.0383 |
| scale | median abs A = 0.0093 vs median abs B = 0.0675 -- **7x too small** |

The gradient prediction points the OPPOSITE way to what the model actually does
when the residue is changed, and it clears its own null on the wrong side.
**Honest caveat:** n = 18, this is one of several correlations computed on the
same small set, and B itself is of noise-scale magnitude. Treat -0.50 as
suggestive of systematic anti-correlation, NOT as established. The scale
mismatch is the more robust half: a first-order term 7x smaller than the effect
it approximates means the linearisation is simply not valid here.

### The geometry confound did NOT fire this time

| | value |
|---|---|
| Spearman(importance, distance to imatinib) | **-0.151** (p = 0.033) |
| median distance, top-20 importance | 8.9 A |
| median distance, all residues | 11.4 A |

Both the L2 and simplex reductions agree (-0.151 vs -0.152), so section 14's
reduction question is moot here. The top residue is **430 at 20.3 A**, and the
top 12 include 407 (14.0 A), 408 (17.0 A), 409 (14.5 A) and 399 (12.5 A).

So unlike Phase 5, this map is **not** just the binding pocket -- which is the
trap section 19 named. But that is not good news: combined with the
anti-correlated faithfulness, the map corresponds to neither the geometry nor
the model's behaviour. There is no third thing it has been shown to track.

### Two bugs found in our own reporting

1. **The gate's printed "Crop coverage: 29/264 (10.98%)" is a units bug.** It
   intersected 0-based construct indices `0..263` with the token map's residue
   NUMBERS `235..456`, so the "surviving" list is exactly `{235..263}` -- an
   arithmetic artifact. **The real coverage is 200 of 264 residues (76%), and
   15 of the 20 measured positions**, losing 343, 351, 355, 459, 486. Fix the
   call site to pass residue numbers.
2. **The token map under-records the ligand**: 237 non-padding tokens = 200
   protein + 37 ligand, but the map holds 200 protein entries plus a single
   ligand entry keyed `('1', 0)`. Harmless for protein-residue attribution,
   wrong if anything ever needs ligand tokens.

### Where this leaves the interpretability goal

"Which residues does the model deem important" is not answerable on this system
**because the model does not meaningfully deem any residue important** -- its
affinity output is near-invariant to the whole sequence. That is a real finding
about a model the field is building probes and SAEs on, and it is cheap to
state and hard to argue with: three numbers, 3.80 / 4.62 / 6.56 uM.

What would sharpen it, cheaply and with forward passes only: scramble the
sequence, substitute an unrelated protein of the same length, and truncate the
MSA differently -- if affinity stays ~3.8-6.6 uM throughout, sequence-invariance
is established beyond the two baselines measured here. That is the next run,
and it is minutes of GPU.

## 21. RETRACTION of section 20's headline -- the sequence result is an artifact. 2026-10-04

**Section 20 called "the affinity head barely uses the protein sequence" THE
HEADLINE. That claim is withdrawn.** The confound noted in passing there is
fatal, and the literature quantifies exactly how fatal.

### Why the three numbers mean nothing as stated

We injected real all-atom 2HYY coordinates, so residue identity stays
recoverable from side-chain geometry. That is not a hand-wave -- it is measured:

| method | native sequence recovery from coordinates alone | source |
|---|---|---|
| ProteinMPNN | **52.4% overall, 90-95% in the deep core**, ~35% surface | Dauparas et al., Science 2022, PMC9997061 |
| ESM-IF1 | 51% overall, **72% buried**, from N/CA/C only | bioRxiv 2022.04.10.487779 |

**Imatinib occupies the buried ATP pocket of 2HYY -- precisely where identity is
most recoverable.** So a 1.2-1.7x shift under sequence erasure is the EXPECTED
behaviour of a redundant input channel and is no evidence about whether the
model uses sequence.

**The "single mutations move more than full erasure" point is also withdrawn.**
A zeroed one-hot is not a valid point in the input space, so that compares two
incomparable regimes. Majdandzic, Rajesh & Koo (Genome Biology 24:109, 2023)
give the right vocabulary and state it covers protein as well as DNA/RNA:
a zeroed `res_type` sums to 0 and is **off the probabilistic simplex entirely**,
where the network's function is unconstrained and gradients acquire a random
component; uniform 1/20 is **on the simplex but off the data manifold**. That
predicts our ordering -- uniform (4.62 uM) moving less than zeros (6.56 uM) --
without any claim about sequence reliance.

### And the conclusion would not have been novel anyway

"Structure-based affinity models learn protein or ligand identity rather than
interactions" is an established literature, and all five papers recalled from
memory were verified:

| paper | finding |
|---|---|
| Yang, Shen & Huang, Front Pharmacol 2020, PMC7052818 | refined set: complex R^2 ~0.80, ligand-only ~0.71, **protein-only ~0.84 -- protein-only BEAT the complex** |
| Volkov et al., J Med Chem 65:7946, 2022 | explicit interaction description gives "no advantage"; "memorization largely dominates true learning" |
| Boyles, Deane & Morris, Bioinformatics 36:758, 2020 | ligand-features-only RF is "predictive at a level similar to classical scoring functions" |
| Sieg, Flachsenberg & Rarey, JCIM 59:947, 2019 | "bias is learned implicitly and unnoticed from standard benchmarks" |
| Chen et al., PLoS ONE 14:e0220113, 2019 | DUD-E CNN enrichment attributable to analogue/decoy bias |
| Graber et al. (GEMS), Nat Mach Intell 2025, PMC12552120 | CASF2016 RMSE **1.402 with ALL protein information removed**; degrades to 1.609 after de-leaking -- so part of the effect is leakage |

### The gap that IS open, and the paper we would be measured against

No published inference-time input ablation of a **co-folding** affinity head
with coordinates pinned (AF3 / Boltz / Chai / Protenix). The near-miss is
"AlphaFold3 for Structure-guided Ligand Discovery", bioRxiv 2025.12.04.692352:
AF3 given **only ligand SMILES, no protein sequence and no template**, gave
non-random enrichment on **84% (36/43)** of DUDE-Z targets and matched or beat
DOCK3 on **47% (20/43)**. Their line -- "AF3 remembers where ligands go, but not
why they bind" -- is the ligand-side dual of our experiment, and it reads clean
precisely because removing the protein entirely leaves no redundancy confound.

**Mean-AA substitution specifically: no published instance found** (searched
flat profiles/PSSMs, masked PLM tokens, uninformative MSAs, composition-only
baselines). Record as UNVERIFIED-because-absent, i.e. a small genuine gap.

### What IS defensible: the IG baseline is invalid, by the authors' own rule

Section 20 reported the completeness failure as a quadrature problem. **That
reading is wrong and is corrected here.**

Sundararajan, Taly & Yan (arXiv:1703.01365), the IG paper this project is built
on, state two things we failed:

1. **"We recommend that developers check that the baseline has a near-zero
   score."** Our `mean_aa` baseline scores 0.6644 against the input's 0.5797 --
   not near-zero relative to the input, and the span is only **0.085**.
2. Completeness should hold **within ~5%** at 20-300 steps.

5% of 0.085 is ~0.004. Our absolute error was **2.1-3.1, i.e. 300-700x the
authors' own tolerance**, with the sign flipping across m = 16/32/64. Lundstrom,
Huang & Razaviyayn (ICML 2022, PMLR v162) establish that completeness error of
an m-step Riemann sum IS the quadrature error of the path integral -- so a
non-monotone, sign-flipping sequence in m indicates a **high-variation
integrand, not a coarse grid**. No step count fixes this.

Nearest named mode: Miglani et al., "Investigating Saturation Effects in
Integrated Gradients" (arXiv:2010.12697) -- gradients in saturated regions
"contribute disproportionately". Ours is the degenerate limit where the
ENDPOINTS barely differ. **No established term of art for that, and no
published numeric minimum-span rule** -- the 5% completeness check is already
span-normalised and suffices.

**Expected gradients would not rescue it.** Erion et al., Nat Mach Intell 2021:
EG replaces one baseline with an expectation over data-drawn baselines, so by
linearity completeness becomes `sum = f(x) - E[f(x')]`. If `f(x) ~= f(x')` for
all plausible baselines the EG span is also ~0. Averaging cuts variance across
baseline choice; it cannot manufacture a span the output surface lacks. (This
step is derived from the completeness axiom, not quoted from the paper.)

### The clean experiment, if this arm continues

1. **Sequence shuffle on fixed coordinates** -- the classic threading-decoy
   construction: permute the sequence, keep the native structure. It breaks
   redundancy in the right direction, because the coordinates then DISAGREE with
   the token, so any model reading the token must move. Doubles as the
   composition-matched IG baseline (the protein analogue of genomics'
   dinucleotide-shuffled reference).
2. **Ligand-channel ablation as positive control.** Without it, "the prediction
   barely moved" cannot be distinguished from a dead path into the head.
3. **More than one target.** ABL1/imatinib alone cannot carry a claim about a
   model.
4. **Do NOT use backbone-only stripping** -- the recovery numbers above make it
   useless, worst of all in a buried pocket.
5. Labelled sensitivity set available: Platinum / mCSM-lig, ~1000 experimental
   ddG-of-binding mutations, and mCSM-lig predicts from wild-type structure plus
   mutation identity -- our exact setup, with ground truth.

### Standing correction to how section 20 should be read

Of section 20, what survives is: the gradient reaches `res_type` through the
full trunk at a 7.12x trunk/shallow ratio; peak VRAM 16.80 GiB; real crop
coverage 200/264 residues and 15/20 measured positions; and **the baseline is
invalid for this model and target**. What does NOT survive is any claim about
the model's sequence reliance.

## 22. The baseline was NOT the problem. IG does not converge on this head. 2026-10-04

Section 21 diagnosed the completeness failure as an invalid baseline and said
one run would settle whether that was the whole story. It was not.

### The span probe -- and the clean threading control

`--span-probe 5`, forward passes only, seconds of GPU. f(x) = 0.579723.

| baseline | f(baseline) | span |
|---|---|---|
| shuffled, 5 seeds | 0.642, 0.647, 0.744, 0.597, 0.657 | mean **0.078**, sd **0.048**, min 0.017, max 0.164 |
| `mean_aa` | 0.664421 | 0.085 |
| zeros | 0.817172 | 0.237 |
| **`unk`** | **1.008340** | **0.429** |

**The shuffled result is the retraction-proof version of section 20's withdrawn
claim.** Composition preserved exactly, a valid one-hot at every position, and
with coordinates pinned at 2HYY the geometry now DISAGREES with the token -- so
the redundancy that killed the section-20 reading is broken in the right
direction. Scrambling all 264 residues moves predicted affinity by a factor of
**1.04 to 1.46**.

And the **sd is 61% of the mean span**: variation BETWEEN scrambles is
comparable to the effect OF scrambling. There is no consistent sequence signal,
it is noise-level. That is stronger than a small mean alone.

`unk` -- boltz's own unknown-residue token, verified `tokens[22] == "UNK"` in
`const.py` with `unk_token["PROTEIN"]` and `prot_letter_to_token["X"]` agreeing
-- gives the largest span at 0.429, 5.1x `mean_aa`. It is also the baseline the
attribution literature prefers, being genuine in-distribution missingness
rather than nonsense (zeros) or a wrong answer (shuffled).

### The decisive comparison: absolute error is BASELINE-INDEPENDENT

| baseline | span | m | ig_sum | abs error | rel error |
|---|---|---|---|---|---|
| `mean_aa` | 0.085 | 16 | -2.234 | 2.150 | 25.4 |
| `mean_aa` | 0.085 | 32 | +2.994 | 3.079 | 36.4 |
| `mean_aa` | 0.085 | 64 | +1.184 | 1.269 | 15.0 |
| `unk` | 0.429 | 16 | +1.789 | 2.217 | 5.17 |
| `unk` | 0.429 | 32 | +1.348 | 1.777 | 4.15 |
| `unk` | 0.429 | 64 | **-1.504** | 1.075 | 2.51 |

**A 5.1x larger span left the ABSOLUTE error unchanged** -- mean 2.17 under
`mean_aa` against 1.69 under `unk`. Relative error improved only because we
divided by a bigger number. **And the sign of `ig_sum` still flips across m
under BOTH baselines** (+1.348 -> -1.504 between m=32 and m=64 for `unk`).

That is the answer. The path integral carries intrinsic variation of magnitude
~1-3 along ANY path through this function, which swamps endpoint differences of
0.085 and 0.429 alike. Per Lundstrom et al. (ICML 2022, PMLR v162), completeness
error IS the quadrature error, so a non-monotone sign-flipping sequence in m
means a **high-variation integrand**. Enlarging the span does not reduce the
variation. **No baseline choice and no step count fixes this.**

### Faithfulness is not merely wrong, it is unstable

Same model, same 18 mutations, same B. Only the baseline changed:

| baseline | Spearman(A, B) |
|---|---|
| `mean_aa` | **-0.4964** (p = 0.036) |
| `unk` | **+0.2487** (p = 0.320) |

Note `grad_res_type` in the artifact is the PATH-AVERAGED gradient at the top
rung, not the gradient at x, so it legitimately depends on the baseline -- but
that is the point: **the attribution flips sign under an arbitrary baseline
choice and neither value is robust.** Section 20's -0.4964, which I called
"suggestive of systematic anti-correlation", is hereby downgraded to
baseline-dependent noise. The IG product at m=64 gives +0.1208 (p=0.633),
likewise null.

### VERDICT on IG for Boltz-2's affinity head

Measured, not inferred:

1. **Completeness does not converge**, under a span-0.085 baseline or a
   span-0.429 one, with absolute error ~1-3 in both and the sign flipping in m.
2. **Faithfulness against the model's own behaviour is unstable**, swinging
   -0.50 to +0.25 on baseline choice alone, neither significant.
3. **The one fixable hypothesis was tested and rejected.** The baseline was a
   real defect -- section 21's diagnosis stands as far as it went -- but it was
   not the cause.

So IG is not salvageable on this target by baseline selection. Any future claim
that it works here must first exhibit a convergent completeness ladder.

What remains untested: whether the non-convergence is specific to the affinity
head or general to this trunk at these token counts. The confidence path
reached rel err 0.0442 at m=32 and 0.0036 at m=5 on the pair layer, so the
trunk is not inherently pathological -- which points at the affinity module's
distance-bucketised, `.detach()`-heavy construction as the likely locus. That is
a hypothesis, not a result.

### Housekeeping

Peak VRAM 23.87 GiB on the `unk` run, up from 16.80 -- so a 24 GB L4 is
marginal, not safe. The printed "Crop coverage: 29/264" units bug from section
20 is STILL present and still needs the call site fixed to pass residue numbers.

## 23. Paper build-out: no GPU this session, four arms made runnable. 2026-10-12

Goal restated by the user: write the paper, on IG applied to Boltz-2 and/or
Protenix. Cost is explicitly not a constraint from here on.
**Nothing ran on a GPU this session.** Everything below is code, controls and
target selection, all uncommitted, all pending the runs listed at the end.

### The GPU never started, and why that was lucky

The faithfulness run (below) was dispatched first and never reached the VM:
`~/.ssh/google_compute_engine` carries a passphrase and the agent is empty, so
SSH failed with `Permission denied (publickey)`. That is the exact trap
`HANDOFF.md` warns about. The VM ran ~12 min and was stopped, TERMINATED, no job
started. **Unblock with `ssh-add --apple-use-keychain
~/.ssh/google_compute_engine` in Terminal.app.**

Lucky, because `04_scan.py` would have produced an uninterpretable scan: see
below. Buying ~2,300 forward passes against zeroed geometry would have been the
most expensive mistake available this session.

### `04_scan.py` was not fit to be the faithfulness ground truth

Four defects, all fixed:

1. **Geometry was empty.** `x_pred = ref_feats["coords"]`, which section 14
   measured as all zeros for sequence-only input, while stage 03 uses
   `predict_structure_coords`. Scan and attribution would have been computed on
   *different geometry*, which voids the comparison outright.
2. **The MSA was not pinned** -- every mutant re-queried the server, against
   section 16's standing instruction.
3. **No wild-type row**, so no `delta` could be formed at all.
4. **No resume on the SKEMPI path**, so any interruption appended duplicates.

### The design decision: `resswap` is the primary ground truth

Attribution differentiates f w.r.t. the one-hot `res_type` with every other
feature held at wild type. So the faithful B is the *same function* evaluated at
a swapped one-hot -- not a re-featurised mutant. `--mode resswap` (now default)
changes `res_type[pos]` and nothing else; `--mode refeaturise` keeps the real
mutant as a secondary measurement of model behaviour.

**The `token_to_rep_atom` hazard is confirmed, not hypothetical.** From the
installed boltz source, `featurizer.py:726` builds
`token_to_rep_atom` from per-token `disto_idx`, and `loss/confidence.py:188-189`
uses it to select one representative atom per token out of the atom table. A
Gly->Trp mutation (4 vs 14 heavy atoms) shifts that table, so reusing wild-type
`x_pred` under `refeaturise` reads the wrong atoms for every token after the
mutation. `refeaturise` now guards on both the shape AND the content of
`token_to_rep_atom` and skips rather than lying.

**Section 18 is unaffected and stands.** `20_affinity_scan.py` already
implemented `resswap` (it featurises once and edits only `res_type`), so the 18
ABL1 deltas ARE the function the gradient differentiates. Had it re-featurised,
section 18's faithfulness numbers would have compared two different functions.

### The affinity-head diagnostic, and the control it was missing

Section 22 left one hypothesis standing: non-convergence is specific to the
affinity module, not the trunk, because the confidence path converges (rel err
0.0442 at m=32). `scripts/23_affinity_path_profile.py` tests it directly --
f(alpha) on a 257-point grid plus the autograd directional derivative
<grad f, x - b> at 65 alphas, **both heads, same path, same complex**.

Review caught three defects that would have produced a confident, wrong figure:

1. **The profile and the gradient ran different code.** f(alpha) went through
   `model.msa_module`; the derivative went through `_msa_forward_checkpointed`.
   FD-vs-autograd disagreement would then have been a code-path artifact. Both
   now call the same `affinity_forward`/`confidence_forward`, and
   `max_profile_vs_grad_diff` is reported as the check.
2. **No noise floor.** TV and sign-flip counts are the whole claim, and nothing
   separated them from fp32 GEMM nondeterminism. Now: 5 alphas x 3 repeats,
   `noise_sd` reported, plus `--deterministic`.
3. **The sign-change threshold was 1e-30**, i.e. every float wobble counted.
   Now `3*noise_sd*sqrt(2)`, and TV is reported both raw and above-noise.

Also fixed: the **"Crop coverage: 29/264" units bug** from sections 20-22 is
closed (`18_gate_affinity.py`, sites now passed as residue NUMBERS), real
coverage 200/264, with a regression test and the misleading
`check_crop_coverage` docstring corrected.

### Protenix is the second model -- for the confidence path only

Verified against `bytedance/Protenix` @ 85767b8 (= v2.0.0), written up in
`notes/draft_protenix_feasibility.md`. `restype` is a float one-hot concatenated
into `s_inputs` with no embedding lookup -- the same seam as Boltz. Four
blockers, all patched at runtime without a fork: the recycling loop under
`set_grad_enabled(self.training)`; an in-place MSA slice update (the root cause
of the unsolved CopySlices error in IG_Agros); `ConfidenceHead.stop_gradient`;
and a `@torch.no_grad()` summary-score function.

- **No affinity head exists**, across every variant checked. The ligand-affinity
  arm stays Boltz-only.
- **PLAN.md's hard-max question is answered: yes.** pTM and ipTM take a max over
  tokens in *both* Protenix and Boltz-2. `complex_pde` avoids it and is the
  score that transfers.
- Trunk is smaller (48 pairformer blocks vs 64). Estimated ~10-21 GiB at 4
  recycles, ~450 tokens -- unmeasured.
- Separate image required: protenix conflicts with `boltz==2.2.1` and pins
  cuequivariance 0.8.0, the A100 crash from IG_Agros. v1.0.0 weights only
  (Apache-2.0); v2 is proprietary and now refused in code.
- **Chai-1 rejected**: compiled trunk blocks per-block checkpointing, and
  full-trunk backward already OOM'd on a 141 GB H200.

### Targets: one complex cannot carry a claim about a model

`notes/draft_ligand_targets.md`. Recommended ABL1 + EGFR + Mpro, with PimA
fourth.

| target | ligand (CCD) | PDB | singles / positions | assay |
|---|---|---|---|---|
| EGFR KD | AQ4 erlotinib, IRE gefitinib | 1M17 | 27 / 20, x18 TKIs | Ba/F3 viability, 3 reps |
| SARS-CoV-2 Mpro | 4WI nirmatrelvir | 8DZ2 (dimer) | 102 / 12 | enzymatic Ki |
| PimA | GDD, GDP | 3OKP / 3OKC | 16 x2 / 13 | Kd |

- **EGFR is not Hauser.** Median replicate sd **0.080 log10** against a 5-log10
  range (S/N 12.2), vs Hauser's ~0.6. Technical repeatability only; the
  IC50-vs-Kd systematic gap stays unmeasured. Sign convention confirmed from the
  data (positive = resistance, as SKEMPI), anchored on T790M **reversing sign**
  between erlotinib (+1.56) and osimertinib (-0.76) -- a stronger qualitative
  anchor than T315I/ponatinib because it is a sign change, not a magnitude.
- **Mpro must be run as the dimer.** On 8DZ2, E166 sits 2.6 A and H172 3.2 A
  from the partner protomer; monomeric 7VH8 deletes a contact the measured Ki
  depends on. Its value is *within-position depth* (~8.5 substitutions/position,
  Q192 has 11) -- the within-position ranking section 17 lost on Hauser.
- **Leakage:** Mpro has **0** Platinum rows and EGFR **1**, so mCSM-lig is a
  usable external baseline on both. All WT structures predate the 2023-06-01
  cutoff; flagged, unfixable.
- **Numbering trap:** 1M17 is mature-protein numbered, offset exactly **-24**
  (unique in [-40,+40], 20/20 agreement). Same class as the 2HYY residue-275 bug.

**Correction to our own record.** ABL1 was written up as having ~2 positions
beyond 6 A of imatinib. Recomputed from 2HYY it is **6 within / 13 beyond,
3.0-27.2 A**. So ABL1 always had the near/far spread, EGFR's spread is not a
unique advantage, and section 17's invalidity flag 1 (near/far confounded with
modellable/unmodellable) **applies to ABL1 too and always did**. PimA is the
only candidate whose far sites are not allosteric.

**Rejected so it is not re-searched:** HIV-1 protease (max 8 positions for a
fixed ligand, best literature panel 7 singles / 5 positions, homodimer); every
kinase but ABL1 for *biochemical* singles (Davis 2011 tops out at 5). **MET DMS
(Estevam 2025, ~5764 variants) is flagged, not buried** -- it is the only thing
that would solve the degrees-of-freedom problem, excluded solely because the
readout is growth enrichment rather than affinity. If that objection is ever
resolved, MET beats everything here.

### Registered before the runs

- Completeness will converge on `complex_pde` and fail on `affinity_pred_value`
  on the SAME complex and path (that is the point of the two-head profile).
- `total_variation_above_noise` will be large for the affinity head and ~0 for
  the confidence head.
- Faithfulness on 3S9D `resswap` will beat the section-22 affinity numbers.
- Protenix `complex_pde` completeness will land near Boltz's (~0.04 at m=32).

Written down so a failure counts.

### The queue, in order

1. **3S9D chain A, confidence path.** Stage 03 IG m=64 then m=128
   (`mean_aa`, `--x-pred predicted`), then `04_scan.py --mode resswap`, then
   `--mode refeaturise`, then `24_faithfulness.py`. ~2,330 forward passes.
   This is the paper's missing headline: the shortcut-vs-slow-way number.
2. **Affinity path profile**, ABL1/imatinib, three baselines + `--deterministic`.
   ~804 forward-equivalents each, ~40-67 min.
3. **Protenix**: build the image, memory ladder, parity test, then `complex_pde`
   plain_grad and IG (m=32, 64) on 1JTG and 3S9D.
4. **EGFR and PimA** affinity runs once ground truth lands; Mpro needs
   `oc3c00538_si_001.pdf` fetched by hand (publisher blocks scripts) and dimer
   support in the affinity CLIs.

### Housekeeping

1,227 tests passing, ruff unchanged at its pre-existing count. Nothing
committed. `04_scan.py`, `18_gate_affinity.py`, `boltz_score.py` and the
Makefile are modified; `23_affinity_path_profile.py`, `24_faithfulness.py`,
`03_attribute_protenix.py`, `08_memscale_protenix.py`, `protenix_score.py`,
`Dockerfile.protenix` and four test files are new.

### ADDENDUM to 23 -- ground truth landed for all four targets, 2026-10-13

All parsed, all guarded, none committed. Mpro's supplement was fetched by hand
(publisher blocks scripts); its sha256 is recorded and verified before parsing.

| target | rows | positions | guard |
|---|---|---|---|
| Mpro / nirmatrelvir (8DZ2) | 102 subs (+10 del, 2 double) | 12 | WT Ki 1.88 nM; 102/102 residues match 8DZ2 |
| EGFR (1M17) | 486 = 27 muts x 18 TKIs | 20 | 27/27 match at offset -24; T790M sign reversal confirmed |
| PimA-GDD (3OKP) | 16 | 13 | 16/16 |
| PimA-GDP (3OKC) | 16 | 13 | 16/16 |

EGFR median replicate sd **0.0797 log units** (ddof=1), confirming the 0.080
figure. **The log base is NOT stated anywhere in the supplement** -- recorded as
`log_base: "unknown"`, not assumed. Magnitudes are consistent with log10
(T790M/erlotinib +1.56 implies ~36-fold) but that is inference. It would
corrupt a calibration slope, not a rank correlation.

Only 54 of 102 Mpro substitutions carry a Ki; 44 are IC50-only. Kept in separate
columns, never merged.

### The distance distributions are the real finding here

Minimum heavy-atom distance from each measured position to the ligand:

| target | within 6 A | beyond 6 A | range |
|---|---|---|---|
| **Mpro** | **12 / 12** (11 within 4 A) | **0** | 1.9 - 9.1 A |
| **EGFR** | 3 / 20 | **17 / 20** | 3.3 - 21.3 A |
| PimA (both ligands) | 5 / 13 | 8 / 13 | 2.7 - 16.9 A |
| ABL1 (recomputed, 2HYY) | 6 / 19 | 13 / 19 | 3.0 - 27.2 A |

**Mpro cannot test the geometry confound at all.** Every measured position is a
drug-site residue, so the distance partial -- the control that saved this
project from Phase 5's false positive -- is unidentifiable on Mpro by
construction. Mpro's value is within-position depth (18 substitutions at Q192,
17 at H172, 16 at S144) and nothing else. Say so before quoting any Mpro
enrichment.

**EGFR is the mirror image and carries the opposite risk.** Only 3 of 20
positions are within 6 A; 17 are distal, out to 21.3 A. That is the regime
section 17 flagged as invalidity 1: distal resistance mutations act through
conformational equilibrium, which a co-folding model scoring a PINNED structure
cannot represent even in principle. So EGFR's 17 far positions are plausibly
*unmodellable* rather than merely hard, and a null on them would not be evidence
about attribution.

**Consequence for the paper: the four targets are complementary, not
redundant, and must not be pooled into one correlation.** Mpro tests
within-position ranking with geometry held constant; PimA and ABL1 carry the
near/far contrast with non-allosteric far sites; EGFR tests the distal regime
and is the one most likely to fail for reasons that have nothing to do with IG.
Report per-target, and state which question each target can answer.

Tests 1,330 passing, ruff down from 14 to 1.

## 25. Replication, partial: 1JTG confirms 3S9D. 2JEL/3HFM stranded. 2026-10-14

The 3S9D faithfulness number (section 24) was one complex, and section 24's own
first caveat said it must not be quoted until the other three had run. They
ran. **Two of the four are read out below; the other two are measured but their
artifacts are stuck on a VM that cannot restart.**

### Faithfulness, resswap ground truth

| complex | n | method | Spearman | CI | null p95 | partial vs distance | scale ratio |
|---|---|---|---|---|---|---|---|
| 3S9D A | 950 | IG m=64 | **0.5937** | [0.540, 0.640] | 0.053 | 0.5722 | 0.748 |
| 3S9D A | 950 | plain_grad | 0.5885 | [0.535, 0.638] | 0.053 | 0.5653 | 0.438 |
| **1JTG B** | **532** | **IG m=32** | **0.6326** | **[0.569, 0.691]** | **0.072** | **0.6317** | **0.721** |
| 1JTG B | 532 | plain_grad | 0.5449 | [0.474, 0.615] | 0.070 | 0.5502 | 0.543 |

**It replicates.** Two complexes, 0.59 and 0.63, both ~9-11x their own
permutation null. On 1JTG the distance partial costs **0.0009** (0.6326 ->
0.6317); on 3S9D it cost 0.0215. Phase 5's hot-spot result died under exactly
this control. This one does not move.

IG beats plain_grad on scale ratio on both complexes (0.72/0.75 vs 0.54/0.44)
while ranking within 0.09, reproducing section 24's reading: path integration
buys calibrated magnitude, not ranking.

### Refeaturise stays null, and n collapses

| complex | n scored | IG | plain_grad |
|---|---|---|---|
| 3S9D | 103 / 950 | -0.0123 (p=0.90) | -0.0783 (p=0.44) |
| 1JTG | **32 / 532** | 0.2053, CI [-0.177, 0.539] | -0.2496, CI [-0.540, 0.077] |

1JTG keeps only **32 of 532** mutants past the `token_to_rep_atom` guard -- a
94% skip against 3S9D's 89%. Both 1JTG CIs straddle zero and both scale ratios
are ~0.007, i.e. A is ~140x too small. Nothing here contradicts section 24;
nothing here is powered enough to confirm it either.

### A prediction registered before the run, and falsified

`notes/_sec25_predictions_registered.md` called it: "the 3S9D 0.5937 will NOT
cleanly replicate at that magnitude... expect mean ~0.4-0.5 with sd >0.15, and
expect at least one complex to land materially lower." 1JTG came in **higher**
(0.6326), with a tighter distance partial. **The prediction is wrong so far**,
and is recorded as wrong. Two complexes cannot settle it; 2JEL and 3HFM can.

### Status of 2JEL and 3HFM -- measured, not yet readable

The overnight chain completed both headline arms (`3HFM headline rc=0` at
09:29 UTC, `2JEL headline rc=0` at 13:27 UTC) and 1JTG's m=64 attribution
(13:27-15:17), then self-halted via `sudo shutdown -h now` as designed.

**The artifacts were never synced.** The VM is TERMINATED and
`us-central1-a` is returning `STOCKOUT` for a2-ultragpu-1g, so it cannot be
restarted to collect them. The boot disk is zonal and retained; the data is
safe but unreachable until capacity frees. A patient retrier is polling.

**Do not quote a four-complex verdict until those two land.** Two of four,
both positive, is not the test section 14 demands -- that section's lesson was
that 1 of 8 clearing its null is what noise looks like at eight looks.

### Operational

- The 12h cap raise worked and the VM's self-shutdown worked. The local
  auto-stop watcher **reported `STOPPED_ON_COMPLETE` while its stop command had
  errored** -- it did not stop anything, and the cap is still 43200s. Do not
  trust a watcher's own success line; verify state. Restoring 2h is pending the
  same stockout.
- 2JEL/3HFM m=32 artifacts from section 14 were revalidated before reuse by
  reproducing f(x) at HEAD: 2JEL 1.4196462631225586, 3HFM 0.9989417791366577,
  both identical to the 0bf19ef values. Completeness also reproduced exactly
  (2.3726%, 11.1378%). Their ladders will be **m=32 only** -- m=64 was
  unaffordable -- and both FAIL the 5% gate at that rung.
- A driver bug was caught that would have voided two complexes silently: `.DONE`
  markers were stamped on stages that were *skipped*, so 2JEL and 3HFM were
  logged `COMPLEX ... COMPLETE` with no scan data. Found by listing artifacts,
  not by reading the log.
- **VRAM: section 24's "peak 20.2 GiB, the 80 GB card is unnecessary" is
  withdrawn.** Measured peaks: 1JTG ig_m32 42.6 GiB, 2JEL plain_grad 61.2 GiB,
  3HFM plain_grad 70.1 GiB of 80. These are `nvidia-smi memory.used`
  (driver-reserved, incl. allocator pool), so an upper bound -- read as "80 GB
  suffices, 24 GB does not", not as live tensor footprint. 20.2 GiB was the
  smallest complex and never generalised.

### COMPLETION of 25 -- all four complexes, 2026-10-14

The stockout cleared, the artifacts synced, the VM is TERMINATED and the 2h
`maxRunDuration` cap is restored (verified 7200).

| complex | n | rung | Spearman | CI | null p95 | partial vs distance | drop | scale |
|---|---|---|---|---|---|---|---|---|
| 1JTG B | 532 | IG m=64 | **0.6678** | [0.608, 0.722] | 0.071 | 0.6653 | 0.4% | 0.759 |
| 3S9D A | 950 | IG m=64 | 0.5937 | [0.540, 0.640] | 0.053 | 0.5722 | 3.6% | 0.748 |
| 3HFM Y | 285 | IG m=32 | 0.4054 | [0.289, 0.509] | 0.098 | 0.4043 | 0.3% | 0.438 |
| 2JEL P | 608 | IG m=32 | 0.4042 | [0.331, 0.471] | 0.068 | **0.2629** | **35.0%** | 0.783 |

**Mean 0.5178, sd 0.1339, range 0.404-0.668. 4 of 4 clear their own
permutation null**, every CI excludes zero, and the weakest complex (0.4042)
still sits ~6x its null. Against section 14, where 1 of 8 cleared its null and
the sign flipped across complexes, this is a different regime.

### The registered prediction was broadly RIGHT, and an earlier call of mine was wrong

`_sec25_predictions_registered.md` said: "expect mean ~0.4-0.5 with sd >0.15,
and expect at least one complex to land materially lower", and that all four
would still clear their null. Outcome: mean **0.518** (just above the band), sd
**0.134** (just below), spread 0.404-0.668 (materially lower: yes, two of
them), 4/4 clear null (correct).

**Section 25's first pass declared this prediction "FALSIFIED" on the strength
of 1JTG alone.** That was premature and is retracted here. 1JTG is the
best-performing complex of the four; calling the prediction on the single most
favourable draw is the same error the project made with 3S9D, one iteration
later. The prediction was substantially correct.

### 2JEL is where the geometry control finally bites

The distance-to-partner partial costs 0.4% on 1JTG, 0.3% on 3HFM, 3.6% on
3S9D -- and **35.0% on 2JEL** (0.4042 -> 0.2629). So "the attribution is not
geometry" is true on three complexes and only two-thirds true on the fourth.
2JEL was also the complex where section 14's corrected pair map was actively
anti-predictive (AUROC 0.365). Report the partial per complex; never quote the
mean partial alone.

### IG beats plain_grad, and the margin is largest where the signal is weakest

| complex | IG | plain_grad | delta |
|---|---|---|---|
| 3HFM | 0.4054 | 0.1819 | **+0.2235** |
| 1JTG | 0.6678 | 0.5449 | +0.1229 |
| 3S9D | 0.5937 | 0.5885 | +0.0052 |
| 2JEL | 0.4042 | 0.3920 | +0.0122 |

On 3HFM plain_grad is barely above its null (0.1819 vs p95 0.098) while IG is
4x it. Section 24 concluded from 3S9D alone that "path integration buys
calibrated magnitude, not ranking" -- that holds where both methods are strong,
but on the hardest complex IG buys ranking too. Amend the claim.

### Completeness ladders remain incomplete

1JTG has the full ladder (m=32 -> m=64) and m=64 improved faithfulness
0.6326 -> 0.6678. **2JEL and 3HFM are m=32 only** -- m=64 at ~234 and ~259 min
did not fit the window. 3HFM FAILS the 5% gate at that rung (11.14%); 2JEL
PASSES (2.37%). So the one complex whose attribution is least arithmetically
accountable, 3HFM, is also the one where IG most outperforms plain_grad. That
is not explained, and it is the obvious thing for a referee to ask about.

### What this does and does not establish

Establishes: on Boltz-2's `complex_pde`, gradient attribution predicts the
model's own one-hot substitution response at 0.40-0.67 across four complexes,
4/4 above null, and on three of four this is not reducible to distance.

Does NOT establish: anything about real mutations. The refeaturise arm stays
null everywhere and its n collapses (103/950 on 3S9D, 32/532 on 1JTG). And the
36x median-|B| gap is still confounded -- refeaturise reuses wild-type
coordinates with rebuilt mutant atom features, so part of that response may be
the model reacting to a sequence-structure inconsistency rather than to the
mutation. **The coordinate-consistent control (re-predict geometry for the
mutant) has not been run and is the next thing that matters.**

## 26. All three comparisons, at last: the shortcut beats the model it reads. 2026-10-14

Section 1 named three comparisons. Two were open until today.

| comparison | what it asks | result | clears null |
|---|---|---|---|
| **shortcut vs slow way** (A-B) | is attribution faithful to the model? | **+0.518** | **4 of 4** |
| **model vs lab** (B-C) | is the model right about biology? | +0.235, p=0.19 | 2 of 4 |
| **shortcut vs lab** (A-C) | what does a practitioner get? | +0.265, p=0.17 | 2 of 4 |

**The shortcut reproduces the model far better than the model reproduces
nature.** That ordering is the project's answer, and it is the one the original
plan said would be most informative: attribution is not the weak link.

### Model vs lab, resswap scan against SKEMPI ddG

| complex | n | Spearman | CI | perm p | partial (distance) |
|---|---|---|---|---|---|
| 3S9D A | 54 | **+0.4205** | [+0.183, +0.626] | **0.0015** | +0.4199 |
| 1JTG B | 30 | **+0.4376** | [+0.011, +0.708] | **0.0151** | **+0.5368** |
| 3HFM Y | 42 | +0.2530 | [-0.039, +0.503] | 0.109 | +0.1921 |
| 2JEL P | 44 | -0.1693 | [-0.453, +0.139] | 0.268 | -0.2090 |

Mean +0.2354, sd 0.2824, t=+1.67, **p=0.19**. Dropping SKEMPI's exact-zero rows
(1-sig-fig Kd) lifts 3S9D to **+0.5521** (n=46, p<0.0001).

### Shortcut vs lab

| complex | n | IG | perm p | plain_grad | perm p |
|---|---|---|---|---|---|
| 3S9D A | 54 | **+0.4678** | **0.0008** | +0.1196 | 0.386 |
| 1JTG B | 30 | **+0.5591** | **0.0013** | +0.2618 | 0.165 |
| 2JEL P | 44 | +0.0385 | 0.802 | +0.1482 | 0.334 |
| 3HFM Y | 42 | -0.0049 | 0.975 | -0.0442 | 0.781 |

IG mean +0.2651 (2/4 clear null); **plain_grad mean +0.1214, 0 of 4**. Against
experiment IG beats plain_grad by +0.35 and +0.30 on the two complexes that
work -- a far larger margin than the +0.09 it wins by against the model.
Section 24's "path integration buys magnitude, not ranking" is now wrong twice
over: it buys ranking on the hardest complex (section 25) and it buys most of
the usable signal against experiment.

The quadrature rung is irrelevant at this n: 1JTG m=32 gives +0.5613 against
m=64's +0.5591.

### Sign orientation, verified rather than assumed

`complex_pde` is a predicted distance error (lower = better, confirmed against
`12_compare.LOWER_IS_BETTER`); SKEMPI ddG is positive = destabilising. Same
axis, no negation, agreement is a POSITIVE correlation. Checked before any
number was quoted: 1JTG's five largest measured ddGs -- W150A, K74A, H41A,
F36A, Y53A, the canonical BLIP hot spots -- **all carry positive deltas**, and
3HFM's top two (K96, K97, the HyHEL-10 epitope) likewise. Hot-spot vs neutral
separation: 3S9D p=0.0004, 1JTG p=0.039. 2JEL's anchor inverts but at p=0.145,
i.e. 2JEL has no signal rather than a flipped sign.

### Why 2JEL fails, and it is not the attribution

**A-vs-B on 2JEL collapses to 0.1574 on the 44 SKEMPI rows against 0.4042 on
the full 608-row scan.** 2JEL's measured panel is only **14% alanine**, against
87% and 80% on 3S9D and 1JTG. So faithfulness fails on precisely the rows
experiment happened to choose. That is the cleanest available explanation of
2JEL's null and it implicates the panel, not the method.

### The bound, and why it is not violated

A-vs-C exceeds B-vs-C numerically on three complexes (+0.047, +0.122, +0.208).
A data-processing bound would only bind if A were a deterministic function of
B; A agrees with B at 0.16-0.76 on these rows, so the two make different errors
against C, and +/-0.2 is ordinary sampling noise at n=30-54. Every CI overlaps.
**No claim here depends on the shortcut beating the model.**

### Caveats that limit what this can support

1. **n = 30-54.** Four complexes, t=+1.67, p=0.19. Section 14's "1 of 8 is what
   noise looks like" is NOT overturned -- 2 of 4 is better, and still not
   significant at four.
2. **The positive complexes are 80-87% alanine.** They mostly answer "which
   positions matter", not the per-amino-acid claim the project set out to make,
   and that confound cannot be controlled with this data.
3. **Refeaturise is untestable against experiment**: the `token_to_rep_atom`
   survivors contain **0** SKEMPI mutations on 3S9D and 1JTG (18 and 3 on the
   others).
4. **3HFM cannot test the geometry confound** -- all 42 mutations lie within
   2.3-3.9 A. A second independent reason to distrust it, beyond section 14's
   15-position flag.
5. **A trap, recorded and NOT quoted as a result**: 2JEL refeaturise plain_grad
   vs ddG reads +0.7234 (n=18, p=0.0010) and is **geometry** -- distance
   correlates -0.938 with ddG on those rows, the partial falls to +0.607 with a
   CI straddling zero, and A-vs-B there is -0.08.
6. The lab column is not the bottleneck: split-half reliability 0.960 / 0.881 /
   0.844 (2JEL not estimable). Where the model works it reaches about half its
   ceiling.

### Supersedes section 14 on two complexes, confirms its verdict

Like-for-like against |ddG|: sec14 simplex +0.4683 / +0.1714 / -0.3396 /
+0.1573 (mean +0.1143, p=0.54) vs new |A_IG| +0.4774 / +0.5195 / -0.0315 /
+0.0422 (mean +0.2519, p=0.18). Complex ordering is preserved (rho +0.80);
1JTG triples; 2JEL's anti-predictive -0.34 relaxes to a flat -0.03. The
four-complex t-test still misses 0.05, so section 14's headline verdict stands.

---

## 27. IG's top-k blind spot: cysteine, where cysteine matters. 2026-11-04

Every A-vs-B number in this project so far is a **whole-ranking** Spearman. Nobody
had asked the question a practitioner actually asks: *does IG find the same top
hits as the scan?* It does not, and the way it fails is specific enough to be
worth chasing.

Computed on CPU from the existing npz + scan CSVs, same `_extract_attribution`
pairing `24_faithfulness.py` uses, so these reproduce the section 26 rhos exactly.

### Top-k overlap, IG vs scan

| complex | n | rho | top-10 | top-25 | top-50 | chance@50 | scan #1 at IG rank |
|---|---|---|---|---|---|---|---|
| 1JTG | 532 | +0.668 | 1/10 | 11/25 | 27/50 | ~4.7 | 159 |
| 3S9D | 950 | +0.594 | 0/10 | 4/25 | 18/50 | ~2.6 | **897 / 950** |
| 3HFM | 285 | +0.405 | 2/10 | 8/25 | 18/50 | ~8.8 | 8 |
| 2JEL | 608 | +0.404 | 5/10 | 8/25 | 19/50 | ~4.1 | 4 |

Top-50 is 4-7x chance everywhere -- the correlation is real and it is carried by
the bulk. The top-10 is mostly missed. **The rho and the top-k tell different
stories, and only the rho has been reported.**

Aggregating to positions (each position's worst mutation) recovers a lot:
6/10, 2/10, 8/10, 8/10 against chance 3.6 / 2.0 / 6.7 / 3.1. On 1JTG the scan's
biggest hits are K73 and R159 and IG ranks those *positions* highly while
misplacing the specific substitutions. So IG is better at "which position" than
at "which amino acid" -- the same split section 26's alanine confound warned about,
arrived at from the other direction.

### The failure is conditional on cysteine

My first read of this was wrong. Pooled over all four complexes, cysteine is
**not** the biggest effect -- proline is (median |delta| 0.0222 vs C 0.0147, C
third behind P and W) -- and IG handles proline **better** than average
(median IG rank percentile 0.40 against a 0.50 null). So "IG misses big effects"
is not the story.

The story is that cysteine's effect size varies by complex, and IG's grip on
cysteine fails exactly where that effect is large:

| complex | C n | med abs delta, C | med abs delta, rest | rho within C | rho within rest |
|---|---|---|---|---|---|
| 3S9D | 50 | 0.0257 (2.2x) | 0.0116 | **+0.042** | +0.632 |
| 3HFM | 15 | 0.1152 (4.5x) | 0.0257 | **-0.179** | +0.478 |
| 1JTG | 28 | 0.0153 (1.1x) | 0.0139 | +0.724 | +0.665 |
| 2JEL | 32 | 0.0038 (0.7x) | 0.0051 | +0.274 | +0.411 |

Where X->C is consequential, IG has **no signal on it at all** (3S9D) or is mildly
**anti**-predictive (3HFM), while retaining +0.63 / +0.48 on everything else in
the same run. Where X->C is ordinary, IG tracks it as well as anything else.

Scan top-10 composition makes the same point: 3S9D is `CCCCCCDFPY` and 3HFM is
`CCCCCCCCQV`, both with median IG percentile 0.50 -- pure chance. 1JTG
(`AADDEEEIPV`) and 2JEL (`FFGPPPPPTW`) have essentially no cysteine and median
percentiles of 0.10 and 0.03 -- IG nails them.

### Why this is a plausible mechanism, and how to kill it

This is the **resswap** scan (section 2.3): only `res_type[pos]` changes, with
coordinates, atom features and MSA held at wild type. A cysteine token therefore
lands in a structure with no disulfide partner and no thiol geometry -- a
sequence-structure inconsistency. A large `complex_pde` response to that is
believable. IG integrates from a `mean_aa` baseline along a straight line in
one-hot space, and a response that only switches on near the cysteine vertex is
exactly what a path-integrated gradient from a smeared baseline would miss.

**Untested.** Candidate checks, cheapest first:
1. Does a WT->mutant path IG (rather than mean_aa->WT) recover the cysteine
   hits? If yes, the baseline is the problem, not IG.
2. Do the 3S9D / 3HFM cysteine positions sit near existing cysteines or
   disulfides? That would support the mechanism directly.
3. Does the same blind spot appear on the refeaturise scan, where the mutant
   actually gets cysteine atom features?

### Caveats

1. **3HFM's C cell is n=15**, and 3HFM already fails the 5% completeness gate at
   m=32 (11.14%, section 25). Its -0.179 is the weakest number in the table.
   3S9D (n=50, m=64, completeness 0.24%) is the load-bearing one.
2. 1JTG and 3S9D are m=64; 3HFM and 2JEL are m=32 only. Step count is confounded
   with complex throughout this table.
3. Top-k overlap is a harsh metric by construction -- with n~500-950, small rank
   perturbations move members in and out of a 10-element set. The position-level
   aggregation is the fairer read.
4. This says nothing about the lab. It is an A-vs-B statement: where IG does and
   does not reproduce **the model's own** scan.

### Why it matters beyond a caveat

For the SAE phase (`docs/SAE_RESEARCH.md`), this is a ready-made test with a known
answer IG gets wrong. If a concept-level explanation can account for the 3S9D
cysteine hits that residue-level IG ranks 897th of 950, that is a sharper result
than matching IG's +0.518 -- it is a case where the concept view sees something
the gradient view cannot.

## 28. Protenix attribution was never reproducible, and `mean_aa` is why IG loses to a single gradient. 2026-11-05/06

First end-to-end IG + scan on Protenix (1JTG, chain B, `complex_pde`, single-seq,
n_cycle=4, 532 mutations). The headline is a **retraction and a fix**, not a result.

### The bug: `ref_pos` was regenerated, unseeded, every process

We froze the *predicted* structure last session (`x_pred` on-disk cache) and
assumed featurisation was deterministic. It is not. Hashing every feature tensor
across two processes with identical inputs, only two differ:

| feature | shape | process A | process B |
|---|---|---|---|
| `ref_pos` | (3257, 3) | `3757a503a3fffb9c` | `db2cca53d9fb5425` |
| `d_lm` | (102, 32, 128, 3) | `025531fa60714275` | `e6fc6f5a4b07411d` |

`ref_pos` is the RDKit reference conformer; `d_lm` is downstream of it. Every
other feature is byte-identical.

### Why that was fatal, and invisible

Within one process everything is bit-exact (4 repeated forwards and 4 repeated
backwards, identical to all printed digits). Across processes the forward moves
only 0.1-3%, but the backward -- 64 Pairformer layers x 4 recycles -- amplifies it:

| quantity | across four separate processes |
|---|---|
| `g.d` at alpha=0.5 | **-1.4882, +12.57155, +3.04140, -0.26294** (two sign flips) |
| `||g||` at alpha=1.0 | **75.74024 vs 46.89751** (38%) |

A 2% forward drift became a >100% gradient swing *at the wild-type itself*. IG was
integrating a gradient field that changed between evaluations.

### Fix

`build_features(use_feature_cache=True, feature_cache_dir=...)` -- atomic on-disk
cache of the finalised feature dict (written after `generate_relp` and
`update_input_feature_dict`), keyed on chains + msa_paths + msa_depth + model_name,
shared by stages 03/04/05 via `data/raw/protenix_features/`. `ref_pos_fingerprint`
now goes into every `.prov.json`, so a matched run is **verifiable, not assumed**.

Verified: two fresh processes gave identical `ref_pos` (`5e0a8e15f1ffeb9b`), f,
`g.d` (-7.59176) and `||g||` (517.61847). In the production run, IG, plain_grad
and the scan all report `f(x) = 9.472452` and `ref_pos c987246d8eb33d2b`.

A first fix attempt was incomplete: stages 03 and 04 passed *different* cache
dirs (`protenix_attr/` vs `protenix_scan/`), so each built its own cache --
reproducible per script, still mismatched between them, which is the pair that
matters. Caught before any run.

### RETRACTION: the pre-fix A-B numbers were artifacts

| | pre-fix (mismatched `ref_pos`) | post-fix (verified matched) |
|---|---|---|
| IG m=32, `mean_aa` | +0.2062 [0.1182, 0.2964], p=0.0002 | **+0.0495 [-0.0293, 0.1325], p=0.26** |
| plain_grad | +0.1176 [0.0337, 0.2020], p=0.0081 | **+0.2358 [0.1586, 0.3145], p<1e-4** |
| completeness (IG) | 351.11% | 401.19% |
| scale ratio median\|A\|/median\|B\| | 0.1853 | 0.2565 (IG), 1.7511 (plain_grad) |

Both pre-fix numbers were wrong **and the ordering was wrong**. I had written that
"IG beats plain_grad, doing real work over its m=1 baseline." With matched features
the ranking reverses: **plain_grad beats IG by ~5x**, and IG's CI spans zero.

I also predicted the fix would *raise* +0.206 ("a floor"). It fell to 0.050. The
reasoning -- noise attenuates correlation, so removing noise raises it -- only holds
for random noise around a fixed signal. Changing `ref_pos` produces a *different*
gradient vector, not a noisier one (`||g||` 75.7 / 46.9 / 517.6 across conditions).

### The determinism bug was NOT causing the completeness failure

401.19% with perfectly matched, reproducible features. That isolates the cause to
the path. Measured along `mean_aa` -> WT (||d|| = 20.1408, 8540 nonzero entries):

| alpha | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| f | 12.176787 | 11.420660 | 9.695648 | **9.955785** | 9.549372 | **9.682346** | 9.460993 |

Non-monotone, reversing direction at least three times. A finite-difference h-sweep
never converges (at alpha=0.5: 4.42, -5.40, -3.73, -5.78, -8.82, -15.61 as h shrinks),
so the path is rough at every scale tested.

**Mechanism.** 31 of 32 Gauss-Legendre nodes sit at alpha<1, inside a region where
every residue is a 20-way uniform smear -- a protein the model never saw. Those
gradients are noise and they swamp the one meaningful region near alpha=1. Hence
both the 401% completeness error and the collapse to +0.050. The gradient is
informative; the `mean_aa` path destroys it.

### This confirms section 27's registered prediction

Section 27 listed, as the cheapest falsification of the cysteine blind spot:
"Does a WT->mutant path IG (rather than mean_aa->WT) recover the cysteine hits?
If yes, the baseline is the problem, not IG." The baseline is the problem,
established here on independent grounds.

`scripts/05_ig_mutation_paths.py` (new) implements per-mutation paths:
`wt_to_mut` and `null_to_mut` (`--null-mode {unk,zeros}`), changing only the
mutated row. In `wt_to_mut`, `f(input) - f(baseline)` IS the scan's delta, so
**completeness becomes checkable against ground truth per mutation** -- impossible
under `mean_aa`. Not yet run.

### Caveats

1. **One complex, one chain, one score.** 1JTG/B, `complex_pde`. No replication.
2. **Single-sequence.** No MSAs exist for Protenix (lost with the us-central1-a
   disk). Protenix deltas are ~10x Boltz's (median |delta| ~0.2 vs 0.014), almost
   certainly because without an MSA the model leans entirely on sequence. The
   Boltz +0.518 comparison is therefore confounded by MSA setting as well as model.
3. **`mean_aa` m=32 only.** Boltz used m=64 on 1JTG. Step count is not controlled.
4. **Boltz has never had this stability check run.** Its +0.518, and the SAE phase
   resting on it, used the same attribution machinery on an equally deep model.
   Boltz has a separate featurisation path so it may well be clean -- but that is
   exactly what we assumed about Protenix.
