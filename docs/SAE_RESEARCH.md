# Sparse autoencoders + attribution — research notes and directions

Written 2026-10-24. This is a literature survey and a set of proposals, not a
result. Nothing here has been run. Results and retractions still go in
`notes/logging.md`; this document is the map for a possible next phase.

Confidence markers on literature claims:

- **[FT]** read in full text (some long pages truncated)
- **[A]** abstract, model card, or search summary only
- **[M]** from memory, not re-checked

Several bioRxiv/PNAS full texts could not be fetched (rate limits / 403), so a
lot of the bio literature below is **[A]**. Re-read anything before citing it or
copying a recipe from it.

---

## 1. Why this, and why alongside IG

The goal is to understand **what Boltz-2 computes**, not to score it against
the lab. Agreement or disagreement with experiment is one lens among several,
not the target.

IG answers one question well: *which input residues* moved a score. On
`complex_pde` its answer matches Boltz-2's own brute-force scan, with A–B
+0.518 across 4 complexes (`docs/OVERVIEW.md` §1). What IG can't say is *why*:
what the model represents internally, and what route a residue's influence takes
to reach the output.

- **SAE alone:** *which concepts* the model represents, at each layer, residue
  and residue pair. This is unsupervised, so it can surface concepts nobody
  thought to look for.
- **SAE + attribution:** *which concepts* moved a given prediction, and at which
  residues. The result is a residue × concept explanation instead of a residue
  heat map.
- **Across settings:** how those concepts shift between layers, recycles, MSA
  vs single-sequence, the confidence vs affinity heads, and Boltz-2 vs other
  models.

So the proposal is to add SAEs to IG, not to replace it. IG and the
brute-force scan remain the yardstick for whether a feature-level explanation is
faithful to the model.

**Note on targets.** The SKEMPI arm attributes the **confidence** head's
`complex_pde` (predicted distance error), not the affinity head
(`docs/OVERVIEW.md` §2, `scripts/03_attribute.py`). The affinity head is used
only on the protein–ligand arm (ABL1/imatinib, Mpro, EGFR, PimA), and IG does
not converge there. Several proposals below say "affinity"; for the SKEMPI
complexes read that as `complex_pde`.

---

## 2. What an SAE is

Boltz-2 holds a 384-d single representation per token and a 128-d pair
representation per token pair. Individual dimensions aren't concepts. Each
concept is spread across many dimensions, and each dimension takes part in many
concepts ("superposition").

A sparse autoencoder is a separate small network trained on cached activations:

```
x (384-d)  →  encoder  →  f (e.g. 16,384 latents, only ~k nonzero)  →  decoder  →  x̂ ≈ x
```

- **Reconstruction:** x̂ must match x.
- **Sparsity:** only k latents may be active per token, enforced by TopK,
  BatchTopK or JumpReLU, or by an L1 penalty in older SAEs.
- **Result:** latents tend to be monosemantic ("contacts ligand",
  "disulfide cysteine", "hydrophobic pocket lining").
- **Unsupervised:** nobody chooses the concepts. You inspect top-activating
  residues afterwards and name them. That makes it a discovery tool. Supervised
  probes still win for concepts you already have labels for (§5).

Boltz-2 is never modified. The SAE reads one layer.

Variants that matter here:

- **Transcoder:** maps the input of an MLP to its output, instead of x → x.
  Gives cleaner circuits through MLPs.
- **Skip-transcoder:** a transcoder with an affine skip connection.
- **Cross-layer transcoder (CLT):** a transcoder whose features write to every
  later layer.
- **Crosscoder:** one dictionary shared across two models or two settings. Used
  for diffing, e.g. MSA vs no-MSA.
- **Matryoshka SAE:** nested dictionaries. More stable, and leads SAEBench on
  disentanglement.

---

## 3. Prior work in biology

### 3.1 SAEs on protein language models

Mature, but not directly useful to us: Boltz-2 has no PLM in its input path, so
these explain ESM, not Boltz.

| work | model / layer | SAE | notes |
|---|---|---|---|
| InterPLM (Simon & Zou) [A] — arXiv 2412.12101 | ESM-2-8M, all layers | L1, 10,240 latents | 433 Swiss-Prot concepts; SAE features cover far more concepts than neurons |
| InterProt (Adams … AlQuraishi, ICML 2025) [A] | ESM-2-650M L24 | TopK, 4,096 | weights at `liambai/InterProt-ESM2-SAEs` |
| Reticular (Parsan et al.) [A] — arXiv 2503.08764 | ESM-2-3B | Matryoshka | steering changed ESMFold solvent accessibility |
| ProtSAE [A] — arXiv 2509.05309 | ESM-2-15B L35 | TopK, partly GO-supervised | a linear probe beat unsupervised SAEs (Fmax 0.537 vs 0.444) |
| ESM C SAEs (Biohub, Jan 2026) [A] | ESMC-6B L60 and others | TopK k=64, 16k/65k | best off-the-shelf PLM SAEs |
| ProtoMech [A] — arXiv 2602.12026 | ESM-2-8M/35M | cross-layer transcoders + attribution graphs | circuits use ~0.8% of latents, recover 79% of family-classification F1 |
| **Mohanty, Phutela & Green (bioRxiv Sept 2026)** [A] — doi 10.64898/2026.08.28.747907 | ESM-2-650M | SAE + **IG** + activation patching | **the most overlapping paper**: 67 DNAJA1 mutations. Circuits chosen by indirect effect beat ones chosen by activation magnitude. They picked mutations where the model **agrees** with DMS — the opposite of our angle |

### 3.2 SAEs on structure / co-folding models

| work | model | what | attribution? |
|---|---|---|---|
| **PairSAE** (Migliorini et al., Flagship) [A] — arXiv 2606.27440 | Boltz-2 trunk L33/L64, 3rd recycle | SVD-compressed pair + single → 512-d per token; Matryoshka BatchTopK, 16k; 40k PLINDER systems, no MSA | no. A LASSO readout predicts **Boltz-2's own** affinity (Spearman 0.81 on PoseBusters). No code seen |
| Boltz-1 SAEs [A] — arXiv 2608.11475 | Boltz-1 trunk + diffusion | TopK k=256, 2,048; 84k proteins; weights `evolve-away/Boltz1-SAEs-*` | steering only. Mean-subtract first (~98% of energy in 10 dims). A decodable β-strand direction had **no causal effect**. Probes beat single features |
| bish-bash-fold (Kim & Vonessen, ICML 2026 GenBio WS) [A] | AF3 + Boltz-2 **diffusion** modules | features sorted by MSA/sequence ablation into MSA-activated / MSA-silenced / sequence-driven / model-prior | no. Probes on features predict wet-lab expression/solubility better than pLDDT. **Closest to our MSA ideas**, but diffusion-module and designability, not ddG/affinity |
| Boltz-2 transcoders [A] — github `rishimj/boltz2-plt` | Boltz-2 trunk | 384→2,048, k=16 | unreviewed repo |
| FoldSAE [A] — arXiv 2511.22519 | RFdiffusion | TopK, 16× | mostly helix/strand features |
| AlphaInterp [A] | AF3 | **linear probes**, not SAEs | — |

**PairSAE's pair handling**, since it's the recipe we'd copy:

1. Take an N-mode SVD of Z (N×N×128).
2. Concatenate the top-64 singular vectors per mode per token, zero-padded when
   N<64. Concatenate that with the 384-d single representation, giving 512-d
   per token.
3. Use one SAE. Its decoder reconstructs the single representation and also
   pair entries as z_ij ≈ row(h_i) + col(h_j) + b, using one random j per i.

Details beyond the abstract are unverified.

### 3.3 Context

- Co-folding models give confident poses for physically impossible binding
  sites (Masters, Mahmoud & Lill, Nat Commun 2025) [A].
- Boltz-2 affinity has leakage/memorisation critiques: DAVIS leakage audit,
  arXiv 2512.00708 [A].
- Crosscoder diffing can come back empty. ProtBert→IgBert found no latents
  distinguishing paired from unpaired fine-tunes (NeurIPS 2026) [A].
- A student project on SAE steering of designed-binder embeddings found the
  features **legible but not causally steerable** (github
  `vigkart/reg3a-binder-design`).

---

## 4. Combining attribution with sparse features, outside biology

None of the methods found run IG over SAE latents for a **scalar regression**
output. The analogues are LM logit differences and image-classifier logits.

### 4.1 Sparse Feature Circuits (Marks et al. 2024) [FT] — arXiv 2403.19647

- SAEs sit on the residual stream, attention outputs and MLP outputs.
- **Decomposition:** each activation is x = Σ fᵢ vᵢ + b + ε. The error ε is
  kept as an explicit **error node**, not dropped.
- **Indirect effects:** estimated by attribution patching
  (∇ₐm · (a_patch − a_clean)) or by **IG with N=10** steps. IG fixes the
  underestimates in early layers.
- **Faithfulness:** (m(C) − m(∅)) / (m(M) − m(∅)), with everything outside the
  circuit mean-ablated.
- **Completeness:** the faithfulness of the complement; 0 is ideal.
- **Result:** ~100 features (Pythia-70M) or ~500 (Gemma-2-2B) reproduce the
  behaviour; neurons need about 15–100× more nodes.
- **Transfer:** this is the template. Attribute the scalar to SAE latents on the
  trunk, keep ε, and report faithfulness, completeness and the share of
  attribution that lands on ε, alongside the Spearman against the brute-force
  scan.

### 4.2 Attribution patching / AtP* (Kramár et al. 2024) [FT] — arXiv 2403.00745

- First-order patching: all nodes for two forwards plus one backward.
- **Failure modes, both producing false negatives:**
  - **saturation:** a flat softmax region, so the gradient is about 0;
  - **cancellation:** direct and indirect effects cancel.
- **Fixes:** recompute the QK softmax exactly; GradDrop. Use AtP as a filter and
  confirm the top nodes with real activation patching.
- EAP-IG (Hanna et al. 2024) [A]: adding IG fixes the zero-gradient problem.
  Judge circuits by faithfulness, not by overlap with a reference circuit.
- **Transfer:** triangle attention is a saturating softmax, which is a plausible
  but **unverified** cause of the affinity-head IG non-convergence. In any case,
  confirm top-k feature attributions by actually ablating them.
- Caveat [A, forum only]: TopK/JumpReLU SAEs are discontinuous, which biases
  discrete IG sums near their thresholds.

### 4.3 Transcoders and circuit tracing

- **Transcoders** (Dunefsky et al., NeurIPS 2024) [A]: attribution through MLPs
  factors into input-dependent × input-invariant terms.
- **Skip-transcoders** (EleutherAI 2025) [A]: more interpretable than SAEs
  trained on the same data.
- **Anthropic attribution graphs** (Mar 2025) [FT, partial]:
  - Cross-layer transcoders inside a "local replacement model" that freezes
    attention and LayerNorm and adds a per-(position, layer) error term.
  - Edges are activation × virtual weight, giving a linear graph.
  - Completeness ~0.80. Attention-formation (QK) circuits are unexplained.
  - Open-source `circuit-tracer` (Gemma-2-2B, Llama-3.2-1B).
- **CLT faithfulness critique** (Lange et al., LessWrong Feb 2026) [FT]: CLTs
  can swap multi-step computation for shallow single-hop paths.
- **Transfer:**
  - Per-layer (skip-)transcoders on the Pairformer transition MLPs are feasible.
  - Prefer them over CLTs.
  - Freezing attention hides exactly the triangle-attention computation that may
    matter most.

### 4.4 Crosscoders / model diffing

- **Lindsey et al. 2024** [FT, intro]: one dictionary across models; features
  exclusive to one side flag what changed.
- **Minder et al. 2025** [A]: L1 crosscoders invent fake "exclusive" latents;
  **BatchTopK** fixes this.
- **OpenAI 2025** [A]: diffing before vs after finetuning found a "toxic persona"
  latent that steers emergent misalignment.
- **Transfer:** compare MSA vs single-sequence, or Boltz-1 vs Boltz-2. The cheap
  version is one shared SAE over both runs, then diff activations.

### 4.5 Concept attribution in vision

- **TCAV** (2018) [M]: directional derivative along a supervised concept
  vector. The ancestor of all of this.
- **CRAFT** (Fel et al., CVPR 2023) [A]: NMF concepts + Sobol importance.
- **"Holistic" concept attribution** (Fel et al., NeurIPS 2023) [FT]:
  - Concept extraction is dictionary learning; concept importance is attribution
    over the concept coefficients.
  - Defines C-Deletion, C-Insertion and C-μFidelity.
  - **Theorem:** with concepts at the penultimate layer feeding a linear head,
    Gradient×Input, IG, Occlusion and RISE are optimal on those metrics.
- **Archetypal SAE** (Fel et al., ICML 2025) [A]: constraining atoms to the
  convex hull of the data stabilises dictionaries across retrains.
- **Dreyer et al. 2025** on CLIP SAE latents [FT]:
  - Act×Grad over latents, with ε kept, beat 10-step IG on deletion/insertion
    curves.
  - Found 57 spurious failure modes out of 220.
  - A "red hue" latent drove melanoma predictions; editing it dropped accuracy
    from 80.5% to 59.9%.
- **Transfer:** put a dictionary at the representation feeding the head. If the
  head is near-linear in it, Gradient×Input ≈ IG and convergence stops mattering
  at that layer. **Unchecked:** whether Boltz-2's affinity/confidence heads are
  anywhere near linear in their input; the affinity module has its own
  Pairformer, so probably not.

### 4.6 Finding model bugs: SHIFT and auditing

- **SHIFT** (in SFC) [FT]:
  - Build a feature circuit, have a human mark task-irrelevant features, and
    zero-ablate them. Optionally retrain the head.
  - Bias in Bios, Gemma-2-2B: worst-group accuracy rose from 18.2% to 92.9%;
    gender decodability fell to chance.
- **Anthropic auditing game** (2025) [A]: 3 of 4 blind teams found an implanted
  hidden objective, with SAE features as a main route.
- **Transfer:** a general way to ask which concepts a prediction leans on, and
  whether removing them changes it. That includes, but isn't limited to, cases
  where the model and the lab differ.

### 4.7 Parameter-space alternatives

APD (Apollo 2025) and SPD (Goodfire 2025) [A] decompose weights rather than
activations. They have only been shown on toy models so far; watch them, don't
build on them.

---

## 5. Pitfalls we must design around

- **Probes beat SAEs on known concepts.** Sources: Kantamneni et al., ICML 2025
  [A]; AxBench [A]; the DeepMind note (Mar 2025, secondary sources); ProtSAE;
  Boltz-1 SAEs. Every feature claim must beat logistic or linear regression on
  the **raw 384-d representation**, plus difference-in-means, on the same labels.
- **Decodable ≠ causal.** See the Boltz-1 β-strand direction and the reg3a
  steering result. Confirm by ablation or patching, not by correlation.
- **Proxy metrics mislead.** SAEBench: reconstruction, L0 and spliced-in loss
  don't track usefulness, and the benchmarks themselves are noisy
  (arXiv 2605.18229) [A]. For us, "spliced-in" means Δ`complex_pde` and Δaffinity
  when x is replaced by x̂.
- **Random-SAE baselines** can match trained SAEs on some tasks
  (arXiv 2602.14111) [A]. Include one.
- **Unstable dictionaries.** Features differ across seeds and split or absorb
  one another. Train ≥3 seeds; consider Matryoshka or Archetypal SAEs.
- **Dead latents and outliers.** Boltz activations concentrate their energy in a
  few dimensions, so mean-subtract and use an auxiliary dead-latent loss.
- **Small N.** Four SKEMPI complexes are an **evaluation** set, not a training
  set. The SAE must be trained on thousands of external complexes.
- **MSA confound.** PairSAE ran without MSA and reports that this changes
  affinity substantially; our scans use MSAs. Match the setting between the
  activations the SAE is trained on and those it is evaluated on.

---

## 6. Applications

These are grouped by the question each one asks about Boltz-2, not ranked
against one goal. Lab data comes in where it's a useful check, not as the aim.
Novelty is relative to everything in §3–4; feasibility assumes 1×A100 and 2 h
runs.

### 6.1 What the model represents

| application | prior work | novelty | feasibility |
|---|---|---|---|
| **Concept atlas of the trunk.** Name single and pair features: contacts, H-bonds, salt bridges, pocket lining, interface vs core, epitope/paratope | PairSAE (protein–ligand) [A], Boltz-1 SAEs [A] | medium. **Protein–protein interfaces are new**: PairSAE was trained on protein–ligand systems | high |
| **Pair-specific concepts.** What lives in z that isn't in s: geometry, coevolution, distance bins | PairSAE row/col decoder only | medium-high | medium |
| **Ligand-side chemistry features**, including on ABL1, Mpro, EGFR and PimA | PairSAE couldn't name its ligand features | medium-high | medium |
| **Layer and recycle dynamics.** When does a binding concept appear, sharpen or vanish across depth and recycles? | Boltz-1 SAEs across layers and recycles [A] | medium; new for Boltz-2 | medium (one SAE per depth, or a crosscoder) |

### 6.2 How a prediction is made (SAE + attribution)

| application | prior work | novelty | feasibility |
|---|---|---|---|
| **Residue × concept explanations.** Attribute `complex_pde` and affinity to latents (SFC-style, with ε kept) and map back to residues | SFC on LMs [FT], CLIP latents [FT]; nothing on co-folding | **high** | medium-high |
| **Mutation mechanism.** For each scanned mutant, which latents switch on or off, and which of those carry the Δscore? Turns the scan into "this mutation breaks salt-bridge feature #812" | Mohanty et al. on ESM-2 [A] | high for a co-folding model | high: the scans already exist |
| **Faithfulness of concept explanations.** Map latent attribution to residues and rerun A–B against the brute-force scan | none | medium | medium. Needed anyway, to know whether the explanations above can be trusted |
| **Confidence vs affinity heads.** Do they read the same latents? Is the affinity head a re-weighting of confidence concepts or something separate? | none found | **high** | medium (affinity IG convergence is still open) |
| **Circuits through the Pairformer** (transcoders, attribution graphs) | ProtoMech (ESM-2) [A], circuit-tracer (LMs) | high | low at first: a quarter of engineering |

### 6.3 What information the model relies on

| application | prior work | novelty | feasibility |
|---|---|---|---|
| **MSA vs single-sequence diffing.** Which concepts come from coevolution and which from the model's prior? | bish-bash-fold, diffusion module only [A] | medium-high on the trunk | high: about 2× forward passes and no training |
| **Model diffing.** Boltz-1 vs Boltz-2, and Boltz-2 vs Protenix (already built here) | LM crosscoders; IgBert diff came back empty | medium-high | low-medium |
| **Memorisation / train-set similarity.** Are there latents that track "seen this complex" rather than physics? | identity-based leakage audits only | medium-high | medium (needs similarity labels) |
| **Covariates.** How much do MSA depth, interface size and complex size organise the feature space? | bish-bash-fold [A] | medium | high |

### 6.4 Using the features

| application | prior work | novelty | feasibility |
|---|---|---|---|
| **Steering.** Amplify or suppress interface/binding latents and watch structure, confidence and affinity respond | FoldSAE, ESMFold steering, low-N design; none on a co-folding head | medium-high | medium. Known steering nulls (Boltz-1 β-strand, reg3a) make this exploratory |
| **Features as a representation.** Probes from latents to lab ddG, DMS, expression, failure flags; compare with raw-representation probes | PairSAE (own output), bish-bash-fold (wet-lab phenotypes) [A] | medium | high once activations are cached |
| **Comparison with the lab as one lens.** Where Boltz-2 and experiment differ, which concepts are involved; SHIFT-style ablation to see what each concept contributes | SHIFT on LMs [FT] | high | medium-high |
| **Hotspot / epitope discovery** | InterPLM, antibody-LM SAEs | low (crowded) | high |

### Suggested shape

1. A **concept atlas**: what Boltz-2 represents at protein–protein and
   protein–ligand interfaces.
2. **Concept-level explanations** of its predictions, checked for faithfulness
   against the brute-force scan. This is where the existing IG work carries
   over directly.
3. **Diffing** (MSA vs none, confidence vs affinity head, Boltz vs Protenix) to
   show what each concept depends on.

Steering, probes and lab comparison are then applications built on 1–3. The
asset nobody else has is a scan-validated attribution pipeline with provenance.
That's what turns "these features look interpretable" into "these features
explain the model's outputs".

---

## 7. First experiments

All assume the existing pipeline in `src/igv/boltz_score.py` and
`src/igv/attrib.py`.

### E0. Activation cache and pilot SAE (prerequisite)

1. **Hook points:** trunk single representation s (L×384), plus PairSAE-style
   SVD summaries of z (L×128) at two Pairformer depths. Recycle and MSA setting
   are to be matched to how the scans were run.
2. **Data:** a pilot of ~500 PLINDER/PDB complexes, then 10–40k. Add our four
   SKEMPI complexes (WT + every scanned mutant) and the protein–ligand series as
   held-out evaluation.
3. **Storage:** ~1 KB/token in fp16 for the 512-d summary, so ~16 GB per layer
   for 40k complexes of ~400 tokens. Fits the 500 GB disk (~58 GB used). Raw z
   does not fit (~400 GB per 10k complexes); never store it.
4. **Runs:** sharded and resumable, with each shard well under the 7200 s
   auto-stop; `.prov.json` sidecars as everywhere else.
5. **Compute:** extraction dominates, at an estimated 100–300 A100-hours for
   40k complexes [estimate]. SAE training is a few GPU-hours.
6. **SAE:**
   - library: `dictionary_learning` (Marks) or `Overcomplete`, over the cached
     shards;
   - architecture: Matryoshka BatchTopK, mean-subtracted inputs, an auxiliary
     dead-latent loss;
   - pilot size: 4,096 latents, k=64;
   - ≥3 seeds.
7. **Gate before anything downstream:** splice x̂ in place of x and report
   Δ`complex_pde` / Δaffinity on the held-out complexes, plus the dead-latent
   fraction and cross-seed feature overlap. If splicing moves the score more
   than the typical mutation effect, the SAE is unusable for attribution. That gate
   blocks every experiment below; it is not a deliverable on its own.

### E1. Concept atlas pass

- **Method:** for each latent, record top-activating residues and residue pairs
  across the held-out set.
- **Auto-labelling:**
  - UniProt annotations;
  - PLIP contacts (protein–ligand);
  - interface/core and SASA;
  - secondary structure;
  - contact distance from z.
- **Report:**
  - concept coverage, the fraction of known concepts with F1 ≥ 0.5;
  - a list of high-attribution latents that match **no** known label. Those
    are the interesting ones.

### E2. Concept-level attribution and its faithfulness

- **Method:** IG over SAE latents with ε held fixed, using `attrib.py`
  quadrature on the latent path. Also run Act×Grad.
- **Targets:** `complex_pde` on 1JTG, 3S9D, 3HFM and 2JEL; affinity on the
  protein–ligand series.
- **Map back to residues:** Σ over latents of attribution × decoder-norm share
  per token. Rerun A–B against the existing scans and compare with residue IG
  at +0.518.
- **Report:**
  - the fraction of attribution on ε;
  - faithfulness and completeness in the SFC sense;
  - ablation of the top-k latents vs their predicted effect.

### E3. Mutation mechanisms

- **Method:** encode WT and each scanned mutant with the SAE. For each mutation,
  list the latents whose activation changes most and their attribution-weighted
  contribution to Δscore.
- **Questions:**
  - Do many mutations route through a few shared latents (interface "hubs")?
  - Do substitutions at the same position reuse latents? Mohanty et al. report
    40–75% reuse on ESM-2.

### E4. MSA vs single-sequence diffing

- **Method:** run the same complexes in both modes, encode both with one SAE,
  and catalogue MSA-dependent vs model-prior latents.
- **Then:** check which group carries the attribution in E2/E3.
- **Cost:** about 2× forward passes and no training.

### E5. Confidence vs affinity head

- **Method:** on the protein–ligand complexes, attribute both heads to the same
  latents.
- **Report:** the overlap of top-k latents, and whether affinity attribution
  concentrates on ligand-contact latents that confidence ignores.
- **Diagnostic:** run the `23_affinity_path_profile.py` diagnostic on the latent
  path, to see whether IG converges better there than on input embeddings.

### E6. Optional lens: comparison with the lab

- **Method:** with E2/E3 in hand, group mutations by which concepts dominate
  their predicted effect, and ask how each group lines up with SKEMPI/DMS.
- **SHIFT-style ablation:** shows what each concept contributes. Removing a
  concept is a probe of the model, not a fix to it.
- **Baselines:** any probe from latents to lab data must beat a probe on the
  raw 384-d representation and a random SAE (§5).

---

## 8. Open questions

- Which layer? The SAE should sit on the input to the heads (better for
  attribution, §4.5) or mid-trunk (PairSAE's L33/L64, richer concepts). Probably
  both: one pilot at each.
- Does the affinity-head IG non-convergence survive when attributing to latents
  just before the affinity module? Run the `23_affinity_path_profile.py`
  diagnostic on the latent path.
- Is PairSAE's code obtainable? Writing to the authors would save the
  reimplementation and give a cross-check.
- Does the protein–protein confidence score `complex_pde` even carry binding
  concepts, or only structure-quality concepts? E1 tells us.

## 9. Tooling

- **`dictionary_learning`** (github `saprmarks/dictionary_learning`): Standard,
  Gated, TopK, BatchTopK, JumpReLU and MatryoshkaBatchTopK; accepts any PyTorch
  dataloader. **Recommended.**
- **`Overcomplete`** (Kempner): trains on a plain (N, d) tensor; TopK, BatchTopK,
  JumpReLU and Archetypal. The most model-agnostic option.
- **SAELens v6**: analysis, SAEBench and Neuronpedia; its training loop is
  LLM-oriented.
- **`sparsify`** (EleutherAI): computes activations on the fly from a
  HuggingFace model, so it's awkward for Boltz.
- **`circuit-tracer`**: LM-only, but its graph code is a reference if we ever do
  #11.
