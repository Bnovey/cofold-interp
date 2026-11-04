"""Protenix confidence-score surface for gradient attribution.

Wraps the Protenix confidence head so that a single scalar score is
differentiable w.r.t. the one-hot residue-type input ``restype``.
The trunk is run with gradient through all recycling cycles (patch P1),
an out-of-place MSA update (patch P2), stop_gradient disabled (P3),
and live-logit score computation (P4).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Protenix residue type indices: ALA=0 .. VAL=19, UNK=20.
_NUM_RESTYPE_CLASSES = 32
_CANONICAL_AA_RANGE = range(0, 20)
_UNK_INDEX = 20

# Bin parameters from configs_base.py (pinned for reproducibility).
_PDE_BIN_PARAMS = {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
_PAE_BIN_PARAMS = {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
_PLDDT_BIN_PARAMS = {"min_bin": 0.0, "max_bin": 1.0, "no_bins": 50}
_DISTOGRAM_BIN_PARAMS = {"min_bin": 2.3125, "max_bin": 21.6875, "no_bins": 64}

_DEFAULT_MODEL_NAME = "protenix_base_default_v1.0.0"
_PROTENIX_REPO_ROOT = "/tmp/protenix_src"

_ALLOWED_WEIGHT_NAMES = frozenset({
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix_base_default_v0.5.0",
    "protenix_base_constraint_v0.5.0",
})
_REJECTED_WEIGHT_PREFIX = "protenix-v2"


# ---------------------------------------------------------------------------
# Score registry (mirrors boltz_score.SCORES)
# ---------------------------------------------------------------------------

PROTENIX_SCORES: dict[str, str] = {
    "complex_pde": "dense",
    "complex_plddt": "dense",
    "pae_interface_mean": "dense",
    "iptm": "hard_max",
    "ptm": "hard_max",
}

PROTENIX_IPTM_SCORES = {"iptm", "ptm"}


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------


def _ensure_protenix_repo_on_path() -> bool:
    import sys

    for candidate in [_PROTENIX_REPO_ROOT, "/opt/protenix", "/root/protenix"]:
        if os.path.isdir(os.path.join(candidate, "configs")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return True
    return False


def _bootstrap_layernorm() -> None:
    os.environ.setdefault("LAYERNORM_TYPE", "torch_layernorm")


# ---------------------------------------------------------------------------
# Config / weights / model loading
# ---------------------------------------------------------------------------


def _get_cache_dir() -> Path:
    env = os.environ.get("PROTENIX_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "protenix"


def _download_weights(weights_dir: Path, model_name: str) -> None:
    import urllib.request

    _BASE = "https://protenix.tos-cn-beijing.volces.com"
    ckpt_dir = weights_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}.pt"
    if not ckpt_path.exists():
        url = f"{_BASE}/checkpoint/{model_name}.pt"
        log.info("Downloading Protenix checkpoint %s ...", model_name)
        urllib.request.urlretrieve(url, ckpt_path)

    common_dir = weights_dir / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    for fname, rel_url in [
        ("components.cif", "common/components.cif"),
        ("components.cif.rdkit_mol.pkl", "common/components.cif.rdkit_mol.pkl"),
        ("clusters-by-entity-40.txt", "common/clusters-by-entity-40.txt"),
        ("obsolete_release_date.csv", "common/obsolete_release_date.csv"),
    ]:
        fpath = common_dir / fname
        if not fpath.exists():
            url = f"{_BASE}/{rel_url}"
            log.info("Downloading %s ...", fname)
            urllib.request.urlretrieve(url, fpath)


def _sync_ccd_paths(weights_dir: Path) -> None:
    common = weights_dir / "common"
    paths = {
        "ccd_components_file": str(common / "components.cif"),
        "ccd_components_rdkit_mol_file": str(common / "components.cif.rdkit_mol.pkl"),
        "pdb_cluster_file": str(common / "clusters-by-entity-40.txt"),
        "obsolete_release_data_csv": str(common / "obsolete_release_date.csv"),
    }
    os.environ["PROTENIX_ROOT_DIR"] = str(weights_dir.resolve())

    if _ensure_protenix_repo_on_path():
        import configs.configs_data as cd

        cd.PROTENIX_ROOT_DIR = str(weights_dir.resolve())
        cd.data_configs.update(paths)

    import protenix.data.core.ccd as ccd_mod

    ccd_mod.COMPONENTS_FILE = paths["ccd_components_file"]
    ccd_mod.RKDIT_MOL_PKL = Path(paths["ccd_components_rdkit_mol_file"])
    for fn in (
        ccd_mod.biotite_load_ccd_cif,
        ccd_mod.get_component_atom_array,
        ccd_mod.get_one_letter_code,
        ccd_mod.get_mol_type,
        ccd_mod.get_all_ccd_code,
        ccd_mod.get_component_rdkit_mol,
    ):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()


def _build_configs(
    model_name: str,
    weights_dir: Path,
    input_json: Path,
    output_dir: Path,
    n_cycle: int = 4,
    n_sample: int = 1,
) -> Any:
    os.environ["PROTENIX_ROOT_DIR"] = str(weights_dir.resolve())
    from protenix.config.config import parse_configs

    if not _ensure_protenix_repo_on_path():
        raise RuntimeError(
            f"Protenix repo not found. Clone to {_PROTENIX_REPO_ROOT}."
        )

    from configs.configs_base import configs as _configs_base
    from configs.configs_data import data_configs as _data_configs
    from configs.configs_inference import inference_configs as _inference_configs
    from configs.configs_model_type import model_configs as _model_configs

    full_configs: dict = {**_configs_base, **{"data": dict(_data_configs)}, **_inference_configs}

    if model_name in _model_configs:
        def _deep_update(d: dict, u: dict) -> None:
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    _deep_update(d[k], v)
                else:
                    d[k] = v
        _deep_update(full_configs, _model_configs[model_name])

    configs = parse_configs(
        configs=full_configs,
        arg_str=f"--model_name {model_name}",
        fill_required_with_null=True,
    )

    configs.load_checkpoint_dir = str(weights_dir / "checkpoint")
    configs.input_json_path = str(input_json)
    configs.dump_dir = str(output_dir)
    configs.num_workers = 0
    configs.load_strict = False
    configs.need_atom_confidence = False
    configs.sorted_by_ranking_score = False
    configs.enable_tf32 = True
    configs.use_deepspeed_evo_attention = False
    configs.triangle_attention = "torch"
    configs.triangle_multiplicative = "torch"
    configs.enable_efficient_fusion = False

    common = weights_dir / "common"
    configs.data.ccd_components_file = str(common / "components.cif")
    configs.data.ccd_components_rdkit_mol_file = str(
        common / "components.cif.rdkit_mol.pkl"
    )
    configs.data.pdb_cluster_file = str(common / "clusters-by-entity-40.txt")
    configs.data.obsolete_release_data_csv = str(common / "obsolete_release_date.csv")
    _sync_ccd_paths(weights_dir)

    configs.model.N_cycle = n_cycle
    configs.N_sample = n_sample
    return configs


def load_model(
    device,
    model_name: str = _DEFAULT_MODEL_NAME,
    weights_dir: Path | None = None,
    n_cycle: int = 4,
):
    """Load the Protenix model. Returns (model, configs).

    Only Apache-2.0 weights (v0.5.0, v1.0.0) are accepted. The protenix-v2
    weights are proprietary and are rejected with a clear error.
    """
    import json
    import tempfile
    from argparse import Namespace

    import torch

    if model_name.startswith(_REJECTED_WEIGHT_PREFIX):
        raise ValueError(
            f"Model {model_name!r} uses proprietary v2 weights that may not "
            "be reproduced or distributed. Use an Apache-2.0 model: "
            f"{sorted(_ALLOWED_WEIGHT_NAMES)}."
        )
    if model_name not in _ALLOWED_WEIGHT_NAMES:
        log.warning(
            "Model name %r is not in the known Apache-2.0 set %s. "
            "Proceeding, but verify the licence before publishing.",
            model_name, sorted(_ALLOWED_WEIGHT_NAMES),
        )

    _bootstrap_layernorm()

    torch.serialization.add_safe_globals([Namespace])

    if weights_dir is None:
        weights_dir = _get_cache_dir()
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    _download_weights(weights_dir, model_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_json = Path(tmp_dir) / "dummy.json"
        tmp_out = Path(tmp_dir) / "out"
        tmp_out.mkdir()
        data = [{"name": "dummy", "sequences": [
            {"proteinChain": {"sequence": "ACDEF", "count": 1}},
            {"proteinChain": {"sequence": "ACDEFGHIKLMNPQRSTVWY", "count": 1}},
        ], "covalent_bonds": []}]
        tmp_json.write_text(json.dumps(data))

        configs = _build_configs(
            model_name=model_name,
            weights_dir=weights_dir,
            input_json=tmp_json,
            output_dir=tmp_out,
            n_cycle=n_cycle,
        )

    from protenix.model.protenix import Protenix

    model = Protenix(configs).to(device)
    ckpt_path = weights_dir / "checkpoint" / f"{model_name}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    sample_key = next(iter(state_dict))
    if sample_key.startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.configs.triangle_attention = "torch"
    model.configs.triangle_multiplicative = "torch"
    model.configs.enable_efficient_fusion = False
    log.info("Protenix %s loaded (%s)", model_name, ckpt_path.name)
    return model, configs


# ---------------------------------------------------------------------------
# Data pipeline helpers
# ---------------------------------------------------------------------------


def write_protenix_json(
    chains: dict[str, str],
    out_path: Path,
    msa_paths: dict[str, str | Path] | None = None,
) -> None:
    """Write a Protenix JSON input for a multi-chain complex.

    When *msa_paths* maps chain ids to ``.a3m`` files, the JSON emits
    ``unpairedMsaPath`` for each protein chain so that Protenix reads the
    pinned wild-type MSA instead of running a server query.
    """
    import json

    sequences = []
    for ch_id, seq in chains.items():
        entry: dict = {"sequence": seq, "count": 1}
        if msa_paths is not None and ch_id in msa_paths:
            entry["unpairedMsaPath"] = str(msa_paths[ch_id])
        sequences.append({"proteinChain": entry})
    data = [{"name": "complex", "sequences": sequences, "covalent_bonds": []}]
    out_path.write_text(json.dumps(data, indent=2))


def resolve_msa_paths(
    chains: dict[str, str],
    msa_dir: str | Path | None,
    single_seq: bool = False,
) -> dict[str, Path] | None:
    """Resolve per-chain MSA .a3m paths from a Boltz-style MSA directory.

    The Boltz pipeline writes MSA files under
    ``data/raw/boltz_attr/<DATASET>_<CHAIN>/processed/msa/``. Each entity
    gets a subdirectory (``0/``, ``1/``, ...) with an ``a3m/`` containing
    ``.a3m`` files. The first ``.a3m`` in each entity dir is used.

    Protenix reads ``.a3m`` natively via the ``unpairedMsaPath`` JSON field.

    Returns None when *single_seq* is True or *msa_dir* is None AND
    *single_seq* is True. Raises if *msa_dir* is given but files are
    missing and *single_seq* is False.
    """
    if single_seq:
        return None

    if msa_dir is None:
        return None

    msa_dir = Path(msa_dir)
    if not msa_dir.is_dir():
        raise FileNotFoundError(
            f"MSA directory does not exist: {msa_dir}. "
            "Supply --msa-dir or pass --single-seq to run without MSA."
        )

    msa_paths: dict[str, Path] = {}
    chain_labels = list(chains.keys())

    # Try Boltz layout: msa_dir/processed/msa/<entity_idx>/a3m/*.a3m
    processed_msa = msa_dir / "processed" / "msa"
    if processed_msa.is_dir():
        entity_dirs = sorted(
            [d for d in processed_msa.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        for i, ch_id in enumerate(chain_labels):
            if i >= len(entity_dirs):
                break
            a3m_dir = entity_dirs[i] / "a3m"
            if a3m_dir.is_dir():
                a3m_files = sorted(a3m_dir.glob("*.a3m"))
                if a3m_files:
                    msa_paths[ch_id] = a3m_files[0]

    # Fallback: msa_dir contains .a3m files directly
    if not msa_paths:
        for ch_id in chain_labels:
            candidates = [
                msa_dir / f"{ch_id}.a3m",
                msa_dir / f"chain_{ch_id}.a3m",
            ]
            for c in candidates:
                if c.exists():
                    msa_paths[ch_id] = c
                    break
            # Also try numbered files
            if ch_id not in msa_paths:
                idx = chain_labels.index(ch_id)
                numbered = msa_dir / f"{idx}.a3m"
                if numbered.exists():
                    msa_paths[ch_id] = numbered

    if not msa_paths:
        raise FileNotFoundError(
            f"No .a3m files found in {msa_dir}. Checked Boltz layout "
            f"({processed_msa}/*/a3m/) and flat layout. "
            "Supply --single-seq to run without MSA."
        )

    missing = set(chain_labels) - set(msa_paths)
    if missing:
        log.warning(
            "MSA files missing for chains %s. Those chains will use "
            "single-sequence mode.",
            sorted(missing),
        )

    return msa_paths if msa_paths else None


def build_features(
    chains: dict[str, str],
    cache_dir: Path,
    device,
    configs: Any,
    model=None,
    msa_paths: dict[str, str | Path] | None = None,
    msa_depth: int | None = None,
) -> tuple[dict, dict[tuple[str, int], int]]:
    """Build Protenix input_feature_dict and a token map.

    When ``model`` is provided, ``relp`` is generated via the model's
    relative_position_encoding module (required for the recycling loop).

    When ``msa_paths`` maps chain ids to .a3m files, the JSON input
    includes ``unpairedMsaPath`` so the pinned wild-type MSA is used.

    When ``msa_depth`` is set, the MSA sampling cutoff is overridden
    to cap the number of MSA rows used by the model.

    Returns (input_feature_dict, token_map) where token_map maps
    (chain_label, residue_index) -> token position.
    """
    import torch
    from protenix.data.inference.infer_dataloader import get_inference_dataloader
    from protenix.model.protenix import update_input_feature_dict

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / "input.json"
    write_protenix_json(chains, json_path, msa_paths=msa_paths)
    configs.input_json_path = str(json_path)
    _sync_ccd_paths(Path(configs.load_checkpoint_dir).parent)

    if msa_depth is not None:
        configs.data.msa.sample_cutoff.test = msa_depth
        configs.data.msa.max_size.test = msa_depth

    dataloader = get_inference_dataloader(configs=configs)
    for batch in dataloader:
        data_tuple = batch[0]
        data, atom_array, error_msg = data_tuple
        if error_msg:
            raise RuntimeError(f"Protenix data pipeline error: {error_msg}")
        input_feature_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data["input_feature_dict"].items()
        }
        break
    else:
        raise RuntimeError("Protenix dataloader returned no samples")

    if "relp" not in input_feature_dict and model is not None:
        with torch.no_grad():
            input_feature_dict = model.relative_position_encoding.generate_relp(
                input_feature_dict
            )

    input_feature_dict = update_input_feature_dict(input_feature_dict)

    token_map = _build_token_map(input_feature_dict, chains)
    return input_feature_dict, token_map


def _build_token_map(
    input_feature_dict: dict,
    chains: dict[str, str],
) -> dict[tuple[str, int], int]:
    """Map (chain_id, 0-based residue index) -> token index.

    Protenix assigns asym_id sequentially to chains in JSON order.
    """
    import torch

    asym_id = input_feature_dict["asym_id"]
    if isinstance(asym_id, torch.Tensor):
        asym_id = asym_id.cpu().squeeze().numpy()
    asym_id = np.asarray(asym_id).ravel()

    residue_index = input_feature_dict.get("residue_index")
    if residue_index is not None:
        if isinstance(residue_index, torch.Tensor):
            residue_index = residue_index.cpu().squeeze().numpy()
        residue_index = np.asarray(residue_index).ravel()

    chain_labels = list(chains.keys())
    unique_asyms = sorted(set(asym_id.tolist()))
    asym_to_label = {}
    for i, a in enumerate(unique_asyms):
        if i < len(chain_labels):
            asym_to_label[int(a)] = chain_labels[i]
        else:
            asym_to_label[int(a)] = str(int(a))

    # Protenix emits ``residue_index`` 1-based: verified on 1JTG, where chain A
    # runs 1..262 and chain B runs 1..165 against sequences of exactly those
    # lengths. Callers look up 0-based offsets into the reference sequence, so
    # normalise by each chain's own minimum. Subtracting the per-chain minimum
    # rather than a hard-coded 1 keeps this correct if Protenix ever emits
    # 0-based indices or genuine PDB numbering with an offset.
    chain_min: dict[int, int] = {}
    if residue_index is not None:
        for tok_idx in range(len(asym_id)):
            a = int(asym_id[tok_idx])
            r = int(residue_index[tok_idx])
            chain_min[a] = r if a not in chain_min else min(chain_min[a], r)

    token_map: dict[tuple[str, int], int] = {}
    chain_counters: dict[int, int] = {}
    for tok_idx in range(len(asym_id)):
        a = int(asym_id[tok_idx])
        chain_label = asym_to_label.get(a, str(a))
        if residue_index is not None:
            resi = int(residue_index[tok_idx]) - chain_min[a]
        else:
            count = chain_counters.get(a, 0)
            resi = count
            chain_counters[a] = count + 1
        key = (chain_label, resi)
        if key not in token_map:
            token_map[key] = tok_idx
    return token_map


# ---------------------------------------------------------------------------
# Canonical AA token indices (Protenix space)
# ---------------------------------------------------------------------------


def canonical_aa_token_indices() -> np.ndarray:
    """Return int64 array of shape (20,) with indices 0..19 for
    the 20 canonical amino acids in Protenix's residue type space.
    """
    return np.arange(20, dtype=np.int64)


# ---------------------------------------------------------------------------
# Patch P2: Out-of-place MSA inference_forward
# ---------------------------------------------------------------------------


def _oop_msa_inference_forward(self, m, z, chunk_size=2048):
    """Out-of-place replacement for MSAStack.inference_forward.

    The original does ``m[start:end] += ...`` in-place, which corrupts
    the autograd version counter and raises CopySlices errors during
    backward. This replacement concatenates chunks out-of-place.
    """
    import torch

    num_msa = m.shape[-3]
    no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
    updated_chunks = []
    for i in range(no_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_msa)
        mc = m[start:end, :, :]
        mc = mc + self.msa_pair_weighted_averaging(mc, z)
        mc = mc + self.transition_m(mc)
        updated_chunks.append(mc)
    return torch.cat(updated_chunks, dim=-3) if len(updated_chunks) > 1 else updated_chunks[0]


def _patch_msa_inplace(model) -> None:
    """Apply patch P2 to all MSAStack instances in the model."""
    import types

    from protenix.model.modules.pairformer import MSAStack

    for module in model.modules():
        if isinstance(module, MSAStack):
            module.inference_forward = types.MethodType(
                _oop_msa_inference_forward, module
            )


# ---------------------------------------------------------------------------
# Patch P1: IG-safe recycling loop
# ---------------------------------------------------------------------------


def _ig_safe_pairformer(
    model,
    s_inputs_interp,
    input_feature_dict: dict,
    n_cycle: int,
):
    """Recycling loop without the set_grad_enabled(self.training) guard.

    Port of IG_Agros/src/ig/protenix.py:504-567 adapted for this repo.
    """
    import torch

    rel_pos = model.relative_position_encoding(input_feature_dict["relp"])
    token_bonds_z = model.linear_no_bias_token_bond(
        input_feature_dict["token_bonds"].unsqueeze(dim=-1)
    )
    z_constraint = None
    if "constraint_feature" in input_feature_dict:
        z_constraint = model.constraint_embedder(
            input_feature_dict["constraint_feature"]
        )

    s_init = model.linear_no_bias_sinit(s_inputs_interp)
    z_init = (
        model.linear_no_bias_zinit1(s_init)[..., None, :]
        + model.linear_no_bias_zinit2(s_init)[..., None, :, :]
        + rel_pos
        + token_bonds_z
    )
    if z_constraint is not None:
        z_init = z_init + z_constraint

    z = torch.zeros_like(z_init)
    s = torch.zeros_like(s_init)

    for _ in range(n_cycle):
        z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
        if model.template_embedder.n_blocks > 0:
            z = z + model.template_embedder(
                input_feature_dict, z, inplace_safe=False
            )
        z = model.msa_module(
            input_feature_dict, z, s_inputs_interp,
            pair_mask=None, inplace_safe=False,
        )
        s = s_init + model.linear_no_bias_s(model.layernorm_s(s))
        s, z = model.pairformer_stack(
            s.clone(), z.clone(), pair_mask=None, inplace_safe=False,
        )

    return s, z


# ---------------------------------------------------------------------------
# Patch P4: Differentiable score computation from live logits
#
# These are local reimplementations of protenix.model.sample_confidence
# functions (get_bin_centers, calculate_iptm, calculate_ptm) so that
# this module and its tests can run without protenix installed.
# ---------------------------------------------------------------------------


def _get_bin_centers(min_bin: float, max_bin: float, no_bins: int):
    import torch

    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(min_bin, max_bin - bin_width, no_bins)
    return boundaries + 0.5 * bin_width


def _calculate_normalization(N: int) -> float:
    return 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8


def _calculate_ptm(pae_prob, has_frame, min_bin, max_bin, no_bins, eps=1e-8):
    has_frame = has_frame.bool()
    N_d = has_frame.shape[-1]
    ptm_norm = _calculate_normalization(N_d)
    bc = _get_bin_centers(min_bin, max_bin, no_bins).to(pae_prob.device)
    per_bin = (1.0 / (1.0 + (bc / ptm_norm) ** 2))
    ttp = (pae_prob * per_bin).sum(dim=-1)
    ptm = ttp.mean(dim=-1)[..., has_frame].max(dim=-1).values
    return ptm


def _calculate_iptm(pae_prob, has_frame, asym_id, min_bin, max_bin, no_bins, eps=1e-8):
    has_frame = has_frame.bool()
    N_d = has_frame.shape[-1]
    ptm_norm = _calculate_normalization(N_d)
    bc = _get_bin_centers(min_bin, max_bin, no_bins).to(pae_prob.device)
    per_bin = (1.0 / (1.0 + (bc / ptm_norm) ** 2))
    ttp = (pae_prob * per_bin).sum(dim=-1)
    is_diff_chain = asym_id[None, :] != asym_id[:, None]
    iptm = (ttp * is_diff_chain.float()).sum(dim=-1) / (
        eps + is_diff_chain.float().sum(dim=-1)
    )
    iptm = iptm[..., has_frame].max(dim=-1).values
    return iptm


def _compute_score_from_logits(
    score_name: str,
    plddt_logits,
    pae_logits,
    pde_logits,
    input_feature_dict: dict,
    distogram_logits=None,
):
    """Compute a differentiable scalar score from confidence logits.

    Never calls compute_full_data_and_summary (decorated @torch.no_grad).
    """
    import torch

    def _logits_to_score(logits, min_bin, max_bin, no_bins):
        prob = torch.softmax(logits.float(), dim=-1)
        centres = _get_bin_centers(min_bin, max_bin, no_bins).to(logits.device)
        return prob @ centres, prob

    asym_id = input_feature_dict["asym_id"].squeeze().long()
    has_frame = input_feature_dict.get(
        "has_frame", input_feature_dict.get("token_exists_mask")
    )
    if has_frame is not None:
        has_frame = has_frame.squeeze().bool()

    pde_l = pde_logits.squeeze()
    if pde_l.dim() == 4:
        pde_l = pde_l[0]
    pae_l = pae_logits.squeeze()
    if pae_l.dim() == 4:
        pae_l = pae_l[0]
    plddt_l = plddt_logits.squeeze()
    if plddt_l.dim() == 3:
        plddt_l = plddt_l[0]

    if score_name == "complex_pde":
        pde_scores, _ = _logits_to_score(pde_l, **_PDE_BIN_PARAMS)
        return pde_scores.mean()

    elif score_name == "complex_plddt":
        plddt_scores, _ = _logits_to_score(plddt_l, **_PLDDT_BIN_PARAMS)
        return plddt_scores.mean() * 100.0

    elif score_name == "pae_interface_mean":
        pae_scores, _ = _logits_to_score(pae_l, **_PAE_BIN_PARAMS)
        is_diff_chain = asym_id[None, :] != asym_id[:, None]
        n_cross = is_diff_chain.float().sum()
        if n_cross > 0:
            return (pae_scores * is_diff_chain.float()).sum() / n_cross
        return pae_scores.mean()

    elif score_name == "iptm":
        _, pae_prob = _logits_to_score(pae_l, **_PAE_BIN_PARAMS)
        return _calculate_iptm(
            pae_prob=pae_prob, has_frame=has_frame, asym_id=asym_id,
            **_PAE_BIN_PARAMS,
        )

    elif score_name == "ptm":
        _, pae_prob = _logits_to_score(pae_l, **_PAE_BIN_PARAMS)
        return _calculate_ptm(
            pae_prob=pae_prob, has_frame=has_frame, **_PAE_BIN_PARAMS,
        )

    raise ValueError(
        f"Unknown score {score_name!r}. Available: {sorted(PROTENIX_SCORES)}"
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def build_mean_aa_res_type_baseline(res_type, input_feature_dict):
    """Uniform 1/20 over canonical AAs at protein positions, unchanged elsewhere."""
    import torch

    baseline = res_type.detach().clone().float()
    n_tokens = res_type.shape[-1]
    uniform = torch.zeros(n_tokens, device=res_type.device, dtype=baseline.dtype)
    for idx in _CANONICAL_AA_RANGE:
        uniform[idx] = 1.0 / 20.0

    restype_raw = input_feature_dict.get("restype")
    if restype_raw is not None:
        protein_mask = restype_raw.squeeze()[..., :20].sum(dim=-1) > 0
    else:
        protein_mask = torch.ones(baseline.shape[-2], dtype=torch.bool, device=res_type.device)
    baseline[..., protein_mask, :] = uniform
    return baseline


def build_unk_res_type_baseline(res_type, input_feature_dict):
    """UNK-token baseline: index 20 at every protein position."""
    import torch

    baseline = res_type.detach().clone().float()
    n_tokens = res_type.shape[-1]
    unk_onehot = torch.zeros(n_tokens, device=res_type.device, dtype=baseline.dtype)
    unk_onehot[_UNK_INDEX] = 1.0

    restype_raw = input_feature_dict.get("restype")
    if restype_raw is not None:
        protein_mask = restype_raw.squeeze()[..., :20].sum(dim=-1) > 0
    else:
        protein_mask = torch.ones(baseline.shape[-2], dtype=torch.bool, device=res_type.device)
    baseline[..., protein_mask, :] = unk_onehot
    return baseline


def build_shuffled_res_type_baseline(res_type, input_feature_dict, seed=0):
    """Composition-matched shuffled-sequence baseline."""
    import torch

    baseline = res_type.detach().clone().float()
    restype_raw = input_feature_dict.get("restype")
    if restype_raw is not None:
        protein_mask = restype_raw.squeeze()[..., :20].sum(dim=-1) > 0
    else:
        protein_mask = torch.ones(baseline.shape[-2], dtype=torch.bool, device=res_type.device)

    protein_indices = protein_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    if len(protein_indices) < 2:
        return baseline
    protein_rows = baseline[..., protein_indices, :].clone()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(protein_indices))
    import torch as _torch
    baseline[..., protein_indices, :] = protein_rows[..., _torch.from_numpy(perm), :]
    return baseline


# ---------------------------------------------------------------------------
# Cache-key helpers
# ---------------------------------------------------------------------------


def _chain_cache_key(chains: dict[str, str]) -> str:
    """Deterministic key encoding chain set identity (same logic as boltz_score)."""
    import hashlib

    lengths = "_".join(f"{cid}{len(seq)}" for cid, seq in sorted(chains.items()))
    payload = "|".join(f"{cid}:{seq}" for cid, seq in sorted(chains.items()))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{lengths}-{digest}"


def _prediction_cache_key(chains: dict[str, str], dataset: str) -> str:
    return f"{dataset}_{_chain_cache_key(chains)}"


def coord_fingerprint(tensor) -> dict[str, Any]:
    """Stable fingerprint of a coordinate tensor for provenance comparison."""
    import hashlib
    import torch

    t = tensor.detach().cpu().float()
    rounded = torch.round(t * 1e6) / 1e6
    h = hashlib.sha256(rounded.numpy().tobytes()).hexdigest()
    return {
        "sha256": h,
        "shape": list(t.shape),
        "abs_max": float(t.abs().max()),
    }


# ---------------------------------------------------------------------------
# Fixed-coordinate injection
# ---------------------------------------------------------------------------


def predict_structure_coords(
    model,
    input_feature_dict,
    configs,
    device,
    n_sample=1,
    *,
    cache_dir=None,
    chains=None,
    dataset=None,
):
    """Run one prediction without gradients and return detached coords.

    When ``cache_dir``, ``chains``, and ``dataset`` are all supplied, the
    result is cached on disk at ``cache_dir / "predicted_coords" / key /
    "x_pred.pt"`` and reused on subsequent calls.  The cache validates that
    the stored tensor's atom count matches ``input_feature_dict["ref_pos"]``.
    """
    import tempfile

    import torch

    _use_cache = cache_dir is not None and chains is not None and dataset is not None

    if _use_cache:
        cache_key = _prediction_cache_key(chains, dataset)
        pred_dir = Path(cache_dir) / "predicted_coords" / cache_key
        cached_path = pred_dir / "x_pred.pt"
        expected_n_atoms = int(input_feature_dict["ref_pos"].shape[-2])

        if cached_path.exists():
            log.info("Loading cached predicted coords from %s", cached_path)
            x_pred = torch.load(
                cached_path,
                map_location=device if isinstance(device, str) else str(device),
                weights_only=True,
            )
            if x_pred.shape[-2] != expected_n_atoms:
                log.warning(
                    "Cached x_pred has %d atoms but feats expects %d; "
                    "re-predicting.",
                    x_pred.shape[-2], expected_n_atoms,
                )
            else:
                log.info(
                    "Cached x_pred validated: shape %s, abs max %.6g",
                    list(x_pred.shape), float(x_pred.abs().max()),
                )
                return x_pred

    with torch.no_grad():
        s_inputs, s, z = model.get_pairformer_output(
            input_feature_dict,
            N_cycle=model.configs.model.N_cycle,
            inplace_safe=False,
        )
        ifd_diffusion = dict(input_feature_dict)
        for key in list(ifd_diffusion.keys()):
            if "template_" in key or key in (
                "msa", "has_deletion", "deletion_value", "profile", "deletion_mean",
            ):
                del ifd_diffusion[key]

        noise_schedule = model.inference_noise_scheduler(
            N_step=200, device=s_inputs.device, dtype=s_inputs.dtype,
        )
        coords = model.sample_diffusion(
            denoise_net=model.diffusion_module,
            input_feature_dict=ifd_diffusion,
            s_inputs=s_inputs, s_trunk=s, z_trunk=z,
            pair_z=None, p_lm=None, c_l=None,
            N_sample=n_sample,
            noise_schedule=noise_schedule,
            inplace_safe=False,
            enable_efficient_fusion=False,
        )

    x_pred = coords.detach()

    if _use_cache:
        pred_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "Structure prediction complete: x_pred shape %s, abs max %.6g",
            list(x_pred.shape), float(x_pred.abs().max()),
        )
        fd, tmp = tempfile.mkstemp(dir=str(pred_dir), suffix=".tmp")
        try:
            os.close(fd)
            torch.save(x_pred, tmp)
            os.replace(tmp, cached_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        log.info("Cached predicted coords at %s", cached_path)

    return x_pred


# ---------------------------------------------------------------------------
# Main confidence forward (full trunk, differentiable)
# ---------------------------------------------------------------------------


def confidence_forward(
    model,
    restype,
    input_feature_dict: dict,
    x_pred,
    score_name: str,
    n_cycle: int = 4,
    gradient_checkpointing: bool = True,
):
    """Full-trunk + confidence forward, differentiable w.r.t. ``restype``.

    Applies patches P1-P4 at runtime.
    """
    import torch
    from torch.utils.checkpoint import checkpoint as _checkpoint

    if score_name not in PROTENIX_SCORES:
        raise ValueError(
            f"Unknown score {score_name!r}. Available: {sorted(PROTENIX_SCORES)}"
        )

    # P2: patch MSA in-place update
    _patch_msa_inplace(model)

    # P3: disable stop_gradient on confidence head
    orig_sg = model.confidence_head.stop_gradient
    model.confidence_head.stop_gradient = False

    def _full_trunk_and_confidence(restype_leaf):
        ifd = dict(input_feature_dict)
        ifd["restype"] = restype_leaf

        # P1: own recycling loop (gradient always enabled)
        s_inputs = model.input_embedder(ifd, inplace_safe=False)
        s, z = _ig_safe_pairformer(model, s_inputs, ifd, n_cycle)

        # Confidence head
        plddt_logits, pae_logits, pde_logits, _ = model.confidence_head(
            ifd,
            s_inputs=s_inputs, s_trunk=s, z_trunk=z,
            pair_mask=None,
            x_pred_coords=x_pred,
            use_embedding=True,
            triangle_multiplicative="torch",
            triangle_attention="torch",
            inplace_safe=False,
        )

        # P4: compute score from live logits
        scalar = _compute_score_from_logits(
            score_name, plddt_logits, pae_logits, pde_logits, ifd,
        )
        if scalar.dim() > 0:
            scalar = scalar.squeeze()
        return scalar

    if gradient_checkpointing:
        # NEITHER checkpoint flavour is usable on this path, so we run without
        # one. This differs deliberately from boltz_score.py, where the outer
        # reentrant checkpoint IS load-bearing for memory.
        #
        #   use_reentrant=True  -- returns a numerically identical scalar whose
        #     .requires_grad is True, but ``restype.grad`` comes back None. The
        #     gradient is silently lost; the only thing that catches it is
        #     res_type_gradient_forward's explicit None check. Measured on 1JTG
        #     (427 tokens, n_cycle=1, complex_pde): score 9.585430 both ways,
        #     grad None reentrant vs 233.2586 without.
        #     Boltz escapes this because its gradient target is s_inputs with
        #     per-block non-reentrant checkpoints nested inside; here the target
        #     is the restype leaf handed straight to the checkpoint, which is
        #     the case reentrant does not propagate to.
        #
        #   use_reentrant=False -- raises CheckpointError: "A different number
        #     of tensors was saved during the original forward and
        #     recomputation." Protenix's trunk is not reproducible between the
        #     forward and the recompute. determinism_check="none" would silence
        #     that, but suppressing a real recompute mismatch risks silently
        #     WRONG gradients, which is worse than no checkpointing.
        #
        # Memory does not force the issue: measured peak is 35.14 GiB at
        # n_cycle=4 on 1JTG (427 tokens), inside an 80 GB A100. Re-measure
        # before trusting this on markedly larger complexes.
        log.warning(
            "gradient_checkpointing=True requested but is unusable on the "
            "Protenix path (reentrant silently drops the restype gradient; "
            "non-reentrant fails its determinism check). Running without it; "
            "measured peak 35.14 GiB at n_cycle=4 on 1JTG."
        )
    scalar = _full_trunk_and_confidence(restype)

    model.confidence_head.stop_gradient = orig_sg

    if torch.is_grad_enabled():
        assert scalar.requires_grad, (
            f"Score {score_name!r} does not require grad after the forward pass."
        )
    assert torch.isfinite(scalar).all(), (
        f"Score {score_name!r} is not finite: {scalar.item()}"
    )

    return scalar


# ---------------------------------------------------------------------------
# res_type gradient forward (like boltz_score.res_type_gradient_forward)
# ---------------------------------------------------------------------------


def res_type_gradient_forward(
    model,
    input_feature_dict: dict,
    score_name: str,
    x_pred,
    n_cycle: int = 4,
    gradient_checkpointing: bool = True,
) -> dict:
    """Gradient of the score w.r.t. the one-hot restype tensor."""
    restype = input_feature_dict["restype"]
    rt_leaf = restype.detach().float().requires_grad_(True)

    scalar = confidence_forward(
        model, rt_leaf, input_feature_dict, x_pred, score_name,
        n_cycle=n_cycle,
        gradient_checkpointing=gradient_checkpointing,
    )
    scalar.backward()

    grad = rt_leaf.grad
    if grad is None:
        raise RuntimeError("No gradient on restype.")

    return {
        "grad_res_type": grad[0].detach().cpu().float().numpy() if grad.dim() == 3
        else grad.detach().cpu().float().numpy(),
        "score_value": float(scalar.detach()),
    }


# ---------------------------------------------------------------------------
# Numerics arm (for provenance parity with boltz)
# ---------------------------------------------------------------------------


def numerics_arm() -> dict:
    return {
        "model": "protenix",
        "model_name": _DEFAULT_MODEL_NAME,
        "autocast": "off",
        "dtype": "fp32",
    }


# ---------------------------------------------------------------------------
# record iptm argmax (mirrors boltz_score.record_iptm_argmax)
# ---------------------------------------------------------------------------


def record_iptm_argmax(
    pae_logits, input_feature_dict, token_to_chain,
) -> dict:
    """Record which token row the ipTM max selects."""
    import torch

    pae_l = pae_logits.squeeze()
    if pae_l.dim() == 4:
        pae_l = pae_l[0]
    prob = torch.softmax(pae_l.float(), dim=-1)

    asym_id = input_feature_dict["asym_id"].squeeze().long()
    has_frame = input_feature_dict.get(
        "has_frame", input_feature_dict.get("token_exists_mask")
    )
    if has_frame is not None:
        has_frame = has_frame.squeeze().bool()
    else:
        has_frame = torch.ones(asym_id.shape[0], dtype=torch.bool, device=asym_id.device)

    N_d = has_frame.shape[-1]
    ptm_norm = _calculate_normalization(N_d)
    bc = _get_bin_centers(**_PAE_BIN_PARAMS).to(prob.device)
    per_bin = (1.0 / (1.0 + (bc / ptm_norm) ** 2))
    ttp = (prob * per_bin).sum(dim=-1)

    is_diff = asym_id[None, :] != asym_id[:, None]
    iptm_per_i = (ttp * is_diff.float()).sum(dim=-1) / (
        1e-8 + is_diff.float().sum(dim=-1)
    )
    iptm_masked = iptm_per_i.clone()
    iptm_masked[~has_frame] = -1e9
    argmax_tok = int(iptm_masked.argmax())

    chain = token_to_chain.get(argmax_tok, "?")
    return {
        "argmax_token": argmax_tok,
        "argmax_chain": chain,
    }
