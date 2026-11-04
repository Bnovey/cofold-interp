"""Boltz-2 confidence-score surface for gradient attribution.

Wraps the Boltz-2 confidence head so that a single scalar score is
differentiable w.r.t. the token-level input embedding ``s_inputs``.
The trunk is run with per-block gradient checkpointing to fit within
80 GB VRAM during backward (the same strategy as the predecessor repo).
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score registry
# ---------------------------------------------------------------------------

# Pairformer blocks per gradient checkpoint. 4 keeps the retained boundary
# tensors to 16 per recycling iteration (~4.1 GiB at 730 tokens, down from
# ~16.3 GiB at one block per checkpoint) while recomputing only 4 blocks at a
# time. Override with IGV_PF_GROUP_SIZE.
_PF_GROUP_SIZE = 4


# ---------------------------------------------------------------------------
# Chunk / precision / kernel configuration
#
# Every knob below has an IGV_* environment variable (the convention set by
# IGV_PF_CHUNK) and a default that reproduces this file's behaviour before the
# knob existed, so a GPU run at default settings is bit-identical. All of these
# resolvers are pure and torch-free at import time, so they are unit-testable on
# a laptop with neither CUDA nor boltz installed.
# ---------------------------------------------------------------------------

# The chunk-size fields that make up one profile. `chunk_profile()` returns
# exactly these six (the frozen public contract); `select_chunk_profile()`
# additionally returns "profile" and "conf_chunk" for internal use.
_CHUNK_PROFILE_KEYS = (
    "pf_chunk",
    "msa_chunk_heads_pwa",
    "msa_chunk_trans_z",
    "msa_chunk_trans_msa",
    "msa_chunk_outer",
    "msa_chunk_tri",
)

# Legal values of IGV_CHUNK_PROFILE.
_CHUNK_PROFILES = ("auto", "large", "small")


def select_chunk_profile(
    n_tokens: int,
    threshold: int = 384,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Resolve the chunk-size configuration for a complex of *n_tokens* tokens.

    Boltz switches chunking algorithm at ``boltz.data.const.chunk_size_threshold``
    (384 in boltz 2.2.1). Above it the trunk runs small triangle-attention chunks
    and all four MSA chunk knobs are engaged; at or below it the chunks are large
    and every MSA knob is off. Those are two *different algorithms*, which is why
    a memory/time sweep must not fit a single curve across L=384 -- force one
    profile with ``IGV_CHUNK_PROFILE`` instead.

    ``IGV_CHUNK_PROFILE``:
      * ``auto`` (default) -- size-dependent, exactly the historical behaviour
        (``n_tokens > threshold`` selects "large").
      * ``large`` / ``small`` -- force that branch regardless of *n_tokens*.

    ``IGV_PF_CHUNK`` overrides the triangle-attention chunk size. NOTE the
    pre-existing asymmetry, preserved here deliberately: in the "large" profile
    it sets both ``pf_chunk`` and ``msa_chunk_tri`` (both defaulting to 128); in
    the "small" profile it sets only ``pf_chunk`` (default 512) while
    ``msa_chunk_tri`` is hardcoded 512. Changing that would change today's
    numbers, so it stays.

    Pure function: no torch, no boltz, no CUDA. *env* defaults to ``os.environ``
    and is injectable for tests.

    Returns
    -------
    dict
        ``profile`` ("large"/"small"), the six keys of ``_CHUNK_PROFILE_KEYS``,
        and ``conf_chunk`` (the confidence pairformer stack's chunk size, which
        :func:`enable_confidence_checkpointing` used to re-derive independently).
    """
    if env is None:
        env = os.environ
    requested = str(env.get("IGV_CHUNK_PROFILE", "auto")).strip().lower() or "auto"
    if requested not in _CHUNK_PROFILES:
        raise ValueError(
            f"IGV_CHUNK_PROFILE={requested!r} is not recognised; legal values "
            f"are {list(_CHUNK_PROFILES)}."
        )

    if requested == "auto":
        profile = "large" if int(n_tokens) > int(threshold) else "small"
    else:
        profile = requested

    if profile == "large":
        return {
            "profile": "large",
            "pf_chunk": int(env.get("IGV_PF_CHUNK", 128)),
            "msa_chunk_heads_pwa": True,
            "msa_chunk_trans_z": 64,
            "msa_chunk_trans_msa": 32,
            "msa_chunk_outer": 4,
            "msa_chunk_tri": int(env.get("IGV_PF_CHUNK", 128)),
            "conf_chunk": 128,
        }
    return {
        "profile": "small",
        "pf_chunk": int(env.get("IGV_PF_CHUNK", 512)),
        "msa_chunk_heads_pwa": False,
        "msa_chunk_trans_z": None,
        "msa_chunk_trans_msa": None,
        "msa_chunk_outer": None,
        # Not env-overridable in the small profile -- matches the historical code.
        "msa_chunk_tri": 512,
        "conf_chunk": 512,
    }


def chunk_profile(
    n_tokens: int, threshold: int = 384, profile: str | None = None
) -> dict:
    """Chunk-size configuration for *n_tokens*, as the six-key public contract.

    Thin projection of :func:`select_chunk_profile` onto exactly
    ``_CHUNK_PROFILE_KEYS``. *profile* (``"large"``/``"small"``) forces a branch;
    when it is None the value of ``IGV_CHUNK_PROFILE`` is used, and when that is
    unset the selection is size-dependent -- today's behaviour.
    """
    env = dict(os.environ)
    if profile is not None:
        env["IGV_CHUNK_PROFILE"] = str(profile)
    resolved = select_chunk_profile(n_tokens, threshold, env)
    return {key: resolved[key] for key in _CHUNK_PROFILE_KEYS}


# IGV_AUTOCAST: name -> torch dtype attribute (None == no autocast at all).
_AUTOCAST_DTYPES: dict[str, str | None] = {
    "off": None,
    "none": None,
    "0": None,
    "fp32": None,
    "float32": None,
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}

# Rejected outright, not merely discouraged -- see resolve_autocast_dtype.
_AUTOCAST_REJECTED = ("fp16", "float16", "half")


def resolve_autocast_dtype(
    spec: str | None = None, env: Mapping[str, str] | None = None
):
    """Resolve ``IGV_AUTOCAST`` to a torch dtype, or None for no autocast.

    Default ``"off"`` returns None and does not even import torch, which is
    today's exact behaviour: this repo runs the trunk in fp32 with no autocast
    anywhere.

    ``bf16``/``bfloat16`` are accepted. fp16 **raises**: there is no
    ``GradScaler`` anywhere in this repo, and fp16's 5-bit exponent would break
    the post-checkpoint guards in :func:`confidence_forward` (finiteness and
    ``abs() > 1e-12``), which only stay meaningful because bf16 shares fp32's
    8-bit exponent (bf16 min normal ~1.2e-38).
    """
    if spec is None:
        env = os.environ if env is None else env
        spec = env.get("IGV_AUTOCAST", "off")
    key = str(spec).strip().lower() or "off"
    if key in _AUTOCAST_REJECTED:
        raise ValueError(
            f"IGV_AUTOCAST={spec!r} is refused. fp16 needs a torch.cuda.amp "
            "GradScaler and this repo has none (grep GradScaler: zero hits), "
            "and fp16's 5-bit exponent would break the finiteness / non-zero "
            "guards after the outer checkpoint in confidence_forward. Use "
            "'bf16', which shares fp32's 8-bit exponent."
        )
    if key not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"IGV_AUTOCAST={spec!r} is not recognised; legal values are "
            f"{sorted(_AUTOCAST_DTYPES)} (fp16 is refused deliberately)."
        )
    name = _AUTOCAST_DTYPES[key]
    if name is None:
        return None
    import torch

    return getattr(torch, name)


_TRUE_FLAGS = ("1", "true", "yes", "on")
_FALSE_FLAGS = ("0", "false", "no", "off", "")


def _parse_bool_flag(name: str, value) -> bool:
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in _TRUE_FLAGS:
        return True
    if key in _FALSE_FLAGS:
        return False
    raise ValueError(
        f"{name}={value!r} is not a boolean; use one of "
        f"{list(_TRUE_FLAGS)} / {list(_FALSE_FLAGS[:-1])}."
    )


def resolve_use_kernels(
    model=None, spec: str | None = None, env: Mapping[str, str] | None = None
) -> tuple[bool, bool]:
    """Resolve ``(use_kernels, use_cuequiv_attn)`` from ``IGV_USE_KERNELS`` /
    ``IGV_TRI_ATTN_KERNEL``. Both default False == today's exact behaviour.

    This collapses eight scattered ``use_kernels`` decision points in this file
    into one flag. One of those eight is invisible to ``grep use_kernels``: a
    bare positional ``False`` passed as ``MSALayer.forward``'s 10th positional
    parameter.

    ENABLING THIS BUYS ZERO MEMORY TODAY and is plumbing/hygiene only. Both
    boltz kernel paths import ``cuequivariance_torch`` (boltz 2.2.1
    ``triangular_attention/primitives.py:201`` and ``triangular_mult.py:22``),
    which ``constraints.txt`` forbids installing and ``scripts/cloud/bootstrap.sh``
    actively enforces. So the flag fails eagerly and loudly here rather than from
    inside a checkpointed backward recompute stack. Lifting that ban is a user
    policy decision that this flag does not make, and it would first require
    re-verifying cuequivariance's dtype/version support (unverifiable in this
    environment) and pinning it on BOTH ends -- boltz declares it only as an
    unbounded optional extra ``>=0.5.0``.

    ``IGV_TRI_ATTN_KERNEL`` is the narrow variant: it turns on the triangle
    ATTENTION kernel only, leaving triangle-multiplication on the reference
    path. It is reachable solely at the two sites where this module calls
    ``PairformerLayer`` directly (``PairformerModule.forward`` does not forward
    ``use_cuequiv_attn``), i.e. the checkpointed trunk group and the confidence
    stack; it is a no-op for ``model.msa_module(...)``,
    ``model.pairformer_module(...)`` and ``model.confidence_module(...)``.

    When *spec* is None and ``IGV_USE_KERNELS`` is unset the result is a literal
    ``False``, which is exactly what the eight call sites hardcoded before this
    refactor. It deliberately does NOT inherit
    ``getattr(model, "use_kernels", False)``: ``Boltz2`` is restored with
    ``load_from_checkpoint``, so that attribute comes from the checkpoint's saved
    hyperparameters and is not verifiable in this environment (boltz is not
    installed here). If it ever came back True, inheriting it would silently
    switch the trunk to the kernel path -- changing the algorithm AND disabling
    triangle-attention chunking -- on a run where nobody asked for it, and would
    then hard-fail on the missing ``cuequivariance_torch``. A disagreement is
    logged instead, so the model's own flag is visible without being obeyed.
    """
    env = os.environ if env is None else env

    if spec is None:
        spec = env.get("IGV_USE_KERNELS")
    if spec is None:
        # Preserve the pre-refactor literal False. See the docstring for why the
        # model's own attribute is reported but not inherited.
        use_kernels = False
        if bool(getattr(model, "use_kernels", False)):
            log.warning(
                "model.use_kernels is True but IGV_USE_KERNELS is unset; "
                "keeping use_kernels=False, which is what this module's call "
                "sites hardcoded before IGV_USE_KERNELS existed. Set "
                "IGV_USE_KERNELS=1 to honour the model's flag."
            )
    else:
        use_kernels = _parse_bool_flag("IGV_USE_KERNELS", spec)

    attn_spec = env.get("IGV_TRI_ATTN_KERNEL")
    use_cuequiv_attn = (
        False if attn_spec is None else _parse_bool_flag("IGV_TRI_ATTN_KERNEL", attn_spec)
    )

    if use_kernels or use_cuequiv_attn:
        log.warning(
            "Kernel path requested (use_kernels=%s, use_cuequiv_attn=%s). This "
            "also DISABLES triangle-attention chunking entirely (boltz "
            "attention.py:157 `if chunk_size is not None and not use_kernels`), "
            "so IGV_PF_CHUNK and IGV_TRI_ATTN_CKPT become inert.",
            use_kernels, use_cuequiv_attn,
        )
        try:
            import cuequivariance_torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "IGV_USE_KERNELS / IGV_TRI_ATTN_KERNEL requested but "
                "cuequivariance_torch is not importable. (a) Both boltz kernel "
                "paths import it (boltz 2.2.1 triangular_attention/"
                "primitives.py:201 and triangular_mult.py:22). (b) "
                "constraints.txt forbids installing cuequivariance and "
                "scripts/cloud/bootstrap.sh enforces that. (c) Lifting the ban "
                "is a user policy decision, not something this flag does. "
                "Unset the flag, or take the policy decision explicitly first."
            ) from exc

    return bool(use_kernels), bool(use_cuequiv_attn)


def resolve_tri_attn_ckpt(
    flag: bool | None = None, env: Mapping[str, str] | None = None
) -> bool:
    """Resolve ``IGV_TRI_ATTN_CKPT``. Default ``"0"`` == today's behaviour.

    See :func:`enable_triangle_attention_chunk_checkpointing` for what it does
    and why IGV_PF_CHUNK is not a substitute.
    """
    if flag is not None:
        return bool(flag)
    env = os.environ if env is None else env
    return _parse_bool_flag("IGV_TRI_ATTN_CKPT", env.get("IGV_TRI_ATTN_CKPT", "0"))


def resolve_msa_spec(
    msa: str | dict | None = None, env: Mapping[str, str] | None = None
) -> str | dict | None:
    """Resolve ``IGV_MSA_SPEC``. Default unset -> None == today's behaviour
    (no ``msa`` key is emitted in the YAML at all).

    ``"server"`` also resolves to None: it is the name callers use for "let the
    MSA server generate it", which in YAML terms means emitting no ``msa`` key
    (boltz then reads ``msa_id == 0`` = auto-generate). Emitting the literal
    string "server" instead would make boltz treat it as a path to a custom
    .a3m/.csv.

    A ``dict[str, str | Path]`` mapping chain id to MSA file path is passed
    through unchanged -- it is consumed by :func:`_build_yaml_sequences` to
    emit per-chain ``msa`` keys."""
    if isinstance(msa, dict):
        return msa
    if msa is None:
        env = os.environ if env is None else env
        msa = env.get("IGV_MSA_SPEC")
    if msa is None:
        return None
    value = str(msa).strip()
    if value == "" or value.lower() == "server":
        return None
    return value


def resolve_assert_feat_seq(env: Mapping[str, str] | None = None) -> bool:
    """Resolve ``IGV_ASSERT_FEAT_SEQ``. Default ``"1"`` (ON): the featurised
    sequence check is a correctness guard, so it is on by default.
    ``IGV_ASSERT_FEAT_SEQ=0`` downgrades the raise to a ``log.error`` so a
    debugging session is never hard-blocked."""
    env = os.environ if env is None else env
    return _parse_bool_flag("IGV_ASSERT_FEAT_SEQ", env.get("IGV_ASSERT_FEAT_SEQ", "1"))


def numerics_arm(env: Mapping[str, str] | None = None) -> dict:
    """The knobs that change the numbers, for a provenance ``arm`` dict.

    Every stage that writes an artifact must stamp this, because two runs that
    differ only in ``IGV_AUTOCAST`` produce DIFFERENT SCORES from IDENTICAL
    inputs. Without it a bf16 artifact is byte-indistinguishable from an fp32
    one at the provenance level, and the first person to compare them gets a
    wrong answer with nothing in the record to warn them. ``08_memscale``
    already writes an equivalent ``pinned_env`` block into its own artifact;
    this is the same information in the shared shape the other stages use.

    Measured 2026-09-04 on igv-gpu at L=730: bf16 moves ``grad_abs_max`` by up
    to 3.6% versus fp32, against a ~1.7% nondeterminism floor established by a
    mathematically-transparent control (per-chunk checkpointing). So the
    perturbation is small but real and must never be silently mixed.

    Returns plain JSON-safe values; ``autocast`` is the resolved string rather
    than a torch dtype so this stays importable with no torch.
    """
    env = os.environ if env is None else env
    autocast = str(env.get("IGV_AUTOCAST", "off")).strip().lower() or "off"
    return {
        "autocast": autocast,
        "dtype": "fp32" if autocast in ("off", "none", "") else autocast,
        "tri_attn_ckpt": resolve_tri_attn_ckpt(env=env),
        "pf_group_size": max(1, int(env.get("IGV_PF_GROUP_SIZE", _PF_GROUP_SIZE))),
        "pf_chunk": env.get("IGV_PF_CHUNK"),
        "chunk_profile": env.get("IGV_CHUNK_PROFILE", "auto"),
        "use_kernels": _parse_bool_flag(
            "IGV_USE_KERNELS", env.get("IGV_USE_KERNELS", "0")
        ),
    }


SCORES: dict[str, Callable[[dict], Any]] = {}


def _register_score(name: str, fn: Callable) -> None:
    SCORES[name] = fn


def _complex_pde(out_dict):
    return out_dict["complex_pde"]


def _complex_iplddt(out_dict):
    return out_dict["complex_iplddt"]


def _complex_plddt(out_dict):
    return out_dict["complex_plddt"]


def _iptm(out_dict):
    return out_dict["iptm"]


def _ptm(out_dict):
    return out_dict["ptm"]


def _protein_iptm(out_dict):
    return out_dict["protein_iptm"]


_register_score("complex_pde", _complex_pde)
_register_score("complex_iplddt", _complex_iplddt)
_register_score("complex_plddt", _complex_plddt)
_register_score("iptm", _iptm)
_register_score("ptm", _ptm)
_register_score("protein_iptm", _protein_iptm)


# ---------------------------------------------------------------------------
# Affinity score registry
# ---------------------------------------------------------------------------

AFFINITY_SCORES: dict[str, Callable[[dict], Any]] = {}


def _register_affinity_score(name: str, fn: Callable) -> None:
    AFFINITY_SCORES[name] = fn


def _affinity_pred_value(out_dict):
    return out_dict["affinity_pred_value"]


def _affinity_logits_binary(out_dict):
    return out_dict["affinity_logits_binary"]


_register_affinity_score("affinity_pred_value", _affinity_pred_value)
_register_affinity_score("affinity_logits_binary", _affinity_logits_binary)


# ---------------------------------------------------------------------------
# Residue-token decoding tables
#
# Pinned copies of boltz 2.2.1 `boltz/data/const.py` (`canonical_tokens` at
# :95-117, `tokens` at :119-135, `prot_letter_to_token`/`prot_token_to_letter`
# at :140-171). They exist ONLY as a fallback so featurised_sequences() -- and
# its unit tests -- work in an environment where boltz is not installed. The
# real tables are preferred at runtime; if boltz ever changes them, boltz wins
# and these are dead weight rather than a silent disagreement.
# ---------------------------------------------------------------------------

_CANONICAL_TOKENS_2_2_1 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
)

_BOLTZ_TOKENS_2_2_1 = (
    "<pad>", "-", *_CANONICAL_TOKENS_2_2_1,
    "A", "G", "C", "U", "N",          # rna
    "DA", "DG", "DC", "DT", "DN",     # dna
)

_PROT_TOKEN_TO_LETTER_2_2_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "UNK": "X", "-": "-",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def select_checkpoint(checkpoint_dir):
    """Return the Boltz-2 *confidence* checkpoint inside *checkpoint_dir*.

    A full ``download_boltz2`` leaves two checkpoints in the cache::

        boltz2_aff.ckpt    affinity head   -- sorts FIRST alphabetically
        boltz2_conf.ckpt   confidence head -- what this module needs

    Every score registered here (iptm, ptm, complex_pde, ...) reads the
    confidence head, so a naive ``sorted(...)[0]`` picks the affinity model and
    the pipeline would attribute gradients of the wrong network without ever
    raising. Kept torch-free so it is testable on CPU.
    """
    from pathlib import Path as _P

    ckpt_dir = _P(checkpoint_dir)
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")

    preferred = [c for c in candidates if "aff" not in c.name.lower()]
    if not preferred:
        raise FileNotFoundError(
            f"Only affinity checkpoints found in {ckpt_dir} "
            f"({[c.name for c in candidates]}). The confidence checkpoint "
            f"(boltz2_conf.ckpt) is required; re-run scripts/cloud/fetch_weights.sh."
        )
    conf = [c for c in preferred if "conf" in c.name.lower()]
    chosen = conf[0] if conf else preferred[0]
    if len(candidates) > 1:
        log.info(
            "Checkpoints present: %s -- selected %s",
            [c.name for c in candidates], chosen.name,
        )
    return chosen


def select_affinity_checkpoint(checkpoint_dir):
    """Return the Boltz-2 *affinity* checkpoint inside *checkpoint_dir*.

    Mirrors :func:`select_checkpoint` but picks ``boltz2_aff.ckpt`` instead
    of ``boltz2_conf.ckpt``.  Kept torch-free so it is testable on CPU.
    """
    from pathlib import Path as _P

    ckpt_dir = _P(checkpoint_dir)
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")

    aff = [c for c in candidates if "aff" in c.name.lower()]
    if not aff:
        raise FileNotFoundError(
            f"No affinity checkpoint found in {ckpt_dir} "
            f"({[c.name for c in candidates]}). The affinity checkpoint "
            f"(boltz2_aff.ckpt) is required."
        )
    chosen = aff[0]
    if len(candidates) > 1:
        log.info(
            "Checkpoints present: %s -- selected affinity %s",
            [c.name for c in candidates], chosen.name,
        )
    return chosen


def load_affinity_model(checkpoint_dir, device):
    """Load Boltz-2 with the affinity head from *checkpoint_dir*.

    Returns ``(model, boltz_version)``.  The affinity checkpoint contains the
    full model (trunk + affinity module); this loader ensures
    ``affinity_prediction`` is True so the ``affinity_module`` attribute
    exists.
    """
    import importlib.metadata as md

    import torch
    from boltz.main import BoltzSteeringParams
    from boltz.model.models.boltz2 import Boltz2
    from dataclasses import asdict

    boltz_version = md.version("boltz")

    ckpt_path = select_affinity_checkpoint(checkpoint_dir)
    log.info("Loading Boltz-2 affinity from %s (boltz %s)", ckpt_path, boltz_version)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "hyper_parameters" in ckpt:
        _scrub_hparams(ckpt["hyper_parameters"], {"mse_rotational_alignment"})
        _ensure_pairformer_v2(ckpt["hyper_parameters"])
        _ensure_affinity_prediction(ckpt["hyper_parameters"])

    import os
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".ckpt")
    os.close(fd)
    try:
        torch.save(ckpt, tmp_path)
        model = Boltz2.load_from_checkpoint(
            tmp_path, map_location=device, strict=False, ema=False,
        )
    finally:
        os.unlink(tmp_path)

    model.affinity_prediction = True
    if model.steering_args is None:
        steering = BoltzSteeringParams()
        steering.fk_steering = False
        steering.physical_guidance_update = False
        steering.contact_guidance_update = False
        model.steering_args = asdict(steering)
    model.eval()
    model.to(device)

    has_aff = (
        hasattr(model, "affinity_module") and model.affinity_module is not None
    )
    has_ens = (
        hasattr(model, "affinity_module1") and model.affinity_module1 is not None
        and hasattr(model, "affinity_module2") and model.affinity_module2 is not None
    )
    log.info(
        "Boltz-2 affinity loaded: affinity_module=%s, ensemble=%s",
        has_aff, has_ens,
    )
    return model, boltz_version


def load_model(checkpoint_dir, device):
    """Load Boltz-2 in eval mode from *checkpoint_dir*.

    Returns ``(model, boltz_version)`` where *boltz_version* is
    the installed ``boltz`` package version string.
    """
    import importlib.metadata as md

    import torch
    from boltz.main import BoltzSteeringParams
    from boltz.model.models.boltz2 import Boltz2
    from dataclasses import asdict

    boltz_version = md.version("boltz")

    ckpt_path = select_checkpoint(checkpoint_dir)
    log.info("Loading Boltz-2 from %s (boltz %s)", ckpt_path, boltz_version)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "hyper_parameters" in ckpt:
        _scrub_hparams(ckpt["hyper_parameters"], {"mse_rotational_alignment"})
        _ensure_pairformer_v2(ckpt["hyper_parameters"])
        _ensure_confidence_prediction(ckpt["hyper_parameters"])

    import os
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".ckpt")
    os.close(fd)
    try:
        torch.save(ckpt, tmp_path)
        model = Boltz2.load_from_checkpoint(
            tmp_path, map_location=device, strict=False, ema=False,
        )
    finally:
        os.unlink(tmp_path)

    model.confidence_prediction = True
    if model.steering_args is None:
        steering = BoltzSteeringParams()
        steering.fk_steering = False
        steering.physical_guidance_update = False
        steering.contact_guidance_update = False
        model.steering_args = asdict(steering)
    model.eval()
    model.to(device)
    log.info("Boltz-2 loaded, confidence_module present: %s",
             hasattr(model, "confidence_module") and model.confidence_module is not None)
    return model, boltz_version


def _scrub_hparams(obj, remove_keys):
    from collections.abc import Mapping
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        for k in remove_keys:
            try:
                obj.pop(k, None)
            except Exception:
                pass
        try:
            vals = list(obj.values())
        except Exception:
            vals = []
        for v in vals:
            _scrub_hparams(v, remove_keys)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scrub_hparams(v, remove_keys)


def _ensure_pairformer_v2(obj):
    from collections.abc import Mapping
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        if "pairformer_args" in obj:
            pa = obj["pairformer_args"]
            if isinstance(pa, Mapping) or hasattr(pa, "items"):
                try:
                    pa["v2"] = True
                except Exception:
                    pass
        try:
            for v in obj.values():
                _ensure_pairformer_v2(v)
        except Exception:
            pass
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _ensure_pairformer_v2(v)


def _ensure_confidence_prediction(obj):
    from collections.abc import Mapping
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        try:
            obj["confidence_prediction"] = True
        except Exception:
            pass


def _ensure_affinity_prediction(obj):
    from collections.abc import Mapping
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        try:
            obj["affinity_prediction"] = True
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------


def _build_yaml_sequences(
    chains: dict[str, str],
    resolved_msa: str | dict | None,
) -> list[dict]:
    """Build the ``sequences`` list for a Boltz YAML spec.

    Pure function: no torch, no boltz, no I/O. Factored out of
    :func:`build_complex_feats` so the YAML-construction logic is unit-testable
    without GPU or network.

    Parameters
    ----------
    chains : dict[str, str]
        Chain id -> amino-acid sequence.
    resolved_msa : str | dict | None
        Already-resolved MSA spec from :func:`resolve_msa_spec`.
        * ``None`` -- emit no ``msa`` key (server auto-generates).
        * ``str`` -- same value for every chain (e.g. ``"empty"`` or a path).
        * ``dict[str, str | Path]`` -- per-chain MSA file paths. Must cover
          every chain in *chains*; every mapped path must exist.

    Returns
    -------
    list[dict]
        Each element is ``{"protein": {"id": ..., "sequence": ..., ["msa": ...]}}``.
    """
    if isinstance(resolved_msa, dict):
        missing = set(chains) - set(resolved_msa)
        if missing:
            raise ValueError(
                f"Per-chain MSA dict is missing chains: {sorted(missing)}. "
                f"Chains in the complex: {sorted(chains)}; "
                f"chains in the MSA dict: {sorted(resolved_msa)}."
            )
        for chain_id, msa_path in resolved_msa.items():
            p = Path(msa_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Per-chain MSA path for chain {chain_id!r} does not exist: "
                    f"{p}. A missing MSA would silently fall back to the server, "
                    f"which is the failure mode per-chain paths exist to prevent."
                )

    sequences = []
    for chain_id, seq in chains.items():
        entry: dict[str, Any] = {"id": chain_id, "sequence": seq}
        if isinstance(resolved_msa, dict):
            entry["msa"] = str(resolved_msa[chain_id])
        elif resolved_msa is not None:
            entry["msa"] = resolved_msa
        sequences.append({"protein": entry})
    return sequences


def build_affinity_yaml_spec(
    protein_chains: dict[str, str],
    ligand_id: str,
    ligand_smiles: str | None = None,
    ligand_ccd: str | None = None,
    resolved_msa: str | dict | None = None,
) -> dict:
    """Build a Boltz YAML spec dict for a protein-ligand affinity prediction.

    Pure function: no torch, no boltz, no I/O.

    Parameters
    ----------
    protein_chains : dict[str, str]
        Protein chain id -> amino-acid sequence.
    ligand_id : str
        Chain id for the ligand.
    ligand_smiles : str or None
        SMILES string. Exactly one of *ligand_smiles* / *ligand_ccd* required.
    ligand_ccd : str or None
        CCD code.  Exactly one of *ligand_smiles* / *ligand_ccd* required.
    resolved_msa : str | dict | None
        MSA spec for the protein chains (same semantics as
        :func:`_build_yaml_sequences`).

    Returns
    -------
    dict
        A complete YAML-serialisable spec with ``version``, ``sequences``,
        and ``properties: [{affinity: {binder: <ligand_id>}}]``.
    """
    if (ligand_smiles is None) == (ligand_ccd is None):
        raise ValueError(
            "Exactly one of ligand_smiles / ligand_ccd must be provided."
        )
    if ligand_id in protein_chains:
        raise ValueError(
            f"ligand_id {ligand_id!r} collides with a protein chain id."
        )

    sequences: list[dict] = []

    if isinstance(resolved_msa, dict):
        missing = set(protein_chains) - set(resolved_msa)
        if missing:
            raise ValueError(
                f"Per-chain MSA dict is missing chains: {sorted(missing)}."
            )

    for chain_id, seq in protein_chains.items():
        entry: dict[str, Any] = {"id": chain_id, "sequence": seq}
        if isinstance(resolved_msa, dict):
            entry["msa"] = str(resolved_msa[chain_id])
        elif resolved_msa is not None:
            entry["msa"] = resolved_msa
        sequences.append({"protein": entry})

    lig_entry: dict[str, Any] = {"id": ligand_id}
    if ligand_smiles is not None:
        lig_entry["smiles"] = ligand_smiles
    else:
        lig_entry["ccd"] = ligand_ccd
    sequences.append({"ligand": lig_entry})

    return {
        "version": 1,
        "sequences": sequences,
        "properties": [{"affinity": {"binder": ligand_id}}],
    }


def _inject_pdb_coords_into_processed_structure(
    proc_out_dir: Path,
    pdb_path: Path,
    resnum_offset: int = 0,
) -> int:
    """Patch the processed structure's atom coordinates from a PDB file.

    After ``process_inputs`` writes a processed structure with (0,0,0)
    coordinates (because the YAML is sequence-only), this function reads a
    real PDB, maps its ATOM records onto the processed structure's atom
    array by (chain letter, residue number, atom name), and writes the real
    coordinates back.  The ``AffinityCropper`` then sees real geometry and
    selects the pocket neighbourhood rather than the first ~200 tokens in
    sequence order.

    For protein chains (``mol_type == 0``), the PDB residue number is
    computed as ``chain_relative_index + resnum_offset``.  For non-polymer
    chains (ligands, ``mol_type == 3``), HETATM records are matched by
    ``(resname, atom_name)`` regardless of chain letter, because the PDB
    may place the ligand in the protein chain while boltz gives it a
    separate chain.

    Residues with **no** matching PDB atoms get ``is_present = False``,
    which propagates through the tokenizer's ``is_present = res & center``
    check and the cropper's ``resolved_mask``, correctly excluding them
    from the crop rather than leaving them at the origin as phantom points.

    **Pre-affinity file synthesis.**  Boltz's affinity path
    (``inferencev2.py:62-65``) loads the structure from
    ``<structures>/<record_id>/pre_affinity_<record_id>.npz``.  That file
    is normally produced by the WRITER (``writer.py:178-180``) during a
    prediction run, containing the top-ranked diffusion sample's
    coordinates.  Because this repo holds geometry fixed at a deposited PDB
    rather than running structure prediction, we synthesise the file here
    with the *experimental* coordinates that were just injected.  This is
    deterministic (the diffusion sampler is not seeded) and matches this
    repo's fixed-wild-type-geometry approach exactly.

    The ``record_id`` is derived from the structure npz filename stem,
    which ``process_inputs`` (``main.py:650``) sets to
    ``target.record.id``, itself the YAML file stem
    (``parse/yaml.py:67``).

    **Non-polymer zero-coordinate guard.**  If any chain whose
    ``mol_type`` marks it as a non-polymer (ligand) ends up with zero
    injected atoms, the function raises ``RuntimeError``.  A ligand at the
    origin would make every protein-to-ligand distance equal to the
    protein atom's distance from the origin, causing the affinity crop to
    select residues near an arbitrary point instead of the binding pocket.
    This failure is silent downstream so a loud crash here is essential.

    Returns the number of atoms whose coordinates were injected.
    """
    import numpy as np

    struct_dir = proc_out_dir / "processed" / "structures"
    npz_files = sorted(struct_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No structure npz in {struct_dir}")
    struct_path = npz_files[0]

    data = dict(np.load(str(struct_path), allow_pickle=True))
    atoms = data["atoms"].copy()
    residues = data["residues"].copy()
    chains = data["chains"]

    # -- Parse PDB: separate ATOM (protein) and HETATM (ligand) records ------
    pdb_protein_coords: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    pdb_hetatm_coords: dict[tuple[str, str], tuple[float, float, float]] = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                chain_letter = line[21]
                resseq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                pdb_protein_coords[(chain_letter, resseq, atom_name)] = (x, y, z)
            elif line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                pdb_hetatm_coords[(resname, atom_name)] = (x, y, z)

    chain_letters = []
    for c in chains:
        chain_letters.append(str(c["name"]).strip())

    def _decode_atom_name(raw_name) -> str:
        """Decode atom name from unicode string (boltz >=2.2) or byte array."""
        if isinstance(raw_name, (str, np.str_)):
            return str(raw_name).strip()
        # Legacy byte-array form (boltz 2.0.3): dtype 4i1
        if hasattr(raw_name, "tolist"):
            raw_name = raw_name.tolist()
        return "".join(chr(b) if b > 0 else "" for b in raw_name).strip()

    n_injected = 0
    n_total_atoms = 0
    unmatched_residues: list[str] = []
    # Track per-chain injection counts for the non-polymer guard.
    chain_injected: dict[int, int] = {}  # chain_idx -> count
    for chain_idx, chain in enumerate(chains):
        chain_letter = chain_letters[chain_idx]
        mol_type = int(chain["mol_type"])
        res_start = int(chain["res_idx"])
        res_count = int(chain["res_num"])
        chain_inj = 0

        for ri in range(res_start, res_start + res_count):
            res = residues[ri]
            atom_start = int(res["atom_idx"])
            atom_count = int(res["atom_num"])
            res_matched = False

            # Protein: (chain_letter, chain_relative_index + offset, atom_name)
            if mol_type == 0:
                res_idx_within_chain = ri - res_start
                pdb_resnum = res_idx_within_chain + resnum_offset

            # Ligand / non-polymer: match HETATM by (resname, atom_name)
            res_name = str(res["name"]).strip() if mol_type != 0 else None

            for ai in range(atom_start, atom_start + atom_count):
                n_total_atoms += 1
                aname = _decode_atom_name(atoms[ai]["name"])

                if mol_type == 0:
                    key = (chain_letter, pdb_resnum, aname)
                    coords = pdb_protein_coords.get(key)
                else:
                    hkey = (res_name, aname)
                    coords = pdb_hetatm_coords.get(hkey)

                if coords is not None:
                    atoms[ai]["coords"] = np.array(coords, dtype=np.float32)
                    atoms[ai]["is_present"] = True
                    n_injected += 1
                    chain_inj += 1
                    res_matched = True

            if not res_matched and atom_count > 0:
                label = (
                    f"{chain_letter}:{pdb_resnum}"
                    if mol_type == 0
                    else f"{chain_letter}:{res_name}"
                )
                unmatched_residues.append(label)
                residues[ri]["is_present"] = False

        chain_injected[chain_idx] = chain_inj

    # -- Non-polymer zero-coordinate guard ----------------------------------
    # A ligand with zero injected atoms stays at the origin, making the
    # affinity crop select residues near an arbitrary point.  This is
    # silent downstream and must be caught here.
    for chain_idx, chain in enumerate(chains):
        mol_type = int(chain["mol_type"])
        n_chain_atoms = int(chain["atom_num"])
        if mol_type != 0 and n_chain_atoms > 0 and chain_injected[chain_idx] == 0:
            chain_letter = chain_letters[chain_idx]
            res_start = int(chain["res_idx"])
            res_names = [
                str(residues[ri]["name"]).strip()
                for ri in range(res_start, res_start + int(chain["res_num"]))
            ]
            raise RuntimeError(
                f"Non-polymer chain {chain_letter} (residues: {res_names}, "
                f"{n_chain_atoms} atoms) received ZERO injected coordinates. "
                f"The ligand would remain at the origin, causing the affinity "
                f"crop to select residues near an arbitrary point instead of "
                f"the binding pocket. This makes the run meaningless. "
                f"Check that the PDB HETATM residue name matches the "
                f"processed structure's residue name (use --ligand-ccd with "
                f"the CCD code matching the PDB, not --ligand-smiles)."
            )

    # -- Sync the top-level ``coords`` array (derived from atoms) ------------
    if "coords" in data:
        coords_arr = data["coords"].copy()
        coords_arr["coords"] = atoms["coords"]
        data["coords"] = coords_arr

    data["atoms"] = atoms
    data["residues"] = residues
    np.savez_compressed(str(struct_path), **data)

    # -- Write pre_affinity file for the affinity dataset path ---------------
    # Boltz's affinity PredictionDataset (inferencev2.py:62-65) loads from
    # <structures>/<record_id>/pre_affinity_<record_id>.npz.  Normally the
    # writer (writer.py:178-180) creates this from the top-ranked diffusion
    # sample.  We synthesise it with the injected experimental coordinates
    # because this repo uses fixed wild-type geometry, not predicted coords.
    record_id = struct_path.stem  # e.g. "input" from "input.npz"
    pre_aff_dir = struct_dir / record_id
    pre_aff_dir.mkdir(parents=True, exist_ok=True)
    pre_aff_path = pre_aff_dir / f"pre_affinity_{record_id}.npz"
    np.savez_compressed(str(pre_aff_path), **data)
    log.info("Wrote pre_affinity file: %s", pre_aff_path)

    frac_matched = n_injected / n_total_atoms if n_total_atoms > 0 else 0.0
    log.info(
        "Coordinate injection: %d/%d atoms matched (%.1f%%), "
        "%d unmatched residues: %s",
        n_injected, n_total_atoms, frac_matched * 100,
        len(unmatched_residues),
        ", ".join(unmatched_residues[:10]) if unmatched_residues else "none",
    )

    if n_total_atoms > 0 and frac_matched < 0.5:
        raise RuntimeError(
            f"Coordinate injection matched only {n_injected}/{n_total_atoms} "
            f"atoms ({frac_matched:.1%}). The PDB chain letters or residue "
            f"numbering likely do not match the processed structure. "
            f"Unmatched residues: {', '.join(unmatched_residues[:10])}"
        )

    return n_injected


def _verify_token_map_identity(
    token_map: dict[tuple[str, int], int],
    pdb_path: Path,
    protein_chain_asym_str: str,
    protein_sequence: str,
    resnum_offset: int,
    pdb_chain: str = "A",
) -> None:
    """Assert the token map's residue numbers match the PDB's amino acids.

    For every protein token in the map, verifies that the residue number
    (with the applied offset) corresponds to the correct amino acid in
    both the construct sequence and the PDB. Raises on any mismatch,
    converting the entire class of offset/isoform/gap errors from
    silent-wrong into a crash.
    """
    _THREE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    pdb_residues: dict[int, str] = {}
    with open(pdb_path) as f:
        seen: set[int] = set()
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != pdb_chain:
                continue
            resseq = int(line[22:26].strip())
            if resseq in seen:
                continue
            seen.add(resseq)
            resname = line[17:20].strip()
            pdb_residues[resseq] = _THREE.get(resname, "X")

    mismatches: list[str] = []
    checked = 0
    for (chain_str, resnum), _tok_idx in token_map.items():
        if chain_str != protein_chain_asym_str:
            continue
        seq_idx = resnum - resnum_offset
        if seq_idx < 0 or seq_idx >= len(protein_sequence):
            mismatches.append(
                f"resnum {resnum} (index {seq_idx}) out of range for "
                f"sequence length {len(protein_sequence)}"
            )
            continue
        construct_aa = protein_sequence[seq_idx]
        pdb_aa = pdb_residues.get(resnum)
        if pdb_aa is not None and pdb_aa != construct_aa:
            mismatches.append(
                f"resnum {resnum}: construct has {construct_aa}, "
                f"PDB has {pdb_aa}"
            )
        checked += 1

    if mismatches:
        sample = "; ".join(mismatches[:5])
        raise RuntimeError(
            f"Token-map identity check FAILED: {len(mismatches)} mismatch(es) "
            f"out of {checked} checked. First: {sample}. "
            f"The resnum_offset ({resnum_offset}) or the construct sequence "
            f"does not match the PDB."
        )
    log.info(
        "Token-map identity check passed: %d protein residues verified "
        "against %s chain %s", checked, pdb_path, pdb_chain,
    )


def build_affinity_feats(
    protein_chains: dict[str, str],
    ligand_id: str,
    cache_dir,
    device,
    ligand_smiles: str | None = None,
    ligand_ccd: str | None = None,
    use_msa_server: bool = False,
    msa: str | dict | None = None,
    feat_seed: int | None = None,
    structure_pdb: "Path | str | None" = None,
    resnum_offset: int = 0,
):
    """Featurise a protein-ligand complex for affinity prediction.

    Like :func:`build_complex_feats` but emits the ``properties: affinity``
    YAML block and passes ``affinity=True`` to boltz's ``PredictionDataset``,
    which triggers the affinity cropper (``max_tokens=256``,
    ``max_tokens_protein=200``, ``neighborhood_size=10``) and MSA depth 1.

    Returns ``(feats, token_map)`` where *token_map* maps
    ``(chain_id, residue_number) -> crop_local_token_index``.

    Parameters
    ----------
    structure_pdb : Path or str or None
        PDB file with real atom coordinates for the protein-ligand complex.
        When provided, the coordinates are injected into the processed
        structure AFTER ``process_inputs`` and BEFORE the
        ``PredictionDataset`` creates the affinity crop.  This makes the
        ``AffinityCropper`` pocket-selective (by distance to the ligand)
        rather than geometry-blind (first ~200 tokens in sequence order).
        Without this, sequence-only YAML sets every coordinate to (0,0,0)
        and the crop is effectively random.
    resnum_offset : int
        Added to the 0-based ``res_idx`` for the protein chain in the
        token map.  For ABL1 with a kinase-domain construct starting at
        W235 (2HYY), pass ``resnum_offset=235`` so that the token map keys
        are P00519 residue numbers (e.g. 315 for T315).  The construct
        must be contiguous (no gaps); fill any PDB gaps from UniProt.

    Crop determinism
    ~~~~~~~~~~~~~~~~
    ``AffinityCropper`` selects tokens by ascending distance to the ligand,
    computed from ``center_coords`` in the tokenized structure -- the
    reference PDB coordinates, not ``x_pred``.  There is no RNG call.
    For a point mutation that changes only ``res_type`` (same PDB geometry),
    the crop selects the identical residue set, so the embedding delta
    ``E_mut - E_ref`` is taken in the same index space.

    Ligand tokens share the same index space as protein tokens (both in
    ``token_data["token_idx"]``).  After cropping, ``x_pred`` and the
    distance bucketisation in the affinity module address the same
    crop-local indices.
    """
    import yaml
    from boltz.main import process_inputs

    from igv.deterministic import deterministic_featurisation

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved_msa_val = resolve_msa_spec(msa)
    spec = build_affinity_yaml_spec(
        protein_chains,
        ligand_id,
        ligand_smiles=ligand_smiles,
        ligand_ccd=ligand_ccd,
        resolved_msa=resolved_msa_val,
    )
    spec_path = cache_dir / "input.yaml"
    spec_path.write_text(yaml.dump(spec))

    proc_kwargs, cache_root = _boltz_process_inputs_kwargs(use_msa_server)

    with deterministic_featurisation(seed=feat_seed):
        process_inputs(data=[spec_path], out_dir=cache_dir, **proc_kwargs)

        if structure_pdb is not None:
            n_inj = _inject_pdb_coords_into_processed_structure(
                cache_dir, Path(structure_pdb),
                resnum_offset=resnum_offset,
            )
            log.info("Coordinate injection: %d atoms patched", n_inj)
            if n_inj == 0:
                log.warning(
                    "No coordinates were injected from %s. The crop will be "
                    "geometry-blind. Check that the PDB chain letters and "
                    "residue numbering match the processed structure.",
                    structure_pdb,
                )

        mol_dir = proc_kwargs.get("mol_dir", cache_root / "mols")
        feats = _featurize_processed_affinity(cache_dir, mol_dir, device)

    n_pad = int(feats["token_pad_mask"][0].sum().item())
    log.info("Affinity feats: %d non-padding tokens (after crop)", n_pad)

    token_map = _build_affinity_token_map(
        feats,
        protein_chain_asym=0,
        resnum_offset=resnum_offset,
    )
    log.info("Affinity token map: %d entries (resnum_offset=%d)", len(token_map), resnum_offset)

    if structure_pdb is not None and resnum_offset != 0:
        _verify_token_map_identity(
            token_map, Path(structure_pdb),
            protein_chain_asym_str="0",
            protein_sequence=next(iter(protein_chains.values())),
            resnum_offset=resnum_offset,
        )

    return feats, token_map


def _build_affinity_token_map(
    feats: dict,
    protein_chain_asym: int = 0,
    resnum_offset: int = 0,
) -> dict[tuple[str, int], int]:
    """Build ``(chain_id, residue_number) -> crop_local_token_index``.

    After the affinity cropper, the feats contain fewer tokens than the
    original protein chain.  ``feats["residue_index"]`` carries the
    **original** ``res_idx`` for each surviving token, and
    ``feats["asym_id"]`` identifies the chain.  Together they give a
    complete map from the original sequence numbering to the crop-local
    index space.

    Chain ids are integers (``asym_id``), not the string ids from the YAML.
    The map uses integer chain ids; callers that need string ids must map
    through the YAML's chain ordering (``asym_id`` 0 = first chain in YAML,
    1 = second, etc.).

    Parameters
    ----------
    feats : dict
        Collated feature dict.
    protein_chain_asym : int
        The asym_id of the protein chain to apply the offset to.
    resnum_offset : int
        Added to the 0-based ``res_idx`` for the protein chain to produce
        the residue numbers stored in the token map. Use this to emit
        PDB author numbering (e.g. ABL1 1b) rather than 0-based indices.
        Non-protein chains are not offset.
    """
    pad = feats["token_pad_mask"][0].bool().cpu()
    asym = feats["asym_id"][0].cpu()[pad]
    res_idx = feats["residue_index"][0].cpu()[pad]

    token_map: dict[tuple[str, int], int] = {}
    for local_idx in range(len(asym)):
        chain = int(asym[local_idx])
        resi = int(res_idx[local_idx])
        if chain == protein_chain_asym and resnum_offset != 0:
            resi = resi + resnum_offset
        key = (str(chain), resi)
        if key not in token_map:
            token_map[key] = local_idx
    return token_map


def check_crop_coverage(
    token_map: dict[tuple[str, int], int],
    chain_id: str,
    sites: list[int],
) -> dict:
    """Check which mutation sites survived the affinity crop.

    Pure function, no torch.

    Parameters
    ----------
    token_map : dict[(chain_id, residue_number), token_index]
        From :func:`_build_affinity_token_map` or :func:`_build_token_map`.
    chain_id : str
        The chain to check (string id or ``str(asym_id)``).
    sites : list[int]
        Residue numbers in the same numbering space as the token map keys.
        When ``resnum_offset`` was used to build the token map, sites must
        include that offset (e.g. ``range(235, 235 + n_protein)`` for ABL1).

    Returns
    -------
    dict
        ``n_requested``, ``n_surviving``, ``n_cropped``,
        ``surviving`` (list of residue indices that survived),
        ``cropped`` (list of residue indices that were cropped out),
        ``surviving_token_indices`` (crop-local token indices),
        ``fraction_surviving``.
    """
    surviving = []
    cropped = []
    surviving_token_indices = []

    for site in sites:
        key = (chain_id, site)
        if key in token_map:
            surviving.append(site)
            surviving_token_indices.append(token_map[key])
        else:
            cropped.append(site)

    n_req = len(sites)
    n_surv = len(surviving)
    return {
        "n_requested": n_req,
        "n_surviving": n_surv,
        "n_cropped": n_req - n_surv,
        "surviving": surviving,
        "cropped": cropped,
        "surviving_token_indices": surviving_token_indices,
        "fraction_surviving": n_surv / n_req if n_req > 0 else 0.0,
    }


def build_mean_aa_res_type_baseline(res_type, feats):
    """Mean-AA baseline for the res_type one-hot in the affinity path.

    At protein positions: uniform 1/20 over the 20 canonical amino acid columns.
    At non-protein positions (ligand, padding): keep the original res_type
    so that attribution there is zero by construction.

    The baseline operates on the CROPPED token set: after AffinityCropper, the
    token indices are crop-local, and this function addresses them in that space.
    """
    import torch

    aa_indices = canonical_aa_token_indices()
    baseline = res_type.detach().clone().float()

    pad_mask = feats["token_pad_mask"][0].bool()
    mol_type = feats.get("mol_type")
    if mol_type is not None:
        protein_mask = (mol_type[0] == 0) & pad_mask
    else:
        protein_mask = pad_mask

    num_tokens = res_type.shape[-1]
    uniform = torch.zeros(num_tokens, device=res_type.device, dtype=baseline.dtype)
    for idx in aa_indices:
        uniform[int(idx)] = 1.0 / len(aa_indices)

    baseline[0, protein_mask] = uniform
    return baseline


# UNK token index, verified against boltz 2.2.1 const.tokens:
#   tokens[22] == "UNK" == the unknown-protein token.
# Boltz maps single-letter X -> three-letter UNK -> index 22.
UNK_TOKEN_INDEX = 22


def _verify_unk_token_index() -> int:
    """Return the UNK protein token index, verifying against the pinned table."""
    idx = list(_BOLTZ_TOKENS_2_2_1).index("UNK")
    if idx != UNK_TOKEN_INDEX:
        raise RuntimeError(
            f"Pinned UNK index {UNK_TOKEN_INDEX} disagrees with "
            f"_BOLTZ_TOKENS_2_2_1 lookup {idx}"
        )
    return idx


def build_unk_res_type_baseline(res_type, feats):
    """UNK-token baseline: set every protein position to the unknown-residue
    one-hot (index 22 in boltz 2.2.1). Ligand and padding rows are unchanged.
    """
    import torch

    unk_idx = _verify_unk_token_index()
    baseline = res_type.detach().clone().float()

    pad_mask = feats["token_pad_mask"][0].bool()
    mol_type = feats.get("mol_type")
    if mol_type is not None:
        protein_mask = (mol_type[0] == 0) & pad_mask
    else:
        protein_mask = pad_mask

    num_tokens = res_type.shape[-1]
    unk_onehot = torch.zeros(num_tokens, device=res_type.device, dtype=baseline.dtype)
    unk_onehot[unk_idx] = 1.0
    baseline[0, protein_mask] = unk_onehot
    return baseline


def build_shuffled_res_type_baseline(res_type, feats, seed=0):
    """Composition-matched shuffled-sequence baseline.

    Permutes the protein chain's res_type rows among themselves (protein tokens
    only -- ligand and padding rows are untouched). Preserves composition
    exactly because it is a permutation, not a resample.

    The shuffle is seeded for reproducibility.
    """
    import numpy as np
    import torch

    baseline = res_type.detach().clone().float()

    pad_mask = feats["token_pad_mask"][0].bool()
    mol_type = feats.get("mol_type")
    if mol_type is not None:
        protein_mask = (mol_type[0] == 0) & pad_mask
    else:
        protein_mask = pad_mask

    protein_indices = protein_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    if len(protein_indices) < 2:
        return baseline

    protein_rows = baseline[0, protein_indices].clone()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(protein_indices))
    baseline[0, protein_indices] = protein_rows[torch.from_numpy(perm)]
    return baseline


def _featurize_processed_affinity(proc_out_dir: Path, mol_dir: Path, device) -> dict:
    """Like :func:`_featurize_processed` but with ``affinity=True``."""
    import torch
    from boltz.data.types import Manifest

    manifest_path = proc_out_dir / "processed" / "manifest.json"
    manifest = Manifest.load(manifest_path)
    processed = proc_out_dir / "processed"
    constraints_dir = processed / "constraints"
    templates_dir = processed / "templates"
    extra_mols_dir = processed / "mols"

    from boltz.data.module.inferencev2 import PredictionDataset, collate
    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=mol_dir,
        constraints_dir=constraints_dir if constraints_dir.exists() else None,
        template_dir=templates_dir if templates_dir.exists() else None,
        extra_mols_dir=extra_mols_dir if extra_mols_dir.exists() else None,
        affinity=True,
    )

    sample = dataset[0]
    if sample is None:
        raise RuntimeError("Boltz PredictionDataset returned None for affinity")
    sample.pop("record", None)
    feats = collate([sample])
    for key, val in feats.items():
        if key in _FEATS_SKIP_DEVICE:
            continue
        if isinstance(val, torch.Tensor):
            feats[key] = val.to(device)
    return feats


def build_complex_feats(
    chains: dict[str, str],
    structure_pdb: Path,
    cache_dir: Path,
    device,
    use_msa_server: bool = True,
    msa: str | dict | None = None,
    msa_spec: str | dict | None = None,
    feat_seed: int | None = None,
) -> tuple[dict, dict[tuple[str, int], int]]:
    """Featurise a multi-chain complex for Boltz-2.

    Parameters
    ----------
    chains : dict[str, str]
        Maps chain id to amino-acid sequence, e.g.
        ``{"H": "EVQL...", "L": "DIQM...", "A": "MKYL..."}``.
    structure_pdb : Path
        ACCEPTED BUT NEVER READ. It is kept because four call sites pass it
        positionally, but ``grep -n structure_pdb`` over this module finds only
        the parameter and its docstring: features come solely from the
        sequences-only YAML written below. Consequence for the reader: the
        "GEOMETRY IS FIXED ... Boltz featurisation places the wild-type atom
        coordinates into feats['coords']" claim logged by scripts/03_attribute.py
        and scripts/04_scan.py is NOT sourced from the deposited PDB. boltz's
        schema parser sets every parsed atom's coords to (0,0,0) for
        sequence-only YAML input (boltz 2.2.1 parse/schema.py:733, :880), so
        ``x_pred`` is very likely all zeros. NOT changed here -- this function
        just logs ``feats["coords"].abs().max()`` at INFO so the next GPU run
        settles it without new code.
    cache_dir : Path
        Scratch directory for Boltz intermediate files. MUST BE UNIQUE PER
        SEQUENCE: boltz's ``process_inputs`` skips any input whose YAML stem is
        already in ``<cache_dir>/processed/records`` (boltz 2.2.1
        main.py:724-742) and this function always writes the stem "input", so a
        reused directory silently returns the FIRST sequence's features. The
        sequence check below (``IGV_ASSERT_FEAT_SEQ``, default ON) catches it.
    device : torch.device
        Target device.
    use_msa_server : bool
        Whether to query ColabFold for MSAs. With explicit per-chain MSA file
        paths (a dict), ``use_msa_server=False`` works: every chain has an
        ``msa`` key pointing at a local file, so boltz never attempts a server
        query and the "Missing MSA's in input" error does not fire.
    msa : str | dict | None
        Per-chain ``msa`` entry for the YAML. Default None emits NO ``msa`` key,
        which is today's exact behaviour. Boltz reads it as
        ``msa = items[0][entity_type].get("msa", 0)``; ``"empty"`` becomes -1 =
        single-sequence mode, and any other non-zero value is treated as a path
        to a custom .a3m/.csv (boltz 2.2.1 parse/schema.py:1112-1140).
        Overridable per-run with ``IGV_MSA_SPEC`` so a sweep can force
        ``IGV_MSA_SPEC=empty`` without editing call sites. ``"server"`` is
        accepted as a synonym for None (emit no key and let the MSA server
        generate one).

        A ``dict[str, str | Path]`` maps each chain id to its own MSA file
        path, emitting a per-chain ``msa`` key. The dict must cover every
        chain; every mapped path must exist (a missing path would silently fall
        back to the server, which is the failure mode this is meant to
        prevent). The intended use is reusing the wild-type MSA for every point
        mutant: the MSA captures evolutionary context that should be held
        constant so the embedding delta isolates the substitution, not a
        different MSA.
    msa_spec : str | dict | None
        Alias of *msa*, accepted because the memory sweep in
        scripts/08_memscale.py resolves this parameter by name. Passing both
        with different values raises.

        Two things to know before using it. (1) With no ``msa`` key a protein
        chain gets ``msa_id == 0`` meaning auto-generate, and then
        ``use_msa_server=False`` RAISES ("Missing MSA's in input and
        --use_msa_server flag not set", boltz 2.2.1 main.py:565-583) -- which is
        exactly what makes offline featurisation impossible today. With explicit
        per-chain paths this is avoided because every chain has an ``msa`` key.
        (2) ``msa="empty"`` changes the memory profile of the 4 MSA blocks
        (the 4 MSA blocks cost ~3.25 GiB under ``_msa_forward_checkpointed``), so
        a reference point measured WITH MSAs is not comparable to one measured
        without. Any artifact must record which was used.
    feat_seed : int or None
        Seed for deterministic featurisation.  ``None`` (the default) reads
        ``IGV_FEAT_SEED`` from the environment, falling back to 42.  Wraps
        the entire featurisation (``process_inputs`` + dataset collation)
        in :func:`igv.deterministic.deterministic_featurisation`, which:

        1. Patches ``center_random_augmentation`` to centre-only (no random
           rotation/translation), eliminating the dominant source of
           ``ref_pos`` nondeterminism.
        2. Seeds torch, numpy, Python ``random`` to cover
           ``random.choice(conf_ids)`` and any other RNG boltz draws from.
        3. Patches RDKit's ``EmbedMolecule`` to inject a fixed
           ``randomSeed`` for ligand/non-standard residue conformer
           generation.

        This makes ``ref_pos`` byte-identical across repeated featurisations
        of the same input AND across a reference and point mutant at their
        shared residues, eliminating the conformer-resample noise from
        embedding deltas.

    Returns
    -------
    feats : dict[str, Tensor]
        Collated feature dict for ``model.forward()`` / ``confidence_module``.
    token_map : dict[(chain_id, residue_index), token_index]
        Maps ``(chain_id, 0-based residue index)`` to the global token index
        in the feats tensors.
    """
    import yaml
    from boltz.main import process_inputs

    from igv.deterministic import deterministic_featurisation

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    effective_msa = msa if msa is not None else msa_spec
    if msa is not None and msa_spec is not None:
        if isinstance(msa, dict) or isinstance(msa_spec, dict):
            if msa is not msa_spec:
                raise ValueError(
                    f"build_complex_feats got conflicting msa={msa!r} and "
                    f"msa_spec={msa_spec!r}; they are aliases, pass one."
                )
        elif str(msa) != str(msa_spec):
            raise ValueError(
                f"build_complex_feats got conflicting msa={msa!r} and "
                f"msa_spec={msa_spec!r}; they are aliases, pass one."
            )
    resolved_msa = resolve_msa_spec(effective_msa)
    sequences = _build_yaml_sequences(chains, resolved_msa)
    if resolved_msa is not None:
        log.info("YAML msa spec: %r (default None emits no msa key)", resolved_msa)

    spec = {"version": 1, "sequences": sequences}
    spec_path = cache_dir / "input.yaml"
    spec_path.write_text(yaml.dump(spec))

    proc_kwargs, cache_root = _boltz_process_inputs_kwargs(use_msa_server)

    with deterministic_featurisation(seed=feat_seed):
        process_inputs(data=[spec_path], out_dir=cache_dir, **proc_kwargs)

        mol_dir = proc_kwargs.get("mol_dir", cache_root / "mols")
        feats = _featurize_processed(cache_dir, mol_dir, device)

    token_map = _build_token_map(chains, feats)

    expected_tokens = sum(len(s) for s in chains.values())
    n_pad = int(feats["token_pad_mask"][0].sum().item())
    assert n_pad == expected_tokens, (
        f"Token count mismatch: expected {expected_tokens} from chain "
        f"sequences, got {n_pad} non-padding tokens in feats"
    )

    # The count check above is invariant under a same-length point mutation, so
    # it cannot see a stale cache. Compare the actual residues.
    _check_featurised_sequences(chains, feats, token_map)

    coords = feats.get("coords")
    if coords is not None:
        try:
            log.info(
                "feats['coords'].abs().max() = %.6g (0.0 means the "
                "sequence-only YAML produced no geometry, i.e. x_pred is zeros)",
                float(coords.abs().max()),
            )
        except Exception as exc:  # instrumentation must never break a run
            log.debug("Could not summarise feats['coords']: %r", exc)

    return feats, token_map


def _boltz_process_inputs_kwargs(use_msa_server: bool) -> tuple[dict, "Path"]:
    import inspect
    import urllib.request
    from boltz.main import get_cache_path, process_inputs

    cache_root = Path(get_cache_path())
    proc_kwargs: dict = {"use_msa_server": use_msa_server}
    sig = inspect.signature(process_inputs)
    if "msa_server_url" in sig.parameters:
        proc_kwargs["msa_server_url"] = "https://api.colabfold.com"
    if "msa_pairing_strategy" in sig.parameters:
        proc_kwargs["msa_pairing_strategy"] = "greedy"
    if "ccd_path" in sig.parameters:
        ccd_path = cache_root / "ccd.pkl"
        if not ccd_path.exists():
            urllib.request.urlretrieve(
                "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl",
                ccd_path,
            )
        proc_kwargs["ccd_path"] = ccd_path
    if "mol_dir" in sig.parameters:
        proc_kwargs["mol_dir"] = cache_root / "mols"
    if "boltz2" in sig.parameters:
        proc_kwargs["boltz2"] = True
    return proc_kwargs, cache_root


_FEATS_SKIP_DEVICE = frozenset({
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
    "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
    "record", "affinity_mw",
})


def _featurize_processed(proc_out_dir: Path, mol_dir: Path, device) -> dict:
    import torch
    from boltz.data.types import Manifest

    manifest_path = proc_out_dir / "processed" / "manifest.json"
    manifest = Manifest.load(manifest_path)
    processed = proc_out_dir / "processed"
    constraints_dir = processed / "constraints"
    templates_dir = processed / "templates"
    extra_mols_dir = processed / "mols"

    try:
        from boltz.data.module.inferencev2 import PredictionDataset, collate
        dataset = PredictionDataset(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=mol_dir,
            constraints_dir=constraints_dir if constraints_dir.exists() else None,
            template_dir=templates_dir if templates_dir.exists() else None,
            extra_mols_dir=extra_mols_dir if extra_mols_dir.exists() else None,
            affinity=False,
        )
    except ImportError:
        from boltz.data.module.inference import PredictionDataset, collate
        dataset = PredictionDataset(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            constraints_dir=constraints_dir if constraints_dir.exists() else None,
        )

    sample = dataset[0]
    if sample is None:
        raise RuntimeError("Boltz PredictionDataset returned None")
    sample.pop("record", None)
    feats = collate([sample])
    for key, val in feats.items():
        if key in _FEATS_SKIP_DEVICE:
            continue
        if isinstance(val, torch.Tensor):
            feats[key] = val.to(device)
    return feats


def _build_token_map(
    chains: dict[str, str], feats: dict
) -> dict[tuple[str, int], int]:
    """Build ``(chain_id, residue_index) -> global token index``.

    The chain-to-token ordering is *derived* from ``feats["asym_id"]`` rather
    than assumed.  Boltz assigns each token an ``asym_id``; tokens of one chain
    are contiguous, so the run lengths of ``asym_id`` give the actual layout.

    This matters because the obvious assumption -- that token order follows the
    alphabetically sorted chain ids -- is not verifiable from the YAML we write,
    and the token *count* assertion passes under any ordering.  A mismatch would
    therefore misalign every per-residue attribution silently, with no error.
    For the 4fqi complex the two candidate orderings differ (sorted gives
    A=324, B=176, H=121, L=109; insertion order gives H, L, A, B), so comparing
    run lengths against the expected chain lengths detects the wrong guess.

    Raises
    ------
    RuntimeError
        If the number of chain runs does not match the number of chains, or if
        no ordering of the supplied chains reproduces the observed run lengths.
    """
    pad = feats["token_pad_mask"][0].bool().cpu()
    asym = feats["asym_id"][0].cpu()[pad]

    # Contiguous runs of equal asym_id == one chain each.
    runs: list[tuple[int, int, int]] = []  # (asym_value, start, length)
    run_start = 0
    for i in range(1, len(asym) + 1):
        if i == len(asym) or int(asym[i]) != int(asym[run_start]):
            runs.append((int(asym[run_start]), run_start, i - run_start))
            run_start = i

    if len(runs) != len(chains):
        raise RuntimeError(
            f"Found {len(runs)} contiguous asym_id runs but {len(chains)} chains "
            f"were supplied: run lengths {[r[2] for r in runs]}, chain lengths "
            f"{ {c: len(s) for c, s in chains.items()} }. Cannot map tokens to "
            "chains; attribution slicing would be wrong."
        )

    observed = [r[2] for r in runs]

    # Try the two plausible orderings, then fall back to a length-multiset match.
    candidates = [sorted(chains.keys()), list(chains.keys())]
    order: list[str] | None = None
    for cand in candidates:
        if [len(chains[c]) for c in cand] == observed:
            order = cand
            break
    if order is None:
        # Unique-length chains can still be matched unambiguously by length.
        by_len: dict[int, list[str]] = {}
        for c, s in chains.items():
            by_len.setdefault(len(s), []).append(c)
        if all(len(by_len.get(n, [])) == 1 for n in observed):
            order = [by_len[n][0] for n in observed]
        else:
            raise RuntimeError(
                "Could not determine chain order from asym_id runs. Observed run "
                f"lengths {observed}; chain lengths "
                f"{ {c: len(s) for c, s in chains.items()} }. Two chains share a "
                "length and neither sorted nor insertion order matches, so the "
                "mapping is ambiguous."
            )

    token_map: dict[tuple[str, int], int] = {}
    for chain_id, (_asym_val, offset, length) in zip(order, runs):
        seq = chains[chain_id]
        if length != len(seq):
            raise RuntimeError(
                f"Chain {chain_id} has {len(seq)} residues but its token run is "
                f"{length} long."
            )
        for resi in range(length):
            token_map[(chain_id, resi)] = offset + resi
    return token_map


def _residue_letter_table() -> list[str]:
    """Token-id -> one-letter code, preferring the installed boltz tables.

    Falls back to the pinned boltz-2.2.1 copies above when boltz is not
    importable, so this is testable on a laptop.
    """
    try:
        from boltz.data import const as _c

        tokens = list(_c.tokens)
        token_to_letter = dict(_c.prot_token_to_letter)
    except Exception:
        tokens = list(_BOLTZ_TOKENS_2_2_1)
        token_to_letter = dict(_PROT_TOKEN_TO_LETTER_2_2_1)

    table = []
    for name in tokens:
        letter = token_to_letter.get(name)
        if letter is None:
            # RNA tokens are already single letters; DNA tokens and <pad> have
            # no protein letter, so they decode to "X".
            letter = name if len(name) == 1 else "X"
        table.append(letter)
    return table


def _canonical_letter(letter: str) -> str:
    """The one-letter code a requested residue will DECODE back to.

    Boltz round-trips a sequence letter through
    ``prot_letter_to_token`` -> ``prot_token_to_letter``, and that is lossy by
    design: B, J, Z, O, U and X all become UNK and decode back to "X". Comparing
    raw letters would therefore report a spurious mismatch on any sequence
    containing one, so the requested letter is canonicalised the same way.
    """
    try:
        from boltz.data import const as _c

        letter_to_token = dict(_c.prot_letter_to_token)
        token_to_letter = dict(_c.prot_token_to_letter)
    except Exception:
        letter_to_token = {v: k for k, v in _PROT_TOKEN_TO_LETTER_2_2_1.items()}
        letter_to_token.update(
            {"J": "UNK", "B": "UNK", "Z": "UNK", "O": "UNK", "U": "UNK"}
        )
        token_to_letter = dict(_PROT_TOKEN_TO_LETTER_2_2_1)

    token = letter_to_token.get(letter.upper())
    if token is None:
        return letter.upper()
    return token_to_letter.get(token, "X")


def _decode_residue_tokens(feats: dict) -> list[tuple[int, str]]:
    """Decode ``feats`` to ``[(asym_id, one_letter), ...]`` in token order.

    Padding positions are dropped, so the index into the returned list is the
    same global token index that :func:`_build_token_map` produces (it filters
    by ``token_pad_mask`` before measuring asym_id runs).

    ``res_type`` is a ONE-HOT tensor ``(1, L, num_tokens)`` -- boltz's
    featurizer stores ``one_hot(res_type.long(), num_classes=const.num_tokens)``
    -- so the token id is ``argmax`` over the last axis.
    """
    table = _residue_letter_table()

    res_type = feats["res_type"]  # KeyError here is the caller's bug, not ours
    ids = res_type[0].argmax(-1)
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)

    asym = feats["asym_id"][0]
    asym = asym.tolist() if hasattr(asym, "tolist") else list(asym)

    pad = feats.get("token_pad_mask")
    if pad is None:
        keep = [True] * len(ids)
    else:
        pad0 = pad[0]
        pad0 = pad0.tolist() if hasattr(pad0, "tolist") else list(pad0)
        keep = [bool(v) for v in pad0]

    out: list[tuple[int, str]] = []
    for tok_id, asym_val, ok in zip(ids, asym, keep):
        if not ok:
            continue
        idx = int(tok_id)
        letter = table[idx] if 0 <= idx < len(table) else "X"
        out.append((int(asym_val), letter))
    return out


def featurised_sequences(feats: dict) -> dict[int, str]:
    """Decode the residue sequence Boltz actually featurised, per chain.

    Returns ``{asym_id_value: one_letter_sequence}``, built only from
    ``feats["res_type"]``, ``feats["asym_id"]`` and ``feats["token_pad_mask"]``.
    Pure apart from an optional lazy boltz import for the token tables, so it
    runs on CPU with a synthetic feats dict and no boltz installed.

    This is what makes the stale-feature hazard checkable rather than assumed:
    the pre-existing guard in :func:`build_complex_feats` compares token COUNTS,
    which are invariant under a same-length point mutation -- and every mutant in
    the 4fqi H1 library is same-length by construction.
    """
    decoded = _decode_residue_tokens(feats)
    seqs: dict[int, list[str]] = {}
    for asym_val, letter in decoded:
        seqs.setdefault(asym_val, []).append(letter)
    return {k: "".join(v) for k, v in seqs.items()}


def _check_featurised_sequences(
    chains: dict[str, str], feats: dict, token_map: dict[tuple[str, int], int]
) -> None:
    """Verify that the featurised residues are the sequences we asked for.

    Uses the chain ordering ``_build_token_map`` already resolved from asym_id
    runs rather than re-deriving it. Raises ``RuntimeError`` (not ``assert``, so
    it survives ``python -O``) unconditionally on the first mismatch.

    Made unconditional after entry 34: the stale-cache hazard fired in
    production and was caught only by ``_build_token_map``'s chain-count guard.
    The sequence check is a pure Python string comparison (no GPU, no torch
    ops), so its cost is negligible. ``IGV_ASSERT_FEAT_SEQ`` is still read for
    backward compatibility but no longer downgrades the check -- a mismatch
    always raises.
    """
    if "res_type" not in feats or "asym_id" not in feats:
        log.warning(
            "Cannot verify featurised sequences: feats lacks res_type/asym_id."
        )
        return

    decoded = _decode_residue_tokens(feats)

    for chain_id, seq in chains.items():
        for resi, raw_expected in enumerate(seq):
            expected = _canonical_letter(raw_expected)
            key = (chain_id, resi)
            if key not in token_map:
                continue
            tok = token_map[key]
            if tok >= len(decoded):
                continue
            observed = decoded[tok][1]
            if observed == expected:
                continue
            expected_seq = "".join(_canonical_letter(c) for c in seq)
            observed_seq = "".join(
                decoded[token_map[(chain_id, i)]][1]
                for i in range(len(seq))
                if (chain_id, i) in token_map
                and token_map[(chain_id, i)] < len(decoded)
            )
            raise RuntimeError(
                f"Featurised sequence does not match the requested chains. "
                f"Chain {chain_id!r} differs first at residue index {resi}: "
                f"requested {expected!r}, featurised {observed!r} "
                f"(requested {expected_seq!r}, featurised {observed_seq!r}). "
                "MOST LIKELY CAUSE: boltz's process_inputs skips any input "
                "whose YAML stem is already present in "
                "<cache_dir>/processed/records, and this repo always writes the "
                "stem \"input\", so a cache_dir reused across different "
                "sequences returns the FIRST sequence's features. Pass a unique "
                "cache_dir per sequence."
            )


# ---------------------------------------------------------------------------
# Cache-key helpers
# ---------------------------------------------------------------------------


def _chain_cache_key(chains: dict[str, str]) -> str:
    """A short deterministic key encoding the identity of a chain set.

    Chain IDs with their lengths, sorted by chain ID, plus a digest of the
    sequences themselves -- e.g. ``"A107_B116_C129-3f9a1c7e"``.

    The digest is load-bearing, not decoration. Keying on IDs and lengths
    alone collides for two different complexes that happen to share chain
    labels and lengths, which is not far-fetched among antibody complexes of
    the same isotype. Such a collision is exactly entry 34's failure: a cache
    hit returning another protein's tensors. ``_check_featurised_sequences``
    would now catch it, but a directory key that cannot collide is better than
    one that relies on a downstream guard.

    Uses ``hashlib``, never the builtin ``hash()``: string hashing is salted
    per process, so ``hash()`` would produce a different directory name on
    every run and defeat caching entirely.
    """
    import hashlib

    lengths = "_".join(f"{cid}{len(seq)}" for cid, seq in sorted(chains.items()))
    payload = "|".join(f"{cid}:{seq}" for cid, seq in sorted(chains.items()))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{lengths}-{digest}"


# ---------------------------------------------------------------------------
# x_pred modes
# ---------------------------------------------------------------------------

X_PRED_MODES = ("zeros", "predicted")


def resolve_x_pred_mode(
    mode: str | None = None, env: Mapping[str, str] | None = None
) -> str:
    """Resolve the x_pred mode from an explicit value or ``IGV_X_PRED``.

    Returns ``"zeros"`` or ``"predicted"``.  Default is ``"zeros"`` so that
    existing callers are not silently changed.
    """
    if mode is None:
        env = os.environ if env is None else env
        mode = env.get("IGV_X_PRED", "zeros")
    mode = str(mode).strip().lower()
    if mode not in X_PRED_MODES:
        raise ValueError(
            f"x_pred mode {mode!r} is not recognised; "
            f"legal values are {list(X_PRED_MODES)}."
        )
    return mode


def _prediction_cache_key(chains: dict[str, str], dataset: str) -> str:
    """Cache key for a structure prediction, keyed on both dataset and chains.

    Uses the same ``_chain_cache_key`` digest as featurisation, plus the
    dataset label, so a cached prediction cannot be reused across complexes.
    """
    return f"{dataset}_{_chain_cache_key(chains)}"


def predict_structure_coords(
    model,
    feats: dict,
    cache_dir: Path,
    chains: dict[str, str],
    dataset: str,
    recycling_steps: int = 1,
    sampling_steps: int | None = None,
    diffusion_samples: int = 1,
):
    """Run boltz's structure prediction and return predicted atom coordinates.

    The confidence head was trained on the model's own predictions, not on
    crystal structures, so the on-manifold ``x_pred`` is the one the model
    itself produces.

    Returns a tensor of shape ``(1, N_atoms, 3)`` (squeezed from the
    diffusion output's multiplicity dimension when ``diffusion_samples=1``).

    Results are cached on disk at ``cache_dir / "predicted_coords" / key``,
    keyed on dataset identity and chain sequences, so re-runs do not
    re-predict.  The cache validates that the stored tensor has the expected
    atom count, matching ``feats["atom_pad_mask"]``.
    """
    import torch

    cache_key = _prediction_cache_key(chains, dataset)
    pred_dir = Path(cache_dir) / "predicted_coords" / cache_key
    cached_path = pred_dir / "x_pred.pt"

    expected_n_atoms = int(feats["atom_pad_mask"].shape[-1])

    if cached_path.exists():
        log.info("Loading cached predicted coords from %s", cached_path)
        x_pred = torch.load(cached_path, map_location=feats["coords"].device,
                            weights_only=True)
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

    log.info("Running structure prediction (this takes ~45s on A100)...")
    with torch.no_grad():
        out = model(
            feats,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            run_confidence_sequentially=False,
        )

    x_pred = out["sample_atom_coords"].detach()
    if x_pred.shape[0] > 1:
        x_pred = x_pred[:1]

    log.info(
        "Structure prediction complete: x_pred shape %s, abs max %.6g",
        list(x_pred.shape), float(x_pred.abs().max()),
    )

    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x_pred, cached_path)
    log.info("Cached predicted coords at %s", cached_path)

    return x_pred


# ---------------------------------------------------------------------------
# Forward functions
# ---------------------------------------------------------------------------


def embedder_only(model, feats):
    """Run only the input embedder under no_grad. Returns ``s_inputs`` (1, N, D)."""
    import torch
    with torch.no_grad():
        return model.input_embedder(feats)


def compute_homopolymer_embeddings(
    model,
    chains: dict[str, str],
    chain: str,
    structure_pdb: Path,
    cache_dir: Path,
    device,
    use_msa_server: bool = False,
    feat_seed: int | None = None,
) -> list:
    """Featurise 20 canonical homopolymer complexes and return per-AA embeddings.

    For each amino acid in ``CANONICAL_AMINO_ACIDS``, replaces the attributed
    *chain* with a homopolymer of that residue (same length) and runs the
    input embedder. Returns a list of 20 tensors, each shaped
    ``(len_chain, D)``, suitable for :func:`igv.attrib.build_mean_aa_baseline`.

    Uses ``msa="empty"`` for every homopolymer complex: a poly-alanine chain
    has no evolutionary profile, so an MSA query would be meaningless (and 20
    novel queries would be slow). This is a deliberate asymmetry with the real
    input, which may use the MSA server.

    Parameters
    ----------
    model
        Loaded Boltz-2 model (used for ``input_embedder``).
    chains : dict[str, str]
        Chain id -> sequence for the full complex.
    chain : str
        The attributed chain id (the one to replace with homopolymers).
    structure_pdb : Path
        Passed through to ``build_complex_feats`` (accepted but never read).
    cache_dir : Path
        Parent cache directory. Each homopolymer gets a unique subdirectory
        at ``cache_dir / "boltz_homopolymer" / aa`` to avoid the stale-cache
        hazard (boltz skips inputs whose YAML stem already exists).
    device
        Torch device.
    use_msa_server : bool
        Passed to ``build_complex_feats``. Defaults to False because
        ``msa="empty"`` makes the server unnecessary.
    feat_seed : int or None
        Seed for deterministic featurisation.

    Returns
    -------
    list[Tensor]
        20 tensors, one per canonical amino acid in the order of
        ``igv.attrib.CANONICAL_AMINO_ACIDS``. Each has shape
        ``(len(chains[chain]), D)``.
    """
    from igv.attrib import CANONICAL_AMINO_ACIDS

    chain_len = len(chains[chain])
    per_aa: list = []

    for aa in CANONICAL_AMINO_ACIDS:
        homo_chains = dict(chains)
        homo_chains[chain] = aa * chain_len

        # Key on what is actually featurised (homo_chains), NOT on the real
        # chains.  Keying on `chains` made the directory independent of which
        # chain was substituted, so a run varying chain A and a run varying
        # chain B collided on the same cache and the second silently received
        # the first's features.  Boltz's process_inputs skips any YAML stem
        # already in <cache_dir>/processed/records and this repo always writes
        # the stem "input", so a colliding directory returns stale features
        # rather than recomputing.  Cf. ERRORS_LOG entries 24, 34 and 40.
        aa_cache = (
            cache_dir / "boltz_homopolymer" / f"{aa}_{_chain_cache_key(homo_chains)}"
        )
        log.info("Homopolymer %s: featurising (%s)", aa, aa_cache)

        # msa="empty": a homopolymer has no evolutionary profile, so querying
        # the MSA server is meaningless. This is a deliberate asymmetry with
        # the real input. Do not "fix" this to use the server.
        feats, token_map = build_complex_feats(
            homo_chains,
            structure_pdb,
            aa_cache,
            device,
            use_msa_server=use_msa_server,
            msa="empty",
            feat_seed=feat_seed,
        )

        s = embedder_only(model, feats)

        token_indices = [token_map[(chain, i)] for i in range(chain_len)]
        emb = s[0, token_indices, :].detach()
        per_aa.append(emb)
        log.info("Homopolymer %s: embedding shape %s", aa, list(emb.shape))

        del feats, s

    return per_aa


def build_mean_aa_pair_baseline(
    model,
    chains: dict[str, str],
    structure_pdb,
    cache_dir,
    feats: dict,
    device,
    recycling_steps: int = 1,
    use_msa_server: bool = False,
    feat_seed: int | None = None,
):
    """Build a z baseline by running the trunk on the mean-AA s_inputs.

    Computes the 20 homopolymer embeddings (one per canonical amino acid),
    builds the mean-AA baseline in s_inputs space via
    :func:`igv.attrib.build_mean_aa_baseline`, then runs the trunk to produce
    the pair tensor ``z``.  This ``z`` is the baseline for pair-layer IG,
    matching section 12's spec: interpolate z between the mean_aa baseline's
    pair tensor and the wild-type's.
    """
    import numpy as np
    import torch

    from igv.attrib import build_mean_aa_baseline

    structure_pdb = Path(structure_pdb)
    cache_dir = Path(cache_dir)

    chain_id = next(iter(chains))
    chain_len = len(chains[chain_id])

    per_aa_embs = compute_homopolymer_embeddings(
        model, chains, chain_id, structure_pdb, cache_dir, device,
        use_msa_server=use_msa_server, feat_seed=feat_seed,
    )

    s_inputs_real = embedder_only(model, feats)

    token_indices_arr = np.arange(chain_len, dtype=np.int64)
    baseline_s = build_mean_aa_baseline(
        s_inputs_real, token_indices_arr, per_aa_embs,
    )

    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    with torch.no_grad():
        rel_pos = model.rel_pos(feats)
        token_bonds_z = model.token_bonds(feats["token_bonds"].float())
        contact_z = model.contact_conditioning(feats)

        s_init = model.s_init(baseline_s)
        z_init = (
            model.z_init_1(baseline_s)[:, :, None, :]
            + model.z_init_2(baseline_s)[:, None, :, :]
            + rel_pos + token_bonds_z + contact_z
        )

        s_ = torch.zeros_like(s_init)
        z_ = torch.zeros_like(z_init)

        for _ in range(recycling_steps + 1):
            s_ = s_init + model.s_recycle(model.s_norm(s_))
            z_ = z_init + model.z_recycle(model.z_norm(z_))
            z_ = z_ + model.msa_module(
                z_, baseline_s, feats, use_kernels=False,
            )
            s_, z_ = model.pairformer_module(
                s_, z_, mask=mask, pair_mask=pair_mask,
                use_kernels=False,
            )

    return z_.detach()


def enable_confidence_checkpointing(model) -> int:
    """Per-block gradient checkpointing inside the CONFIDENCE pairformer stack.

    The trunk (``pairformer_module`` / ``msa_module``) is checkpointed per block
    by :func:`confidence_forward`, but ``confidence_module.pairformer_stack`` was
    not, so during the outer checkpoint's backward recompute every one of its
    layers' L x L activations materialised at once. On a 730-token complex
    (4fqi: HA A=324 + B=176, Fab H=121 + L=109 = 730, as parsed by
    igv.data.read_pdb_chains from data/raw/4fqi_hlab.pdb; the "753" in
    an earlier debug log recorded "753" tokens from a misparsed chain split
    and is left as the historical record) that overruns an 80 GiB A100, OOMing at
    ``transition.py: silu(fc1(x)) * fc2(x)`` while trying to allocate 1.02 GiB
    with 78.43 GiB already held.

    This is the same failure the predecessor repo hit and fixed in the trunk --
    "checkpointed the whole pairformer_module as ONE unit, so backward recompute
    still materialized all 64 blocks' activations at once" -- just relocated to
    the one module that still fit at their smaller L~540.

    Boltz's PairformerModule already implements exactly this, but gates it on
    ``self.activation_checkpointing and self.training``. Calling ``.train()`` to
    unlock it is not an option: it enables dropout, which makes the score
    stochastic and its gradient meaningless, and it sets
    ``chunk_size_tri_attn = None``, removing the triangle-attention chunking
    that eval mode gives us. So we rebind ``forward`` instead, keeping the
    module in eval and replicating upstream's eval-mode chunk selection.

    Idempotent. Returns the number of layers now checkpointed.
    """
    import types

    from torch.utils.checkpoint import checkpoint as _ckpt

    confidence = getattr(model, "confidence_module", None)
    stack = getattr(confidence, "pairformer_stack", None)
    if stack is None:
        log.warning("No confidence_module.pairformer_stack; nothing to checkpoint.")
        return 0
    if getattr(stack, "_igv_checkpointed", False):
        return len(stack.layers)

    try:
        from boltz.data import const as _c
        _threshold = _c.chunk_size_threshold
    except Exception:
        _threshold = 384

    # use_kernels itself is NOT switched here: boltz's ConfidenceModule forwards
    # its own use_kernels into pairformer_stack by keyword and _checkpointed_forward
    # takes it as a parameter, so the value already flows in from the
    # model.confidence_module(...) call in confidence_forward. A second switch
    # here is how the two paths drift apart. use_cuequiv_attn is different: it is
    # NOT forwarded by PairformerModule.forward, so the only way to reach it is
    # this direct-layer call. Resolved once, at rebind time; the default is False.
    _, _use_cuequiv_attn = resolve_use_kernels(model)

    def _checkpointed_forward(self, s, z, mask, pair_mask, use_kernels: bool = False):
        # Mirrors PairformerModule.forward's eval-mode branch. The chunk size
        # used to be re-derived here as `128 if z.shape[1] > _threshold else 512`;
        # it now comes from the same pure function as the trunk's, so
        # IGV_CHUNK_PROFILE moves both branches together.
        chunk_size_tri_attn = select_chunk_profile(
            int(z.shape[1]), _threshold
        )["conf_chunk"]
        for layer in self.layers:
            s, z = _ckpt(
                layer, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels,
                False, _use_cuequiv_attn,
                use_reentrant=False,
            )
        return s, z

    stack.forward = types.MethodType(_checkpointed_forward, stack)
    stack._igv_checkpointed = True
    log.info(
        "Confidence pairformer stack: per-block checkpointing enabled (%d layers)",
        len(stack.layers),
    )
    return len(stack.layers)


def enable_triangle_attention_chunk_checkpointing(model) -> int:
    """Checkpoint each triangle-attention CHUNK so its softmax is not retained.

    This is the memory lever ``IGV_PF_CHUNK`` was believed to be and is not.
    Verified in the pinned boltz 2.2.1 source: ``chunk_layer``
    (``triangular_attention/utils.py:322-375``) is a plain
    ``for _ in range(no_chunks)`` loop calling ``layer(**chunks)`` at :339 with
    NO ``torch.no_grad``, NO ``detach`` and NO recompute, so EVERY chunk's
    softmax output is saved for backward and the retained total is
    chunk-size-INVARIANT (chunking is marginally worse: it also allocates the
    ``out`` buffer via ``t.new_zeros`` at :343).

    Wrapping the per-chunk ``self.mha`` call in a checkpoint changes that: the
    softmax is recomputed during backward instead of retained, so the retained
    cost becomes O(one chunk) at recompute time rather than O(L). The trade is
    one extra forward of triangle attention per chunk during backward -- pure
    compute for memory, with no numerical change (bit-identical fp32 recompute).

    Nesting, so nobody re-derives it: this puts a non-reentrant checkpoint inside
    the non-reentrant per-group pairformer checkpoint inside the reentrant OUTER
    checkpoint of :func:`confidence_forward`. Non-reentrant inside non-reentrant
    is torch's standard recursive case. The ``torch.is_grad_enabled()`` bypass in
    the wrapper is REQUIRED, not an optimisation: the outer checkpoint is
    ``use_reentrant=True``, which runs the whole wrapped function under
    ``torch.no_grad()``, so during the forward nothing requires grad and a
    non-reentrant checkpoint would emit "None of the inputs have
    requires_grad=True" once per chunk per attention per layer -- thousands of
    lines. All the savings come from the backward recompute, where grad IS
    enabled.

    Scope: ``model.modules()`` catches the trunk pairformer (64 layers), the
    confidence pairformer stack (8 layers) and the MSA module's triangle
    attention -- two ``TriangleAttention`` instances per pairformer layer
    (start + end). No narrower selector until a measurement asks for one.

    Like :func:`enable_confidence_checkpointing` this is a ONE-WAY monkeypatch of
    boltz internals: it rebinds a bound method and is never undone, so a later
    call with the feature disabled does not get the plain implementation back on
    the same model object. Idempotent via the ``_igv_chunk_ckpt`` sentinel. We
    stay in eval mode throughout and never call ``.train()``.

    Returns the number of ``TriangleAttention`` modules newly wrapped.

    NEEDS GPU CONFIRMATION -- what to measure, stated so it can be falsified: at
    IGV_PF_GROUP_SIZE=1 the softmax term should drop from the 11.59 GiB
    ERRORS_LOG entry 12 attributes to primitives.py:170 toward ~1.3 GiB per layer
    (retained per-attention outputs ~0.254 GiB x2 plus one live ~1.02 GiB chunk
    transient during recompute), i.e. roughly 77.64 -> ~67 GiB peak, and
    IGV_PF_CHUNK should for the first time visibly move the peak. If the measured
    peak does NOT drop by ~10 GiB, suspect that the rebind did not take effect --
    check the count returned here is non-zero (expect 64*2 + 8*2 + the MSA
    module's attentions).
    """
    import types

    import torch
    from torch.utils.checkpoint import checkpoint as _ckpt

    from boltz.model.layers.triangular_attention.attention import TriangleAttention
    from boltz.model.layers.triangular_attention.utils import chunk_layer

    def _make_chunk(module):
        def _chunk(self, x, tri_bias, mask_bias, mask, chunk_size,
                   use_kernels: bool = False):
            # Reproduces TriangleAttention._chunk exactly except for the
            # checkpoint. chunk_layer calls layer(**chunks), so the wrapper must
            # accept precisely these five keyword names.
            def _ckpt_mha(q_x, kv_x, tri_bias, mask_bias, mask):
                if not torch.is_grad_enabled():
                    return self.mha(q_x, kv_x, tri_bias, mask_bias, mask, use_kernels)
                return _ckpt(
                    self.mha, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels,
                    use_reentrant=False,
                )

            mha_inputs = {
                "q_x": x,
                "kv_x": x,
                "tri_bias": tri_bias,
                "mask_bias": mask_bias,
                "mask": mask,
            }
            return chunk_layer(
                _ckpt_mha,
                mha_inputs,
                chunk_size=chunk_size,
                no_batch_dims=len(x.shape[:-2]),
                _out=None,
            )

        return _chunk

    wrapped = 0
    for module in model.modules():
        if not isinstance(module, TriangleAttention):
            continue
        if getattr(module, "_igv_chunk_ckpt", False):
            continue
        module._chunk = types.MethodType(_make_chunk(module), module)
        module._igv_chunk_ckpt = True
        wrapped += 1

    log.info(
        "Triangle-attention per-chunk checkpointing: %d newly wrapped, %d already "
        "instrumented (one-way rebind; softmax is recomputed in backward, not "
        "retained)",
        wrapped,
        count_tri_attn_chunk_ckpt(model) - wrapped,
    )
    return wrapped


def count_tri_attn_chunk_ckpt(model) -> int:
    """How many ``TriangleAttention`` modules currently carry the chunk rebind.

    The counterpart to
    :func:`enable_triangle_attention_chunk_checkpointing`'s return value, which
    counts only what *this* call wrapped. Reusing one model object -- which
    ``08_memscale`` does across its whole ladder and ``07_sanity`` does across
    ``signal_control``'s mutants -- makes that return 0 while the lever is
    fully active, so "is it on?" has to be asked of the model, not of the
    last call.
    """
    from boltz.model.layers.triangular_attention.attention import TriangleAttention

    return sum(
        1
        for module in model.modules()
        if isinstance(module, TriangleAttention) and getattr(module, "_igv_chunk_ckpt", False)
    )


def confidence_forward(
    model,
    s_inputs,
    feats,
    x_pred,
    score_name: str,
    gradient_checkpointing: bool = True,
    recycling_steps: int = 1,
    autocast_dtype: str | None = None,
    triangle_attention_chunk_checkpointing: bool | None = None,
    activation_callback=None,
):
    """Run the full trunk from ``s_inputs``, then the confidence head.

    Returns the requested scalar score as a differentiable tensor.

    The trunk is run with per-block gradient checkpointing to fit within
    80 GB: each pairformer and MSA block is individually checkpointed so
    that backward recompute never materialises all 64 blocks' activations
    simultaneously. When ``gradient_checkpointing`` is set, the confidence
    module's own pairformer stack gets the same treatment -- see
    :func:`enable_confidence_checkpointing`.

    Optional knobs, all defaulting to today's exact behaviour:

    autocast_dtype : str | None
        Overrides ``IGV_AUTOCAST`` (default "off" = no autocast anywhere, i.e.
        full fp32 as this repo has always run). ``"bf16"`` wraps ONLY the
        trunk+confidence dispatch below in ``torch.autocast("cuda", ...)``; the
        post-checkpoint guards and every caller's ``score.backward()`` stay
        outside it, which is sufficient because both checkpoint flavours stash
        the autocast state at forward and re-enter it during the backward
        recompute (verified on torch 2.5.1 with this module's exact reentrant/
        non-reentrant nesting; RE-VERIFY on the VM's pinned torch 2.7.1+cu126).
        ``fp16`` raises -- see :func:`resolve_autocast_dtype`.

        What bf16 does NOT fix, so it is not re-budgeted as a 2x saving: boltz
        2.2.1 ``pairformer.py:104-110`` opens ``with torch.autocast("cuda",
        enabled=False)`` and calls ``s.float()``, ``z.float()``,
        ``mask.float()``, so the SEQUENCE track of all 64 trunk layers stays
        fp32 and each layer takes a full fp32 copy of z (0.254 GiB at L=730) as
        attention bias. The softmax term does move to boltz's bf16 branch
        (``primitives.py:168`` instead of ``:170``, the site ERRORS_LOG blames);
        the boundary-z term should follow but MUST be confirmed by printing
        ``z_.dtype`` at the group-checkpoint boundary. And the score VALUE will
        change: every recorded score, signal_control and the completeness gate
        has to be re-derived under bf16, never compared across dtypes, and the
        dtype must land in the provenance arm dict of the calling stage.
    triangle_attention_chunk_checkpointing : bool | None
        Overrides ``IGV_TRI_ATTN_CKPT`` (default off). Meaningful only when
        ``gradient_checkpointing`` is true -- the saving is entirely in the
        backward recompute -- and ignored with a warning otherwise. See
        :func:`enable_triangle_attention_chunk_checkpointing`.
    """
    import torch
    from torch.utils.checkpoint import checkpoint as _checkpoint
    from boltz.data import const as _boltz_const

    if score_name not in SCORES:
        raise ValueError(
            f"Unknown score {score_name!r}; available: {sorted(SCORES)}"
        )

    # Knobs. Every default reproduces the pre-knob behaviour exactly.
    _amp_dtype = resolve_autocast_dtype(autocast_dtype)
    if _amp_dtype is not None:
        log.warning(
            "IGV_AUTOCAST=%s -- scores are NOT comparable to fp32 runs. "
            "Re-derive every reference score, signal_control and the "
            "completeness gate under this dtype and record it in the "
            "provenance arm dict.",
            _amp_dtype,
        )
    _use_kernels, _use_cuequiv_attn = resolve_use_kernels(model)
    _tri_attn_ckpt = resolve_tri_attn_ckpt(triangle_attention_chunk_checkpointing)

    if gradient_checkpointing:
        # The confidence stack needs the same per-block treatment as the trunk;
        # without it the backward recompute OOMs at 730 tokens. See
        # enable_confidence_checkpointing for why .train() is not the answer.
        enable_confidence_checkpointing(model)
        if _tri_attn_ckpt:
            _n_tri = enable_triangle_attention_chunk_checkpointing(model)
            # "0 newly wrapped" has two causes with OPPOSITE meanings: the
            # rebind is idempotent (guarded by ``_igv_chunk_ckpt``), so any
            # caller that reuses one model object -- 08_memscale across its
            # ladder, 07_sanity across signal_control's mutants -- legitimately
            # wraps 0 on every call after the first while the lever stays fully
            # in effect. Only "nothing is instrumented" deserves the warning;
            # emitting it for the reuse case told the operator to "expect no
            # memory change" on a run where all 156 modules were checkpointed,
            # which would misread the whole sweep.
            _n_live = count_tri_attn_chunk_ckpt(model)
            if _n_live == 0:
                log.warning(
                    "IGV_TRI_ATTN_CKPT is on but no TriangleAttention module is "
                    "instrumented (the module layout changed). Expect no memory "
                    "change."
                )
            elif _n_tri == 0:
                log.info(
                    "Triangle-attention chunk checkpointing already active on %d "
                    "modules from an earlier call on this model object; still in "
                    "effect (the rebind is idempotent).",
                    _n_live,
                )
    elif _tri_attn_ckpt:
        log.warning(
            "IGV_TRI_ATTN_CKPT ignored: it only saves memory inside the "
            "backward recompute, which requires gradient_checkpointing=True."
        )

    device = s_inputs.device
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    with torch.no_grad():
        rel_pos = model.rel_pos(feats)
        token_bonds_z = model.token_bonds(feats["token_bonds"].float())
        contact_z = model.contact_conditioning(feats)

    _n_tokens = int(mask.shape[1])
    try:
        _threshold = _boltz_const.chunk_size_threshold
    except Exception:
        _threshold = 384

    # Triangle-attention chunk size, and the MSA chunk knobs that move with it.
    #
    # CORRECTION (this comment previously claimed the opposite). Halving
    # IGV_PF_CHUNK does NOT halve the retained softmax, and it cannot reduce
    # total retained memory at all. boltz 2.2.1
    # triangular_attention/utils.py:322-375 (`chunk_layer`) is a plain
    # `for _ in range(no_chunks)` loop calling `layer(**chunks)` at :339 with NO
    # torch.no_grad, NO detach and NO recompute, so every chunk's softmax output
    # is saved for backward and the retained total is chunk-size-INVARIANT.
    # Chunking is in fact marginally WORSE: it also allocates the `out` buffer
    # via `t.new_zeros` at :343.
    #
    # The arithmetic confirms it exactly. chunk_layer chunks over the batch dims
    # x.shape[:-2] == [1, I], and _prep_qkv gives q/k/v of
    # [1, I, H, J, C_hidden], so the softmax tensor at primitives.py:191 is
    # [chunk, H, J, J]. At L=730, H=4, fp32:
    #     one chunk of 128 : 128*4*730*730*4B = 1.0164 GiB
    #     full attention   : 730*4*730*730*4B = 5.7968 GiB
    #     per PairformerLayer (tri_att_start + tri_att_end) = 11.5936 GiB
    # 11.59 GiB matches the observed OOM profile: primitives.py:170
    # held 12 tensors at IGV_PF_GROUP_SIZE=1
    # (12 = 2 attentions x 6 chunks, since 730 = 5x128 + 90), and 1.02 GiB is
    # precisely the failing allocation in every recorded OOM. That is the whole
    # explanation of commit b1c29cc ("chunk knobs do not fix the OOM").
    #
    # So IGV_PF_CHUNK is a TRANSIENT / FRAGMENTATION control ONLY: it bounds the
    # largest single LIVE allocation (1.0164 GiB at 128, 0.1271 GiB at 16) and
    # therefore the size of the request that fails, not the total held. For the
    # knob that actually reduces the retained softmax, see
    # enable_triangle_attention_chunk_checkpointing / IGV_TRI_ATTN_CKPT.
    _profile = select_chunk_profile(_n_tokens, _threshold)
    log.info(
        "chunk profile: %s (n_tokens=%d, threshold=%d, pf_chunk=%d)",
        _profile["profile"], _n_tokens, _threshold, _profile["pf_chunk"],
    )

    # How many pairformer blocks share one checkpoint. Env-overridable so the
    # memory/recompute tradeoff can be tuned per complex size without a code
    # change; see the loop below for what it buys.
    _pf_group = max(1, int(os.environ.get("IGV_PF_GROUP_SIZE", _PF_GROUP_SIZE)))

    _n_pf_blocks = len(model.pairformer_module.layers)
    _n_msa_blocks = model.msa_module.msa_blocks

    # recycling_steps is a caller parameter (default 1, matching the
    # predecessor repo's "1 for speed, 3 for quality"). It is also the single
    # biggest VRAM lever: the trunk runs recycling_steps+1 iterations and each
    # one saves a checkpoint boundary tensor per block, so at 730 tokens
    # 64 blocks x 2 iterations x 0.254 GiB = 32.5 GiB of saved z alone.

    # (A dead _pairformer_block_fn used to sit here with no caller. Deleted:
    # two near-identical block runners is how the two paths drift apart. The
    # live path is _pairformer_group_fn, called from the loop below.)

    _recycle_counter = [0]

    def _pairformer_group_fn(s_, z_, mask_, pair_mask_, span):
        """Run pairformer layers [span[0], span[1]) as ONE checkpoint unit."""
        start, stop = int(span[0].item()), int(span[1].item())
        for i in range(start, stop):
            # Positional args 6-8 are use_kernels, use_cuequiv_mul,
            # use_cuequiv_attn. Reaching use_cuequiv_attn is only possible
            # because we call the layer directly: PairformerModule.forward does
            # not forward it. Both flags default False.
            s_, z_ = model.pairformer_module.layers[i](
                s_, z_, mask_, pair_mask_, _profile["pf_chunk"],
                _use_kernels, False, _use_cuequiv_attn,
            )
            if activation_callback is not None:
                cb_result = activation_callback(_recycle_counter[0], i, s_, z_)
                if cb_result is not None:
                    s_, z_ = cb_result
        return s_, z_

    def _msa_block_fn(z_, m_, token_mask_, msa_mask_, layer_idx_dummy):
        idx = int(layer_idx_dummy.item())
        return model.msa_module.layers[idx](
            z_, m_, token_mask_, msa_mask_,
            _profile["msa_chunk_heads_pwa"],
            _profile["msa_chunk_trans_z"],
            _profile["msa_chunk_trans_msa"],
            _profile["msa_chunk_outer"],
            _profile["msa_chunk_tri"],
            # MSALayer.forward's 10th POSITIONAL parameter is use_kernels. This
            # was a bare `False` that no grep for "use_kernels" could find.
            _use_kernels,
        )

    def _msa_forward_checkpointed(z_, s_interp_):
        try:
            from boltz.data import const as _c
            _num_tokens = _c.num_tokens
        except Exception:
            _num_tokens = 33

        msa = feats["msa"]
        msa_oh = torch.nn.functional.one_hot(msa, num_classes=_num_tokens)
        has_deletion = feats["has_deletion"].unsqueeze(-1)
        deletion_value = feats["deletion_value"].unsqueeze(-1)
        msa_mask_ = feats["msa_mask"]
        token_mask_ = feats["token_pad_mask"].float()
        token_mask_ = token_mask_[:, :, None] * token_mask_[:, None, :]

        if model.msa_module.use_paired_feature:
            is_paired = feats["msa_paired"].unsqueeze(-1)
            m = torch.cat([msa_oh, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa_oh, has_deletion, deletion_value], dim=-1)

        m = model.msa_module.msa_proj(m)
        m = m + model.msa_module.s_proj(s_interp_).unsqueeze(1)

        for blk_i in range(_n_msa_blocks):
            idx_t = torch.tensor(blk_i, device=device)
            z_, m = _checkpoint(
                _msa_block_fn, z_, m, token_mask_, msa_mask_, idx_t,
                use_reentrant=False,
            )
        return z_

    def _full_trunk_and_confidence(s_interp):
        s_init = model.s_init(s_interp)
        z_init = (
            model.z_init_1(s_interp)[:, :, None, :]
            + model.z_init_2(s_interp)[:, None, :, :]
            + rel_pos + token_bonds_z + contact_z
        )

        s_ = torch.zeros_like(s_init)
        z_ = torch.zeros_like(z_init)

        for _recycle_i in range(recycling_steps + 1):
            _recycle_counter[0] = _recycle_i
            s_ = s_init + model.s_recycle(model.s_norm(s_))
            z_ = z_init + model.z_recycle(model.z_norm(z_))

            if gradient_checkpointing:
                z_ = z_ + _msa_forward_checkpointed(z_, s_interp)
                # Checkpoint GROUPS of blocks, not single blocks. Each
                # checkpoint retains its input z (1, L, L, 128); at L=730 that
                # is 254 MiB apiece, and a profile of the peak attributed
                # 34.30 GiB across 135 such tensors -- the largest single term
                # by far. Grouping trades those boundaries against a bigger
                # transient during recompute (the classic sqrt(N) tradeoff):
                # group_size=1 -> 64 boundaries/iteration, minimum recompute;
                # group_size=8 ->  8 boundaries/iteration, 8 blocks of
                # activations live at once. See _PF_GROUP_SIZE.
                for start in range(0, _n_pf_blocks, _pf_group):
                    stop = min(start + _pf_group, _n_pf_blocks)
                    span = torch.tensor([start, stop], device=device)
                    s_, z_ = _checkpoint(
                        _pairformer_group_fn, s_, z_, mask, pair_mask, span,
                        use_reentrant=False,
                    )
            elif activation_callback is not None:
                z_ = z_ + model.msa_module(
                    z_, s_interp, feats, use_kernels=_use_kernels
                )
                for start in range(0, _n_pf_blocks, _pf_group):
                    stop = min(start + _pf_group, _n_pf_blocks)
                    span = torch.tensor([start, stop], device=device)
                    s_, z_ = _pairformer_group_fn(
                        s_, z_, mask, pair_mask, span,
                    )
            else:
                # IGV_TRI_ATTN_KERNEL cannot reach these two module-level entry
                # points (PairformerModule.forward does not forward
                # use_cuequiv_attn), so only the combined flag applies here.
                z_ = z_ + model.msa_module(
                    z_, s_interp, feats, use_kernels=_use_kernels
                )
                s_, z_ = model.pairformer_module(
                    s_, z_, mask=mask, pair_mask=pair_mask,
                    use_kernels=_use_kernels,
                )

        pdistogram = model.distogram_module(z_)
        pred_distogram_logits = pdistogram[:, :, :, 0]

        out_dict = model.confidence_module(
            s_inputs=s_interp,
            s=s_,
            z=z_,
            x_pred=x_pred,
            feats=feats,
            pred_distogram_logits=pred_distogram_logits,
            multiplicity=1,
            run_sequentially=False,
            # Flows on into confidence_module.pairformer_stack, which boltz
            # calls with use_kernels=use_kernels -- so the rebound
            # _checkpointed_forward receives it and must not switch it again.
            use_kernels=_use_kernels,
        )

        scalar = SCORES[score_name](out_dict)
        if scalar.dim() > 0:
            scalar = scalar.squeeze()

        # Guards deliberately live OUTSIDE this function, after the checkpoint
        # boundary -- see below. requires_grad is not meaningful in here.
        return scalar

    # The ONE autocast injection point. Every caller inherits it, including the
    # torch.no_grad() reference forwards used by the completeness check, so
    # f(x) - f(baseline) and the path integral are never compared across dtypes.
    # s_inputs is deliberately NOT cast: the fp32 leaf is load-bearing (the
    # Gauss-Legendre weights in attrib.py are built at its dtype, and the .npz
    # writer calls .numpy() with no .float()), and autocast gives bf16 compute
    # while leaving the leaf and its .grad fp32.
    _amp_ctx = (
        torch.autocast("cuda", dtype=_amp_dtype)
        if _amp_dtype is not None
        else contextlib.nullcontext()
    )
    with _amp_ctx:
        if gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint as _ckpt
            # use_reentrant=True is load-bearing for memory, and is why the guards
            # below sit out here rather than inside _full_trunk_and_confidence.
            #
            # Reentrant checkpointing runs the wrapped function under
            # torch.no_grad(), so NO autograd graph is built during the forward at
            # all; the trunk is recomputed with grad during backward, where the
            # per-block non-reentrant checkpoints bound peak memory. That two-level
            # scheme is what makes this complex fit in 80 GB. Switching the outer
            # call to use_reentrant=False builds the full forward graph and OOMs
            # (measured: 79.25 GiB capacity, ~32 MiB free).
            #
            # The cost is that requires_grad is False for every tensor inside the
            # function, so the guards cannot live there. They run here instead,
            # where the value is real.
            scalar = _ckpt(
                _full_trunk_and_confidence, s_inputs, use_reentrant=True,
            )
        else:
            scalar = _full_trunk_and_confidence(s_inputs)
    # Guards run OUTSIDE the autocast region, and callers run .backward()
    # outside it too -- both checkpoint flavours stash the autocast state at
    # forward and re-enter it during the backward recompute, so wrapping
    # backward would add nothing and is unsupported.

    # Only demand a gradient when the caller actually asked for one. Callers
    # that just want a score -- the signal_control sanity check scores 30
    # mutants under torch.no_grad() -- are legitimate and must not trip this.
    if torch.is_grad_enabled():
        assert scalar.requires_grad, (
            f"Score {score_name!r} does not require grad. This usually means "
            f"compute_ptms silently failed (check stdout for 'Error in "
            f"compute_ptms') and returned a zero tensor without grad."
        )
    assert torch.isfinite(scalar).all(), (
        f"Score {score_name!r} is not finite: {scalar.item()}"
    )
    # boltz wraps compute_ptms in a bare `except` that assigns
    # torch.zeros_like(complex_plddt) and only prints. A zero score therefore
    # looks perfectly well-formed, and IG would silently integrate a constant.
    assert scalar.abs().item() > 1e-12, (
        f"Score {score_name!r} is exactly zero. compute_ptms almost "
        "certainly failed silently -- check stdout for 'Error in "
        "compute_ptms'. Do not interpret this run."
    )

    return scalar


def confidence_head_forward(
    model,
    s_inputs,
    s,
    z,
    x_pred,
    feats,
    score_name: str,
    autocast_dtype=None,
):
    """Score from the confidence head with an externally supplied ``z``.

    This is the seam for pair-layer Integrated Gradients: ``z`` is supplied by
    the caller (typically an interpolated tensor with ``requires_grad=True``)
    and is **not** recomputed from the trunk.  Everything upstream of the
    confidence head -- ``s``, ``s_inputs``, ``x_pred``, ``feats`` -- is fixed
    across interpolation steps.

    Does NOT wrap in the outer ``use_reentrant=True`` checkpoint that
    :func:`confidence_forward` uses at line 1832: the inner per-layer
    checkpoints are already ``use_reentrant=False``, and bypassing the outer
    wrapper is exactly the intent (see ``notes/logging.md`` section 12).

    ``z`` must be fp32; it is not cast even under autocast, so that its
    ``.grad`` stays fp32.

    The distogram is recomputed from the supplied ``z`` rather than passed in
    frozen, because the distogram is a direct function of ``z`` and freezing it
    would break the path integral: the completeness identity requires that
    every ``z``-dependent intermediate tracks the interpolated ``z``.
    """
    import torch

    _amp_dtype = resolve_autocast_dtype(autocast_dtype)
    _use_kernels, _ = resolve_use_kernels(model)

    if score_name not in SCORES:
        raise ValueError(
            f"Unknown score: {score_name!r}. Available: {sorted(SCORES)}"
        )

    _amp_ctx = (
        torch.autocast("cuda", dtype=_amp_dtype)
        if _amp_dtype is not None
        else contextlib.nullcontext()
    )

    with _amp_ctx:
        # Duplicated from _full_trunk_and_confidence lines 1777-1798 rather
        # than refactored into a shared helper: the existing path is wrapped in
        # reentrant checkpointing with load-bearing use_reentrant=True, and any
        # extraction risks changing its behaviour.  ~15 lines of duplication is
        # the safer choice.
        pred_distogram_logits = model.distogram_module(z)[:, :, :, 0]

        out_dict = model.confidence_module(
            s_inputs=s_inputs,
            s=s,
            z=z,
            x_pred=x_pred,
            feats=feats,
            pred_distogram_logits=pred_distogram_logits,
            multiplicity=1,
            run_sequentially=False,
            # Flows on into confidence_module.pairformer_stack, which boltz
            # calls with use_kernels=use_kernels -- so the rebound
            # _checkpointed_forward receives it and must not switch it again.
            use_kernels=_use_kernels,
        )

        scalar = SCORES[score_name](out_dict)
        if scalar.dim() > 0:
            scalar = scalar.squeeze()

    return scalar


# ---------------------------------------------------------------------------
# Affinity head forward -- the differentiable seam
# ---------------------------------------------------------------------------


def affinity_forward(
    model,
    s_inputs,
    feats,
    x_pred,
    score_name: str,
    gradient_checkpointing: bool = True,
    recycling_steps: int = 1,
    autocast_dtype: str | None = None,
    triangle_attention_chunk_checkpointing: bool | None = None,
    activation_callback=None,
):
    """Full-trunk forward ending at the affinity head, differentiable to ``s_inputs``.

    This is the affinity analogue of :func:`confidence_forward`.  Gradient
    flows ``res_type`` -> ``InputEmbedder`` -> trunk -> z -> affinity module,
    with the same two-level gradient checkpointing scheme that makes the
    confidence path fit in 80 GiB.

    The outer ``checkpoint(..., use_reentrant=True)`` runs the trunk under
    ``no_grad`` during the forward; during backward the trunk is recomputed
    with grad, and the inner per-block ``use_reentrant=False`` checkpoints
    bound peak memory.  See :func:`confidence_forward` for the detailed
    explanation and the reason this nesting is load-bearing.

    Memory estimate for the affinity path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The affinity cropper caps the complex at 256 tokens (``max_tokens=256``).
    Trunk VRAM scales roughly as L^2 for the pair tensor.  At L=730 with
    checkpointing the confidence path measured 55.2 GiB
    (docs/MEMSCALE_RESULTS.md).  At L=256: ``(256/730)^2 * 55.2 ~ 6.8 GiB``.
    With ``IGV_TRI_ATTN_CKPT=1`` and ``IGV_AUTOCAST=bf16`` this is
    comfortably within 80 GiB.

    Differences from the confidence path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``model.input_embedder(feats, affinity=True)`` selects
      ``profile_affinity`` / ``deletion_mean_affinity`` (MSA depth 1).
    * ``z_affinity = z * cross_pair_mask`` masks to protein-ligand pairs.
    * ``x_pred`` enters via distance bucketisation (``cdist`` -> ``.long()``
      -> ``Embedding``), which is non-differentiable.  It is held fixed.
    * The MW correction is applied when the model has it enabled.
    """
    import torch
    from torch.utils.checkpoint import checkpoint as _checkpoint
    from boltz.data import const as _boltz_const

    if score_name not in AFFINITY_SCORES:
        raise ValueError(
            f"Unknown affinity score {score_name!r}; "
            f"available: {sorted(AFFINITY_SCORES)}"
        )

    _amp_dtype = resolve_autocast_dtype(autocast_dtype)
    if _amp_dtype is not None:
        log.warning(
            "IGV_AUTOCAST=%s -- affinity scores are NOT comparable to fp32 "
            "runs. Record the dtype in provenance.", _amp_dtype,
        )
    _use_kernels, _use_cuequiv_attn = resolve_use_kernels(model)
    _tri_attn_ckpt = resolve_tri_attn_ckpt(triangle_attention_chunk_checkpointing)

    if gradient_checkpointing:
        enable_confidence_checkpointing(model)
        if _tri_attn_ckpt:
            _n_tri = enable_triangle_attention_chunk_checkpointing(model)
            _n_live = count_tri_attn_chunk_ckpt(model)
            if _n_live == 0:
                log.warning(
                    "IGV_TRI_ATTN_CKPT is on but no TriangleAttention module "
                    "is instrumented."
                )
            elif _n_tri == 0:
                log.info(
                    "Triangle-attention chunk checkpointing already active on "
                    "%d modules.", _n_live,
                )
    elif _tri_attn_ckpt:
        log.warning(
            "IGV_TRI_ATTN_CKPT ignored: requires gradient_checkpointing=True."
        )

    device = s_inputs.device
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    with torch.no_grad():
        rel_pos = model.rel_pos(feats)
        token_bonds_z = model.token_bonds(feats["token_bonds"].float())
        contact_z = model.contact_conditioning(feats)

    _n_tokens = int(mask.shape[1])
    try:
        _threshold = _boltz_const.chunk_size_threshold
    except Exception:
        _threshold = 384

    _profile = select_chunk_profile(_n_tokens, _threshold)
    log.info(
        "affinity_forward chunk profile: %s (n_tokens=%d, pf_chunk=%d)",
        _profile["profile"], _n_tokens, _profile["pf_chunk"],
    )

    _pf_group = max(1, int(os.environ.get("IGV_PF_GROUP_SIZE", _PF_GROUP_SIZE)))
    _n_pf_blocks = len(model.pairformer_module.layers)
    _n_msa_blocks = model.msa_module.msa_blocks

    # Activations captured here come from the main trunk's Pairformer,
    # NOT the affinity module's own Pairformer stack. The affinity module's
    # internal Pairformer re-processes z_affinity but its layers are not
    # indexed here. We capture at the same layer indices as confidence_forward
    # so the two paths produce comparable activation caches.
    _recycle_counter = [0]

    def _pairformer_group_fn(s_, z_, mask_, pair_mask_, span):
        start, stop = int(span[0].item()), int(span[1].item())
        for i in range(start, stop):
            s_, z_ = model.pairformer_module.layers[i](
                s_, z_, mask_, pair_mask_, _profile["pf_chunk"],
                _use_kernels, False, _use_cuequiv_attn,
            )
            if activation_callback is not None:
                cb_result = activation_callback(_recycle_counter[0], i, s_, z_)
                if cb_result is not None:
                    s_, z_ = cb_result
        return s_, z_

    def _msa_block_fn(z_, m_, token_mask_, msa_mask_, layer_idx_dummy):
        idx = int(layer_idx_dummy.item())
        return model.msa_module.layers[idx](
            z_, m_, token_mask_, msa_mask_,
            _profile["msa_chunk_heads_pwa"],
            _profile["msa_chunk_trans_z"],
            _profile["msa_chunk_trans_msa"],
            _profile["msa_chunk_outer"],
            _profile["msa_chunk_tri"],
            _use_kernels,
        )

    def _msa_forward_checkpointed(z_, s_interp_):
        try:
            from boltz.data import const as _c
            _num_tokens = _c.num_tokens
        except Exception:
            _num_tokens = 33

        msa = feats["msa"]
        msa_oh = torch.nn.functional.one_hot(msa, num_classes=_num_tokens)
        has_deletion = feats["has_deletion"].unsqueeze(-1)
        deletion_value = feats["deletion_value"].unsqueeze(-1)
        msa_mask_ = feats["msa_mask"]
        token_mask_ = feats["token_pad_mask"].float()
        token_mask_ = token_mask_[:, :, None] * token_mask_[:, None, :]

        if model.msa_module.use_paired_feature:
            is_paired = feats["msa_paired"].unsqueeze(-1)
            m = torch.cat([msa_oh, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa_oh, has_deletion, deletion_value], dim=-1)

        m = model.msa_module.msa_proj(m)
        m = m + model.msa_module.s_proj(s_interp_).unsqueeze(1)

        for blk_i in range(_n_msa_blocks):
            idx_t = torch.tensor(blk_i, device=device)
            z_, m = _checkpoint(
                _msa_block_fn, z_, m, token_mask_, msa_mask_, idx_t,
                use_reentrant=False,
            )
        return z_

    def _full_trunk_and_affinity(s_interp):
        s_init = model.s_init(s_interp)
        z_init = (
            model.z_init_1(s_interp)[:, :, None, :]
            + model.z_init_2(s_interp)[:, None, :, :]
            + rel_pos + token_bonds_z + contact_z
        )

        s_ = torch.zeros_like(s_init)
        z_ = torch.zeros_like(z_init)

        for _recycle_i in range(recycling_steps + 1):
            _recycle_counter[0] = _recycle_i
            s_ = s_init + model.s_recycle(model.s_norm(s_))
            z_ = z_init + model.z_recycle(model.z_norm(z_))

            if gradient_checkpointing:
                z_ = z_ + _msa_forward_checkpointed(z_, s_interp)
                for start in range(0, _n_pf_blocks, _pf_group):
                    stop = min(start + _pf_group, _n_pf_blocks)
                    span = torch.tensor([start, stop], device=device)
                    s_, z_ = _checkpoint(
                        _pairformer_group_fn, s_, z_, mask, pair_mask, span,
                        use_reentrant=False,
                    )
            elif activation_callback is not None:
                z_ = z_ + model.msa_module(
                    z_, s_interp, feats, use_kernels=_use_kernels
                )
                for start in range(0, _n_pf_blocks, _pf_group):
                    stop = min(start + _pf_group, _n_pf_blocks)
                    span = torch.tensor([start, stop], device=device)
                    s_, z_ = _pairformer_group_fn(
                        s_, z_, mask, pair_mask, span,
                    )
            else:
                z_ = z_ + model.msa_module(
                    z_, s_interp, feats, use_kernels=_use_kernels
                )
                s_, z_ = model.pairformer_module(
                    s_, z_, mask=mask, pair_mask=pair_mask,
                    use_kernels=_use_kernels,
                )

        # Build z_affinity: mask to protein-ligand pairs.
        pad_mask_1d = feats["token_pad_mask"][0]
        mol_type = feats.get("mol_type")
        aff_mask = feats.get("affinity_token_mask")

        if mol_type is not None:
            rec_mask_ = (mol_type[0] == 0).float() * pad_mask_1d.float()
        else:
            rec_mask_ = pad_mask_1d.float()

        if aff_mask is not None:
            lig_mask_ = aff_mask[0].float() * pad_mask_1d.float()
        else:
            lig_mask_ = torch.zeros_like(pad_mask_1d.float())

        cross_pair_mask_ = (
            lig_mask_[:, None] * rec_mask_[None, :]
            + rec_mask_[:, None] * lig_mask_[None, :]
            + lig_mask_[:, None] * lig_mask_[None, :]
        )

        z_affinity = z_ * cross_pair_mask_[None, :, :, None]

        x_pred_local = x_pred.detach()
        if x_pred_local.dim() == 3:
            x_pred_local = x_pred_local.unsqueeze(1)

        has_ensemble = (
            hasattr(model, "affinity_ensemble") and model.affinity_ensemble
            and hasattr(model, "affinity_module1")
            and hasattr(model, "affinity_module2")
        )

        if has_ensemble:
            out1 = model.affinity_module1(
                s_inputs=s_interp,
                z=z_affinity,
                x_pred=x_pred_local,
                feats=feats,
                multiplicity=1,
                use_kernels=_use_kernels,
            )
            out2 = model.affinity_module2(
                s_inputs=s_interp,
                z=z_affinity,
                x_pred=x_pred_local,
                feats=feats,
                multiplicity=1,
                use_kernels=_use_kernels,
            )
            out_dict = {
                "affinity_pred_value": (
                    out1["affinity_pred_value"] + out2["affinity_pred_value"]
                ) / 2,
                "affinity_logits_binary": (
                    out1["affinity_logits_binary"] + out2["affinity_logits_binary"]
                ) / 2,
            }
        else:
            aff_out = model.affinity_module(
                s_inputs=s_interp,
                z=z_affinity,
                x_pred=x_pred_local,
                feats=feats,
                multiplicity=1,
                use_kernels=_use_kernels,
            )
            out_dict = {
                "affinity_pred_value": aff_out["affinity_pred_value"],
                "affinity_logits_binary": aff_out["affinity_logits_binary"],
            }

        if hasattr(model, "affinity_mw_correction") and model.affinity_mw_correction:
            mw_val = feats.get("affinity_mw")
            if mw_val is not None:
                mw = mw_val[0] ** 0.3
                out_dict["affinity_pred_value"] = (
                    1.03525938 * out_dict["affinity_pred_value"]
                    + (-0.59992683) * mw
                    + 2.83288489
                )

        scalar = AFFINITY_SCORES[score_name](out_dict)
        if scalar.dim() > 0:
            scalar = scalar.squeeze()

        return scalar

    _amp_ctx = (
        torch.autocast("cuda", dtype=_amp_dtype)
        if _amp_dtype is not None
        else contextlib.nullcontext()
    )
    with _amp_ctx:
        if gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint as _ckpt
            scalar = _ckpt(
                _full_trunk_and_affinity, s_inputs, use_reentrant=True,
            )
        else:
            scalar = _full_trunk_and_affinity(s_inputs)

    if torch.is_grad_enabled():
        assert scalar.requires_grad, (
            f"Affinity score {score_name!r} does not require grad."
        )
    assert torch.isfinite(scalar).all(), (
        f"Affinity score {score_name!r} is not finite: {scalar.item()}"
    )

    return scalar


def affinity_head_forward(
    model,
    s_inputs,
    z,
    x_pred,
    feats,
    score_name: str,
    flow_through_z: bool = False,
    autocast_dtype=None,
):
    """Score from the affinity head with an externally supplied ``z``.

    **DIAGNOSTIC ONLY** -- this path severs gradient from the trunk and
    measures only how the affinity head responds to the residue-identity
    one-hot at its input, holding the trunk's view of the protein fixed at
    wild type.  It will produce a non-zero gradient that carries almost no
    mutation signal.  Use :func:`affinity_forward` for real attribution.

    This function exists for two purposes:
    1. A fast smoke test that the affinity module itself is callable.
    2. Measuring the s_inputs-only grad norm beside the full-trunk norm
       to quantify the trunk's contribution.

    For the full-trunk differentiable path (gradient through
    ``res_type`` -> ``InputEmbedder`` -> trunk -> z -> affinity module),
    use :func:`affinity_forward`, which has the same two-level
    gradient-checkpointing scheme as :func:`confidence_forward`.

    Bypasses the three ``.detach()`` calls at boltz 2.2.1
    ``boltz2.py:632-633,646-647,705-706`` by calling the affinity module
    directly.  ``x_pred`` is always held fixed (detached).  The affinity
    module bucketises distances (``cdist`` -> ``.long()`` -> ``Embedding``),
    so ``x_pred`` is non-differentiable by construction.

    The *flow_through_z* flag controls whether ``z`` retains its gradient
    state (True) or is detached (False, default).  When the caller sets
    ``z.requires_grad_(True)`` this gives pair-layer attribution analogous
    to the confidence path's ``confidence_head_forward``.
    """
    import torch

    if score_name not in AFFINITY_SCORES:
        raise ValueError(
            f"Unknown affinity score: {score_name!r}. "
            f"Available: {sorted(AFFINITY_SCORES)}"
        )

    _amp_dtype = resolve_autocast_dtype(autocast_dtype)
    _use_kernels, _ = resolve_use_kernels(model)

    _amp_ctx = (
        torch.autocast("cuda", dtype=_amp_dtype)
        if _amp_dtype is not None
        else contextlib.nullcontext()
    )

    pad_mask = feats["token_pad_mask"][0]
    mol_type = feats["mol_type"][0] if "mol_type" in feats else None
    aff_mask = feats.get("affinity_token_mask")

    if mol_type is not None:
        rec_mask = (mol_type == 0).float() * pad_mask.float()
    else:
        rec_mask = pad_mask.float()

    if aff_mask is not None:
        lig_mask = aff_mask[0].float() * pad_mask.float()
    else:
        lig_mask = torch.zeros_like(pad_mask.float())

    cross_pair_mask = (
        lig_mask[:, None] * rec_mask[None, :]
        + rec_mask[:, None] * lig_mask[None, :]
        + lig_mask[:, None] * lig_mask[None, :]
    )

    z_affinity = z * cross_pair_mask[None, :, :, None]
    if not flow_through_z:
        z_affinity = z_affinity.detach()

    x_pred_detached = x_pred.detach()
    if x_pred_detached.dim() == 3:
        x_pred_detached = x_pred_detached.unsqueeze(1)

    with _amp_ctx:
        has_ensemble = (
            hasattr(model, "affinity_ensemble") and model.affinity_ensemble
            and hasattr(model, "affinity_module1")
            and hasattr(model, "affinity_module2")
        )

        if has_ensemble:
            out1 = model.affinity_module1(
                s_inputs=s_inputs,
                z=z_affinity,
                x_pred=x_pred_detached,
                feats=feats,
                multiplicity=1,
                use_kernels=_use_kernels,
            )
            out2 = model.affinity_module2(
                s_inputs=s_inputs,
                z=z_affinity,
                x_pred=x_pred_detached,
                feats=feats,
                multiplicity=1,
                use_kernels=_use_kernels,
            )
            out_dict = {
                "affinity_pred_value": (
                    out1["affinity_pred_value"] + out2["affinity_pred_value"]
                ) / 2,
                "affinity_logits_binary": (
                    out1["affinity_logits_binary"] + out2["affinity_logits_binary"]
                ) / 2,
            }
        else:
            aff_out = model.affinity_module(
                s_inputs=s_inputs,
                z=z_affinity,
                x_pred=x_pred_detached,
                feats=feats,
                multiplicity=1,
                use_kernels=_use_kernels,
            )
            out_dict = {
                "affinity_pred_value": aff_out["affinity_pred_value"],
                "affinity_logits_binary": aff_out["affinity_logits_binary"],
            }

        if hasattr(model, "affinity_mw_correction") and model.affinity_mw_correction:
            mw_val = feats.get("affinity_mw")
            if mw_val is not None:
                mw = mw_val[0] ** 0.3
                out_dict["affinity_pred_value"] = (
                    1.03525938 * out_dict["affinity_pred_value"]
                    + (-0.59992683) * mw
                    + 2.83288489
                )

        scalar = AFFINITY_SCORES[score_name](out_dict)
        if scalar.dim() > 0:
            scalar = scalar.squeeze()

    return scalar


# ---------------------------------------------------------------------------
# Simplex-gradient seam: gradient w.r.t. the one-hot res_type tensor
# ---------------------------------------------------------------------------


def canonical_aa_token_indices(
    tokens: tuple[str, ...] | None = None,
    token_to_letter: dict[str, str] | None = None,
):
    """Map each canonical amino acid to its column index in the token axis.

    Returns an ``int64`` array of shape ``(20,)`` whose *i*-th entry is the
    position of ``CANONICAL_AMINO_ACIDS[i]`` in the boltz token list.

    The token list is derived at runtime from ``boltz.data.const.tokens``
    (preferred) or the pinned 2.2.1 fallback.  Pass *tokens* and
    *token_to_letter* explicitly only in tests.

    Raises
    ------
    RuntimeError
        If any canonical amino acid is absent from the token list.
    """
    import numpy as np
    from igv.attrib import CANONICAL_AMINO_ACIDS

    if tokens is None or token_to_letter is None:
        try:
            from boltz.data import const as _c
        except ImportError:
            # boltz genuinely absent (local dev / CI).  Fall back, but say so
            # loudly: a silent substitution here mislabels every column of the
            # (L, 20) map without crashing.  Cf. ERRORS_LOG entry 34.
            log.warning(
                "boltz is not installed; using the PINNED 2.2.1 token "
                "vocabulary for canonical_aa_token_indices. Any (L, 20) map "
                "built this way is unverified against the live model and must "
                "not be reported. Install boltz to derive indices at runtime."
            )
            tokens = _BOLTZ_TOKENS_2_2_1
            token_to_letter = dict(_PROT_TOKEN_TO_LETTER_2_2_1)
        else:
            # boltz IS present: derive at runtime and let any API change raise
            # rather than degrade to the pinned tables.
            tokens = tuple(_c.tokens)
            token_to_letter = dict(_c.prot_token_to_letter)
            if tuple(tokens) != tuple(_BOLTZ_TOKENS_2_2_1):
                log.warning(
                    "Live boltz token vocabulary differs from the pinned "
                    "2.2.1 table (%d live vs %d pinned). Using the live one, "
                    "which is correct, but the pinned fallback and any test "
                    "asserting fixed indices are now stale.",
                    len(tokens), len(_BOLTZ_TOKENS_2_2_1),
                )

    letter_to_token: dict[str, str] = {}
    for tok_name, letter in token_to_letter.items():
        if letter not in letter_to_token:
            letter_to_token[letter] = tok_name

    token_name_to_idx = {name: i for i, name in enumerate(tokens)}

    indices = np.empty(len(CANONICAL_AMINO_ACIDS), dtype=np.intp)
    missing: list[str] = []
    for i, aa_letter in enumerate(CANONICAL_AMINO_ACIDS):
        tok_name = letter_to_token.get(aa_letter)
        if tok_name is None or tok_name not in token_name_to_idx:
            missing.append(aa_letter)
            continue
        indices[i] = token_name_to_idx[tok_name]

    if missing:
        raise RuntimeError(
            f"Canonical amino acid(s) {missing} not found in the boltz token "
            f"list ({len(tokens)} tokens). The token vocabulary may have "
            f"changed; update the mapping."
        )
    return indices


def build_vocab_order(
    tokens: tuple[str, ...] | None = None,
    token_to_letter: dict[str, str] | None = None,
):
    """Build a (num_tokens,) array of one-letter codes for the token vocabulary.

    Protein tokens map to their one-letter amino acid codes. Non-protein tokens
    (pad, gap, RNA, DNA) map to empty strings. The result is suitable for
    serialisation as ``vocab_order`` in gradient artifacts, so consumers can
    locate amino acid columns without importing boltz.
    """
    import numpy as np

    if tokens is None or token_to_letter is None:
        try:
            from boltz.data import const as _c
        except ImportError:
            tokens = _BOLTZ_TOKENS_2_2_1
            token_to_letter = dict(_PROT_TOKEN_TO_LETTER_2_2_1)
        else:
            tokens = tuple(_c.tokens)
            token_to_letter = dict(_c.prot_token_to_letter)

    vocab = np.array(
        [token_to_letter.get(tok, "") for tok in tokens],
        dtype=object,
    )
    return vocab


def res_type_gradient_forward(
    model,
    feats: dict,
    score_name: str,
    x_pred=None,
    gradient_checkpointing: bool = True,
    recycling_steps: int = 1,
    autocast_dtype: str | None = None,
    triangle_attention_chunk_checkpointing: bool | None = None,
):
    """Gradient of a scalar score w.r.t. the one-hot ``res_type`` tensor.

    This is the seam for simplex-corrected attribution: the gradient lands
    directly on the categorical one-hot input (shape ``(1, L, num_tokens)``)
    rather than on the 384-dim embedding, so no linearity assumption is
    needed.

    The function:
      1. Detaches ``feats["res_type"]`` and enables its gradient.
      2. Runs the input embedder to produce ``s_inputs``.
      3. Forwards through the full trunk and confidence head via
         :func:`confidence_forward`.
      4. Calls ``.backward()`` and returns the gradient.

    Returns
    -------
    dict
        ``"grad_res_type"`` : numpy array, shape ``(L, num_tokens)``
            The gradient ``feats["res_type"].grad[0]``, detached and on CPU.
        ``"score"`` : float
            The scalar score value.

    Notes
    -----
    ``feats["res_type"]`` is restored to its original state (no gradient,
    original data) after the call so the dict is safe to reuse.
    """
    import torch

    res_type_orig = feats["res_type"]
    res_type_leaf = res_type_orig.detach().clone().float().requires_grad_(True)
    feats["res_type"] = res_type_leaf

    try:
        s_inputs = model.input_embedder(feats)

        if x_pred is None:
            # Zeros are legitimate for a gradient-flow smoke test -- that is
            # what Gate A ran -- but wrong for a reported attribution: the
            # confidence head then sees no geometry at all.  Gate A measured
            # complex_pde 1.988813 -> 1.029308 (-48.2%) once real coordinates
            # were supplied.  Warn rather than fabricate silently.
            log.warning(
                "res_type_gradient_forward: x_pred not supplied, using zeros. "
                "The confidence head sees no geometry; Gate A measured a "
                "-48.2%% shift in complex_pde when real coordinates were "
                "passed. Fine for a gradient-flow check, not for a reported "
                "attribution -- pass the predicted coordinates."
            )
            x_pred = torch.zeros(
                1, s_inputs.shape[1], 3, device=s_inputs.device
            )

        scalar = confidence_forward(
            model,
            s_inputs,
            feats,
            x_pred=x_pred,
            score_name=score_name,
            gradient_checkpointing=gradient_checkpointing,
            recycling_steps=recycling_steps,
            autocast_dtype=autocast_dtype,
            triangle_attention_chunk_checkpointing=triangle_attention_chunk_checkpointing,
        )

        scalar.backward()

        grad = res_type_leaf.grad
        if grad is None:
            raise RuntimeError(
                "No gradient on res_type after backward. The graph may have "
                "been detached by an intermediate operation."
            )
        grad_np = grad[0].detach().cpu().float().numpy()
        score_val = scalar.detach().cpu().item()
    finally:
        feats["res_type"] = res_type_orig

    return {"grad_res_type": grad_np, "score": score_val}


# ---------------------------------------------------------------------------
# ipTM argmax recording
# ---------------------------------------------------------------------------


def record_iptm_argmax(out_dict, token_to_chain):
    """Record which token holds the ipTM argmax frame index.

    Parameters
    ----------
    out_dict : dict
        Confidence module output dict (must contain ``iptm``-related tensors
        from ``compute_ptms``).
    token_to_chain : dict[int, str]
        Maps token index to chain id.

    Returns
    -------
    dict with ``argmax_token``, ``argmax_chain``, ``argmax_resi``.
    """
    import torch

    pae_logits = out_dict.get("pae_logits")
    if pae_logits is None:
        return {"argmax_token": None, "argmax_chain": None, "argmax_resi": None}

    num_bins = pae_logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_value = torch.arange(
        start=0.5 * bin_width, end=32.0, step=bin_width, device=pae_logits.device
    ).unsqueeze(0)
    probs = torch.nn.functional.softmax(pae_logits, dim=-1)

    # Pinned to fp32 rather than pae_logits.dtype on purpose: under bf16
    # autocast torch.tensor(730.).to(torch.bfloat16) == 728.0 (bf16 spacing is 4
    # in that binade), which would shift d0 below and silently move the reported
    # ipTM argmax token. This path runs under no_grad, so only the recorded
    # argmax is affected -- but a silent wrong number is still a wrong number.
    N_res = torch.tensor(
        [pae_logits.shape[1]], device=pae_logits.device, dtype=torch.float32
    )
    d0 = 1.24 * (torch.clip(N_res, min=19) - 15) ** (1.0 / 3.0) - 1.8
    tm_value = 1.0 / (1.0 + (pae_value / d0) ** 2)
    tm_value = tm_value.unsqueeze(1).unsqueeze(2)
    tm_expected = (probs * tm_value).sum(dim=-1)

    per_frame = tm_expected.mean(dim=-1)
    argmax_token = int(per_frame[0].argmax().item())

    chain_id = token_to_chain.get(argmax_token, "?")

    chain_offsets = {}
    for tok_idx, cid in sorted(token_to_chain.items()):
        if cid not in chain_offsets:
            chain_offsets[cid] = tok_idx
    resi = argmax_token - chain_offsets.get(chain_id, 0)

    return {
        "argmax_token": argmax_token,
        "argmax_chain": chain_id,
        "argmax_resi": resi,
    }
