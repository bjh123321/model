from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from surrogate_model.models.blocks import MFDNN as _RealMFDNN
from surrogate_model.models.checkpoint_utils import restore_model_config_from_checkpoint

# Reuse the existing prediction script for task parsing, response reconstruction,
# and PR/HF/HG/HA/VF/VG/VA file writing. Only replace the model-output adapter.
try:
    _base = importlib.import_module("test.Predict_user")
except ModuleNotFoundError:
    _base = importlib.import_module("surrogate_model.test.Predict_user")

_LAST_STATE = None
_LAST_CHECKPOINT = None
_REAL_TORCH_LOAD = _base.torch.load


def _capture_torch_load(*args, **kwargs):
    global _LAST_CHECKPOINT, _LAST_STATE
    ckpt = _REAL_TORCH_LOAD(*args, **kwargs)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt)
        if isinstance(state, dict):
            _LAST_CHECKPOINT = ckpt
            _LAST_STATE = state
    return ckpt


class _MFDNNFromCheckpoint(_RealMFDNN):
    """Build MFDNN from the complete structural checkpoint configuration."""

    def __init__(self, cfg):
        if _LAST_CHECKPOINT is None:
            raise RuntimeError("MFDNN was constructed before a checkpoint was captured")
        restored_cfg = restore_model_config_from_checkpoint(_LAST_CHECKPOINT)
        print(
            f"[Fixed-SF] model embedding sizes from checkpoint: "
            f"exc={restored_cfg.embed_excitation_num}x{restored_cfg.embed_excitation_dim}, "
            f"ch={restored_cfg.embed_channel_num}x{restored_cfg.embed_channel_dim}, "
            f"modulation_mode={restored_cfg.modulation_mode}, use_lora={restored_cfg.use_lora}",
            flush=True,
        )
        super().__init__(restored_cfg)


@torch.no_grad()
def run_infer_batched(
    model,
    in_norm,
    omegaN: float,
    device: torch.device,
    H_all: np.ndarray,
    M_all: np.ndarray,
    exc_all: np.ndarray,
    ch_all: np.ndarray,
    batch_size: int = 20000,
    predict_output: str = "low",
    mode_freq_ranges: Dict[str, List[Optional[Tuple[float, float]]]] | None = None,
):
    """Run fixed-SF inference using the current polar network output.

    Current fixed-SF MFDNN output is:
      alpha      : continuous pole real part, normally negative
      omega      : normalized continuous pole imaginary part, omega_d / (pi * fs)
      rho        : log residue magnitude in the trained data representation
      sin_theta  : residue phase sine
      cos_theta  : residue phase cosine

    Downstream reconstruction uses continuous-domain poles and residue magnitude:
      alpha_phys = alpha
      omega_phys = omega * omegaN
      rho_mag    = exp(rho)
      theta      = atan2(sin_theta, cos_theta)
    """
    N = len(H_all)
    kmax = int(getattr(getattr(model, "cfg", None), "K_max", 3))
    ch_limit = int(getattr(getattr(model, "emb_ch", None), "num_embeddings", 0) or 0)
    exc_limit = int(getattr(getattr(model, "emb_exc", None), "num_embeddings", 0) or 0)
    warned_invalid = False
    outs_acc = {k: [] for k in ["alpha", "omega", "rho", "sin_theta", "cos_theta", "valid_mask"]}

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        nb = e - s

        alpha_full = np.zeros((nb, kmax), dtype=np.float64)
        omega_full = np.zeros((nb, kmax), dtype=np.float64)
        rho_full = np.zeros((nb, kmax), dtype=np.float64)
        sin_full = np.zeros((nb, kmax), dtype=np.float64)
        cos_full = np.ones((nb, kmax), dtype=np.float64)
        valid_full = np.zeros((nb, kmax), dtype=bool)

        ch_batch = np.asarray(ch_all[s:e], dtype=np.int64)
        exc_batch = np.asarray(exc_all[s:e], dtype=np.int64)
        valid_input = np.ones((nb,), dtype=bool)
        if ch_limit > 0:
            valid_input &= (ch_batch >= 0) & (ch_batch < ch_limit)
        if exc_limit > 0:
            valid_input &= (exc_batch >= 0) & (exc_batch < exc_limit)

        if not np.all(valid_input) and not warned_invalid:
            bad_ch = sorted(set(ch_batch[~valid_input].tolist()))
            bad_exc = sorted(set(exc_batch[~valid_input].tolist()))
            print(
                f"[WARN] Some requested channels/excitations are outside checkpoint embeddings; "
                f"they will be written as zero/empty outputs. bad_ch={bad_ch}, bad_exc={bad_exc}",
                flush=True,
            )
            warned_invalid = True

        if np.any(valid_input):
            local_idx = np.where(valid_input)[0]
            x = torch.tensor(
                np.stack([H_all[s:e][valid_input], M_all[s:e][valid_input]], axis=1),
                dtype=torch.float32,
            )
            x_norm = in_norm.transform(x).to(device)
            Hn = x_norm[:, :1]
            Mn = x_norm[:, 1:]

            exc_t = torch.tensor(exc_batch[valid_input], dtype=torch.long, device=device)
            ch_t = torch.tensor(ch_batch[valid_input], dtype=torch.long, device=device)

            model_out = model(Hn, Mn, exc_t, ch_t)
            if isinstance(model_out, tuple):
                outL, outH = model_out
                out = outL if predict_output == "low" else outH
            else:
                out = model_out

            required = ("alpha", "omega", "rho", "sin_theta", "cos_theta")
            missing = [k for k in required if k not in out]
            if missing:
                raise KeyError(
                    f"Fixed-SF prediction expects model outputs {required}, but missing {missing}. "
                    f"Available keys: {sorted(out.keys())}"
                )

            alpha = out["alpha"].detach().cpu().numpy().astype(np.float64)
            omega_norm = out["omega"].detach().cpu().numpy().astype(np.float64)
            rho_raw = out["rho"].detach().cpu().numpy().astype(np.float64)
            sin_t = out["sin_theta"].detach().cpu().numpy().astype(np.float64)
            cos_t = out["cos_theta"].detach().cpu().numpy().astype(np.float64)

            # Numeric cleanup and conversion to the representation used by residue_pred.py.
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            omega = np.nan_to_num(omega_norm, nan=0.0, posinf=0.0, neginf=0.0) * float(omegaN)
            rho = np.exp(np.clip(np.nan_to_num(rho_raw, nan=-80.0, posinf=80.0, neginf=-80.0), -80.0, 80.0))

            norm = np.sqrt(np.maximum(sin_t * sin_t + cos_t * cos_t, 1e-12))
            sin_t = np.nan_to_num(sin_t / norm, nan=0.0, posinf=0.0, neginf=0.0)
            cos_t = np.nan_to_num(cos_t / norm, nan=1.0, posinf=1.0, neginf=1.0)

            freq = omega / (2.0 * np.pi)
            denom = np.sqrt(np.maximum(alpha * alpha + omega * omega, 1e-12))
            zeta = -alpha / denom

            valid_mask = (
                np.isfinite(alpha)
                & np.isfinite(omega)
                & np.isfinite(rho)
                & np.isfinite(sin_t)
                & np.isfinite(cos_t)
                & (alpha < 0.0)
                & (omega > 0.0)
                & (freq > 0.0)
                & (zeta > 0.0)
                & (zeta < 0.5)
                & (rho > 1e-12)
            )

            if mode_freq_ranges is not None:
                for local_j, ch_idx in enumerate(ch_batch[valid_input]):
                    ch_name = _base.channels[int(ch_idx)] if 0 <= int(ch_idx) < len(_base.channels) else f"C{int(ch_idx):02d}"
                    ranges = mode_freq_ranges.get(ch_name, [None, None, None])
                    for slot in range(valid_mask.shape[1]):
                        rg = ranges[slot] if slot < len(ranges) else None
                        if rg is None:
                            valid_mask[local_j, slot] = False
                            continue
                        fmin, fmax = rg
                        f_val = float(freq[local_j, slot])
                        valid_mask[local_j, slot] = bool(valid_mask[local_j, slot] and (fmin <= f_val <= fmax))

            alpha_full[local_idx] = alpha
            omega_full[local_idx] = omega
            rho_full[local_idx] = rho
            sin_full[local_idx] = sin_t
            cos_full[local_idx] = cos_t
            valid_full[local_idx] = valid_mask

        outs_acc["alpha"].append(alpha_full)
        outs_acc["omega"].append(omega_full)
        outs_acc["rho"].append(rho_full)
        outs_acc["sin_theta"].append(sin_full)
        outs_acc["cos_theta"].append(cos_full)
        outs_acc["valid_mask"].append(valid_full.astype(np.float32))

    return {k: np.concatenate(v, axis=0) for k, v in outs_acc.items()}


# Replace old components before calling original main.
_base.torch.load = _capture_torch_load
_base.MFDNN = _MFDNNFromCheckpoint
_base.run_infer_batched = run_infer_batched


if __name__ == "__main__":
    _base.main()
