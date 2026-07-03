from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Reuse the existing prediction script for task parsing, response reconstruction,
# and PR/HF/HG/HA/VF/VG/VA file writing. Only replace the model-output adapter.
try:
    _base = importlib.import_module("test.Predict_user")
except ModuleNotFoundError:
    _base = importlib.import_module("surrogate_model.test.Predict_user")


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
    outs_acc = {k: [] for k in ["alpha", "omega", "rho", "sin_theta", "cos_theta", "valid_mask"]}

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)

        x = torch.tensor(np.stack([H_all[s:e], M_all[s:e]], axis=1), dtype=torch.float32)
        x_norm = in_norm.transform(x).to(device)
        Hn = x_norm[:, :1]
        Mn = x_norm[:, 1:]

        exc_t = torch.tensor(exc_all[s:e], dtype=torch.long, device=device)
        ch_t = torch.tensor(ch_all[s:e], dtype=torch.long, device=device)

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
            for local_i, ch_idx in enumerate(ch_all[s:e]):
                ch_name = _base.channels[int(ch_idx)] if 0 <= int(ch_idx) < len(_base.channels) else f"C{int(ch_idx):02d}"
                ranges = mode_freq_ranges.get(ch_name, [None, None, None])
                for slot in range(valid_mask.shape[1]):
                    rg = ranges[slot] if slot < len(ranges) else None
                    if rg is None:
                        valid_mask[local_i, slot] = False
                        continue
                    fmin, fmax = rg
                    f_val = float(freq[local_i, slot])
                    valid_mask[local_i, slot] = bool(valid_mask[local_i, slot] and (fmin <= f_val <= fmax))

        outs_acc["alpha"].append(alpha)
        outs_acc["omega"].append(omega)
        outs_acc["rho"].append(rho)
        outs_acc["sin_theta"].append(sin_t)
        outs_acc["cos_theta"].append(cos_t)
        outs_acc["valid_mask"].append(valid_mask.astype(np.float32))

    return {k: np.concatenate(v, axis=0) for k, v in outs_acc.items()}


# Replace the old freq/damping/residue_re/residue_im adapter before calling original main.
_base.run_infer_batched = run_infer_batched


if __name__ == "__main__":
    _base.main()
