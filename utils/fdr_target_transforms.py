from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch


DAMPING_MAX_BY_EXCITATION = {
    0: 0.15,  # FXD_FDC
    1: 0.15,  # FXD_DC
    2: 0.30,  # PW_FDC
    3: 0.30,  # PW_DC
    4: 0.30,  # JFY_FDC
    5: 0.30,  # JFY_DC
}


def damping_max_for_excitation(excitation_id: int) -> float:
    try:
        return float(DAMPING_MAX_BY_EXCITATION[int(excitation_id)])
    except KeyError as exc:
        raise ValueError(f"No damping upper bound is configured for excitation_id={excitation_id}") from exc


def build_target_transform_specs(
    target_names: Iterable[str], excitation_ids: Iterable[int], *, enabled: bool
) -> Dict[str, dict]:
    names = [str(name) for name in target_names]
    if not enabled:
        return {name: {"kind": "identity"} for name in names}
    specs: Dict[str, dict] = {}
    for name in names:
        if name == "freq":
            specs[name] = {"kind": "log", "epsilon": 1e-8}
        elif name == "damping":
            # Preserve the smooth direct-target behavior of the first split
            # FDR models. Non-negativity is enforced only at inference time.
            specs[name] = {"kind": "identity"}
        elif name == "magnitude":
            specs[name] = {"kind": "log", "epsilon": 1e-12}
        else:
            specs[name] = {"kind": "identity"}
    return specs


def _spec(spec: Optional[Mapping]) -> dict:
    return dict(spec or {"kind": "identity"})


def encode_numpy(values, spec: Optional[Mapping]):
    values = np.asarray(values)
    cfg = _spec(spec)
    kind = str(cfg.get("kind", "identity"))
    if kind == "identity":
        return values
    if kind == "log":
        epsilon = float(cfg.get("epsilon", 1e-12))
        return np.log(np.maximum(values, epsilon))
    if kind == "scaled_logit":
        minimum = float(cfg.get("minimum", 0.0))
        maximum = float(cfg["maximum"])
        epsilon = float(cfg.get("epsilon", 1e-5))
        unit = (values - minimum) / (maximum - minimum)
        unit = np.clip(unit, epsilon, 1.0 - epsilon)
        return np.log(unit) - np.log1p(-unit)
    if kind == "scaled_linear":
        minimum = float(cfg.get("minimum", 0.0))
        maximum = float(cfg["maximum"])
        return (values - minimum) / (maximum - minimum)
    raise ValueError(f"Unsupported target transform kind={kind!r}")


def decode_numpy(values, spec: Optional[Mapping]):
    values = np.asarray(values)
    cfg = _spec(spec)
    kind = str(cfg.get("kind", "identity"))
    if kind == "identity":
        return values
    if kind == "log":
        return np.exp(np.clip(values, -80.0, 80.0))
    if kind == "scaled_logit":
        minimum = float(cfg.get("minimum", 0.0))
        maximum = float(cfg["maximum"])
        clipped = np.clip(values, -80.0, 80.0)
        unit = 1.0 / (1.0 + np.exp(-clipped))
        return minimum + (maximum - minimum) * unit
    if kind == "scaled_linear":
        minimum = float(cfg.get("minimum", 0.0))
        maximum = float(cfg["maximum"])
        unit = np.clip(values, 0.0, 1.0)
        return minimum + (maximum - minimum) * unit
    raise ValueError(f"Unsupported target transform kind={kind!r}")


def decode_torch(values: torch.Tensor, spec: Optional[Mapping]) -> torch.Tensor:
    cfg = _spec(spec)
    kind = str(cfg.get("kind", "identity"))
    if kind == "identity":
        return values
    if kind == "log":
        return torch.exp(torch.clamp(values, min=-80.0, max=80.0))
    if kind == "scaled_logit":
        minimum = float(cfg.get("minimum", 0.0))
        maximum = float(cfg["maximum"])
        return minimum + (maximum - minimum) * torch.sigmoid(values)
    if kind == "scaled_linear":
        minimum = float(cfg.get("minimum", 0.0))
        maximum = float(cfg["maximum"])
        return minimum + (maximum - minimum) * torch.clamp(values, 0.0, 1.0)
    raise ValueError(f"Unsupported target transform kind={kind!r}")
