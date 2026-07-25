"""Standalone OE modal-identification kernel aligned with analyze_signal_app.

This module deliberately contains its own implementation.  It does not import
the app, but follows the same model-order semantics, scaling, peak selection,
pole filtering, matching, and reduced reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.optimize import least_squares


SPECTRUM_PREPROCESS_OFF = {
    "demean": False,
    "detrend": False,
    "normalize": False,
    "max_normalize": False,
}

DEFAULT_PEAK_PARAMS: Dict[str, object] = {
    "fmin": 0.0,
    "fmax": 60.0,
    "median_width_hz": 0.25,
    "sg_width_hz": 0.8,
    "sg_order": 3,
    "min_distance_hz": 2.0,
    "min_prominence_db": 0.0,
    "min_width_hz": None,
    "max_width_hz": None,
    "min_height_ratio": 0.05,
    "linear_prominence_ratio": 0.01,
    "prominence_k": 6.0,
    "refine_width_hz": 0.4,
    "max_peaks": 2,
}

DEFAULT_POLE_FILTER_PARAMS = {
    "min_frequency": 5.0,
    "max_frequency": 70.0,
    "min_damping": 0.0,
    "max_damping": 0.3,
}

DEFAULT_MODAL_BANDS = [(0.0, 20.0), (20.0, 40.0), (40.0, 50.0)]
DEFAULT_MODAL_TOLERANCES = (2.0, 2.0, 5.0)


def _build_arx_regression(
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: int,
    nk: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    if y.shape != u.shape:
        raise ValueError(f"y and u must have same length, got {y.shape} vs {u.shape}")

    signal_size = y.size
    max_lag = max(na, nb + nk - 1)
    if signal_size <= max_lag:
        raise ValueError(
            f"data length {signal_size} too short for na={na}, nb={nb}, nk={nk}; "
            f"need at least {max_lag + 1} samples"
        )

    columns = []
    for shift in range(1, na + 1):
        column = np.zeros(signal_size)
        column[shift:] = -y[: signal_size - shift]
        columns.append(column)
    for shift in range(nk, nb + nk):
        column = np.zeros(signal_size)
        if shift < signal_size:
            column[shift:] = u[: signal_size - shift]
        columns.append(column)
    phi = np.column_stack(columns)
    return phi[max_lag:, :], y[max_lag:]


def _arx_init(
    y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    phi, target = _build_arx_regression(y, u, na, nb, nk=nk)
    theta, *_ = np.linalg.lstsq(phi, target, rcond=None)
    return np.hstack(([1.0], theta[:na])), theta[na:]


def _log_fact(n: int) -> float:
    return lgamma(n + 1.0)


def _rpm_filter(na: int, khat: int) -> np.ndarray:
    if khat < na:
        raise ValueError(f"khat must be >= na, got khat={khat}, na={na}")
    weights = np.zeros(khat - na + 1, dtype=float)
    log_denominator = _log_fact(na - 1) + (_log_fact(khat) - _log_fact(khat - na))
    for index in range(weights.size):
        log_term_1 = _log_fact(index + na - 1) - _log_fact(index)
        log_term_2 = _log_fact(khat - index) - _log_fact(khat - index - na)
        weights[index] = np.exp(log_term_1 + log_term_2 - log_denominator)
    return weights


def _rpm_init(
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: int,
    nk: int = 0,
    khat: int = 75,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    if y.shape != u.shape:
        raise ValueError("y and u must have same length")
    m = _rpm_filter(na=na, khat=khat)
    filtered_u = signal.lfilter(m, [1.0], u)
    filtered_y = signal.lfilter(m, [1.0], y)
    gamma = y - filtered_y
    signal_size = y.size
    max_lag = max(na, nb + nk - 1)
    target = (y - gamma)[max_lag:]
    columns = []
    for b_shift in range(nb):
        total_shift = nk + b_shift
        column = np.zeros(signal_size)
        if total_shift < signal_size:
            column[total_shift:] = filtered_u[: signal_size - total_shift]
        columns.append(column)
    for a_shift in range(1, na + 1):
        column = np.zeros(signal_size)
        column[a_shift:] = -filtered_y[: signal_size - a_shift]
        columns.append(column)
    phi = np.column_stack(columns)[max_lag:, :]
    theta, *_ = np.linalg.lstsq(phi, target, rcond=None)
    return np.hstack(([1.0], theta[nb:])), theta[:nb]


def stabilize_monic_polynomial(poly: np.ndarray, radius: float = 0.98) -> np.ndarray:
    poly = np.asarray(poly, dtype=float).ravel()
    if poly.size < 2:
        return poly
    roots = np.roots(poly)
    magnitudes = np.abs(roots)
    unstable = magnitudes > radius
    if not np.any(unstable):
        return poly
    roots[unstable] = roots[unstable] / magnitudes[unstable] * radius
    stabilized = np.real_if_close(np.poly(roots), tol=1e5).astype(float)
    return stabilized / stabilized[0]


def _oe_residual(params: np.ndarray, u: np.ndarray, y: np.ndarray, nb: int, nk: int) -> np.ndarray:
    b = params[:nb]
    f = np.hstack(([1.0], params[nb:]))
    b_effective = np.hstack((np.zeros(nk, dtype=float), b)) if nk > 0 else b
    return y - signal.lfilter(b_effective, f, u)


@dataclass
class OEModel:
    """Discrete OE model where ``nb`` is the number of B coefficients."""

    na: int = 30
    nb: int = 30
    nk: int = 0
    max_nfev: int = 50
    demean: bool = False
    rpm_khat: int = 75
    init_method: str = "rpm"
    B_: Optional[np.ndarray] = None
    F_: Optional[np.ndarray] = None
    fit_info_: Optional[Dict[str, Any]] = None

    def fit(self, y: np.ndarray, u: np.ndarray, x0: Optional[np.ndarray] = None) -> "OEModel":
        y = np.asarray(y, dtype=float).ravel()
        u = np.asarray(u, dtype=float).ravel()
        if y.shape != u.shape:
            raise ValueError(f"y and u must have same length, got {y.shape} vs {u.shape}")
        if y.size < 4:
            raise ValueError("OE identification needs at least 4 samples.")
        na = max(1, int(self.na))
        nb = max(1, int(self.nb))
        nk = max(0, int(self.nk))
        parameter_count = nb + na
        if y.size < parameter_count:
            raise ValueError(
                f"OE parameter count is {parameter_count}, but only {y.size} samples are available. "
                "Reduce A/F order or B order."
            )
        if self.demean:
            y = y - float(np.mean(y))
            u = u - float(np.mean(u))
        if x0 is None:
            if self.init_method.lower() == "rpm":
                khat = max(int(self.rpm_khat), na)
                f0, b0 = _rpm_init(y, u, na=na, nb=nb, nk=nk, khat=khat)
                f0 = stabilize_monic_polynomial(f0, radius=0.98)
            elif self.init_method.lower() == "arx":
                f0, b0 = _arx_init(y, u, na=na, nb=nb, nk=nk)
            else:
                raise ValueError("init_method must be 'rpm' or 'arx'")
            x0 = np.hstack((b0, f0[1:]))
        x0 = np.asarray(x0, dtype=float).ravel()
        if x0.size != parameter_count:
            raise ValueError(f"x0 length mismatch: expected {parameter_count}, got {x0.size}")
        initial_residual = _oe_residual(x0, u, y, nb, nk)
        if not np.all(np.isfinite(initial_residual)):
            f0, b0 = _arx_init(y, u, na=na, nb=nb, nk=nk)
            f0 = stabilize_monic_polynomial(f0, radius=0.98)
            x0 = np.hstack((b0, f0[1:]))
            initial_residual = _oe_residual(x0, u, y, nb, nk)
            if not np.all(np.isfinite(initial_residual)):
                raise ValueError("Initial OE residuals are not finite. Check data scaling or reduce orders.")
        result = least_squares(
            _oe_residual,
            x0,
            args=(u, y, nb, nk),
            method="lm",
            max_nfev=max(1, int(self.max_nfev)),
        )
        x = result.x
        self.na, self.nb, self.nk = na, nb, nk
        self.B_ = x[:nb].copy()
        self.F_ = np.hstack(([1.0], x[nb:])).copy()
        self.fit_info_ = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "init_method": self.init_method,
            "rpm_khat": int(self.rpm_khat) if self.init_method.lower() == "rpm" else None,
        }
        return self

    def _check_fitted(self) -> None:
        if self.B_ is None or self.F_ is None:
            raise RuntimeError("OEModel is not fitted yet. Call fit(y, u) first.")

    def simulate(self, u: np.ndarray) -> np.ndarray:
        self._check_fitted()
        u = np.asarray(u, dtype=float).ravel()
        b_effective = np.hstack((np.zeros(self.nk, dtype=float), self.B_)) if self.nk > 0 else self.B_
        return signal.lfilter(b_effective, self.F_, u)

    def predict(self, u: np.ndarray) -> np.ndarray:
        return self.simulate(u)

    def residuals(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        return y - self.simulate(u)

    def residues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_fitted()
        b_effective = np.hstack((np.zeros(self.nk, dtype=float), self.B_)) if self.nk > 0 else self.B_
        return signal.residuez(b_effective, self.F_)

    def continuous_poles(self, fs: float) -> np.ndarray:
        _, poles, _ = self.residues()
        return np.log(poles) * float(fs)

    def coefficients(self) -> np.ndarray:
        self._check_fitted()
        return np.hstack((self.B_, self.F_[1:]))


@dataclass(slots=True)
class ResidueMode:
    alpha: float
    omega: float
    residue_abs: float
    residue_phase: float

    @property
    def frequency(self) -> float:
        return abs(complex(self.alpha, self.omega)) / (2.0 * np.pi)


def standardize_pair(
    u: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    u_mean = float(np.mean(u))
    y_mean = float(np.mean(y))
    u_centered = u - u_mean
    y_centered = y - y_mean
    u_scale = float(np.std(u_centered))
    y_scale = float(np.std(y_centered))
    if u_scale <= 0 or y_scale <= 0:
        raise ValueError("Input or output signal has zero variance; OE identification is impossible.")
    return u_centered / u_scale, y_centered / y_scale, u_mean, u_scale, y_mean, y_scale


def fit_oe_modal(
    u: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    nb: int = 30,
    nf: int = 30,
    nk: int = 0,
    max_iter: int = 50,
) -> Dict[str, object]:
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if u.shape != y.shape:
        raise ValueError("Input and output signals must have the same length.")
    valid = np.isfinite(u) & np.isfinite(y)
    u, y = u[valid], y[valid]
    min_length = max(int(nf) + int(nb) + int(nk) + 20, 200)
    if u.size < min_length:
        raise ValueError("Not enough valid samples for OE identification.")
    u_fit, y_fit, u_mean, u_scale, y_mean, y_scale = standardize_pair(u, y)
    model = OEModel(
        na=int(nf),
        nb=int(nb),
        nk=int(nk),
        max_nfev=int(max_iter),
        init_method="arx",
        demean=False,
    ).fit(y_fit, u_fit)
    reconstructed = model.simulate(u_fit) * y_scale + y_mean
    if model.B_ is None or model.F_ is None:
        raise ValueError("OE model did not return B/F coefficients.")
    b_effective = np.hstack((np.zeros(int(nk), dtype=float), model.B_)) if int(nk) > 0 else model.B_
    residues, poles, direct_terms = signal.residuez(b_effective, model.F_)
    physical_scale = y_scale / u_scale
    b_physical = np.asarray(b_effective, dtype=float) * physical_scale
    residues = np.asarray(residues, dtype=complex) * physical_scale
    direct_terms = np.asarray(direct_terms, dtype=complex) * physical_scale
    frequency, damping = pole_frequency_damping(poles, fs)
    return {
        "model": model,
        "input": u,
        "output": y,
        "reconstructed": np.asarray(reconstructed, dtype=float),
        "B": b_physical,
        "F": np.asarray(model.F_, dtype=float),
        "poles": np.asarray(poles, dtype=complex),
        "residues": residues,
        "direct_terms": direct_terms,
        "frequency": frequency,
        "damping": damping,
        "reduced_excitation": np.asarray(u - u_mean, dtype=float),
        "output_scale": 1.0,
        "output_offset": y_mean,
        "iteration_text": str(model.fit_info_["nfev"] if model.fit_info_ else "--"),
    }


def pole_frequency_damping(
    poles: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    poles = np.asarray(poles, dtype=complex).ravel()
    frequency = np.full(poles.shape, np.nan, dtype=float)
    damping = np.full(poles.shape, np.nan, dtype=float)
    if poles.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            s_poles = np.log(poles) * float(fs)
            wn = np.abs(s_poles)
            frequency = wn / (2.0 * np.pi)
            damping = -np.real(s_poles) / wn
    return frequency, damping


def amplitude_spectrum(
    x: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(x, dtype=float).ravel()
    if y.size < 4 or not np.all(np.isfinite(y)):
        raise ValueError("Spectrum input must contain at least four finite samples.")
    coefficients = np.fft.rfft(y)
    frequency = np.fft.rfftfreq(y.size, d=1.0 / float(fs))
    amplitude = np.abs(coefficients) / y.size
    if amplitude.size > 2 and y.size % 2 == 0:
        amplitude[1:-1] *= 2.0
    elif amplitude.size > 1:
        amplitude[1:] *= 2.0
    return frequency.astype(float), amplitude.astype(float)


def robust_spectrum_peaks(freq, amp, **overrides) -> Dict[str, Any]:
    params = dict(DEFAULT_PEAK_PARAMS)
    params.update(overrides)
    freq = np.asarray(freq, dtype=float).reshape(-1)
    amp = np.abs(np.asarray(amp, dtype=float).reshape(-1))
    if freq.size != amp.size:
        raise ValueError("freq and amp must have the same length.")
    valid = np.isfinite(freq) & np.isfinite(amp)
    freq, amp = freq[valid], amp[valid]
    order = np.argsort(freq)
    freq, amp = freq[order], amp[order]
    eps = np.finfo(float).eps
    amp = np.where(amp > 0.0, amp, eps)
    mask = np.ones(freq.size, dtype=bool)
    if params["fmin"] is not None:
        mask &= freq >= float(params["fmin"])
    if params["fmax"] is not None:
        mask &= freq <= float(params["fmax"])
    freq, amp = freq[mask], amp[mask]
    if freq.size < 3:
        raise ValueError("Too few spectrum samples in the requested frequency range.")
    diffs = np.diff(freq)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("Frequency axis must be strictly increasing.")
    df = float(np.mean(diffs))
    use_db = bool(params.get("use_db", True))
    y_db = 20.0 * np.log10(amp + eps) if use_db else amp.copy()
    median_window = _hz_to_odd_window(float(params["median_width_hz"]), df, freq.size, minimum=3)
    sg_order = int(params["sg_order"])
    sg_window = _hz_to_odd_window(float(params["sg_width_hz"]), df, freq.size, minimum=max(5, sg_order + 2))
    if sg_window <= sg_order:
        sg_window = min(freq.size if freq.size % 2 else freq.size - 1, sg_order + 2 + ((sg_order + 2) % 2 == 0))
    y_median = signal.medfilt(y_db, kernel_size=median_window) if median_window >= 3 else y_db.copy()
    y_smooth = y_median.copy() if sg_window <= sg_order or sg_window < 3 else signal.savgol_filter(y_median, sg_window, min(sg_order, sg_window - 1))
    amp_median = signal.medfilt(amp, kernel_size=median_window) if median_window >= 3 else amp.copy()
    amp_smooth = amp_median.copy() if sg_window <= sg_order or sg_window < 3 else signal.savgol_filter(amp_median, sg_window, min(sg_order, sg_window - 1))
    min_prominence_db = params.get("min_prominence_db")
    if min_prominence_db is None:
        residual = y_db - y_smooth
        center = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - center)))
        min_prominence_db = max(3.0, float(params["prominence_k"]) * sigma)
    min_prominence_db = float(min_prominence_db)
    distance = max(1, int(round(float(params["min_distance_hz"]) / df)))
    width_arg = _width_points_arg(params, df)
    db_candidates, _ = signal.find_peaks(y_smooth, distance=distance, prominence=min_prominence_db, width=width_arg)
    amp_span = float(np.nanmax(amp_smooth) - np.nanmin(amp_smooth))
    linear_prominence = max(eps, float(params["linear_prominence_ratio"]) * amp_span)
    linear_candidates, _ = signal.find_peaks(amp_smooth, distance=distance, prominence=linear_prominence, width=width_arg)
    candidates = np.array(sorted(set(map(int, db_candidates)) | set(map(int, linear_candidates))), dtype=int)
    min_height = 0.0 if params["min_height_ratio"] is None else float(params["min_height_ratio"]) * float(np.nanmax(amp))
    rows: List[Dict[str, float]] = []
    for index in candidates:
        refined_freq, peak_amp, peak_amp_db = _refine_peak(freq, amp, y_db, y_median, index, float(params["refine_width_hz"]))
        if peak_amp < min_height:
            continue
        prominence_db = _safe_peak_prominence(y_smooth, index)
        prominence_linear = _safe_peak_prominence(amp_smooth, index)
        width_hz = _safe_peak_width(amp_smooth, index) * df
        if params["min_width_hz"] is not None and width_hz < float(params["min_width_hz"]):
            continue
        if params["max_width_hz"] is not None and width_hz > float(params["max_width_hz"]):
            continue
        relative_amp = peak_amp / max(float(np.nanmax(amp)), eps)
        rows.append({
            "peak_freq": refined_freq,
            "peak_amp": peak_amp,
            "peak_amp_db": peak_amp_db,
            "prominence_db": prominence_db,
            "prominence_linear": prominence_linear,
            "width_hz": width_hz,
            "candidate_freq": float(freq[index]),
            "score": prominence_db + 20.0 * relative_amp,
        })
    rows = _deduplicate_peaks(rows, min_distance_hz=float(params["min_distance_hz"]))
    max_peaks = params.get("max_peaks")
    if max_peaks is not None and int(max_peaks) > 0 and len(rows) > int(max_peaks):
        rows = sorted(rows, key=lambda item: (item["score"], item["prominence_linear"], item["peak_amp"]), reverse=True)[: int(max_peaks)]
    rows = sorted(rows, key=lambda item: item["peak_freq"])
    return {
        "peak_freq": np.array([row["peak_freq"] for row in rows], dtype=float),
        "peak_amp": np.array([row["peak_amp"] for row in rows], dtype=float),
        "peak_amp_db": np.array([row["peak_amp_db"] for row in rows], dtype=float),
        "prominence_db": np.array([row["prominence_db"] for row in rows], dtype=float),
        "prominence_linear": np.array([row["prominence_linear"] for row in rows], dtype=float),
        "width_hz": np.array([row["width_hz"] for row in rows], dtype=float),
        "candidate_freq": np.array([row["candidate_freq"] for row in rows], dtype=float),
        "score": np.array([row["score"] for row in rows], dtype=float),
        "freq_used": freq,
        "amp_used": amp,
        "smooth_db": y_smooth,
        "smooth_amp": amp_smooth,
        "median_db": y_median,
        "params": dict(params, median_window=median_window, sg_window=sg_window, distance=distance),
    }


def marker_slots_from_bands(
    frequency: np.ndarray,
    amplitude: np.ndarray,
    bands: List[Optional[Tuple[float, float]]],
    *,
    peak_params: Optional[Dict[str, object]] = None,
) -> List[Optional[float]]:
    freq = np.asarray(frequency, dtype=float).ravel()
    amp = np.abs(np.asarray(amplitude, dtype=float).ravel())
    params = dict(DEFAULT_PEAK_PARAMS)
    params.update(peak_params or {})
    band_slots: List[Optional[Tuple[float, float]]] = []
    for band in (list(bands[:3]) + [None, None, None])[:3]:
        if band is None:
            band_slots.append(None)
            continue
        low, high = band
        fmin = max(float(low), float(params["fmin"])) if params.get("fmin") is not None else float(low)
        fmax = min(float(high), float(params["fmax"])) if params.get("fmax") is not None else float(high)
        band_slots.append((fmin, fmax) if np.isfinite(fmin) and np.isfinite(fmax) and fmax > fmin else None)
    valid_bands = [band for band in band_slots if band is not None]
    if not valid_bands:
        return [None, None, None]
    search_fmin = min(low for low, _ in valid_bands)
    search_fmax = max(high for _, high in valid_bands)
    candidate_freq, candidate_amp = _panel_controlled_peak_candidates(freq, amp, search_fmin, search_fmax, params)
    if candidate_freq.size == 0:
        return [None, None, None]
    markers: List[Optional[float]] = []
    for band in band_slots:
        if band is None:
            markers.append(None)
            continue
        low, high = band
        band_mask = np.isfinite(freq) & np.isfinite(amp) & (freq >= low) & (freq < high)
        candidate_mask = np.isfinite(candidate_freq) & np.isfinite(candidate_amp) & (candidate_freq >= low) & (candidate_freq < high)
        if params.get("min_height_ratio") is not None and np.any(band_mask):
            candidate_mask &= candidate_amp >= float(params["min_height_ratio"]) * float(np.nanmax(amp[band_mask]))
        indexes = np.flatnonzero(candidate_mask)
        markers.append(None if indexes.size == 0 else float(candidate_freq[indexes[int(np.argmax(candidate_amp[indexes]))]]))
    return (markers + [None, None, None])[:3]


def match_modes_to_marker_frequencies(
    peak_frequencies: np.ndarray,
    poles: np.ndarray,
    residues: np.ndarray,
    fs: float,
    *,
    f_threshold: float = 5.0,
    f_thresholds: Optional[np.ndarray] = None,
    min_frequency: float = 5.0,
    max_frequency: float = 70.0,
    min_damping: float = 0.0,
    max_damping: float = 0.3,
) -> List[Tuple[float, ResidueMode]]:
    poles = np.asarray(poles, dtype=complex).ravel()
    residues = np.asarray(residues, dtype=complex).ravel()
    count = min(poles.size, residues.size)
    poles, residues = poles[:count], residues[:count]
    if count == 0:
        return []
    with np.errstate(divide="ignore", invalid="ignore"):
        s_poles = np.log(poles) * float(fs)
        frequencies = np.abs(s_poles) / (2.0 * np.pi)
        damping = -np.real(s_poles) / np.abs(s_poles)
    valid = np.isfinite(frequencies) & np.isfinite(damping)
    valid &= (frequencies >= min_frequency) & (frequencies <= max_frequency)
    valid &= (damping >= min_damping) & (damping <= max_damping)
    markers = np.asarray(peak_frequencies, dtype=float).ravel()
    thresholds = np.full(markers.size, float(f_threshold)) if f_thresholds is None else np.asarray(f_thresholds, dtype=float).ravel()
    if thresholds.size < markers.size:
        thresholds = np.pad(thresholds, (0, markers.size - thresholds.size), constant_values=float(f_threshold))
    used = np.zeros(count, dtype=bool)
    matched: List[Tuple[float, ResidueMode]] = []
    for marker_index, target in enumerate(markers):
        threshold = float(thresholds[marker_index])
        candidates = np.flatnonzero(valid & (np.abs(frequencies - target) <= threshold))
        positive_indexes, conjugate_indexes = [], []
        for pole_index in candidates:
            if used[pole_index] or np.imag(s_poles[pole_index]) <= 1e-8:
                continue
            conjugate_target = complex(np.real(s_poles[pole_index]), -np.imag(s_poles[pole_index]))
            positive_indexes.append(int(pole_index))
            conjugate_indexes.append(int(np.argmin(np.abs(s_poles - conjugate_target))))
        if not positive_indexes:
            continue
        distances = np.abs(frequencies[positive_indexes] - target)
        best_local = int(np.argmin(distances))
        best_index = positive_indexes[best_local]
        used[[best_index, conjugate_indexes[best_local]]] = True
        pole, residue = s_poles[best_index], residues[best_index]
        matched.append((float(target), ResidueMode(float(np.real(pole)), float(np.imag(pole)), float(np.abs(residue)), float(np.angle(residue)))))
    return matched


def build_ba_from_residue_modes(
    modes: List[ResidueMode], fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    if not modes:
        return np.array([0.0]), np.array([1.0])
    ts = 1.0 / float(fs)
    p_plus = np.exp(np.array([complex(mode.alpha, mode.omega) for mode in modes]) * ts)
    poles = np.concatenate([p_plus, np.conj(p_plus)])
    r_plus = np.array([mode.residue_abs * np.exp(1j * mode.residue_phase) for mode in modes])
    residues = np.concatenate([r_plus, np.conj(r_plus)])
    b, a = signal.invresz(residues, poles, 0)
    b = np.real_if_close(b, tol=1e5).astype(float)
    a = np.real_if_close(a, tol=1e5).astype(float)
    if a.size and a[0] != 0.0:
        b, a = b / a[0], a / a[0]
    return b, a


def simulate_reduced_response(
    excitation: np.ndarray,
    modes: List[ResidueMode],
    fs: float,
    *,
    output_scale: float = 1.0,
    output_offset: float = 0.0,
) -> np.ndarray:
    b, a = build_ba_from_residue_modes(modes, fs)
    y = signal.lfilter(b, a, np.asarray(excitation, dtype=float).ravel())
    return np.asarray(y, dtype=float) * float(output_scale) + float(output_offset)


def _panel_controlled_peak_candidates(freq, amp, fmin, fmax, params):
    mask = np.isfinite(freq) & np.isfinite(amp) & (freq >= fmin) & (freq < fmax)
    indexes = np.flatnonzero(mask)
    if indexes.size < 3:
        return np.array([]), np.array([])
    local_freq, local_amp = freq[indexes], np.abs(amp[indexes])
    order = np.argsort(local_freq)
    local_freq, local_amp = local_freq[order], local_amp[order]
    diffs = np.diff(local_freq)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.array([]), np.array([])
    df = float(np.median(diffs))
    peak_signal = 20.0 * np.log10(np.where(local_amp > 0, local_amp, np.finfo(float).eps))
    sg_order = int(params.get("sg_order", 3) or 3)
    if params.get("median_width_hz") is not None:
        window = _hz_to_odd_window(float(params["median_width_hz"]), df, peak_signal.size, minimum=3)
        if window >= 3:
            peak_signal = signal.medfilt(peak_signal, kernel_size=window)
    if params.get("sg_width_hz") is not None:
        window = _hz_to_odd_window(float(params["sg_width_hz"]), df, peak_signal.size, minimum=max(5, sg_order + 2))
        if window > sg_order and window >= 3:
            peak_signal = signal.savgol_filter(peak_signal, window, min(sg_order, window - 1))
    distance = max(1, int(round(float(params.get("min_distance_hz", 1.0) or 1.0) / df)))
    kwargs: Dict[str, object] = {"distance": distance}
    if params.get("min_prominence_db") is not None:
        kwargs["prominence"] = max(0.0, float(params["min_prominence_db"]))
    width = _width_points_arg(params, df)
    if width is not None:
        kwargs["width"] = width
    peaks, _ = signal.find_peaks(peak_signal, **kwargs)
    return local_freq[peaks], local_amp[peaks]


def _hz_to_odd_window(width_hz: float, df: float, size: int, *, minimum: int) -> int:
    points = max(int(round(width_hz / df)), int(minimum))
    if points % 2 == 0:
        points += 1
    points = min(points, size if size % 2 else size - 1)
    if points % 2 == 0:
        points -= 1
    return max(points, 1)


def _width_points_arg(params: Dict[str, object], df: float):
    min_width, max_width = params.get("min_width_hz"), params.get("max_width_hz")
    if min_width is None and max_width is None:
        return None
    min_points = None if min_width is None else _optional_width_points(min_width, df, "min_width_hz")
    max_points = None if max_width is None else _optional_width_points(max_width, df, "max_width_hz")
    if min_points is not None and max_points is not None and min_points > max_points:
        raise ValueError("min_width_hz cannot be greater than max_width_hz.")
    return min_points, max_points


def _optional_width_points(value: object, df: float, label: str) -> float:
    width = float(value)
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"{label} must be a positive finite number.")
    return max(1.0, width / df)


def _safe_peak_prominence(values: np.ndarray, index: int) -> float:
    try:
        return float(signal.peak_prominences(values, np.array([index]))[0][0])
    except Exception:
        return 0.0


def _safe_peak_width(values: np.ndarray, index: int) -> float:
    try:
        return float(signal.peak_widths(values, np.array([index]), rel_height=0.5)[0][0])
    except Exception:
        return 1.0


def _refine_peak(freq, amp, y_db, y_median, candidate_index, refine_width_hz):
    center = float(freq[candidate_index])
    local = np.flatnonzero((freq >= center - refine_width_hz) & (freq <= center + refine_width_hz))
    if local.size == 0:
        local = np.array([candidate_index])
    local_best = int(local[int(np.argmax(y_median[local]))])
    fit_idx = np.arange(max(0, local_best - 4), min(freq.size, local_best + 5))
    refined = float(freq[local_best])
    if fit_idx.size >= 5:
        a, b, _ = map(float, np.polyfit(freq[fit_idx], y_db[fit_idx], 2))
        vertex = -b / (2.0 * a) if a != 0 else np.nan
        if a < 0 and np.isfinite(vertex) and freq[local[0]] <= vertex <= freq[local[-1]]:
            refined = float(vertex)
    peak_amp = float(np.interp(refined, freq, amp))
    return refined, peak_amp, float(20.0 * np.log10(max(peak_amp, np.finfo(float).eps)))


def _deduplicate_peaks(rows, *, min_distance_hz):
    kept = []
    for row in sorted(rows, key=lambda item: (item["score"], item["prominence_linear"], item["peak_amp"]), reverse=True):
        if all(abs(row["peak_freq"] - other["peak_freq"]) >= min_distance_hz for other in kept):
            kept.append(row)
    return kept
