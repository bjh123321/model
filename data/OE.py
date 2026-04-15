# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
from math import lgamma


def _build_arx_regression(
    y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int = 0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    构造 ARX 回归矩阵 Phi 与目标 Y_valid.
    采用与你脚本一致的约定：y 滞后项系数前带负号。
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    if y.shape != u.shape:
        raise ValueError(f"y and u must have same length, got {y.shape} vs {u.shape}")

    N = len(y)
    max_lag = max(na, nb + nk)
    if N <= max_lag:
        raise ValueError(
            f"data length {N} too short for na={na}, nb={nb}, nk={nk}; need >= {max_lag + 1}"
        )

    phi_cols = []

    # y 的滞后项：-y(t-1..t-na)
    for i in range(1, na + 1):
        col = np.zeros(N)
        col[i:] = -y[: N - i]
        phi_cols.append(col)

    # u 的滞后项：u(t-nk .. t-(nk+nb))
    for i in range(nk, nb + nk + 1):
        col = np.zeros(N)
        if i < N:
            col[i:] = u[: N - i]
        phi_cols.append(col)

    Phi = np.column_stack(phi_cols)
    Y_valid = y[max_lag:]
    Phi_valid = Phi[max_lag:, :]
    return Phi_valid, Y_valid, max_lag


def arx_init(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    用最小二乘辨识 ARX，返回 (A, B)：
    A = [1, a1, ..., ana]
    B = [b0, b1, ..., bnb]  (对应 u 的 nk 延迟已体现在回归构造中)
    """
    Phi_valid, Y_valid, _ = _build_arx_regression(y, u, na, nb, nk=nk)
    theta, *_ = np.linalg.lstsq(Phi_valid, Y_valid, rcond=None)

    A = np.hstack(([1.0], theta[:na]))
    B = theta[na:]
    return A, B


# ------------------ RPM 初始化（论文对齐） ------------------
def _log_fact(n: int) -> float:
    # log(n!) = lgamma(n+1)
    return lgamma(n + 1.0)


def _rpm_filter_m(na: int, khat: int) -> np.ndarray:
    """
    生成论文里 RPM 的隐式 FIR 滤波器 M(q)=sum_{i=0..khat-na} m_i q^{-i}
    为了数值稳定，用 log-gamma 在 log 域计算：
      m_i = ((i+na-1)!/i!) * ((khat-i)!/(khat-i-na)!) / ((na-1)! * (khat)!/(khat-na)!)
    """
    if khat < na:
        raise ValueError(f"khat must be >= na, got khat={khat}, na={na}")

    L = khat - na + 1
    m = np.zeros(L, dtype=float)

    # 常数项：log((na-1)! * khat!/(khat-na)!)
    log_den = _log_fact(na - 1) + (_log_fact(khat) - _log_fact(khat - na))

    for i in range(L):
        # log((i+na-1)!/i!)
        log_term1 = _log_fact(i + na - 1) - _log_fact(i)
        # log((khat-i)!/(khat-i-na)!)
        log_term2 = _log_fact(khat - i) - _log_fact(khat - i - na)
        m[i] = np.exp(log_term1 + log_term2 - log_den)

    return m


def rpm_init(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int = 0, khat: int = 75) -> Tuple[np.ndarray, np.ndarray]:
    """
    用论文中的 DT-RPM 模型做 OE 初始化，得到 (A,B) 供 OE 的 (F,B) 初值使用。
    这里实现的是论文的“隐式滤波 + 线性最小二乘”的 RPM 初始化思路，
    并保持 nk 结构（把 nk 作为输入延迟等价实现）。

    返回：
      A = [1, a1, ..., ana]  (作为 OE 的 F 初值)
      B = [b0, ..., bnb]     (作为 OE 的 B 初值；延迟 nk 不在系数里体现，而在仿真时处理)
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    if y.shape != u.shape:
        raise ValueError("y and u must have same length")

    # 构造隐式滤波器 M(q)
    m = _rpm_filter_m(na=na, khat=khat)

    # M(q)u, M(q)y
    Mu = signal.lfilter(m, [1.0], u)
    My = signal.lfilter(m, [1.0], y)

    # gamma_y = (1 - M(q)) y = y - M(q)y
    gamma = y - My

    N = len(y)
    # RPM 回归里需要用到 u、y 的滞后项；并且 nk 会让 u 的“有效起点”后移
    max_lag = max(na, nb + nk)

    # 目标：y - gamma = Σ b_m * q^{-(nk+m)} Mu + Σ a_n * ( - q^{-n} My )
    Y = (y - gamma)[max_lag:]

    cols = []

    # beta_u_m(k) = q^{-(nk+m)} Mu(k)
    for m_shift in range(0, nb + 1):
        total_shift = nk + m_shift
        col = np.zeros(N)
        if total_shift < N:
            col[total_shift:] = Mu[: N - total_shift]
        cols.append(col)

    # alpha_y_n(k) = - q^{-n} My(k), n=1..na
    for n_shift in range(1, na + 1):
        col = np.zeros(N)
        col[n_shift:] = -My[: N - n_shift]
        cols.append(col)

    Phi = np.column_stack(cols)[max_lag:, :]

    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    B = theta[: nb + 1]
    a = theta[nb + 1:]
    A = np.hstack(([1.0], a))
    return A, B


def _stabilize_monic_poly(F: np.ndarray, radius: float = 0.98) -> np.ndarray:
    """
    把 monic 多项式 F(z)=1+f1 z^-1+... 的根投影到 |z|<radius 内。
    只做初始化护栏：不改变优化目标，只保证初始点可用。
    """
    F = np.asarray(F, dtype=float).ravel()
    if F.size < 2:
        return F

    # roots of F in z-domain (for polynomial in z^-1, roots() 仍按系数序列求根)
    r = np.roots(F)

    # 投影：|r|>radius 的根缩放到 radius
    mag = np.abs(r)
    mask = mag > radius
    if np.any(mask):
        r[mask] = r[mask] / mag[mask] * radius
        F_stab = np.real_if_close(np.poly(r), tol=1e5).astype(float)

        # 归一化为 monic（理论上 poly 返回首项=1，但做个保险）
        F_stab = F_stab / F_stab[0]
        return F_stab

    return F



# ------------------ OE 残差（对齐 MATLAB：LM + zero IC） ------------------
def _oe_residual(params: np.ndarray, u: np.ndarray, y: np.ndarray, nb: int, nk: int) -> np.ndarray:
    """
    OE 残差：e = y - lfilter(B(q) q^{-nk}, F(q), u)
    params = [b0..bnb, f1..f_na] 其中 F=[1,f1..]
    - 初始条件：zero（lfilter 默认就是零初始状态，行为对齐你 MATLAB 的 InitialCondition='zero'）
    """
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    b = params[: nb + 1]
    f = np.hstack(([1.0], params[nb + 1:]))

    # 把 nk 作为纯延迟：等价于 B 前补 nk 个零
    if nk > 0:
        b_eff = np.hstack((np.zeros(nk, dtype=float), b))
    else:
        b_eff = b

    y_hat = signal.lfilter(b_eff, f, u)
    return y - y_hat


@dataclass
class OEModel:
    """
    轻量 OE：
      y(t) = (B(q)/F(q)) u(t) + e(t)

    对齐要求：
    - 初始化：默认使用论文的 RPM（用于 OE 初值），替代原 ARX 初始化
    - 优化：LM（scipy least_squares(method='lm')），zero 初始条件
    """

    na: int = 100
    nb: int = 50
    nk: int = 0

    enforce_stability: bool = False
    stability_weight: float = 0.0

    # MATLAB SearchOptions.MaxIterations 对齐：这里用 max_nfev 近似（LM 的最大函数评估次数）
    max_nfev: int = 50

    # MATLAB 脚本里你并没有去均值；默认 False 更贴近
    demean: bool = False

    # RPM 设计参数（论文建议典型 60~90，示例最优 75）
    rpm_khat: int = 75

    # 初始化方式：'rpm' 或 'arx'
    init_method: str = "rpm"

    # 拟合后参数
    B_: Optional[np.ndarray] = None
    F_: Optional[np.ndarray] = None
    fit_info_: Optional[Dict[str, Any]] = None

    def fit(self, y: np.ndarray, u: np.ndarray, x0: Optional[np.ndarray] = None) -> "OEModel":
        y = np.asarray(y, dtype=float).ravel()
        u = np.asarray(u, dtype=float).ravel()
        if y.shape != u.shape:
            raise ValueError(f"y and u must have same length, got {y.shape} vs {u.shape}")

        if self.demean:
            y = y - y.mean()
            u = u - u.mean()

        if x0 is None:
            if self.init_method.lower() == "rpm":
                A0, B0 = rpm_init(y, u, na=self.na, nb=self.nb, nk=self.nk, khat=self.rpm_khat)
            elif self.init_method.lower() == "arx":
                A0, B0 = arx_init(y, u, na=self.na, nb=self.nb, nk=self.nk)
            else:
                raise ValueError("init_method must be 'rpm' or 'arx'")

            # 稳定化护栏：仅对 RPM 初值做
            if self.init_method.lower() == "rpm":
                A0 = _stabilize_monic_poly(A0, radius=0.98)

            x0 = np.hstack((B0, A0[1:]))

        # ---- 初始点有限性检查：不通过就回退到 ARX（再稳定化一次）----
        x0 = np.asarray(x0, dtype=float).ravel()
        expected = (self.nb + 1) + self.na
        if x0.size != expected:
            raise ValueError(f"x0 length mismatch: expect {(self.nb+1)}+{self.na}={expected}, got {x0.size}")

        e0 = _oe_residual(x0, u, y, self.nb, self.nk)
        if not np.all(np.isfinite(e0)):
            # 回退 ARX 初值（更稳），并做一次稳定化
            A1, B1 = arx_init(y, u, na=self.na, nb=self.nb, nk=self.nk)
            A1 = _stabilize_monic_poly(A1, radius=0.98)
            x0 = np.hstack((B1, A1[1:]))
            e1 = _oe_residual(x0, u, y, self.nb, self.nk)
            if not np.all(np.isfinite(e1)):
                raise ValueError("Initial residuals are not finite even after fallback (ARX). Check data scaling or reduce na/nb.")


        res = least_squares(
            _oe_residual,
            x0,
            args=(u, y, self.nb, self.nk),
            method="lm",               # 对齐 MATLAB SearchMethod='lm'
            max_nfev=self.max_nfev,    # 近似对齐 MaxIterations
        )

        x = res.x
        self.B_ = x[: self.nb + 1].copy()
        self.F_ = np.hstack(([1.0], x[self.nb + 1 :])).copy()

        njev = getattr(res, "njev", None)
        self.fit_info_ = {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "cost": float(res.cost),
            "nfev": int(res.nfev),
            "njev": -1 if njev is None else int(njev),
            "init_method": self.init_method,
            "rpm_khat": int(self.rpm_khat) if self.init_method.lower() == "rpm" else None,
        }
        return self

    def simulate(self, u: np.ndarray) -> np.ndarray:
        self._check_fitted()
        u = np.asarray(u, dtype=float).ravel()

        if self.nk > 0:
            b_eff = np.hstack((np.zeros(self.nk, dtype=float), self.B_))
        else:
            b_eff = self.B_

        return signal.lfilter(b_eff, self.F_, u)

    def predict(self, u: np.ndarray) -> np.ndarray:
        return self.simulate(u)

    def residuals(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y = np.asarray(y, dtype=float).ravel()
        u = np.asarray(u, dtype=float).ravel()
        if y.shape != u.shape:
            raise ValueError(f"y and u must have same length, got {y.shape} vs {u.shape}")
        return y - self.simulate(u)

    def residues(self):
        self._check_fitted()
        # 注意：这里 residuez 用的是“无延迟的 B_”；延迟 nk 是结构信息，不改变 B_ 系数本身
        return signal.residuez(self.B_, self.F_)

    def continuous_poles(self, fs: float) -> np.ndarray:
        self._check_fitted()
        Ts = 1.0 / float(fs)
        _, poles, _ = self.residues()
        return np.log(poles + 1e-12) / Ts

    def _check_fitted(self):
        if self.B_ is None or self.F_ is None:
            raise RuntimeError("OEModel is not fitted yet. Call fit(y,u) first.")


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd
    from scipy.signal import find_peaks

    from envelop import compute_spectrum, envelope_rms_tv, moving_rms

    def block_mean(x, window_len):
        x = np.asarray(x)
        means = []
        starts = []   # 每个块在原始信号中的起点
        ends = []
        for i in range(0, len(x), window_len):
            j = min(i + window_len, len(x))
            means.append(np.mean(x[i:j]))
            starts.append(i)
            ends.append(j)
        return np.array(means), np.array(starts), np.array(ends)
    

    def find_excitation_breakpoint_from_mean(means, starts, ratio=0.2):
        """
        means  : block_mean 得到的均值序列
        starts : 每个块的起点索引
        ratio  : 去除比例（0.2 表示前后各去 20%）
        """
        n = len(means)
        k = int(n * ratio)

        # 中间部分
        mid_means = means[k : n-k]

        # 找最小值
        idx_mid = np.argmin(mid_means)

        # 映射回原始 block 索引
        idx_block = idx_mid + k

        # 对应原始信号起点
        i_break = starts[idx_block]
        return i_break, idx_block


    fs = 512
    data_file = Path(r"D:\BaiduSyncdisk\项目\试飞数据\35A\试飞数据-ReConstr\H8d23-ReConstr\20250919_H8d23_M1d4-ReConstr\H8d23_M1d4_FXD_反_幅值0d6-ReConstr.txt")
    # data_file = Path(r"C:\Users\Leo\Desktop\扫频响应（F=10.000.01&d=32.0000.020&noise=0.100）仿真信号.txt")
    df = pd.read_table(data_file, sep=r"[,\t ]+", engine="python")
    Y = df.to_numpy(dtype=np.float64)
    Y = Y - Y.mean(axis=0, keepdims=True)

    window_len = fs // 4  # 128
    Mean, starts, ends = block_mean(abs(Y[:, 0]), window_len=window_len)
    i_break, idx_block = find_excitation_breakpoint_from_mean(Mean, starts, ratio=0.2)

    # Y = np.concatenate([Y[:i2, :], Y[20000:, :]], axis=0)
    u = Y[:, 0]
    y = Y[:, 19]
    t = np.arange(len(u)) / fs

    f, y_fft = compute_spectrum(y, fs, window=False, mode="amplitude")
    mask = f <= 60
    f = f[mask]
    y_fft = y_fft[mask]
    rms, env = envelope_rms_tv(y_fft, rms_win=32, n_iter=50, lam=50, rho=2)
    rms, y_env = envelope_rms_tv(y, rms_win=128, n_iter=100, lam=5, rho=2)

    # ---------- 关键参数 ----------
    prom = 0.01 * (np.percentile(env, 95) - np.percentile(env, 5))
    min_width_hz = 1             # 最小峰宽(Hz)
    min_dist_hz  = 3.0             # 峰之间最小间距(Hz)

    df = f[1] - f[0]
    min_width = int(min_width_hz / df)
    min_dist  = int(min_dist_hz  / df)
    
    idx, _ = find_peaks(env, prominence=prom, width=min_width, distance=min_dist)

    # y_idx, _ = find_peaks(-y_env, distance=5, width=50, wlen=100)
    print(f"峰值频率：{f[idx]}")
    # print(f"峰值位置（时域包络）：{y_idx}")

    model = OEModel(
    na=30, nb=30, nk=0,
    max_nfev=50,
    init_method="arx",
    rpm_khat=75,
    demean=False
    ).fit(y, u)

    y_hat = model.simulate(u)
    r, p, k = model.residues()

    Ts = 1.0 / fs

    # r: 留数, p: 离散极点 (z 域)
    r = np.asarray(r, dtype=np.complex128)
    p = np.asarray(p, dtype=np.complex128)

# ------------------ OE 预处理：稳定性 + 阻尼比阈值 ------------------
    # z -> s（连续域），用 +1e-12 避免 log(0) 崩
    s = np.log(p + 1e-12) / Ts
    sigma = np.real(s)
    omega_d = np.imag(s)

    # 频率 & 阻尼比
    freq_hz = np.abs(omega_d) / (2 * np.pi)
    omega_n = np.sqrt(sigma**2 + omega_d**2)

    # 避免除 0 / nan
    valid = np.isfinite(freq_hz) & np.isfinite(omega_n) & (omega_n > 0)

    zeta = np.empty_like(freq_hz, dtype=float)
    zeta[:] = np.nan
    zeta[valid] = -sigma[valid] / omega_n[valid]

    # 1) 只保留“正频率”分支（去掉共轭重复）
    mask_pos = omega_d > 0

    # 2) 稳定根：连续域 Re(s)<0（等价于离散域 |p|<1 的更严格表达，尤其你已经算了 s）
    mask_stable = sigma < 0

    # 3) 阻尼比阈值：ζ <= 0.8
    mask_zeta = np.isfinite(zeta) & (zeta <= 0.8) & (zeta >= 0.0)  # 负阻尼也顺手剔掉

    # 合并所有筛选
    mask = valid & mask_pos & mask_stable & mask_zeta

    # 同步筛选
    p = p[mask]
    r = r[mask]
    s = s[mask]
    freq_hz = freq_hz[mask]
    zeta = zeta[mask]

# ------------------ 频谱峰 ↔ OE 模态匹配（±0.5 Hz）------------------
    peak_freqs = f[idx]                  # 频谱包络找出的峰频率（Hz）
    oe_freqs = freq_hz                   # 初筛后的 OE 频率（Hz），与 p/r 一一对应
    oe_p = p
    oe_r = r

    match_half_width = 2               # ±2 Hz
    oe_used = np.zeros(len(oe_freqs), dtype=bool)

    matched = []  # list of dict: {peak_f, oe_i, oe_f, p, r}

    # 为了稳定：按峰强到弱匹配（避免一个强峰先占走好的 OE 模态后，弱峰没得选）
    # 你这里没有把峰幅值取出来，我就用 env[idx] 来排序（有的话）
    peak_order = np.argsort(env[idx])[::-1]

    for jj in peak_order:
        fp = float(peak_freqs[jj])

        cand = np.where((np.abs(oe_freqs - fp) <= match_half_width) & (~oe_used))[0]
        if cand.size == 0:
            continue  # 该峰丢弃（窗口内无 OE 候选）

        # 冲突处理：窗口内多个 OE 候选 -> 选 |r| 最大的
        best = cand[np.argmax(np.abs(oe_r[cand]))]

        oe_used[best] = True
        matched.append({
            "peak_f": fp,
            "oe_i": int(best),
            "oe_f": float(oe_freqs[best]),
            "p": oe_p[best],
            "r": oe_r[best],
            "res_abs": float(np.abs(oe_r[best])),
        })

    # 匹配结果：抽出选中的 p、r、freq
    p_sel = np.array([m["p"] for m in matched], dtype=np.complex128)
    r_sel = np.array([m["r"] for m in matched], dtype=np.complex128)
    f_sel = np.array([m["oe_f"] for m in matched], dtype=float)

    print(f"谱峰数: {len(peak_freqs)}，匹配到 OE 模态数: {len(p_sel)}")
    print(f"谱峰频率: {peak_freqs}")
    print(f"匹配的 OE 模态频率: {f_sel}")
    # ---------------------------------------------------------------


    # ------------------ 用筛选的极点/留数重构输出 y_rec ------------------
    # 离散域：每个模态项 G_i(z)= r_i / (1 - p_i z^-1)
    # 因为匹配阶段只保留了正频率分支(共轭对的一半)，重构时用 2*Re 来补共轭，保证实信号
    import scipy.signal as signal

    u1 = np.asarray(u, dtype=float).ravel()

    y_rec = np.zeros_like(u1, dtype=float)
    for ri, pi in zip(r_sel, p_sel):
        # 单个复极点项的输出（复数）
        yi = signal.lfilter([ri], [1.0, -pi], u1)
        y_rec += 2.0 * np.real(yi)   # 补共轭 -> 实信号贡献

    # 可选：若你希望加上 direct term（residuez 里的 k），这里可以加：
    # y_rec += np.real_if_close(np.polyval(k, ???))  # 你的模型 k 通常是 FIR 直通项/多项式项
    # 但你目前是“用筛出来的极点和留数重构”，一般就不加 k；需要再说。

    # 对比：y_hat 是全模型(na/nb)的 simulate(u)，y_rec 是“选定模态子集”的重构
    # plt.figure()
    # plt.plot(t, y, label="y true")
    # plt.plot(t, y_hat, label="OE full simulate", alpha=0.7)
    # plt.plot(t, y_rec, label="Reconstruction (matched modes)", alpha=0.9)
    # plt.legend()
    # plt.title("Reconstruction using matched poles & residues")
    # plt.grid(True)
    # # ---------------------------------------------------------------


    # # %%
    # plt.figure()
    # plt.plot(y, label="y (true)")
    # plt.plot(y_hat)
    # plt.legend()
    # plt.title("OE Model Fit Example")
    # # plt.show()

    # # plt.figure()
    # plt.plot(y_env)
    # plt.xlabel('频率/Hz')
    # plt.ylabel('包络幅度/g')
    # # plt.show()

    # plt.figure()
    # plt.plot(t, abs(u))
    # plt.xlabel('时间/s')
    # plt.ylabel('幅度/g')
    # plt.show()

    # plt.figure()
    # plt.plot(f, y_fft)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Spectrum Amplitude")
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(f, env)
    # plt.xlabel('频率/Hz')
    # plt.ylabel('包络幅度/g')
    # # plt.show()

    # plt.figure()
    # plt.plot(Mean)
    # plt.xlabel('时间/s')
    # plt.ylabel('平均响应/g')
    # plt.show()

# %%
