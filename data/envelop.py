# %%
# from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.fftpack import hilbert
from scipy.interpolate import interp1d


# plt.rcParams.update({
#     "figure.autolayout": False,
#     # "figure.subplot.left": 0.15,
#     # "figure.subplot.right": 0.98,
#     # "figure.subplot.bottom": 0.15,
#     # "figure.subplot.top": 0.92,
#     "figure.subplot.left": 0.12,
#     "figure.subplot.right": 0.98,
#     "figure.subplot.bottom": 0.12,
#     "figure.subplot.top": 0.94,
#     "axes.formatter.use_mathtext": True,
#     "axes.formatter.useoffset": True,
#     "axes.formatter.limits": (0, 0),
#     # "legend.loc": "upper left",
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
#     # ===== 字体 =====
#     "font.family": ["Times New Roman", "SimSun"],
#     "font.size": 8,                        # 主体字体大小
#     "axes.labelsize": 7,
#     "axes.titlesize": 10,
#     "xtick.labelsize": 6,
#     "ytick.labelsize": 6,
#     "legend.fontsize": 6,
#     # ===== 线条 =====
#     "lines.linewidth": 0.5,
#     "axes.linewidth": 0.5,

#     # ===== 刻度线方向 =====
#     "xtick.direction": "in",
#     "ytick.direction": "in",

#     # ===== 刻度线长度 =====
#     "xtick.major.size": 3,
#     "ytick.major.size": 3,

#     # ===== 刻度线宽度 =====
#     "xtick.major.width": 0.5,
#     "ytick.major.width": 0.5,

#     # "figure.figsize": (2.626, 1.969),  # 一般大小（英寸），对应6.67cm×5cm
#     "figure.figsize": (3.543, 2.657),  # 一般大小（英寸），对应9cm×6.75cm
#     # "figure.figsize": (5.315, 3.543),  # 一般大小（英寸），对应13.5cm×9cm
#     "figure.dpi": 300                  # 分辨率 dpi
# })


def compute_spectrum(x, fs, window=True, mode="amplitude"):
    """
    计算单边频谱（Amplitude Spectrum / Power Spectrum / PSD）

    参数
    ----
    x : 1D array
        输入信号
    fs : float
        采样率（Hz）
    window : bool
        是否使用汉宁窗减轻泄漏
    mode : str, optional
        "amplitude" → 单边幅度谱 (默认)
        "power"     → 功率谱 |X|^2
        "psd"       → 功率谱密度 (Power Spectral Density)

    返回
    ----
    f : 1D array
        频率轴（Hz）
    X : 1D array
        对应模式的频谱值
    """

    x = np.asarray(x, float)
    N = len(x)

    # 加窗
    if window:
        w = np.hanning(N)
        xw = x * w
        U = (w**2).sum() / N   # PSD 归一化用
    else:
        xw = x
        U = 1

    # FFT
    Xf = np.fft.rfft(xw)
    f = np.fft.rfftfreq(N, d=1/fs)

    # 单边幅度谱
    if mode == "amplitude":
        X_amp = np.abs(Xf) / (N / 2)
        X_amp[0] /= 2  # DC
        return f, X_amp

    # 功率谱（|X|^2）
    elif mode == "power":
        X_pow = (np.abs(Xf)**2) / (N**2)
        X_pow[1:-1] *= 2  # 单边谱修正
        return f, X_pow

    # 功率谱密度 PSD（与 Welch 一致的定义）
    elif mode == "psd":
        X_psd = (np.abs(Xf)**2) / (fs * N * U)
        X_psd[1:-1] *= 2
        return f, X_psd

    else:
        raise ValueError(f"mode 必须是 amplitude / power / psd，但现在是: {mode}")


def moving_rms(x, win):
    """滑动 RMS（适合提取能量随时间变化的包络）"""
    x = np.asarray(x, float)
    N = x.size
    if win <= 1:
        return np.abs(x)

    pad = win // 2
    x2 = np.pad(x**2, (pad, pad), mode='edge')
    csum = np.cumsum(x2)

    rms2 = (csum[win:] - csum[:-win]) / win
    return np.sqrt(rms2)


def _soft_thresh(x, k):
    """软阈值函数：soft thresholding"""
    return np.sign(x) * np.maximum(np.abs(x) - k, 0.0)


def tv_envelope(x, lam=0.1, rho=1.0, n_iter=100):
    """
    基于 Total Variation 的 1D 包络提取（TV-Envelope）

    解的模型是：
        min_e 0.5 * ||e - y||^2 + lam * TV(e)
    其中 y = |x|，TV(e) = sum_i |e[i+1] - e[i]|

    参数
    ----
    x : 1D np.ndarray
        输入信号（实数序列）
    lam : float
        TV 正则化系数，越大包络越平滑，越小越贴近 |x|
    rho : float
        ADMM 的惩罚参数，一般 1~10 都可以，调节收敛速度
    n_iter : int
        ADMM 迭代次数，几十到几百次一般就够

    返回
    ----
    e : 1D np.ndarray
        提取到的包络（与 x 等长，非负）
    """
    # 用绝对值作为“观测包络”，再做 TV 去噪/平滑
    y = np.abs(np.asarray(x, dtype=float))
    n = y.size
    if n < 2:
        return y.copy()

    # 差分算子 D 与其转置 D^T，1D 情况不用显式构造矩阵
    def Dx(v):
        # v: (n,) -> (n-1,)
        return v[1:] - v[:-1]

    def DTz(z):
        # z: (n-1,) -> (n,)
        out = np.zeros(n, dtype=float)
        out[0] = -z[0]
        out[1:-1] = z[:-1] - z[1:]
        out[-1] = z[-1]
        return out

    # ADMM 变量
    e = y.copy()          # 主变量（包络）
    z = Dx(e)             # 辅助变量，近似 e 的一阶差分
    u = np.zeros_like(z)  # 对偶变量

    # 要解的线性系统： (I + rho * D^T D) e = y + rho * D^T (z - u)
    # 1D 情况下这是三对角矩阵，手写 Thomas 算法即可 O(n)

    # 主对角线
    diag = np.ones(n, dtype=float)
    if n == 2:
        diag[0] += rho * 1.0
        diag[1] += rho * 1.0
        off = np.array([-rho], dtype=float)
    else:
        diag[0] += rho * 1.0
        diag[1:-1] += rho * 2.0
        diag[-1] += rho * 1.0
        # 上/下对角线相同
        off = -rho * np.ones(n - 1, dtype=float)

    def solve_tridiag(b):
        """解三对角线性系统 A x = b，其中 A 由 diag, off 定义"""
        c = off.copy()
        d = diag.copy()
        bb = b.copy()

        # 前向消去
        for i in range(1, n):
            w = c[i - 1] / d[i - 1]
            d[i] -= w * c[i - 1]
            bb[i] -= w * bb[i - 1]

        # 回代
        x = np.zeros_like(bb)
        x[-1] = bb[-1] / d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (bb[i] - c[i] * x[i + 1]) / d[i]
        return x

    # ADMM 迭代
    for _ in range(n_iter):
        # e 更新：解线性方程
        rhs = y + rho * DTz(z - u)
        e = solve_tridiag(rhs)

        # z 更新：soft-thresholding 作用在 e 的一阶差分上
        De = Dx(e)
        z = _soft_thresh(De + u, lam / rho)

        # u 更新：对偶变量
        u = u + De - z

    return e


def envelope_rms_tv(x, rms_win=500, lam=1.0, rho=5.0, n_iter=200):
    """
    提取扫频试飞信号的“能量大包络”
    1) 滑动 RMS 压制载波 + 噪声
    2) TV smoothing 得平滑包线
    """
    rms = moving_rms(x, rms_win)
    env = tv_envelope(rms, lam=lam, rho=rho, n_iter=n_iter)
    return rms, env


# def main():
#     # dir = Path(r'D:\BaiduSyncdisk\毕业\硕士\论文\数据\第二章')
#     dir = Path(r'D:\BaiduSyncdisk\项目\试飞数据\35A\试飞数据35A\H3-ReConstr\M0.9-ReConstr')
#     filename = 'H3_M0.9_方向舵对称_幅值0.6-ReConstr.txt'
#     data = np.loadtxt(dir/filename, skiprows=1)
#     # %%
#     fs = 512
#     dt = 1/fs
#     duration = data.shape[0] * dt
#     t = np.arange(0, duration, dt)
#     LRY = data[:, 15]
#     LRY = LRY - np.mean(LRY)
#     f, LRY_fft = compute_spectrum(LRY, fs, window=False, mode="amplitude")
#     mask = f <= 60
#     f = f[mask]
#     LRY_fft = LRY_fft[mask]

#     # %% 希尔伯特变换
#     # analytic_signal = hilbert(LRY)
#     # rough_envelope = np.abs(analytic_signal)

#     # envelope = tv_envelope(LRY, n_iter=200, lam=5, rho=5) # 频谱平滑
#     rms, env = envelope_rms_tv(LRY_fft, rms_win=32, n_iter=100, lam=1, rho=1)

#     # envelope = tv_envelope(LRY, lam=1, rho=1, n_iter=200)
#     # rms, env = envelope_rms_tv(LRY, rms_win=2000, lam=5)

#     plt.figure()
#     plt.plot(LRY_fft)
#     plt.xlabel('频率/Hz')
#     plt.ylabel('幅值')
#     plt.title('LRY Signal')

#     # plt.figure()
#     # LRY = LRY[LRY>0]
#     # plt.plot(LRY)

#     plt.figure()
#     plt.plot(rms)
#     plt.legend('RMS')

#     plt.figure()
#     plt.plot(env*10)
#     # plt.legend(['RMS-TV 包络'], loc='center left', bbox_to_anchor=(1.02, 0.5))
#     plt.xlabel('频率/Hz')
#     plt.ylabel('包络幅度/g')
#     # plt.xlim(0, 60)
#     # savefig = Path(r'C:\Users\Leo\Desktop\频谱包络.png')
#     # plt.savefig(savefig, dpi=300, bbox_inches='tight')
#     # 
#     # plt.figure()
#     # plt.plot(envelope)
#     # plt.legend('tv Envelope')

#     plt.show()
    # %%
# if __name__ == "__main__":
    # main()