import numpy as np
from scipy.signal import cont2discrete

def build_BA_from_zeros(alpha, omega, gain, zero, Ts):
    """
    连续域参数 (alpha, omega, gain, zero) -> 先构建连续域二阶模态传函，再用 ZOH 离散化，最后串联得到离散 IIR。
    连续域单模态：
        H_k(s) = K_k * (s - z0_k) / [(s - (α_k + jω_k)) (s - (α_k - jω_k))]
               = K_k * (s - z0_k) / (s^2 - 2α_k s + (α_k^2 + ω_k^2))
    返回：B, A （float64，变量为 z^{-1}，可直接给 lfilter）
    """
    alpha = np.asarray(alpha, dtype=float)
    omega = np.asarray(omega, dtype=float)
    gain  = np.asarray(gain,  dtype=float)
    zero  = np.asarray(zero,  dtype=float)

    K = alpha.size
    if not (omega.size == gain.size == zero.size == K):
        raise ValueError("alpha, omega, gain, zero 形状必须一致 (K,)")

    # 总的离散分子/分母（z^{-1} 变量）
    Bz = np.array([1.0])
    Az = np.array([1.0])

    for k in range(K):
        # 连续域二阶分母: s^2 - 2α s + (α^2 + ω^2)
        den_s = np.array([1.0, -2.0*alpha[k], alpha[k]**2 + omega[k]**2], dtype=float)
        # 连续域一阶分子: K * (s - z0)
        num_s = gain[k] * np.array([1.0, -zero[k]], dtype=float)

        # 使用 ZOH 离散化该二阶节
        # cont2discrete 接受 (num, den)，返回 bz, az, dt；bz, az 是 z^{-1} 变量系数
        bz, az, _ = cont2discrete((num_s, den_s), Ts, method='zoh')[:3]
        bz = np.squeeze(bz).astype(float)
        az = np.squeeze(az).astype(float)

        # 串联（卷积）
        Bz = np.convolve(Bz, bz)
        Az = np.convolve(Az, az)

    # 数值清理与规范化
    Bz = np.real_if_close(Bz, tol=1e5).astype(float)
    Az = np.real_if_close(Az, tol=1e5).astype(float)
    if Az[0] != 0.0:
        Bz /= Az[0]
        Az /= Az[0]
    return Bz, Az
