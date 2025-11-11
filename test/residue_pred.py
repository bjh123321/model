import numpy as np
from scipy.signal import invresz

def build_BA_from_residues(alpha, omega, rho, theta, Ts,
                           *, p_exist=None, exist_thresh: float = 0.5,
                           residues_are_discrete: bool = False):
    """
    根据 (alpha, omega, rho, theta) 构造离散 IIR 系统系数 (B, A)，变量为 z^{-1}。
    - alpha, omega: 连续域极点实部/虚部，形状 (K,)
    - rho, theta  : 留数极坐标（幅值/相位，theta 为弧度），形状 (K,)
    - Ts          : 采样周期 (秒)
    - residues_are_discrete: 若 True，表示 (rho, theta) 已对应离散 z 域的残差；否则默认按连续残差直接配对使用（与常见的经验流程一致）
    返回:
      B, A (float64)，满足 H(z)=B(z^{-1})/A(z^{-1})
    """
    alpha  = np.asarray(alpha,  dtype=np.float64)
    omega  = np.asarray(omega,  dtype=np.float64)
    rho    = np.asarray(rho,    dtype=np.float64)
    theta  = np.asarray(theta,  dtype=np.float64)

    if not (alpha.shape == omega.shape == rho.shape == theta.shape):
        raise ValueError("alpha, omega, rho, theta 形状必须一致")
    
    if p_exist is not None:
        p_exist = np.asarray(p_exist, dtype=np.float64)
        if p_exist.shape != alpha.shape:
            raise ValueError("p_exist 形状必须与 alpha 等一致")
        mask = p_exist > exist_thresh
    else:
        mask = np.ones_like(alpha, dtype=bool)

    if not np.any(mask):
        return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
    
    alpha = alpha[mask]
    omega = omega[mask]
    rho   = rho[mask]
    theta = theta[mask]

    print(mask)
    # ===== 连续 -> 离散 极点 =====
    p_plus  = np.exp((alpha + 1j*omega) * Ts)   # e^{(α+jω)Ts}
    p_minus = np.exp((alpha - 1j*omega) * Ts)   # e^{(α-jω)Ts}
    poles = np.concatenate([p_plus, p_minus], axis=0)  # (2K,)

    # ===== 留数（默认视为与离散极点配对的复数，直接成对使用）=====
    # 若 residues_are_discrete=False，这里仍按经验直接使用；与很多 MATLAB/工程流程保持一致
    r_complex = rho * (np.cos(theta) + 1j*np.sin(theta))
    residues = np.concatenate([r_complex, np.conj(r_complex)], axis=0)  # (2K,)

    # ===== 合成 B,A （z^{-1} 变量）=====
    # 直通项 k = 0
    B, A = invresz(residues, poles, 0)

    # 数值清理 + 规范化 A[0]=1
    B = np.real_if_close(B, tol=1e5).astype(np.float64)
    A = np.real_if_close(A, tol=1e5).astype(np.float64)
    if A[0] != 0.0:
        B /= A[0]
        A /= A[0]

    return B, A
