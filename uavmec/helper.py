
import numpy as np
from typing import List, Dict, Optional, Tuple

# ======================================
# Helper functions (channel etc.)
# ======================================
np.random.seed(42)
def poisson_arrivals(lam: float) -> int:
    """Sample Poisson arrivals for a given rate λ."""
    return np.random.poisson(lam)


def los_probability(mu1: float, mu2: float, H: float, H0: float,
                    uav_xy: np.ndarray, rsu_xy: np.ndarray) -> float:
    """
    LoS probability between UAV and RSU as in eq. (6).
    uav_xy, rsu_xy: np.array([x, y])
    """
    dist_horizontal = np.linalg.norm(uav_xy - rsu_xy)
    # avoid division by zero (if UAV exactly above RSU)
    dist_horizontal = max(dist_horizontal, 1e-6)
    beta = np.arctan((H - H0) / dist_horizontal)  # radians
    # convert elevation angle to degrees inside expression as in the paper
    beta_deg = (180.0 / np.pi) * beta # check this once
    p_los = 1.0 / (1.0 + mu1 * np.exp(-mu2 * (beta_deg - mu1)))
    return p_los


def uav_rsu_channel_gain(mu1: float, mu2: float, H: float, H0: float,
                        g0: float, zeta: float,
                        uav_xy: np.ndarray, rsu_xy: np.ndarray) -> float:
    """
    Channel gain g_{u,k}^t as in eq. (7).
    """
    p_los = los_probability(mu1, mu2, H, H0, uav_xy, rsu_xy)
    dist_sq = (H - H0) ** 2 + np.sum((uav_xy - rsu_xy) ** 2)
    gain = g0 * (p_los + zeta * (1.0 - p_los)) / dist_sq
    return float(gain)


def shannon_rate(bandwidth: float, power: float, gain: float, noise: float,
                interference: float = 0.0) -> float:
    """
    r = B log2(1 + S / (N + I))
    Returns rate in bits per second (if B in Hz). as in eqn (8)
    """
    denom = noise + interference
    # avoid division by zero
    if denom <= 0:
        denom = 1e-12
    snr = power * gain / denom
    return float(bandwidth * np.log2(1.0 + snr))


def mm1_queue_delay(tau: float, xi: float) -> float:
    """
    M/M/1 queue delay for overloaded RSU as in eq. (13):
    T_queue = tau / (1 - tau * xi)
    tau: expected delay per task without queue
    xi: arrival rate in 'tasks per second' equivalent
    """
    rho = tau * xi
    if rho >= 1.0:
        # System unstable – cap with a large delay
        return 1e6
    return float(tau / (1.0 - rho))


def gpu_processing_time_and_energy(gpu_ops: float, gpu_flops: float, power_active: float) -> Tuple[float, float]:
    if gpu_flops <= 0:
        return float('inf'), float('inf')
    t = gpu_ops / gpu_flops
    e = power_active * t
    return float(t), float(e)


def cpu_processing_time_and_energy(cpu_cycles: float, cpu_freq: float, energy_coeff: float, cpu_power_active: Optional[float]=None) -> Tuple[float, float]:
    if cpu_freq <= 0:
        return float('inf'), float('inf')
    t = cpu_cycles / cpu_freq
    e = energy_coeff * (cpu_freq ** 2) * cpu_cycles
    if cpu_power_active is not None:
        e = cpu_power_active * t
    return float(t), float(e)

