#import numpy as np
#from typing import List, Dict, Optional, Tuple
from .helper import *
from dataclasses import dataclass


# =======================================
# Data classes (Vehicle, RSU, UAV)
# =======================================

@dataclass
class Vehicle:
    vid: int
    lam: float               # Poisson arrival rate λ_v (tasks / slot)
    c_per_task: float        # CPU cycles per task
    d_per_task: float        # bits per task
    rsu_id: int              # which RSU covers this vehicle (simple mapping)
    gpu_frac:float = 0.0
    # runtime stats (not required but helpful)
    last_generated_tasks: int = 0


@dataclass
class RSU:
    rid: int
    x: float
    y: float
    f_max: float             # CPU cycles per second (computing ability)
    bandwidth: float         # total bandwidth (Hz) for uplink
    noise_power: float
    tx_power: float          # RSU tx power for response (for simplicity)

    gpu_flops:float = 0.0
    gpu_memory:float = 0.0
    gpu_power_active:float = 0.0

    # dynamic state
    workload_cycles: float = 0.0   # accumulated workload (cycles) this slot
    workload_gpu_ops: float = 0.0   # accumulated gpu ops
    workload_bits: float = 0.0     # accumulated bits (upload) this slot

    def reset_slot(self):
        self.workload_cycles = 0.0
        self.workload_gpu_ops = 0.0
        self.workload_bits = 0.0


@dataclass
class UAV:
    uid: int
    x: float
    y: float
    H: float

    f_u: float            # CPU cycles per second (UAV computing ability)
    hover_power: float    # ψ_u (W)
    energy_coeff: float   # b for comp energy: E = b f_u^2 c_k
    fly_coeff: float      # ρ for fly energy: E_fly = ρ * v^2


    # energy model
    E_max: float          # full battery energy
    E_batt: float         # current battery energy
    EH_max: float         # max harvest per slot e_max

    # environment-related channel params
    mu1: float
    mu2: float
    g0: float
    zeta: float
    H0: float             # RSU altitude (ground)

     # GPU-specific
    gpu_flops: float = 0.0        # GPU capability in FLOPS
    gpu_memory: float = 0.0       # GPU memory (MB)
    gpu_power_active: float = 0.0 # watts when GPU active
    # runtime tracking
    last_v: float = 0.0   # last-slot flight speed


    def harvest_energy(self) -> float:
        """Sample harvested energy e_t for this slot."""
        e_t = np.random.uniform(0.0, self.EH_max)
        # cannot exceed remaining capacity
        space = self.E_max - self.E_batt
        gained = min(e_t, space)
        self.E_batt += gained
        return gained

    def move_to(self, new_x: float, new_y: float, delta_t: float) -> float:
        """Move UAV, return flight energy for this slot."""
        dist = np.linalg.norm(np.array([new_x - self.x, new_y - self.y]))
        v = dist / max(delta_t, 1e-6)
        self.last_v = v
        self.x, self.y = new_x, new_y
        E_fly = self.fly_coeff * (v ** 2)
        return E_fly

    def can_afford(self, energy_needed: float) -> bool:
        return self.E_batt >= energy_needed

    def spend_energy(self, energy: float):
        self.E_batt = max(0.0, self.E_batt - energy)


# ===========================================
# Cell 4: VEC + UAV world model environment
# ===========================================

class UavVecWorld:
    """
    World model for UAV-assisted Vehicular Edge Computing.

    This class does NOT implement any learning or optimization.
    You call `step()` with your own decision about:
      - which RSU(s) each UAV serves
      - what fraction of that RSU's workload each UAV offloads
    and it will simulate task arrivals, transmission, computing, and energy.
    """

    def __init__(
        self,
        vehicles: List[Vehicle],
        rsus: List[RSU],
        uavs: List[UAV],                      
        delta_t: float,
        rsu_capacity_threshold: float,
        uplink_power_ue: float,
        uplink_noise_ue: float,
        rsu_radius: float
    ):
        self.vehicles = vehicles
        self.rsus = rsus
        self.uavs = uavs                     

        self.delta_t = delta_t
        self.rsu_capacity_threshold = rsu_capacity_threshold
        self.uplink_power_ue = uplink_power_ue
        self.uplink_noise_ue = uplink_noise_ue
        self.rsu_radius = rsu_radius

        self.t = 0

    # ---------- High-level API ----------

    def reset(self):
        self.t = 0
        for uav in self.uavs:               
            uav.E_batt = uav.E_max
        for rsu in self.rsus:
            rsu.reset_slot()

    def step(self, action: Dict) -> Dict:
        """
        action format :

        {
          rsu_id: {
              uav_id: offload_ratio,
              ...
          },
          ...
        }
        """
        self.t += 1

        # --- 1. Harvest energy ---
        for uav in self.uavs:                 
            uav.harvest_energy()

        # --- 2. Vehicular uploads ---
        veh_stats = self._simulate_vehicular_uploads()
        overloaded_indices = self._get_overloaded_rsus()

        # --- 3. UAV-assisted offloading (multi-UAV, cooperative) ---
        uav_results = {uav.uid: [] for uav in self.uavs}   

        for rsu_id, uav_dict in action.items():            
            if rsu_id not in overloaded_indices:
                continue

            for uav_id, offload_ratio in uav_dict.items():
                offload_ratio = float(np.clip(offload_ratio, 0.0, 1.0))
                uav = self.uavs[uav_id]

                result = self._simulate_uav_offloading(
                    uav=uav,                              
                    rsu_id=rsu_id,
                    offload_ratio=offload_ratio
                )

                if result is not None:
                    uav_results[uav_id].append(result)

        # --- 4. RSU local processing ---
        rsu_stats = self._simulate_rsu_processing(None)

        return {
            "time_slot": self.t,
            "veh_stats": veh_stats,
            "uav_results": uav_results,      
            "rsu_stats": rsu_stats,
            "uav_energy": {uav.uid: uav.E_batt for uav in self.uavs},
            "overloaded_rsus": overloaded_indices
        }

    # ---------- Internal methods ----------

    def _simulate_vehicular_uploads(self) -> Dict:
        """
        1) Sample Poisson tasks from each vehicle.
        2) For each RSU, accumulate workload (cycles, bits).
        3) Compute transmission delay V2I using OFDMA-like sharing.
           For simplicity we equally divide RSU bandwidth among its vehicles.
        """
        # reset RSUs' per-slot workload
        for rsu in self.rsus:
            rsu.reset_slot()

        veh_tasks = {}
        rsu_vehicle_map: Dict[int, List[Vehicle]] = {r.rid: [] for r in self.rsus}

        for v in self.vehicles:
            k = poisson_arrivals(v.lam)
            v.last_generated_tasks = k
            veh_tasks[v.vid] = k
            rsu_vehicle_map[v.rsu_id].append(v)

        rsu_trans_delay = {r.rid: 0.0 for r in self.rsus}

        for rsu in self.rsus:
            vehicles_here = rsu_vehicle_map[rsu.rid]
            if not vehicles_here:
                continue

            # equal bandwidth allocation among vehicles
            B_per_v = rsu.bandwidth / len(vehicles_here)

            total_delay = 0.0
            for v in vehicles_here:
                tasks = veh_tasks[v.vid]
                bits = tasks * v.d_per_task
                cycles = tasks * v.c_per_task

                gpu_frac = getattr(v, "gpu_frac", 0.0)
                gpu_ops = gpu_frac * cycles
                cpu_cycles = cycles - gpu_ops

                rsu.workload_cycles += cpu_cycles
                rsu.workload_gpu_ops += gpu_ops
                rsu.workload_bits += bits
                # simple distance-based channel gain (you can plug a better model)
                rsu_xy = np.array([rsu.x, rsu.y])
                # for world model, assume vehicles at RSU location for now
                veh_xy = rsu_xy.copy()
                dist_sq = np.sum((veh_xy - rsu_xy) ** 2) + 1.0
                gain = 1.0 / dist_sq

                rate = shannon_rate(
                    bandwidth=B_per_v,
                    power=self.uplink_power_ue,
                    gain=gain,
                    noise=self.uplink_noise_ue,
                    interference=0.0,  # inter-cell ignored for simplicity
                )

                # avoid division by zero
                if rate <= 0:
                    T_trans = 1e6
                else:
                    T_trans = bits / rate

                total_delay += T_trans


            rsu_trans_delay[rsu.rid] = total_delay

        return {
            "veh_tasks": veh_tasks,
            "rsu_upload_delay": rsu_trans_delay
        }

    def _get_overloaded_rsus(self) -> List[int]:
        overloaded = []
        for rsu in self.rsus:
            # threshold in "cycles per slot"
           if rsu.workload_cycles + rsu.workload_gpu_ops > self.rsu_capacity_threshold:
                overloaded.append(rsu.rid)
        return overloaded

    def _simulate_uav_offloading(self, uav: UAV, rsu_id: int, offload_ratio: float) -> Dict:
        """
        Simulate ONE UAV assisting ONE RSU.
        Multiple UAVs can call this for the same RSU in the same slot.
        """
        rsu = self.rsus[rsu_id]

        total_equiv = rsu.workload_cycles + rsu.workload_gpu_ops
        if total_equiv <= 0:
            return None

        d_total = rsu.workload_bits
        offload_equiv = offload_ratio * total_equiv

        gpu_share = rsu.workload_gpu_ops / max(total_equiv, 1e-9)
        c_k_gpu = offload_equiv * gpu_share
        c_k_cpu = offload_equiv * (1.0 - gpu_share)
        d_k = offload_ratio * d_total

        # --- UAV flight ---
        E_fly = uav.move_to(rsu.x, rsu.y, self.delta_t)

        # --- RSU --> UAV transmission ---
        g_uk = uav_rsu_channel_gain(
            mu1=uav.mu1, mu2=uav.mu2, H=uav.H, H0=uav.H0,
            g0=uav.g0, zeta=uav.zeta,
            uav_xy=np.array([uav.x, uav.y]),
            rsu_xy=np.array([rsu.x, rsu.y])
        )

        r_uk = shannon_rate(rsu.bandwidth, rsu.tx_power, g_uk, rsu.noise_power)
        T_trans = d_k / r_uk if r_uk > 0 else 1e6

        # --- UAV computing ---
        T_gpu, E_gpu = gpu_processing_time_and_energy(c_k_gpu, uav.gpu_flops, uav.gpu_power_active)
        T_cpu, E_cpu = cpu_processing_time_and_energy(c_k_cpu, uav.f_u, uav.energy_coeff)

        T_comp = max(T_gpu, T_cpu)
        E_comp = E_gpu + E_cpu

        # --- Output ---
        o_t = 0.1
        T_out = (o_t * d_k) / max(r_uk, 1e-9)

        E_hover = uav.hover_power * (T_trans + T_comp + T_out)
        E_total = E_fly + E_comp + E_hover

        if not uav.can_afford(E_total):
            return None

        uav.spend_energy(E_total)

        # --- Subtract ONLY this UAV's share ---
        rsu.workload_gpu_ops -= c_k_gpu
        rsu.workload_cycles -= c_k_cpu
        rsu.workload_bits -= d_k

        return {
            "served_rsu": rsu_id,
            "offloaded_cycles_cpu": c_k_cpu,
            "offloaded_cycles_gpu": c_k_gpu,
            "offloaded_bits": d_k,
            "T_uav_trans": T_trans,
            "T_uav_comp": T_comp,
            "T_uav_output": T_out,
            "E_comp": E_comp,
            "E_hover": E_hover,
            "E_fly": E_fly,
            "E_total": E_total
        }


    def _simulate_rsu_processing(self, uav_result: Dict) -> Dict:
        """
        Compute RSU-side computing delays and (optionally) queueing delays.
        This is a simplified version of eqs. (12)–(14).
        """
        rsu_comp_delay = {}
        rsu_queue_delay = {}
        rsu_output_delay = {}

        for rsu in self.rsus:
            gpu_ops = rsu.workload_gpu_ops
            cpu_cycles_local = rsu.workload_cycles

            T_gpu_rsu = 0.0; E_gpu_rsu = 0.0
            if rsu.gpu_flops > 0 and gpu_ops > 0:
                T_gpu_rsu, E_gpu_rsu = gpu_processing_time_and_energy(gpu_ops, rsu.gpu_flops, rsu.gpu_power_active)
                gpu_ops = 0.0

            T_cpu_rsu, E_cpu_rsu = cpu_processing_time_and_energy(cpu_cycles_local, rsu.f_max, energy_coeff=1e-28)

            T_comp = max(T_gpu_rsu, T_cpu_rsu)
            E_total_rsu = E_gpu_rsu + E_cpu_rsu

            # simple queue delay heuristic
            total_equiv = rsu.workload_cycles + rsu.workload_gpu_ops
            if total_equiv > self.rsu_capacity_threshold:
                tau = self.delta_t / max(total_equiv / self.rsu_capacity_threshold, 1.0)
                xi = total_equiv / max(self.delta_t, 1e-9)
                T_queue = mm1_queue_delay(tau, xi)
            else:
                T_queue = 0.0

            # output to vehicles
            o_t = 0.1
            bits_out = o_t * rsu.workload_bits
            rate_out = rsu.bandwidth
            T_out = bits_out / max(rate_out, 1e-9)

            rsu_comp_delay[rsu.rid] = T_comp
            rsu_queue_delay[rsu.rid] = T_queue
            rsu_output_delay[rsu.rid] = T_out

        return {
            "T_comp_rsu": rsu_comp_delay,
            "T_queue_rsu": rsu_queue_delay,
            "T_output_rsu": rsu_output_delay
        }

# ===========================================
# TODO: Create a gym.Env class to simulate UAV world
# ... also add multiple uavs
# ===========================================

import gymnasium as gym

class UavEnv(gym.Env):

    def __init__(self):
        self.action_space = ...
        self.observation_space = ...
        self.current_state = ...

    def reset(self, seed=None):
        ...

    def step(self, action, seed=None):
        ...


