import matplotlib.pyplot as plt
import numpy as np


def plot_world(
    vehicles,
    rsus,
    uavs,
    t,
    show=True,
    save_path=None,
    rsu_radius=None
):
    """
    Plot positions of vehicles, RSUs, and UAVs at a given time step.

    Parameters
    ----------
    vehicles : List[Vehicle]
    rsus : List[RSU]
    uavs : List[UAV]
    t : int
        Current time step
    show : bool
        Whether to display the plot
    save_path : str or None
        If provided, save the plot to this path
    rsu_radius : float or None
        If provided, draw RSU coverage circles
    """

    plt.figure(figsize=(8, 8))

    # ---------------- RSUs ----------------
    rsu_x = [r.x for r in rsus]
    rsu_y = [r.y for r in rsus]

    plt.scatter(
        rsu_x, rsu_y,
        c="red", s=120, marker="s",
        label="RSUs", zorder=3
    )

    # Optional RSU coverage
    if rsu_radius is not None:
        for r in rsus:
            circle = plt.Circle(
                (r.x, r.y),
                rsu_radius,
                color="red",
                fill=False,
                linestyle="--",
                alpha=0.3
            )
            plt.gca().add_patch(circle)

    # ---------------- Vehicles ----------------
    veh_x = [v.x for v in vehicles]
    veh_y = [v.y for v in vehicles]

    plt.scatter(
        veh_x, veh_y,
        c="blue", s=15, alpha=0.6,
        label="Vehicles", zorder=2
    )

    # ---------------- UAVs ----------------
    uav_x = [u.x for u in uavs]
    uav_y = [u.y for u in uavs]

    plt.scatter(
        uav_x, uav_y,
        c="green", s=160, marker="^",
        label="UAVs", zorder=4
    )

    # ---------------- Formatting ----------------
    plt.title(f"UAV-assisted MEC system — time step {t}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()
