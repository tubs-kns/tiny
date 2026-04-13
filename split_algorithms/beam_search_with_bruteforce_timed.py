"""
beam_search_with_bruteforce_timed.py
------------------------------------
Extended version including:
 - Brute-force optimal split search
 - Beam Search comparison
 - Processing time measurement and dual-axis plot
"""

import pandas as pd
import heapq
import itertools
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Callable
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

# =========================================================
# 1. Load data
# =========================================================

inference_df = pd.read_csv("data/inference_times.csv")
inference_times = inference_df["Inference_Time"].tolist()
L = len(inference_times)

trans_df = pd.read_csv("data/layer_transmission_times.csv")
transmission_times = trans_df["Transmission_Time"].tolist()

print(f"Loaded {L} layers and {len(transmission_times)} transmission entries.")

# =========================================================
# 2. Cost model
# =========================================================

def cost_segment(a: int, b: int, device_idx: int) -> float:
    """
    Compute latency of assigning layers [a, b] to device device_idx.
    Combines local inference latency and transmission delay.
    """
    processing_time = inference_times[b - 1] - inference_times[a - 1]
    if processing_time < 0:
        processing_time = abs(processing_time)

    transmission_time = transmission_times[b - 1] if b < L else 0.0
    return processing_time + transmission_time


# =========================================================
# 3. Beam Search Algorithm
# =========================================================

def beam_search_split(
    L: int,
    N: int,
    B: int,
    cost_segment: Callable[[int, int, int], float]
) -> Tuple[List[int], float]:
    beam = [(0.0, 0, [])]  # (total_cost, last_layer, split_list)

    for k in range(1, N + 1):
        new_beam = []
        for cost, pos, splits in beam:
            upper_limit = L if k == N else L - (N - k)
            for nxt in range(pos + 1, upper_limit + 1):
                c_seg = cost_segment(pos + 1, nxt, k)
                total_cost = cost + c_seg
                new_beam.append((total_cost, nxt, splits + [nxt]))

        # Retain best candidates
        if k == N:
            valid_final = [item for item in new_beam if item[1] == L]
            beam = heapq.nsmallest(B, valid_final if valid_final else new_beam, key=lambda x: x[0])
        else:
            beam = heapq.nsmallest(B, new_beam, key=lambda x: x[0])

    final_candidates = [(c, s) for c, pos, s in beam if pos == L]
    if not final_candidates:
        raise RuntimeError("No valid configuration reached the final layer.")
    best_cost, best_splits = min(final_candidates, key=lambda x: x[0])
    return best_splits[:-1], best_cost


# =========================================================
# 4. Brute-Force Optimal Algorithm
# =========================================================

def brute_force_optimal_split(L: int, N: int, cost_segment: Callable[[int, int, int], float]) -> Tuple[List[int], float]:
    """
    Exhaustive brute-force search for the optimal split configuration.
    """
    best_splits = None
    best_latency = float("inf")

    for splits in itertools.combinations(range(1, L), N - 1):
        # Compute total latency
        s = [0] + list(splits) + [L]
        total_latency = 0.0
        for i in range(1, N + 1):
            total_latency += cost_segment(s[i - 1] + 1, s[i], i)
        if total_latency < best_latency:
            best_latency = total_latency
            best_splits = list(splits)

    return best_splits, best_latency


# =========================================================
# 5. Evaluate latency and processing time
# =========================================================

def evaluate_latency_vs_devices(L: int, max_devices: int, beam_width: int):
    results = []
    for N in range(2, max_devices + 1):
        start = time.time()
        best_splits, best_cost = beam_search_split(L, N, beam_width, cost_segment)
        end = time.time()
        results.append((N, best_cost, end - start))
        print(f"[Beam] Devices={N} | Latency={best_cost:.3f} | Time={end - start:.3f}s")
    return pd.DataFrame(results, columns=["Devices", "Beam_Latency", "Beam_Time"])


def evaluate_latency_vs_devices_bruteforce(L: int, max_devices: int):
    results = []
    for N in range(2, max_devices + 1):
        start = time.time()
        best_splits, best_cost = brute_force_optimal_split(L, N, cost_segment)
        end = time.time()
        results.append((N, best_cost, end - start))
        print(f"[Brute] Devices={N} | Latency={best_cost:.3f} | Time={end - start:.3f}s")
    return pd.DataFrame(results, columns=["Devices", "Optimal_Latency", "Brute_Time"])


# =========================================================
# 6. Plot results with dual y-axis
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_latency_and_time(df_beam: pd.DataFrame, df_opt: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # --- X ticks as integers ---
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # --- Latency (lines, left y-axis) ---
    ax1.plot(df_beam["Devices"], df_beam["Beam_Latency"],
             marker="o", color="tab:blue", linewidth=2, label="Beam Search Latency")
    ax1.plot(df_opt["Devices"], df_opt["Optimal_Latency"],
             marker="s", color="tab:red", linewidth=2, label="Brute-Force Latency")
    ax1.set_xlabel("Number of Devices (N)", fontsize=16)
    ax1.set_ylabel("Expected Total Latency (ms)", fontsize=16)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim(0, 4000)

    # --- Processing time (bars, right y-axis) ---
    width = 0.35
    bars_beam = ax2.bar(df_beam["Devices"] - width/2, df_beam["Beam_Time"],
                        width=width, color="tab:blue", alpha=0.3, label="Beam Search Time (s)")
    bars_brute = ax2.bar(df_opt["Devices"] + width/2, df_opt["Brute_Time"],
                         width=width, color="tab:red", alpha=0.3, label="Brute-Force Time (s)")
    ax2.set_ylabel("Processing Time (s)", fontsize=16)

    # --- Add labels above bars (reusable) ---
    def add_bar_labels(axis, bars, fontsize=12, va='bottom'):
        ymin, ymax = axis.get_ylim()
        headroom = (ymax - ymin) * 0.03 if ymax > ymin else 0.05
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                axis.text(
                    bar.get_x() + bar.get_width()/2,
                    h + headroom,                          # small vertical offset
                    f"{h:.0f}" if h >= 1 else f"{h:.2f}",
                    ha="center", va="top", fontsize=fontsize
                )
    def add_bar_labelsx(axis, bars, fontsize=12, va='bottom'):
        ymin, ymax = axis.get_ylim()
        headroom = (ymax - ymin) * 0.03 if ymax > ymin else 0.05
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                axis.text(
                    bar.get_x() + bar.get_width()/2,
                    h + headroom,                          # small vertical offset
                    f"{h:.0f}s" if h >= 1 else f"{h:.2f}s",
                    ha="center", va="top", fontsize=9
                )

    add_bar_labels(ax2, bars_beam)
    add_bar_labels(ax2, bars_brute)

    # --- Legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11,
               loc="lower left", bbox_to_anchor=(0.0, 0.2))  # move up from bottom-left

    # --- Highlight zoom region on main plot (around N=4) ---
    #ax1.axvspan(3.8, 4.2, alpha=0.08)

    # --- Inset (replicates main: lines + bars + twin y-axis) ---
    ax_ins = inset_axes(
        ax1,
        width="50%", height="50%",
        loc="lower left",
        bbox_to_anchor=(0.478, 0.22, 0.48, 0.48),  # (x0, y0, w, h) in axes coords
        bbox_transform=ax1.transAxes,
        borderpad=0.0
    )
    ax_ins2 = ax_ins.twinx()

    # Corrected filter for N≈4 (matches axvspan above)
    dfb4 = df_beam[(df_beam["Devices"] >= 3.8) & (df_beam["Devices"] <= 4.2)]
    dfo4 = df_opt[(df_opt["Devices"] >= 3.8) & (df_opt["Devices"] <= 4.2)]
    if dfb4.empty: dfb4 = df_beam[df_beam["Devices"] == 4]
    if dfo4.empty: dfo4 = df_opt[df_opt["Devices"] == 4]

    # Lines (latency) in inset
    
    # Bars (processing time) in inset (narrower width)
    w_inset = 0.15
    inset_beam = inset_brute = None
    if not dfb4.empty:
        inset_beam = ax_ins2.bar(dfb4["Devices"] - w_inset/2, dfb4["Beam_Time"],
                                 width=w_inset, color="tab:blue", alpha=0.3)
    if not dfo4.empty:
        inset_brute = ax_ins2.bar(dfo4["Devices"] + w_inset/2, dfo4["Brute_Time"],
                                  width=w_inset, color="tab:red", alpha=0.3)

    # Inset formatting
    ax_ins.set_xlim(3.8, 4.2)
    #ax_ins.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax_ins.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    #ax_ins.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax_ins.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    #ax_ins2.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax_ins2.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # Latency y-lims with padding
    yvals = []
    if not dfb4.empty: yvals += dfb4["Beam_Latency"].tolist()
    if not dfo4.empty: yvals += dfo4["Optimal_Latency"].tolist()
    if yvals:
        ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
        pad = max(10.0, 0.08 * (ymax - ymin if ymax > ymin else 50.0))
        ax_ins.set_ylim(ymin - pad, ymax + pad)

    ax_ins.grid(True, linestyle="--", alpha=0.6)
    ax_ins.set_title("Zoom @ N=4", fontsize=10, pad=2)
    ax_ins.tick_params(labelsize=9)
    ax_ins2.tick_params(labelsize=9)

    # ---- VALUE LABELS on inset bars ----
    if inset_beam is not None:
        add_bar_labelsx(ax_ins2, inset_beam)
    if inset_brute is not None:
        add_bar_labelsx(ax_ins2, inset_brute)

    # Layout + save/show
    plt.tight_layout()
    fig.savefig('Latency_and_Processing_Time_with_Zoom_N4.pdf')
    plt.show()



# =========================================================
# 7. Main entry
# =========================================================

if __name__ == "__main__":
    BEAM_WIDTH = 300
    MAX_DEVICES = 4 # smaller range recommended for brute force

    print("\n--- Beam Search ---")
    df_beam = evaluate_latency_vs_devices(L, MAX_DEVICES, BEAM_WIDTH)

    print("\n--- Brute Force ---")
    df_opt = evaluate_latency_vs_devices_bruteforce(L, MAX_DEVICES)

    # Merge on "Devices"
    merged = pd.merge(df_beam, df_opt, on="Devices")
    print("\nCombined results:\n", merged)

    # Plot latency + processing time
    plot_latency_and_time(df_beam, df_opt)

