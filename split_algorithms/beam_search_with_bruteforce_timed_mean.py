"""
compare_beam_brute_random.py
----------------------------
Combined comparison of:
 - Brute-force optimal split search
 - Beam Search
 - Random-Fit (stochastic baseline)

Loads:
 - inference_times.csv   with column: Inference_Time
 - layer_transmission_times.csv  with column: Transmission_Time

Outputs:
 - One figure with all three latency curves and processing-time bars
"""

import time
import heapq
import itertools
import random
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =========================================================
# 1) Load data
# =========================================================

def load_data(
    inference_csv: str = "data/inference_times.csv",
    transmission_csv: str = "data/layer_transmission_times.csv"
):
    inference_df = pd.read_csv(inference_csv)
    trans_df = pd.read_csv(transmission_csv)

    inference_times = inference_df["Inference_Time"].tolist()
    transmission_times = trans_df["Transmission_Time"].tolist()

    L = len(inference_times)
    if len(transmission_times) < L:
        raise ValueError(
            f"Transmission list shorter than layers: {len(transmission_times)} < {L}"
        )

    print(f"Loaded {L} layers and {len(transmission_times)} transmission entries.")
    return inference_times, transmission_times, L


# =========================================================
# 2) Cost model
# =========================================================

def make_cost_segment(inference_times: List[float], transmission_times: List[float], L: int
) -> Callable[[int, int, int], float]:
    """
    Returns a cost_segment(a,b,device_idx) closure consistent with your script.
    Layers are 1-based inclusive: [a..b]
    """
    def cost_segment(a: int, b: int, device_idx: int) -> float:
        processing_time = inference_times[b - 1] - inference_times[a - 1]
        if processing_time < 0:
            processing_time = abs(processing_time)
        transmission_time = transmission_times[b - 1] if b < L else 0.0
        return processing_time + transmission_time
    return cost_segment


# =========================================================
# 3) Beam Search
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
# 4) Brute force (optimal)
# =========================================================

def brute_force_optimal_split(L: int, N: int, cost_segment: Callable[[int, int, int], float]) -> Tuple[List[int], float]:
    best_splits = None
    best_latency = float("inf")

    for splits in itertools.combinations(range(1, L), N - 1):
        s = [0] + list(splits) + [L]
        total_latency = 0.0
        for i in range(1, N + 1):
            total_latency += cost_segment(s[i - 1] + 1, s[i], i)
        if total_latency < best_latency:
            best_latency = total_latency
            best_splits = list(splits)

    return best_splits, best_latency


# =========================================================
# 5) Random-Fit (stochastic baseline)
# =========================================================

def random_fit_split(
    L: int,
    N: int,
    cost_segment: Callable[[int, int, int], float],
    trials: int = 1,
    seed: int | None = None
) -> Tuple[List[int], float]:
    """
    Uniformly sample N-1 unique cut positions from {1..L-1}.
    If trials > 1, keep the best found.
    """
    rng = random.Random(seed)
    best_splits = None
    best_cost = float("inf")

    if N < 2:
        raise ValueError("N must be >= 2.")
    if N > L:
        raise ValueError("N cannot exceed L.")

    for _ in range(trials):
        splits = sorted(rng.sample(range(1, L), N - 1))
        s = [0] + splits + [L]
        total = 0.0
        for i in range(1, N + 1):
            total += cost_segment(s[i - 1] + 1, s[i], i)
        if total < best_cost:
            best_cost = total
            best_splits = splits

    return best_splits, best_cost


# =========================================================
# 6) Evaluation loops
# =========================================================

def evaluate_latency_vs_devices_beam(L: int, max_devices: int, beam_width: int, cost_segment):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        _, best_cost = beam_search_split(L, N, beam_width, cost_segment)
        t1 = time.time()
        rows.append((N, best_cost, t1 - t0))
        print(f"[Beam]  N={N} | Latency={best_cost:.3f} | Time={t1 - t0:.3f}s")
    return pd.DataFrame(rows, columns=["Devices", "Beam_Latency", "Beam_Time"])


def evaluate_latency_vs_devices_bruteforce(L: int, max_devices: int, cost_segment):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        _, best_cost = brute_force_optimal_split(L, N, cost_segment)
        t1 = time.time()
        rows.append((N, best_cost, t1 - t0))
        print(f"[Brute] N={N} | Latency={best_cost:.3f} | Time={t1 - t0:.3f}s")
    return pd.DataFrame(rows, columns=["Devices", "Optimal_Latency", "Brute_Time"])


def evaluate_latency_vs_devices_random(L: int, max_devices: int, cost_segment, trials: int = 200, seed: int = 42):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        latencies = []
        for t in range(trials):
            s, c = random_fit_split(L, N, cost_segment, trials=1, seed=seed + N * 1000 + t)
            latencies.append(c)
        t1 = time.time()
        latencies = np.array(latencies, dtype=float)
        mean_lat, std_lat, best_lat = float(latencies.mean()), float(latencies.std()), float(latencies.min())
        rows.append((N, mean_lat, std_lat, best_lat, t1 - t0))
        print(f"[Rand]  N={N} | mean={mean_lat:.3f} ± {std_lat:.3f} | best={best_lat:.3f} | Time={t1 - t0:.3f}s")
    return pd.DataFrame(rows, columns=[
        "Devices", "Random_Mean_Latency", "Random_Std_Latency", "Random_Best_Latency", "Random_Time"
    ])


# =========================================================
# 7) Plot (all in one figure)
# =========================================================

def plot_latency_and_time_all(df_beam: pd.DataFrame, df_opt: pd.DataFrame, df_rand: pd.DataFrame, inset_at: int | None = 4):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    # integer X axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # ---- Latency curves (left axis) ----
    ax1.plot(df_beam["Devices"], df_beam["Beam_Latency"],
             marker="o", linewidth=2, label="Beam Search (latency)")
    ax1.plot(df_opt["Devices"], df_opt["Optimal_Latency"],
             marker="s", linewidth=2, label="Brute-Force Optimal (latency)")
    ax1.plot(df_rand["Devices"], df_rand["Random_Mean_Latency"],
             marker="^", linewidth=2, label="Random-Fit (latency)")
    # variability band for random


    ax1.set_xlabel("Number of Devices (N)", fontsize=14)
    ax1.set_ylabel("Total Latency (ms)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---- Processing-time bars (right axis) ----
    width = 0.25
    x = df_beam["Devices"].values
    bars_beam = ax2.bar(x - width, df_beam["Beam_Time"], width=width, alpha=0.3, label="Beam Time (s)")
    bars_brut = ax2.bar(x + 0.0,    df_opt["Brute_Time"], width=width, alpha=0.3, label="Brute Time (s)")
    bars_rand = ax2.bar(x + width,   df_rand["Random_Time"], width=width, alpha=0.3, label="Random-Fit Time (s)")
    ax2.set_ylabel("Processing Time (s)", fontsize=14)

    # Labels above bars
    def add_bar_labels(axis, bars, fontsize=9):
        ymin, ymax = axis.get_ylim()
        headroom = (ymax - ymin) * 0.03 if ymax > ymin else 0.05
        for b in bars:
            h = b.get_height()
            if h > 0:
                axis.text(b.get_x() + b.get_width()/2, h ,
                          f"{h:.0f}s" if h >= 1 else f"{h:.2f}s",
                          ha="center", va="bottom", fontsize=fontsize)
    add_bar_labels(ax2, bars_beam)
    add_bar_labels(ax2, bars_brut)
    add_bar_labels(ax2, bars_rand)

    # ---- Legend (merge both axes) ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # ---- Optional inset zoom around inset_at ----
    if inset_at is not None and inset_at in df_beam["Devices"].values:
        ax_ins = inset_axes(ax1, width="48%", height="48%", loc="lower left",
                            bbox_to_anchor=(0.50, 0.18, 0.48, 0.48), bbox_transform=ax1.transAxes, borderpad=0.0)
        ax_ins2 = ax_ins.twinx()

        def filter_at(df, col="Devices", val=inset_at):
            f = df[df[col] == val]
            if f.empty:
                # fallback to small window if non-integer used
                f = df[(df[col] >= val - 0.2) & (df[col] <= val + 0.2)]
            return f

        db = filter_at(df_beam)
        do = filter_at(df_opt)
        dr = filter_at(df_rand)

        # plot latency points at inset
        if not db.empty:
            ax_ins.plot(db["Devices"], db["Beam_Latency"], marker="o", linewidth=2)
        if not do.empty:
            ax_ins.plot(do["Devices"], do["Optimal_Latency"], marker="s", linewidth=2)
        if not dr.empty:
            ax_ins.plot(dr["Devices"], dr["Random_Mean_Latency"], marker="^", linewidth=2)

        # bars for processing time at inset
        w_inset = 0.15
        if not db.empty:
            ax_ins2.bar(db["Devices"] - w_inset, db["Beam_Time"], width=w_inset, alpha=0.3)
        if not do.empty:
            ax_ins2.bar(do["Devices"] + 0.0, do["Brute_Time"], width=w_inset, alpha=0.3)
        if not dr.empty:
            ax_ins2.bar(dr["Devices"] + w_inset, dr["Random_Time"], width=w_inset, alpha=0.3)

        # y-lims with padding for inset latency axis
        yvals = []
        if not db.empty: yvals += db["Beam_Latency"].tolist()
        if not do.empty: yvals += do["Optimal_Latency"].tolist()
        if not dr.empty: yvals += dr["Random_Mean_Latency"].tolist()
        if yvals:
            ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
            pad = max(10.0, 0.08 * (ymax - ymin if ymax > ymin else 50.0))
            ax_ins.set_ylim(ymin - pad, ymax + pad)

        ax_ins.grid(True, linestyle="--", alpha=0.6)
        ax_ins.set_title(f"Zoom @ N={inset_at}", fontsize=10, pad=2)
        ax_ins.tick_params(labelsize=9)
        ax_ins2.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig("Latency_and_Processing_Time_All.pdf")
    plt.show()


# =========================================================
# 8) Main
# =========================================================

if __name__ == "__main__":
    # ---- Config ----
    BEAM_WIDTH   = 300
    MAX_DEVICES  = 5        # keep modest if brute force is enabled
    RAND_TRIALS  = 200      # random samples per N (set to 1 for single-shot)
    RAND_SEED    = 42
    INSET_AT     = 4        # set to None to disable inset

    # Load data and cost
    inference_times, transmission_times, L = load_data()
    cost_segment = make_cost_segment(inference_times, transmission_times, L)

    # Evaluate
    print("\n--- Beam Search ---")
    df_beam = evaluate_latency_vs_devices_beam(L, MAX_DEVICES, BEAM_WIDTH, cost_segment)

    print("\n--- Brute Force ---")
    df_opt  = evaluate_latency_vs_devices_bruteforce(L, MAX_DEVICES, cost_segment)

    print("\n--- Random-Fit ---")
    df_rand = evaluate_latency_vs_devices_random(L, MAX_DEVICES, cost_segment, trials=RAND_TRIALS, seed=RAND_SEED)

    # Optional: print merged table
    merged = (df_beam.merge(df_opt, on="Devices").merge(df_rand, on="Devices"))
    print("\nCombined results:\n", merged.to_string(index=False))

    # Plot all three in one figure
    plot_latency_and_time_all(df_beam, df_opt, df_rand, inset_at=INSET_AT)