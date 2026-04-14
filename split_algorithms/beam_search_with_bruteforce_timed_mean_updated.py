"""
compare_beam_greedy_firstfit_random.py
--------------------------------------
Combined comparison of:
 - Beam Search
 - Greedy Search
 - First-Fit
 - Random-Fit (stochastic baseline)

Loads:
 - inference_times.csv   with column: Inference_Time
 - layer_transmission_times.csv  with column: Transmission_Time

Outputs:
 - One figure with latency curves and processing-time bars
"""

import time
import heapq
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

def make_cost_segment(
    inference_times: List[float],
    transmission_times: List[float],
    L: int
) -> Callable[[int, int, int], float]:
    """
    Returns cost_segment(a, b, device_idx).
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

        if k == N:
            valid_final = [item for item in new_beam if item[1] == L]
            beam = heapq.nsmallest(
                B,
                valid_final if valid_final else new_beam,
                key=lambda x: x[0]
            )
        else:
            beam = heapq.nsmallest(B, new_beam, key=lambda x: x[0])

    final_candidates = [(c, s) for c, pos, s in beam if pos == L]
    if not final_candidates:
        raise RuntimeError("No valid configuration reached the final layer.")

    best_cost, best_splits = min(final_candidates, key=lambda x: x[0])
    return best_splits[:-1], best_cost


# =========================================================
# 4) Greedy Search
# =========================================================

def greedy_split(
    L: int,
    N: int,
    cost_segment: Callable[[int, int, int], float]
) -> Tuple[List[int], float]:
    """
    At each step, choose the feasible next split point with the minimum
    immediate segment cost.
    """
    if N < 2:
        raise ValueError("N must be >= 2.")
    if N > L:
        raise ValueError("N cannot exceed L.")

    splits = []
    total_cost = 0.0
    pos = 0

    for k in range(1, N):
        upper_limit = L - (N - k)
        best_next = None
        best_seg_cost = float("inf")

        for nxt in range(pos + 1, upper_limit + 1):
            seg_cost = cost_segment(pos + 1, nxt, k)
            if seg_cost < best_seg_cost:
                best_seg_cost = seg_cost
                best_next = nxt

        if best_next is None:
            raise RuntimeError("Greedy search failed to find a feasible split.")

        splits.append(best_next)
        total_cost += best_seg_cost
        pos = best_next

    total_cost += cost_segment(pos + 1, L, N)
    return splits, total_cost


# =========================================================
# 5) First-Fit Search
# =========================================================

def first_fit_split(
    L: int,
    N: int,
    cost_segment: Callable[[int, int, int], float],
    threshold_mode: str = "average"
) -> Tuple[List[int], float]:
    """
    Scan feasible split points from left to right and select the first one
    whose segment cost is below a threshold.

    threshold_mode:
      - "average": use average segment budget estimated from full-model cost / N
      - "median":  use median of feasible segment costs at each step
    """
    if N < 2:
        raise ValueError("N must be >= 2.")
    if N > L:
        raise ValueError("N cannot exceed L.")

    splits = []
    total_cost = 0.0
    pos = 0

    # coarse global budget estimate
    global_budget = cost_segment(1, L, 1) / N

    for k in range(1, N):
        upper_limit = L - (N - k)
        candidates = []

        for nxt in range(pos + 1, upper_limit + 1):
            seg_cost = cost_segment(pos + 1, nxt, k)
            candidates.append((nxt, seg_cost))

        if not candidates:
            raise RuntimeError("First-Fit search found no feasible candidates.")

        if threshold_mode == "median":
            tau = float(np.median([c for _, c in candidates]))
        else:
            tau = global_budget

        chosen_next = None
        chosen_cost = None

        for nxt, seg_cost in candidates:
            if seg_cost <= tau:
                chosen_next = nxt
                chosen_cost = seg_cost
                break

        # fallback: choose last feasible split if none satisfies the threshold
        if chosen_next is None:
            chosen_next, chosen_cost = candidates[-1]

        splits.append(chosen_next)
        total_cost += chosen_cost
        pos = chosen_next

    total_cost += cost_segment(pos + 1, L, N)
    return splits, total_cost


# =========================================================
# 6) Random-Fit
# =========================================================

def random_fit_split(
    L: int,
    N: int,
    cost_segment: Callable[[int, int, int], float],
    trials: int = 1,
    seed: int | None = None
) -> Tuple[List[int], float]:
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
# 7) Evaluation loops
# =========================================================

def evaluate_latency_vs_devices_beam(L: int, max_devices: int, beam_width: int, cost_segment):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        _, best_cost = beam_search_split(L, N, beam_width, cost_segment)
        t1 = time.time()
        rows.append((N, best_cost, t1 - t0))
        print(f"[Beam]      N={N} | Latency={best_cost:.3f} | Time={t1 - t0:.3f}s")
    return pd.DataFrame(rows, columns=["Devices", "Beam_Latency", "Beam_Time"])


def evaluate_latency_vs_devices_greedy(L: int, max_devices: int, cost_segment):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        _, best_cost = greedy_split(L, N, cost_segment)
        t1 = time.time()
        rows.append((N, best_cost, t1 - t0))
        print(f"[Greedy]    N={N} | Latency={best_cost:.3f} | Time={t1 - t0:.3f}s")
    return pd.DataFrame(rows, columns=["Devices", "Greedy_Latency", "Greedy_Time"])


def evaluate_latency_vs_devices_firstfit(L: int, max_devices: int, cost_segment, threshold_mode: str = "average"):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        _, best_cost = first_fit_split(L, N, cost_segment, threshold_mode=threshold_mode)
        t1 = time.time()
        rows.append((N, best_cost, t1 - t0))
        print(f"[First-Fit] N={N} | Latency={best_cost:.3f} | Time={t1 - t0:.3f}s")
    return pd.DataFrame(rows, columns=["Devices", "FirstFit_Latency", "FirstFit_Time"])


def evaluate_latency_vs_devices_random(L: int, max_devices: int, cost_segment, trials: int = 200, seed: int = 42):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        latencies = []
        for t in range(trials):
            _, c = random_fit_split(L, N, cost_segment, trials=1, seed=seed + N * 1000 + t)
            latencies.append(c)
        t1 = time.time()

        latencies = np.array(latencies, dtype=float)
        mean_lat = float(latencies.mean())
        std_lat = float(latencies.std())
        best_lat = float(latencies.min())

        rows.append((N, mean_lat, std_lat, best_lat, (t1 - t0)/trials))
        print(
            f"[Random]    N={N} | mean={mean_lat:.3f} ± {std_lat:.3f} "
            f"| best={best_lat:.3f} | Time={t1 - t0:.3f}s"
        )

    return pd.DataFrame(rows, columns=[
        "Devices", "Random_Mean_Latency", "Random_Std_Latency", "Random_Best_Latency", "Random_Time"
    ])


# =========================================================
# 8) Plot
# =========================================================

def plot_latency_and_time_all(
    df_beam: pd.DataFrame,
    df_greedy: pd.DataFrame,
    df_firstfit: pd.DataFrame,
    df_rand: pd.DataFrame | None = None,
    inset_at: int | None = 4
):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # ---- Latency curves ----
    ax1.plot(df_beam["Devices"], df_beam["Beam_Latency"],
             marker="o", linewidth=2, label="Beam Search (latency)")
    ax1.plot(df_greedy["Devices"], df_greedy["Greedy_Latency"],
             marker="s", linewidth=2, label="Greedy Search (latency)")
    ax1.plot(df_firstfit["Devices"], df_firstfit["FirstFit_Latency"],
             marker="D", linewidth=2, label="First-Fit (latency)")
    #ax1.plot(df_rand["Devices"], df_rand["Random_Mean_Latency"],
    #         marker="^", linewidth=2, label="Random-Fit (latency)")

    ax1.set_xlabel("Number of Devices (N)", fontsize=16)
    ax1.set_ylabel("Total Latency (ms)", fontsize=16)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---- Processing-time bars ----
    width = 0.18
    x = df_beam["Devices"].values

    bars_beam = ax2.bar(x - 1.5 * width, df_beam["Beam_Time"], width=width, alpha=0.3, label="Beam Time (s)")
    bars_greedy = ax2.bar(x - 0.5 * width, df_greedy["Greedy_Time"], width=width, alpha=0.3, label="Greedy Time (s)")
    bars_firstfit = ax2.bar(x + 0.5 * width, df_firstfit["FirstFit_Time"], width=width, alpha=0.3, label="First-Fit Time (s)")
    #bars_rand = ax2.bar(x + 1.5 * width, df_rand["Random_Time"], width=width, alpha=0.3, label="Random-Fit Time (s)")

    ax2.set_ylabel("Processing Time (s)", fontsize=16)

    def add_bar_labels(axis, bars, fontsize=8):
        for b in bars:
            h = b.get_height()
            if h < 0.005 and h > 0:
                axis.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    f"0",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize
                )
            elif h > 0.005:
                axis.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    f"{h:.0f}s" if h >= 1 else f"{h:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize
                )

    add_bar_labels(ax2, bars_beam)
    add_bar_labels(ax2, bars_greedy)
    add_bar_labels(ax2, bars_firstfit)
    #add_bar_labels(ax2, bars_rand)

    # ---- Legend ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=14, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    '''
    # ---- Optional inset ----
    if inset_at is not None and inset_at in df_beam["Devices"].values:
        ax_ins = inset_axes(
            ax1, width="48%", height="48%", loc="lower left",
            bbox_to_anchor=(0.50, 0.18, 0.48, 0.48),
            bbox_transform=ax1.transAxes, borderpad=0.0
        )
        ax_ins2 = ax_ins.twinx()

        def filter_at(df, col="Devices", val=inset_at):
            f = df[df[col] == val]
            if f.empty:
                f = df[(df[col] >= val - 0.2) & (df[col] <= val + 0.2)]
            return f

        db = filter_at(df_beam)
        dg = filter_at(df_greedy)
        dff = filter_at(df_firstfit)
        #dr = filter_at(df_rand)

        if not db.empty:
            ax_ins.plot(db["Devices"], db["Beam_Latency"], marker="o", linewidth=2)
            ax_ins2.bar(db["Devices"] - 1.5 * 0.10, db["Beam_Time"], width=0.10, alpha=0.3)

        if not dg.empty:
            ax_ins.plot(dg["Devices"], dg["Greedy_Latency"], marker="s", linewidth=2)
            ax_ins2.bar(dg["Devices"] - 0.5 * 0.10, dg["Greedy_Time"], width=0.10, alpha=0.3)

        if not dff.empty:
            ax_ins.plot(dff["Devices"], dff["FirstFit_Latency"], marker="D", linewidth=2)
            ax_ins2.bar(dff["Devices"] + 0.5 * 0.10, dff["FirstFit_Time"], width=0.10, alpha=0.3)

        #if not dr.empty:
        #    ax_ins.plot(dr["Devices"], dr["Random_Mean_Latency"], marker="^", linewidth=2)
        #    ax_ins2.bar(dr["Devices"] + 1.5 * 0.10, dr["Random_Time"], width=0.10, alpha=0.3)

        yvals = []
        if not db.empty:
            yvals += db["Beam_Latency"].tolist()
        if not dg.empty:
            yvals += dg["Greedy_Latency"].tolist()
        if not dff.empty:
            yvals += dff["FirstFit_Latency"].tolist()
        #if not dr.empty:
        #     yvals += dr["Random_Mean_Latency"].tolist()

        if yvals:
            ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
            pad = max(10.0, 0.08 * (ymax - ymin if ymax > ymin else 50.0))
            ax_ins.set_ylim(ymin - pad, ymax + pad)

        ax_ins.grid(True, linestyle="--", alpha=0.6)
        ax_ins.set_title(f"Zoom @ N={inset_at}", fontsize=14, pad=2)
        ax_ins.tick_params(labelsize=12)
        ax_ins2.tick_params(labelsize=12)
    '''

    plt.tight_layout()
    plt.savefig("Latency_and_Processing_Time_Heuristics.pdf")
    plt.show()


# =========================================================
# 9) Main
# =========================================================

if __name__ == "__main__":
    # ---- Config ----
    BEAM_WIDTH = 300
    MAX_DEVICES = 20
    RAND_TRIALS = 4000
    RAND_SEED = 42
    INSET_AT = 3
    FIRSTFIT_THRESHOLD_MODE = "average"   # or "median"

    # Load data and cost
    inference_times, transmission_times, L = load_data()
    cost_segment = make_cost_segment(inference_times, transmission_times, L)

    # Evaluate
    print("\n--- Beam Search ---")
    df_beam = evaluate_latency_vs_devices_beam(L, MAX_DEVICES, BEAM_WIDTH, cost_segment)

    print("\n--- Greedy Search ---")
    df_greedy = evaluate_latency_vs_devices_greedy(L, MAX_DEVICES, cost_segment)

    print("\n--- First-Fit ---")
    df_firstfit = evaluate_latency_vs_devices_firstfit(
        L, MAX_DEVICES, cost_segment, threshold_mode=FIRSTFIT_THRESHOLD_MODE
    )

    #print("\n--- Random-Fit ---")
    #df_rand = evaluate_latency_vs_devices_random(
    #    L, MAX_DEVICES, cost_segment, trials=RAND_TRIALS, seed=RAND_SEED
    #)

    # Optional: print merged table
    #merged = (
    #    df_beam
    #    .merge(df_greedy, on="Devices")
    #    .merge(df_firstfit, on="Devices")
    #    .merge(df_rand, on="Devices")
    #)
    #print("\nCombined results:\n", merged.to_string(index=False))

    # Plot
    plot_latency_and_time_all(
        df_beam, df_greedy, df_firstfit, inset_at=INSET_AT
    )