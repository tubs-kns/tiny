"""
random_fit_baseline.py
----------------------
Standalone Random-Fit baseline for split selection.

- Loads inference_times.csv and layer_transmission_times.csv
- Defines the same cost model as your main script
- Implements Random-Fit (single-shot or multi-trial)
- Evaluates latency vs. number of devices
- Prints a results table and can plot (optional)
"""
import random
import time
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


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
# 2) Cost model (same as your script)
# =========================================================

def make_cost_segment(inference_times: List[float], transmission_times: List[float], L: int
) -> Callable[[int, int, int], float]:
    """
    Returns a cost_segment(a,b,device_idx) closure consistent with your script.
    """
    def cost_segment(a: int, b: int, device_idx: int) -> float:
        # layers [a..b], 1-based inclusive
        processing_time = inference_times[b - 1] - inference_times[a - 1]
        if processing_time < 0:
            processing_time = abs(processing_time)
        transmission_time = transmission_times[b - 1] if b < L else 0.0
        return processing_time + transmission_time
    return cost_segment


# =========================================================
# 3) Random-Fit splitter
# =========================================================

def random_fit_split(
    L: int,
    N: int,
    cost_segment: Callable[[int, int, int], float],
    trials: int = 1,
    seed: int | None = None
) -> Tuple[List[int], float]:
    """
    Randomly selects N-1 unique split points from {1..L-1}.
    If trials > 1, samples multiple random configurations and returns the best.
    Returns (best_splits, best_cost), where splits is a sorted list of cut indices.
    """
    rng = random.Random(seed)
    best_splits = None
    best_cost = float("inf")

    if N < 2:
        raise ValueError("N must be >= 2 (at least one split creates 2 segments).")
    if N > L:
        raise ValueError("N cannot exceed number of layers L.")

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
# 4) Evaluation loop
# =========================================================

def evaluate_random_fit_vs_devices(
    L: int,
    cost_segment: Callable[[int, int, int], float],
    max_devices: int,
    trials: int = 200,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    For N = 2..max_devices:
      - run Random-Fit 'trials' times with different seeds
      - report mean, std, best latency and wall time

    Returns a DataFrame with columns:
      Devices, Random_Mean_Latency, Random_Std_Latency, Random_Best_Latency, Random_Time
    """
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        latencies = []
        for t in range(trials):
            seed = base_seed + N * 1000 + t
            _, cost = random_fit_split(L, N, cost_segment, trials=1, seed=seed)
            latencies.append(cost)
        t1 = time.time()

        latencies = np.array(latencies, dtype=float)
        mean_lat = float(latencies.mean())
        std_lat = float(latencies.std())
        best_lat = float(latencies.min())
        wall = t1 - t0

        print(f"[RandomFit] N={N} | mean={mean_lat:.3f} ± {std_lat:.3f} | best={best_lat:.3f} | time={wall:.3f}s")
        rows.append((N, mean_lat, std_lat, best_lat, wall))

    return pd.DataFrame(rows, columns=[
        "Devices", "Random_Mean_Latency", "Random_Std_Latency", "Random_Best_Latency", "Random_Time"
    ])


# =========================================================
# 5) Plotting (optional)
# =========================================================

def plot_random_fit(df_rand: pd.DataFrame, title: str = "Random-Fit Baseline"):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # Latency (mean ± 1 std)
    ax1.plot(df_rand["Devices"], df_rand["Random_Mean_Latency"],
             marker="^", linewidth=2, label="Random-Fit Latency (mean)")
    ax1.fill_between(
        df_rand["Devices"],
        df_rand["Random_Mean_Latency"] - df_rand["Random_Std_Latency"],
        df_rand["Random_Mean_Latency"] + df_rand["Random_Std_Latency"],
        alpha=0.15, label="±1σ"
    )
    ax1.plot(df_rand["Devices"], df_rand["Random_Best_Latency"],
             marker="o", linewidth=1.5, linestyle="--", label="Random-Fit Latency (best)")

    ax1.set_xlabel("Number of Devices (N)", fontsize=14)
    ax1.set_ylabel("Total Latency (ms)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title(title, fontsize=15)

    # Processing time bars
    width = 0.25
    bars = ax2.bar(df_rand["Devices"] + 0.0, df_rand["Random_Time"],
                   width=width, alpha=0.3, label="Processing Time (s)")
    ax2.set_ylabel("Processing Time (s)", fontsize=14)

    # Labels on bars
    ymin, ymax = ax2.get_ylim()
    headroom = (ymax - ymin) * 0.03 if ymax > ymin else 0.05
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax2.text(b.get_x() + b.get_width()/2, h + headroom,
                     f"{h:.0f}s" if h >= 1 else f"{h:.2f}s",
                     ha="center", va="bottom", fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower left")

    plt.tight_layout()
    fig.savefig("RandomFit_baseline.pdf")
    plt.show()


# =========================================================
# 6) Main
# =========================================================

if __name__ == "__main__":
    # --- Config ---
    MAX_DEVICES = 6       # evaluate N = 2..MAX_DEVICES
    TRIALS = 200          # random samples per N (set to 1 for single-shot Random-Fit)
    BASE_SEED = 42        # reproducibility
    PLOT = True           # set False to skip plotting

    # Load data
    inference_times, transmission_times, L = load_data()
    cost_segment = make_cost_segment(inference_times, transmission_times, L)

    # Evaluate Random-Fit
    df_rand = evaluate_random_fit_vs_devices(
        L=L,
        cost_segment=cost_segment,
        max_devices=MAX_DEVICES,
        trials=TRIALS,
        base_seed=BASE_SEED
    )

    print("\nRandom-Fit results:\n", df_rand.to_string(index=False))

    # Plot
    if PLOT:
        plot_random_fit(df_rand)
