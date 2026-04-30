import time
import heapq
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


# =========================================================
# LOAD DATA (NEW CSV FORMAT)
# =========================================================

def load_data(inference_csv, transmission_csv):

    def extract_number(x):
        match = re.match(r"(\d+)", str(x))
        return int(match.group(1)) if match else None

    df_inf = pd.read_csv(inference_csv)
    df_trans = pd.read_csv(transmission_csv)

    latency_dict = {}

    for _, row in df_inf.iterrows():
        a = extract_number(row["Start_Layer"])
        b = extract_number(row["End_Layer"])
        latency = float(str(row["latency_ms"]).replace(",", "."))

        latency_dict[(a, b)] = latency

    transmission_times = df_trans["Transmission_Time"] \
        .astype(str).str.replace(",", ".", regex=False) \
        .astype(float).tolist()

    L = max(b for (_, b) in latency_dict.keys())

    print(f"Loaded {len(latency_dict)} segments. Max layer = {L}")

    return latency_dict, transmission_times, L


# =========================================================
# MIN STEPS
# =========================================================

def compute_min_steps_to_end(L, latency_dict):

    from collections import defaultdict, deque

    reverse_graph = defaultdict(list)

    for (a, b) in latency_dict.keys():
        reverse_graph[b].append(a)

    dist = {L: 0}
    queue = deque([L])

    while queue:
        node = queue.popleft()
        for prev in reverse_graph[node]:
            if prev not in dist:
                dist[prev] = dist[node] + 1
                queue.append(prev)

    return dist


# =========================================================
# COST FUNCTION
# =========================================================

def make_cost_segment(latency_dict, transmission_times, L):

    def cost_segment(a, b, device_idx):

        if (a, b) not in latency_dict:
            return None

        processing_time = latency_dict[(a, b)]
        transmission_time = transmission_times[b - 1] if b < L else 0.0

        return processing_time + transmission_time

    return cost_segment


# =========================================================
# 🔥 BEAM (FIXED)
# =========================================================

def beam_search_split(L, N, B, cost_segment, latency_dict, min_steps):

    beam = [(0.0, 0, [])]

    for k in range(1, N + 1):

        new_beam = []

        for cost, pos, splits in beam:

            upper_limit = L if k == N else L - (N - k)

            for nxt in range(pos + 1, upper_limit + 1):

                seg_cost = cost_segment(pos + 1, nxt, k)
                if seg_cost is None:
                    continue

                if k < N:
                    remaining = N - k
                    if nxt not in min_steps or min_steps[nxt] > remaining:
                        continue

                new_beam.append((cost + seg_cost, nxt, splits + [nxt]))

        if not new_beam:
            continue

        best = sorted(new_beam, key=lambda x: x[0])[:B // 2]
        far = sorted(new_beam, key=lambda x: -x[1])[:B // 2]

        beam = best + far

    final = [(c, s) for c, pos, s in beam if pos == L]

    if not final:
        # 🔥 fallback (CRITICAL FIX)
        best_cost, best_pos, best_splits = min(beam, key=lambda x: x[0])
        return best_splits, best_cost

    best_cost, best_splits = min(final, key=lambda x: x[0])

    return best_splits[:-1], best_cost


# =========================================================
# GREEDY
# =========================================================

def greedy_split(L, N, cost_segment, latency_dict, min_steps):

    splits = []
    total_cost = 0.0
    pos = 0

    for k in range(1, N):

        upper_limit = L - (N - k)

        best_next = None
        best_cost = float("inf")

        for nxt in range(pos + 1, upper_limit + 1):

            seg_cost = cost_segment(pos + 1, nxt, k)
            if seg_cost is None:
                continue

            remaining = N - k
            if nxt not in min_steps or min_steps[nxt] > remaining:
                continue

            if seg_cost < best_cost:
                best_cost = seg_cost
                best_next = nxt

        if best_next is None:
            raise RuntimeError("Greedy failed")

        splits.append(best_next)
        total_cost += best_cost
        pos = best_next

    final_cost = cost_segment(pos + 1, L, N)
    if final_cost is None:
        raise RuntimeError("Greedy final failed")

    total_cost += final_cost

    return splits, total_cost


# =========================================================
# FIRST-FIT
# =========================================================

def first_fit_split(L, N, cost_segment, latency_dict, min_steps):

    splits = []
    total_cost = 0.0
    pos = 0

    for k in range(1, N):

        upper_limit = L - (N - k)

        for nxt in range(pos + 1, upper_limit + 1):

            seg_cost = cost_segment(pos + 1, nxt, k)
            if seg_cost is None:
                continue

            remaining = N - k
            if nxt not in min_steps or min_steps[nxt] > remaining:
                continue

            splits.append(nxt)
            total_cost += seg_cost
            pos = nxt
            break
        else:
            raise RuntimeError("First-Fit failed")

    final_cost = cost_segment(pos + 1, L, N)
    if final_cost is None:
        raise RuntimeError("First-Fit final failed")

    total_cost += final_cost

    return splits, total_cost


# =========================================================
# EVALUATION (OLD STYLE)
# =========================================================

def evaluate_latency_vs_devices_beam(L, max_devices, B, cost_segment, latency_dict, min_steps):
    rows = []
    for N in range(2, max_devices + 1):
        t0 = time.time()
        _, cost = beam_search_split(L, N, B, cost_segment, latency_dict, min_steps)
        t1 = time.time()
        print(f"[Beam] N={N} | Latency={cost:.3f} | Time={t1 - t0:.3f}s")
        rows.append((N, cost, t1 - t0))
    return pd.DataFrame(rows, columns=["Devices", "Beam_Latency", "Beam_Time"])


def evaluate_latency_vs_devices_greedy(L, max_devices, cost_segment, latency_dict, min_steps):

    rows = []

    for N in range(2, max_devices + 1):

        try:
            t0 = time.time()

            _, cost = greedy_split(L, N, cost_segment, latency_dict, min_steps)

            t1 = time.time()

            print(f"[Greedy] N={N} | Latency={cost:.3f} | Time={t1 - t0:.3f}s")

            rows.append((N, cost, t1 - t0))

        except Exception as e:
            print(f"[Greedy] N={N} | FAILED")

            rows.append((N, np.nan, np.nan))

    return pd.DataFrame(rows, columns=["Devices", "Greedy_Latency", "Greedy_Time"])


def evaluate_latency_vs_devices_firstfit(L, max_devices, cost_segment, latency_dict, min_steps):

    rows = []

    for N in range(2, max_devices + 1):

        try:
            t0 = time.time()

            _, cost = first_fit_split(L, N, cost_segment, latency_dict, min_steps)

            t1 = time.time()

            print(f"[First-Fit] N={N} | Latency={cost:.3f} | Time={t1 - t0:.3f}s")

            rows.append((N, cost, t1 - t0))

        except Exception as e:
            print(f"[First-Fit] N={N} | FAILED")

            rows.append((N, np.nan, np.nan))

    return pd.DataFrame(rows, columns=["Devices", "FirstFit_Latency", "FirstFit_Time"])


# =========================================================
# PLOT (SAME AS OLD)
# =========================================================

def plot_latency_and_time_all(
    df_beam,
    df_greedy,
    df_firstfit,
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

    ax1.set_xlabel("Number of Devices (N)", fontsize=16)
    ax1.set_ylabel("Total Latency (ms)", fontsize=16)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---- Processing-time bars ----
    width = 0.18
    x = df_beam["Devices"].values

    bars_beam = ax2.bar(x - 1.5 * width, df_beam["Beam_Time"],
                        width=width, alpha=0.3, label="Beam Time (s)")
    bars_greedy = ax2.bar(x - 0.5 * width, df_greedy["Greedy_Time"],
                          width=width, alpha=0.3, label="Greedy Time (s)")
    bars_firstfit = ax2.bar(x + 0.5 * width, df_firstfit["FirstFit_Time"],
                            width=width, alpha=0.3, label="First-Fit Time (s)")

    ax2.set_ylabel("Processing Time (s)", fontsize=16)

    # ---- BAR LABELS (IMPORTANT) ----
    def add_bar_labels(axis, bars):
        for b in bars:
            h = b.get_height()
            if h > 0.005:
                axis.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    f"{h:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )
            else:
                axis.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    add_bar_labels(ax2, bars_beam)
    add_bar_labels(ax2, bars_greedy)
    add_bar_labels(ax2, bars_firstfit)

    # ---- LEGEND (CRITICAL) ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower left",
        bbox_to_anchor=(0.55, 0.12),
        fontsize=12,
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig("Latency_and_Processing_Time_Heuristics_RestNet50.pdf")
    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    latency_dict, transmission_times, L = load_data(
        "/home/zied/Documents/work folder/Journal-extension-code/split_algorithms/data_1/inference_times_ResNet50.csv",
        "/home/zied/Documents/work folder/Journal-extension-code/split_algorithms/data_1/layer_transmission_times_ResNet50.csv"
    )

    cost_segment = make_cost_segment(latency_dict, transmission_times, L)
    min_steps = compute_min_steps_to_end(L, latency_dict)

    df_beam = evaluate_latency_vs_devices_beam(L, 20, 300, cost_segment, latency_dict, min_steps)
    df_greedy = evaluate_latency_vs_devices_greedy(L, 20, cost_segment, latency_dict, min_steps)
    df_firstfit = evaluate_latency_vs_devices_firstfit(L, 20, cost_segment, latency_dict, min_steps)

    plot_latency_and_time_all(df_beam, df_greedy, df_firstfit)