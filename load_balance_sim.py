"""
load_balance_sim.py
Simulate Round-Robin vs Best-Fit-Decreasing load balancing for a batch of independent tasks.
Produces printed metrics and a saved/visible matplotlib figure.

Requirements:
  pip install matplotlib numpy
Run:
  python load_balance_sim.py
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------
NUM_TASKS = 200               # number of tasks in the batch
NUM_PROCESSORS = 8            # number of processors
PROC_SPEEDS = None            # None -> homogeneous speeds of 1.0, or set list length NUM_PROCESSORS
TASK_DIST = "pareto"         # "uniform", "exponential", "pareto"
UNIFORM_RANGE = (1, 10)       # for uniform
EXP_SCALE = 4.0               # scale for exponential
PARETO_A = 2.5                # shape for pareto (Pareto Type I), heavier tail when close to 1
SEED = 42                     # random seed for reproducibility
SAVE_PLOT = True
PLOT_FILENAME = "lb_result.png"

random.seed(SEED)
np.random.seed(SEED)

# -----------------------
# Helper: generate tasks
# -----------------------
def generate_tasks(n, dist="uniform"):
    if dist == "uniform":
        low, high = UNIFORM_RANGE
        workload = np.random.randint(low, high + 1, size=n).astype(float)
    elif dist == "exponential":
        workload = np.random.exponential(scale=EXP_SCALE, size=n)
    elif dist == "pareto":
        # numpy pareto gives samples from Pareto Type I with shape a: X ~ pareto(a) -> heavy-tail.
        # shift to make minimum ~1 and scale to reasonable range
        a = PARETO_A
        raw = np.random.pareto(a, size=n) + 1.0
        workload = raw * (EXP_SCALE / 2.0)  # scale factor to keep magnitudes reasonable
    else:
        raise ValueError("Unknown dist")
    return workload.tolist()

# -----------------------
# Scheduling algorithms
# -----------------------
def round_robin_assign(tasks, num_procs, proc_speeds):
    """
    Assign tasks by round-robin order.
    Returns list of lists (tasks assigned per processor) and per-processor load in time units.
    """
    proc_tasks = [[] for _ in range(num_procs)]
    proc_load = [0.0 for _ in range(num_procs)]
    rr = 0
    for t in tasks:
        i = rr % num_procs
        proc_tasks[i].append(t)
        proc_load[i] += t / proc_speeds[i]
        rr += 1
    return proc_tasks, proc_load

def best_fit_decreasing_assign(tasks, num_procs, proc_speeds):
    """
    Best-Fit Decreasing: sort tasks descending by size, then assign each to the processor
    with the current minimum load (considering speed).
    """
    sorted_tasks = sorted(tasks, reverse=True)
    proc_tasks = [[] for _ in range(num_procs)]
    proc_load = [0.0 for _ in range(num_procs)]  # measured in time (work / speed)
    for t in sorted_tasks:
        # choose processor with minimum current load (time)
        victim = min(range(num_procs), key=lambda i: proc_load[i])
        proc_tasks[victim].append(t)
        proc_load[victim] += t / proc_speeds[victim]
    return proc_tasks, proc_load

# -----------------------
# Metrics & Reporting
# -----------------------
def compute_metrics(proc_load):
    makespan = max(proc_load)
    total_work_time = sum(proc_load)
    num_procs = len(proc_load)
    utilization = [L / makespan if makespan > 0 else 0.0 for L in proc_load]  # fraction of time active
    avg_util = sum(utilization) / num_procs
    load_std = float(np.std(proc_load))
    load_mean = float(np.mean(proc_load))
    imbalance_pct = (load_std / load_mean * 100.0) if load_mean > 0 else 0.0
    return {
        "makespan": makespan,
        "total_work_time": total_work_time,
        "avg_util": avg_util,
        "load_std": load_std,
        "imbalance_pct": imbalance_pct,
        "per_proc": proc_load
    }

def print_summary(name, metrics):
    print(f"=== {name} ===")
    print(f"Makespan (time until all tasks finished): {metrics['makespan']:.4f}")
    print(f"Total work-time (sum of processor times): {metrics['total_work_time']:.4f}")
    print(f"Average utilization (fraction): {metrics['avg_util']:.4f}")
    print(f"Load std dev: {metrics['load_std']:.4f}")
    print(f"Imbalance (% of mean): {metrics['imbalance_pct']:.2f}%")
    print()

# -----------------------
# Main simulation
# -----------------------
def main():
    # processor speeds
    if PROC_SPEEDS is None:
        proc_speeds = [1.0] * NUM_PROCESSORS
    else:
        if len(PROC_SPEEDS) != NUM_PROCESSORS:
            raise ValueError("PROC_SPEEDS length mismatch")
        proc_speeds = PROC_SPEEDS.copy()

    # generate tasks
    tasks = generate_tasks(NUM_TASKS, TASK_DIST)
    print(f"Generated {NUM_TASKS} tasks (dist={TASK_DIST}). Example workloads: {tasks[:8]}")

    # round-robin
    rr_tasks, rr_load = round_robin_assign(tasks, NUM_PROCESSORS, proc_speeds)
    rr_metrics = compute_metrics(rr_load)
    print_summary("Round-Robin", rr_metrics)

    # best-fit decreasing
    bf_tasks, bf_load = best_fit_decreasing_assign(tasks, NUM_PROCESSORS, proc_speeds)
    bf_metrics = compute_metrics(bf_load)
    print_summary("Best-Fit-Decreasing", bf_metrics)

    # Plotting: side-by-side bar charts of per-processor load
    procs = list(range(NUM_PROCESSORS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11,5))
    ax.bar([p - width/2 for p in procs], rr_load, width=width, label="Round-Robin")
    ax.bar([p + width/2 for p in procs], bf_load, width=width, label="Best-Fit-Decreasing")
    ax.set_xlabel("Processor ID")
    ax.set_ylabel("Total execution time (work/speed)")
    ax.set_title(f"Load per processor: Round-Robin vs Best-Fit (Ntasks={NUM_TASKS}, Nprocs={NUM_PROCESSORS})")
    ax.set_xticks(procs)
    ax.legend()
    plt.tight_layout()
    if SAVE_PLOT:
        plt.savefig(PLOT_FILENAME, dpi=200)
        print(f"Saved plot to {PLOT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    main()
    print("Script is running...")

