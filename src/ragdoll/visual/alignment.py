from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from ragdoll.config import VisualizeAlignmentConfig
from ragdoll.jsonl import read_jsonl
from ragdoll.nuggetizer.metrics import METRIC_COLUMNS, cell_metric, mean, run_means
from ragdoll.stats import kendall_tau

TASKS = ("GEN", "AUGGEN", "COMBINED", "all")
TOPIC_COLORS = {"GEN": "#1a1ae6", "AUGGEN": "#5512c0", "COMBINED": "#9aa0a6", "all": "#1a1ae6"}
RUN_COLORS = {"GEN": "#ff0000", "AUGGEN": "#ffa500", "COMBINED": "#555555", "all": "#ff0000"}
MARKERS = {"GEN": "o", "AUGGEN": "s", "COMBINED": "D", "all": "o"}
TASK_LABEL = {"GEN": "RAG", "AUGGEN": "AG", "COMBINED": "combined", "all": ""}
TOPIC_ALPHA = 0.30
METRIC_TEX = {
    "V_strict": r"$V_{\mathrm{strict}}$",
    "A_strict": r"$A_{\mathrm{strict}}$",
    "V": r"$V$",
    "A": r"$A$",
}

Pairs = list[tuple[str, float, float]]
Tau = tuple[float, float, float]


def _pyplot():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for visualization; install with `uv sync --extra viz` "
            "or `pip install 'ragdoll[viz]'`"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def alignment_from_assignments(
    system: Path,
    reference: Path,
    *,
    metric: str,
    group_of: Callable[[str], str] | None = None,
) -> tuple[Pairs, Pairs, Tau]:
    column = METRIC_COLUMNS[metric]
    sys_cells = [cell_metric(row) for row in read_jsonl(Path(system))]
    ref_cells = [cell_metric(row) for row in read_jsonl(Path(reference))]
    ref_by_key = {(c.qid, c.run_id): c for c in ref_cells}

    cell_pairs: Pairs = []
    per_qid: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for cell in sys_cells:
        ref = ref_by_key.get((cell.qid, cell.run_id))
        if ref is None:
            continue
        group = group_of(cell.run_id) if group_of else "all"
        x, y = cell.metric(metric), ref.metric(metric)
        cell_pairs.append((group, x, y))
        per_qid[cell.qid].append((x, y))

    sys_runs, ref_runs = run_means(sys_cells), run_means(ref_cells)
    shared_runs = sorted(set(sys_runs) & set(ref_runs))
    run_pairs: Pairs = [
        (group_of(run_id) if group_of else "all", sys_runs[run_id][column], ref_runs[run_id][column])
        for run_id in shared_runs
    ]

    run_tau = kendall_tau(
        [sys_runs[r][column] for r in shared_runs],
        [ref_runs[r][column] for r in shared_runs],
    )
    cell_tau = kendall_tau([x for _, x, _ in cell_pairs], [y for _, _, y in cell_pairs])
    per_topic_tau = mean(
        kendall_tau([x for x, _ in vals], [y for _, y in vals]) for vals in per_qid.values()
    )
    return cell_pairs, run_pairs, (run_tau, cell_tau, per_topic_tau)


def plot_alignment(
    *,
    cell_pairs: Pairs,
    run_pairs: Pairs,
    x_label: str,
    y_label: str,
    metric: str,
    out_path: Path,
    tau: Tau | None = None,
    compact: bool = False,
) -> Path:
    plt, Line2D = _pyplot()
    present = {g for g, _, _ in cell_pairs} | {g for g, _, _ in run_pairs}
    order = [task for task in TASKS if task in present] or ["all"]

    fig, ax = plt.subplots(figsize=(3.6, 3.6) if compact else (5.0, 5.0))
    s_topic, s_run = (9, 34) if compact else (14, 46)

    for task in order:
        pts = [(x, y) for g, x, y in cell_pairs if g == task]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=s_topic, alpha=TOPIC_ALPHA,
                       marker=MARKERS.get(task, "o"), color=TOPIC_COLORS.get(task, "#1a1ae6"), linewidths=0)
    for task in order:
        pts = [(x, y) for g, x, y in run_pairs if g == task]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=s_run, edgecolor="none",
                       marker=MARKERS.get(task, "o"), color=RUN_COLORS.get(task, "#ff0000"))

    ax.plot([0, 1], [0, 1], color="#8c8c8c", linewidth=1.0, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    m = METRIC_TEX.get(metric, metric)
    ax.set_xlabel(f"{x_label}  {m}", fontsize=15 if compact else 12)
    ax.set_ylabel(f"{y_label}  {m}", fontsize=15 if compact else 12)
    if compact:
        ax.tick_params(labelsize=10)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])

    if tau is not None:
        rt, trt, ptt = tau
        ax.annotate(
            f"Kendall $\\tau$\nrun: {rt:.3f}\ntopic/run: {trt:.3f}\nper-topic avg: {ptt:.3f}",
            xy=(0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=10 if compact else 9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    if not compact and order != ["all"]:
        handles = []
        for task in order:
            label = TASK_LABEL.get(task, task)
            handles.append(Line2D([], [], marker=MARKERS.get(task, "o"), color=TOPIC_COLORS.get(task),
                                  alpha=0.55, linestyle="", label=f"{label} topic/run".strip()))
            handles.append(Line2D([], [], marker=MARKERS.get(task, "o"), color=RUN_COLORS.get(task),
                                  linestyle="", label=f"{label} run".strip()))
        ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def render_alignment(config: VisualizeAlignmentConfig) -> None:
    cell_pairs, run_pairs, tau = alignment_from_assignments(
        config.system, config.reference, metric=config.metric
    )
    out = plot_alignment(
        cell_pairs=cell_pairs,
        run_pairs=run_pairs,
        x_label=config.x_label,
        y_label=config.y_label,
        metric=config.metric,
        out_path=config.output_file,
        tau=tau,
        compact=config.compact,
    )
    print(f"wrote {out} cells={len(cell_pairs)} runs={len(run_pairs)} run_tau={tau[0]:.3f}")
