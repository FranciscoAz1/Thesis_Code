import argparse
import csv
import os
from typing import List, Dict, Tuple

# We keep it simple: no pandas; only matplotlib is required for plotting.
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required. Please install it with: python -m pip install matplotlib"
    ) from e

def _cumsum(xs: List[int]) -> List[int]:
    total = 0
    out: List[int] = []
    for v in xs:
        total += int(v)
        out.append(total)
    return out


def read_benchmark(csv_path: str) -> Tuple[List[int], List[int], List[int], Dict[str, List[float]]]:
    """Read the benchmark CSV and return X arrays and metrics.

    Returns:
        total_files (List[int]): num_entidades * num_pastas_per_entidade * num_ficheiros_per_pasta (for insert)
        total_steps (List[int]): num_fluxos * num_etapas_per_fluxo (for queries)
        metrics (Dict[str, List[float]]): time series for insert and queries
            keys: 'insert_sample_data', 'query_fluxo_etapas', 'query_entidade_hierarchy', 'global_semantic_search'
    """
    total_files: List[int] = []
    total_steps: List[int] = []
    total_objects: List[int] = []  # Total objects created in Weaviate per dataset
    metrics: Dict[str, List[float]] = {
        'insert_sample_data': [],
        'query_fluxo_etapas': [],
        'query_entidade_hierarchy': [],
        'global_semantic_search': [],
        # Optional/experimental columns for flux-level semantic search timing
        # We'll support multiple aliases to be resilient to CSV naming.
        'flux_semantic_search': [],
        'entity_flux_semantic_search': [],
    }

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # For insert X: total files per dataset
                num_entidades = int(row['num_entidades'])
                num_pastas_per_entidade = int(row['num_pastas_per_entidade'])
                num_ficheiros_per_pasta = int(row['num_ficheiros_per_pasta'])
                files_count = num_entidades * num_pastas_per_entidade * num_ficheiros_per_pasta
                total_files.append(files_count)

                # For queries X: total steps per dataset
                num_fluxos = int(row['num_fluxos'])
                num_etapas_per_fluxo = int(row['num_etapas_per_fluxo'])
                steps_count = num_fluxos * num_etapas_per_fluxo
                total_steps.append(steps_count)

                # Compute total objects in Weaviate per dataset
                # Objects per dataset: entidades + pastas + ficheiros + fluxos + etapas
                obj_count = (
                    num_entidades
                    + (num_entidades * num_pastas_per_entidade)
                    + files_count
                    + num_fluxos
                    + steps_count
                )
                total_objects.append(obj_count)

                # Collect metrics of interest (default to 0.0 if missing)
                for key in metrics.keys():
                    val = float(row.get(key, '0') or 0)
                    metrics[key].append(val)
            except (KeyError, ValueError):
                # Skip malformed rows
                continue

    # Do not sort here; return raw aligned arrays. Sorting is handled in plot functions per-axis need.
    return total_files, total_steps, total_objects, metrics


def _fit_line(x: List[int], y: List[float]) -> Tuple[float, float, float]:
    """Return slope m, intercept b, and R^2 of linear fit y = m*x + b."""
    import numpy as np
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    m, b = np.polyfit(x_arr, y_arr, 1)
    # Compute R^2
    y_pred = m * x_arr + b
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(m), float(b), float(r2)


def _fit_exponential(x: List[int], y: List[float]) -> Tuple[float, float, float] | None:
    """Fit y ≈ A * exp(k x). Returns (A, k, R^2) or None if not enough positive points.

    Uses linear least squares on ln(y) for points with y > 0.
    """
    import numpy as np
    pairs = [(float(xi), float(yi)) for xi, yi in zip(x, y) if yi > 0]
    if len(pairs) < 2:
        return None
    x_arr = np.array([p[0] for p in pairs], dtype=float)
    y_arr = np.array([p[1] for p in pairs], dtype=float)
    logy = np.log(y_arr)
    k, lnA = np.polyfit(x_arr, logy, 1)
    A = float(np.exp(lnA))
    # Predictions and R^2 in original domain
    y_pred = A * np.exp(k * x_arr)
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(A), float(k), float(r2)


def plot_insert_only(total_files: List[int], insert_latency: List[float], out_path: str):
    plt.figure(figsize=(9.5, 6))
    # Sort by total files
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [insert_latency[i] for i in order]

    plt.plot(x_sorted, y_sorted, marker='o', linewidth=2, color='#2563eb', label='Insert sample data')

    # Title with growth factor
    if len(insert_latency) >= 2 and insert_latency[0] > 0:
        growth = insert_latency[-1] / insert_latency[0]
        title = f"Insert Latency vs Number of Files (x{growth:.1f} growth)"
    else:
        title = "Insert Latency vs Number of Files"
    plt.title(title)

    # Linear fit for insert
    m, b, r2 = _fit_line(x_sorted, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_sorted, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98,
        0.93,
        eq_text,
        transform=ax.transAxes,
        fontsize=9,
        family='monospace',
        va='top',
        ha='right',
        alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Total files (num_entidades × num_pastas × num_ficheiros)")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_queries(total_steps: List[int], metrics: Dict[str, List[float]], out_path: str):
    plt.figure(figsize=(9.5, 6))

    series_order = [
        ('query_fluxo_etapas', '#16a34a', 'Query Fluxo → Etapas'),
        ('query_entidade_hierarchy', '#f59e0b', 'Query Entidade Hierarchy'),
        ('global_semantic_search', '#ef4444', 'Global Semantic Search'),
    ]

    # Sort by total steps once, then apply to all y series
    order = sorted(range(len(total_steps)), key=lambda i: total_steps[i])
    x_sorted = [total_steps[i] for i in order]

    for key, color, label in series_order:
        y = metrics.get(key, [])
        if not y:
            continue
        # Skip series that are all zeros
        if all(abs(v) < 1e-12 for v in y):
            continue
        y_sorted = [y[i] for i in order]
        plt.plot(x_sorted, y_sorted, marker='o', linewidth=2, color=color, label=label)

    plt.title("Query Latency vs Number of Files")

    # Linear fits for queries
    eq_lines = []
    for key, color, label in series_order:
        y = metrics.get(key, [])
        if not y or all(abs(v) < 1e-12 for v in y):
            continue
        y_sorted = [y[i] for i in order]
        m, b, r2 = _fit_line(x_sorted, y_sorted)
        eq_lines.append(f"{label} — Lin: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})")
        exp_fit = _fit_exponential(x_sorted, y_sorted)
        if exp_fit is not None:
            A, k, r2e = exp_fit
            eq_lines.append(f"{label} — Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")

    eq_text = "\n".join(eq_lines)
    if eq_text:
        ax = plt.gca()
        ax.text(
            0.98,
            0.93,
            eq_text,
            transform=ax.transAxes,
            fontsize=9,
            family='monospace',
            va='top',
            ha='right',
            alpha=0.95,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
        )

    plt.xlabel("Total (num_fluxos × num_etapas_per_fluxo)")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_semantic_vs_files(total_files: List[int], semantic_latency: List[float], out_path: str):
    """Plot Global Semantic Search latency vs total files (same X as insert)."""
    # Guard
    if not semantic_latency or all(abs(v) < 1e-12 for v in semantic_latency):
        print("Global Semantic Search series is empty or all zeros; skipping semantic plot.")
        return

    plt.figure(figsize=(9.5, 6))

    # Sort by total files
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [semantic_latency[i] for i in order]

    plt.plot(x_sorted, y_sorted, marker='o', linewidth=2, color='#ef4444', label='Global Semantic Search')

    # Title with growth factor
    if len(y_sorted) >= 2 and y_sorted[0] > 0:
        growth = y_sorted[-1] / y_sorted[0]
        title = f"Global Semantic Search vs Total Files (x{growth:.1f} growth)"
    else:
        title = "Global Semantic Search vs Total Files"
    plt.title(title)

    # Fits
    m, b, r2 = _fit_line(x_sorted, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_sorted, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)

    ax = plt.gca()
    ax.text(
        0.98, 0.95, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Total files (num_entidades × num_pastas × num_ficheiros)")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_queries_vs_files(total_files: List[int], metrics: Dict[str, List[float]], out_path: str):
    """Plot query latencies vs total files, focusing on Entidade Hierarchy (and optionally others)."""
    y_ent = metrics.get('query_entidade_hierarchy', [])
    if not y_ent or all(abs(v) < 1e-12 for v in y_ent):
        print("Query Entidade Hierarchy series is empty or all zeros; skipping queries-vs-files plot.")
        return

    plt.figure(figsize=(9.5, 6))

    # Sort by total files
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [y_ent[i] for i in order]

    plt.plot(x_sorted, y_sorted, marker='o', linewidth=2, color='#f59e0b', label='Query Entidade Hierarchy')

    # Title and fits
    plt.title("Query Latency vs Total Files (Entidade Hierarchy)")
    m, b, r2 = _fit_line(x_sorted, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_sorted, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98, 0.93, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Total files (num_entidades × num_pastas × num_ficheiros)")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_fluxo_vs_steps(total_steps: List[int], fluxo_latency: List[float], out_path: str):
    """Plot Query Fluxo → Etapas latency vs total steps only."""
    if not fluxo_latency or all(abs(v) < 1e-12 for v in fluxo_latency):
        print("Query Fluxo → Etapas series is empty or all zeros; skipping fluxo-vs-steps plot.")
        return

    plt.figure(figsize=(9.5, 6))

    # Sort by total steps
    order = sorted(range(len(total_steps)), key=lambda i: total_steps[i])
    x_sorted = [total_steps[i] for i in order]
    y_sorted = [fluxo_latency[i] for i in order]

    plt.plot(x_sorted, y_sorted, marker='o', linewidth=2, color='#16a34a', label='Query Fluxo → Etapas')
    plt.title("Query Fluxo → Etapas vs Total Steps")

    m, b, r2 = _fit_line(x_sorted, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_sorted, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98, 0.93, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Total (num_fluxos × num_etapas_per_fluxo)")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_fluxo_vs_files(total_files: List[int], fluxo_latency: List[float], out_path: str):
    """Plot Query Fluxo → Etapas latency vs total files only."""
    if not fluxo_latency or all(abs(v) < 1e-12 for v in fluxo_latency):
        print("Query Fluxo → Etapas series is empty or all zeros; skipping fluxo-vs-files plot.")
        return

    plt.figure(figsize=(9.5, 6))

    # Sort by total files
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [fluxo_latency[i] for i in order]

    plt.plot(x_sorted, y_sorted, marker='o', linewidth=2, color='#16a34a', label='Query Fluxo → Etapas')
    plt.title("Query Fluxo → Etapas vs Total Files")

    m, b, r2 = _fit_line(x_sorted, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_sorted, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98, 0.93, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Total files (num_entidades × num_pastas × num_ficheiros)")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")

def plot_semantic_combined(
    total_files: List[int],
    global_semantic: List[float],
    flux_semantic: List[float] | None,
    out_path: str,
    *,
    flux_label: str = 'Flux Semantic Search',
    flux_color: str = '#8b5cf6',
):
    """Plot Global vs Flux Semantic Search latency vs total files on the same chart.

    If one series is empty or all zeros, it will be skipped with a console note.
    """
    # Determine which series are valid
    has_global = bool(global_semantic) and not all(abs(v) < 1e-12 for v in global_semantic)
    has_flux = bool(flux_semantic) and not all(abs(v) < 1e-12 for v in flux_semantic or [])

    if not has_global and not has_flux:
        print("Both semantic series are empty or all zeros; skipping combined semantic plot.")
        return

    plt.figure(figsize=(9.5, 6))

    # Sort by total files
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]

    eq_lines: List[str] = []

    # Plot global
    if has_global:
        y_g = [global_semantic[i] for i in order]
        plt.plot(x_sorted, y_g, marker='o', linewidth=2, color='#ef4444', label='Global Semantic Search')
        m, b, r2 = _fit_line(x_sorted, y_g)
        eq_lines.append(f"Global — Lin: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})")
        exp_fit = _fit_exponential(x_sorted, y_g)
        if exp_fit is not None:
            A, k, r2e = exp_fit
            eq_lines.append(f"Global — Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")

    # Plot flux
    if has_flux and flux_semantic is not None:
        y_f = [flux_semantic[i] for i in order]
        plt.plot(x_sorted, y_f, marker='o', linewidth=2, color=flux_color, label=flux_label)
        m, b, r2 = _fit_line(x_sorted, y_f)
        eq_lines.append(f"{flux_label} — Lin: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})")
        exp_fit = _fit_exponential(x_sorted, y_f)
        if exp_fit is not None:
            A, k, r2e = exp_fit
            eq_lines.append(f"{flux_label} — Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")

    # Title
    title = "Semantic Latency vs Total Files"
    plt.title(title)

    # Equation box
    if eq_lines:
        eq_text = "\n".join(eq_lines)
        ax = plt.gca()
        ax.text(
            0.98, 0.95, eq_text, transform=ax.transAxes,
            fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
        )

    plt.xlabel("Total files (num_entidades × num_pastas × num_ficheiros)")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


# =========================
# CUMULATIVE-X VARIANTS
# =========================

def plot_insert_cumulative(total_files: List[int], insert_latency: List[float], out_path: str):
    plt.figure(figsize=(9.5, 6))
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [insert_latency[i] for i in order]
    x_cum = _cumsum(x_sorted)

    plt.plot(x_cum, y_sorted, marker='o', linewidth=2, color='#2563eb', label='Insert sample data')
    plt.title("Insert Latency vs Cumulative Objects")

    m, b, r2 = _fit_line(x_cum, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_cum, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98, 0.93, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Cumulative objects in weaviate")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_semantic_vs_files_cumulative(total_files: List[int], semantic_latency: List[float], out_path: str):
    if not semantic_latency or all(abs(v) < 1e-12 for v in semantic_latency):
        print("Global Semantic Search series is empty or all zeros; skipping cumulative semantic plot.")
        return
    plt.figure(figsize=(9.5, 6))
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [semantic_latency[i] for i in order]
    x_cum = _cumsum(x_sorted)

    plt.plot(x_cum, y_sorted, marker='o', linewidth=2, color='#ef4444', label='Global Semantic Search')
    plt.title("Global Semantic Search vs Cumulative Objects")

    m, b, r2 = _fit_line(x_cum, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_cum, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98, 0.95, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Cumulative objects in weaviate")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_queries_vs_files_cumulative(total_files: List[int], metrics: Dict[str, List[float]], out_path: str):
    y_ent = metrics.get('query_entidade_hierarchy', [])
    if not y_ent or all(abs(v) < 1e-12 for v in y_ent):
        print("Query Entidade Hierarchy series is empty or all zeros; skipping cumulative queries-vs-files plot.")
        return
    plt.figure(figsize=(9.5, 6))
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [y_ent[i] for i in order]
    x_cum = _cumsum(x_sorted)

    plt.plot(x_cum, y_sorted, marker='o', linewidth=2, color='#f59e0b', label='Query Entidade Hierarchy')
    plt.title("Query Latency vs Cumulative Objects (Entidade Hierarchy)")

    m, b, r2 = _fit_line(x_cum, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_cum, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98, 0.93, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Cumulative objects in weaviate")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_fluxo_vs_files_cumulative(total_files: List[int], fluxo_latency: List[float], out_path: str):
    if not fluxo_latency or all(abs(v) < 1e-12 for v in fluxo_latency):
        print("Query Fluxo → Etapas series is empty or all zeros; skipping cumulative fluxo-vs-files plot.")
        return
    plt.figure(figsize=(9.5, 6))
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    y_sorted = [fluxo_latency[i] for i in order]
    x_cum = _cumsum(x_sorted)

    plt.plot(x_cum, y_sorted, marker='o', linewidth=2, color='#16a34a', label='Query Fluxo → Etapas')
    plt.title("Query Fluxo → Etapas vs Cumulative Objects")

    m, b, r2 = _fit_line(x_cum, y_sorted)
    eq_lines = [f"Linear: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})"]
    exp_fit = _fit_exponential(x_cum, y_sorted)
    if exp_fit is not None:
        A, k, r2e = exp_fit
        eq_lines.append(f"Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")
    eq_text = "\n".join(eq_lines)
    ax = plt.gca()
    ax.text(
        0.98, 0.93, eq_text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )

    plt.xlabel("Cumulative objects in weaviate")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def plot_semantic_combined_cumulative(
    total_files: List[int],
    global_semantic: List[float],
    flux_semantic: List[float] | None,
    out_path: str,
    *,
    flux_label: str = 'Flux Semantic Search',
    flux_color: str = '#8b5cf6',
):
    has_global = bool(global_semantic) and not all(abs(v) < 1e-12 for v in global_semantic)
    has_flux = bool(flux_semantic) and not all(abs(v) < 1e-12 for v in flux_semantic or [])
    if not has_global and not has_flux:
        print("Both semantic series are empty or all zeros; skipping combined cumulative semantic plot.")
        return
    plt.figure(figsize=(9.5, 6))
    order = sorted(range(len(total_files)), key=lambda i: total_files[i])
    x_sorted = [total_files[i] for i in order]
    x_cum = _cumsum(x_sorted)
    eq_lines: List[str] = []

    if has_global:
        y_g = [global_semantic[i] for i in order]
        plt.plot(x_cum, y_g, marker='o', linewidth=2, color='#ef4444', label='Global Semantic Search')
        m, b, r2 = _fit_line(x_cum, y_g)
        eq_lines.append(f"Global — Lin: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})")
        exp_fit = _fit_exponential(x_cum, y_g)
        if exp_fit is not None:
            A, k, r2e = exp_fit
            eq_lines.append(f"Global — Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")

    if has_flux and flux_semantic is not None:
        y_f = [flux_semantic[i] for i in order]
        plt.plot(x_cum, y_f, marker='o', linewidth=2, color=flux_color, label=flux_label)
        m, b, r2 = _fit_line(x_cum, y_f)
        eq_lines.append(f"{flux_label} — Lin: y = {m:.4e} x + {b:.3f}  (R²={r2:.3f})")
        exp_fit = _fit_exponential(x_cum, y_f)
        if exp_fit is not None:
            A, k, r2e = exp_fit
            eq_lines.append(f"{flux_label} — Exp: y = {A:.3e} e^({k:.3e} x)  (R²={r2e:.3f})")

    plt.title("Semantic Latency vs Cumulative Objects")
    if eq_lines:
        eq_text = "\n".join(eq_lines)
        ax = plt.gca()
        ax.text(
            0.98, 0.95, eq_text, transform=ax.transAxes,
            fontsize=9, family='monospace', va='top', ha='right', alpha=0.95,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85),
        )
    plt.xlabel("Cumulative objects in weaviate")
    plt.ylabel("Latency (s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot latency vs number of files from benchmark CSV.")
    parser.add_argument(
        "--csv",
        default=os.path.join("tests", "benchmark_results.csv"),
        help="Path to the benchmark CSV (default: tests/benchmark_results256.csv)",
    )
    parser.add_argument(
        "--out-insert",
        default=os.path.join("scripts", "outputs", "insert_latency_vs_files.png"),
        help="Output PNG for insert latency plot",
    )
    parser.add_argument(
        "--out-insert-cum",
        default=os.path.join("scripts", "outputs", "insert_latency_vs_files_cum.png"),
        help="Output PNG for insert latency plot (cumulative X)",
    )
    parser.add_argument(
        "--out-queries",
        default=os.path.join("scripts", "outputs", "query_latency_vs_files.png"),
        help="Output PNG for queries latency plot",
    )
    parser.add_argument(
        "--out-semantic-files",
        default=os.path.join("scripts", "outputs", "semantic_latency_vs_files.png"),
        help="Output PNG for Global Semantic Search vs total files plot",
    )
    parser.add_argument(
        "--out-semantic-files-cum",
        default=os.path.join("scripts", "outputs", "semantic_latency_vs_files_cum.png"),
        help="Output PNG for Global Semantic Search vs cumulative total files plot",
    )
    parser.add_argument(
        "--out-semantic-combined",
        default=os.path.join("scripts", "outputs", "semantic_latency_combined.png"),
        help="Output PNG for combined Global vs Flux Semantic Search vs total files plot",
    )
    parser.add_argument(
        "--out-semantic-combined-cum",
        default=os.path.join("scripts", "outputs", "semantic_latency_combined_cum.png"),
        help="Output PNG for combined Global vs Flux Semantic Search vs cumulative total files plot",
    )
    parser.add_argument(
        "--out-queries-files",
        default=os.path.join("scripts", "outputs", "query_latency_vs_total_files.png"),
        help="Output PNG for Query (Entidade Hierarchy) latency vs total files",
    )
    parser.add_argument(
        "--out-queries-files-cum",
        default=os.path.join("scripts", "outputs", "query_latency_vs_total_files_cum.png"),
        help="Output PNG for Query (Entidade Hierarchy) latency vs cumulative total files",
    )
    parser.add_argument(
        "--out-fluxo-steps",
        default=os.path.join("scripts", "outputs", "query_fluxo_vs_steps.png"),
        help="Output PNG for Query Fluxo → Etapas vs total steps",
    )
    parser.add_argument(
        "--out-fluxo-files",
        default=os.path.join("scripts", "outputs", "query_fluxo_vs_files.png"),
        help="Output PNG for Query Fluxo → Etapas vs total files",
    )
    parser.add_argument(
        "--out-fluxo-files-cum",
        default=os.path.join("scripts", "outputs", "query_fluxo_vs_files_cum.png"),
        help="Output PNG for Query Fluxo → Etapas vs cumulative total files",
    )
    # Backward-compat arg (ignored) to avoid breaking previous calls
    parser.add_argument(
        "--out",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    total_files, total_steps, total_objects, metrics = read_benchmark(args.csv)
    if not total_files and not total_steps:
        raise SystemExit("No data parsed from CSV. Check the file format/headers.")

    # Ensure we explicitly include insert_sample_data latency as requested
    if 'insert_sample_data' not in metrics or not metrics['insert_sample_data']:
        raise SystemExit("CSV missing 'insert_sample_data' column or values.")

    # Produce separate plots
    plot_insert_only(total_files, metrics['insert_sample_data'], args.out_insert)
    plot_insert_cumulative(total_objects, metrics['insert_sample_data'], args.out_insert_cum)
    plot_queries(total_steps, metrics, args.out_queries)
    plot_semantic_vs_files(total_files, metrics.get('global_semantic_search', []), args.out_semantic_files)
    plot_semantic_vs_files_cumulative(total_objects, metrics.get('global_semantic_search', []), args.out_semantic_files_cum)
    plot_queries_vs_files(total_files, metrics, args.out_queries_files)
    plot_queries_vs_files_cumulative(total_objects, metrics, args.out_queries_files_cum)
    plot_fluxo_vs_steps(total_steps, metrics.get('query_fluxo_etapas', []), args.out_fluxo_steps)
    plot_fluxo_vs_files(total_files, metrics.get('query_fluxo_etapas', []), args.out_fluxo_files)
    plot_fluxo_vs_files_cumulative(total_objects, metrics.get('query_fluxo_etapas', []), args.out_fluxo_files_cum)

    # Determine flux semantic series using supported aliases; fallback to Query Fluxo if missing
    flux_series: List[float] | None = None
    flux_label = 'Flux Semantic Search'
    flux_color = '#8b5cf6'
    for alias in ("flux_semantic_search", "entity_flux_semantic_search"):
        series = metrics.get(alias, [])
        if series and not all(abs(v) < 1e-12 for v in series):
            flux_series = series
            break

    # Fallback to Query Fluxo -> Etapas if no flux semantic series
    if flux_series is None:
        q_flux = metrics.get('query_fluxo_etapas', [])
        if q_flux and not all(abs(v) < 1e-12 for v in q_flux):
            flux_series = q_flux
            flux_label = 'Query Fluxo → Etapas'
            flux_color = '#16a34a'
            print("No flux semantic series found; using 'query_fluxo_etapas' in combined semantic plot.")

    # Combined semantic plot (global + flux)
    plot_semantic_combined(
        total_files,
        metrics.get('global_semantic_search', []),
        flux_series,
        args.out_semantic_combined,
        flux_label=flux_label,
        flux_color=flux_color,
    )
    plot_semantic_combined_cumulative(
        total_objects,
        metrics.get('global_semantic_search', []),
        flux_series,
        args.out_semantic_combined_cum,
        flux_label=flux_label,
        flux_color=flux_color,
    )


if __name__ == "__main__":
    main()
