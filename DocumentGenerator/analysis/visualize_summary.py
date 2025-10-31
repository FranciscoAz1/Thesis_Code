import json
from pathlib import Path



from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results_summ"
PLOTS_DIR = RESULTS_DIR / "plots"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hero metrics subset for the thesis main text
HERO_METRICS = ["token_recall", "jaccard", "overlap", "bert_cos"]
HERO_DIR = PLOTS_DIR / "hero"
HERO_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_summary(summary_path: Path) -> Dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_dataframe(summary: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = pd.DataFrame(summary.get("final_threshold_rows", []))
    medians = pd.DataFrame(summary.get("final_median_rows", []))

    # Normalize and type-cast for safety
    if not thresholds.empty:
        # Ensure numeric columns are numeric
        numeric_cols = [
            "best_threshold", "true_pass_rate", "false_pass_rate", "difference",
            "tpr_ci_low", "tpr_ci_high", "fpr_ci_low", "fpr_ci_high",
            "auc", "p_value", "median_gap", "n_true", "n_false",
            "class_balance_ratio", "median_overall", "median_true", "median_false",
            "doc_retrieved_pct", "avg_pass_rate", "overall_pass_rate"
        ]
        for c in numeric_cols:
            if c in thresholds.columns:
                thresholds[c] = pd.to_numeric(thresholds[c], errors="coerce")

        # Create readable labels
        thresholds["label"] = thresholds["metric"] + " | " + thresholds["source_file"].astype(str)

    if not medians.empty:
        for c in [
            "median_overall", "median_true", "median_false",
            "n_overall", "n_true", "n_false"
        ]:
            if c in medians.columns:
                medians[c] = pd.to_numeric(medians[c], errors="coerce")

        medians["label"] = medians["metric"] + " | " + medians["source_file"].astype(str)

    return thresholds, medians


def save_csvs(thresholds: pd.DataFrame, medians: pd.DataFrame, outdir: Path = PLOTS_DIR):
    if not thresholds.empty:
        thresholds.to_csv(outdir / "final_threshold_rows_flat.csv", index=False)
    if not medians.empty:
        medians.to_csv(outdir / "final_median_rows_flat.csv", index=False)


# ---------- New: Option B helpers ----------
def _ensure_metric_order(df: pd.DataFrame, col: str = "metric") -> pd.DataFrame:
    if col in df.columns:
        order = [
            # retrieval-facing
            "token_recall",
            "context_token_recall",
            # lexical/overlap
            "context_rouge1_f",
            "rouge1_f",
            "context_overlap",
            "overlap",
            # set-based
            "context_jaccard",
            "jaccard",
            # semantic
            "bert_cos",
        ]
        try:
            existing = list(pd.unique(df[col].astype(str)))
        except Exception:
            existing = list(set(df[col]))
        extra = [m for m in existing if m not in order]
        df[col] = pd.Categorical(df[col], categories=order + sorted(extra), ordered=True)
    return df


def export_median_first_table(medians: pd.DataFrame, outdir: Path = PLOTS_DIR):
    """Save median-first table (CSV + LaTeX) sorted by median_true - median_false (median_gap)."""
    if medians.empty:
        return
    df = medians.copy()
    df["median_gap"] = df["median_true"] - df["median_false"]
    df = df.sort_values(["metric", "median_gap"], ascending=[True, False])
    df = _ensure_metric_order(df)
    out_csv = outdir / "median_first_table.csv"
    cols = [
        "metric", "source_file", "median_overall", "median_true", "median_false", "median_gap", "n_true", "n_false",
    ]
    df[cols].to_csv(out_csv, index=False)

    # Minimal LaTeX export
    try:
        latex = df[cols].to_latex(index=False, float_format="%.3f")
        (outdir / "median_first_table.tex").write_text(latex, encoding="utf-8")
    except Exception:
        pass


def export_best_j_per_metric(thresholds: pd.DataFrame, outdir: Path = PLOTS_DIR):
    """For each metric, keep the best-J row per source_file, output as CSV + LaTeX, and a heatmap of J."""
    if thresholds.empty:
        return
    df = thresholds.copy()
    df = df.dropna(subset=["difference"])  # J = TPR - FPR

    # For each (metric, source_file) keep the row with max J; robust single-index selection
    # Build a boolean mask equal to True only at the idxmax row per group
    try:
        idx = df.groupby(["metric", "source_file"])['difference'].idxmax()
        # idx is a Series indexed by group, values are row indices
        df_best = df.loc[idx.dropna().astype(int)].reset_index(drop=True)
    except Exception:
        # Fallback: sort and drop_duplicates to emulate idxmax per group
        df_best = (
            df.sort_values(["metric", "source_file", "difference"], ascending=[True, True, False])
              .drop_duplicates(subset=["metric", "source_file"], keep="first")
              .reset_index(drop=True)
        )
    df_best = df_best.sort_values(["metric", "difference"], ascending=[True, False])
    df_best = _ensure_metric_order(df_best)

    cols = [
        "metric", "source_file", "best_threshold", "true_pass_rate", "false_pass_rate", "difference",
        "median_false", "median_true", "median_overall", "auc", "doc_retrieved_pct",
    ]
    out_csv = outdir / "best_j_per_metric_per_file.csv"
    df_best[cols].to_csv(out_csv, index=False)
    try:
        latex = df_best[cols].to_latex(index=False, float_format="%.3f")
        (outdir / "best_j_per_metric_per_file.tex").write_text(latex, encoding="utf-8")
    except Exception:
        pass

    # Heatmap of J (difference) by metric x source_file
    if df_best.empty:
        return
    pivot = df_best.pivot(index="metric", columns="source_file", values="difference")
    plt.figure(figsize=(max(8, 0.5 * pivot.shape[1] + 3), max(5, 0.5 * pivot.shape[0] + 2)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Youden's J (TPR−FPR)"})
    plt.title("Best J per metric and source (at optimal threshold)")
    plt.tight_layout()
    plt.savefig(outdir / "07_heatmap_best_j_per_metric_source.png", dpi=200)
    plt.close()

    # Scatter: J vs median_false (lower median_false at high J is desirable)
    plt.figure(figsize=(10, 6))
    highlight = df_best["source_file"].astype(str).str.contains("Naive1AI", case=False, na=False)
    sns.scatterplot(data=df_best, x="median_false", y="difference", hue="metric", style="source_file", alpha=0.9)
    # Emphasize Naive1AI points
    if highlight.any():
        sns.scatterplot(data=df_best[highlight], x="median_false", y="difference", color="red", s=120, marker="X", label="weaviateNaive1AI (highlight)")
    plt.xlabel("Median False (lower better)")
    plt.ylabel("Youden's J (higher better)")
    plt.title("Trade-off: J vs median_false across metrics and sources")
    plt.tight_layout()
    plt.savefig(outdir / "08_scatter_j_vs_median_false.png", dpi=200)
    plt.close()


def plot_per_metric_panels(thresholds: pd.DataFrame, outdir: Path = PLOTS_DIR):
    """Panels per metric showing TPR, FPR, and J by source_file at best thresholds."""
    if thresholds.empty:
        return
    df = thresholds.copy()
    try:
        idx = df.groupby(["metric", "source_file"])['difference'].idxmax()
        df = df.loc[idx.dropna().astype(int)].reset_index(drop=True)
    except Exception:
        df = (
            df.sort_values(["metric", "source_file", "difference"], ascending=[True, True, False])
              .drop_duplicates(subset=["metric", "source_file"], keep="first")
              .reset_index(drop=True)
        )

    metrics = list(df["metric"].dropna().unique())
    metrics = [m for m in ["token_recall", "rouge1_f", "jaccard", "overlap", "bleu", "bert_cos"] if m in metrics] or list(metrics)

    for m in metrics:
        sub = df[df["metric"] == m].copy()
        if sub.empty:
            continue
        # ensure numeric dtypes for plotting/annotation
        for col in ["true_pass_rate", "false_pass_rate", "difference", "best_threshold"]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.sort_values("difference", ascending=False)
        fig, axes = plt.subplots(1, 3, figsize=(16, max(4, 0.4 * len(sub))), sharey=True)

        sns.barplot(data=sub, y="source_file", x="true_pass_rate", ax=axes[0], color="#2ca02c")
        axes[0].set_title(f"TPR at best thr ({m})")
        axes[0].set_xlabel("TPR")
        axes[0].set_ylabel("")

        sns.barplot(data=sub, y="source_file", x="false_pass_rate", ax=axes[1], color="#d62728")
        axes[1].set_title(f"FPR at best thr ({m})")
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("")

        sns.barplot(data=sub, y="source_file", x="difference", ax=axes[2], color="#1f77b4")
        axes[2].set_title(f"J = TPR−FPR ({m})")
        axes[2].set_xlabel("Youden's J")
        axes[2].set_ylabel("")

        for ax in axes:
            ax.set_xlim(0, 1)
            for i, row in enumerate(sub.itertuples(index=False)):
                if ax is axes[2]:
                    # Use numpy .item() when available to ensure Python float
                    diff_raw = getattr(row, "difference", None)
                    thr_raw = getattr(row, "best_threshold", None)
                    if diff_raw is None:
                        continue
                    try:
                        diff_val = diff_raw.item() if hasattr(diff_raw, "item") else float(diff_raw)
                    except Exception:
                        continue
                    thr_val = None
                    if thr_raw is not None:
                        try:
                            thr_val = thr_raw.item() if hasattr(thr_raw, "item") else float(thr_raw)
                        except Exception:
                            thr_val = None
                    x_annot = min(0.98, float(diff_val) + 0.01)
                    label = f"thr={thr_val:.2f}" if isinstance(thr_val, float) else "thr=?"
                    ax.text(x_annot, i, label, va="center", fontsize=8)

        fig.suptitle(f"Per-source comparison at best threshold for metric: {m}")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(outdir / f"09_panels_tpr_fpr_j_{m}.png", dpi=200)
        plt.close()


def plot_weaviate_best_j(thresholds: pd.DataFrame, metric: str = "token_recall", outdir: Path = PLOTS_DIR):
    """Compare Weaviate variants only for a given metric using best-J rows.
    Produces a bar chart of J by weaviate* source_file and highlights Naive1AI.
    Also saves a CSV/LaTeX table for the filtered subset.
    """
    if thresholds.empty:
        return
    df = thresholds.copy().dropna(subset=["difference"])
    # best-J per (metric, source)
    try:
        idx = df.groupby(["metric", "source_file"])['difference'].idxmax()
        best = df.loc[idx.dropna().astype(int)].reset_index(drop=True)
    except Exception:
        best = (
            df.sort_values(["metric", "source_file", "difference"], ascending=[True, True, False])
              .drop_duplicates(subset=["metric", "source_file"], keep="first")
              .reset_index(drop=True)
        )
    best = best[best["metric"] == metric].copy()
    if best.empty:
        return
    weav = best[best["source_file"].astype(str).str.startswith("weaviate")].copy()
    if weav.empty:
        return
    weav = weav.sort_values("difference", ascending=False)

    # Export subset table
    cols = ["source_file", "best_threshold", "true_pass_rate", "false_pass_rate", "difference", "median_false", "auc"]
    (outdir / f"weaviate_best_j_{metric}.csv").write_text(weav[cols].to_csv(index=False), encoding="utf-8")
    try:
        (outdir / f"weaviate_best_j_{metric}.tex").write_text(weav[cols].to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    except Exception:
        pass

    plt.figure(figsize=(10, max(3.5, 0.4 * len(weav))))
    ax = sns.barplot(data=weav, y="source_file", x="difference", color="#1f77b4")
    ax.set_title(f"Weaviate variants — best J for {metric}")
    ax.set_xlabel("Youden's J (TPR−FPR)")
    ax.set_ylabel("")
    # Highlight Naive1AI row
    bars = ax.patches  # bars drawn by seaborn
    for i, row in enumerate(weav.itertuples(index=False)):
        label = getattr(row, "source_file", "")
        j = getattr(row, "difference", 0.0)
        thr = getattr(row, "best_threshold", None)
        try:
            jv = float(j)
        except Exception:
            jv = float("nan")
        txt = f"J={jv:.3f}"
        if thr is not None:
            try:
                thrv = float(thr)
                txt += f" | thr={thrv:.2f}"
            except Exception:
                pass
        ax.text(min(0.98, (jv if pd.notna(jv) else 0.0) + 0.01), i, txt, va="center", fontsize=8)
        if isinstance(label, str) and "Naive1AI" in label:
            # Recolor the corresponding bar (rect) safely
            try:
                bar = bars[i]
                bar.set_facecolor("#d62728")
            except Exception:
                pass
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(outdir / f"10_weaviate_best_j_{metric}.png", dpi=200)
    plt.close()


def plot_difference_bar(thresholds: pd.DataFrame, outdir: Path = PLOTS_DIR):
    if thresholds.empty:
        return
    df = thresholds.copy()
    # Order by difference descending
    df = df.sort_values(by=["difference"], ascending=False)

    plt.figure(figsize=(12, max(4, 0.28 * len(df))))
    ax = sns.barplot(data=df, x="difference", y="label", hue="metric", dodge=False, palette="viridis")
    ax.set_title("True Pass Rate - False Pass Rate (difference) by metric/source")
    ax.set_xlabel("TPR - FPR (higher is better)")
    ax.set_ylabel("")
    for i, v in enumerate(df["difference" ].values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, min(1.05, max(1.0, df["difference"].max() + 0.1)))
    plt.tight_layout()
    plt.savefig(outdir / "01_difference_by_metric_source.png", dpi=200)
    plt.close()


def plot_auc_bar(thresholds: pd.DataFrame, outdir: Path = PLOTS_DIR):
    if thresholds.empty or "auc" not in thresholds.columns:
        return
    df = thresholds.sort_values(by=["auc"], ascending=False)

    plt.figure(figsize=(12, max(4, 0.28 * len(df))))
    ax = sns.barplot(data=df, x="auc", y="label", hue="metric", dodge=False, palette="crest")
    ax.set_title("AUC by metric/source")
    ax.set_xlabel("AUC (ROC)")
    ax.set_ylabel("")
    for i, v in enumerate(df["auc"].values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, min(1.05, max(1.0, df["auc"].max() + 0.1)))
    plt.tight_layout()
    plt.savefig(outdir / "02_auc_by_metric_source.png", dpi=200)
    plt.close()


def plot_tpr_fpr_scatter(thresholds: pd.DataFrame, outdir: Path = PLOTS_DIR):
    if thresholds.empty:
        return
    df = thresholds.copy()

    g = sns.FacetGrid(df, col="metric", col_wrap=3, height=4, sharex=True, sharey=True)
    g.map_dataframe(
        sns.scatterplot,
        x="false_pass_rate",
        y="true_pass_rate",
        hue="source_file",
        style="source_file",
        s=80,
        alpha=0.9,
    )
    g.set_axis_labels("False Pass Rate (FPR)", "True Pass Rate (TPR)")
    g.add_legend(title="Source file")

    # Add diagonal line and annotations per metric axis using axes_dict to ensure correct mapping
    for metric, ax in g.axes_dict.items():
        ax.plot([0, 1], [0, 1], ls="--", c="gray", linewidth=1)
        subdf = df[df["metric"] == metric]
        for _, row in subdf.iterrows():
            x, y = row["false_pass_rate"], row["true_pass_rate"]
            if pd.notna(x) and pd.notna(y):
                thr = row.get("best_threshold", None)
                label_bits = [f"{row['source_file']}", f"FPR={x:.2f}", f"TPR={y:.2f}"]
                if pd.notna(thr):
                    label_bits.append(f"thr={float(thr):.2f}")
                label = " \n".join(label_bits)
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"TPR vs FPR | {metric}")

    plt.tight_layout()
    g.savefig(outdir / "03_tpr_vs_fpr_scatter_by_metric.png", dpi=200)
    plt.close()


def plot_median_overall_bar(medians: pd.DataFrame, outdir: Path = PLOTS_DIR):
    if medians.empty:
        return
    df = medians.copy().sort_values(by=["median_overall"], ascending=False)

    plt.figure(figsize=(12, max(4, 0.28 * len(df))))
    ax = sns.barplot(data=df, x="median_overall", y="label", hue="metric", dodge=False, palette="mako")
    ax.set_title("Median overall score by metric/source")
    ax.set_xlabel("Median overall")
    ax.set_ylabel("")
    for i, v in enumerate(df["median_overall" ].values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, min(1.05, max(1.0, df["median_overall"].max() + 0.1)))
    plt.tight_layout()
    plt.savefig(outdir / "04_median_overall_by_metric_source.png", dpi=200)
    plt.close()


def plot_doc_retrieved_pct(thresholds: pd.DataFrame, outdir: Path = PLOTS_DIR):
    if thresholds.empty or "doc_retrieved_pct" not in thresholds.columns:
        return
    # Aggregate by source_file (doc_retrieved_pct is per file; drop metric from labels)
    cols = ["source_file", "doc_retrieved_pct"]
    df = (
        thresholds[cols]
        .dropna(subset=["doc_retrieved_pct"])  # safety
        .drop_duplicates(subset=["source_file"])  # many rows share the same file value
        .sort_values(by=["doc_retrieved_pct"], ascending=False)
    )

    plt.figure(figsize=(10, max(4, 0.28 * len(df))))
    ax = sns.barplot(data=df, x="doc_retrieved_pct", y="source_file", color=sns.color_palette("flare", 1)[0])
    ax.set_title("Documents retrieved % by source file")
    ax.set_xlabel("Doc Retrieved %")
    ax.set_ylabel("")
    for i, v in enumerate(df["doc_retrieved_pct" ].values):
        ax.text(min(1.0, v + 0.01), i, f"{v:.2%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(outdir / "05_doc_retrieved_pct_by_source_file.png", dpi=200)
    plt.close()


def plot_confidence_intervals(thresholds: pd.DataFrame, outdir: Path = PLOTS_DIR):
    # Errorbar chart of TPR and FPR with CI ranges per label
    if thresholds.empty:
        return
    df = thresholds.copy()
    df = df.sort_values("difference", ascending=False)

    # TPR CIs
    plt.figure(figsize=(12, max(4, 0.28 * len(df))))
    ax = plt.gca()
    y_pos = range(len(df))
    ax.errorbar(df["tpr_ci_low"], y_pos, xerr=(df["tpr_ci_high"] - df["tpr_ci_low"]) / 2, fmt="o", color="tab:green", ecolor="lightgreen", label="TPR CI", alpha=0.8)
    ax.errorbar(df["fpr_ci_low"], y_pos, xerr=(df["fpr_ci_high"] - df["fpr_ci_low"]) / 2, fmt="o", color="tab:red", ecolor="#f5b7b1", label="FPR CI", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Rate with 95% CI (approx.)")
    ax.set_title("TPR/FPR confidence intervals by metric/source")
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(outdir / "06_tpr_fpr_confidence_intervals.png", dpi=200)
    plt.close()


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Could not find {SUMMARY_PATH}")

    summary = load_summary(SUMMARY_PATH)
    thresholds, medians = to_dataframe(summary)

    # Save flat CSVs for quick inspection
    save_csvs(thresholds, medians)

    # Generate plots
    plot_difference_bar(thresholds)
    plot_auc_bar(thresholds)
    plot_tpr_fpr_scatter(thresholds)
    plot_median_overall_bar(medians)
    plot_doc_retrieved_pct(thresholds)
    plot_confidence_intervals(thresholds)
    # New Option B assets
    export_median_first_table(medians)
    export_best_j_per_metric(thresholds)
    plot_per_metric_panels(thresholds)
    # Focused: compare Weaviate variants and highlight Naive1AI
    plot_weaviate_best_j(thresholds, metric="token_recall")

    # Also produce a filtered "hero metrics" pack for the thesis main text
    if not thresholds.empty:
        th_hero = thresholds[thresholds["metric"].isin(HERO_METRICS)].copy()
    else:
        th_hero = thresholds
    if not medians.empty:
        med_hero = medians[medians["metric"].isin(HERO_METRICS)].copy()
    else:
        med_hero = medians

    # Save flat CSVs for hero subset
    save_csvs(th_hero, med_hero, outdir=HERO_DIR)
    # Plots/tables for hero subset
    plot_difference_bar(th_hero, outdir=HERO_DIR)
    plot_auc_bar(th_hero, outdir=HERO_DIR)
    plot_tpr_fpr_scatter(th_hero, outdir=HERO_DIR)
    plot_median_overall_bar(med_hero, outdir=HERO_DIR)
    plot_doc_retrieved_pct(th_hero, outdir=HERO_DIR)
    plot_confidence_intervals(th_hero, outdir=HERO_DIR)
    export_median_first_table(med_hero, outdir=HERO_DIR)
    export_best_j_per_metric(th_hero, outdir=HERO_DIR)
    plot_per_metric_panels(th_hero, outdir=HERO_DIR)
    plot_weaviate_best_j(th_hero, metric="token_recall", outdir=HERO_DIR)

    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
