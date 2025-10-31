"""
Benchmark Weaviate sample data configurations incrementally to observe scaling.

This script imports `benchmark_sample_configs` from `weaviate_manager.py`
and runs a series of configurations that grow with the number of levels.

Supported growth shapes:
- linear (default): increases selected parameters proportionally to the level index
- exponential: increases selected parameters by a base^level factor

It prints per-config timings and writes a CSV summary if requested.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure repository root is on sys.path so we can import weaviate_manager
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib


def build_linear_configs(
    levels: int,
    *,
    num_entidades: int = 2,
    num_pastas_per_entidade: int = 5,
    num_ficheiros_per_pasta: int = 8,
    num_metadados_per_ficheiro: int = 1,
    num_fluxos: int = 5,
    num_etapas_per_fluxo: int = 10,
) -> List[Dict[str, int]]:
    """
    Create a list of configs that scale linearly with the level index.

    Keeps `num_entidades` and `num_etapas_per_fluxo` constant by default,
    while increasing pastas/entidade, ficheiros/pasta and fluxos.
    """
    configs: List[Dict[str, int]] = []
    for i in range(1, levels + 1):
        factor = i  # simple linear growth
        cfg = {
            "num_entidades": num_entidades,
            "num_pastas_per_entidade": num_pastas_per_entidade * factor,
            "num_ficheiros_per_pasta": num_ficheiros_per_pasta * factor,
            "num_metadados_per_ficheiro": num_metadados_per_ficheiro,
            "num_fluxos": num_fluxos * factor,
            "num_etapas_per_fluxo": num_etapas_per_fluxo,
        }
        configs.append(cfg)
    return configs


def build_exponential_configs(
    levels: int,
    *,
    num_entidades: int = 2,
    num_pastas_per_entidade: int = 5,
    num_ficheiros_per_pasta: int = 8,
    num_metadados_per_ficheiro: int = 1,
    num_fluxos: int = 5,
    num_etapas_per_fluxo: int = 10,
    growth: float = 2.0,
) -> List[Dict[str, int]]:
    """
    Create a list of configs that scale exponentially with the level index.

    Keeps `num_entidades` and `num_etapas_per_fluxo` constant by default,
    while increasing pastas/entidade, ficheiros/pasta and fluxos by
    factor = growth ** (i-1) for level i in [1..levels].
    """
    if growth <= 0:
        raise ValueError("growth must be > 0 for exponential scaling")

    configs: List[Dict[str, int]] = []
    for i in range(1, levels + 1):
        factor = growth ** (i - 1)
        # Ensure we keep integers and never go below 1
        def scale(x: int) -> int:
            v = int(round(x * factor))
            return max(1, v)

        cfg = {
            "num_entidades": num_entidades,
            "num_pastas_per_entidade": scale(num_pastas_per_entidade),
            "num_ficheiros_per_pasta": scale(num_ficheiros_per_pasta),
            "num_metadados_per_ficheiro": num_metadados_per_ficheiro,
            "num_fluxos": scale(num_fluxos),
            "num_etapas_per_fluxo": num_etapas_per_fluxo,
        }
        configs.append(cfg)
    return configs


def write_csv(results: List[Dict[str, Any]], out_path: Path) -> None:
    """Write aggregated benchmark results to CSV."""
    fieldnames = [
        "num_entidades",
        "num_pastas_per_entidade",
        "num_ficheiros_per_pasta",
        "num_metadados_per_ficheiro",
        "num_fluxos",
        "num_etapas_per_fluxo",
        "setup_collections",
        "insert_sample_data",
        "query_fluxo_etapas",
        "query_entidade_hierarchy",
        "global_semantic_search",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            cfg = item.get("config", {})
            t = item.get("timings", {})
            writer.writerow({
                "num_entidades": cfg.get("num_entidades"),
                "num_pastas_per_entidade": cfg.get("num_pastas_per_entidade"),
                "num_ficheiros_per_pasta": cfg.get("num_ficheiros_per_pasta"),
                "num_metadados_per_ficheiro": cfg.get("num_metadados_per_ficheiro"),
                "num_fluxos": cfg.get("num_fluxos"),
                "num_etapas_per_fluxo": cfg.get("num_etapas_per_fluxo"),
                "setup_collections": f"{t.get('setup_collections', 0.0):.6f}",
                "insert_sample_data": f"{t.get('insert_sample_data', 0.0):.6f}",
                "query_fluxo_etapas": f"{t.get('query_fluxo_etapas', 0.0):.6f}",
                "query_entidade_hierarchy": f"{t.get('query_entidade_hierarchy', 0.0):.6f}",
                "global_semantic_search": f"{t.get('global_semantic_search', 0.0):.6f}",
            })
    print(f"\nCSV written to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark incremental Weaviate sample data configs")
    p.add_argument("--levels", type=int, default=200, help="Number of incremental levels to run")
    p.add_argument("--entidades", type=int, default=2, help="Base number of entidades")
    p.add_argument("--pastas", type=int, default=8, help="Base pastas per entidade (scaled)")
    p.add_argument("--ficheiros", type=int, default=12, help="Base ficheiros per pasta (scaled)")
    p.add_argument("--metadados", type=int, default=1, help="Metadados per ficheiro (constant)")
    p.add_argument("--fluxos", type=int, default=8, help="Base fluxos (scaled)")
    p.add_argument("--etapas", type=int, default=10, help="Etapas per fluxo (constant)")
    p.add_argument(
        "--shape",
        choices=["linear", "exponential"],
        default="linear",
        help="Scaling shape to generate configs",
    )
    p.add_argument(
        "--growth",
        type=float,
        default=5.0,
        help="Exponential base when --shape=exponential (ignored otherwise)",
    )
    p.add_argument("--limit-fluxo-etapas", type=int, default=100, help="Limit for fluxo-etapas query")
    p.add_argument("--limit-entidade-hierarchy", type=int, default=100, help="Limit for entidade hierarchy query")
    p.add_argument("--limit-per-collection", type=int, default=100, help="Limit per collection for global search")
    p.add_argument("--no-clean-start-each", action="store_true",default=True, help="Reuse schema between levels instead of recreating")
    p.add_argument("--no-local", action="store_true", help="Do not connect to local Weaviate (use custom connection in manager)")
    p.add_argument("--csv", type=str, default=str(REPO_ROOT / "tests" / "benchmark_results.csv"), help="CSV output path")
    p.add_argument("--manager", choices=["normal", "optimized"], default="optimized", help="Choose which manager implementation to use")
    p.add_argument("--workers", type=int, default=None, help="Max concurrent workers (only for optimized manager)")
    return p.parse_args()




def main() -> None:
    args = parse_args()

    if args.shape == "exponential":
        configs = build_exponential_configs(
            args.levels,
            num_entidades=args.entidades,
            num_pastas_per_entidade=args.pastas,
            num_ficheiros_per_pasta=args.ficheiros,
            num_metadados_per_ficheiro=args.metadados,
            num_fluxos=args.fluxos,
            num_etapas_per_fluxo=args.etapas,
            growth=args.growth,
        )
    else:
        configs = build_linear_configs(
            args.levels,
            num_entidades=args.entidades,
            num_pastas_per_entidade=args.pastas,
            num_ficheiros_per_pasta=args.ficheiros,
            num_metadados_per_ficheiro=args.metadados,
            num_fluxos=args.fluxos,
            num_etapas_per_fluxo=args.etapas,
        )

    banner = "Planned configurations (linear growth)" if args.shape == "linear" else f"Planned configurations (exponential base={args.growth:g})"
    print(f"\n{banner}:")
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[{i}] ent={cfg['num_entidades']}, "
            f"pastas/ent={cfg['num_pastas_per_entidade']}, "
            f"ficheiros/pasta={cfg['num_ficheiros_per_pasta']}, "
            f"metadados/ficheiro={cfg['num_metadados_per_ficheiro']}, "
            f"fluxos={cfg['num_fluxos']}, etapas/fluxo={cfg['num_etapas_per_fluxo']}"
        )

    # Select manager implementation
    run_bench = None
    if args.manager == "optimized":
        try:
            mod = importlib.import_module("weaviate_manager_paralyzed")
            run_bench = getattr(mod, "benchmark_sample_configs")
        except Exception as e:
            print("Warning: failed to load optimized manager:", e)
            from weaviate_manager import benchmark_sample_configs as run_bench  # type: ignore
    else:
        from weaviate_manager import benchmark_sample_configs as run_bench  # type: ignore

    bench_kwargs = {}
    if args.manager == "optimized" and args.workers:
        bench_kwargs["max_workers"] = args.workers

    results = run_bench(
        configs,
        limit_fluxo_etapas=args.limit_fluxo_etapas,
        limit_entidade_hierarchy=args.limit_entidade_hierarchy,
        global_query="Find documents about contract approvals",
        limit_per_collection=args.limit_per_collection,
        clean_start_each=not args.no_clean_start_each,
        connect_to_local=not args.no_local,
        # Stream CSV writes per level when using optimized manager
        csv_path=(Path(args.csv).resolve() if args.manager == "optimized" else None),
        **bench_kwargs,
    )

    # For optimized manager, rows were already appended after each level
    if args.manager != "optimized":
        out_csv = Path(args.csv).resolve()
        write_csv(results, out_csv)


if __name__ == "__main__":
    main()
