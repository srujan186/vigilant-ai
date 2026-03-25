"""
SmartGuard - Master Runner
===========================
Run everything in the right order for your submission.

Usage:
    python run_all.py             # Full pipeline
    python run_all.py --quick     # Skip latency benchmark (faster)
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def print_banner(title: str):
    print(f"\n{'█'*65}")
    print(f"  {title}")
    print(f"{'█'*65}")


def main(quick=False):
    print_banner("SmartGuard — Full Evaluation Pipeline")

    # ── Step 1: Load classifier once ──────────────────────────
    print_banner("Step 1/4 — Loading SmartGuard Classifier")
    from guard import SmartGuard
    guard = SmartGuard(threshold=0.5)

    # ── Step 2: Red-team evaluation ────────────────────────────
    print_banner("Step 2/4 — Red-Team Evaluation (45 prompts)")
    from runner import run_redteam, sweep_thresholds
    run_redteam(threshold=0.5)
    print("\n  Sweeping thresholds for accuracy vs strictness curve...")
    sweep_thresholds()

    # ── Step 3: Keyword vs ML benchmark ───────────────────────
    print_banner("Step 3/4 — Keyword Filter vs ML Benchmark")
    from benchmark import run_benchmark
    run_benchmark(guard=guard)

    # ── Step 4: Latency benchmark ──────────────────────────────
    if not quick:
        print_banner("Step 4/4 — Latency Benchmark (100 runs per prompt)")
        print("  ⏳ This takes ~3-5 minutes. Use --quick to skip.")
        from latency_benchmark import run_latency_benchmark
        run_latency_benchmark(guard=guard)
    else:
        print_banner("Step 4/4 — Latency Benchmark (SKIPPED with --quick)")

    # ── Step 5: Failure analysis ───────────────────────────────
    print_banner("Step 5/5 — Failure Analysis")
    from failure_analysis import analyze_failures, print_evasion_taxonomy
    print_evasion_taxonomy()
    analyze_failures()

    # ── Done ───────────────────────────────────────────────────
    print_banner("ALL DONE — Launch Dashboard")
    print("""
  All results saved to results/ directory:
    ✅ results/redteam_results.json
    ✅ results/threshold_sweep.json
    ✅ results/benchmark_results.json
    ✅ results/latency_benchmark.json    (if not --quick)
    ✅ results/failure_analysis.json

  Now launch the dashboard:
    streamlit run app.py

  Or start the API server:
    uvicorn main:app --reload --port 8000
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Skip latency benchmark")
    args = parser.parse_args()
    main(quick=args.quick)
