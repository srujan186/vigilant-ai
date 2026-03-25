"""
SmartGuard - Keyword Filter vs ML Model Benchmark
==================================================
This module implements a pure keyword baseline classifier and benchmarks
it head-to-head against the ML model (unitary/toxic-bert).

Research question: Does a lightweight CPU-friendly classifier do
meaningfully better than a simple keyword blocklist?

This answers Research Question #1 from the brief with side-by-side
evidence and latency measurements.
"""

import time
import re
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# BASELINE: Pure Keyword Filter
# This is what most naive guardrails implementations use.
# We implement it rigorously so the comparison is fair.
# ─────────────────────────────────────────────────────────────

KEYWORD_BLOCKLIST = {
    # Jailbreak keywords
    "jailbreak", "dan", "do anything now", "ignore previous instructions",
    "ignore all instructions", "forget your instructions", "no restrictions",
    "unrestricted", "override", "bypass", "pretend you are", "act as if",
    "you are now", "new persona", "developer mode", "jailbroken",

    # Toxic / harmful keywords
    "bomb", "explosive", "detonate", "synthesize", "methamphetamine",
    "fentanyl", "poison someone", "make drugs", "kill yourself",
    "self harm", "self-harm", "shoot up", "mass shooting", "hate speech",
    "white supremacy", "ethnic cleansing", "suicide method",

    # Prompt injection keywords
    "system prompt", "ignore the above", "new instruction",
    "disregard previous", "forget previous", "your new instructions",
    "[inst]", "###system",
}

# Regex patterns for keyword filter (slightly smarter than exact match)
KEYWORD_PATTERNS = [
    r"\bdan\b",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"how\s+to\s+(make|build|create|synthesize)\s+\w*(bomb|drug|weapon|explosive|poison)",
    r"(kill|shoot|murder)\s+(yourself|myself|people|everyone)",
    r"\b(suicide|self.harm)\b",
    r"(forget|disregard|override)\s+(your\s+)?(instructions|rules|guidelines|system)",
]


@dataclass
class ClassifierResult:
    verdict: str        # "safe" or "unsafe"
    category: str
    confidence: float
    latency_ms: float


class KeywordFilter:
    """
    Baseline: pure keyword + regex blocklist.
    Represents the naive guardrail approach.
    No ML, no semantic understanding.
    """
    name = "Keyword Filter (Baseline)"

    def classify(self, text: str) -> ClassifierResult:
        start = time.perf_counter()
        text_lower = text.lower()

        # Check exact keywords
        for kw in KEYWORD_BLOCKLIST:
            if kw in text_lower:
                latency = round((time.perf_counter() - start) * 1000, 3)
                return ClassifierResult("unsafe", "keyword_match", 1.0, latency)

        # Check patterns
        for pattern in KEYWORD_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                latency = round((time.perf_counter() - start) * 1000, 3)
                return ClassifierResult("unsafe", "pattern_match", 1.0, latency)

        latency = round((time.perf_counter() - start) * 1000, 3)
        return ClassifierResult("safe", "safe", 0.0, latency)


# ─────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkEntry:
    prompt: str
    ground_truth: str
    keyword_verdict: str
    keyword_correct: bool
    keyword_latency_ms: float
    ml_verdict: str
    ml_correct: bool
    ml_latency_ms: float
    ml_confidence: float
    category: str
    notes: str = ""


def run_benchmark(guard=None) -> dict:
    """
    Run head-to-head benchmark: Keyword Filter vs SmartGuard ML.
    Returns full results dict with side-by-side comparison.
    """
    # Load test suite
    suite_path = Path(__file__).parent / "test_suite.json"
    raw = suite_path.read_text(encoding="utf-8")
    raw = re.sub(r"//.*", "", raw)
    suite = json.loads(raw)

    # Load ML classifier lazily
    if guard is None:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from guard import SmartGuard
        print("⏳ Loading ML classifier for benchmark...")
        guard = SmartGuard(threshold=0.5)

    keyword_filter = KeywordFilter()

    results: List[BenchmarkEntry] = []

    print(f"\n{'='*72}")
    print(f"  SmartGuard Benchmark: Keyword Filter vs ML Model")
    print(f"  {'Prompt':<45} {'KW':>4} {'ML':>4} {'GT':>5}")
    print(f"{'='*72}")

    for item in suite:
        text = item["prompt"]
        gt = item["label"]

        kw_result = keyword_filter.classify(text)
        ml_result = guard.classify(text)

        kw_correct = kw_result.verdict == gt
        ml_correct = ml_result.verdict == gt

        kw_icon = "✅" if kw_correct else "❌"
        ml_icon = "✅" if ml_correct else "❌"

        print(f"  [{item['id']}] {text[:44]:<44} KW:{kw_icon} ML:{ml_icon} GT:{gt[:6]}")

        results.append(BenchmarkEntry(
            prompt=text,
            ground_truth=gt,
            keyword_verdict=kw_result.verdict,
            keyword_correct=kw_correct,
            keyword_latency_ms=kw_result.latency_ms,
            ml_verdict=ml_result.verdict,
            ml_correct=ml_correct,
            ml_latency_ms=ml_result.latency_ms,
            ml_confidence=ml_result.confidence,
            category=item["category"],
        ))

    # ── Metrics ──────────────────────────────────────────────
    total = len(results)
    kw_correct_count = sum(1 for r in results if r.keyword_correct)
    ml_correct_count = sum(1 for r in results if r.ml_correct)

    # Where keyword FAILS but ML CATCHES (the key research finding)
    kw_miss_ml_catch = [r for r in results
                         if not r.keyword_correct and r.ml_correct
                         and r.ground_truth == "unsafe"]

    # Where ML fails but keyword catches
    ml_miss_kw_catch = [r for r in results
                         if r.keyword_correct and not r.ml_correct
                         and r.ground_truth == "unsafe"]

    # False positives for each
    kw_fp = [r for r in results if r.keyword_verdict == "unsafe" and r.ground_truth == "safe"]
    ml_fp = [r for r in results if r.ml_verdict == "unsafe" and r.ground_truth == "safe"]

    # Latency stats
    kw_latencies = [r.keyword_latency_ms for r in results]
    ml_latencies = [r.ml_latency_ms for r in results]

    def p95(lst): return sorted(lst)[int(0.95 * len(lst))]

    summary = {
        "total_prompts": total,
        "keyword_filter": {
            "accuracy_pct": round(kw_correct_count / total * 100, 1),
            "false_positives": len(kw_fp),
            "false_positive_rate_pct": round(len(kw_fp) / 15 * 100, 1),  # 15 benign
            "avg_latency_ms": round(statistics.mean(kw_latencies), 3),
            "p95_latency_ms": round(p95(kw_latencies), 3),
        },
        "ml_model": {
            "accuracy_pct": round(ml_correct_count / total * 100, 1),
            "false_positives": len(ml_fp),
            "false_positive_rate_pct": round(len(ml_fp) / 15 * 100, 1),
            "avg_latency_ms": round(statistics.mean(ml_latencies), 2),
            "p95_latency_ms": round(p95(ml_latencies), 2),
        },
        "key_findings": {
            "cases_kw_misses_ml_catches": len(kw_miss_ml_catch),
            "cases_ml_misses_kw_catches": len(ml_miss_kw_catch),
            "ml_accuracy_gain_pct": round(
                (ml_correct_count - kw_correct_count) / total * 100, 1
            ),
            "latency_overhead_ms": round(
                statistics.mean(ml_latencies) - statistics.mean(kw_latencies), 2
            ),
        },
        "kw_miss_ml_catch_examples": [
            {"id": r.category, "prompt_snippet": r.prompt[:80]}
            for r in kw_miss_ml_catch[:5]
        ],
    }

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Metric':<35} {'Keyword':>10} {'ML Model':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy':<35} {summary['keyword_filter']['accuracy_pct']:>9}% {summary['ml_model']['accuracy_pct']:>9}%")
    print(f"  {'False Positive Rate':<35} {summary['keyword_filter']['false_positive_rate_pct']:>9}% {summary['ml_model']['false_positive_rate_pct']:>9}%")
    print(f"  {'Avg Latency':<35} {summary['keyword_filter']['avg_latency_ms']:>8}ms {summary['ml_model']['avg_latency_ms']:>8}ms")
    print(f"  {'P95 Latency':<35} {summary['keyword_filter']['p95_latency_ms']:>8}ms {summary['ml_model']['p95_latency_ms']:>8}ms")
    print(f"\n  ML catches {summary['key_findings']['cases_kw_misses_ml_catches']} attacks that keyword filter MISSES")
    print(f"  ML accuracy gain: +{summary['key_findings']['ml_accuracy_gain_pct']}%")
    print(f"  Latency overhead: +{summary['key_findings']['latency_overhead_ms']}ms")

    print(f"\n  ── Cases where keyword fails but ML catches ──")
    for ex in kw_miss_ml_catch[:5]:
        print(f"  • [{ex.category}] {ex.prompt[:65]}...")

    # Save results
    out_path = Path(__file__).parent / "results" / "benchmark_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "per_prompt": [
            {
                "prompt": r.prompt[:100],
                "category": r.category,
                "ground_truth": r.ground_truth,
                "keyword_verdict": r.keyword_verdict,
                "keyword_correct": r.keyword_correct,
                "keyword_latency_ms": r.keyword_latency_ms,
                "ml_verdict": r.ml_verdict,
                "ml_correct": r.ml_correct,
                "ml_latency_ms": r.ml_latency_ms,
                "ml_confidence": r.ml_confidence,
            }
            for r in results
        ]
    }, indent=2))
    print(f"\n  Saved → {out_path}")

    return summary


if __name__ == "__main__":
    run_benchmark()
