"""
eval/benchmark.py

Latency and Throughput Benchmarking
=====================================
Runs 50 sample queries with per-stage timing collection.
Reports p50 and p95 for each pipeline stage.

Separate from run_ragas.py because:
  - Benchmarking needs controlled, repeatable conditions
  - Uses a fixed query set designed to stress different pipeline paths
  - Measures throughput (requests/second) in addition to latency
  - Can run with --concurrent flag for concurrency testing

Usage:
  python eval/benchmark.py
  python eval/benchmark.py --queries 50 --concurrent 1
  python eval/benchmark.py --queries 20 --output eval/benchmark_results.json

Author: Enterprise RAG Assistant
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Benchmark Query Set ────────────────────────────────────────────────────
# 50 queries designed to cover:
#   - All 6 pillars (varied coverage)
#   - Different query lengths (short factual vs long analytical)
#   - Different retrieval difficulty levels
#   - Cross-pillar queries (hardest retrieval case)

BENCHMARK_QUERIES = [
    # Reliability pillar (8 queries)
    "What is the AWS recommendation for designing against single points of failure?",
    "How should I implement health checks for services?",
    "What monitoring does AWS recommend for workload reliability?",
    "How does AWS define high availability?",
    "What is chaos engineering and how does the Reliability pillar address it?",
    "How should database backups be configured for reliability?",
    "What are the recommended retry and backoff strategies?",
    "How does the Reliability pillar address cross-region resilience?",
    # Security pillar (8 queries)
    "How should I implement IAM policies for least privilege?",
    "What encryption standards does AWS recommend for data at rest?",
    "How should I handle security incidents according to AWS?",
    "What is AWS recommendation for network segmentation?",
    "How should I audit and monitor AWS API calls?",
    "What are the AWS recommendations for protecting root account?",
    "How should I implement multi-factor authentication?",
    "What does the Security pillar say about vulnerability management?",
    # Operational Excellence (7 queries)
    "How should operational runbooks be structured?",
    "What metrics should I track for operational health?",
    "How does AWS recommend handling deployment failures?",
    "What is the recommended approach for configuration management?",
    "How should I implement change management processes?",
    "What are the best practices for incident post-mortems?",
    "How should observability be implemented in AWS workloads?",
    # Performance Efficiency (7 queries)
    "How should I select the right EC2 instance type for my workload?",
    "What are the AWS recommendations for caching strategies?",
    "How does AWS recommend optimizing database query performance?",
    "What is the approach for auto-scaling in AWS?",
    "How should I benchmark and measure application performance?",
    "What are the trade-offs between different storage types in AWS?",
    "How should I optimize network performance for AWS workloads?",
    # Cost Optimization (7 queries)
    "How should I implement cost allocation and tagging?",
    "What are the AWS recommendations for Reserved Instance planning?",
    "How should I right-size EC2 instances?",
    "What is the AWS recommendation for using Spot Instances?",
    "How should I optimize data transfer costs?",
    "What cost governance practices does AWS recommend?",
    "How should I set up cost anomaly detection?",
    # Sustainability (5 queries)
    "How does AWS recommend reducing carbon footprint of cloud workloads?",
    "What are sustainability best practices for compute resources?",
    "How should I optimize for energy efficiency in AWS?",
    "What does the Sustainability pillar say about right-sizing for sustainability?",
    "How does using managed services improve sustainability?",
    # Cross-pillar queries (8 queries — hardest retrieval)
    "How do reliability and cost optimization trade off in AWS architecture?",
    "What is the relationship between security and operational excellence?",
    "How do performance efficiency and sustainability goals interact?",
    "How should I balance reliability and cost when choosing availability zones?",
    "What are the common trade-offs between security controls and performance?",
    "How does operational excellence support reliability goals?",
    "How do cost optimization and performance efficiency interact for databases?",
    "What sustainability practices also improve cost optimization?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RAG API latency")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--output", default="eval/benchmark_results.json")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup requests before measuring")
    return parser.parse_args()


def make_single_request(
    query: str,
    api_url: str,
    request_num: int,
) -> dict:
    """
    Make a single /generate request and return detailed timing.

    Returns timing at every instrumented level:
      - wall_clock_ms: total time including network
      - pipeline fields from the response (retrieval_ms, generation_ms, etc.)
    """
    import httpx

    start = time.perf_counter()
    try:
        response = httpx.post(
            f"{api_url}/generate",
            json={
                "query": query,
                "use_fine_tuned": False,
                "top_k": 5,
            },
            timeout=60.0,
        )
        wall_ms = round((time.perf_counter() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            return {
                "request_num": request_num,
                "status": "success",
                "http_status": 200,
                "wall_clock_ms": wall_ms,
                "retrieval_latency_ms": data.get("retrieval_latency_ms", 0),
                "generation_latency_ms": data.get("generation_latency_ms", 0),
                "total_pipeline_ms": data.get("total_latency_ms", 0),
                "cache_hit": data.get("cache_hit", False),
                "is_refusal": data.get("is_refusal", False),
                "confidence": data.get("confidence", ""),
                "tokens_used": data.get("tokens_used", 0),
                "sources_count": len(data.get("sources", [])),
                "query_length": len(query),
            }
        else:
            return {
                "request_num": request_num,
                "status": "error",
                "http_status": response.status_code,
                "wall_clock_ms": wall_ms,
                "error": response.text[:200],
            }
    except Exception as e:
        return {
            "request_num": request_num,
            "status": "error",
            "error": str(e),
            "wall_clock_ms": round((time.perf_counter() - start) * 1000),
        }


def run_benchmark(
    api_url: str,
    query_count: int,
    concurrent: int,
    warmup_count: int,
) -> dict:
    """
    Run the full benchmark suite.

    Args:
        api_url: API base URL.
        query_count: Number of queries to benchmark.
        concurrent: Number of concurrent requests.
        warmup_count: Number of warmup requests.

    Returns:
        Full benchmark results including per-stage statistics.
    """
    queries = (BENCHMARK_QUERIES * ((query_count // len(BENCHMARK_QUERIES)) + 1))
    queries = queries[:query_count]

    # ── Warmup ─────────────────────────────────────────────────────────────
    print(f"\nWarming up with {warmup_count} requests...")
    for i in range(warmup_count):
        make_single_request(queries[i % len(queries)], api_url, i)
        print(f"  Warmup {i+1}/{warmup_count} done")

    # ── Benchmark ──────────────────────────────────────────────────────────
    print(f"\nRunning {query_count} benchmark requests (concurrency={concurrent})...")
    results = []
    benchmark_start = time.perf_counter()

    if concurrent == 1:
        # Sequential execution — cleaner timing measurements
        for i, query in enumerate(queries):
            result = make_single_request(query, api_url, i)
            results.append(result)
            status = "✅" if result["status"] == "success" else "❌"
            wall = result.get("wall_clock_ms", 0)
            retr = result.get("retrieval_latency_ms", 0)
            gen = result.get("generation_latency_ms", 0)
            cache = "CACHE" if result.get("cache_hit") else ""
            print(
                f"  [{i+1:2d}/{query_count}] {status} "
                f"wall={wall:>5}ms  retr={retr:>4}ms  "
                f"gen={gen:>5}ms  {cache}"
            )
    else:
        # Concurrent execution
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = {
                executor.submit(make_single_request, query, api_url, i): i
                for i, query in enumerate(queries)
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                print(
                    f"  [{completed:2d}/{query_count}] "
                    f"{'✅' if result['status'] == 'success' else '❌'} "
                    f"wall={result.get('wall_clock_ms', 0)}ms"
                )

    total_benchmark_time = time.perf_counter() - benchmark_start
    throughput_rps = query_count / total_benchmark_time

    # ── Compute statistics ─────────────────────────────────────────────────
    successful = [r for r in results if r.get("status") == "success"]
    cache_hits = [r for r in successful if r.get("cache_hit")]
    cache_misses = [r for r in successful if not r.get("cache_hit")]

    def stats(values: list[float], label: str) -> dict:
        if not values:
            return {}
        arr = np.array(values)
        return {
            "p50_ms": int(np.percentile(arr, 50)),
            "p95_ms": int(np.percentile(arr, 95)),
            "p99_ms": int(np.percentile(arr, 99)),
            "min_ms": int(np.min(arr)),
            "max_ms": int(np.max(arr)),
            "mean_ms": int(np.mean(arr)),
            "std_ms": int(np.std(arr)),
            "samples": len(values),
        }

    # Per-stage latencies (cache misses only — cache hits skip retrieval/generation)
    miss_results = cache_misses

    benchmark_stats = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "total_requests": query_count,
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "cache_hits": len(cache_hits),
            "cache_misses": len(cache_misses),
            "cache_hit_rate": len(cache_hits) / max(len(successful), 1),
            "concurrency": concurrent,
            "total_benchmark_time_seconds": round(total_benchmark_time, 2),
            "throughput_rps": round(throughput_rps, 2),
        },
        "stages": {
            "retrieval": stats(
                [r["retrieval_latency_ms"] for r in miss_results if r.get("retrieval_latency_ms")],
                "retrieval",
            ),
            "generation": stats(
                [r["generation_latency_ms"] for r in miss_results if r.get("generation_latency_ms")],
                "generation",
            ),
            "total_pipeline": stats(
                [r["total_pipeline_ms"] for r in miss_results if r.get("total_pipeline_ms")],
                "total_pipeline",
            ),
            "wall_clock_all_requests": stats(
                [r["wall_clock_ms"] for r in successful],
                "wall_clock",
            ),
            "wall_clock_cache_hits": stats(
                [r["wall_clock_ms"] for r in cache_hits],
                "cache_hits",
            ),
        },
        "sla_compliance": {},
        "raw_results": results,
    }

    # SLA compliance
    sla_targets = {
        "retrieval": {"target_ms": 200, "field": "p95_ms"},
        "generation": {"target_ms": 6000, "field": "p95_ms"},
        "total_pipeline": {"target_ms": 7000, "field": "p95_ms"},
    }

    for stage, target_config in sla_targets.items():
        stage_stats = benchmark_stats["stages"].get(stage, {})
        actual = stage_stats.get(target_config["field"], 0)
        target = target_config["target_ms"]
        benchmark_stats["sla_compliance"][stage] = {
            "p95_ms": actual,
            "target_ms": target,
            "meets_sla": actual <= target if actual > 0 else None,
        }

    return benchmark_stats


def print_benchmark_table(stats: dict) -> None:
    """Print a formatted benchmark results table."""
    print("\n" + "=" * 65)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 65)

    meta = stats["metadata"]
    print(f"\nRequests:    {meta['total_requests']} total, "
          f"{meta['successful']} succeeded, {meta['failed']} failed")
    print(f"Concurrency: {meta['concurrency']} concurrent request(s)")
    print(f"Throughput:  {meta['throughput_rps']:.2f} req/s")
    print(f"Cache hits:  {meta['cache_hits']} ({meta['cache_hit_rate']:.1%})")

    print(f"\n{'Stage':<25} {'p50':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8}")
    print("-" * 65)

    stage_labels = {
        "retrieval": "Retrieval (embed+FAISS)",
        "generation": "Generation (LLM)",
        "total_pipeline": "Total Pipeline",
        "wall_clock_all_requests": "Wall Clock (all)",
        "wall_clock_cache_hits": "Wall Clock (cache hits)",
    }

    for stage_key, label in stage_labels.items():
        s = stats["stages"].get(stage_key, {})
        if not s:
            continue
        print(
            f"  {label:<23} "
            f"{s.get('p50_ms', 0):>6}ms "
            f"{s.get('p95_ms', 0):>6}ms "
            f"{s.get('p99_ms', 0):>6}ms "
            f"{s.get('min_ms', 0):>6}ms "
            f"{s.get('max_ms', 0):>6}ms"
        )

    print(f"\n{'SLA Compliance':}")
    print("-" * 40)
    for stage, sla in stats.get("sla_compliance", {}).items():
        if sla.get("meets_sla") is None:
            continue
        status = "✅" if sla["meets_sla"] else "❌"
        print(
            f"  {status} {stage:<25} "
            f"p95={sla['p95_ms']}ms  "
            f"(target ≤{sla['target_ms']}ms)"
        )


def main():
    args = parse_args()

    print("=" * 65)
    print("RAG API LATENCY BENCHMARK")
    print("=" * 65)
    print(f"API URL:     {args.api_url}")
    print(f"Queries:     {args.queries}")
    print(f"Concurrency: {args.concurrent}")
    print(f"Warmup:      {args.warmup}")

    results = run_benchmark(
        api_url=args.api_url,
        query_count=args.queries,
        concurrent=args.concurrent,
        warmup_count=args.warmup,
    )

    print_benchmark_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results written to: {output_path}")


if __name__ == "__main__":
    main()