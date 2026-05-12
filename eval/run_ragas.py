"""
eval/run_ragas.py

RAGAS Evaluation Pipeline
==========================
Runs the gold evaluation dataset through the RAG pipeline and scores
the results using RAGAS metrics.

RAGAS Metrics:
  faithfulness:      Are all claims in the answer supported by the retrieved context?
                     Measures: LLM decomposes answer into atomic claims, checks each
                     against context. Score: fraction of claims supported by context.
                     Target: >= 0.75

  context_precision: Are the retrieved chunks relevant to the question?
                     Measures: fraction of retrieved chunks that are actually useful
                     for answering the question (precision, not recall).
                     Target: >= 0.70

  context_recall:    Does the retrieved context contain all the information needed?
                     Measures: fraction of the ground truth answer that can be
                     attributed to the retrieved context (recall).
                     Target: >= 0.70

  answer_relevancy:  Is the generated answer relevant to the question?
                     Measures: how directly the answer addresses the question.
                     Does not measure factual correctness — measures relevance.
                     Target: >= 0.75

Why RAGAS and not just ROUGE-L?
  ROUGE-L measures lexical overlap with a reference answer.
  It rewards verbosity and penalizes paraphrasing.
  RAGAS measures semantic quality of the RAG pipeline:
    - faithfulness catches hallucination (ROUGE-L cannot)
    - context_precision measures retrieval quality (ROUGE-L ignores retrieval)
    - context_recall measures whether we retrieved enough context
    - answer_relevancy measures response coherence (ROUGE-L measures coverage)

  RAGAS is the industry standard for RAG evaluation. Its metrics directly
  correspond to the failure modes we designed against in Phase 3.

Limitation acknowledgment:
  RAGAS uses an LLM (GPT-4 by default) as the judge for faithfulness and
  context_precision. This means:
    1. The evaluation costs ~$0.10-0.50 per full run (25 questions × 4 metrics)
    2. The judge LLM introduces its own bias and can disagree with humans
    3. Scores vary slightly between runs (LLM is stochastic)
  These are accepted limitations of the approach — RAGAS scores are
  directional indicators, not ground truth measurements.

Usage:
  # From project root
  python eval/run_ragas.py

  # With specific config
  python eval/run_ragas.py --questions 10 --output eval/ragas_results.json

Author: Enterprise RAG Assistant
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the gold dataset"
    )
    parser.add_argument(
        "--gold-dataset",
        default="eval/gold_dataset.json",
        help="Path to gold dataset JSON",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (None = all)",
    )
    parser.add_argument(
        "--output",
        default="eval/ragas_results.json",
        help="Path to write evaluation results JSON",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running RAG API",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Collect pipeline outputs only, skip RAGAS scoring",
    )
    parser.add_argument(
        "--use-fine-tuned",
        action="store_true",
        default=False,
        help="Use fine-tuned model (default: OpenAI fallback for eval consistency)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Output Collector
# ─────────────────────────────────────────────────────────────────────────────

def collect_pipeline_outputs(
    questions: list[dict],
    api_url: str,
    use_fine_tuned: bool = False,
) -> list[dict]:
    """
    Run each gold dataset question through the RAG API and collect outputs.

    For each question we collect:
      - generated answer (the system's response)
      - retrieved contexts (the chunks that informed the response)
      - latency breakdown (for benchmarking)
      - metadata (confidence, model used, grounding score)

    Args:
        questions: List of gold dataset question dicts.
        api_url: Base URL of the running RAG API.
        use_fine_tuned: Whether to request fine-tuned model.

    Returns:
        List of result dicts with question + pipeline output.
    """
    import httpx

    results = []
    total = len(questions)
    failed = 0

    print(f"\nCollecting pipeline outputs for {total} questions...")
    print(f"API: {api_url}")
    print(f"Model: {'fine-tuned' if use_fine_tuned else 'OpenAI fallback'}")
    print("-" * 60)

    for i, question in enumerate(questions, start=1):
        q_id = question["id"]
        query = question["question"]
        pillar = question.get("pillar", "")

        print(f"[{i:2d}/{total}] {q_id} ({pillar[:20]:<20}) ", end="", flush=True)

        request_start = time.time()
        try:
            response = httpx.post(
                f"{api_url}/generate",
                json={
                    "query": query,
                    "use_fine_tuned": use_fine_tuned,
                    "top_k": 5,
                },
                timeout=60.0,  # generous timeout for evaluation
            )

            if response.status_code != 200:
                print(f"FAILED (HTTP {response.status_code})")
                failed += 1
                results.append({
                    "question_id": q_id,
                    "question": query,
                    "error": f"HTTP {response.status_code}",
                    "status": "failed",
                })
                continue

            data = response.json()
            request_ms = round((time.time() - request_start) * 1000)

            # Extract retrieved contexts from sources for RAGAS
            # RAGAS expects: list of strings (one per retrieved chunk excerpt)
            retrieved_contexts = [
                source["excerpt"]
                for source in data.get("sources", [])
            ]

            result = {
                "question_id": q_id,
                "question": query,
                "pillar": pillar,
                "difficulty": question.get("difficulty", ""),
                # RAGAS inputs
                "answer": data.get("answer", ""),
                "contexts": retrieved_contexts,
                "ground_truth": question.get("expected_answer", ""),
                # Metadata
                "confidence": data.get("confidence", ""),
                "model_used": data.get("model_used", ""),
                "grounding_score": data.get("grounding_score", 0),
                "grounding_flag": data.get("grounding_flag", False),
                "is_refusal": data.get("is_refusal", False),
                "sources_count": len(data.get("sources", [])),
                "source_pillars": [s["pillar"] for s in data.get("sources", [])],
                # Latency
                "retrieval_latency_ms": data.get("retrieval_latency_ms", 0),
                "generation_latency_ms": data.get("generation_latency_ms", 0),
                "total_latency_ms": data.get("total_latency_ms", 0),
                "wall_clock_ms": request_ms,
                "tokens_used": data.get("tokens_used"),
                "status": "success",
            }

            # Check source retrieval precision
            expected_sources = question.get("expected_sources", [])
            actual_sources = [s["source_file"] for s in data.get("sources", [])]
            source_hit = any(
                exp in actual_source
                for exp in expected_sources
                for actual_source in actual_sources
            )
            result["source_retrieval_correct"] = source_hit

            results.append(result)
            status_icon = "✅" if not data.get("is_refusal") else "⚠️ "
            print(
                f"{status_icon} {request_ms:>5}ms | "
                f"conf={data.get('confidence', '?'):<6} | "
                f"src={'✓' if source_hit else '✗'}"
            )

        except httpx.ConnectError:
            print(f"CONNECTION ERROR — is the API running at {api_url}?")
            failed += 1
            results.append({
                "question_id": q_id,
                "question": query,
                "error": "Connection refused",
                "status": "failed",
            })
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            results.append({
                "question_id": q_id,
                "question": query,
                "error": str(e),
                "status": "failed",
            })

    print(f"\nCollection complete: {total - failed}/{total} succeeded, {failed} failed")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Scoring
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_evaluation(
    pipeline_outputs: list[dict],
) -> dict:
    """
    Score pipeline outputs using RAGAS metrics.

    RAGAS requires:
      - questions: list[str]        — the original questions
      - answers: list[str]          — generated answers from the pipeline
      - contexts: list[list[str]]   — retrieved chunks for each question
      - ground_truths: list[str]    — reference answers (for context_recall)

    Args:
        pipeline_outputs: Results from collect_pipeline_outputs().

    Returns:
        Dict with per-metric scores and per-question breakdown.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
        return {}

    # Filter to successful, non-refusal results only
    # Refusals cannot be scored on faithfulness or answer_relevancy
    scoreable = [
        r for r in pipeline_outputs
        if r.get("status") == "success" and not r.get("is_refusal", False)
    ]

    if not scoreable:
        print("No scoreable results — all questions resulted in refusals or errors")
        return {}

    print(f"\nRunning RAGAS on {len(scoreable)} scoreable results...")
    print("(Refusals and errors excluded from RAGAS scoring)")
    print("Note: RAGAS uses GPT-4 as the judge — requires OPENAI_API_KEY")

    # Build RAGAS dataset
    ragas_data = {
        "question": [r["question"] for r in scoreable],
        "answer": [r["answer"] for r in scoreable],
        "contexts": [r["contexts"] for r in scoreable],
        "ground_truth": [r["ground_truth"] for r in scoreable],
    }

    ragas_dataset = Dataset.from_dict(ragas_data)

    try:
        ragas_start = time.time()
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        ragas_time = round(time.time() - ragas_start, 1)

        # Convert to dict
        scores_dict = result.to_pandas().to_dict(orient="list")
        aggregate_scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }

        print(f"\nRAGAS evaluation complete in {ragas_time}s")
        print("\nAggregate scores:")
        targets = {
            "faithfulness": 0.75,
            "answer_relevancy": 0.75,
            "context_precision": 0.70,
            "context_recall": 0.70,
        }
        for metric, score in aggregate_scores.items():
            target = targets.get(metric, 0.70)
            status = "✅" if score >= target else "⚠️ "
            print(f"  {status} {metric:<22} {score:.4f}  (target: ≥{target})")

        return {
            "aggregate": aggregate_scores,
            "per_question": scores_dict,
            "questions_scored": len(scoreable),
            "evaluation_time_seconds": ragas_time,
        }

    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        print("Common causes:")
        print("  - OPENAI_API_KEY not set (RAGAS uses GPT-4 as judge)")
        print("  - RAGAS version incompatibility")
        print("  - Network timeout during LLM judge calls")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Source Retrieval Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_source_retrieval(
    pipeline_outputs: list[dict],
    gold_questions: list[dict],
) -> dict:
    """
    Evaluate whether the correct source documents were retrieved.

    This is a simpler, cheaper complement to RAGAS — it checks
    whether the retrieved sources match the expected_sources field
    in the gold dataset. No LLM judge required.

    Metrics:
      source_hit_rate: fraction of questions where at least one retrieved
                       source matches the expected source document.
      exact_match_rate: fraction where ALL expected sources were retrieved.
    """
    gold_by_id = {q["id"]: q for q in gold_questions}
    successful = [r for r in pipeline_outputs if r.get("status") == "success"]

    source_hits = 0
    exact_matches = 0
    pillar_hits: dict[str, list[bool]] = {}

    for result in successful:
        q_id = result["question_id"]
        gold = gold_by_id.get(q_id, {})
        expected = set(gold.get("expected_sources", []))
        actual = set(result.get("source_pillars", []))  # actual pillar names retrieved

        # Check if any expected source was retrieved
        hit = result.get("source_retrieval_correct", False)
        if hit:
            source_hits += 1

        # Track by pillar
        pillar = result.get("pillar", "Unknown")
        if pillar not in pillar_hits:
            pillar_hits[pillar] = []
        pillar_hits[pillar].append(hit)

    total = len(successful)
    source_hit_rate = source_hits / total if total > 0 else 0

    pillar_breakdown = {
        pillar: {
            "hit_rate": sum(hits) / len(hits) if hits else 0,
            "questions": len(hits),
        }
        for pillar, hits in pillar_hits.items()
    }

    return {
        "source_hit_rate": round(source_hit_rate, 4),
        "questions_evaluated": total,
        "source_hits": source_hits,
        "pillar_breakdown": pillar_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Latency Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def compute_latency_benchmarks(pipeline_outputs: list[dict]) -> dict:
    """
    Compute p50 and p95 latency for each pipeline stage.

    Uses numpy percentile on the collected timing data.

    Args:
        pipeline_outputs: Results from collect_pipeline_outputs().

    Returns:
        Dict with p50 and p95 for each latency dimension.
    """
    import numpy as np

    successful = [
        r for r in pipeline_outputs
        if r.get("status") == "success"
    ]

    if not successful:
        return {}

    def percentiles(values: list[float]) -> dict:
        if not values:
            return {"p50": 0, "p95": 0, "min": 0, "max": 0, "mean": 0}
        arr = np.array(values)
        return {
            "p50": int(np.percentile(arr, 50)),
            "p95": int(np.percentile(arr, 95)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": int(np.mean(arr)),
        }

    retrieval_latencies = [r["retrieval_latency_ms"] for r in successful]
    generation_latencies = [r["generation_latency_ms"] for r in successful]
    total_latencies = [r["total_latency_ms"] for r in successful]
    wall_clock_latencies = [r["wall_clock_ms"] for r in successful]
    tokens = [r["tokens_used"] for r in successful if r.get("tokens_used")]

    benchmarks = {
        "retrieval_ms": percentiles(retrieval_latencies),
        "generation_ms": percentiles(generation_latencies),
        "total_pipeline_ms": percentiles(total_latencies),
        "wall_clock_ms": percentiles(wall_clock_latencies),
        "samples": len(successful),
    }

    if tokens:
        benchmarks["tokens_per_response"] = percentiles([float(t) for t in tokens])

    # SLA compliance check
    targets = {
        "retrieval_ms": {"p95_target": 200, "field": "p95"},
        "generation_ms": {"p95_target": 6000, "field": "p95"},
        "total_pipeline_ms": {"p95_target": 7000, "field": "p95"},
    }

    sla_results = {}
    for stage, target_config in targets.items():
        actual_p95 = benchmarks[stage][target_config["field"]]
        target_p95 = target_config["p95_target"]
        sla_results[stage] = {
            "actual_p95_ms": actual_p95,
            "target_p95_ms": target_p95,
            "meets_sla": actual_p95 <= target_p95,
        }

    benchmarks["sla_compliance"] = sla_results
    return benchmarks


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Testing
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial_tests(
    adversarial_cases: list[dict],
    api_url: str,
) -> list[dict]:
    """
    Run adversarial test cases and verify the system handles them correctly.

    Each adversarial case has an expected_behavior that the system must
    produce. Test failures are documented in the evaluation report.

    Args:
        adversarial_cases: List of adversarial case dicts from gold_dataset.json.
        api_url: Base URL of the running RAG API.

    Returns:
        List of test results with pass/fail status for each case.
    """
    import httpx

    results = []
    print(f"\nRunning {len(adversarial_cases)} adversarial test cases...")
    print("-" * 60)

    for case in adversarial_cases:
        case_id = case["id"]
        case_type = case["type"]
        expected_behavior = case["expected_behavior"]

        print(f"[{case_id}] {case_type:<25} ", end="", flush=True)

        # Build query string
        if case.get("query_length"):
            # Generate a query of specified length
            base = case.get("query_pattern", "Test query. {padding}")
            padding = "x" * (case["query_length"] - len(base.replace("{padding}", "")) + 1)
            query = base.replace("{padding}", padding)
        else:
            query = case.get("query", "")

        try:
            response = httpx.post(
                f"{api_url}/generate",
                json={
                    "query": query,
                    "use_fine_tuned": False,
                    "top_k": 5,
                },
                timeout=30.0,
            )

            actual_status_code = response.status_code
            result = {
                "case_id": case_id,
                "type": case_type,
                "expected_behavior": expected_behavior,
                "query_preview": query[:80] + "..." if len(query) > 80 else query,
                "actual_http_status": actual_status_code,
            }

            if actual_status_code == 200:
                body = response.json()
                result["actual_is_refusal"] = body.get("is_refusal", False)
                result["actual_answer_preview"] = body.get("answer", "")[:100]

            # ── Evaluate whether expected behavior was achieved ────────────
            passed = False
            failure_reason = ""

            if expected_behavior == "validation_error":
                # Expect a 422 from Pydantic validation
                expected_http = case.get("expected_http_status", 422)
                passed = actual_status_code == expected_http
                if not passed:
                    failure_reason = (
                        f"Expected HTTP {expected_http}, got {actual_status_code}"
                    )

            elif expected_behavior == "guardrail_block":
                # Expect a 200 with is_refusal=True
                # (guardrails return structured responses, not HTTP errors)
                if actual_status_code == 200:
                    body = response.json()
                    passed = body.get("is_refusal", False)
                    if not passed:
                        failure_reason = (
                            "Expected is_refusal=True, got is_refusal=False. "
                            f"Answer: {body.get('answer', '')[:80]}"
                        )
                else:
                    failure_reason = (
                        f"Expected HTTP 200 with refusal, got HTTP {actual_status_code}"
                    )

            elif expected_behavior == "refusal":
                # Expect a 200 with is_refusal=True (no relevant context)
                if actual_status_code == 200:
                    body = response.json()
                    passed = body.get("is_refusal", False)
                    if not passed:
                        failure_reason = (
                            "Expected refusal (no relevant context) but got an answer. "
                            "Retrieval may be returning false positives."
                        )
                else:
                    failure_reason = f"Unexpected HTTP {actual_status_code}"

            result["passed"] = passed
            result["failure_reason"] = failure_reason if not passed else ""

            status_icon = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status_icon} | HTTP {actual_status_code}")
            if not passed:
                print(f"         Reason: {failure_reason}")

        except Exception as e:
            result = {
                "case_id": case_id,
                "type": case_type,
                "expected_behavior": expected_behavior,
                "passed": False,
                "failure_reason": str(e),
                "error": str(e),
            }
            print(f"ERROR: {e}")

        results.append(result)

    passed_count = sum(1 for r in results if r.get("passed"))
    print(f"\nAdversarial tests: {passed_count}/{len(results)} passed")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    pipeline_outputs: list[dict],
    ragas_scores: dict,
    source_retrieval: dict,
    latency_benchmarks: dict,
    adversarial_results: list[dict],
    output_path: str,
) -> None:
    """
    Write the full evaluation results to a JSON file.

    The JSON is then used to generate the markdown evaluation report.

    Args:
        pipeline_outputs: Raw pipeline results.
        ragas_scores: RAGAS metric scores.
        source_retrieval: Source retrieval precision metrics.
        latency_benchmarks: Per-stage latency statistics.
        adversarial_results: Adversarial test results.
        output_path: Path to write the JSON output.
    """
    report = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "questions_evaluated": len(pipeline_outputs),
            "successful": sum(1 for r in pipeline_outputs if r.get("status") == "success"),
            "failed": sum(1 for r in pipeline_outputs if r.get("status") == "failed"),
            "refusals": sum(1 for r in pipeline_outputs if r.get("is_refusal")),
        },
        "ragas_scores": ragas_scores,
        "source_retrieval": source_retrieval,
        "latency_benchmarks": latency_benchmarks,
        "adversarial_results": adversarial_results,
        "pipeline_outputs": pipeline_outputs,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults written to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("ENTERPRISE RAG ASSISTANT — EVALUATION PIPELINE")
    print("=" * 60)

    # ── Load gold dataset ──────────────────────────────────────────────────
    gold_path = Path(args.gold_dataset)
    if not gold_path.exists():
        print(f"Gold dataset not found: {gold_path}")
        sys.exit(1)

    with open(gold_path) as f:
        gold_data = json.load(f)

    questions = gold_data["questions"]
    adversarial_cases = gold_data.get("adversarial_cases", [])

    if args.questions:
        questions = questions[: args.questions]
        print(f"Evaluating {args.questions} of {len(gold_data['questions'])} questions")

    print(f"Gold dataset: {len(questions)} questions, {len(adversarial_cases)} adversarial cases")

    # ── Collect pipeline outputs ───────────────────────────────────────────
    pipeline_outputs = collect_pipeline_outputs(
        questions=questions,
        api_url=args.api_url,
        use_fine_tuned=args.use_fine_tuned,
    )

    # ── Source retrieval evaluation ────────────────────────────────────────
    print("\nEvaluating source retrieval accuracy...")
    source_retrieval = evaluate_source_retrieval(pipeline_outputs, questions)
    print(f"Source hit rate: {source_retrieval['source_hit_rate']:.1%}")

    # ── Latency benchmarks ─────────────────────────────────────────────────
    print("\nComputing latency benchmarks...")
    latency_benchmarks = compute_latency_benchmarks(pipeline_outputs)
    if latency_benchmarks:
        print(f"Retrieval p95:   {latency_benchmarks['retrieval_ms']['p95']}ms")
        print(f"Generation p95:  {latency_benchmarks['generation_ms']['p95']}ms")
        print(f"Total p95:       {latency_benchmarks['total_pipeline_ms']['p95']}ms")

    # ── RAGAS evaluation ───────────────────────────────────────────────────
    ragas_scores = {}
    if not args.skip_ragas:
        ragas_scores = run_ragas_evaluation(pipeline_outputs)
    else:
        print("\nSkipping RAGAS evaluation (--skip-ragas flag set)")

    # ── Adversarial testing ────────────────────────────────────────────────
    adversarial_results = run_adversarial_tests(adversarial_cases, args.api_url)

    # ── Write results ──────────────────────────────────────────────────────
    generate_report(
        pipeline_outputs=pipeline_outputs,
        ragas_scores=ragas_scores,
        source_retrieval=source_retrieval,
        latency_benchmarks=latency_benchmarks,
        adversarial_results=adversarial_results,
        output_path=args.output,
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if ragas_scores.get("aggregate"):
        print("\nRAGAS Scores:")
        targets = {
            "faithfulness": 0.75,
            "answer_relevancy": 0.75,
            "context_precision": 0.70,
            "context_recall": 0.70,
        }
        for metric, score in ragas_scores["aggregate"].items():
            target = targets.get(metric, 0.70)
            status = "✅" if score >= target else "⚠️ "
            print(f"  {status} {metric:<22} {score:.4f}  target ≥{target}")

    if latency_benchmarks.get("sla_compliance"):
        print("\nSLA Compliance:")
        for stage, sla in latency_benchmarks["sla_compliance"].items():
            status = "✅" if sla["meets_sla"] else "❌"
            print(
                f"  {status} {stage:<25} "
                f"p95={sla['actual_p95_ms']}ms  "
                f"(target ≤{sla['target_p95_ms']}ms)"
            )

    adv_passed = sum(1 for r in adversarial_results if r.get("passed"))
    print(f"\nAdversarial Tests: {adv_passed}/{len(adversarial_results)} passed")

    print(f"\nSource Retrieval Hit Rate: {source_retrieval.get('source_hit_rate', 0):.1%}")
    print("\nFull results written to:", args.output)


if __name__ == "__main__":
    main()