"""
evaluation/run_eval.py
-----------------------
Evaluation harness for the Ontario Oversight CRAG pipeline.

Runs all 12 QA benchmark prompts through the pipeline and scores each answer
against expected topics and citations. Calculates accuracy % across the test suite.

SCORING METHODOLOGY:
Each answer is scored on two criteria:
1. Citation presence (50% weight) — does the answer cite at least one expected CSPA section?
2. Topic coverage (50% weight) — does the answer mention expected key topics?

A question PASSES if its combined score >= 0.5 (50%).
Overall accuracy = number of passing questions / 12

Usage:
    python evaluation/run_eval.py

Output:
    - Console report with per-question results
    - evaluation/results_YYYY-MM-DD.json with full results

Branch: develop
Issue:  #12 — Evaluation harness
"""

import os
import sys
import json
import re
from datetime import date

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()
from src.ingestion.pipeline import run_ingestion
from src.agent.graph import build_graph, run_query
import time

def run_query_with_retry(graph, prompt, retries=3, wait=60):
    """Run a query with retry logic for 429 rate limit errors."""
    for attempt in range(retries):
        try:
            return run_query(graph, prompt)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  ⏳ Rate limited. Waiting {wait}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait)
            else:
                raise


def check_citation_present(answer: str, expected_citations: list[str]) -> tuple[bool, list[str]]:
    """
    Check if the answer contains any valid CSPA, LECA, or O. Reg. citation.
    """
    pattern = r'\[?(?:CSPA s\.\d+(?:\(\d+\))*(?:\([a-z]\))?|O\.\s*Reg\.(?:\s*\d+/\d+)?(?:\s*s\.\d+(?:\(\d+\))*(?:\([a-z]\))?)?|LECA Guideline\s+\d+|LECA Rules|LECA [—-])\]?'
    found = re.findall(pattern, answer)
    found = list(set(found))  # deduplicate
    return len(found) > 0, found


def check_topic_coverage(answer: str, expected_topics: list[str]) -> tuple[float, list[str]]:
    """
    Check how many expected topics are covered in the answer.

    Args:
        answer: The generated answer text.
        expected_topics: List of expected topic keywords/phrases.

    Returns:
        Tuple of (coverage_score: float 0-1, found_topics: list)
    """
    found = []
    answer_lower = answer.lower()

    for topic in expected_topics:
        if topic.lower() in answer_lower:
            found.append(topic)

    score = len(found) / len(expected_topics) if expected_topics else 0
    return score, found


def evaluate_answer(
    answer: str,
    expected_citations: list[str],
    expected_topics: list[str]
) -> dict:
    """
    Score a single answer against expected citations and topics.

    Returns a dict with:
    - citation_passed: bool
    - citation_found: list of found citations
    - topic_score: float 0-1
    - topics_found: list of found topics
    - overall_score: float 0-1 (weighted average)
    - passed: bool (overall_score >= 0.5)
    """
    citation_passed, citation_found = check_citation_present(answer, expected_citations)
    topic_score, topics_found = check_topic_coverage(answer, expected_topics)

    # Citation is binary (50% weight), topics are proportional (50% weight)
    citation_score = 1.0 if citation_passed else 0.0
    overall_score = (citation_score * 0.5) + (topic_score * 0.5)

    return {
        "citation_passed": citation_passed,
        "citation_found": citation_found,
        "topic_score": round(topic_score, 2),
        "topics_found": topics_found,
        "overall_score": round(overall_score, 2),
        "passed": overall_score >= 0.5,
    }

def llm_judge(question: str, answer: str, expected_topics: list[str], retrieved_chunks) -> dict | None:
    """
    Score the answer using the Qwen model via OpenRouter (OpenAI-compatible API).
    """
    try:
        from openai import OpenAI
        import json
        import os
        client = OpenAI(
            base_url=os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        system_prompt = "You are an expert evaluator. Evaluate the assistant's answer on Faithfulness, Relevance, and Completeness. Return ONLY a valid JSON object."
        user_prompt = f"""
        Question: {question}
        Answer: {answer}
        Expected Topics: {expected_topics}
        Retrieved Context Chunks: {retrieved_chunks}
        
        Provide a JSON object with strictly these keys and 0 to 1 float values:
        - "faithfulness": (is the answer grounded in the retrieved chunks?)
        - "relevance": (does it directly answer the question?)
        - "completeness": (does it cover the expected topics?)
        """

        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-plus-02-15"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        scores = json.loads(response.choices[0].message.content)
        f_score = float(scores.get("faithfulness", 0.0))
        r_score = float(scores.get("relevance", 0.0))
        c_score = float(scores.get("completeness", 0.0))
        avg_score = (f_score + r_score + c_score) / 3.0
        
        return {
            "faithfulness": round(f_score, 2),
            "relevance": round(r_score, 2),
            "completeness": round(c_score, 2),
            "average": round(avg_score, 2)
        }
    except Exception as e:
        print(f"  ❌ LLM Judge failed: {e}")
        return None


def main():
    """Run the full evaluation pipeline."""

    print("=" * 70)
    print("Ontario Oversight CRAG Evaluation Harness")
    print(f"Date: {date.today()}")
    print("=" * 70)

    # Load QA pairs
    qa_path = os.path.join(os.path.dirname(__file__), "qa_pairs.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"\nLoaded {len(qa_pairs)} QA prompts.")

    # Initialise pipeline
    print("\nInitialising pipeline...")
   

    chunks = run_ingestion()
    graph = build_graph(chunks)
    print("Pipeline ready.\n")

    # Run evaluation
    results = []
    passed_count = 0

    print("-" * 70)

    for qa in qa_pairs:
        prompt_id = qa["id"]
        prompt = qa["prompt"]
        expected_citations = qa["expected_citations"]
        expected_topics = qa["expected_topics"]
        category = qa["category"]

        print(f"\nQ{prompt_id} [{category}]")
        print(f"Prompt: {prompt[:80]}...")

        try:
            # Run through pipeline
            result = run_query_with_retry(graph, prompt)
            answer = result.get("answer", "")
            
            # Extract retrieved chunks flexibly depending on the node states
            retrieved_chunks = result.get("documents", result.get("context", result.get("retrieved_chunks", [])))

            # Score the answer
            scores = evaluate_answer(answer, expected_citations, expected_topics)

            status = "✅ PASS" if scores["passed"] else "❌ FAIL"
            print(f"Status: {status}")
            print(f"Citations found: {scores['citation_found'] or 'None'}")
            print(f"Topics found: {scores['topics_found']} ({scores['topic_score']*100:.0f}%)")
            print(f"Overall score: {scores['overall_score']*100:.0f}%")
            
            # Add LLM judge
            judge_scores = llm_judge(prompt, answer, expected_topics, retrieved_chunks)
            if judge_scores:
                print(f"LLM Judge: Faithfulness={judge_scores['faithfulness']}, Relevance={judge_scores['relevance']}, Completeness={judge_scores['completeness']} (Avg: {judge_scores['average']})")
            else:
                print("LLM Judge: Failed/Unavailable")

            if scores["passed"]:
                passed_count += 1

            results.append({
                "id": prompt_id,
                "category": category,
                "prompt": prompt,
                "answer": answer,
                "expected_citations": expected_citations,
                "expected_topics": expected_topics,
                "llm_judge": judge_scores,
                **scores,
            })

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "id": prompt_id,
                "category": category,
                "prompt": prompt,
                "answer": "",
                "error": str(e),
                "passed": False,
                "overall_score": 0,
            })
        time.sleep(4)  # stay under RPM limit
        print("-" * 70)

    # Summary
    accuracy = passed_count / len(qa_pairs)
    print(f"\n{'=' * 70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results: {passed_count}/{len(qa_pairs)} passed")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print(f"{'=' * 70}")

    # Category breakdown
    print("\nResults by category:")
    categories = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.get("passed", False):
            categories[cat]["passed"] += 1

    for cat, stats in categories.items():
        pct = stats["passed"] / stats["total"] * 100
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")

    # Save results
    output_path = os.path.join(
        os.path.dirname(__file__),
        f"results_{date.today()}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": str(date.today()),
            "accuracy": round(accuracy, 3),
            "passed": passed_count,
            "total": len(qa_pairs),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")
    return accuracy


if __name__ == "__main__":
    main()
