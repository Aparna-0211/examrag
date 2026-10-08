"""Scoring helpers for the evaluation harness (kept dependency-free so they are easy to test)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CaseResult:
    question: str
    language: str
    should_answer: bool
    found: bool
    retrieval_hit: bool | None  # None when no expected_page was given
    keyword_recall: float | None  # None when no expected_keywords / no LLM run
    passed: bool


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords that appear in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(1 for k in keywords if k.lower() in text) / len(keywords)


def retrieval_hit(hit_pages: list[tuple[str, int]], source: str | None, page: int | None) -> bool | None:
    """Did retrieval return the expected page (optionally within the expected source)?"""
    if page is None:
        return None
    return any(p == page and (source is None or s == source) for s, p in hit_pages)


def judge_case(
    case: dict,
    found: bool,
    hit_pages: list[tuple[str, int]],
    answer_text: str | None,
    pass_threshold: float = 0.6,
) -> CaseResult:
    should_answer = bool(case.get("should_answer", True))
    rh = retrieval_hit(hit_pages, case.get("source"), case.get("expected_page"))
    recall = None
    if answer_text is not None and case.get("expected_keywords"):
        recall = keyword_recall(answer_text, case["expected_keywords"])

    if not should_answer:
        passed = not found  # correct behaviour is to refuse
    else:
        checks = [found]
        if rh is not None:
            checks.append(rh)
        if recall is not None:
            checks.append(recall >= pass_threshold)
        passed = all(checks)

    return CaseResult(
        question=case["question"],
        language=case.get("language", "English"),
        should_answer=should_answer,
        found=found,
        retrieval_hit=rh,
        keyword_recall=recall,
        passed=passed,
    )


def summarize(results: list[CaseResult]) -> dict:
    """Overall and per-language pass rates plus retrieval/keyword averages."""

    def rate(items: list[CaseResult]) -> float:
        return round(100 * sum(r.passed for r in items) / len(items), 1) if items else 0.0

    by_lang: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_lang[r.language].append(r)

    rh = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    kr = [r.keyword_recall for r in results if r.keyword_recall is not None]
    refusals = [r for r in results if not r.should_answer]

    return {
        "cases": len(results),
        "pass_rate": rate(results),
        "by_language": {lang: {"cases": len(v), "pass_rate": rate(v)} for lang, v in sorted(by_lang.items())},
        "retrieval_hit_rate": round(100 * sum(rh) / len(rh), 1) if rh else None,
        "avg_keyword_recall": round(100 * sum(kr) / len(kr), 1) if kr else None,
        "correct_refusal_rate": rate(refusals) if refusals else None,
    }
