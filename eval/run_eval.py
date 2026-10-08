"""Evaluate ExamRAG on a question set.

Examples
--------
# Free, offline: only measures whether retrieval finds the right page
python eval/run_eval.py --docs data/uploads --questions eval/questions.json --retrieval-only

# Full run (calls the configured LLM) and compare chunk sizes
python eval/run_eval.py --docs data/uploads --questions eval/questions.json --chunk-size 600
python eval/run_eval.py --docs data/uploads --questions eval/questions.json --chunk-size 1200

Question file format: see eval/questions.example.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from examrag.config import load_settings  # noqa: E402
from examrag.embeddings import Embedder  # noqa: E402
from examrag.evaluation import judge_case, summarize  # noqa: E402
from examrag.llm import get_llm  # noqa: E402
from examrag.pipeline import index_pdf  # noqa: E402
from examrag.rag import answer_question, retrieve  # noqa: E402
from examrag.vectorstore import VectorStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--docs", required=True, help="Folder containing the PDFs to index")
    ap.add_argument("--questions", required=True, help="JSON file of test cases")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--retrieval-only", action="store_true", help="Skip the LLM; score retrieval only")
    ap.add_argument("--out", default=None, help="Write full results as JSON to this path")
    args = ap.parse_args()

    settings = load_settings()
    overrides = {
        k: v
        for k, v in {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
        }.items()
        if v is not None
    }
    settings = dataclasses.replace(settings, **overrides)

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    pdfs = sorted(Path(args.docs).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.docs}")

    embedder = Embedder(settings.embedding_model)
    llm = None if args.retrieval_only else get_llm(settings)

    with tempfile.TemporaryDirectory() as tmp:  # fresh index so settings changes take effect
        store = VectorStore(tmp)
        for pdf in pdfs:
            n = index_pdf(pdf, store, embedder, settings)
            print(f"indexed {pdf.name}: {n} chunks")

        results = []
        for case in cases:
            if llm is None:
                hits = retrieve(case["question"], store, embedder, settings, source=case.get("source"))
                relevant = [h for h in hits if h.score >= settings.min_similarity]
                res = judge_case(case, bool(relevant), [(h.source, h.page) for h in hits], None)
            else:
                ans = answer_question(
                    case["question"], store, embedder, llm, settings,
                    marks=case.get("marks", 0),
                    language=case.get("answer_language", "Same as question"),
                    source=case.get("source"),
                )
                res = judge_case(case, ans.found, [(h.source, h.page) for h in ans.hits], ans.text)
            results.append(res)
            print(("PASS" if res.passed else "FAIL"), "|", res.language, "|", res.question[:70])

    summary = summarize(results)
    summary["settings"] = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "embedding_model": settings.embedding_model,
        "llm": "retrieval-only" if llm is None else settings.llm_provider,
    }
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "cases": [dataclasses.asdict(r) for r in results]},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
