from examrag.evaluation import judge_case, keyword_recall, retrieval_hit, summarize


def test_keyword_recall():
    assert keyword_recall("A circular WAIT for resources", ["circular", "wait", "cpu"]) == 2 / 3
    assert keyword_recall("anything", []) == 1.0


def test_retrieval_hit():
    pages = [("a.pdf", 3), ("b.pdf", 47)]
    assert retrieval_hit(pages, "b.pdf", 47) is True
    assert retrieval_hit(pages, "a.pdf", 47) is False
    assert retrieval_hit(pages, None, 47) is True
    assert retrieval_hit(pages, None, None) is None


def test_answerable_case_passes_when_all_checks_pass():
    case = {"question": "q", "expected_page": 5, "expected_keywords": ["x", "y"]}
    r = judge_case(case, True, [("d.pdf", 5)], "x and y")
    assert r.passed and r.retrieval_hit and r.keyword_recall == 1.0


def test_answerable_case_fails_on_wrong_page():
    case = {"question": "q", "expected_page": 5}
    assert not judge_case(case, True, [("d.pdf", 9)], None).passed


def test_answerable_case_fails_when_refused():
    assert not judge_case({"question": "q"}, False, [], None).passed


def test_out_of_scope_case_requires_refusal():
    case = {"question": "q", "should_answer": False}
    assert judge_case(case, False, [], None).passed
    assert not judge_case(case, True, [], "made-up answer").passed


def test_summary_per_language():
    cases = [
        ({"question": "a", "language": "English"}, True),
        ({"question": "b", "language": "Tamil"}, False),
        ({"question": "c", "language": "Tamil"}, True),
    ]
    results = [judge_case(c, found, [], None) for c, found in cases]
    s = summarize(results)
    assert s["cases"] == 3
    assert s["by_language"]["English"]["pass_rate"] == 100.0
    assert s["by_language"]["Tamil"]["pass_rate"] == 50.0
    assert s["correct_refusal_rate"] is None


def test_summary_empty():
    assert summarize([])["pass_rate"] == 0.0
