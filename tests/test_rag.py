from dataclasses import replace

from examrag.config import Settings
from examrag.rag import (
    NOT_FOUND,
    answer_question,
    build_prompt,
    format_context,
    not_found_message,
)
from examrag.vectorstore import Hit

SETTINGS = Settings(min_similarity=0.35, top_k=3)


def hit(score=0.8, page=5, unit="Unit 1: Basics", text="A deadlock is a circular wait."):
    return Hit(text=text, source="os.pdf", page=page, unit=unit, score=score)


class FakeEmbedder:
    def embed_one(self, text):
        return [0.0]


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.last_filters = None

    def query(self, embedding, k=5, source=None, unit=None):
        self.last_filters = (source, unit)
        return self.hits


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, system, user, max_tokens=1200):
        self.calls.append((system, user, max_tokens))
        return "  A deadlock is a circular wait [1].  "


def test_context_has_numbered_passages_with_page_and_unit():
    ctx = format_context([hit(page=47), hit(page=48, unit="")])
    assert "[1] (os.pdf, page 47, Unit 1: Basics)" in ctx
    assert "[2] (os.pdf, page 48)" in ctx


def test_prompt_marks_and_language():
    system, user = build_prompt("What is deadlock?", [hit()], marks=2, language="Tamil")
    assert "2-mark" in system
    assert "Tamil" in system
    assert "<context>" in user and "What is deadlock?" in user


def test_prompt_marks_16_is_structured():
    system, _ = build_prompt("q", [hit()], marks=16)
    assert "16-mark" in system and "conclusion" in system


def test_prompt_no_marks_has_no_mark_template():
    system, _ = build_prompt("q", [hit()], marks=0)
    assert "-mark answer" not in system


def test_prompt_treats_context_as_untrusted():
    system, _ = build_prompt("q", [hit()])
    assert "untrusted" in system


def test_not_found_by_question_script():
    assert not_found_message("Same as question", "டெட்லாக் என்றால் என்ன?") == NOT_FOUND["Tamil"]
    assert not_found_message("Same as question", "डेडलॉक क्या है?") == NOT_FOUND["Hindi"]
    assert not_found_message("Same as question", "what is it?") == NOT_FOUND["English"]
    assert not_found_message("Hindi", "what is it?") == NOT_FOUND["Hindi"]


def test_answer_uses_llm_when_relevant():
    llm = FakeLLM()
    res = answer_question("What is deadlock?", FakeStore([hit(0.9)]), FakeEmbedder(), llm, SETTINGS, marks=2)
    assert res.found and res.text == "A deadlock is a circular wait [1]."
    assert len(llm.calls) == 1
    assert llm.calls[0][2] == 300  # 2-mark answers get a small token budget


def test_below_threshold_is_not_found_and_skips_llm():
    llm = FakeLLM()
    res = answer_question("Who won the World Cup?", FakeStore([hit(0.2)]), FakeEmbedder(), llm, SETTINGS)
    assert not res.found
    assert res.text == NOT_FOUND["English"]
    assert llm.calls == []


def test_no_hits_is_not_found():
    res = answer_question("anything", FakeStore([]), FakeEmbedder(), FakeLLM(), SETTINGS)
    assert not res.found


def test_only_relevant_hits_reach_the_prompt():
    llm = FakeLLM()
    hits = [hit(0.9, page=1), hit(0.1, page=2)]
    res = answer_question("q", FakeStore(hits), FakeEmbedder(), llm, SETTINGS)
    assert [h.page for h in res.hits] == [1]
    assert "page 2" not in llm.calls[0][1]


def test_filters_are_passed_to_the_store():
    store = FakeStore([hit()])
    answer_question("q", store, FakeEmbedder(), FakeLLM(), SETTINGS, source="os.pdf", unit="Unit 3")
    assert store.last_filters == ("os.pdf", "Unit 3")


def test_threshold_is_configurable():
    strict = replace(SETTINGS, min_similarity=0.95)
    res = answer_question("q", FakeStore([hit(0.9)]), FakeEmbedder(), FakeLLM(), strict)
    assert not res.found
