import pytest

from examrag.ingest import chunk_text, chunks_from_pages, clean_text, detect_unit


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} ends here." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # overlap means consecutive chunks share some text
    assert chunks[0][-20:].strip().split()[-1] in chunks[1]


def test_chunks_cover_all_text():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), size=200, overlap=40)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)


def test_prefers_sentence_boundary():
    text = ("A" * 40 + ". ") + ("B" * 40 + ". ") + ("C" * 40 + ". ")
    chunks = chunk_text(text, size=100, overlap=10)
    assert chunks[0].endswith(".")


def test_hindi_danda_boundary():
    text = ("यह पहला वाक्य है।" * 3) + (" यह दूसरा वाक्य है।" * 10)
    chunks = chunk_text(text, size=60, overlap=10)
    assert len(chunks) > 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size=size, overlap=overlap)


def test_clean_text_normalises_whitespace():
    assert clean_text("a \t b c\n\n\n\nd") == "a b c\n\nd"


@pytest.mark.parametrize(
    "page,expected",
    [
        ("UNIT III\nMemory management", "Unit III"),
        ("Unit 2: Process Scheduling\nbody", "Unit 2: Process Scheduling"),
        ("CHAPTER 4 - Sorting", "Chapter 4: Sorting"),
    ],
)
def test_detect_unit(page, expected):
    assert detect_unit(page).startswith(expected.split(":")[0])


def test_detect_unit_keeps_current_when_none_found():
    assert detect_unit("just body text", current="Unit 1") == "Unit 1"


def test_chunks_carry_page_and_unit():
    pages = [
        (1, "UNIT 1: Basics\nIntro text about basics."),
        (2, "More text continuing unit one."),
        (3, "UNIT 2: Advanced\nAdvanced material."),
    ]
    chunks = chunks_from_pages(pages, source="notes.pdf", size=500, overlap=50)
    by_page = {c.page: c for c in chunks}
    assert by_page[1].unit.startswith("Unit 1")
    assert by_page[2].unit.startswith("Unit 1")  # carried forward
    assert by_page[3].unit.startswith("Unit 2")
    assert all(c.source == "notes.pdf" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
