# ExamRAG 📚

An exam-aware, multilingual study assistant that answers **only from your own documents**
(notes, textbooks, past papers) and cites the page. It can run fully offline.

> Existing "chat with your PDF" tools answer questions. ExamRAG is built for *exam prep*:
> it shapes answers to the marks asked, filters by syllabus unit, refuses to guess, and
> works in English, Tamil and Hindi.

## Features

| Feature | How it works |
|---|---|
| **Page-level citations** | Every chunk keeps its source file and page; the model cites `[1]`, `[2]`, and the UI shows the passages. |
| **Exam-aware answers** | Choose 2 / 5 / 8 / 16 marks; the prompt asks for a matching length and structure. |
| **Syllabus mapping** | Headings like `UNIT III` / `Chapter 2` are detected and tagged on each chunk; filter retrieval by unit. |
| **Honest "not found"** | If the best match is below a similarity threshold, the app says so *without calling the LLM*. |
| **Multilingual** | Multilingual embeddings let a Tamil/Hindi question retrieve from English text; answer language is selectable. |
| **Offline mode** | Local embeddings + Chroma + Ollama. No API calls. |
| **Measured quality** | `eval/run_eval.py` reports retrieval hit rate, keyword recall and correct-refusal rate, per language. |

## Architecture

```
PDF ──► pypdf (per page) ──► unit tagging ──► chunking (size/overlap)
                                                    │
                                   multilingual embeddings (BGE-M3)
                                                    ▼
question ─► embed ─► Chroma (cosine) ─► top-k ─► threshold ─┬─► "not found"
                                                            └─► exam-aware prompt ─► LLM ─► answer + sources
                                                                      (Anthropic API  or  Ollama)
```

## Quickstart

```bash
git clone https://github.com/Aparna-0211/examrag.git
cd examrag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and set:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-key        # create one at console.groq.com
GROQ_MODEL=openai/gpt-oss-120b    # use any chat model your key can access
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Then run:

```bash
streamlit run app.py
```

The first run downloads the embedding model, which needs internet once.
BGE-M3 gives better multilingual retrieval but is much larger; see *Choosing models*.

**Windows note:** if loading the model fails with a `FileNotFoundError` in the Hugging Face cache, download it into the project folder instead:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', local_dir='models/minilm')"
```

and set `EMBEDDING_MODEL=models/minilm` in `.env`.

## Offline mode

```bash
# 1. Install Ollama (https://ollama.com) and pull a model while you still have internet
ollama pull qwen2.5:7b
# 2. In .env
LLM_PROVIDER=ollama
HF_HUB_OFFLINE=1        # after the embedding model has been downloaded once
# 3. Run
streamlit run app.py
```

Or with Docker: `docker compose up -d` (see `docker-compose.yml`).

## Choosing models

| Setting | Quality | Size / speed |
|---|---|---|
| `EMBEDDING_MODEL=BAAI/bge-m3` | Strong multilingual retrieval | Large; slower on CPU |
| `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Lower, fine for demos | Much smaller; fits free hosting |

Small local LLMs are noticeably weaker in Tamil and Hindi than in English. Don't assume
quality: run the eval per language and report what you measure.

## Evaluate

1. Put your PDFs in a folder, e.g. `data/uploads/`.
2. Copy `eval/questions.example.json` to `eval/questions.json` and write 20+ real questions
   with the expected page and a few expected keywords. Include some questions that are
   **not** in the documents (`"should_answer": false`).
3. Run:

```bash
python eval/run_eval.py --docs data/uploads --questions eval/questions.json --retrieval-only
python eval/run_eval.py --docs data/uploads --questions eval/questions.json --chunk-size 600 --out results_600.json
python eval/run_eval.py --docs data/uploads --questions eval/questions.json --chunk-size 1200 --out results_1200.json
```

**Calibrate `MIN_SIMILARITY` with this set.** The default (0.35) is a placeholder. Too low and
the app answers out-of-scope questions; too high and it refuses valid ones. Pick the value
that gives the best pass rate on your own questions.

## Tests

```bash
pytest          # no model downloads needed
```

## Project layout

```
app.py                  Streamlit UI
src/examrag/
  config.py             settings from env
  ingest.py             PDF -> pages -> unit tags -> chunks
  embeddings.py         sentence-transformers wrapper
  vectorstore.py        Chroma store with source/unit filters
  llm.py                Anthropic / Ollama backends
  rag.py                retrieval, threshold, exam-aware prompts
  pipeline.py           index a PDF
  evaluation.py         scoring helpers
eval/run_eval.py        evaluation CLI
tests/                  unit tests
docs/DEPLOYMENT.md      GitHub + deployment steps
```

## Known limitations

- **Scanned PDFs** have no text layer and are rejected with a clear message; run OCR first.
- **Tables, equations and diagrams** are extracted as plain text and may be garbled.
- **Shared index:** all users of one deployment share one document collection. Don't deploy
  publicly with private notes (see `docs/DEPLOYMENT.md`).
- **Unit detection** relies on headings like "Unit 3" / "Chapter 2"; other layouts are untagged.
- Answers can still be wrong; always check the cited page.

## Roadmap

- [ ] Past-paper analysis: which topics repeat across years
- [ ] Quiz generation per unit and weak-topic tracking
- [ ] OCR for scanned notes
- [ ] Per-session document isolation for public demos
- [ ] Hybrid search (BM25 + vectors) and a re-ranker
