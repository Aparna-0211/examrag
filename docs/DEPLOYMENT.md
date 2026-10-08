# From empty folder to deployed app

## 1. Put it on GitHub

```bash
cd examrag
git init -b main
git add .
git commit -m "Initial commit: ExamRAG"
```

Create an **empty** repo on github.com (no README/licence), then:

```bash
git remote add origin https://github.com/<you>/examrag.git
git push -u origin main
```

Never commit `.env` (it is in `.gitignore`). If a key is ever committed, revoke it and
create a new one; deleting the file is not enough because git keeps history.

CI (`.github/workflows/ci.yml`) runs the tests on every push.

## 2. Pick a deployment target

### A. Run locally (best for the offline story)

`streamlit run app.py`, or `docker compose up -d` for the Ollama stack.

### B. Hugging Face Spaces (free, good for a portfolio link)

1. New Space, SDK **Docker** (or Streamlit), hardware: CPU basic.
2. Push this repo to the Space (or connect the GitHub repo).
3. Space *Settings -> Variables and secrets*: add secret `ANTHROPIC_API_KEY`, and variables
   `LLM_PROVIDER=anthropic` and
   `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. For the Docker SDK, add `app_port: 8501` to the Space README front matter.

### C. Streamlit Community Cloud

1. share.streamlit.io -> New app -> pick the repo, branch `main`, file `app.py`.
2. *Advanced settings -> Secrets*:
   ```toml
   ANTHROPIC_API_KEY = "..."
   LLM_PROVIDER = "anthropic"
   EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
   ```
   (Streamlit exposes top-level secrets as environment variables.)

### D. Any VPS or Render/Railway

`docker build -t examrag . && docker run -p 8501:8501 --env-file .env examrag`

## 3. Things to know before sharing a public link

- **Free hosts have little RAM.** BGE-M3 is likely too big for them; use the small
  multilingual model above and run your eval to see what you lose in Tamil/Hindi.
- **Ollama will not run on free hosts.** Offline mode is for your own machine or a server
  with enough RAM/GPU. A public demo should use the hosted API.
- **Everyone shares one index.** Anyone who opens your public app can query documents
  uploaded by anyone else. For a public demo, pre-load a non-sensitive sample PDF and tell
  visitors not to upload private files, or keep the app private.
- **API cost.** A public app spends your API credits. Set a spending limit on the key.
- **Ephemeral storage.** Free hosts wipe `data/` on restart, so documents must be re-indexed.
