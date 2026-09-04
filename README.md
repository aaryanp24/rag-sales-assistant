# RAG Sales Assistant

A retrieval-augmented generation app that generates **company-specific cold-call
scripts, cold emails, pricing estimates, and deliverable recommendations**,
grounded in your own uploaded documents (pricing sheets, case studies, templates)
plus live research on the target company.

Built entirely on free-tier tools — no paid API required.

## Architecture

```
Your documents (PDF/DOCX/PPTX)      Target company name
        │                                   │
        ▼                                   ▼
   parse + chunk                    Tavily search (free tier)
   (ingestion.py)                    (research.py, cached)
        │                                   │
        ▼                                   │
 sentence-transformers embeddings           │
   (local, free, no API)                    │
        │                                   │
        ▼                                   │
  ChromaDB vector store                     │
  (one collection per org_id)               │
        │                                   │
        └──────────────┬────────────────────┘
                        ▼
              retrieval (filtered by
              org_id + doc_type)
                        │
                        ▼
          feature-specific prompt template
        (cold call / email / pricing / match)
                        │
                        ▼
              Groq or Gemini (free tier)
                        │
                        ▼
                 Streamlit UI
```

## Why it's structured this way (for your resume writeup / interview talking points)

- **Multi-tenant vector store**: each org gets its own ChromaDB collection
  (`org_<id>`), so retrieval never leaks one client's data into another's
  generated content — a real RAG design concern, not just a toy demo.
- **Filtered retrieval, not just similarity search**: pricing queries are
  filtered to `doc_type="pricing"` so a cold-call-script query can't
  accidentally pull rate-card numbers into a script, and vice versa.
- **One prompt template per feature**, not one generic prompt — each feature
  needs different context shape, tone, and constraints (see `prompts.py`).
- **Grounding constraint on pricing**: the pricing prompt is explicitly
  instructed to only use numbers that appear in retrieved context, and to
  say "not enough info" rather than invent a number. This is the most
  common RAG failure mode worth calling out that you thought about.
- **Swappable LLM provider**: `llm.py` abstracts Groq vs Gemini behind one
  `generate()` call, so the rest of the app doesn't care which free provider
  is behind it.

## Setup

```bash
cd rag-marketing
pip install -r requirements.txt
cp .env.example .env
# edit .env and add at least one free API key (see below)
streamlit run app.py
```

## Free API keys (pick at least one LLM provider)

| Service | Free tier | Get a key |
|---|---|---|
| **Groq** (recommended — fast, generous limits) | Yes | https://console.groq.com/keys |
| **Gemini** (alternative) | Yes | https://aistudio.google.com/apikey |
| **Tavily** (optional — company research) | 1,000 searches/mo free | https://app.tavily.com |

Without a Tavily key, the app still works — it just skips external company
research and generates from your uploaded docs alone.

Embeddings (`sentence-transformers`) run **locally on CPU**, no key needed —
the first run downloads a small (~90MB) model from Hugging Face.

## Project structure

```
config.py        # env var / settings loader
ingestion.py      # PDF/DOCX/PPTX parsing + chunking
vectorstore.py    # ChromaDB wrapper, multi-tenant, filtered retrieval
research.py       # Tavily company lookup, cached
llm.py            # Groq/Gemini abstraction
prompts.py        # one template per feature
app.py            # Streamlit UI
```

## Try it

1. Upload a fake "rate card" doc tagged `pricing` and a "case study" doc
   tagged `case_study` under org `demo_org`.
2. Go to the Pricing tab, describe a project, generate an estimate — note it
   only uses numbers from your uploaded doc.
3. Go to the Cold Call tab, enter a real company name — it researches them
   live and grounds the script in your case study.

## Known limitations (intentional, given project scope)

- SQLite/local-disk only — not designed for concurrent multi-user production use.
- Chunking is a simple sliding window on character count, not token- or
  semantic-boundary-aware.
- No auth — `org_id` is just a text field, anyone can type any org name.
  Fine for a demo, would need real auth for anything beyond that.
