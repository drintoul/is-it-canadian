# Is It Canadian?

An evidence-driven checker that vets whether a company is Canadian, powered by an
agentic AI workflow built on LangGraph. Enter a **company name** or paste a
**URL** — the agent finds the official website, scrapes the homepage and
corporate pages, then asks a local Ollama model to classify the company strictly
from the collected evidence. Every answer is verified by deterministic rules:
no evidence, no verdict.

Results stream back to a React UI in real time as the agent runs.

## What it does

- **Company name → website lookup** — SearXNG first, then Brave Search API, then
  Tavily Search API.
- **Candidate validation** — Ranks results by name-match score; rejects
  aggregators (Wikipedia, LinkedIn, app stores, review sites), non-production
  hosts (sandbox/staging/dev), portal subdomains (careers.*, shop.*), and deep
  pages — preferring the apex domain. Ambiguous matches surface a warning and
  alternate candidates.
- **Direct URL support** — Paste a URL (bare domains like `cbc.ca` work) to skip
  the lookup.
- **Resilient scraping** — Firecrawl with escalating retries (JS-render wait,
  then direct HTTP fetch). Thin-but-real homepages still proceed to evidence
  discovery.
- **Evidence discovery** — Finds corporate pages (About, Legal, Investors,
  Contact, History, Careers, Newsroom, Privacy) via homepage links and canonical
  path guesses, trying multiple path variants per type. If no page has identity
  signals, it also tries alternate candidate domains and searches for an
  external careers portal.
- **Evidence-driven classification** — A local Ollama model answers
  `Yes / No / Unclear` (plus `employs_canadians`) as strict JSON, citing
  verbatim evidence excerpts with source URLs.
- **Deterministic validation** — A `Yes` requires Canadian evidence, a `No`
  requires foreign evidence; unsupported or self-contradictory answers are
  downgraded to `Unclear`. Careers pages listing Canadian locations
  deterministically upgrade `employs_canadians` to `Yes`.
- **Streaming UI** — Three-column React UI: live Mermaid workflow diagram,
  per-step explanations, and the answer card with execution trace, warnings,
  and alternate candidates.

## How it works

```text
company name or URL
    ↓
Normalize Input
    ↓
SearXNG → Brave → Tavily  (search fallback chain)
    ↓
Validate Candidate  (name-match ranking, aggregator/host rejection)
    ↓
Scrape Homepage → Validate Scrape → Retry Scrape (waitFor, direct fetch)
    ↓
Assess Evidence → Discover Evidence Pages → Scrape Evidence Pages
    ↓                    (incl. alternate domains + careers-portal search)
Validate Evidence  (bundle with detected Canadian/foreign location signals)
    ↓
Classify (Ollama) → Validate Classification (deterministic rules)
    ↓
Terminal → Yes / No / Unclear + employs_canadians + cited evidence
```

The backend is a FastAPI app with a LangGraph graph that exposes:

- `POST /check` — synchronous result
- `POST /check/stream` — Server-Sent Events stream of graph updates
- `GET /graph/mermaid` — Mermaid definition of the workflow

## Stack

- **Backend**: FastAPI + LangGraph + httpx
- **Frontend**: React + Vite + Tailwind CSS
- **Search**: SearXNG container on the shared Docker network, Brave, Tavily
- **Scraper**: Firecrawl container on the shared Docker network
- **LLM**: Ollama (default model `llama3.1:8b`)

## Quick start

1. Ensure the shared Docker network exists:

```bash
docker network inspect app-network >/dev/null 2>&1 || docker network create app-network
```

2. Start the Firecrawl project and SearXNG container on that network.

3. Copy and adjust environment variables:

```bash
cp .env.example .env
```

   SearXNG is used first. If it returns no results, set `BRAVE_API_KEY` and/or
   `TAVILY_API_KEY` to enable the search API fallbacks.

4. Start this project:

```bash
docker compose up --build -d
```

5. Open http://localhost:5173 and enter a company name.

## API

- `GET /health` — health check
- `POST /check` — run the analysis synchronously
  - Body: `{ "company_name": "Shopify" }` or `{ "url": "https://www.shopify.com" }`
- `POST /check/stream` — stream graph updates via SSE
  - Body: `{ "company_name": "Shopify" }` or `{ "url": "https://www.shopify.com" }`
- `GET /graph/mermaid` — Mermaid diagram source

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `SEARXNG_BASE_URL` | `http://searxng:8080` | Primary search |
| `BRAVE_API_KEY` | — | Fallback search |
| `TAVILY_API_KEY` | — | Final fallback search |
| `FIRECRAWL_BASE_URL` | `http://firecrawl-api:3002` | Scraper endpoint |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model used for analysis |
| `LLM_NUM_CTX` | `16384` | Model context window |
| `LLM_NUM_PREDICT` | `2048` | Max output tokens |
| `LLM_CONTENT_BUDGET` | `20000` | Evidence bundle size cap |
| `MAX_SCRAPE_RETRIES` | `2` | Homepage scrape retry count |
| `MIN_SCRAPE_CONTENT_LENGTH` | `300` | Minimum usable scrape length |
