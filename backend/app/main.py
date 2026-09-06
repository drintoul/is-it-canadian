import json
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from typing import Optional

from pydantic import BaseModel

from .graph import build_graph

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Is It Canadian?", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()
logger = logging.getLogger(__name__)


class CheckRequest(BaseModel):
    company_name: Optional[str] = None
    url: Optional[str] = None


@app.get("/")
def root():
    return {"message": "Is It Canadian? API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph/mermaid", response_class=PlainTextResponse)
def graph_mermaid():
    return """graph TD
    normalize_input[\"Normalize Input\"]
    searxng[\"SearXNG\"]
    brave[\"Brave Search\"]
    tavily[\"Tavily Search\"]
    validate_candidate[\"Validate Candidate\"]
    scrape_homepage[\"Scrape Homepage\"]
    validate_scrape[\"Validate Scrape\"]
    retry_scrape[\"Retry Scrape\"]
    assess_evidence[\"Assess Evidence\"]
    discover_evidence[\"Discover Evidence Pages\"]
    scrape_evidence[\"Scrape Evidence Pages\"]
    validate_evidence[\"Validate Evidence\"]
    classify[\"Classify with Ollama\"]
    validate_classification[\"Validate Classification\"]
    terminal[\"Terminal\"]
    fail[\"✗ Any failure path<br/>(invalid input, no results,<br/>no candidate, retries exhausted,<br/>no usable evidence)\"]
    normalize_input -->|company name| searxng
    normalize_input -->|url supplied| scrape_homepage
    searxng -->|no usable candidate| brave
    brave -->|no usable candidate| tavily
    searxng --> validate_candidate
    brave --> validate_candidate
    tavily --> validate_candidate
    validate_candidate -->|valid| scrape_homepage
    scrape_homepage --> validate_scrape
    validate_scrape -->|invalid| retry_scrape
    retry_scrape --> validate_scrape
    validate_scrape -->|valid| assess_evidence
    validate_scrape -->|thin but real content| discover_evidence
    assess_evidence -->|insufficient| discover_evidence
    discover_evidence --> scrape_evidence
    scrape_evidence --> validate_evidence
    assess_evidence -->|sufficient| validate_evidence
    validate_evidence --> classify
    classify --> validate_classification
    validate_classification --> terminal
    fail -.-> terminal
"""


def _initial_state(req: CheckRequest) -> dict:
    return {
        "company_name": req.company_name or "",
        "input_url": req.url or "",
        "url": "",
        "content": "",
        "answer": "",
        "reasoning": "",
        "confidence": "",
        "employs_canadians": "",
        "classification_status": "",
        "trace": [],
        "errors": [],
        "warnings": [],
        "error": "",
    }


def _result_payload(result: dict) -> dict:
    return {
        "company_name": result.get("company_name", ""),
        "url": result.get("url", ""),
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", ""),
        "employs_canadians": result.get("employs_canadians", ""),
        "reasoning": result.get("reasoning", ""),
        "classification_status": result.get("classification_status", ""),
        "failure_type": result.get("failure_type", ""),
        "trace": result.get("trace", []),
        "warnings": result.get("warnings", []),
        "alternate_candidates": result.get("alternate_candidates", []),
        "error": result.get("error", ""),
        "content_length": len(result.get("content", "")),
    }


@app.post("/check")
def check(req: CheckRequest):
    try:
        result = graph.invoke(_initial_state(req))
        return _result_payload(result)
    except Exception as e:
        logger.exception("Graph invocation failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/check/stream")
def check_stream(req: CheckRequest):
    logger.info("[stream] company_name=%s url=%s", req.company_name, req.url)

    def event_generator():
        try:
            for chunk in graph.stream(
                _initial_state(req),
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    update = update or {}
                    logger.info("[stream] node=%s update_keys=%s", node, list(update.keys()))
                    payload = json.dumps({"node": node, "update": update})
                    yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'node': '__end__', 'done': True})}\n\n"
        except Exception as exc:
            logger.exception("Stream failed")
            yield f"data: {json.dumps({'node': '__error__', 'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
