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


def _mermaid_from_builder() -> str:
    """Render a Mermaid diagram from the graph's actual builder so the
    diagram can never drift from the real workflow.

    LangGraph's draw_mermaid() drops conditional edges whose targets are
    resolved at runtime, so we derive branch targets by scanning each
    router function's source for returned node names.
    """
    import ast
    import inspect
    import re

    builder = graph.builder
    hidden = {"terminate"}
    node_names = set(builder.nodes.keys()) - hidden

    labels = {
        "searxng": "SearXNG",
        "tavily": "Tavily Search",
        "brave": "Brave Search",
        "classify": "Classify with Ollama",
    }

    # Collect edges first so nodes and edges can be emitted in flow order —
    # Mermaid's layout (dagre) uses declaration order as a ranking hint, so
    # BFS ordering from the entry point minimizes line crossings.
    static_edges = []
    for source, target in builder.edges:
        if source == "__start__" or target == "__end__":
            continue
        if target in node_names:
            static_edges.append((source, target))

    # Extract branch targets from each router's source: `return "node"`,
    # `or "node"` / `else "node"` fallbacks, ternary `"a" if ... else "b"`,
    # and lookup-dict values. When the router is a closure bound to a key
    # of that dict (e.g. route_after_search("searxng")), resolve only the
    # mapped target instead of every value.
    literal_re = re.compile(
        r'return\s+"([a-z_]+)"|or\s+"([a-z_]+)"|else\s+"([a-z_]+)"|'
        r'"([a-z_]+)"\s+if'
    )
    dict_re = re.compile(r"\{[^{}]*:[^{}]*\}")

    def _closure_vars(func):
        vars_ = {}
        if func and getattr(func, "__closure__", None):
            for name, cell in zip(
                func.__code__.co_freevars, func.__closure__
            ):
                try:
                    vars_[name] = cell.cell_contents
                except ValueError:
                    pass
        return vars_

    cond_edges = []
    for source, spec in builder.branches.items():
        for branch in spec.values():
            func = getattr(branch.path, "func", None) or getattr(
                branch.path, "afunc", None
            )
            try:
                src = inspect.getsource(func or branch.path)
            except (OSError, TypeError):
                continue
            targets = {
                name
                for m in literal_re.finditer(src)
                for name in m.groups()
                if name and name in node_names and name != source
            }
            for dict_src in dict_re.findall(src):
                try:
                    mapping = ast.literal_eval(dict_src)
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(mapping, dict):
                    continue
                bound = {
                    v for v in _closure_vars(func).values()
                    if isinstance(v, str)
                }
                if bound:
                    # Router is bound to a specific lookup key (e.g. a
                    # search provider); a key absent from the map means
                    # the fallback chain ends there.
                    values = {mapping[k] for k in bound if k in mapping}
                else:
                    values = set(mapping.values())
                targets |= {
                    v for v in values
                    if v in node_names and v != source
                }
            for tgt in sorted(targets):
                cond_edges.append((source, tgt))

    # BFS from the entry point over static edges first, then conditional
    # edges, so declaration order follows the workflow's natural flow.
    adjacency = {}
    for source, target in static_edges + cond_edges:
        adjacency.setdefault(source, []).append(target)
    entry = next(
        (t for s, t in builder.edges if s == "__start__"),
        next(iter(builder.nodes), None),
    )
    order = []
    seen = set()
    queue = [entry] if entry else []
    while queue:
        node = queue.pop(0)
        if node in seen or node not in node_names:
            continue
        seen.add(node)
        order.append(node)
        queue.extend(adjacency.get(node, []))
    # Append any nodes unreachable from the entry point.
    order.extend(n for n in builder.nodes if n not in seen and n not in hidden)
    rank = {name: i for i, name in enumerate(order)}

    lines = ["graph TD"]
    for name in order:
        label = labels.get(name, name.replace("_", " ").title())
        lines.append(f'    {name}["{label}"]')
    for source, target in sorted(
        static_edges, key=lambda e: (rank.get(e[0], 999), rank.get(e[1], 999))
    ):
        lines.append(f"    {source} --> {target}")
    for source, target in sorted(
        cond_edges, key=lambda e: (rank.get(e[0], 999), rank.get(e[1], 999))
    ):
        lines.append(f"    {source} -.-> {target}")

    return "\n".join(lines) + "\n"


@app.get("/graph/mermaid", response_class=PlainTextResponse)
def graph_mermaid():
    return PlainTextResponse(
        _mermaid_from_builder(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "CDN-Cache-Control": "no-store",
            "Cloudflare-CDN-Cache-Control": "no-store",
        },
    )


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
