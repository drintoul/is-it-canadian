import html
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, List, Optional, Tuple, TypedDict
from urllib.parse import urljoin, urlparse

import httpx
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from .prompts import ANALYSIS_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIRECRAWL_BASE_URL = os.getenv("FIRECRAWL_BASE_URL", "http://firecrawl-api:3002")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_TIMEOUT = int(os.getenv("FIRECRAWL_TIMEOUT", "60"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://searxng:8080")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

MIN_SCRAPE_CONTENT_LENGTH = int(os.getenv("MIN_SCRAPE_CONTENT_LENGTH", "300"))
MAX_SCRAPE_RETRIES = int(os.getenv("MAX_SCRAPE_RETRIES", "2"))
MAX_EVIDENCE_PAGES = int(os.getenv("MAX_EVIDENCE_PAGES", "8"))
MAX_SEARCH_CANDIDATES = int(os.getenv("MAX_SEARCH_CANDIDATES", "10"))
LLM_CONTENT_BUDGET = int(os.getenv("LLM_CONTENT_BUDGET", "8000"))
REDIRECT_TIMEOUT = int(os.getenv("REDIRECT_TIMEOUT", "10"))

USER_AGENT = "Mozilla/5.0 (compatible; IsItCanadian/1.0)"


# ---------------------------------------------------------------------------
# Error / failure categories
# ---------------------------------------------------------------------------


class FailureType:
    SEARCH_FAILED = "search_failed"
    NO_OFFICIAL_URL = "no_official_url"
    URL_INVALID = "url_invalid"
    REDIRECT_FAILED = "redirect_failed"
    SCRAPE_NETWORK_ERROR = "scrape_network_error"
    SCRAPE_HTTP_ERROR = "scrape_http_error"
    SCRAPE_EMPTY = "scrape_empty"
    SCRAPE_TOO_SHORT = "scrape_too_short"
    SCRAPE_ACCESS_DENIED = "scrape_access_denied"
    SCRAPE_JS_ONLY = "scrape_js_only"
    SCRAPE_ERROR_PAGE = "scrape_error_page"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    LLM_ERROR = "llm_error"
    LLM_INVALID_OUTPUT = "llm_invalid_output"
    CLASSIFICATION_VALIDATION_FAILED = "classification_validation_failed"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _concat(a: Optional[list], b: Optional[list]) -> list:
    return (a or []) + (b or [])


class EvidenceItem(TypedDict, total=False):
    claim: str
    source_url: str
    quote_or_excerpt: str


class EvidencePage(TypedDict, total=False):
    url: str
    page_type: str
    content: str
    content_length: int


class GraphState(TypedDict, total=False):
    # Inputs
    company_name: str
    input_url: str
    url: str  # selected/final URL (kept for API + frontend compatibility)

    # URL handling
    normalized_url: str
    final_url: str
    redirect_chain: List[str]

    # Discovery
    search_results: List[str]
    search_provider: str
    selected_candidate: str
    candidate_validation: str
    alternate_candidates: List[str]
    alternates_exhausted: bool

    # Retrieval
    content: str  # homepage content (kept for compatibility)
    homepage_status: int
    scrape_valid: bool
    scrape_attempts: int
    scrape_failure_reason: str

    # Evidence
    evidence_pages: List[EvidencePage]
    canadian_evidence: List[EvidenceItem]
    non_canadian_evidence: List[EvidenceItem]
    employment_evidence: List[EvidenceItem]

    # Classification
    classification_status: str  # "performed" | "not_performed"
    failure_type: str
    answer: str
    confidence: str
    employs_canadians: str
    reasoning: str

    # Diagnostics
    errors: Annotated[List[str], _concat]
    warnings: Annotated[List[str], _concat]
    trace: Annotated[List[str], _concat]
    error: str  # top-level error string (frontend compatibility)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

AGGREGATOR_HOSTS = (
    "wikipedia.org",
    "wikidata.org",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "reddit.com",
    "pinterest.com",
    "crunchbase.com",
    "bloomberg.com",
    "glassdoor.",
    "indeed.",
    "yelp.com",
    "tripadvisor.",
    "trustpilot.com",
    "foursquare.com",
    "mapquest.com",
    "manta.com",
    "bbb.org",
    "dnb.com",
    "zoominfo.com",
    "owler.com",
    "yellowpages.",
    "amazon.",
    "ebay.",
    "walmart.",
    "play.google.com",
    "apps.apple.com",
    "store.google.com",
    "chromewebstore.google.com",
    # Complaint / review aggregators
    "pissedconsumer.com",
    "complaintsboard.com",
    "consumeraffairs.com",
    "sitejabber.com",
    "ripoffreport.com",
    # Encyclopedias / reference
    "britannica.com",
    "thecanadianencyclopedia.ca",
    "encyclopedia.com",
    # Job aggregators
    "ziprecruiter.com",
    "monster.com",
    "workopolis.com",
    "simplyhired.com",
    "careerbuilder.com",
    "job-applications.com",
    # Form/document hosts — not corporate sites or real careers portals
    "pdffiller.com",
    "signnow.com",
    "jotform.com",
    "docs.google.com",
    "forms.gle",
)

# Subdomain labels that indicate a non-production environment — never an
# official company site (e.g. arbysca.sandbox.opentender.io).
NON_PRODUCTION_SUBDOMAINS = {
    "sandbox", "staging", "stage", "dev", "test", "testing", "demo",
    "beta", "qa", "uat", "preview", "poc",
}

TRACKING_HOST_HINTS = (
    "google.com/url",
    "l.facebook.com",
    "t.co",
    "bit.ly",
    "goo.gl",
    "ow.ly",
    "tinyurl.com",
)


def normalize_url(raw: str) -> str:
    """Return a normalized absolute https URL, or '' if the input is invalid.

    Always upgrades to https and adds a www. prefix for bare apex domains
    (e.g. ``example.com`` -> ``https://www.example.com/``). Hosts that already
    have a subdomain are left untouched.
    """
    if not raw:
        return ""
    candidate = html.unescape(raw.strip())
    if not candidate:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate):
        candidate = "https://" + candidate
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    host, _, port = parsed.netloc.lower().partition(":")
    # Add www. only for bare apex domains (two labels), not subdomains.
    if not host.startswith("www.") and host.count(".") == 1:
        host = f"www.{host}"
    netloc = host + (f":{port}" if port else "")
    path = parsed.path or "/"
    normalized = f"https://{netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def resolve_redirects(url: str) -> Tuple[str, List[str]]:
    """Follow redirects and return (final_url, redirect_chain). Non-fatal."""
    chain: List[str] = []
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=REDIRECT_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(url)
            for hop in response.history:
                chain.append(str(hop.url))
            final = str(response.url)
            chain.append(final)
            return final, chain
    except Exception as exc:
        logger.warning("Redirect resolution failed for %s: %s", url, exc)
        return url, chain


def validate_candidate_url(url: str) -> Tuple[bool, str]:
    """Decide whether a search result looks like an official company site."""
    if not url:
        return False, "empty URL"
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "malformed URL"
    host = (parsed.netloc or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return False, "malformed URL"
    if any(hint in url.lower() for hint in TRACKING_HOST_HINTS):
        return False, "tracking/redirect URL"
    if any(host == h or host.endswith("." + h) or h in host for h in AGGREGATOR_HOSTS):
        return False, "aggregator/social/directory/news result"
    labels = host.removeprefix("www.").split(".")
    if len(labels) > 2 and any(l in NON_PRODUCTION_SUBDOMAINS for l in labels[:-2]):
        return False, "non-production environment host"
    return True, "looks like an official company site"


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    """Crude but dependency-free HTML-to-text fallback."""
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _firecrawl_scrape(url: str, formats=("markdown",), **options) -> Tuple[str, int]:
    """Scrape a URL via Firecrawl. Returns (content, page_status).

    Extra keyword arguments are merged into the Firecrawl request payload
    (e.g. ``waitFor``, ``onlyMainContent``, ``timeout``).
    """
    headers = {}
    if FIRECRAWL_API_KEY:
        headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"

    payload = {"url": url, "formats": list(formats)}
    payload.update(options)
    endpoint = f"{FIRECRAWL_BASE_URL.rstrip('/')}/v1/scrape"

    response = httpx.post(endpoint, json=payload, headers=headers, timeout=FIRECRAWL_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    payload_data = data.get("data", {}) or {}
    metadata = payload_data.get("metadata", {}) or {}
    status = metadata.get("statusCode") or data.get("statusCode") or 200

    content = payload_data.get("markdown") or payload_data.get("content") or ""
    if not content and payload_data.get("html"):
        content = _html_to_text(payload_data["html"])
    return content or "", int(status)


_META_REFRESH = re.compile(
    r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\s*\d+\s*;\s*url=([^"\'>\s]+)',
    re.IGNORECASE,
)


def _firecrawl_map(url: str, limit: int = 200) -> List[str]:
    """List a site's URLs via Firecrawl's /v1/map endpoint.

    Returns discovered links (same-site only), or [] on failure — callers
    should fall back to guessed paths.
    """
    headers = {}
    if FIRECRAWL_API_KEY:
        headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"
    endpoint = f"{FIRECRAWL_BASE_URL.rstrip('/')}/v1/map"
    try:
        response = httpx.post(
            endpoint,
            json={"url": url, "limit": limit},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        links = data.get("links") or []
        base_host = (urlparse(url).netloc or "").lower().removeprefix("www.")
        out: List[str] = []
        for link in links:
            if not isinstance(link, str):
                link = link.get("url", "") if isinstance(link, dict) else ""
            host = (urlparse(link).netloc or "").lower().removeprefix("www.")
            if link and host == base_host:
                out.append(link.split("#")[0])
        return out
    except Exception as exc:
        logger.warning("Firecrawl map failed for %s: %s", url, exc)
        return []


def _wikipedia_summary(company_name: str) -> Tuple[str, str]:
    """Fetch the Wikipedia lead for a company as supplementary identity
    evidence. Returns (content, article_url), or ('', '') on failure or
    when no matching article exists."""
    if not company_name:
        return "", ""
    try:
        # Find the best-matching article title via the search API.
        r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": company_name,
                "srlimit": 3, "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return "", ""
        title = results[0]["title"]
        article_url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        # Fetch the plain-text lead extract.
        r2 = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts", "exintro": True,
                "explaintext": True, "titles": title, "format": "json",
                "redirects": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        r2.raise_for_status()
        pages = r2.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                # Wikipedia leads often contain IPA pronunciations in parens
                # (e.g. "Yahoo! (/ˈjɑːhuː/)") — when non-ASCII is stripped
                # downstream they collapse to "()". Remove empty or
                # punctuation-only parentheticals so quotes stay clean.
                extract = re.sub(r"\(\s*[^A-Za-z0-9()]*\s*\)", "", extract)
                extract = re.sub(r"  +", " ", extract)
                return f"Wikipedia article '{title}':\n{extract}", article_url
        return "", ""
    except Exception as exc:
        logger.warning("Wikipedia lookup failed for %s: %s", company_name, exc)
        return "", ""


def _direct_fetch(url: str, _hops: int = 0) -> Tuple[str, int]:
    """Direct HTTP fetch fallback. Returns (text, status).

    Follows <meta http-equiv="refresh"> redirects, which httpx does not —
    some sites (e.g. rbcroyalbank.com) serve a stub page that meta-refreshes
    to the real homepage.
    """
    response = httpx.get(
        url,
        timeout=30,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )
    response.raise_for_status()
    text = _html_to_text(response.text)
    if len(text) < 100 and _hops < 2:
        m = _META_REFRESH.search(response.text[:5000])
        if m:
            target = urljoin(str(response.url), m.group(1))
            if target != url:
                return _direct_fetch(target, _hops + 1)
    return text, response.status_code


ERROR_PAGE_PATTERNS = [
    (r"enable javascript|javascript is required|javascript is disabled|"
     r"please (enable|turn on) javascript|needs javascript|javascript to run",
     FailureType.SCRAPE_JS_ONLY),
    (r"captcha|are you a robot|verify you are human|human verification|"
     r"recaptcha|hcaptcha",
     FailureType.SCRAPE_ACCESS_DENIED),
    (r"access denied|access to this page has been denied|request blocked|"
     r"bot detection|security check|attention required|just a moment|"
     r"cloudflare|cf-ray|403 forbidden|error 403|rate limit",
     FailureType.SCRAPE_ACCESS_DENIED),
    (r"404 not found|page not found|page doesn'?t exist|"
     r"this page could not be found|error 404",
     FailureType.SCRAPE_ERROR_PAGE),
]


def is_error_page(content: str) -> Optional[str]:
    """Return a FailureType if the content looks like an error/challenge page."""
    sample = content[:4000].lower()
    for pattern, failure in ERROR_PAGE_PATTERNS:
        if re.search(pattern, sample):
            return failure
    return None


def validate_scrape(content, status: Optional[int] = None) -> Tuple[bool, str]:
    """Deterministically decide whether scraped content is usable."""
    if content is None:
        return False, FailureType.SCRAPE_EMPTY
    if not isinstance(content, str):
        try:
            content = str(content)
        except Exception:
            return False, FailureType.SCRAPE_EMPTY
    stripped = content.strip()
    if not stripped:
        return False, FailureType.SCRAPE_EMPTY
    if status and status >= 400:
        return False, FailureType.SCRAPE_HTTP_ERROR
    # Check error/JS-only patterns before length — a JS shell or captcha page
    # is short, but "scrape_too_short" would mislabel the real problem.
    error_kind = is_error_page(stripped)
    if error_kind:
        return False, error_kind
    if len(stripped) < MIN_SCRAPE_CONTENT_LENGTH:
        return False, FailureType.SCRAPE_TOO_SHORT
    if len(stripped.split()) < 40:
        return False, FailureType.SCRAPE_TOO_SHORT
    return True, ""


# ---------------------------------------------------------------------------
# Search providers (return candidate URL lists)
# ---------------------------------------------------------------------------


def _search_searxng(company_name: str) -> List[str]:
    base = SEARXNG_BASE_URL.rstrip("/")
    endpoint = f"{base}/search"
    queries = [
        f'"{company_name}" official website',
        f"{company_name} official website",
        company_name,
    ]
    candidates: List[str] = []
    seen = set()
    for query in queries:
        try:
            response = httpx.get(
                endpoint,
                params={"q": query, "format": "json", "language": "en-US"},
                timeout=30,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            for result in response.json().get("results", []):
                url = (result.get("url") or "").strip()
                if url and url not in seen:
                    seen.add(url)
                    candidates.append(url)
                    if len(candidates) >= MAX_SEARCH_CANDIDATES:
                        return candidates
        except Exception as exc:
            logger.warning("SearXNG query failed for '%s': %s", query, exc)
            continue
    return candidates


def _search_brave(company_name: str) -> List[str]:
    if not BRAVE_API_KEY:
        raise RuntimeError("Brave API key not configured")
    response = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={
            "q": f'"{company_name}" official website',
            "count": MAX_SEARCH_CANDIDATES,
            "offset": 0,
            "text_decorations": False,
        },
        headers={
            "X-Subscription-Token": BRAVE_API_KEY,
            "Accept": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("web", {}).get("results", [])
    return [r["url"].strip() for r in results if r.get("url")]


def _search_tavily(company_name: str) -> List[str]:
    if not TAVILY_API_KEY:
        raise RuntimeError("Tavily API key not configured")
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": f'"{company_name}" official website',
            "search_depth": "basic",
            "max_results": MAX_SEARCH_CANDIDATES,
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return [r["url"].strip() for r in results if r.get("url")]


# ---------------------------------------------------------------------------
# Evidence discovery / bundling
# ---------------------------------------------------------------------------

# Ordered by corporate-identity relevance.
EVIDENCE_PATH_PRIORITY = [
    ("about", ("about", "about-us", "our-company", "company", "who-we-are", "our-story", "corporate", "corporate-info", "our-business")),
    ("legal", ("legal", "terms", "terms-of-service", "terms-of-use", "terms-and-conditions", "imprint", "impressum", "legal-notice", "corporate-governance", "governance")),
    ("investors", ("investor", "investors", "investor-relations", "ir", "sec-filings", "financials", "financial-information")),
    ("contact", ("contact", "contact-us", "head-office", "offices", "locations", "our-locations", "find-us")),
    ("history", ("history", "heritage", "our-history", "timeline", "milestones")),
    ("careers", ("careers", "career", "jobs", "job", "work-with-us", "join-us", "join-our-team", "employment", "career-opportunities", "opportunities", "work-here")),
    ("newsroom", ("newsroom", "press", "media", "news", "press-releases")),
    ("privacy", ("privacy", "privacy-policy", "privacy-notice")),
]

STRONG_IDENTITY_PATTERNS = [
    r"headquartered in",
    r"headquarters",
    r"head office",
    r"incorporated in",
    r"incorporation",
    r"canadian (company|corporation|business)",
    r"(company|corporation) (based|headquartered|founded|incorporated) in",
    r"founded in",
    r"registered office",
]


def _extract_links(markdown: str, base_url: str) -> List[str]:
    links = re.findall(r"\]\(\s*(https?://[^\s)]+|/[^\s)]+)", markdown)
    base_host = (urlparse(base_url).netloc or "").lower().removeprefix("www.")
    out: List[str] = []
    for link in links:
        absolute = urljoin(base_url, link).split("#")[0]
        host = (urlparse(absolute).netloc or "").lower().removeprefix("www.")
        if host == base_host:
            out.append(absolute)
    return out


# Canonical paths to try when the homepage doesn't link an obvious page of
# that type. Ordered to mirror EVIDENCE_PATH_PRIORITY.
GUESSED_EVIDENCE_PATHS = [
    ("about", ("/about", "/about-us", "/company", "/our-company", "/who-we-are", "/our-story", "/corporate")),
    ("legal", ("/legal", "/terms", "/terms-of-service", "/terms-of-use", "/corporate-governance")),
    ("investors", ("/investors", "/investor-relations", "/ir", "/financials")),
    ("contact", ("/contact", "/contact-us", "/locations", "/offices")),
    ("history", ("/history", "/our-history", "/heritage")),
    ("careers", ("/careers", "/jobs", "/career-opportunities", "/work-with-us", "/join-us", "/employment")),
    ("newsroom", ("/newsroom", "/press", "/media")),
    ("privacy", ("/privacy", "/privacy-policy")),
]


def _path_matches_type(path: str, keywords) -> bool:
    """Match keywords against path segments, not raw substrings.

    ``/en-us/company/insights/foo`` should not win over ``/about`` just because
    it contains the substring ``company`` — the segment must be (or start with)
    a keyword, and deeper paths are deprioritized by the caller.
    """
    segments = [
        re.sub(r"\.(html?|php|aspx?|jsp)$", "", s)
        for s in path.lower().split("/")
        if s
    ]
    for segment in segments:
        for keyword in keywords:
            if segment == keyword or segment.startswith(keyword + "-") or segment.startswith(keyword + "_"):
                return True
    return False


def discover_evidence_pages(homepage_content: str, homepage_url: str) -> List[EvidencePage]:
    """Find likely corporate-evidence pages linked from the homepage.

    Discovered links are ranked by (page-type priority, path depth) so shallow
    canonical pages beat deep blog/article URLs. High-priority page types that
    were not discovered are supplemented with guessed canonical paths.
    """
    # Prefer Firecrawl's site map — real URLs beat homepage-link extraction
    # and path guessing. Fall back to homepage links when the map is empty.
    links = _firecrawl_map(homepage_url)
    if not links:
        links = _extract_links(homepage_content, homepage_url)
    # Locale preference: Canadian locales and unlocalized paths first, foreign
    # locales (uk-en, in-en, …) last — a Canadian checker shouldn't read the UK
    # about page when a generic or /en-ca/ one exists.
    def _locale_rank(path: str) -> int:
        first = (path.lstrip("/").split("/") or [""])[0]
        if first in ("en-ca", "fr-ca", "ca-en", "ca-fr"):
            return 0
        if re.fullmatch(r"[a-z]{2}-[a-z]{2}", first):
            return 2
        return 1

    shallow = []  # depth <= 2: likely real nav/corporate pages
    deep = []     # deeper: likely blog/article/case-study pages
    for link in links:
        path = urlparse(link).path.lower().rstrip("/")
        if not path:
            continue
        depth = path.count("/")
        for priority, (page_type, keywords) in enumerate(EVIDENCE_PATH_PRIORITY):
            if _path_matches_type(path, keywords):
                (shallow if depth <= 2 else deep).append(
                    (priority, _locale_rank(path), depth, link, page_type)
                )
                break

    shallow.sort(key=lambda item: (item[0], item[1], item[2]))
    deep.sort(key=lambda item: (item[0], item[1], item[2]))

    pages: List[EvidencePage] = []
    seen = set()
    covered_types = set()
    # Candidates may exceed MAX_EVIDENCE_PAGES — scrape_evidence validates each
    # and stops once it has enough usable pages.
    candidate_cap = MAX_EVIDENCE_PAGES * 3

    def _add(link: str, page_type: str, covers: bool) -> bool:
        if link in seen or len(pages) >= candidate_cap:
            return False
        seen.add(link)
        if covers:
            covered_types.add(page_type)
        pages.append({"url": link, "page_type": page_type})
        return True

    # 1. Shallow discovered links (real corporate pages) — at most 2 per type,
    # interleaved so the first page of each type precedes second pages. This
    # keeps the scrape budget covering every type instead of exhausting it on
    # high-priority types (e.g. dozens of */terms-and-conditions promos).
    by_type: dict = {}
    for _p, _l, _d, link, page_type in shallow:
        by_type.setdefault(page_type, []).append(link)
    for round_idx in range(2):
        for page_type, _kw in EVIDENCE_PATH_PRIORITY:
            links_for_type = by_type.get(page_type, [])
            if round_idx < len(links_for_type):
                _add(links_for_type[round_idx], page_type, covers=True)

    # 2. Guessed canonical paths for uncovered high-priority types. Emit ALL
    # candidate paths per type — scrape_evidence validates each and stops once
    # it has enough usable pages, so a 404 on /about doesn't block /about-us.
    # Use the domain root so localized homepages (e.g. /en-us/) don't produce
    # paths like /en-us/about that may not exist.
    parsed_home = urlparse(homepage_url)
    base = f"{parsed_home.scheme}://{parsed_home.netloc}"
    # Round-robin across types: emit the first path of each uncovered type,
    # then the second, etc. — so the cap can't starve lower-priority types
    # (e.g. careers) when earlier types have many path variants.
    uncovered = [
        (page_type, list(paths))
        for page_type, paths in GUESSED_EVIDENCE_PATHS
        if page_type not in covered_types
    ]
    idx = 0
    while uncovered and len(pages) < candidate_cap:
        remaining = []
        for page_type, paths in uncovered:
            if idx < len(paths) and len(pages) < candidate_cap:
                _add(base + paths[idx], page_type, covers=True)
            if idx + 1 < len(paths):
                remaining.append((page_type, paths))
        uncovered = remaining
        idx += 1

    # 3. Deep discovered links fill any remaining slots.
    for _p, _l, _d, link, page_type in deep:
        if len(pages) >= candidate_cap:
            break
        _add(link, page_type, covers=False)

    return pages


# Canadian locations used to detect employment/presence signals in scraped
# content (especially careers pages). Long names are matched case-insensitively;
# two-letter province codes require word boundaries to avoid false positives.
CANADIAN_CITY_PATTERNS = [
    "toronto", "vancouver", "montreal", "montréal", "calgary", "ottawa",
    "edmonton", "winnipeg", "quebec city", "québec", "waterloo", "mississauga",
    "burnaby", "victoria", "halifax", "london, on", "kitchener", "saskatoon",
    "regina", "kelowna", "surrey", "brampton", "hamilton", "laval",
]

CANADIAN_PROVINCE_PATTERNS = [
    "ontario", "quebec", "québec", "british columbia", "alberta", "manitoba",
    "saskatchewan", "nova scotia", "new brunswick", "newfoundland",
    "prince edward island", "yukon", "northwest territories", "nunavut",
]

# Province/state abbreviations — matched with surrounding non-letters.
CANADIAN_PROVINCE_CODES = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "YT", "NT", "NU"]

CANADIAN_GENERAL_PATTERNS = [
    r"\bcanada\b", r"\bcanadian\b", r"\bcanada[- ]wide\b", r"\bremote\s*[-–—]\s*canada\b",
]


# Foreign locations used to detect non-Canadian corporate/presence signals.
# US states/cities plus common foreign country mentions.
FOREIGN_CITY_PATTERNS = [
    "new york", "san francisco", "los angeles", "chicago", "boston", "seattle",
    "austin", "dallas", "houston", "atlanta", "denver", "miami", "westlake",
    "menlo park", "mountain view", "palo alto", "cupertino", "redmond",
    "london", "berlin", "munich", "paris", "tokyo", "sydney", "singapore",
    "hong kong", "dublin", "amsterdam", "frankfurt", "zurich", "beijing",
    "shanghai", "bangalore", "mumbai", "mexico city", "são paulo", "sao paulo",
]

FOREIGN_REGION_PATTERNS = [
    r"\bunited states\b", r"\bu\.s\.a?\.?\b", r"\bamerica(n)?\b",
    r"\bgermany\b", r"\bgerman\b", r"\bunited kingdom\b", r"\bu\.k\.\b",
    r"\bengland\b", r"\bfrance\b", r"\bfrench\b", r"\bjapan\b", r"\bjapanese\b",
    r"\bchina\b", r"\bchinese\b", r"\bindia\b", r"\baustralia\b",
    r"\bswitzerland\b", r"\bswiss\b", r"\bnetherlands\b", r"\bdutch\b",
    r"\bireland\b", r"\birish\b", r"\bsweden\b", r"\bswedish\b",
    r"\bsouth korea\b", r"\bkorean\b", r"\btaiwan\b", r"\bmexico\b",
    r"\bbrazil\b", r"\bitaly\b", r"\bitalian\b", r"\bspain\b", r"\bspanish\b",
]

US_STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
]


def detect_foreign_signals(content: str) -> List[str]:
    """Return distinct non-Canadian location signals found in page content."""
    if not content:
        return []
    lower = content.lower()
    found = set()
    for name in FOREIGN_CITY_PATTERNS:
        if name in lower:
            found.add(name.title())
    for pattern in FOREIGN_REGION_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            found.add(m.group(0).title())
    for code in US_STATE_CODES:
        if re.search(rf"(?<![A-Za-z]){code}(?![A-Za-z])", content):
            found.add(code)
    return sorted(found)


def detect_canadian_signals(content: str) -> List[str]:
    """Return the distinct Canadian location signals found in page content."""
    if not content:
        return []
    lower = content.lower()
    found = set()
    for name in CANADIAN_CITY_PATTERNS + CANADIAN_PROVINCE_PATTERNS:
        if name in lower:
            found.add(name.title())
    for code in CANADIAN_PROVINCE_CODES:
        if re.search(rf"(?<![A-Za-z]){code}(?![A-Za-z])", content):
            found.add(code)
    for pattern in CANADIAN_GENERAL_PATTERNS:
        if re.search(pattern, lower):
            found.add("Canada")
    return sorted(found)


def assess_evidence_sufficiency(content: str) -> bool:
    """Heuristic: does the homepage already contain corporate-identity signals?"""
    lower = content.lower()
    return any(re.search(pattern, lower) for pattern in STRONG_IDENTITY_PATTERNS)


def build_evidence_bundle(pages: List[EvidencePage]) -> str:
    """Format evidence pages with source markers for the LLM prompt."""
    parts: List[str] = []
    budget = LLM_CONTENT_BUDGET
    # Reference pages (e.g. Wikipedia) carry the identity statement — put them
    # first so the content budget can't truncate them away.
    ordered = sorted(pages, key=lambda p: 0 if p.get("page_type") == "reference" else 1)
    for page in ordered:
        content = page.get("content", "")
        signals = detect_canadian_signals(content)
        foreign = detect_foreign_signals(content)
        signal_note = ""
        if signals:
            signal_note += (
                f"\n[CONTEXT: this page mentions Canadian locations: "
                f"{', '.join(signals)}. Location mentions alone are NOT corporate "
                f"identity evidence — they may indicate employment or operations only.]"
            )
        if foreign:
            signal_note += (
                f"\n[CONTEXT: this page mentions non-Canadian locations: "
                f"{', '.join(foreign)}. Location mentions alone are NOT corporate "
                f"identity evidence — they may indicate employment or operations only.]"
            )
        header = (
            f"\n\n===== SOURCE: {page['url']} (type: {page.get('page_type', 'page')}) ====="
            f"{signal_note}\n"
        )
        if budget <= 0:
            break
        chunk = content[:budget]
        parts.append(header + chunk)
        budget -= len(chunk) + len(header)
    return "".join(parts)


# ---------------------------------------------------------------------------
# LLM output parsing / validation
# ---------------------------------------------------------------------------


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from LLM output.

    Robust to markdown fences, surrounding prose, trailing commas, and
    truncated output (attempts brace-balance repair).
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)

    def _try_load(candidate: str) -> Optional[dict]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None

    # Complete object: first { ... } span.
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        result = _try_load(match.group(0))
        if result is not None:
            return result

    # Truncated output: an opening brace with no closing brace.
    start = cleaned.find("{")
    if start == -1:
        return None
    fragment = cleaned[start:]
    # Close any open string, then balance brackets/braces.
    if fragment.count('"') % 2 == 1:
        fragment += '"'
    open_brackets = fragment.count("[") - fragment.count("]")
    open_braces = fragment.count("{") - fragment.count("}")
    fragment = fragment.rstrip(", \n\t")
    fragment += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    return _try_load(fragment)


def _normalize_choice(value, allowed, default):
    text = str(value or "").strip().lower()
    for option in allowed:
        if text == option.lower():
            return option
    return default


def _normalize_evidence_list(value) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    if not isinstance(value, list):
        return items
    for entry in value:
        if isinstance(entry, dict):
            items.append(
                {
                    "claim": str(entry.get("claim", "")),
                    "source_url": str(entry.get("source_url", "")),
                    "quote_or_excerpt": str(entry.get("quote_or_excerpt", "")),
                }
            )
        elif isinstance(entry, str) and entry.strip():
            items.append({"claim": entry.strip(), "source_url": "", "quote_or_excerpt": ""})
    return items


LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "2048"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "16384"))


def _create_llm():
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=0.0,
        num_predict=LLM_NUM_PREDICT,
        num_ctx=LLM_NUM_CTX,
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def normalize_input_node(state: GraphState) -> dict:
    """Normalize a user-supplied URL, or pass through to search."""
    raw = (state.get("input_url") or state.get("url") or "").strip()
    company = (state.get("company_name") or "").strip()

    if not raw and not company:
        return {
            "classification_status": "not_performed",
            "failure_type": FailureType.URL_INVALID,
            "error": "Provide a company name or a URL.",
            "errors": ["No company name or URL supplied"],
            "trace": ["✗ No company name or URL supplied"],
        }

    if not raw:
        return {"trace": [f"Searching for official site of '{company}'"]}

    normalized = normalize_url(raw)
    if not normalized:
        return {
            "classification_status": "not_performed",
            "failure_type": FailureType.URL_INVALID,
            "error": f"Invalid URL: {raw}",
            "errors": [f"Invalid URL supplied: {raw}"],
            "trace": [f"✗ Input URL invalid: {raw}"],
        }

    final_url, chain = resolve_redirects(normalized)
    trace = [f"✓ Input URL normalized\n  {raw} → {final_url}"]
    if chain and chain[-1] != normalized:
        trace.append(f"  Redirect chain: {' → '.join(chain)}")
    return {
        "normalized_url": normalized,
        "final_url": final_url,
        "redirect_chain": chain,
        "selected_candidate": final_url,
        "candidate_validation": "user-supplied URL",
        "url": final_url,
        "trace": trace,
    }


def _search_node(state: GraphState, provider: str, search_fn) -> dict:
    company = (state.get("company_name") or "").strip()
    label = {"searxng": "SearXNG", "brave": "Brave Search", "tavily": "Tavily Search"}[provider]
    trace = [f"Searching {label} for '{company}'"]
    try:
        candidates = search_fn(company)
    except Exception as exc:
        logger.warning("%s lookup failed: %s", label, exc)
        trace.append(f"{label} failed: {exc}")
        update = {"search_results": [], "search_provider": provider, "trace": trace}
        if provider == "tavily":
            update.update(
                {
                    "classification_status": "not_performed",
                    "failure_type": FailureType.SEARCH_FAILED,
                    "error": f"All search providers failed. Last error: {exc}",
                    "errors": [f"{label}: {exc}"],
                }
            )
        return update

    if not candidates:
        trace.append(f"{label} returned no results")
        update = {"search_results": [], "search_provider": provider, "trace": trace}
        if provider == "tavily":
            update.update(
                {
                    "classification_status": "not_performed",
                    "failure_type": FailureType.SEARCH_FAILED,
                    "error": "All search providers returned no usable results.",
                    "errors": ["All search providers returned no usable results"],
                }
            )
        return update

    trace.append(f"{label} returned {len(candidates)} candidate(s)")
    return {"search_results": candidates, "search_provider": provider, "trace": trace}


def searxng_node(state: GraphState) -> dict:
    return _search_node(state, "searxng", _search_searxng)


def brave_node(state: GraphState) -> dict:
    return _search_node(state, "brave", _search_brave)


def tavily_node(state: GraphState) -> dict:
    return _search_node(state, "tavily", _search_tavily)


# Generic tokens that don't help distinguish a company name.
_NAME_STOPWORDS = {
    "the", "a", "an", "inc", "incorporated", "corp", "corporation", "co",
    "company", "ltd", "limited", "llc", "group", "holdings", "of", "and",
}


# Subdomains that are section portals, not the corporate homepage. A candidate
# on one of these is deprioritized in favor of the apex domain.
_NON_CORPORATE_SUBDOMAINS = {
    "careers", "jobs", "shop", "store", "blog", "news", "help", "support",
    "docs", "developer", "developers", "community", "forum", "forums",
    "app", "apps", "my", "account", "login", "mail",
}


def _name_match_score(company_name: str, url: str) -> float:
    """Fraction of the query's distinctive tokens found in the candidate domain.

    ``Kimberton`` vs ``kimbertonwholefoods.com`` -> 1.0 (token present).
    ``Whole Foods`` vs ``kimbertonwholefoods.com`` -> 0.5 (only 'whole'/'foods'
    partially). Used to rank candidates and flag ambiguous matches.
    """
    tokens = {
        t for t in re.split(r"[^a-z0-9]+", company_name.lower())
        if len(t) > 1 and t not in _NAME_STOPWORDS
    }
    if not tokens:
        return 1.0
    host = (urlparse(url).netloc or "").lower().removeprefix("www.")
    host = host.split(":")[0]
    # Match against the registrable domain (last two labels) so a keyword in a
    # subdomain can't fake a match — e.g. arbysca.sandbox.opentender.io.
    labels = host.split(".")
    domain = ".".join(labels[-2:]) if len(labels) >= 2 else host
    hits = sum(1 for t in tokens if t in domain)
    return hits / len(tokens)


def _domain_extra_chars(company_name: str, url: str) -> int:
    """Characters in the registrable domain's first label left over after
    removing the matched name tokens. ``shopify-en.com`` vs 'Shopify' -> 2
    ('-en'); ``shopify.com`` -> 0. Ranks exact-name domains above lookalikes."""
    tokens = sorted(
        (t for t in re.split(r"[^a-z0-9]+", company_name.lower())
         if len(t) > 1 and t not in _NAME_STOPWORDS),
        key=len,
        reverse=True,
    )
    host = (urlparse(url).netloc or "").lower().removeprefix("www.")
    labels = host.split(".")
    sld = labels[-2] if len(labels) >= 2 else labels[0]
    for t in tokens:
        sld = sld.replace(t, "", 1)
    return len(sld)


# ccTLDs that are plausible homes for a Canadian company's official site.
# Foreign ccTLDs (.cn, .ru, …) hosting a name-lookalike are almost always
# parked/scam domains — penalize them below .ca/.com/generic TLDs.
_TRUSTED_TLDS = {"ca", "com", "org", "net", "io", "co", "ai", "app", "dev"}


def _candidate_rank_key(company_name: str, url: str):
    """Sort key: name match first, then prefer exact-name domains over
    lookalikes, trusted TLDs over foreign ccTLDs, apex over portal
    subdomains, then shallower paths."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().removeprefix("www.")
    labels = host.split(".")
    subdomain = labels[0] if len(labels) > 2 else ""
    portal_penalty = 1 if subdomain in _NON_CORPORATE_SUBDOMAINS else 0
    tld = labels[-1] if labels else ""
    tld_penalty = 0 if tld in _TRUSTED_TLDS else 1
    depth = (parsed.path or "/").count("/")
    return (
        _name_match_score(company_name, url),
        -_domain_extra_chars(company_name, url),
        -tld_penalty,
        -portal_penalty,
        -depth,
    )


def validate_candidate_node(state: GraphState) -> dict:
    """Pick the best official-company candidate from search results.

    Candidates are ranked by name-match score (query tokens vs domain) so an
    ambiguous query prefers the domain that actually contains the company name.
    Other valid candidates are kept as ``alternate_candidates`` and a warning
    is emitted when the best match is weak.
    """
    candidates = state.get("search_results") or []
    provider = state.get("search_provider", "")
    company_name = state.get("company_name", "")
    rejected = []
    valid: List[str] = []
    for url in candidates:
        ok, reason = validate_candidate_url(url)
        if ok:
            valid.append(url)
        else:
            rejected.append(f"{url} ({reason})")

    if valid:
        ranked = sorted(
            valid,
            key=lambda u: _candidate_rank_key(company_name, u),
            reverse=True,
        )
        best = ranked[0]
        score = _name_match_score(company_name, best)

        # A 0% name match means no query token appears in the domain — the
        # site is almost certainly a different company (e.g. a partner page
        # like triangle.canadiantire.ca for 'Petro Canada'). Failing is more
        # honest than classifying the wrong site. Exception: acronym domains —
        # 'Electronic Arts' → ea.com, 'Canadian Broadcasting Corporation' →
        # cbc.ca — where the domain is the company's initials.
        acronym = "".join(
            t[0] for t in re.split(r"[^a-z0-9]+", company_name.lower())
            if t and t not in _NAME_STOPWORDS
        )
        best_host = (urlparse(best).netloc or "").lower().removeprefix("www.")
        best_sld = best_host.split(".")[-2] if len(best_host.split(".")) >= 2 else best_host
        is_acronym = len(acronym) >= 2 and best_sld == acronym
        if company_name and score == 0 and not is_acronym:
            trace = [
                f"✗ Best candidate has no name match: {best} "
                f"(searched for '{company_name}')"
            ]
            if rejected:
                trace.append("  Rejected: " + "; ".join(rejected[:5]))
            update = {
                "selected_candidate": "",
                "candidate_validation": "no usable candidate",
                "trace": trace,
            }
            if provider == "tavily":
                update.update(
                    {
                        "classification_status": "not_performed",
                        "failure_type": FailureType.NO_OFFICIAL_URL,
                        "error": "Search results contained no usable official company website.",
                        "errors": ["No usable official company URL found"],
                    }
                )
            return update

        normalized = normalize_url(best) or best
        alternates = [normalize_url(u) or u for u in ranked[1:4]]

        trace = [f"✓ Official company URL validated: {normalized}"]

        # Portal-subdomain fallback: if the best candidate is e.g.
        # careers.example.com, prefer the apex domain (example.com) and keep
        # the portal URL as the first alternate.
        best_host = (urlparse(best).netloc or "").lower().removeprefix("www.")
        labels = best_host.split(".")
        if len(labels) > 2 and labels[0] in _NON_CORPORATE_SUBDOMAINS:
            apex = ".".join(labels[-2:])
            apex_url = normalize_url(apex)
            if apex_url:
                trace.append(
                    f"↳ {best_host} is a portal subdomain; trying apex domain {apex_url} instead"
                )
                alternates = [normalized] + [a for a in alternates if a != normalized]
                normalized = apex_url
                best = apex_url

        # Brand-TLD fallback: if the candidate lives on a TLD that is itself a
        # company-name token (m365.cloud.microsoft for 'Microsoft'), the real
        # corporate site is almost certainly <name>.com — prefer it and keep
        # the brand-TLD URL as an alternate.
        best_labels = best_host.split(".")
        if company_name and len(best_labels) >= 2:
            tld = best_labels[-1]
            name_tokens = [
                t for t in re.split(r"[^a-z0-9]+", company_name.lower())
                if len(t) > 1 and t not in _NAME_STOPWORDS
            ]
            if tld in name_tokens:
                com_url = normalize_url(f"{tld}.com")
                if com_url and com_url != normalized:
                    trace.append(
                        f"↳ {best_host} is on the .{tld} brand TLD; trying {com_url} instead"
                    )
                    alternates = [normalized] + [a for a in alternates if a != normalized]
                    normalized = com_url
                    best = com_url

        # Deep-path/query fallback: a result like example.com/profile,
        # example.com/ca/en/aco/flights, or example.com/?country=us is a page,
        # not the corporate homepage. Prefer the domain root and keep the
        # original URL as an alternate.
        parsed_best = urlparse(best)
        best_path = (parsed_best.path or "/").rstrip("/")
        if best_path or parsed_best.query or parsed_best.fragment:
            root_url = normalize_url(f"{parsed_best.scheme}://{parsed_best.netloc}")
            if root_url and root_url != normalized:
                trace.append(
                    f"↳ {best} is a deep page; trying domain root {root_url} instead"
                )
                alternates = [normalized] + [a for a in alternates if a != normalized]
                normalized = root_url
        warnings: List[str] = []
        if company_name and score < 0.5:
            warnings.append(
                f"Ambiguous name match: '{company_name}' weakly matches {normalized}"
            )
            trace.append(
                f"⚠ Weak name match ({score:.0%}) — the site may be a different company"
            )
        if alternates:
            trace.append("  Other candidates: " + "; ".join(alternates))

        return {
            "selected_candidate": normalized,
            "candidate_validation": "ok",
            "alternate_candidates": alternates,
            "url": normalized,
            "warnings": warnings,
            "trace": trace,
        }

    trace = ["✗ No usable official-company candidate from search results"]
    if rejected:
        trace.append("  Rejected: " + "; ".join(rejected[:5]))

    update = {
        "selected_candidate": "",
        "candidate_validation": "no usable candidate",
        "trace": trace,
    }
    if provider == "tavily":
        update.update(
            {
                "classification_status": "not_performed",
                "failure_type": FailureType.NO_OFFICIAL_URL,
                "error": "Search results contained no usable official company website.",
                "errors": ["No usable official company URL found"],
            }
        )
    return update


def scrape_homepage_node(state: GraphState) -> dict:
    url = state.get("selected_candidate") or state.get("url") or ""
    trace = [f"Scraping homepage with Firecrawl: {url}"]
    try:
        content, status = _firecrawl_scrape(url)
    except Exception as exc:
        logger.warning("Firecrawl scrape failed for %s: %s", url, exc)
        return {
            "content": "",
            "homepage_status": 0,
            "scrape_attempts": 1,
            "scrape_failure_reason": f"{FailureType.SCRAPE_NETWORK_ERROR}: {exc}",
            "trace": trace + [f"Firecrawl error: {exc}"],
        }
    trace.append(f"Homepage scrape returned {len(content)} characters (status {status})")
    return {
        "content": content,
        "homepage_status": status,
        "scrape_attempts": 1,
        "trace": trace,
    }


def try_alternate_node(state: GraphState) -> dict:
    """Promote the next alternate candidate after the selected site's scrape
    retries are exhausted — e.g. a lookalike domain that fails to load while
    the real site sits in the alternates list."""
    alternates = list(state.get("alternate_candidates") or [])
    company = state.get("company_name", "")
    failed_root = state.get("selected_candidate") or state.get("url") or ""
    while alternates:
        nxt = alternates[0]
        parsed = urlparse(nxt)
        root = normalize_url(f"{parsed.scheme}://{parsed.netloc}") or nxt
        # Skip alternates that normalize to the domain we just failed on —
        # retrying the same dead root wastes the whole retry budget.
        if root == failed_root:
            alternates.pop(0)
            continue
        # Skip lookalike domains — 'searsseating.com' contains 'sears' but has
        # many leftover characters, marking it a different company. Allow small
        # suffixes (searspr.com → 'pr' = 2) but reject long ones.
        if company and _domain_extra_chars(company, nxt) > 4:
            skipped = alternates.pop(0)
            logger.info("Skipping lookalike alternate: %s", skipped)
            continue
        break
    if not alternates:
        return {
            "alternates_exhausted": True,
            "trace": ["✗ No usable alternate candidates to try"],
        }
    nxt = alternates.pop(0)
    # Normalize to the domain root — alternates may be deep pages.
    parsed = urlparse(nxt)
    root = normalize_url(f"{parsed.scheme}://{parsed.netloc}") or nxt
    return {
        "selected_candidate": root,
        "url": root,
        "alternate_candidates": alternates,
        "scrape_attempts": 0,
        "scrape_valid": False,
        "scrape_failure_reason": "",
        "content": "",
        "trace": [f"↳ Trying alternate candidate: {root}"],
    }


def validate_scrape_node(state: GraphState) -> dict:
    content = state.get("content", "")
    status = state.get("homepage_status")
    valid, reason = validate_scrape(content, status)
    if valid:
        return {
            "scrape_valid": True,
            "scrape_failure_reason": "",
            "trace": [f"✓ Scrape validated ({len(content.strip())} characters)"],
        }
    return {
        "scrape_valid": False,
        "scrape_failure_reason": reason,
        "trace": [f"✗ Scrape rejected\n  Reason: {reason} ({len(content.strip()) if content else 0} characters)"],
    }


def retry_scrape_node(state: GraphState) -> dict:
    """Retry the scrape with escalating strategies.

    Attempt 2: Firecrawl with JS-render wait and full-page content.
    Attempt 3+: direct HTTP fetch fallback.
    """
    url = state.get("selected_candidate") or state.get("url") or ""
    attempts = int(state.get("scrape_attempts") or 0) + 1

    if attempts == 2:
        trace = [f"Retrying scrape (attempt {attempts}): Firecrawl with waitFor/full-page for {url}"]
        scrape_fn = lambda: _firecrawl_scrape(  # noqa: E731
            url,
            waitFor=3000,
            onlyMainContent=False,
            timeout=FIRECRAWL_TIMEOUT * 1000,
        )
        label = "Firecrawl retry"
    else:
        trace = [f"Retrying scrape (attempt {attempts}): direct HTTP fetch for {url}"]
        scrape_fn = lambda: _direct_fetch(url)  # noqa: E731
        label = "Direct fetch"

    try:
        content, status = scrape_fn()
    except Exception as exc:
        logger.warning("%s failed for %s: %s", label, url, exc)
        return {
            "content": "",
            "homepage_status": 0,
            "scrape_attempts": attempts,
            "scrape_failure_reason": f"{FailureType.SCRAPE_NETWORK_ERROR}: {exc}",
            "trace": trace + [f"{label} error: {exc}"],
        }
    trace.append(f"Retry returned {len(content)} characters (status {status})")
    return {
        "content": content,
        "homepage_status": status,
        "scrape_attempts": attempts,
        "trace": trace,
    }


def assess_evidence_node(state: GraphState) -> dict:
    content = state.get("content", "")
    if assess_evidence_sufficiency(content):
        return {"trace": ["✓ Homepage contains corporate-identity signals; proceeding to classification"]}
    return {"trace": ["Homepage lacks corporate-identity evidence; discovering evidence pages"]}


def discover_evidence_node(state: GraphState) -> dict:
    content = state.get("content", "")
    url = state.get("selected_candidate") or state.get("url") or ""
    pages = discover_evidence_pages(content, url)
    return {
        "evidence_pages": pages,
        "trace": [f"✓ Corporate evidence discovery: {len(pages)} candidate page(s)"],
    }


def _scrape_one(url: str) -> Tuple[str, str, str]:
    """Scrape one URL. Returns (url, content, rejection_reason)."""
    try:
        content, _status = _firecrawl_scrape(url)
    except Exception as exc:
        return url, "", f"failed: {exc}"
    valid, reason = validate_scrape(content)
    if not valid:
        return url, "", reason
    return url, content, ""


def scrape_evidence_node(state: GraphState) -> dict:
    pages = state.get("evidence_pages") or []
    scraped: List[EvidencePage] = [
        {
            "url": state.get("selected_candidate") or state.get("url") or "",
            "page_type": "homepage",
            "content": state.get("content", ""),
            "content_length": len(state.get("content", "")),
        }
    ]
    trace = []

    # Scrape evidence pages in parallel — each is an independent HTTP call.
    candidates = [p for p in pages if p.get("url")][: MAX_EVIDENCE_PAGES]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda p: _scrape_one(p["url"]), candidates))
    for page, (url, content, reason) in zip(candidates, results):
        if not content:
            label = "failed" if reason.startswith("failed") else "rejected"
            trace.append(f"✗ Evidence page {label}: {url} ({reason})")
            continue
        scraped.append(
            {
                "url": url,
                "page_type": page.get("page_type", "page"),
                "content": content,
                "content_length": len(content),
            }
        )
        trace.append(f"✓ Evidence page scraped: {url} ({len(content)} chars)")

    # Reference fallback — always fetch the Wikipedia lead when a company name
    # is available. It's one cheap API call, and the identity-pattern heuristic
    # can false-positive on pages that mention "headquarters" without stating
    # nationality (e.g. ea.com/about). For companies without an article the
    # lookup returns "" and costs nothing.
    has_identity = any(
        assess_evidence_sufficiency(p.get("content", "")) for p in scraped
    )
    wiki, wiki_url = _wikipedia_summary(state.get("company_name", ""))
    if wiki:
        scraped.append(
            {
                "url": wiki_url,
                "page_type": "reference",
                "content": wiki,
                "content_length": len(wiki),
            }
        )
        trace.append(f"✓ Wikipedia reference added ({len(wiki)} chars)")
        has_identity = True

    # Alternate-domain fallback: only when Wikipedia didn't settle identity.
    # Cap consecutive failures per domain so a dead host can't burn the budget.
    if not has_identity:
        for alt in (state.get("alternate_candidates") or [])[:2]:
            parsed_alt = urlparse(alt)
            alt_base = f"{parsed_alt.scheme}://{parsed_alt.netloc}"
            alt_urls = [alt] + [
                alt_base + path
                for _t, paths in GUESSED_EVIDENCE_PATHS[:4]
                for path in paths[:2]
            ]
            consecutive_failures = 0
            for alt_url in alt_urls:
                if len(scraped) > MAX_EVIDENCE_PAGES + 2 or consecutive_failures >= 3:
                    break
                url, content, reason = _scrape_one(alt_url)
                if not content:
                    consecutive_failures += 1
                    label = "failed" if reason.startswith("failed") else "rejected"
                    trace.append(f"✗ Alternate site {label}: {alt_url} ({reason})")
                    continue
                consecutive_failures = 0
                scraped.append(
                    {
                        "url": alt_url,
                        "page_type": "alternate_site",
                        "content": content,
                        "content_length": len(content),
                    }
                )
                trace.append(f"✓ Alternate site scraped: {alt_url} ({len(content)} chars)")
                if assess_evidence_sufficiency(content):
                    break
            if any(assess_evidence_sufficiency(p.get("content", "")) for p in scraped):
                break

    # Careers-search fallback: if no careers-type page yielded content, the
    # company's job board may live on an external ATS (e.g. jobs.*.com or a
    # Workday/Greenhouse board). Search for it and scrape the top valid hit.
    has_careers = any(
        p.get("page_type") == "careers" and p.get("content", "").strip()
        for p in scraped
    )
    company = state.get("company_name", "")
    if not has_careers and company:
        careers_urls: List[str] = []
        for search_fn in (_search_searxng, _search_brave, _search_tavily):
            try:
                careers_urls = search_fn(f"{company} careers Canada")
            except Exception as exc:
                logger.warning("Careers search failed: %s", exc)
                continue
            if careers_urls:
                break
        for url in careers_urls[:5]:
            ok, reason = validate_candidate_url(url)
            if not ok:
                trace.append(f"✗ Careers site skipped: {url} ({reason})")
                continue
            # Skip unrelated/lookalike domains — a different company's careers
            # page is not evidence for the target. The host must contain a
            # company-name token somewhere (jobs.ca has no 'yahoo' → unrelated
            # job board; company.myworkdayjobs.com does → legit portal). Then
            # reject lookalikes with long leftover suffixes (searsseating.com
            # ≠ Sears), except brand TLDs where the TLD is the name
            # (research.google for 'Google').
            if company:
                host = (urlparse(url).netloc or "").lower().removeprefix("www.")
                tld = host.split(".")[-1] if "." in host else ""
                name_tokens = {
                    t for t in re.split(r"[^a-z0-9]+", company.lower())
                    if len(t) > 1 and t not in _NAME_STOPWORDS
                }
                if name_tokens and not any(t in host for t in name_tokens):
                    trace.append(f"✗ Careers site skipped: {url} (unrelated domain)")
                    continue
                is_brand_tld = tld in name_tokens
                if not is_brand_tld and _domain_extra_chars(company, url) > 4:
                    trace.append(f"✗ Careers site skipped: {url} (lookalike domain)")
                    continue
            url, content, reason = _scrape_one(url)
            if not content:
                label = "failed" if reason.startswith("failed") else "rejected"
                trace.append(f"✗ Careers site {label}: {url} ({reason})")
                continue
            scraped.append(
                {
                    "url": url,
                    "page_type": "careers",
                    "content": content,
                    "content_length": len(content),
                }
            )
            trace.append(f"✓ Careers site scraped: {url} ({len(content)} chars)")
            break
    return {"evidence_pages": scraped, "trace": trace}


def validate_evidence_node(state: GraphState) -> dict:
    """Pre-LLM gate: classification requires at least one usable evidence page."""
    pages = [p for p in (state.get("evidence_pages") or []) if p.get("content", "").strip()]

    # If the homepage was deemed sufficient on its own, the discovery path was
    # skipped and evidence_pages is empty — synthesize the homepage entry.
    if not pages and state.get("scrape_valid") and (state.get("content") or "").strip():
        pages = [
            {
                "url": state.get("selected_candidate") or state.get("url") or "",
                "page_type": "homepage",
                "content": state["content"],
                "content_length": len(state["content"]),
            }
        ]
        return {
            "evidence_pages": pages,
            "trace": ["✓ Evidence bundle created (1 page: homepage)"],
        }

    if not pages:
        return {
            "classification_status": "not_performed",
            "failure_type": FailureType.EVIDENCE_INSUFFICIENT,
            "error": "No usable evidence could be retrieved for classification.",
            "errors": ["No usable evidence pages"],
            "trace": ["✗ No usable evidence pages; classification will not run"],
        }
    return {"trace": [f"✓ Evidence bundle created ({len(pages)} page(s))"]}


def classify_node(state: GraphState) -> dict:
    url = state.get("selected_candidate") or state.get("url") or ""
    bundle = build_evidence_bundle(state.get("evidence_pages") or [])
    if not bundle.strip():
        return {
            "classification_status": "not_performed",
            "failure_type": FailureType.EVIDENCE_INSUFFICIENT,
            "error": "No usable evidence could be retrieved for classification.",
            "trace": ["✗ Empty evidence bundle; classification skipped"],
        }

    prompt = ANALYSIS_PROMPT.format(url=url, content=bundle)
    text = ""
    parsed = None
    try:
        llm = _create_llm()
        for attempt in range(2):
            response = llm.invoke([SystemMessage(content=prompt)])
            text = response.content.strip()
            parsed = _parse_llm_json(text)
            if parsed is not None:
                break
            logger.warning(
                "LLM returned unparseable output (attempt %d): %s",
                attempt + 1,
                text[:500],
            )
    except Exception as exc:
        logger.exception("LLM classification failed")
        return {
            "classification_status": "not_performed",
            "failure_type": FailureType.LLM_ERROR,
            "error": f"Classification failed: {exc}",
            "errors": [f"LLM error: {exc}"],
            "trace": [f"✗ Ollama classification error: {exc}"],
        }

    if parsed is None:
        return {
            "classification_status": "performed",
            "answer": "Unclear",
            "confidence": "Low",
            "employs_canadians": "Unclear",
            "reasoning": "The model returned output that could not be parsed; treating evidence as insufficient.",
            "warnings": [f"{FailureType.LLM_INVALID_OUTPUT}: unparseable model output"],
            "trace": ["✗ LLM output could not be parsed; defaulting to Unclear"],
        }

    return {
        "classification_status": "performed",
        "answer": _normalize_choice(parsed.get("answer"), ("Yes", "No", "Unclear"), "Unclear"),
        "confidence": _normalize_choice(parsed.get("confidence"), ("High", "Medium", "Low"), "Low"),
        "employs_canadians": _normalize_choice(
            parsed.get("employs_canadians"), ("Yes", "Unclear"), "Unclear"
        ),
        "canadian_evidence": _normalize_evidence_list(parsed.get("canadian_evidence")),
        "non_canadian_evidence": _normalize_evidence_list(parsed.get("non_canadian_evidence")),
        "employment_evidence": _normalize_evidence_list(parsed.get("employment_evidence")),
        "reasoning": str(parsed.get("reasoning", "")),
        "trace": [
            f"✓ Ollama classification: {parsed.get('answer', 'Unclear')} / {parsed.get('confidence', 'Low')}"
        ],
    }


_INSUFFICIENT_REASONING = re.compile(
    r"(no|insufficient|lack of|lacks|without|not enough|cannot determine|"
    r"unable to determine|does not (provide|contain|establish)) .{0,40}evidence",
    re.IGNORECASE,
)

# Direction-specific evidence-gap phrases. A gap about the *opposite* direction
# is consistent with the answer (e.g. "no evidence of Canadian identity"
# supports a No); a gap about the *same* direction contradicts it.
_GAP_PHRASE = (
    r"(?:no|insufficient|lack of|lacks|without|not enough|cannot determine|"
    r"unable to determine|does not (?:provide|contain|establish)).{0,60}evidence"
)
_GAP_CANADIAN = re.compile(_GAP_PHRASE + r".{0,80}(canadian|canada)", re.IGNORECASE)
_GAP_FOREIGN = re.compile(
    _GAP_PHRASE + r".{0,80}(non[- ]canadian|outside canada|foreign|based outside|not canadian)",
    re.IGNORECASE,
)


def validate_classification_node(state: GraphState) -> dict:
    """Deterministic post-LLM validation. The LLM is not the final authority."""
    if state.get("classification_status") != "performed":
        return {"trace": ["Classification was not performed; skipping validation"]}

    answer = _normalize_choice(state.get("answer"), ("Yes", "No", "Unclear"), "Unclear")
    employs = _normalize_choice(state.get("employs_canadians"), ("Yes", "Unclear"), "Unclear")
    can_ev = state.get("canadian_evidence") or []
    non_ev = state.get("non_canadian_evidence") or []
    emp_ev = state.get("employment_evidence") or []
    reasoning = state.get("reasoning", "")
    warnings: List[str] = []
    trace: List[str] = []

    # Quote verification: drop evidence items whose quote_or_excerpt does not
    # appear in the scraped content — the model sometimes fabricates claims
    # (e.g. citing "headquarters in Toronto" when no page says that).
    corpus = " ".join(
        (p.get("content") or "") for p in (state.get("evidence_pages") or [])
    )
    # Normalize: lowercase, strip punctuation, collapse whitespace — so
    # "Ottawa, Ontario" matches "ottawa ontario" in the corpus.
    corpus_norm = re.sub(r"[^a-z0-9 ]", " ", corpus.lower())
    corpus_norm = re.sub(r"\s+", " ", corpus_norm)

    def _verified(items):
        kept, dropped = [], 0
        for item in items:
            quote = (item.get("quote_or_excerpt") or "").strip()
            quote_norm = re.sub(r"[^a-z0-9 ]", " ", quote.lower())
            quote_norm = re.sub(r"\s+", " ", quote_norm).strip()
            if not quote_norm or quote_norm in corpus_norm:
                kept.append(item)
                continue
            # Tolerate paraphrasing/ellipses: keep the item if any 5-word
            # window of the quote appears in the corpus. For very short
            # quotes, require the full quote.
            words = quote_norm.split()
            window = min(5, len(words))
            found = any(
                " ".join(words[i:i + window]) in corpus_norm
                for i in range(len(words) - window + 1)
            )
            if found:
                kept.append(item)
            else:
                dropped += 1
        return kept, dropped

    can_ev, d1 = _verified(can_ev)
    non_ev, d2 = _verified(non_ev)
    emp_ev, d3 = _verified(emp_ev)
    if d1 + d2 + d3:
        warnings.append(
            "Some cited evidence could not be verified against the page content and was removed"
        )
        trace.append(f"⚠ {d1 + d2 + d3} evidence quote(s) not found in scraped content; discarded")

    dropped_quotes = d1 + d2 + d3
    if answer == "Yes" and not can_ev:
        answer = "Unclear"
        warnings.append("Answer 'Yes' had no Canadian evidence; downgraded to Unclear")
    if answer == "No" and not non_ev:
        answer = "Unclear"
        warnings.append("Answer 'No' had no non-Canadian evidence; downgraded to Unclear")
    # If fabricated quotes caused the downgrade, the model's reasoning may
    # repeat the hallucinated claim — replace it with an honest explanation.
    if dropped_quotes and answer == "Unclear":
        reasoning = (
            "The available page content did not contain verifiable evidence "
            "to confirm whether this company is Canadian."
        )
    # Downgrade on "insufficient evidence" phrasing — but a gap about the
    # opposite direction is consistent with the answer (e.g. "no evidence of
    # Canadian identity" supports No), so only downgrade when the gap targets
    # the answer's own direction or is direction-neutral.
    if answer != "Unclear" and _INSUFFICIENT_REASONING.search(reasoning):
        same_gap = _GAP_FOREIGN if answer == "No" else _GAP_CANADIAN
        opposite_gap = _GAP_CANADIAN if answer == "No" else _GAP_FOREIGN
        if same_gap.search(reasoning) or not opposite_gap.search(reasoning):
            answer = "Unclear"
            warnings.append("The available evidence was insufficient to confirm the answer")

    # Deterministic employment upgrade: if the model said Unclear but a
    # careers/jobs-type evidence page contains Canadian location signals,
    # that is affirmative evidence of Canadian employment. Runs before the
    # downgrade check so a verified upgrade isn't followed by a stale warning.
    if employs != "Yes" or not emp_ev:
        for page in state.get("evidence_pages") or []:
            if page.get("page_type") not in ("careers", "homepage", "contact"):
                continue
            signals = detect_canadian_signals(page.get("content", ""))
            # A careers page on a .ca domain or a Canadian-locale path
            # (/en-ca/, /fr-ca/) is itself a Canadian employment signal — a
            # dedicated Canadian job portal implies Canadian hiring.
            page_url = page.get("url", "")
            page_host = (urlparse(page_url).netloc or "").lower()
            page_path = (urlparse(page_url).path or "").lower()
            if page.get("page_type") == "careers" and (
                page_host.endswith(".ca")
                or re.search(r"/(en|fr)-ca(/|$)", page_path)
            ):
                signals = signals or ["careers portal on Canadian domain/locale"]
            # Require a careers/jobs page for the upgrade, or explicit
            # employment phrasing on other pages.
            is_careers = page.get("page_type") == "careers"
            has_jobs_text = bool(
                re.search(r"(job|career|position|role|hiring|employ)", page.get("content", ""), re.I)
            )
            # A bare 2-letter province code (NU, ON, …) is too weak on its own —
            # it can appear in unrelated text. Require a city/province name,
            # 'Canada', the locale signal, or multiple signals.
            strong = [
                s for s in signals
                if not (len(s) == 2 and s.isupper())
            ]
            if strong and (is_careers or has_jobs_text):
                employs = "Yes"
                emp_ev = emp_ev + [{
                    "claim": "Canadian job locations detected on company page",
                    "source_url": page.get("url", ""),
                    "quote_or_excerpt": f"Canadian locations detected: {', '.join(signals)}",
                }]
                trace.append(
                    f"↳ Employs Canadians upgraded to Yes: {page.get('url')} "
                    f"mentions {', '.join(signals[:5])}"
                )
                break

    # Post-upgrade downgrade: only warn if employment is still unconfirmed.
    if employs == "Yes" and not emp_ev:
        employs = "Unclear"
        warnings.append("Canadian employment could not be confirmed by verifiable evidence")

    if warnings:
        trace.append("✗ Deterministic validation adjusted the result")
        trace.extend(f"  {w}" for w in warnings)
    else:
        trace.append("✓ Deterministic validation passed")

    return {
        "answer": answer,
        "employs_canadians": employs,
        "canadian_evidence": can_ev,
        "non_canadian_evidence": non_ev,
        "employment_evidence": emp_ev,
        "reasoning": reasoning,
        "warnings": warnings,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_normalize(state: GraphState) -> str:
    if state.get("classification_status") == "not_performed":
        return "terminate"
    if state.get("selected_candidate"):
        return "scrape_homepage"
    return "searxng"


def route_after_search(provider: str):
    def _route(state: GraphState) -> str:
        if state.get("search_results"):
            return "validate_candidate"
        nxt = {"searxng": "brave", "brave": "tavily"}.get(provider)
        return nxt or "terminate"

    return _route


def route_after_validate_candidate(state: GraphState) -> str:
    if state.get("selected_candidate"):
        return "scrape_homepage"
    nxt = {"searxng": "brave", "brave": "tavily"}.get(state.get("search_provider", ""))
    return nxt or "terminate"


def route_after_validate_scrape(state: GraphState) -> str:
    if state.get("scrape_valid"):
        return "assess_evidence"
    attempts = int(state.get("scrape_attempts") or 0)
    if attempts <= MAX_SCRAPE_RETRIES:
        return "retry_scrape"
    # Retries exhausted. If the site returned some real content (e.g. a thin
    # JS-heavy marketing page), still try evidence-page discovery — canonical
    # pages like /about may scrape fine even when the homepage doesn't.
    if len((state.get("content") or "").strip()) >= 50:
        return "discover_evidence"
    # Nothing usable — try the next alternate candidate before giving up.
    if state.get("alternate_candidates"):
        return "try_alternate"
    return "terminate"


def route_after_assess(state: GraphState) -> str:
    if assess_evidence_sufficiency(state.get("content", "")):
        # Homepage alone is usable evidence.
        return "validate_evidence"
    return "discover_evidence"


def route_after_validate_evidence(state: GraphState) -> str:
    if state.get("classification_status") == "not_performed":
        return "terminate"
    return "classify"


def _terminal_update(state: GraphState) -> dict:
    """Ensure terminal failure states surface a consistent top-level error."""
    if state.get("classification_status") == "not_performed" and not state.get("error"):
        reason = state.get("scrape_failure_reason") or state.get("failure_type") or "unknown"
        return {"error": f"Classification not performed: {reason}"}
    return {}


def terminate_node(state: GraphState) -> dict:
    update = _terminal_update(state)

    # Scrape-failure path: validation failed and retries were exhausted without
    # classification_status being set. Mark it explicitly here.
    if (
        state.get("classification_status") != "performed"
        and state.get("scrape_attempts")
        and not state.get("scrape_valid")
        and not state.get("classification_status")
    ):
        reason = state.get("scrape_failure_reason") or FailureType.SCRAPE_NETWORK_ERROR
        update["classification_status"] = "not_performed"
        update["failure_type"] = reason.split(":")[0]
        update["error"] = f"Classification not performed: website content could not be reliably retrieved ({reason})."
        update.setdefault("errors", []).append(update["error"])

    if update.get("error"):
        update["trace"] = [f"✗ Workflow ended: {update['error']}"]
    elif state.get("classification_status") == "not_performed":
        update["trace"] = [f"✗ Workflow ended: {state.get('error', 'classification not performed')}"]
    else:
        update["trace"] = ["✓ Workflow complete"]
    return update


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("normalize_input", normalize_input_node)
    workflow.add_node("searxng", searxng_node)
    workflow.add_node("brave", brave_node)
    workflow.add_node("tavily", tavily_node)
    workflow.add_node("validate_candidate", validate_candidate_node)
    workflow.add_node("scrape_homepage", scrape_homepage_node)
    workflow.add_node("validate_scrape", validate_scrape_node)
    workflow.add_node("retry_scrape", retry_scrape_node)
    workflow.add_node("try_alternate", try_alternate_node)
    workflow.add_node("assess_evidence", assess_evidence_node)
    workflow.add_node("discover_evidence", discover_evidence_node)
    workflow.add_node("scrape_evidence", scrape_evidence_node)
    workflow.add_node("validate_evidence", validate_evidence_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("validate_classification", validate_classification_node)
    workflow.add_node("terminate", terminate_node)

    workflow.set_entry_point("normalize_input")
    workflow.add_conditional_edges("normalize_input", route_after_normalize)
    workflow.add_conditional_edges("searxng", route_after_search("searxng"))
    workflow.add_conditional_edges("brave", route_after_search("brave"))
    workflow.add_conditional_edges("tavily", route_after_search("tavily"))
    workflow.add_conditional_edges("validate_candidate", route_after_validate_candidate)
    workflow.add_edge("scrape_homepage", "validate_scrape")
    workflow.add_edge("retry_scrape", "validate_scrape")
    workflow.add_conditional_edges(
        "try_alternate",
        # All candidates failed to scrape — still try the Wikipedia reference
        # path so a well-known company gets an evidence-based answer.
        lambda s: "scrape_evidence" if s.get("alternates_exhausted") else "scrape_homepage",
    )
    workflow.add_conditional_edges("validate_scrape", route_after_validate_scrape)
    workflow.add_conditional_edges("assess_evidence", route_after_assess)
    workflow.add_edge("discover_evidence", "scrape_evidence")
    workflow.add_edge("scrape_evidence", "validate_evidence")
    workflow.add_conditional_edges("validate_evidence", route_after_validate_evidence)
    workflow.add_edge("classify", "validate_classification")
    workflow.add_edge("validate_classification", "terminate")
    workflow.add_edge("terminate", END)

    return workflow.compile()
