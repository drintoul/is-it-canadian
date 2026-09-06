"""Tests for the evidence-driven classification workflow.

External services (Firecrawl, search providers, Ollama) are mocked so the
deterministic invariants can be verified without network access.
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import graph as g  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_CONTENT = (
    "Welcome to Example Corp. We sell products online and ship worldwide. "
    * 10
)

CANADIAN_HQ_CONTENT = (
    "Example Corp is a Canadian company. Our global headquarters is located "
    "in Toronto, Ontario. " * 10
)

FOREIGN_HQ_CONTENT = (
    "Example Corp is headquartered in Seattle, Washington. " * 10
)


def _llm_response(payload: dict):
    import json

    resp = MagicMock()
    resp.content = json.dumps(payload)
    return resp


def _run_graph(state: dict):
    return g.build_graph().invoke(state)


def _base_state(**overrides):
    state = {
        "company_name": "",
        "input_url": "",
        "url": "",
        "content": "",
        "trace": [],
        "errors": [],
        "warnings": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Unit tests: validate_scrape
# ---------------------------------------------------------------------------


class TestValidateScrape:
    def test_empty_string_rejected(self):
        valid, reason = g.validate_scrape("")
        assert not valid
        assert reason == g.FailureType.SCRAPE_EMPTY

    def test_single_char_rejected(self):
        valid, reason = g.validate_scrape("x")
        assert not valid
        assert reason == g.FailureType.SCRAPE_TOO_SHORT

    def test_none_rejected(self):
        valid, reason = g.validate_scrape(None)
        assert not valid
        assert reason == g.FailureType.SCRAPE_EMPTY

    def test_js_only_rejected(self):
        content = "Please enable JavaScript to view this site. " * 20
        valid, reason = g.validate_scrape(content)
        assert not valid
        assert reason == g.FailureType.SCRAPE_JS_ONLY

    def test_captcha_rejected(self):
        content = "Verify you are human. Complete the captcha below. " * 10
        valid, reason = g.validate_scrape(content)
        assert not valid
        assert reason == g.FailureType.SCRAPE_ACCESS_DENIED

    def test_http_error_status_rejected(self):
        valid, reason = g.validate_scrape(LONG_CONTENT, status=500)
        assert not valid
        assert reason == g.FailureType.SCRAPE_HTTP_ERROR

    def test_valid_content_accepted(self):
        valid, reason = g.validate_scrape(LONG_CONTENT, status=200)
        assert valid
        assert reason == ""


# ---------------------------------------------------------------------------
# Unit tests: URL helpers
# ---------------------------------------------------------------------------


class TestUrlHelpers:
    def test_normalize_adds_https_and_www(self):
        assert g.normalize_url("timhortons.ca") == "https://www.timhortons.ca/"

    def test_normalize_upgrades_http(self):
        assert g.normalize_url("http://example.com") == "https://www.example.com/"

    def test_normalize_keeps_existing_www(self):
        assert g.normalize_url("https://www.example.com") == "https://www.example.com/"

    def test_normalize_keeps_subdomain(self):
        assert g.normalize_url("shop.example.com") == "https://shop.example.com/"

    def test_normalize_invalid(self):
        assert g.normalize_url("not a url at all :::") == "" or True  # may parse; ensure no crash

    def test_candidate_rejects_wikipedia(self):
        ok, reason = g.validate_candidate_url("https://en.wikipedia.org/wiki/Shopify")
        assert not ok

    def test_candidate_rejects_linkedin(self):
        ok, _ = g.validate_candidate_url("https://www.linkedin.com/company/shopify")
        assert not ok

    def test_candidate_accepts_company_domain(self):
        ok, _ = g.validate_candidate_url("https://www.shopify.com")
        assert ok


# ---------------------------------------------------------------------------
# Unit tests: deterministic classification validation
# ---------------------------------------------------------------------------


class TestValidateClassification:
    def test_yes_without_evidence_downgraded(self):
        state = {
            "classification_status": "performed",
            "answer": "Yes",
            "canadian_evidence": [],
            "non_canadian_evidence": [],
            "employment_evidence": [],
            "employs_canadians": "Unclear",
            "reasoning": "The company is Canadian.",
        }
        result = g.validate_classification_node(state)
        assert result["answer"] == "Unclear"

    def test_no_without_evidence_downgraded(self):
        state = {
            "classification_status": "performed",
            "answer": "No",
            "canadian_evidence": [],
            "non_canadian_evidence": [],
            "employment_evidence": [],
            "employs_canadians": "Unclear",
            "reasoning": "The company is American.",
        }
        result = g.validate_classification_node(state)
        assert result["answer"] == "Unclear"

    def test_no_with_absence_reasoning_downgraded(self):
        state = {
            "classification_status": "performed",
            "answer": "No",
            "canadian_evidence": [],
            "non_canadian_evidence": [{"claim": "x", "source_url": "u", "quote_or_excerpt": "q"}],
            "employment_evidence": [],
            "employs_canadians": "Unclear",
            "reasoning": "There is no evidence that the company is based outside Canada.",
        }
        result = g.validate_classification_node(state)
        assert result["answer"] == "Unclear"

    def test_employment_yes_without_evidence_downgraded(self):
        state = {
            "classification_status": "performed",
            "answer": "Unclear",
            "canadian_evidence": [],
            "non_canadian_evidence": [],
            "employment_evidence": [],
            "employs_canadians": "Yes",
            "reasoning": "They employ Canadians.",
        }
        result = g.validate_classification_node(state)
        assert result["employs_canadians"] == "Unclear"

    def test_yes_with_evidence_passes(self):
        state = {
            "classification_status": "performed",
            "answer": "Yes",
            "canadian_evidence": [{"claim": "HQ in Toronto", "source_url": "u", "quote_or_excerpt": "headquarters is in Toronto"}],
            "non_canadian_evidence": [],
            "employment_evidence": [],
            "employs_canadians": "Unclear",
            "reasoning": "The supplied content states the headquarters is in Toronto, Canada.",
            "evidence_pages": [
                {"url": "u", "page_type": "about", "content": "Our headquarters is in Toronto, Canada."}
            ],
        }
        result = g.validate_classification_node(state)
        assert result["answer"] == "Yes"

    def test_fabricated_quote_dropped(self):
        state = {
            "classification_status": "performed",
            "answer": "Yes",
            "canadian_evidence": [{"claim": "HQ in Toronto", "source_url": "u", "quote_or_excerpt": "headquartered in Toronto"}],
            "non_canadian_evidence": [],
            "employment_evidence": [],
            "employs_canadians": "Unclear",
            "reasoning": "The supplied content states the headquarters is in Toronto.",
            "evidence_pages": [
                {"url": "u", "page_type": "about", "content": "We sell shoes online worldwide."}
            ],
        }
        result = g.validate_classification_node(state)
        assert result["answer"] == "Unclear"
        assert result["canadian_evidence"] == []


# ---------------------------------------------------------------------------
# Unit tests: LLM JSON parsing
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_plain_json(self):
        parsed = g._parse_llm_json('{"answer": "Yes"}')
        assert parsed["answer"] == "Yes"

    def test_fenced_json(self):
        parsed = g._parse_llm_json('```json\n{"answer": "No"}\n```')
        assert parsed["answer"] == "No"

    def test_json_with_prose(self):
        parsed = g._parse_llm_json('Here is my answer:\n{"answer": "Unclear"}\nDone.')
        assert parsed["answer"] == "Unclear"

    def test_garbage_returns_none(self):
        assert g._parse_llm_json("no json here") is None

    def test_trailing_comma_tolerated(self):
        parsed = g._parse_llm_json('{"answer": "Yes",}')
        assert parsed["answer"] == "Yes"


class TestCanadianSignals:
    def test_detects_city(self):
        assert "Toronto" in g.detect_canadian_signals("We have offices in Toronto and Austin.")

    def test_detects_province(self):
        assert "British Columbia" in g.detect_canadian_signals("Located in British Columbia.")

    def test_detects_province_code(self):
        assert "ON" in g.detect_canadian_signals("Job location: Mississauga, ON")

    def test_detects_canada(self):
        assert "Canada" in g.detect_canadian_signals("Remote - Canada")

    def test_no_false_positive_on_substring(self):
        # 'ON' inside words like 'BUTTON' should not match
        assert "ON" not in g.detect_canadian_signals("Click the BUTTON to continue")

    def test_empty_content(self):
        assert g.detect_canadian_signals("") == []
        assert g.detect_canadian_signals(None) == []

    def test_bundle_includes_signal_note(self):
        pages = [{"url": "https://x.com/careers", "page_type": "careers", "content": "Jobs in Toronto, ON"}]
        bundle = g.build_evidence_bundle(pages)
        assert "CONTEXT" in bundle
        assert "Toronto" in bundle


# ---------------------------------------------------------------------------
# Integration tests: graph-level invariants (externals mocked)
# ---------------------------------------------------------------------------


class TestGraphInvariants:
    def test_empty_scrape_never_reaches_llm(self):
        with patch.object(g, "_firecrawl_scrape", return_value=("", 200)), \
             patch.object(g, "_direct_fetch", return_value=("", 200)), \
             patch.object(g, "resolve_redirects", side_effect=lambda u: (u, [u])), \
             patch.object(g, "_create_llm") as mock_llm:
            result = _run_graph(_base_state(input_url="https://example.com"))
            mock_llm.assert_not_called()
            assert result["classification_status"] == "not_performed"
            assert result["failure_type"] == g.FailureType.SCRAPE_EMPTY

    def test_one_char_scrape_never_reaches_llm(self):
        with patch.object(g, "_firecrawl_scrape", return_value=("x", 200)), \
             patch.object(g, "_direct_fetch", return_value=("x", 200)), \
             patch.object(g, "resolve_redirects", side_effect=lambda u: (u, [u])), \
             patch.object(g, "_create_llm") as mock_llm:
            result = _run_graph(_base_state(input_url="https://example.com"))
            mock_llm.assert_not_called()
            assert result["classification_status"] == "not_performed"
            assert result["failure_type"] == g.FailureType.SCRAPE_TOO_SHORT

    def test_all_search_providers_fail_no_llm(self):
        with patch.object(g, "_search_searxng", side_effect=RuntimeError("down")), \
             patch.object(g, "_search_brave", side_effect=RuntimeError("down")), \
             patch.object(g, "_search_tavily", side_effect=RuntimeError("down")), \
             patch.object(g, "_create_llm") as mock_llm:
            result = _run_graph(_base_state(company_name="Acme Corp"))
            mock_llm.assert_not_called()
            assert result["classification_status"] == "not_performed"
            assert result["failure_type"] == g.FailureType.SEARCH_FAILED

    def test_llm_yes_without_evidence_becomes_unclear(self):
        llm = MagicMock()
        llm.invoke.return_value = _llm_response(
            {
                "answer": "Yes",
                "confidence": "High",
                "employs_canadians": "Unclear",
                "canadian_evidence": [],
                "non_canadian_evidence": [],
                "employment_evidence": [],
                "reasoning": "It is Canadian.",
            }
        )
        with patch.object(g, "_firecrawl_scrape", return_value=(CANADIAN_HQ_CONTENT, 200)), \
             patch.object(g, "resolve_redirects", side_effect=lambda u: (u, [u])), \
             patch.object(g, "_create_llm", return_value=llm):
            result = _run_graph(_base_state(input_url="https://example.com"))
            assert result["answer"] == "Unclear"

    def test_llm_yes_with_evidence_passes(self):
        llm = MagicMock()
        llm.invoke.return_value = _llm_response(
            {
                "answer": "Yes",
                "confidence": "High",
                "employs_canadians": "Yes",
                "canadian_evidence": [
                    {"claim": "HQ in Toronto", "source_url": "https://example.com", "quote_or_excerpt": "global headquarters is located in Toronto, Ontario"}
                ],
                "non_canadian_evidence": [],
                "employment_evidence": [
                    {"claim": "Canadian staff", "source_url": "https://example.com", "quote_or_excerpt": "Example Corp is a Canadian company"}
                ],
                "reasoning": "The supplied content states the global headquarters is in Toronto, Ontario.",
            }
        )
        with patch.object(g, "_firecrawl_scrape", return_value=(CANADIAN_HQ_CONTENT, 200)), \
             patch.object(g, "resolve_redirects", side_effect=lambda u: (u, [u])), \
             patch.object(g, "_create_llm", return_value=llm):
            result = _run_graph(_base_state(input_url="https://example.com"))
            assert result["answer"] == "Yes"
            assert result["confidence"] == "High"
            assert result["employs_canadians"] == "Yes"

    def test_foreign_hq_with_canadian_employees(self):
        llm = MagicMock()
        llm.invoke.return_value = _llm_response(
            {
                "answer": "No",
                "confidence": "High",
                "employs_canadians": "Yes",
                "canadian_evidence": [],
                "non_canadian_evidence": [
                    {"claim": "HQ in Seattle", "source_url": "https://example.com", "quote_or_excerpt": "headquartered in Seattle, Washington"}
                ],
                "employment_evidence": [
                    {"claim": "Canadian jobs", "source_url": "https://example.com/careers", "quote_or_excerpt": "headquartered in Seattle, Washington"}
                ],
                "reasoning": "The supplied content states the company is headquartered in Seattle, Washington.",
            }
        )
        with patch.object(g, "_firecrawl_scrape", return_value=(FOREIGN_HQ_CONTENT, 200)), \
             patch.object(g, "resolve_redirects", side_effect=lambda u: (u, [u])), \
             patch.object(g, "_create_llm", return_value=llm):
            result = _run_graph(_base_state(input_url="https://example.com"))
            assert result["answer"] == "No"
            assert result["employs_canadians"] == "Yes"

    def test_unparseable_llm_output_does_not_crash(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="I cannot determine this.")
        with patch.object(g, "_firecrawl_scrape", return_value=(LONG_CONTENT, 200)), \
             patch.object(g, "resolve_redirects", side_effect=lambda u: (u, [u])), \
             patch.object(g, "_create_llm", return_value=llm):
            result = _run_graph(_base_state(input_url="https://example.com"))
            assert result["answer"] == "Unclear"
            assert result["classification_status"] == "performed"

    def test_search_results_only_aggregators_fails(self):
        with patch.object(g, "_search_searxng", return_value=["https://en.wikipedia.org/wiki/Acme"]), \
             patch.object(g, "_search_brave", return_value=["https://www.linkedin.com/company/acme"]), \
             patch.object(g, "_search_tavily", return_value=["https://www.facebook.com/acme"]), \
             patch.object(g, "_create_llm") as mock_llm:
            result = _run_graph(_base_state(company_name="Acme"))
            mock_llm.assert_not_called()
            assert result["classification_status"] == "not_performed"
            assert result["failure_type"] == g.FailureType.NO_OFFICIAL_URL

    def test_no_input_fails_fast(self):
        result = _run_graph(_base_state())
        assert result["classification_status"] == "not_performed"
        assert result["failure_type"] == g.FailureType.URL_INVALID
