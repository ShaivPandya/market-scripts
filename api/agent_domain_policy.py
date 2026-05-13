"""Deterministic domain guardrails for Stan agent chat."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

DomainDecision = Literal["allow", "block", "clarify"]

DOMAIN_BLOCK_RESPONSE = (
    "I can only help with investing related questions."
)
DOMAIN_CLARIFY_RESPONSE = "Can you frame that as an investing question?"
MIXED_DOMAIN_INSTRUCTION = (
    "The current user turn contains both supported finance/business content and unsupported content. "
    "Answer only the supported finance, markets, investing, portfolio, macro, business, company, "
    "valuation, accounting, industry, risk, thesis, or Talisman workflow portion. Explicitly decline "
    "the unrelated portion in one concise sentence. Do not provide instructions, facts, or advice for "
    "the unrelated request."
)


@dataclass(frozen=True)
class AgentDomainClassification:
    decision: DomainDecision
    reason: str
    contains_unsupported_request: bool = False


_CASUAL_RX = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|cool|ok|okay|who are you|what can you do)[\s!.?]*$",
    re.IGNORECASE,
)
_SUPPORTED_RX = re.compile(
    r"\b("
    r"finance|financial|invest|investing|investment|portfolio|holding|holdings|position|positions|"
    r"market|markets|stock|stocks|equity|equities|security|securities|bond|bonds|rates?|yield|curve|"
    r"s&p|spx|nasdaq|dow|russell|"
    r"macro|economy|economic|inflation|deflation|recession|growth|gdp|labor|housing|liquidity|"
    r"fed|fomc|ecb|boj|central bank|currency|currencies|fx|forex|eurusd|usdjpy|gbpusd|"
    r"commodity|commodities|oil|copper|gold|natural gas|futures?|options?|volatility|vix|"
    r"business|company|companies|industry|industries|sector|sectors|strategy|operations|competition|"
    r"accounting|revenue|earnings|eps|ebitda|margin|margins|cash flow|balance sheet|income statement|"
    r"valuation|valuations|multiple|multiples|dcf|price target|fair value|"
    r"risk|risks|hedge|hedging|beta|exposure|p&l|pnl|drawdown|"
    r"thesis|catalyst|catalysts|kill condition|dossier|conviction|sizing|sizer|analyzer|optimizer|"
    r"screener|screen|chart|technical analysis|breadth|sentiment|positioning|management quality|"
    r"talisman|stan|workspace|approval|approvals|watch trigger|workflow"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_RX = re.compile(
    r"\b("
    r"recipe|recipes|cook|cooking|bake|baking|dinner|lunch|breakfast|meal|ingredients?|"
    r"travel|trip|vacation|hotel|hotels|flight|flights|itinerary|tourist|tourism|"
    r"sports?|nba|nfl|mlb|nhl|soccer|football|basketball|baseball|lakers|warriors|yankees|"
    r"medical|medicine|doctor|symptom|symptoms|diagnos(?:e|is)|treatment|advil|ibuprofen|"
    r"tylenol|acetaminophen|therapy|therapist|relationship|dating|marriage|personal advice|"
    r"trivia|movie|movies|song|songs|lyrics|poem|poetry|joke|story|horoscope|"
    r"quicksort|leetcode|algorithm|algorithms|binary tree|linked list"
    r")\b",
    re.IGNORECASE,
)
_GENERIC_CODING_RX = re.compile(
    r"\b(write|implement|code|debug|fix|build)\b.*\b("
    r"quicksort|mergesort|binary search|linked list|binary tree|leetcode|algorithm|python|javascript|"
    r"typescript|java|c\+\+|rust|go"
    r")\b",
    re.IGNORECASE,
)
_TALISMAN_CODING_RX = re.compile(
    r"\b(write|implement|code|debug|fix|build|test)\b.*\b("
    r"portfolio|agent|stan|talisman|analyzer|optimizer|sizer|screener|valuation|dcf|"
    r"thesis|market|macro|finance|investment|risk|dashboard|workflow"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_WITHOUT_DOMAIN_RX = re.compile(
    r"^\s*(what do you think|is it good|is this good|thoughts|any thoughts|should i|what about it)[\s?.!]*$",
    re.IGNORECASE,
)
_UPPER_TICKER_RX = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,3})?\b")
_TICKER_STOP_WORDS = {
    "AND",
    "ARE",
    "CAN",
    "DO",
    "FOR",
    "HAS",
    "HOW",
    "MY",
    "THE",
    "THAT",
    "THIS",
    "WHAT",
    "WHO",
    "WHY",
}
_AGENT_META_RX = re.compile(r"\b(what model are you|which model are you|model are you using)\b", re.IGNORECASE)
_BRAND_OR_COMPANY_RX = re.compile(
    r"\b("
    r"apple|amazon|google|alphabet|microsoft|meta|netflix|tesla|nvidia|nvda|amd|intel|"
    r"micron|mu|crowdstrike|crwd|palantir|pltr|oracle|salesforce|adobe|costco|walmart|"
    r"jpmorgan|jpm|goldman|boeing|airbus|uber|spotify|shopify|coinbase"
    r")\b",
    re.IGNORECASE,
)
_SHORT_TEXT_RX = re.compile(r"^[\w .&'-]{1,32}$")


def classify_agent_domain(text: str, screen_context: Any | None = None) -> DomainDecision:
    return analyze_agent_domain(text, screen_context=screen_context).decision


def analyze_agent_domain(text: str, screen_context: Any | None = None) -> AgentDomainClassification:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return AgentDomainClassification("clarify", "empty_message")

    if _CASUAL_RX.match(normalized):
        return AgentDomainClassification("allow", "casual")

    has_screen_context = _has_finance_screen_context(screen_context)
    has_supported_signal = _has_supported_signal(normalized, has_screen_context=has_screen_context)
    has_unsupported_signal = _has_unsupported_signal(normalized)

    if has_supported_signal:
        return AgentDomainClassification(
            "allow",
            "supported_domain_with_unsupported_request" if has_unsupported_signal else "supported_domain",
            contains_unsupported_request=has_unsupported_signal,
        )

    if has_unsupported_signal:
        return AgentDomainClassification("block", "unsupported_domain")

    if has_screen_context and _looks_like_followup(normalized):
        return AgentDomainClassification("allow", "screen_context_followup")

    if _QUESTION_WITHOUT_DOMAIN_RX.match(normalized) or _looks_like_short_ambiguous_text(normalized):
        return AgentDomainClassification("clarify", "ambiguous_without_domain_signal")

    return AgentDomainClassification("block", "outside_supported_domain")


def _has_supported_signal(text: str, *, has_screen_context: bool) -> bool:
    if _SUPPORTED_RX.search(text):
        return True
    if _TALISMAN_CODING_RX.search(text):
        return True
    if _BRAND_OR_COMPANY_RX.search(text):
        return True
    if _AGENT_META_RX.search(text):
        return True
    if _has_ticker_signal(text):
        return True
    return has_screen_context and _looks_like_followup(text)


def _has_unsupported_signal(text: str) -> bool:
    if _UNSUPPORTED_RX.search(text):
        return not _TALISMAN_CODING_RX.search(text)
    return bool(_GENERIC_CODING_RX.search(text) and not _TALISMAN_CODING_RX.search(text))


def _has_ticker_signal(text: str) -> bool:
    return any(token not in _TICKER_STOP_WORDS for token in _UPPER_TICKER_RX.findall(text))


def _has_finance_screen_context(screen_context: Any | None) -> bool:
    if screen_context is None:
        return False
    route = str(getattr(screen_context, "route", "") or "").lower()
    page_name = str(getattr(screen_context, "page_name", "") or getattr(screen_context, "pageName", "") or "").lower()
    ticker = str(getattr(screen_context, "ticker", "") or "").strip()
    summary = str(getattr(screen_context, "summary", "") or "").lower()
    if ticker:
        return True
    joined = " ".join([route, page_name, summary])
    return bool(_SUPPORTED_RX.search(joined))


def _looks_like_followup(text: str) -> bool:
    return bool(
        re.search(
            r"\b(this|that|it|these|those|here|there|same|above|current|why|how|what|should|explain|compare)\b",
            text,
            re.IGNORECASE,
        )
    )


def _looks_like_short_ambiguous_text(text: str) -> bool:
    if not _SHORT_TEXT_RX.match(text):
        return False
    words = [part for part in re.split(r"\s+", text.strip()) if part]
    if len(words) > 4:
        return False
    if _UPPER_TICKER_RX.fullmatch(text.strip()):
        return False
    if _BRAND_OR_COMPANY_RX.search(text):
        return False
    return True
