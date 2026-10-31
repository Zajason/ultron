"""What a turn cost, from the API's usage block. List prices in USD per million tokens (October 2026);
check anthropic.com/pricing if they look stale. Cache writes cost 1.25x input for the 5-minute
TTL and 2x for the 1-hour TTL; cache reads 0.1x (Haiku, Sonnet) or 0.05x (Opus 5.5)."""

from __future__ import annotations

from typing import Any

PRICES = {  # input, output, cache read
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
WEB_SEARCH_USD = 10.00 / 1000  # per search


def usd(model: str, u: Any) -> float:
    p_in, p_out, p_read = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    cc = getattr(u, "cache_creation", None)
    w1h = getattr(cc, "ephemeral_1h_input_tokens", 0) or 0 if cc else 0
    w5m = getattr(cc, "ephemeral_5m_input_tokens", 0) or 0 if cc else 0
    if not cc:
        w5m = u.cache_creation_input_tokens or 0
    stu = getattr(u, "server_tool_use", None)
    searches = (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
    return (u.input_tokens * p_in + u.output_tokens * p_out + (u.cache_read_input_tokens or 0) * p_read
            + w5m * p_in * 1.25 + w1h * p_in * 2.0) / 1e6 + searches * WEB_SEARCH_USD
