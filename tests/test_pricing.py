from types import SimpleNamespace as NS

from ultron.pricing import usd


def test_haiku_cached_turn():
    u = NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=3500, cache_creation_input_tokens=0,
           cache_creation=NS(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0), server_tool_use=None)
    assert abs(usd("claude-haiku-5-5", u) - (100 * 0.10 + 50 * 0.50 + 3500 * 0.01) / 1e6) < 1e-12


def test_writes_and_searches():
    u = NS(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=1000,
           cache_creation=NS(ephemeral_1h_input_tokens=1000, ephemeral_5m_input_tokens=0),
           server_tool_use=NS(web_search_requests=2, web_fetch_requests=0))
    assert abs(usd("claude-sonnet-5-5", u) - (1000 * 2.0 * 2 / 1e6 + 0.02)) < 1e-12
