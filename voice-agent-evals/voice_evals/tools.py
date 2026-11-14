"""Mock tools for the agent under test: deterministic fixture data, no network.

The registry only knows names, JSON schemas and plain callables, so the same agent could call a
real MCP server later without changing the agent or the scorers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import mlflow
from mlflow.entities import SpanType

from .normalize import normalize_tokens
from .tracing import set_inputs, set_outputs

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "data" / "fixtures.json"
FIXTURES = json.loads(FIXTURES_PATH.read_text())

_STOPWORDS = {
    "a", "an", "the", "of", "in", "for", "and", "or", "to", "is", "what", "does", "do", "come",
    "it", "with", "on", "my", "me", "about", "product",
}


def _key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def search_products(query: str) -> dict:
    """Find catalog products by name or description. Returns SKUs and one-line summaries."""
    wanted = set(normalize_tokens(query)) - _STOPWORDS
    scored = []
    for sku, p in FIXTURES["products"].items():
        have = set(normalize_tokens(f"{p['name']} {p['summary']} {sku}")) - _STOPWORDS
        scored.append((len(wanted & have), sku, p))
    best = max((s for s, _, _ in scored), default=0)
    hits = [
        {"sku": sku, "name": p["name"], "summary": p["summary"]}
        for s, sku, p in sorted(scored, key=lambda t: -t[0])
        if s > 0 and s >= best - 1
    ][:3]
    return {"results": hits} if hits else {"results": [], "message": f"No products match '{query}'."}


def get_product(sku: str) -> dict:
    """Full product record for one SKU: price, stock, sizes, pack quantities."""
    for key, p in FIXTURES["products"].items():
        if _key(key) == _key(sku):
            return {"sku": key, **{k: v for k, v in p.items() if k != "summary"}}
    return {"error": f"No product found for SKU '{sku}'."}


def get_order_status(order_id: str) -> dict:
    """Status, ship dates and line items for one order."""
    digits = re.sub(r"\D", "", str(order_id))
    for key, order in FIXTURES["orders"].items():
        if re.sub(r"\D", "", key) == digits and digits:
            return {"order_id": key, **order}
    return {"error": f"No order found with ID '{order_id}'."}


def escalate_to_human(reason: str) -> dict:
    """Hand the conversation to a person, e.g. for complaints or anything the agent can't handle."""
    ticket = "ESC-" + hashlib.sha1(reason.encode()).hexdigest()[:6].upper()
    return {
        "ticket_id": ticket,
        "status": "queued",
        "message": "A support specialist will call back within one business day.",
    }


def _schema(fn: Callable, arg: str, arg_description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": fn.__doc__,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string", "description": arg_description}},
                "required": [arg],
            },
        },
    }


class ToolRegistry:
    """Names, schemas and callables. Every call is recorded as a TOOL span."""

    def __init__(self, tools: dict[str, tuple[Callable, dict]]):
        self._tools = tools

    def schemas(self) -> list[dict]:
        return [schema for _, schema in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str]) -> "ToolRegistry":
        return ToolRegistry({n: t for n, t in self._tools.items() if n in names})

    def call(self, name: str, args: dict) -> dict:
        with mlflow.start_span(name=name, span_type=SpanType.TOOL) as span:
            set_inputs(span, args)
            if name not in self._tools:
                result = {"error": f"Unknown tool '{name}'."}
            else:
                try:
                    result = self._tools[name][0](**args)
                except TypeError as e:
                    result = {"error": f"Bad arguments for {name}: {e}"}
            set_outputs(span, result)
        return result


def default_tools() -> ToolRegistry:
    return ToolRegistry(
        {
            "search_products": (search_products, _schema(search_products, "query", "Product name or description, e.g. 'TriLock femoral stem'")),
            "get_product": (get_product, _schema(get_product, "sku", "Product SKU, e.g. 'TLF-2210'")),
            "get_order_status": (get_order_status, _schema(get_order_status, "order_id", "Order ID, e.g. 'ORD-58213' or '58213'")),
            "escalate_to_human": (escalate_to_human, _schema(escalate_to_human, "reason", "Short reason for the hand-off")),
        }
    )
