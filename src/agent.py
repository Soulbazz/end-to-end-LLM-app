"""Tool-calling agent for TechGear Solutions (Phase 3).

Observe -> Think -> Act loop (Lecture 10) over two orthogonal tools:
``search_docs`` (hybrid RAG over policies) and ``run_sql`` (guarded text-to-SQL).
Falls back to a deterministic local router when no usable LLM endpoint exists.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.retriever import format_passages_for_prompt, search_hybrid
from src.sql_tool import execute_guarded_sql, get_db_schema

load_dotenv()

PLACEHOLDER_KEY = "your_proxy_api_key_here"
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-E4B-it")


def _make_client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY")
    if not key or key == PLACEHOLDER_KEY:
        return None
    return OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL"), timeout=30.0, max_retries=1)


client = _make_client()


class StructuredAgentResponse(BaseModel):
    answer: str
    citations: list[str]
    tools_used: list[str]
    confidence: str  # "high" | "medium" | "low"


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search TechGear company policies, warranty rules, return procedures, and hardware "
                "troubleshooting guides. Call this ONLY when you need factual policy or troubleshooting "
                "information not already in the conversation. Do NOT call for greetings, chit-chat, or "
                "database stock/sales numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "mode": {"type": "string", "enum": ["hybrid"], "description": "Retrieval mode (default hybrid)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run a read-only SQLite SELECT query over the TechGear database tables "
                "(`products(sku, name, category, price, stock, warranty_months)` and "
                "`sales(order_id, sku, region, units, revenue, order_date)`). Call this ONLY for product "
                "prices, inventory stock counts, or sales/revenue aggregations."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql_query": {"type": "string", "description": "A single SQLite SELECT statement."}},
                "required": ["sql_query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are the TechGear Solutions assistant (electronics & IT hardware store).
Tools: search_docs for policies/troubleshooting; run_sql for prices, stock and sales numbers.
Do not call tools for greetings or chit-chat. If SQL fails, read the error and fix the query.
Ground answers ONLY in tool output; cite policy passages like [doc_04] and SQL as sql:<table>.
Database schema:
{schema}

When you are done, reply with ONLY a JSON object:
{{"answer": str, "citations": [str], "tools_used": [str], "confidence": "high"|"medium"|"low"}}"""


# ----------------------------- tool implementations -----------------------------
def search_docs(query: str, mode: str = "hybrid") -> str:
    docs = search_hybrid(query, top_k=4)
    return format_passages_for_prompt(docs) if docs else "No matching documents found."


def run_sql(sql_query: str) -> str:
    return json.dumps(execute_guarded_sql(sql_query), default=str)


def _dispatch(name: str, args: Dict[str, Any]) -> str:
    try:
        if name == "search_docs":
            return search_docs(str(args.get("query", "")), args.get("mode", "hybrid"))
        if name == "run_sql":
            return run_sql(str(args.get("sql_query", "")))
        return f"Error: unknown tool '{name}'."
    except Exception as exc:
        return f"Error: tool '{name}' failed: {type(exc).__name__}: {exc}"


# ----------------------------- helpers -----------------------------
_DOC_ID = re.compile(r"\[(doc_\d+)\]")
_TABLES = ("products", "sales")


def _citations_from_trace(trace: List[Dict[str, Any]]) -> List[str]:
    """Citations that are actually supported by tool output."""
    cites: List[str] = []
    for t in trace:
        for c in t.get("citations") or []:
            if c not in cites:
                cites.append(c)
    return cites


def _trace_citations(tool: str, args: Dict[str, Any], result: str) -> List[str]:
    if tool == "search_docs":
        return list(dict.fromkeys(_DOC_ID.findall(result)))
    if tool == "run_sql":
        sql = str(args.get("sql_query", "")).lower()
        return [f"sql:{t}" for t in _TABLES if re.search(rf"\b{t}\b", sql)]
    return []


def _parse_final(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _finalize(raw_text: str, trace: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """Validate against StructuredAgentResponse, keeping only supported citations."""
    supported = _citations_from_trace(trace)
    tools_used = list(dict.fromkeys(t["tool"] for t in trace))
    data = _parse_final(raw_text) or {"answer": (raw_text or "").strip() or "I could not produce an answer.",
                                      "confidence": "low"}
    cites = [c for c in (data.get("citations") or []) if isinstance(c, str) and c in supported]
    if not cites:
        cites = supported
    conf = data.get("confidence") if data.get("confidence") in ("high", "medium", "low") else (
        "medium" if trace else "high")
    try:
        resp = StructuredAgentResponse(answer=str(data.get("answer", "")), citations=cites,
                                       tools_used=tools_used, confidence=conf)
    except ValidationError:
        resp = StructuredAgentResponse(answer=str(raw_text), citations=cites, tools_used=tools_used,
                                       confidence="low")
    out = resp.model_dump()
    out["trace"] = trace
    out["mode"] = mode
    return out


# ----------------------------- offline deterministic router -----------------------------
_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|howdy)\b[\s!.,?a-z']{0,40}$",
                       re.IGNORECASE)
_SQL_HINT = re.compile(r"\b(stock|inventory|in stock|price|cost how much|how much is|revenue|sales|sold|units|"
                       r"best[- ]selling|total|average|how many)\b", re.IGNORECASE)
_REGIONS = ("north", "south", "east", "west", "central")


def _route_sql(question: str) -> str:
    q = question.lower()
    region = next((r for r in _REGIONS if re.search(rf"\b{r}\b", q)), None)
    sku = re.search(r"\b(sku-[a-z0-9-]+|chg-pd-140w)\b", q)
    if re.search(r"revenue|sales|sold|units", q):
        if region:
            return f"SELECT region, SUM(units) AS units, SUM(revenue) AS revenue FROM sales WHERE region = '{region.title()}' GROUP BY region"
        return "SELECT region, SUM(units) AS units, SUM(revenue) AS revenue FROM sales GROUP BY region ORDER BY revenue DESC"
    if sku:
        return f"SELECT sku, name, price, stock FROM products WHERE lower(sku) = '{sku.group(1)}'"
    if re.search(r"stock|inventory|how many", q):
        return "SELECT sku, name, stock FROM products ORDER BY stock ASC"
    return "SELECT sku, name, price FROM products ORDER BY price DESC"


def _offline_agent(question: str) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []
    if _GREETING.match(question):
        answer = "Hello! I can help with TechGear policies, troubleshooting, and product stock or sales data. What do you need?"
        return _finalize(json.dumps({"answer": answer, "citations": [], "confidence": "high"}), trace, "offline")

    if _SQL_HINT.search(question) and not re.search(r"\b(warranty|return|refund|shipping|policy|error|err-)", question, re.I):
        tool, args = "run_sql", {"sql_query": _route_sql(question)}
    else:
        tool, args = "search_docs", {"query": question, "mode": "hybrid"}
    result = _dispatch(tool, args)
    cites = _trace_citations(tool, args, result)
    trace.append({"step": 1, "tool": tool, "args": args, "result_preview": result[:300], "citations": cites})

    if tool == "run_sql":
        data = json.loads(result)
        if data.get("ok"):
            lines = [", ".join(f"{c}={v}" for c, v in zip(data["columns"], row)) for row in data["rows"][:10]]
            answer = "Query result:\n" + "\n".join(lines)
            conf = "high"
        else:
            answer, conf = f"The query failed: {data.get('error')}", "low"
    else:
        top = result.split("\n\n")[0]
        answer = f"Most relevant policy passage: {top}"
        cites = [c for c in cites if f"[{c}]" in top]
        conf = "medium"
    return _finalize(json.dumps({"answer": answer, "citations": cites, "confidence": conf}), trace, "offline")


# ----------------------------- LLM agent loop -----------------------------
def _llm_agent(question: str, history: Optional[list], max_steps: int) -> Dict[str, Any]:
    assert client is not None
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT.format(schema=get_db_schema())}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})
    trace: List[Dict[str, Any]] = []
    last_sig: Optional[str] = None
    final_text: Optional[str] = None
    force_final = False

    for step in range(1, max_steps + 1):
        kwargs: Dict[str, Any] = dict(model=MODEL_NAME, messages=messages, temperature=0)
        if not force_final:
            kwargs.update(tools=TOOLS_SPEC, tool_choice="auto")
        msg = client.chat.completions.create(**kwargs).choices[0].message
        if not msg.tool_calls or force_final:
            final_text = msg.content or ""
            break

        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in msg.tool_calls],
        })
        repeated = False
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise json.JSONDecodeError("arguments must be an object", call.function.arguments, 0)
            except json.JSONDecodeError as exc:
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"Error: invalid JSON arguments ({exc}). Retry with valid JSON."})
                continue
            sig = name + json.dumps(args, sort_keys=True)
            if sig == last_sig:
                repeated = True
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": "Error: duplicate call. Answer now using the results you already have."})
                continue
            last_sig = sig
            result = _dispatch(name, args)
            trace.append({"step": step, "tool": name, "args": args, "result_preview": result[:300],
                          "citations": _trace_citations(name, args, result)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if repeated:
            force_final = True
        if force_final or step == max_steps - 1:
            force_final = True
            messages.append({"role": "user", "content": "Stop calling tools. Give your final JSON answer now."})

    if final_text is None:  # loop ended without a final message
        messages.append({"role": "user", "content": "Give your final JSON answer now."})
        final_text = client.chat.completions.create(model=MODEL_NAME, messages=messages,
                                                    temperature=0).choices[0].message.content or ""
    return _finalize(final_text, trace, "llm")


def run_agent(question: str, history: Optional[list] = None, max_steps: int = 5) -> Dict[str, Any]:
    """Answer ``question``; returns StructuredAgentResponse fields plus ``trace`` and ``mode``."""
    if client is None:
        return _offline_agent(question)
    try:
        return _llm_agent(question, history, max_steps)
    except Exception as exc:  # network/auth/proxy failure -> deterministic fallback
        print(f"[agent] LLM unavailable ({type(exc).__name__}); using offline router.")
        return _offline_agent(question)


if __name__ == "__main__":
    tests = [
        ("policy", "How do I recover my motherboard with BIOS-FL-202?", "search_docs"),
        ("sql", "What is the total sales revenue by region?", "run_sql"),
        ("greeting", "Hello!", None),
    ]
    failed = 0
    print(f"Mode: {'LLM (' + MODEL_NAME + ')' if client else 'offline (no API key)'}")
    for label, q, expected_tool in tests:
        out = run_agent(q)
        used = out["tools_used"]
        ok = (expected_tool in used) if expected_tool else (not used)
        failed += 0 if ok else 1
        print(f"\n[{'PASS' if ok else 'FAIL'}] {label}: {q}")
        print(f"  mode={out['mode']} tools_used={used} citations={out['citations']} confidence={out['confidence']}")
        print(f"  answer: {out['answer'][:300]}")
        for t in out["trace"]:
            print(f"  trace: step={t['step']} {t['tool']} {t['args']}")
    print(f"\n{'ALL TESTS PASSED' if not failed else str(failed) + ' TEST(S) FAILED'}")
    raise SystemExit(1 if failed else 0)
