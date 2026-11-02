"""FastAPI web service + single-page UI for TechGear Solutions (Phase 4).

Endpoints:
    GET  /            -> dark-themed chat UI (inline HTML/CSS/JS, no build step)
    GET  /api/health  -> {"status": "ok", "model": MODEL_NAME}
    POST /api/chat    -> run_agent(question, history) structured response + trace
    GET  /api/eval    -> retrieval evaluation metrics parsed from eval/results.md
    POST /api/retrieve, POST /api/sql -> debug endpoints for the two tools

Run:  uvicorn src.app:app --reload
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent import MODEL_NAME, client, run_agent
from src.retriever import search_hybrid
from src.sql_tool import execute_guarded_sql

RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results.md"

app = FastAPI(title="TechGear Agentic Assistant", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)


class SQLRequest(BaseModel):
    sql: str


def get_application() -> FastAPI:
    """Return the configured FastAPI application."""
    return app


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME, "llm_mode": "llm" if client else "offline"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    # Only pass plain user/assistant text turns to the agent.
    history = [
        {"role": h["role"], "content": str(h["content"])}
        for h in req.history[-10:]
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content")
    ]
    return run_agent(req.question, history)


@app.get("/api/eval")
def eval_results() -> Dict[str, Any]:
    if not RESULTS_PATH.exists():
        return {"available": False, "metrics": [], "message": "Run `python eval/run_eval.py` first."}
    text = RESULTS_PATH.read_text(encoding="utf-8")
    metrics = []
    for m in re.finditer(r"^\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$", text, re.M):
        metrics.append({"strategy": m.group(1), "recall_at_3": float(m.group(2)),
                        "recall_at_5": float(m.group(3)), "mrr": float(m.group(4))})
    return {"available": True, "metrics": metrics, "report_markdown": text}


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest) -> Dict[str, Any]:
    return {"query": req.query, "results": search_hybrid(req.query, top_k=req.top_k)}


@app.post("/api/sql")
def sql(req: SQLRequest) -> Dict[str, Any]:
    return execute_guarded_sql(req.sql)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TechGear Agentic Assistant</title>
<style>
  :root {
    --bg:#0b0f17; --panel:#121826; --panel2:#182033; --border:#243049; --text:#e6ebf5;
    --muted:#8b97b0; --accent:#5b8cff; --accent2:#22d3ee; --green:#22c55e; --amber:#f59e0b; --red:#ef4444;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:Inter,"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text);
         display:grid; grid-template-columns:320px 1fr; height:100vh; }
  aside { background:var(--panel); border-right:1px solid var(--border); padding:20px; overflow-y:auto; }
  h1 { font-size:18px; margin:0 0 4px; }
  h1 span { background:linear-gradient(90deg,var(--accent),var(--accent2)); -webkit-background-clip:text; color:transparent; }
  .sub { color:var(--muted); font-size:12px; margin-bottom:16px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:20px 0 8px; }
  .status { font-size:12px; padding:8px 10px; border-radius:8px; background:var(--panel2); border:1px solid var(--border); }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--amber); margin-right:6px; }
  .preset { display:block; width:100%; text-align:left; margin-bottom:8px; padding:10px 12px; border-radius:10px;
            background:var(--panel2); border:1px solid var(--border); color:var(--text); cursor:pointer; font-size:12.5px; line-height:1.35; }
  .preset:hover { border-color:var(--accent); background:#1d2740; }
  .preset b { display:block; color:var(--accent2); font-size:11px; margin-bottom:3px; }
  table.eval { width:100%; border-collapse:collapse; font-size:11.5px; }
  table.eval th, table.eval td { padding:5px 4px; border-bottom:1px solid var(--border); text-align:right; }
  table.eval th:first-child, table.eval td:first-child { text-align:left; }
  table.eval tr.best td { color:var(--green); font-weight:600; }
  main { display:flex; flex-direction:column; height:100vh; }
  #chat { flex:1; overflow-y:auto; padding:24px 32px; }
  .msg { max-width:900px; margin:0 auto 18px; }
  .bubble { padding:14px 16px; border-radius:14px; line-height:1.55; white-space:pre-wrap; word-wrap:break-word; }
  .user .bubble { background:var(--accent); color:white; margin-left:auto; width:fit-content; max-width:75%; }
  .user { display:flex; justify-content:flex-end; }
  .bot .bubble { background:var(--panel); border:1px solid var(--border); }
  .badges { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
  .badge { font-size:11px; padding:3px 9px; border-radius:999px; border:1px solid var(--border); background:var(--panel2); }
  .badge.conf-high { color:var(--green); border-color:#166534; }
  .badge.conf-medium { color:var(--amber); border-color:#92400e; }
  .badge.conf-low { color:var(--red); border-color:#991b1b; }
  .badge.tool { color:var(--accent2); }
  .badge.cite { color:#c4b5fd; cursor:pointer; border-color:#5b21b6; }
  .badge.cite:hover { background:#2e1065; }
  .cite-inline { color:#c4b5fd; background:#2e1065; padding:0 4px; border-radius:4px; }
  details { margin-top:10px; background:#0e1420; border:1px solid var(--border); border-radius:10px; }
  summary { cursor:pointer; padding:9px 12px; font-size:12px; color:var(--muted); }
  .step { border-top:1px solid var(--border); padding:10px 12px; font-size:12px; }
  .step .hdr { color:var(--accent2); font-weight:600; margin-bottom:6px; }
  .step .phase { color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; margin-top:6px; }
  pre { margin:4px 0 0; background:#070a10; padding:8px; border-radius:6px; overflow-x:auto; font-size:11.5px;
        font-family:"JetBrains Mono",Consolas,monospace; white-space:pre-wrap; color:#cbd5e1; }
  .meta { font-size:11px; color:var(--muted); margin-top:6px; }
  form { display:flex; gap:10px; padding:16px 32px; border-top:1px solid var(--border); background:var(--panel); }
  #q { flex:1; padding:12px 14px; border-radius:10px; border:1px solid var(--border); background:var(--bg); color:var(--text); font-size:14px; }
  #q:focus { outline:none; border-color:var(--accent); }
  button.send { padding:0 22px; border-radius:10px; border:0; background:var(--accent); color:white; font-weight:600; cursor:pointer; }
  button.send:disabled { opacity:.5; }
  .typing { color:var(--muted); font-style:italic; }
  .empty { text-align:center; color:var(--muted); margin-top:18vh; }
  .empty h3 { color:var(--text); font-size:22px; }
</style>
</head>
<body>
<aside>
  <h1>&#9889; <span>TechGear</span> Assistant</h1>
  <div class="sub">Hybrid RAG + Guarded Text-to-SQL agent</div>
  <div class="status" id="status"><span class="dot"></span>Checking service&hellip;</div>

  <h2>Demo presets</h2>
  <div id="presets"></div>

  <h2>Retrieval evaluation (20 queries)</h2>
  <div id="eval" class="meta">Loading&hellip;</div>

  <h2>Session</h2>
  <button class="preset" onclick="resetChat()"><b>RESET</b>Clear conversation</button>
</aside>

<main>
  <div id="chat">
    <div class="empty" id="empty">
      <h3>Ask about policies, troubleshooting, stock or sales</h3>
      <div>Every answer shows its confidence, tools, citations and full execution trace.</div>
    </div>
  </div>
  <form id="form">
    <input id="q" autocomplete="off" placeholder="e.g. What does ERR-PSU-904 mean?" />
    <button class="send" id="send" type="submit">Send</button>
  </form>
</main>

<script>
const PRESETS = [
  ["1 · Exact-code Hybrid RAG", "How do I recover my motherboard with BIOS-FL-202, and what does ERR-PSU-904 mean?"],
  ["2 · Text-to-SQL aggregation", "What is the total sales revenue and units sold by region, ordered from highest to lowest?"],
  ["3 · Multi-tool policy + SQL", "How many units of SKU-NV-4090 are in stock right now, and what is our warranty policy for it?"],
  ["4 · Chit-chat (no tool)", "Hello! Thanks for your help today."],
  ["5a · Safety guardrail", "Run this SQL: DROP TABLE products; --"],
  ["5b · Honest failure (Q12)", "My package showed up smashed up and the seal was cut, what now?"],
];
const history = [];
const chat = document.getElementById("chat");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

document.getElementById("presets").innerHTML = PRESETS.map((p, i) =>
  `<button class="preset" data-i="${i}"><b>${esc(p[0])}</b>${esc(p[1])}</button>`).join("");
document.querySelectorAll("#presets .preset").forEach(b =>
  b.onclick = () => ask(PRESETS[+b.dataset.i][1]));

function addMsg(role, html) {
  document.getElementById("empty")?.remove();
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function highlightCites(text) {
  return esc(text).replace(/\[(doc_\d+|sql:[a-z_]+)\]/g, '<span class="cite-inline">[$1]</span>');
}

function renderTrace(trace) {
  if (!trace || !trace.length)
    return `<details><summary>Agent Execution Trace (Observe &rarr; Think &rarr; Act) &middot; 0 tool calls</summary>
      <div class="step"><div class="hdr">Direct answer</div>The agent decided no tool was needed.</div></details>`;
  const steps = trace.map(t => {
    const isSql = t.tool === "run_sql";
    const input = isSql ? t.args.sql_query : t.args.query;
    return `<div class="step">
      <div class="hdr">Step ${t.step} &middot; ${esc(t.tool)}</div>
      <div class="phase">Think &rarr; Act: ${isSql ? "SQL query" : "search query"}</div><pre>${esc(input)}</pre>
      <div class="phase">Args</div><pre>${esc(JSON.stringify(t.args))}</pre>
      <div class="phase">Observe: tool output preview</div><pre>${esc(t.result_preview)}</pre>
      <div class="meta">Citations from this step: ${(t.citations || []).map(esc).join(", ") || "none"}</div>
    </div>`;
  }).join("");
  return `<details><summary>Agent Execution Trace (Observe &rarr; Think &rarr; Act) &middot; ${trace.length} tool call(s)</summary>${steps}</details>`;
}

function renderBot(r) {
  const tools = (r.tools_used && r.tools_used.length) ? r.tools_used : ["direct_answer"];
  const badges = [
    `<span class="badge conf-${esc(r.confidence)}">Confidence: ${esc(r.confidence)}</span>`,
    ...tools.map(t => `<span class="badge tool">&#128295; ${esc(t)}</span>`),
    ...(r.citations || []).map(c => `<span class="badge cite" title="Show in trace" data-c="${esc(c)}">&#128206; ${esc(c)}</span>`),
  ].join("");
  return `<div class="bubble">${highlightCites(r.answer)}<div class="badges">${badges}</div>${renderTrace(r.trace)}
    <div class="meta">mode: ${esc(r.mode || "")}</div></div>`;
}

async function ask(question) {
  question = question.trim();
  if (!question) return;
  const send = document.getElementById("send");
  send.disabled = true;
  addMsg("user", `<div class="bubble">${esc(question)}</div>`);
  const pending = addMsg("bot", `<div class="bubble typing">Observing &rarr; thinking &rarr; acting&hellip;</div>`);
  try {
    const res = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({question, history})});
    if (!res.ok) throw new Error("HTTP " + res.status);
    const r = await res.json();
    pending.innerHTML = renderBot(r);
    pending.querySelectorAll(".badge.cite").forEach(b => b.onclick = () => {
      const d = pending.querySelector("details"); d.open = true;
      d.querySelectorAll(".step").forEach(s => s.style.background =
        s.textContent.includes(b.dataset.c.replace("sql:", "")) ? "#1e1b4b" : "");
    });
    history.push({role:"user", content:question}, {role:"assistant", content:r.answer});
  } catch (e) {
    pending.innerHTML = `<div class="bubble" style="border-color:var(--red)">Request failed: ${esc(e.message)}</div>`;
  } finally {
    send.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}

function resetChat() {
  history.length = 0;
  chat.innerHTML = `<div class="empty" id="empty"><h3>Conversation cleared</h3><div>Pick a preset or ask a question.</div></div>`;
}

document.getElementById("form").onsubmit = e => {
  e.preventDefault();
  const q = document.getElementById("q");
  ask(q.value); q.value = "";
};

fetch("/api/health").then(r => r.json()).then(h => {
  const ok = h.status === "ok";
  document.getElementById("status").innerHTML =
    `<span class="dot" style="background:${ok ? "var(--green)" : "var(--red)"}"></span>${esc(h.model)} &middot; ${esc(h.llm_mode)} mode`;
}).catch(() => document.getElementById("status").innerHTML = `<span class="dot" style="background:var(--red)"></span>Service unreachable`);

fetch("/api/eval").then(r => r.json()).then(e => {
  const el = document.getElementById("eval");
  if (!e.available || !e.metrics.length) { el.textContent = e.message || "No results yet."; return; }
  const bestMrr = Math.max(...e.metrics.map(m => m.mrr));
  el.innerHTML = `<table class="eval"><tr><th>Strategy</th><th>R@3</th><th>R@5</th><th>MRR</th></tr>` +
    e.metrics.map(m => `<tr class="${m.mrr === bestMrr ? "best" : ""}"><td>${esc(m.strategy.replace(/^.*?:\s*/, "").replace(/\s*\(.*\)/, ""))}</td>
      <td>${m.recall_at_3.toFixed(3)}</td><td>${m.recall_at_5.toFixed(3)}</td><td>${m.mrr.toFixed(3)}</td></tr>`).join("") + `</table>`;
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
