"""Guarded Text-to-SQL tool for data/techgear.db (Phase 3).

Defense in depth (Lecture 11): regex SELECT-only gate + LIMIT enforcement +
read-only SQLite connection (``mode=ro``). Nothing here raises into the agent
loop; failures come back as ``{"ok": False, "error": ...}``.
"""

import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "techgear.db"
INIT_DB_SCRIPT = ROOT / "data" / "init_db.py"

_BANNED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|"
    r"vacuum|reindex|truncate|grant|revoke)\b",
    re.IGNORECASE,
)
_SELECT_START = re.compile(r"^\s*select\b", re.IGNORECASE)
_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


def is_safe_select(sql: str) -> bool:
    """Return True only for a single, comment-free, read-only SELECT statement."""
    if not isinstance(sql, str) or not sql.strip():
        return False
    if "--" in sql or "/*" in sql or "*/" in sql:
        return False
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    if ";" in stripped:  # stacked statements
        return False
    if not _SELECT_START.match(stripped):
        return False
    if _BANNED.search(stripped):
        return False
    return True


def clean_sql(text: str) -> str:
    """Strip markdown ```sql fences and trailing semicolons/whitespace."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def ensure_limit(sql: str, default_limit: int = 50) -> str:
    """Append ``LIMIT <default_limit>`` when the query has no LIMIT clause."""
    if _LIMIT.search(sql):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {default_limit}"


def _ensure_db() -> None:
    if not DB_PATH.exists():
        subprocess.run([sys.executable, str(INIT_DB_SCRIPT)], check=True, capture_output=True)


def get_db_schema() -> str:
    """Return the CREATE TABLE statements of the database (auto-initializing it)."""
    _ensure_db()
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0] for r in rows if r[0])


def execute_guarded_sql(sql: str) -> Dict[str, Any]:
    """Clean, validate, limit and execute ``sql`` on a read-only connection."""
    original = sql
    try:
        cleaned = clean_sql(sql)
        if not is_safe_select(cleaned):
            return {"ok": False, "sql": cleaned or original,
                    "error": "Rejected: only a single read-only SELECT statement without comments is allowed."}
        final_sql = ensure_limit(cleaned)
        _ensure_db()
        conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
        try:
            cur = conn.execute(final_sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [list(r) for r in cur.fetchall()]
        finally:
            conn.close()
        return {"ok": True, "sql": final_sql, "columns": cols, "rows": rows}
    except Exception as exc:  # never leak into the agent loop
        return {"ok": False, "sql": original if isinstance(original, str) else str(original),
                "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    failures = 0

    def check(label: str, got: Any, expected: Any) -> None:
        global failures
        ok = got == expected
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print("--- is_safe_select ---")
    check("simple SELECT allowed", is_safe_select("SELECT * FROM products"), True)
    check("trailing ; allowed", is_safe_select("select sku from products;"), True)
    check("aggregation allowed", is_safe_select("SELECT region, SUM(revenue) FROM sales GROUP BY region"), True)
    check("DROP TABLE blocked", is_safe_select("DROP TABLE products"), False)
    check("stacked ; blocked", is_safe_select("SELECT 1; DROP TABLE products"), False)
    check("-- comment blocked", is_safe_select("SELECT * FROM products -- hi"), False)
    check("/* */ comment blocked", is_safe_select("SELECT /* x */ 1"), False)
    check("DELETE inside SELECT blocked", is_safe_select("SELECT * FROM products WHERE 1=1 OR delete"), False)
    check("non-string blocked", is_safe_select(None), False)  # type: ignore[arg-type]
    check("empty blocked", is_safe_select("   "), False)

    print("--- helpers ---")
    check("clean_sql strips fence", clean_sql("```sql\nSELECT 1;\n```"), "SELECT 1")
    check("ensure_limit appends", ensure_limit("SELECT * FROM products"), "SELECT * FROM products LIMIT 50")
    check("ensure_limit keeps existing", ensure_limit("SELECT * FROM products LIMIT 3"), "SELECT * FROM products LIMIT 3")

    print("--- schema ---")
    schema = get_db_schema()
    check("schema lists both tables", "CREATE TABLE products" in schema and "CREATE TABLE sales" in schema, True)

    print("--- execute_guarded_sql ---")
    r = execute_guarded_sql("```sql\nSELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY total DESC;\n```")
    print(r)
    check("aggregation executes", r["ok"] and len(r["rows"]) == 5, True)
    r = execute_guarded_sql("DROP TABLE products")
    print(r)
    check("DROP rejected", r["ok"], False)
    r = execute_guarded_sql("SELECT * FROM nonexistent")
    print(r)
    check("bad SQL returns error, no raise", r["ok"], False)
    r = execute_guarded_sql("SELECT COUNT(*) FROM products")
    check("products still intact", r["rows"] == [[12]], True)

    print(f"\n{'ALL TESTS PASSED' if not failures else str(failures) + ' TEST(S) FAILED'}")
    sys.exit(1 if failures else 0)
