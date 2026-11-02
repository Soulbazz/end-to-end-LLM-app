"""Generate the 5-to-6 page academic PDF report at report/Capstone_Project_Report.pdf."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String

REPORT_DIR = Path(__file__).resolve().parent
PDF_PATH = REPORT_DIR / "Capstone_Project_Report.pdf"

NAVY = colors.HexColor("#1e293b")
ACCENT = colors.HexColor("#1d4ed8")
TEAL = colors.HexColor("#0f766e")
LIGHT_BG = colors.HexColor("#f8fafc")
CODE_BG = colors.HexColor("#f1f5f9")
BORDER = colors.HexColor("#cbd5e1")
MUTED = colors.HexColor("#64748b")
GREEN_BG = colors.HexColor("#dcfce7")
RED_BG = colors.HexColor("#fee2e2")


class NumberedDocTemplate(SimpleDocTemplate):
    """SimpleDocTemplate that tracks total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_count = 0

    def afterPage(self):
        self.page_count = self.canv.getPageNumber()


def draw_header_footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    # Header
    if doc.page > 1:
        canvas.setFont("Helvetica-Oblique", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, h - 0.5 * inch, "TechGear Solutions: Hybrid RAG & Guarded Text-to-SQL Agentic Assistant")
        canvas.drawRightString(w - 0.75 * inch, h - 0.5 * inch, "67070255 · Capstone Project Report")
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0.75 * inch, h - 0.55 * inch, w - 0.75 * inch, h - 0.55 * inch)
    # Footer
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, 0.6 * inch, w - 0.75 * inch, 0.6 * inch)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.42 * inch, "Building LLM-Powered Applications — Q2 Capstone Report")
    canvas.drawRightString(w - 0.75 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_architecture_drawing() -> Drawing:
    """Create a vector architecture diagram showing the full pipeline."""
    d = Drawing(480, 250)

    def box(x, y, w, h, title, sub1="", sub2="", fill=colors.HexColor("#eff6ff"), stroke=ACCENT):
        d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=1.1))
        d.add(String(x + w / 2, y + h - 13, title, fontName="Helvetica-Bold", fontSize=8.5, fillColor=NAVY, textAnchor="middle"))
        if sub1:
            d.add(String(x + w / 2, y + h - 24, sub1, fontName="Helvetica", fontSize=7.3, fillColor=NAVY, textAnchor="middle"))
        if sub2:
            d.add(String(x + w / 2, y + h - 34, sub2, fontName="Helvetica-Oblique", fontSize=6.8, fillColor=MUTED, textAnchor="middle"))

    def arrow(x1, y1, x2, y2, label=""):
        d.add(Line(x1, y1, x2, y2, strokeColor=NAVY, strokeWidth=1))
        # Small arrowhead at (x2, y2)
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        d.add(Polygon([x2, y2, x2 - 5 * ux + 2.5 * px, y2 - 5 * uy + 2.5 * py, x2 - 5 * ux - 2.5 * px, y2 - 5 * uy - 2.5 * py],
                      fillColor=NAVY, strokeColor=NAVY))
        if label:
            d.add(String((x1 + x2) / 2 + 4, (y1 + y2) / 2 - 2, label, fontName="Helvetica-Bold", fontSize=6.8, fillColor=ACCENT))

    # Top row: User + FastAPI UI
    box(115, 206, 250, 38, "1. User / FastAPI Single-Page Web UI (src/app.py)",
        "GET /  |  POST /api/chat  |  GET /api/eval  |  GET /api/health",
        "Dark UI: Presets, Confidence/Tool/Citation Badges, Expandable Trace",
        fill=colors.HexColor("#f0f9ff"), stroke=ACCENT)

    arrow(240, 206, 240, 186, "")

    # Second row: Agent Loop
    box(75, 142, 330, 44, "2. Observe -> Think -> Act Agent Loop (src/agent.py :: run_agent)",
        "Model: gemma-4-E4B-it (temp=0)  |  max_steps=5  |  Repeat-Call & JSON Guardrails",
        "Includes automatic deterministic Offline Router fallback when API key/network unavailable",
        fill=colors.HexColor("#eef2ff"), stroke=ACCENT)

    # Branches to Tool 1 and Tool 2
    arrow(165, 142, 115, 118, "search_docs()")
    arrow(315, 142, 365, 118, "run_sql()")

    # Tool 1: Hybrid Retriever
    box(8, 56, 222, 62, "3A. Hybrid Retriever (src/retriever.py)",
        "BM25 (k1=1.5, b=0.75, exact hyphen codes)",
        "+ Vector Cosine Sim -> Fused via RRF (k=60)",
        fill=colors.HexColor("#ecfdf5"), stroke=TEAL)
    d.add(String(119, 63, "Corpus: data/corpus.json (20 policy chunks, [doc_01]..[doc_20])",
                 fontName="Helvetica", fontSize=6.8, fillColor=TEAL, textAnchor="middle"))

    # Tool 2: Guarded SQL Tool
    box(250, 56, 222, 62, "3B. Guarded Text-to-SQL (src/sql_tool.py)",
        "clean_sql -> is_safe_select() regex gate + LIMIT 50",
        "+ Read-Only SQLite URI (file:techgear.db?mode=ro)",
        fill=colors.HexColor("#fffbeb"), stroke=colors.HexColor("#b45309"))
    d.add(String(361, 63, "DB: data/techgear.db (12 products, 20 regional sales rows)",
                 fontName="Helvetica", fontSize=6.8, fillColor=colors.HexColor("#b45309"), textAnchor="middle"))

    # Arrows down to Citation Verifier
    arrow(115, 56, 185, 38, "")
    arrow(365, 56, 295, 38, "")

    # Bottom row: Code-verified output
    box(65, 2, 350, 36, "4. Code-Verified Citations & Pydantic StructuredAgentResponse",
        "Intersects LLM citations against actual trace ([doc_XX], sql:<table>) + returns trace",
        fill=colors.HexColor("#f5f3ff"), stroke=colors.HexColor("#6d28d9"))

    return d


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=17.5,
            leading=22, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4
        ),
        "Subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10,
            leading=14, textColor=MUTED, spaceAfter=10
        ),
        "H1": ParagraphStyle(
            "SectionH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=13,
            leading=17, textColor=ACCENT, spaceBefore=10, spaceAfter=5, keepWithNext=True
        ),
        "H2": ParagraphStyle(
            "SectionH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=3, keepWithNext=True
        ),
        "Body": ParagraphStyle(
            "BodyTextJustified", parent=base["Normal"], fontName="Helvetica", fontSize=9.4,
            leading=13.4, textColor=colors.HexColor("#1f2937"), alignment=TA_JUSTIFY, spaceAfter=5
        ),
        "Bullet": ParagraphStyle(
            "BulletItem", parent=base["Normal"], fontName="Helvetica", fontSize=9.3,
            leading=13.2, leftIndent=14, firstLineIndent=-10, textColor=colors.HexColor("#1f2937"),
            alignment=TA_JUSTIFY, spaceAfter=3.5
        ),
        "Formula": ParagraphStyle(
            "FormulaBox", parent=base["Normal"], fontName="Courier-Bold", fontSize=8.8,
            leading=12.5, textColor=NAVY, backColor=CODE_BG, borderColor=BORDER,
            borderWidth=0.6, borderPadding=6, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6
        ),
        "Callout": ParagraphStyle(
            "CalloutBox", parent=base["Normal"], fontName="Helvetica", fontSize=9.0,
            leading=12.8, textColor=NAVY, backColor=LIGHT_BG, borderColor=ACCENT,
            borderWidth=0.8, borderPadding=7, spaceBefore=5, spaceAfter=6
        ),
        "TableCell": ParagraphStyle(
            "TblCell", parent=base["Normal"], fontName="Helvetica", fontSize=8.4,
            leading=11.2, textColor=colors.HexColor("#1f2937"), alignment=TA_LEFT
        ),
        "TableHeader": ParagraphStyle(
            "TblHdr", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.5,
            leading=11.2, textColor=colors.white, alignment=TA_LEFT
        ),
        "Code": ParagraphStyle(
            "CodeBlock", parent=base["Code"], fontName="Courier", fontSize=7.9,
            leading=10.5, textColor=NAVY, backColor=CODE_BG, borderColor=BORDER,
            borderWidth=0.5, borderPadding=6, leftIndent=0, rightIndent=0, spaceBefore=3, spaceAfter=6
        ),
    }
    return styles


def make_table(headers, rows, col_widths, highlight_last=False):
    st = build_styles()
    data = [[Paragraph(h, st["TableHeader"]) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), st["TableCell"]) if not isinstance(c, Paragraph) else c for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]
    if highlight_last:
        cmds.append(("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), GREEN_BG))
    t.setStyle(TableStyle(cmds))
    return t


def generate_pdf() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = NumberedDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="TechGear Solutions: Hybrid RAG & Guarded Text-to-SQL Agentic Assistant",
        author="Student 67070255",
    )
    st = build_styles()
    story = []

    # ==================== PAGE 1: TITLE + SECTION 1 ====================
    story.append(Paragraph("TechGear Solutions: Hybrid RAG &amp; Guarded Text-to-SQL Agentic Assistant", st["Title"]))
    story.append(Paragraph(
        "<b>Course:</b> Building LLM-Powered Applications (Capstone Deliverable Q2) &nbsp;|&nbsp; "
        "<b>Student ID:</b> 67070255 &nbsp;|&nbsp; <b>Model:</b> <font name='Courier'>gemma-4-E4B-it</font> (temperature=0)",
        st["Subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "<b>Executive Summary:</b> This report presents the design, implementation, and quantitative evaluation of an "
        "end-to-end agentic assistant for <b>TechGear Solutions</b>, an electronics and enterprise IT hardware retailer. "
        "The assistant bridges unstructured policy/troubleshooting manuals (<font name='Courier'>data/corpus.json</font>, "
        "20 chunks) and structured operational data (<font name='Courier'>data/techgear.db</font>, 12 hardware products "
        "and 20 regional sales transactions) through an Observe &rarr; Think &rarr; Act tool-calling loop. By fusing "
        "lexical BM25 and vector similarity via Reciprocal Rank Fusion (RRF, <i>k</i>=60), the retrieval engine achieves "
        "<b>0.950 Recall@3, 0.950 Recall@5, and 0.950 MRR</b> on our 20-query evaluation suite—outperforming both "
        "Vector-only (0.842 MRR) and BM25-only (0.835 MRR) baselines while enforcing strict SQL read-only safety gates "
        "and code-verified citation grounding.",
        st["Callout"]
    ))

    story.append(Paragraph("1. Problem Statement &amp; Target Users", st["H1"]))
    story.append(Paragraph("1.1 Real-World Operational Challenge at TechGear Solutions", st["H2"]))
    story.append(Paragraph(
        "TechGear Solutions operates an IT hardware storefront and commercial server support desk where incoming "
        "questions span two incompatible data representations that cannot be served by a single static prompt or "
        "a single retrieval mechanism:",
        st["Body"]
    ))
    story.append(Paragraph(
        "&bull; <b>Unstructured Policy &amp; Diagnostic Manuals (<font name='Courier'>data/corpus.json</font>):</b> "
        "Contains 20 curated policy documents (<font name='Courier'>doc_01</font> to <font name='Courier'>doc_20</font>) "
        "evenly distributed across five operational categories: <font name='Courier'>warranty</font>, "
        "<font name='Courier'>return</font>, <font name='Courier'>shipping</font>, <font name='Courier'>troubleshooting</font>, "
        "and <font name='Courier'>enterprise_sla</font>. Crucially, these documents combine natural-language rules with "
        "exact hardware error codes, regulatory standards, and SKUs—including <font name='Courier'>ERR-PSU-904</font> "
        "(PSU over-current tripping), <font name='Courier'>BIOS-FL-202</font> (USB BIOS flashback recovery), "
        "<font name='Courier'>GPU-coil-77</font> (45 dB inductor acoustic threshold), <font name='Courier'>RMA-TG-88</font> "
        "(14-day return authorization token), <font name='Courier'>PIXEL-ISO-9241</font> (Class II sub-pixel defect rule), "
        "<font name='Courier'>DOA-48HR</font> (48-hour dead-on-arrival swap), <font name='Courier'>CHG-PD-140W</font> "
        "(140W GaN charger thermal recall), <font name='Courier'>RAID-rebuild-05</font> (degraded array parity rebuild), "
        "<font name='Courier'>SLA-TIER-1</font> (4-hour mission-critical dispatch), and <font name='Courier'>SKU-NV-4090</font>.",
        st["Bullet"]
    ))
    story.append(Paragraph(
        "&bull; <b>Structured Relational Inventory &amp; Sales Database (<font name='Courier'>data/techgear.db</font>):</b> "
        "A SQLite database seeded idempotently via <font name='Courier'>data/init_db.py</font> containing two tables: "
        "(1) <font name='Courier'>products(sku, name, category, price, stock, warranty_months)</font> with 12 hardware items "
        "(e.g., <font name='Courier'>SKU-NV-4090</font> at $1,799.99, 15 units in stock, 36-month warranty; "
        "<font name='Courier'>CHG-PD-140W</font> at $89.99, 120 in stock), and "
        "(2) <font name='Courier'>sales(order_id, sku, region, units, revenue, order_date)</font> with 20 order records "
        "across five regions (<font name='Courier'>North</font>, <font name='Courier'>Central</font>, "
        "<font name='Courier'>East</font>, <font name='Courier'>West</font>, <font name='Courier'>South</font>).",
        st["Bullet"]
    ))

    story.append(Paragraph("1.2 Target Users &amp; Key Use Cases", st["H2"]))
    story.append(make_table(
        ["Target User Group", "Representative Questions", "Required System Capability"],
        [
            ["Tier-1 / Tier-2 Support Engineers",
             "\"How do I recover a board with BIOS-FL-202, and what does ERR-PSU-904 mean?\"",
             "Exact-code lexical retrieval + semantic context with verified [doc_XX] citations."],
            ["Store & Regional Ops Managers",
             "\"What is total sales revenue and units sold by region, highest to lowest?\"",
             "Guarded read-only Text-to-SQL aggregation (GROUP BY, SUM, ORDER BY) over sales."],
            ["Account & Warranty Specialists",
             "\"How many units of SKU-NV-4090 are in stock, and what is our warranty policy?\"",
             "Multi-tool orchestration: run_sql on products + search_docs on warranty corpus."],
        ],
        [1.65 * inch, 2.75 * inch, 2.65 * inch]
    ))
    story.append(Spacer(1, 5))

    story.append(Paragraph("1.3 Why Standard LLMs and Naive Vector-Only RAG Fail", st["H2"]))
    story.append(Paragraph(
        "1. <b>Hallucinated Tabular Arithmetic:</b> An ungrounded LLM cannot know live stock counts or compute exact "
        "regional revenue totals ($6,319.87 in North vs. $3,589.78 in South); embedding tabular rows into chunks fails "
        "on <font name='Courier'>SUM</font>/<font name='Courier'>GROUP BY</font> queries.<br/>"
        "2. <b>Dense Embedding Failure on Bare Error Codes:</b> Vector representations smooth over arbitrary character "
        "strings. When a technician queries a bare fault code like <font name='Courier'>ERR-PSU-904</font> (Query Q3), "
        "our vector-only baseline fails to match any document (RR = 0.00).<br/>"
        "3. <b>BM25 Failure on Paraphrased Intent:</b> Conversely, when a customer asks <i>\"By what hour must I pay "
        "to get my order shipped tonight?\"</i> (Query Q15), BM25 ranks the shipping cutoff policy (<font name='Courier'>doc_09</font>, "
        "which uses <i>\"daily order cutoff timestamp of 3:00 PM EST\"</i>) at position 5 (RR = 0.20).<br/>"
        "4. <b>Wasted Tool Calls &amp; Citation Hallucination:</b> Always-on retrieval wastes compute on simple greetings "
        "(<i>\"Hello!\"</i>), while unvalidated LLM output frequently cites document IDs that were never retrieved.",
        st["Body"]
    ))

    # ==================== SECTION 2 (ARCHITECTURE) ====================
    story.append(Spacer(1, 6))
    story.append(Paragraph("2. System Design &amp; Architecture Diagram", st["H1"]))
    story.append(Paragraph("2.1 End-to-End Architecture Diagram", st["H2"]))
    story.append(Paragraph(
        "Figure 1 illustrates the complete execution pipeline implemented across <font name='Courier'>src/app.py</font>, "
        "<font name='Courier'>src/agent.py</font>, <font name='Courier'>src/retriever.py</font>, and "
        "<font name='Courier'>src/sql_tool.py</font>.",
        st["Body"]
    ))
    story.append(build_architecture_drawing())
    story.append(Spacer(1, 6))

    story.append(Paragraph("2.2 Mapping to the 6 Course Capability Families (Lectures 1–14)", st["H2"]))
    story.append(make_table(
        ["Capability Family", "Module & Key Symbols", "Design Rationale & Technical Mechanism"],
        [
            ["1. Structured Output & Code Verification (Lec 4–5)",
             "src/agent.py: StructuredAgentResponse, _finalize()",
             "Enforces Pydantic schema {answer, citations, tools_used, confidence}. _finalize() intersects LLM citations against _citations_from_trace() so hallucinated doc IDs are stripped, and auto-attaches sql:<table> tags whenever run_sql executes."],
            ["2. Hybrid RAG with RRF & Citations (Lec 6–8)",
             "src/retriever.py: tokenize(), search_bm25(), search_vector(), rrf()",
             "Hyphen-aware tokenizer indexes both 'err-psu-904' and ['err','psu','904']. Fuses top-20 BM25 (k1=1.5, b=0.75) and top-20 Vector rankings via Reciprocal Rank Fusion (k=60), avoiding raw score calibration mismatch."],
            ["3. Retrieval Evaluation Suite (Lec 8)",
             "eval/run_eval.py, data/eval_set.json, eval/results.md",
             "Evaluates 20 queries (10 exact-code, 10 semantic paraphrases) using Recall@3, Recall@5, and MRR across Vector-only, BM25-only, and Hybrid RRF (k=60)."],
            ["4. Tool-Calling Agent Loop (Lec 10–11)",
             "src/agent.py: run_agent(), _llm_agent(), _offline_agent()",
             "Observe -> Think -> Act loop at temperature=0. Guardrails: max_steps=5 cap, consecutive duplicate call detector (last_sig), defensive json.JSONDecodeError recovery back to role='tool', plus deterministic offline fallback."],
            ["5. Guarded Text-to-SQL (Lec 11)",
             "src/sql_tool.py: is_safe_select(), clean_sql(), ensure_limit(), execute_guarded_sql()",
             "4-layer defense in depth: strips ```sql fences; regex gate blocks comments (--, /* */), stacked semicolons, non-SELECT starts, and 15 DDL/DML/admin verbs; appends LIMIT 50; connects via SQLite URI mode=ro."],
            ["6. Served FastAPI Web UI (Lec 13–14)",
             "src/app.py: GET /, POST /api/chat, GET /api/eval, GET /api/health",
             "Zero-build dark single-page web UI with Confidence/Tools/Citations badges, clickable citation-to-trace highlighting, expandable step-by-step trace inspector, live evaluation metrics table, and 6 one-click demo presets."],
        ],
        [1.45 * inch, 1.65 * inch, 3.95 * inch]
    ))

    story.append(Paragraph("2.3 Deep Dive: Defense-in-Depth SQL Guardrails &amp; Agent Loop Controls", st["H2"]))
    story.append(Paragraph(
        "In <font name='Courier'>src/sql_tool.py</font>, <font name='Courier'>execute_guarded_sql(sql)</font> never raises "
        "unhandled exceptions into the agent loop; database errors are returned as structured JSON "
        "<font name='Courier'>{\"ok\": False, \"sql\": ..., \"error\": ...}</font> so the LLM can self-correct on syntax "
        "or column-name mistakes. In <font name='Courier'>src/agent.py</font>, tool descriptions follow Lecture 11 "
        "control-surface rules: <font name='Courier'>search_docs</font> explicitly instructs <i>\"Do NOT call for greetings, "
        "chit-chat, or database stock/sales numbers\"</i>, while <font name='Courier'>run_sql</font> specifies the exact "
        "table schemas and restricts usage to prices, stock counts, and sales aggregations.",
        st["Body"]
    ))

    # ==================== SECTION 3 (EVALUATION METHODOLOGY) ====================
    story.append(Spacer(1, 6))
    story.append(Paragraph("3. Evaluation Methodology", st["H1"]))
    story.append(Paragraph("3.1 The 20-Query Labeled Benchmark (<font name='Courier'>data/eval_set.json</font>)", st["H2"]))
    story.append(Paragraph(
        "To evaluate retrieval accuracy without relying on subjective inspection, we built a 20-query labeled benchmark "
        "in <font name='Courier'>data/eval_set.json</font> paired with the 20 policy chunks in "
        "<font name='Courier'>data/corpus.json</font>. Every single document (<font name='Courier'>doc_01</font> through "
        "<font name='Courier'>doc_20</font>) appears in the ground-truth relevance sets. Four queries have two relevant "
        "documents (e.g., <font name='Courier'>ERR-PSU-904</font> is covered under warranty in <font name='Courier'>doc_04</font> "
        "and troubleshooting in <font name='Courier'>doc_16</font>), making <font name='Courier'>Recall@k</font> sensitive "
        "to retrieving all applicable policies:",
        st["Body"]
    ))
    story.append(make_table(
        ["Subset", "Query IDs", "Design Characteristics & Representative Examples", "Target Challenge"],
        [
            ["Subset A: Exact-Code & SKU Queries (10 queries)",
             "Q1 – Q10",
             "Contains exact hyphenated identifiers from the corpus, either standalone (Q3: 'ERR-PSU-904') or with natural phrasing (Q2: 'BIOS-FL-202 and my board will not start anymore, how do I revive it?'; Q8: 'DOA-48HR: window to flag something broken right out of the box').",
             "Tests whether the retriever preserves exact rare identifier tokens that dense/n-gram vector models smooth away."],
            ["Subset B: Colloquial & Semantic Queries (10 queries)",
             "Q11 – Q20",
             "Natural-language questions using synonyms and everyday phrasing not verbatim in policy titles (Q13: 'Can power banks fly on a plane?' -> doc_10; Q15: 'By what hour must I pay to get my order shipped tonight?' -> doc_09; Q19: 'Our systems were down for hours, what credit money do we get back?' -> doc_20).",
             "Tests whether the retriever bridges lexical vocabulary mismatch where BM25 misses or ranks distractors higher."],
        ],
        [1.35 * inch, 0.75 * inch, 3.05 * inch, 1.90 * inch]
    ))

    story.append(Paragraph("3.2 Mathematical Definitions of Retrieval &amp; Fusion Metrics (Lecture 8)", st["H2"]))
    story.append(Paragraph(
        "Let <i>Q</i> be the evaluation set of |<i>Q</i>| = 20 queries. For each query <i>q</i> &isin; <i>Q</i>, let "
        "<i>R<sub>q</sub></i> be the ground-truth set of relevant document IDs and "
        "<i>L<sub>q</sub></i> = [<i>d</i><sub>1</sub>, <i>d</i><sub>2</sub>, &hellip;, <i>d<sub>m</sub></i>] be the "
        "1-based ranked list returned by a retriever:",
        st["Body"]
    ))
    story.append(Paragraph(
        "Recall@k(q) = |{d_1, ..., d_k} ∩ R_q| / |R_q|     ;     Mean Recall@k = (1 / |Q|) * Σ_{q ∈ Q} Recall@k(q)",
        st["Formula"]
    ))
    story.append(Paragraph(
        "RR(q) = 1 / min{ i : d_i ∈ R_q }  (or 0 if none)   ;     MRR = (1 / |Q|) * Σ_{q ∈ Q} RR(q)",
        st["Formula"]
    ))
    story.append(Paragraph(
        "RRF_Score(d ; k=60) = Σ_{m ∈ {BM25, Vector}}  1.0 / (60 + rank_m(d))   [1-based ranks, tie-break by first seen]",
        st["Formula"]
    ))

    story.append(Paragraph("3.3 Retrieval Strategies Evaluated in <font name='Courier'>eval/run_eval.py</font>", st["H2"]))
    story.append(Paragraph(
        "&bull; <b>Baseline 1 — Vector-Only (<font name='Courier'>search_vector</font>):</b> Computes cosine similarity "
        "over TF-IDF weighted semantic word stems (with morphological suffix stripping and a domain synonym map, plus "
        "character 3-grams <font name='Courier'>#w#</font>) or <font name='Courier'>baai/bge-m3</font> API embeddings "
        "when <font name='Courier'>USE_API_EMBEDDINGS=1</font>. Any token containing digits or hyphens is excluded in the "
        "offline vectorizer, faithfully reproducing dense embedding weakness on arbitrary serial/fault codes.<br/>"
        "&bull; <b>Baseline 2 — BM25-Only (<font name='Courier'>search_bm25</font>):</b> Pure-Python Okapi BM25 "
        "(<i>k</i><sub>1</sub> = 1.5, <i>b</i> = 0.75, IDF = ln((<i>N</i> &minus; df + 0.5)/(df + 0.5) + 1)) indexed over "
        "<font name='Courier'>title + text + category</font> using hyphen-aware tokenization (<font name='Courier'>tokenize()</font>).<br/>"
        "&bull; <b>Upgraded System — Hybrid Search via RRF (<font name='Courier'>search_hybrid</font>, <i>k</i>=60):</b> "
        "Retrieves top-20 candidates from <font name='Courier'>search_bm25</font> and top-20 from "
        "<font name='Courier'>search_vector</font>, fuses their ranked ID lists with <font name='Courier'>rrf(k=60)</font>, "
        "and returns the top-5 documents with <font name='Courier'>[doc_XX]</font> citation tags.",
        st["Body"]
    ))

    # ==================== PAGE 4: SECTION 4 (EXPERIMENTAL RESULTS) ====================
    story.append(Paragraph("4. Experimental Results &amp; Honest Failure Analysis", st["H1"]))
    story.append(Paragraph("4.1 Overall Benchmark Comparison (<font name='Courier'>eval/results.md</font>)", st["H2"]))
    story.append(make_table(
        ["Retrieval Strategy", "Recall@3", "Recall@5", "MRR", "Gain vs. Vector (MRR)", "Gain vs. BM25 (MRR)"],
        [
            ["Baseline 1: Vector-only", "0.875", "0.875", "0.842", "—", "+0.007"],
            ["Baseline 2: BM25-only", "0.900", "0.950", "0.835", "-0.007", "—"],
            ["Upgraded: Hybrid (BM25 + Vector, RRF k=60)", "0.950", "0.950", "0.950", "+0.108 (+12.8%)", "+0.115 (+13.8%)"],
        ],
        [2.35 * inch, 0.80 * inch, 0.80 * inch, 0.80 * inch, 1.15 * inch, 1.15 * inch],
        highlight_last=True
    ))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "<b>Summary of Findings:</b> Hybrid RRF (<i>k</i>=60) achieves the highest score on every metric: "
        "<b>0.950 Recall@3</b> (+7.5 points over Vector-only, +5.0 points over BM25-only), <b>0.950 Recall@5</b>, and "
        "<b>0.950 MRR</b> (+10.8 points over Vector-only and +11.5 points over BM25-only). Out of the 20 queries, "
        "Hybrid RRF places the relevant document at <b>Rank 1 on 19 out of 20 queries</b> (and captures both relevant "
        "documents within the top 2 on all multi-label queries Q3, Q4, Q7, and Q18).",
        st["Body"]
    ))

    story.append(Paragraph("4.2 Concrete Win Case Analysis: How RRF Recovers Single-Retriever Drops", st["H2"]))
    story.append(make_table(
        ["Query & Ground Truth", "Vector-Only Top-5 (RR)", "BM25-Only Top-5 (RR)", "Hybrid RRF Top-5 (RR) & Mechanism"],
        [
            ["Q3: \"ERR-PSU-904\"\nRelevant: ['doc_04', 'doc_16']",
             "[]\nRR = 0.00, R@5 = 0.00\n(Zero non-code tokens)",
             "['doc_04', 'doc_16']\nRR = 1.00, R@5 = 1.00\n(Exact rare token match)",
             "['doc_04', 'doc_16']\nRR = 1.00, R@5 = 1.00\nBM25 carries the RRF sum (1/61, 1/62) when Vector returns empty."],
            ["Q2: \"BIOS-FL-202 and my board will not start anymore, how do I revive it?\"\nRelevant: ['doc_13']",
             "['doc_03','doc_13','doc_05',...]\nRR = 0.50 (Rank 2)\n(Distracted by 'board')",
             "['doc_13','doc_15','doc_14',...]\nRR = 1.00 (Rank 1)\n(Matches 'bios-fl-202')",
             "['doc_13','doc_03','doc_05',...]\nRR = 1.00 (Rank 1)\ndoc_13 gets 1/61 + 1/62 = 0.03252 vs doc_03's 1/61 + 1/65 = 0.03178."],
            ["Q15: \"By what hour must I pay to get my order shipped tonight?\"\nRelevant: ['doc_09']",
             "['doc_09','doc_02','doc_12',...]\nRR = 1.00 (Rank 1)\n(Matches hour/deadline/ship)",
             "['doc_06','doc_15','doc_08','doc_16','doc_09']\nRR = 0.20 (Rank 5)\n(Only 'order' overlaps)",
             "['doc_09','doc_06','doc_02',...]\nRR = 1.00 (Rank 1)\ndoc_09 appears in BOTH lists (1/61 + 1/65 = 0.03178), beating doc_06 (1/61 only)."],
            ["Q16: \"How slow is getting stuff through foreign border checks?\"\nRelevant: ['doc_11']",
             "['doc_20','doc_09','doc_11',...]\nRR = 0.33 (Rank 3)\n(Colloquial 'slow/stuff')",
             "['doc_11','doc_06','doc_18',...]\nRR = 1.00 (Rank 1)\n(Matches sub-token 'border')",
             "['doc_11','doc_20','doc_18',...]\nRR = 1.00 (Rank 1)\ndoc_11 combines Rank 1 BM25 + Rank 3 Vector (1/61 + 1/63 = 0.03227)."],
        ],
        [1.75 * inch, 1.55 * inch, 1.55 * inch, 2.20 * inch]
    ))

    # ==================== FAILURE ANALYSIS & SECTION 5 ====================
    story.append(Spacer(1, 6))
    story.append(Paragraph("4.3 Honest Failure Case &amp; Guardrail Boundary Analysis", st["H2"]))
    story.append(Paragraph(
        "<b>1. Retrieval Failure Case (Query Q12 — Extreme Colloquial Vocabulary Mismatch):</b>",
        st["Body"]
    ))
    story.append(Preformatted(
        "Q12: \"My package showed up smashed up and the seal was cut, what now?\"  (Ground truth: ['doc_12'])\n"
        "  - Baseline 1 (Vector-only): top-5 = ['doc_05', 'doc_13', 'doc_15', 'doc_11', 'doc_06']  | RR = 0.00\n"
        "  - Baseline 2 (BM25-only):   top-5 = ['doc_07', 'doc_08', 'doc_03', 'doc_05', 'doc_09']  | RR = 0.00\n"
        "  - Upgraded (Hybrid RRF):    top-5 = ['doc_05', 'doc_15', 'doc_08', 'doc_11', 'doc_13']  | RR = 0.00",
        st["Code"]
    ))
    story.append(Paragraph(
        "<b>Root Cause of Q12 Failure:</b> Target document <font name='Courier'>doc_12</font> (<i>High-Value Signature "
        "Delivery &amp; Transit Damage Reporting</i>) uses formal logistics phrasing: <i>\"equipment consignment\"</i>, "
        "<i>\"severe exterior carton crushing\"</i>, <i>\"puncture\"</i>, and <i>\"opened tamper-evident security tape\"</i>. "
        "Query Q12 uses everyday colloquial idioms: <i>\"package\"</i>, <i>\"smashed up\"</i>, and <i>\"seal was cut\"</i>. "
        "There is zero lexical token overlap for BM25, and our offline stem/trigram vectorizer does not map "
        "<i>smashed</i> &harr; <i>crushing</i> or <i>seal</i> &harr; <i>tamper-evident tape</i>. Because neither base "
        "retriever ranks <font name='Courier'>doc_12</font> in its top 5, Reciprocal Rank Fusion cannot surface it.<br/>"
        "<b>Agent-Level Recovery in LLM Mode vs. Offline Mode:</b> Notably, when Q12 is executed through the full "
        "FastAPI chat endpoint (<font name='Courier'>POST /api/chat</font>, Preset 5b) with <font name='Courier'>gemma-4-E4B-it</font>, "
        "the LLM agent autonomously reformulates the user's colloquial question into two targeted tool calls: "
        "Step 1 <font name='Courier'>search_docs(query='damaged or tampered package return policy')</font> and "
        "Step 2 <font name='Courier'>search_docs(query='damaged in transit hardware claim procedure')</font>. That second "
        "rewritten query matches <i>\"Transit Damage\"</i> in <font name='Courier'>doc_12</font>, allowing the full agent "
        "to return a high-confidence answer citing <font name='Courier'>['doc_12', 'doc_06']</font>! In offline mode "
        "(where the raw query is passed un-rewritten), the failure remains visible end-to-end.",
        st["Body"]
    ))

    story.append(Paragraph(
        "<b>2. Text-to-SQL Guardrail Verification &amp; False-Positive Boundary Analysis:</b>",
        st["Body"]
    ))
    story.append(Preformatted(
        "$ python -m src.sql_tool\n"
        "[PASS] simple SELECT allowed  | [PASS] DROP TABLE blocked       | [PASS] stacked ; blocked\n"
        "[PASS] -- comment blocked     | [PASS] /* */ comment blocked    | [PASS] DELETE inside SELECT blocked\n"
        "{'ok': False, 'sql': 'DROP TABLE products', 'error': 'Rejected: only a single read-only SELECT ...'}",
        st["Code"]
    ))
    story.append(Paragraph(
        "When tested against Preset 5a (<font name='Courier'>\"Run this SQL: DROP TABLE products; --\"</font>), defense "
        "occurs at three independent layers: (1) <font name='Courier'>gemma-4-E4B-it</font> refuses to call "
        "<font name='Courier'>run_sql</font> because the tool spec restricts it to read-only queries; "
        "(2) if <font name='Courier'>execute_guarded_sql</font> is called directly, <font name='Courier'>is_safe_select</font> "
        "rejects both the <font name='Courier'>--</font> comment and the <font name='Courier'>DROP</font> verb; and "
        "(3) SQLite's <font name='Courier'>mode=ro</font> URI blocks writes at the OS level. However, an honest boundary "
        "limitation of our regex (<font name='Courier'>^\\s*select\\b</font>) is that legitimate read-only Common Table "
        "Expressions starting with <font name='Courier'>WITH regional AS (SELECT ...) SELECT *</font> are rejected as "
        "false positives.",
        st["Body"]
    ))

    story.append(Paragraph("5. Limitations &amp; Next Steps", st["H1"]))
    story.append(Paragraph("5.1 Current Architectural Limitations", st["H2"]))
    story.append(Paragraph(
        "1. <b>Colloquial Vocabulary Gap in Single-Shot Retrieval:</b> Without LLM query rewriting or dense neural "
        "embeddings enabled by default, extreme idiom shifts (such as Q12's <i>\"smashed up / seal was cut\"</i>) miss "
        "in single-shot retrieval.<br/>"
        "2. <b>Equal-Weight RRF Without Cross-Encoder Reranking:</b> RRF weights BM25 and Vector ranks equally (<i>k</i>=60). "
        "When one retriever returns a high-confidence exact code match at Rank 1 while the other returns an unrelated "
        "document at Rank 1, RRF ties them at 1/61 unless broken by list order or secondary rank support.<br/>"
        "3. <b>Regex SQL Gate Blocks Read-Only <font name='Courier'>WITH</font> CTEs:</b> Requiring <font name='Courier'>^\\s*select\\b</font> "
        "prevents the LLM from writing multi-step <font name='Courier'>WITH</font> queries for complex analytical ratios.<br/>"
        "4. <b>Full DDL Injection in System Prompt:</b> Injecting all <font name='Courier'>CREATE TABLE</font> schemas "
        "into <font name='Courier'>SYSTEM_PROMPT</font> works well for 2 tables (<font name='Courier'>products</font>, "
        "<font name='Courier'>sales</font>), but would overflow context or degrade attention on 100+ enterprise tables.",
        st["Body"]
    ))

    story.append(Paragraph("5.2 Concrete, Measurable Roadmap for Next Iteration", st["H2"]))
    story.append(make_table(
        ["Proposed Upgrade", "Concrete Implementation Plan", "Target Measurable Improvement"],
        [
            ["1. Cross-Encoder Reranker + Dense Embeddings",
             "Enable baai/bge-m3 dense vectors by default for search_vector, fuse top-20 BM25 + Vector via RRF (k=60), and rerank the top-10 fused candidates with BAAI/bge-reranker-v2-m3 before returning top-4.",
             "Recover Q12 in single-shot retrieval; raise Recall@5 from 0.950 -> 1.000 and MRR from 0.950 -> >= 0.975."],
            ["2. AST-Based SQL Validation (sqlglot)",
             "Replace regex gate in is_safe_select() with sqlglot.parse_one(sql, read='sqlite'), permitting exp.Select with read-only WITH CTEs while blocking all DDL/DML AST nodes.",
             "100% pass rate on read-only CTE analytical queries while keeping 100% block rate on destructive/stacked SQL."],
            ["3. 50-Query Multi-Hop & Faithfulness Suite",
             "Expand data/eval_set.json from 20 to 50 queries (adding 15 SQL+RAG multi-hop queries and 15 adversarial prompts) with automated LLM-as-Judge faithfulness scoring.",
             "Achieve >= 95% citation precision and >= 90% end-to-end answer faithfulness across all 50 benchmark cases."],
        ],
        [1.55 * inch, 3.25 * inch, 2.25 * inch]
    ))

    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    return doc.page_count


if __name__ == "__main__":
    pages = generate_pdf()
    print(f"Generated {PDF_PATH} ({pages} pages)")
