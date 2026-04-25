"""
workbench_app.py — 8D Expert Workbench
Combines the 8D Report Builder (ported from Streamlit) and the CAPA/8D Expert RAG
into a single Gradio app with two tabs and shared state.

Usage:
    uv run workbench_app.py
    uv run workbench_app.py --share

Dependencies:
    - scripts/answer.py  (RAG pipeline, unchanged)
    - .env with OPENAI_API_KEY and ANTHROPIC_API_KEY
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date

# ── Env ────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env if _env.exists() else None)
except ImportError:
    pass

import gradio as gr
import anthropic

# ── RAG pipeline import ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from answer import answer, answer_stream, AnswerResult, RankedChunk

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

D_STEPS = [
    ("D0", "Problem Initiation",       "Capture the initial problem description and context"),
    ("D1", "Team Formation",           "Define the cross-functional team and champion"),
    ("D2", "Problem Description",      "Is/Is-Not analysis and quantified problem statement"),
    ("D3", "Containment Actions",      "Immediate actions to protect the customer"),
    ("D4", "Root Cause Analysis",      "Identify and verify the root cause(s)"),
    ("D5", "Corrective Actions",       "Define permanent corrective actions"),
    ("D6", "Implementation & Verify",  "Implement and verify corrective actions"),
    ("D7", "Prevent Recurrence",       "Update systems, procedures, and lessons learned"),
    ("D8", "Team Recognition",         "Close the 8D and recognise the team"),
]

EXPERT_TEMPLATES = {
    "D0": "I have a potential quality issue that may need a full 8D investigation: {problem} on {product}. "
          "Should I open a full 8D or is a correction-only path appropriate here?",
    "D1": "I'm forming the 8D team for this problem: {problem} on {product}. "
          "Who should be on the cross-functional team — which functions and roles are essential?",
    "D2": "Help me write a rigorous 5W2H problem statement for this issue: {problem} on {product}. "
          "What Is/Is-Not dimensions should I define to properly scope the suspect population?",
    "D3": "What containment locations must I check and prioritise for this issue: {problem} on {product}? "
          "Are there any locations that are commonly missed?",
    "D4": "Which RCA tool should I use for this defect: {problem} on {product}? "
          "And what does good look like for both the occurrence root cause and the escape root cause?",
    "D5": "The verified root cause for our 8D is: {rc}. "
          "What permanent corrective actions are typically effective for this type of root cause, "
          "and what are the common mistakes to avoid?",
    "D6": "We have implemented corrective actions for: {rc}. "
          "What validation evidence and pass criteria should we define to close D6 with confidence?",
    "D7": "Our corrective action was: {ca}. "
          "What systemic updates are required in D7 — FMEA, Control Plan, procedures, lateral search — "
          "for this type of corrective action?",
    "D8": "We are closing our 8D for: {problem} on {product}. "
          "What are the closure criteria to check before D8 sign-off, "
          "and what should the lessons-learned entry include to be genuinely reusable?",
}

CATEGORY_COLOURS = {
    "methodology": "#4A90D9",
    "example":     "#7B68EE",
    "procedure":   "#2E8B57",
    "reference":   "#D4822A",
    "tool":        "#9B59B6",
    "compliance":  "#C0392B",
    "general":     "#7F8C8D",
}

EXAMPLE_QUESTIONS = [
    "What are the most common mistakes teams make in D3 containment?",
    "How do I update the FMEA after closing a CAPA?",
    "What is the difference between ICA and PCA in 8D?",
    "When should I use Ishikawa instead of 5 Whys?",
    "What locations must I check when scoping the suspect population?",
    "What does a good VoE pass criterion look like for a dimensional fix?",
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Global resets ── */
*, *::before, *::after { box-sizing: border-box; }
footer { display: none !important; }

/* ── Root container ── */
.gradio-container, .gradio-container * {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.gradio-container {
    max-width: 1500px !important;
    background: #0d1117 !important;
}

/* Gradio 5 wraps everything in .contain and svelte-* divs —
   force background on every generic wrapper */
.gradio-container > div,
.gradio-container > div > div,
section.gradio-container,
div.svelte-1gfkn6j,
div.svelte-vt1mxs,
div.gap,
.contain {
    background: #0d1117 !important;
    color: #e6edf3 !important;
}

/* ── Tabs (Gradio 5 uses role="tab" + .tabs) ── */
.tabs { background: #0d1117 !important; }
.tab-nav, [role="tablist"] {
    background: #161b22 !important;
    border-bottom: 1px solid #21262d !important;
}
[role="tab"] {
    font-weight: 500 !important;
    color: #8b949e !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[role="tab"].selected, [role="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* ── All panels / blocks ── */
.block, .panel, .form, .gap, .compact, .padded,
.gradio-container .block {
    background: #161b22 !important;
    border-color: #21262d !important;
    color: #e6edf3 !important;
}

/* ── Labels ── */
label, .label-wrap span, span.svelte-1f354aw {
    color: #c9d1d9 !important;
    font-size: 13px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* ── Inputs & textareas ── */
input[type="text"], input[type="number"], textarea,
.scroll-hide, [data-testid="textbox"] textarea,
[data-testid="textbox"] input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
input:focus, textarea:focus {
    border-color: #58a6ff !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* ── Dropdowns / selects ── */
select, .wrap-inner, [data-testid="dropdown"] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
}
.options, .option { background: #161b22 !important; color: #e6edf3 !important; }
.option:hover, .option.selected { background: #1f6feb !important; color: white !important; }

/* ── Checkboxes ── */
input[type="checkbox"] { accent-color: #1f6feb; }
.checkbox-group label, [data-testid="checkbox"] + span {
    color: #c9d1d9 !important;
}

/* ── Buttons — Gradio 5 uses button[variant] attributes ── */
button.primary, button[variant="primary"],
[data-testid="button"].primary {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
button.primary:hover, button[variant="primary"]:hover { background: #388bfd !important; }

button.secondary, button[variant="secondary"],
[data-testid="button"].secondary {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    background: #2d333b !important;
    border-color: #58a6ff !important;
}

/* ── Ask Expert button ── */
.ask-expert-btn button, .ask-expert-btn button.secondary {
    background: #1a2e1a !important;
    color: #3fb950 !important;
    border: 1px solid rgba(46,160,67,0.15) !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.ask-expert-btn button:hover {
    background: #1f3a2e !important;
    border-color: #3fb950 !important;
}

/* ── Accordions — Gradio 5 uses <details> ── */
details, .accordion {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
details summary, .accordion > .label-wrap {
    background: #161b22 !important;
    padding: 10px 16px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    color: #58a6ff !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    list-style: none !important;
}
details[open] { border-color: #30363d !important; }

/* ── Chatbot — Gradio 5 ── */
[data-testid="chatbot"], .chatbot {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
/* Message bubbles */
[data-testid="chatbot"] .message,
.chatbot .message {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
[data-testid="chatbot"] .message.user,
.chatbot .message.user,
[data-testid="user"] {
    background: #1f3a2e !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
[data-testid="chatbot"] .message.bot,
.chatbot .message.bot,
[data-testid="bot"],
[data-testid="chatbot"] [data-testid="bot"] {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
/* Markdown inside bot messages */
[data-testid="bot"] p,
[data-testid="bot"] li,
[data-testid="bot"] h1,
[data-testid="bot"] h2,
[data-testid="bot"] h3,
[data-testid="bot"] strong,
[data-testid="bot"] code {
    color: #e6edf3 !important;
}
[data-testid="bot"] code {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

/* ── Markdown component ── */
.prose, .md, [data-testid="markdown"] {
    color: #e6edf3 !important;
    background: transparent !important;
}
.prose h3, .md h3 { color: #58a6ff !important; }
"""

# ══════════════════════════════════════════════════════════════════════════════
# Claude helper (for Builder AI suggestions — direct Anthropic, not RAG)
# ══════════════════════════════════════════════════════════════════════════════

def call_claude_direct(prompt: str, max_tokens: int = 600) -> str:
    """Direct Claude call for Builder AI suggestions. Not RAG-grounded."""
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            system=(
                "You are an expert quality engineer specialising in 8D problem solving "
                "for manufacturing. Provide concise, practical, industry-specific guidance. "
                "Use bullet points. Keep responses under 250 words."
            ),
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"⚠️ AI unavailable: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# RAG Expert helpers
# ══════════════════════════════════════════════════════════════════════════════

def category_badge(category: str) -> str:
    colour = CATEGORY_COLOURS.get(category, "#7F8C8D")
    return (
        f'<span style="background:{colour};color:white;font-size:10px;'
        f'font-weight:600;padding:2px 7px;border-radius:10px;margin-right:4px;">'
        f'{category.upper()}</span>'
    )


def format_sources_panel(result: AnswerResult) -> str:
    if not result.ranked_chunks:
        return "<p style='color:#888'>No sources retrieved.</p>"

    reranker_badge = (
        '<span style="background:#2E8B57;color:white;font-size:10px;font-weight:600;'
        'padding:2px 7px;border-radius:10px;margin-left:6px;">BGE</span>'
        if result.reranker_used == "bge" else
        '<span style="background:#7B68EE;color:white;font-size:10px;font-weight:600;'
        'padding:2px 7px;border-radius:10px;margin-left:6px;">LLM</span>'
    )
    parts = [
        f'<div style="font-size:12px;color:#888;margin-bottom:12px;">'
        f'{len(result.ranked_chunks)} chunks · {len(result.sources)} documents'
        f'{reranker_badge}</div>'
    ]
    for chunk in result.ranked_chunks:
        score = chunk.relevance_score
        score_colour = "#2E8B57" if score >= 7 else "#D4822A" if score >= 4 else "#C0392B"
        preview = chunk.original_text[:300].replace("\n", " ").strip()
        if len(chunk.original_text) > 300:
            preview += "…"
        parts.append(f"""
<div style="border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:10px;background:#161b22;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <div>
      {category_badge(chunk.doc_category)}
      <span style="font-size:11px;color:#8b949e;font-weight:500;">
        {chunk.source_file.replace('.md','').replace('_',' ')}
      </span>
    </div>
    <span style="font-size:12px;font-weight:700;color:{score_colour};">{score:.1f}/10</span>
  </div>
  <div style="font-size:12px;font-weight:600;color:#c9d1d9;margin-bottom:4px;">{chunk.headline}</div>
  <div style="font-size:11px;color:#8b949e;line-height:1.5;">{preview}</div>
</div>""")

    if result.rewritten_queries:
        rewrites = "".join(
            f'<div style="font-size:11px;color:#8b949e;padding:2px 0;">• {q}</div>'
            for q in result.rewritten_queries
        )
        parts.append(f"""
<details style="margin-top:12px;">
  <summary style="font-size:11px;color:#8b949e;cursor:pointer;">
    Query rewrites ({len(result.rewritten_queries)})
  </summary>
  <div style="padding:6px 0 0 8px;">{rewrites}</div>
</details>""")

    return "\n".join(parts)


def _format_sources_from_sink(sink: dict) -> str:
    """Build sources HTML directly from the _sink dict populated by answer_stream()."""
    ranked_chunks  = sink.get("ranked_chunks", [])
    sources        = sink.get("sources", [])
    reranker_used  = sink.get("reranker_used", "bge")
    rewritten_queries = sink.get("rewritten_queries", [])

    if not ranked_chunks:
        return "<p style='color:#888'>No sources retrieved.</p>"

    # Reuse format_sources_panel logic via a lightweight AnswerResult stand-in
    dummy = AnswerResult(
        question="",
        rewritten_queries=rewritten_queries,
        ranked_chunks=ranked_chunks,
        answer="",
        sources=sources,
        reranker_used=reranker_used,
        checker_score=1.0,
    )
    return format_sources_panel(dummy)


def _content_str(m: dict) -> str:
    """Normalise Gradio message content to plain string."""
    c = m.get("content", "")
    if isinstance(c, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c) if c else ""


# ══════════════════════════════════════════════════════════════════════════════
# Streaming RAG chat handler
# ══════════════════════════════════════════════════════════════════════════════

def rag_chat_stream(
    message: str,
    history: list[dict],
    use_rewrite: bool,
):
    """
    Streaming RAG chat handler for Gradio.

    Yields (updated_history, cleared_input, sources_html) at each step:
      - During streaming: sources_html shows "Retrieving..." then "Streaming..."
      - After stream ends: sources_html shows populated sources panel
    """
    if not message.strip():
        yield history, "", "<p style='color:#8b949e'>Ask a question to see sources.</p>"
        return

    past_turns = [
        {"role": m["role"], "content": _content_str(m)}
        for m in history
        if _content_str(m).strip()
    ]

    # Add user message immediately so it appears in the chatbot
    new_history = history + [{"role": "user", "content": message}]

    # Show a retrieving indicator while the non-streaming pipeline steps run
    # (rewrite + embed + BGE all happen before the first token)
    yield (
        new_history + [{"role": "assistant", "content": "▌"}],
        "",
        "<p style='color:#8b949e;font-size:12px;'>⏳ Retrieving and ranking sources…</p>",
    )

    sink = {}  # populated by answer_stream() before first token arrives
    accumulated = ""

    try:
        for partial in answer_stream(
            question=message,
            use_rewrite=use_rewrite,
            reranker_mode="auto",
            history=past_turns if past_turns else None,
            _sink=sink,
        ):
            accumulated = partial
            # While streaming: show sources panel (sink is populated after first yield from answer_stream)
            sources_html = (
                _format_sources_from_sink(sink)
                if sink
                else "<p style='color:#8b949e;font-size:12px;'>⏳ Streaming…</p>"
            )
            yield (
                new_history + [{"role": "assistant", "content": accumulated}],
                "",
                sources_html,
            )

    except FileNotFoundError as e:
        accumulated = (
            f"**Knowledge base not found.**\n\n"
            f"Run: `uv run scripts/ingest.py --reset`\n\nError: {e}"
        )
        yield (
            new_history + [{"role": "assistant", "content": accumulated}],
            "",
            "<p style='color:#C0392B'>Knowledge base not initialised.</p>",
        )
        return

    except Exception as e:
        accumulated = f"**Error:** {str(e)}"
        yield (
            new_history + [{"role": "assistant", "content": accumulated}],
            "",
            f"<p style='color:#C0392B'>Error: {str(e)}</p>",
        )
        return

    # Final yield — clean up any streaming cursor artifact, finalise sources
    final_history = new_history + [{"role": "assistant", "content": accumulated}]
    yield final_history, "", _format_sources_from_sink(sink)


# ══════════════════════════════════════════════════════════════════════════════
# Expert question template builder
# ══════════════════════════════════════════════════════════════════════════════

def build_expert_question(discipline: str, d_data: dict) -> str:
    """Interpolate context from d_data into the discipline template."""
    d0 = d_data.get(0, {})
    d4 = d_data.get(4, {})
    d5 = d_data.get(5, {})

    problem = d0.get("description", "").strip() or d0.get("title", "").strip() or "this quality issue"
    product = d0.get("product", "").strip() or "the product"
    rc      = d4.get("root_cause", "").strip() or "the identified root cause"
    ca      = d5.get("pca", "").strip() or "the corrective action"

    template = EXPERT_TEMPLATES.get(discipline, f"Tell me about {discipline} in the 8D process.")
    return template.format(problem=problem, product=product, rc=rc, ca=ca)


# ══════════════════════════════════════════════════════════════════════════════
# Progress bar HTML
# ══════════════════════════════════════════════════════════════════════════════

def build_progress_html(d_data: dict) -> str:
    completed = set(d_data.keys())
    chips = []
    for i, (code, _, _) in enumerate(D_STEPS):
        if i in completed:
            style = "color:#3fb950;font-weight:700;"
            label = f"{code} ✓"
        else:
            style = "color:#484f58;"
            label = code
        chips.append(
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
            f'padding:3px 10px;border-radius:4px;{style}">{label}</span>'
        )
    chips_html = " ".join(chips)
    pct = int(len(completed) / len(D_STEPS) * 100)
    return f"""
<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px 16px;margin-bottom:16px;">
  <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">{chips_html}</div>
  <div style="background:#21262d;border-radius:4px;height:4px;">
    <div style="background:#1f6feb;height:4px;border-radius:4px;width:{pct}%;transition:width 0.3s;"></div>
  </div>
  <div style="font-size:11px;color:#8b949e;margin-top:4px;">{len(completed)} of {len(D_STEPS)} disciplines saved</div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Report text export
# ══════════════════════════════════════════════════════════════════════════════

def build_report_text(d_data: dict) -> str:
    d0 = d_data.get(0, {})
    d1 = d_data.get(1, {})
    d2 = d_data.get(2, {})
    d3 = d_data.get(3, {})
    d4 = d_data.get(4, {})
    d5 = d_data.get(5, {})
    d6 = d_data.get(6, {})
    d7 = d_data.get(7, {})
    d8 = d_data.get(8, {})
    sep = "=" * 60
    return f"""{sep}
8D PROBLEM SOLVING REPORT
{sep}
Report No:  {d0.get('report_no', '-')}
Date:       {d0.get('date', '-')}
Product:    {d0.get('product', '-')}
Customer:   {d0.get('customer', '-')}
Severity:   {d0.get('severity', '-')}
{sep}

D0 — PROBLEM INITIATION
Title: {d0.get('title', '-')}
{d0.get('description', '-')}

D1 — TEAM FORMATION
Champion: {d1.get('champion', '-')}
Team: {d1.get('team', '-')}

D2 — PROBLEM DESCRIPTION
{d2.get('problem_statement', '-')}

D3 — CONTAINMENT ACTIONS
{d3.get('actions', '-')}
Owner: {d3.get('owner', '-')} | Effectiveness: {d3.get('effectiveness', '-')}

D4 — ROOT CAUSE ANALYSIS
Root Cause: {d4.get('root_cause', '-')}
Escape Point: {d4.get('escape_point', '-')}
Method: {', '.join(d4.get('method', []))}

D5 — CORRECTIVE ACTIONS
{d5.get('pca', '-')}

D6 — IMPLEMENTATION & VERIFICATION
Status: {d6.get('status', '-')}
Verification: {d6.get('verification', '-')}
Verified By: {d6.get('verified_by', '-')}

D7 — PREVENT RECURRENCE
{d7.get('systemic', '-')}
Lessons Learned: {d7.get('lessons', '-')}

D8 — TEAM RECOGNITION & CLOSURE
{d8.get('recognition', '-')}
Closed By: {d8.get('closed_by', '-')} | Date: {d8.get('closure_date', '-')}

{sep}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""


# ══════════════════════════════════════════════════════════════════════════════
# AI suggestion HTML wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _ai_html(text: str) -> str:
    escaped = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    return (
        f'<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;'
        f'padding:14px;font-size:13px;color:#c9d1d9;line-height:1.6;">{escaped}</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# UI builder
# ══════════════════════════════════════════════════════════════════════════════

DARK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill="#0d1117",
    body_background_fill_dark="#0d1117",
    block_background_fill="#161b22",
    block_background_fill_dark="#161b22",
    block_border_color="#21262d",
    block_border_color_dark="#21262d",
    block_label_text_color="#c9d1d9",
    block_label_text_color_dark="#c9d1d9",
    body_text_color="#e6edf3",
    body_text_color_dark="#e6edf3",
    background_fill_primary="#0d1117",
    background_fill_primary_dark="#0d1117",
    background_fill_secondary="#161b22",
    background_fill_secondary_dark="#161b22",
    border_color_primary="#30363d",
    border_color_primary_dark="#30363d",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
    input_border_color="#30363d",
    input_border_color_dark="#30363d",
    input_placeholder_color="#484f58",
    input_placeholder_color_dark="#484f58",
    button_primary_background_fill="#1f6feb",
    button_primary_background_fill_dark="#1f6feb",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="#21262d",
    button_secondary_background_fill_dark="#21262d",
    button_secondary_text_color="#c9d1d9",
    button_secondary_text_color_dark="#c9d1d9",
    button_secondary_border_color="#30363d",
    button_secondary_border_color_dark="#30363d",
)


def build_ui() -> gr.Blocks:

    with gr.Blocks(title="8D Expert Workbench") as demo:

        # ── Shared state ────────────────────────────────────────────────────
        d_data_state = gr.State({})   # {0: {...}, 1: {...}, ...} per discipline
        expert_hist  = gr.State([])   # RAG chat history (messages format)

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
<div style="background:#161b22;border-bottom:1px solid #21262d;padding:16px 24px;margin-bottom:0;">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:700;color:#58a6ff;">8D</span>
    <div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:16px;font-weight:600;color:#e6edf3;">
        Expert Workbench
      </div>
      <div style="font-size:11px;color:#8b949e;font-family:'IBM Plex Sans',sans-serif;">
        8D Builder · RAG Expert Q&A · Claude Haiku · BGE Reranker · GPT-4o-mini
      </div>
    </div>
  </div>
</div>""")

        # ── Tabs ─────────────────────────────────────────────────────────────
        with gr.Tabs(elem_id="main-tabs") as tabs:

            # ================================================================
            # TAB 0 — 8D Builder
            # ================================================================
            with gr.Tab("📋  8D Builder", id=0):

                progress_display = gr.HTML(build_progress_html({}))

                # ── D0 ───────────────────────────────────────────────────────
                with gr.Accordion("D0 — Problem Initiation", open=True):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Capture the initial problem description and context</p>")
                    with gr.Row():
                        with gr.Column(scale=3):
                            d0_title    = gr.Textbox(label="Problem Title",              placeholder="e.g. Bore diameter out of spec on PN BRK-7742")
                            d0_desc     = gr.Textbox(label="Initial Problem Description", lines=4,  placeholder="Describe what happened, when, where, and the impact...")
                            d0_product  = gr.Textbox(label="Product / Process",           placeholder="e.g. Machined bracket BRK-7742, Line 3")
                            d0_customer = gr.Textbox(label="Customer / Internal",         placeholder="e.g. OEM Customer / Internal – Line 3")
                        with gr.Column(scale=2):
                            d0_date = gr.Textbox(label="Date Opened", placeholder=str(date.today()))
                            d0_sev  = gr.Dropdown(label="Severity", choices=["Low","Medium","High","Critical"], value="High")
                            d0_no   = gr.Textbox(label="Report Number", placeholder="e.g. 8D-2026-001")
                            d0_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                            d0_save   = gr.Button("Save D0", variant="primary")
                    d0_ai_btn = gr.Button("🤖 AI Suggestions for D0", variant="secondary", size="sm")
                    d0_ai_out = gr.HTML()

                # ── D1 ───────────────────────────────────────────────────────
                with gr.Accordion("D1 — Team Formation", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Define the cross-functional team and champion</p>")
                    with gr.Row():
                        with gr.Column(scale=3):
                            d1_champion = gr.Textbox(label="Team Champion",   placeholder="Name, role, authority to approve actions")
                            d1_team     = gr.Textbox(label="Team Members",    lines=3, placeholder="Name — Function — Role\nName — Function — Role")
                        with gr.Column(scale=2):
                            d1_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                            d1_save   = gr.Button("Save D1", variant="primary")
                    d1_ai_btn = gr.Button("🤖 AI Suggestions for D1", variant="secondary", size="sm")
                    d1_ai_out = gr.HTML()

                # ── D2 ───────────────────────────────────────────────────────
                with gr.Accordion("D2 — Problem Description", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Is/Is-Not analysis and quantified problem statement</p>")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**IS** — what is affected")
                            d2_is_what  = gr.Textbox(label="What",  placeholder="The specific defect")
                            d2_is_where = gr.Textbox(label="Where", placeholder="Location / process step")
                            d2_is_when  = gr.Textbox(label="When",  placeholder="First occurrence date/batch")
                            d2_is_how   = gr.Textbox(label="How Many", placeholder="Count / rate / PPM")
                        with gr.Column():
                            gr.Markdown("**IS NOT** — what is not affected")
                            d2_isnot_what  = gr.Textbox(label="What",  placeholder="Similar parts not affected")
                            d2_isnot_where = gr.Textbox(label="Where", placeholder="Lines / machines not affected")
                            d2_isnot_when  = gr.Textbox(label="When",  placeholder="Periods with no defects")
                            d2_isnot_how   = gr.Textbox(label="How Many", placeholder="Zero defects elsewhere")
                    d2_stmt   = gr.Textbox(label="Problem Statement (5W2H)", lines=3,
                                           placeholder="Concise quantified statement: What / Where / When / How Many / Why Significant")
                    with gr.Row():
                        d2_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                        d2_save   = gr.Button("Save D2", variant="primary")
                    d2_ai_btn = gr.Button("🤖 AI Suggestions for D2", variant="secondary", size="sm")
                    d2_ai_out = gr.HTML()

                # ── D3 ───────────────────────────────────────────────────────
                with gr.Accordion("D3 — Containment Actions", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Immediate actions to protect the customer</p>")
                    d3_actions = gr.Textbox(label="ICA Actions",     lines=3, placeholder="List all containment actions taken...")
                    d3_date    = gr.Textbox(label="ICA Date",        placeholder=str(date.today()))
                    d3_owner   = gr.Textbox(label="ICA Owner",       placeholder="Name responsible for containment")
                    d3_eff     = gr.Dropdown(label="Containment Effectiveness",
                                             choices=["Effective","Partially Effective","Ineffective","Pending"], value="Pending")
                    with gr.Row():
                        d3_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                        d3_save   = gr.Button("Save D3", variant="primary")
                    d3_ai_btn = gr.Button("🤖 AI Suggestions for D3", variant="secondary", size="sm")
                    d3_ai_out = gr.HTML()

                # ── D4 ───────────────────────────────────────────────────────
                with gr.Accordion("D4 — Root Cause Analysis", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Identify and verify the root cause(s)</p>")
                    d4_rc     = gr.Textbox(label="Root Cause (Occurrence)", lines=3, placeholder="Why did the defect occur?")
                    d4_escape = gr.Textbox(label="Escape Root Cause",       lines=2, placeholder="Why was the defect not detected earlier?")
                    d4_method = gr.CheckboxGroup(label="RCA Method Used",
                                                 choices=["5 Whys","Ishikawa","Is/Is-Not","FTA","Other"])
                    with gr.Row():
                        d4_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                        d4_save   = gr.Button("Save D4", variant="primary")
                    d4_ai_btn = gr.Button("🤖 AI Suggestions for D4", variant="secondary", size="sm")
                    d4_ai_out = gr.HTML()

                # ── D5 ───────────────────────────────────────────────────────
                with gr.Accordion("D5 — Corrective Actions", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Define permanent corrective actions</p>")
                    d5_pca     = gr.Textbox(label="Permanent Corrective Action (PCA)", lines=3,
                                            placeholder="Describe the permanent fix and why it addresses the root cause")
                    d5_actions = gr.Textbox(label="Supporting Actions", lines=2,
                                            placeholder="Training, tooling, procedure updates, etc.")
                    with gr.Row():
                        d5_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                        d5_save   = gr.Button("Save D5", variant="primary")
                    d5_ai_btn = gr.Button("🤖 AI Suggestions for D5", variant="secondary", size="sm")
                    d5_ai_out = gr.HTML()

                # ── D6 ───────────────────────────────────────────────────────
                with gr.Accordion("D6 — Implementation & Verification", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Implement and verify corrective actions</p>")
                    d6_status = gr.Dropdown(label="Implementation Status",
                                            choices=["In Progress","Implemented","Verified Effective","Failed Verification"],
                                            value="In Progress")
                    d6_verif  = gr.Textbox(label="Verification Evidence (VoE)", lines=3,
                                           placeholder="Cpk value, zero escapes period, SPC data, etc.")
                    d6_date   = gr.Textbox(label="Verification Date", placeholder=str(date.today()))
                    d6_by     = gr.Textbox(label="Verified By",       placeholder="Name / role")
                    with gr.Row():
                        d6_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                        d6_save   = gr.Button("Save D6", variant="primary")
                    d6_ai_btn = gr.Button("🤖 AI Suggestions for D6", variant="secondary", size="sm")
                    d6_ai_out = gr.HTML()

                # ── D7 ───────────────────────────────────────────────────────
                with gr.Accordion("D7 — Prevent Recurrence", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Update systems, procedures, and lessons learned</p>")
                    d7_systemic = gr.Textbox(label="Systemic Actions", lines=3,
                                             placeholder="FMEA update, Control Plan revision, procedure changes, lateral search...")
                    d7_lessons  = gr.Textbox(label="Lessons Learned", lines=2,
                                             placeholder="What would we do differently? What applies to other processes?")
                    d7_similar  = gr.Textbox(label="Similar Products / Processes Checked", lines=2,
                                             placeholder="Lateral search scope and findings")
                    with gr.Row():
                        d7_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                        d7_save   = gr.Button("Save D7", variant="primary")
                    d7_ai_btn = gr.Button("🤖 AI Suggestions for D7", variant="secondary", size="sm")
                    d7_ai_out = gr.HTML()

                # ── D8 ───────────────────────────────────────────────────────
                with gr.Accordion("D8 — Team Recognition & Closure", open=False):
                    gr.HTML("<p style='font-size:12px;color:#8b949e;margin:0 0 12px 0;'>Close the 8D and recognise the team</p>")
                    d8_recog = gr.Textbox(label="Team Recognition", lines=3,
                                          placeholder="Acknowledge the team's contribution...")
                    d8_date  = gr.Textbox(label="Closure Date", placeholder=str(date.today()))
                    d8_by    = gr.Textbox(label="Closed By",    placeholder="Quality Manager / Champion name")
                    with gr.Row():
                        d8_expert = gr.Button("↗ Ask Expert", elem_classes=["ask-expert-btn"])
                        d8_save   = gr.Button("Save D8", variant="primary")
                    d8_ai_btn = gr.Button("🤖 AI Suggestions for D8", variant="secondary", size="sm")
                    d8_ai_out = gr.HTML()

                # ── Builder footer ────────────────────────────────────────────
                with gr.Row():
                    reset_btn    = gr.Button("🗑 Reset All",        variant="secondary")
                    download_btn = gr.Button("⬇ Download Report",  variant="primary")
                download_file = gr.File(label="Download", visible=False)

            # ================================================================
            # TAB 1 — Expert Q&A (streaming)
            # ================================================================
            with gr.Tab("🔍  Expert Q&A", id=1):

                with gr.Row():
                    # ── Left: Chat ──────────────────────────────────────────
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="CAPA/8D Expert",
                            height=540,
                            placeholder=(
                                "## Welcome to the CAPA/8D Expert\n\n"
                                "Ask anything about:\n"
                                "- **8D disciplines** D0–D8 and when to use them\n"
                                "- **CAPA procedures** and phase requirements\n"
                                "- **RCA tools** — 5 Whys, Ishikawa, Is/Is Not, FTA\n"
                                "- **FMEA and Control Plans** — how to update after a CAPA\n"
                                "- **Containment** — suspect populations, ICA methods\n"
                                "- **Standards** — ISO 9001, IATF 16949, AS9100"
                            ),
                        )
                        with gr.Row():
                            msg_input  = gr.Textbox(
                                placeholder="Ask a CAPA or 8D question...",
                                label="", scale=5, lines=1, max_lines=3, autofocus=True,
                            )
                            submit_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)
                        with gr.Row():
                            clear_btn   = gr.Button("Clear", variant="secondary", size="sm")
                            use_rewrite = gr.Checkbox(
                                value=True,
                                label="Query rewriting (better recall, slightly slower)",
                                scale=3,
                            )
                        gr.Examples(
                            examples=EXAMPLE_QUESTIONS,
                            inputs=msg_input,
                            label="Example questions",
                        )

                    # ── Right: Sources ──────────────────────────────────────
                    with gr.Column(scale=2):
                        gr.Markdown("### 📄 Source Documents")
                        sources_panel = gr.HTML(
                            value="<p style='color:#8b949e;font-size:13px;'>Ask a question to see retrieved sources.</p>",
                        )

        # ════════════════════════════════════════════════════════════════════
        # Builder AI suggestion handlers
        # ════════════════════════════════════════════════════════════════════

        def d0_ai(d_data):
            d0 = d_data.get(0, {})
            return _ai_html(call_claude_direct(
                f"Problem: {d0.get('title','')}\n{d0.get('description','')}\n"
                "For this quality issue, is a full 8D warranted or a simpler CAPA path? "
                "List 3-4 key triage questions the team should answer first (D0 phase)."
            ))
        d0_ai_btn.click(d0_ai, inputs=[d_data_state], outputs=[d0_ai_out])

        def d1_ai(d_data):
            d0 = d_data.get(0, {})
            return _ai_html(call_claude_direct(
                f"Product: {d0.get('product','')}\nProblem: {d0.get('description','')}\n"
                "List the key functions and roles that should be on the D1 8D team. "
                "Specify why each role is needed for this type of problem."
            ))
        d1_ai_btn.click(d1_ai, inputs=[d_data_state], outputs=[d1_ai_out])

        def d2_ai(d_data):
            d0 = d_data.get(0, {})
            return _ai_html(call_claude_direct(
                f"Problem: {d0.get('description','')}\nProduct: {d0.get('product','')}\n"
                "Suggest the most important IS/IS-NOT dimensions to investigate for this problem. "
                "Focus on what differentiates affected from non-affected."
            ))
        d2_ai_btn.click(d2_ai, inputs=[d_data_state], outputs=[d2_ai_out])

        def d3_ai(d_data):
            d0 = d_data.get(0, {})
            return _ai_html(call_claude_direct(
                f"Problem: {d0.get('description','')}\nProduct: {d0.get('product','')}\n"
                "List all containment locations that must be checked (D3). "
                "Include any commonly missed locations for this type of defect."
            ))
        d3_ai_btn.click(d3_ai, inputs=[d_data_state], outputs=[d3_ai_out])

        def d4_ai(d_data):
            d0 = d_data.get(0, {})
            return _ai_html(call_claude_direct(
                f"Problem: {d0.get('description','')}\nProduct: {d0.get('product','')}\n"
                "Recommend an RCA method (5 Whys, Ishikawa, Is/Is-Not, FTA) and explain why. "
                "List the top 3 likely root cause hypotheses to investigate first."
            ))
        d4_ai_btn.click(d4_ai, inputs=[d_data_state], outputs=[d4_ai_out])

        def d5_ai(d_data):
            d4 = d_data.get(4, {})
            return _ai_html(call_claude_direct(
                f"Root cause: {d4.get('root_cause','')}\n"
                "Suggest effective permanent corrective actions for this root cause. "
                "Flag the most common mistakes that make PCAs ineffective."
            ))
        d5_ai_btn.click(d5_ai, inputs=[d_data_state], outputs=[d5_ai_out])

        def d6_ai(d_data):
            d5 = d_data.get(5, {})
            d4 = d_data.get(4, {})
            return _ai_html(call_claude_direct(
                f"PCA: {d5.get('pca','')}\nRoot cause: {d4.get('root_cause','')}\n"
                "Define specific VoE pass criteria for this corrective action. "
                "Include Cpk thresholds, monitoring period, and zero-escape requirements."
            ))
        d6_ai_btn.click(d6_ai, inputs=[d_data_state], outputs=[d6_ai_out])

        def d7_ai(d_data):
            d5 = d_data.get(5, {})
            return _ai_html(call_claude_direct(
                f"PCA implemented: {d5.get('pca','')}\n"
                "List the D7 systemic actions required: FMEA update, Control Plan revision, "
                "Control Plan updates, procedure changes, training, lateral search scope. "
                "Flag the two most commonly missed D7 items."
            ))
        d7_ai_btn.click(d7_ai, inputs=[d_data_state], outputs=[d7_ai_out])

        def d8_ai(d_data):
            d1 = d_data.get(1, {})
            d0 = d_data.get(0, {})
            return _ai_html(call_claude_direct(
                f"Team: {d1.get('team','')}\nProblem resolved: {d0.get('title','')}\n"
                "Write a short professional team recognition paragraph for closing an 8D report. "
                "Acknowledge collaboration and quality of solution. Under 80 words."
            ))
        d8_ai_btn.click(d8_ai, inputs=[d_data_state], outputs=[d8_ai_out])

        # ════════════════════════════════════════════════════════════════════
        # Save handlers
        # ════════════════════════════════════════════════════════════════════

        def save_d0(d_data, title, desc, product, customer, dt, sev, no):
            d = dict(d_data)
            d[0] = {"title": title, "description": desc, "product": product,
                    "customer": customer, "date": dt, "severity": sev, "report_no": no}
            return d, build_progress_html(d)

        def save_d1(d_data, champion, team):
            d = dict(d_data)
            d[1] = {"champion": champion, "team": team}
            return d, build_progress_html(d)

        def save_d2(d_data, is_what, is_where, is_when, is_how,
                    isnot_what, isnot_where, isnot_when, isnot_how, stmt):
            d = dict(d_data)
            d[2] = {"is": {"what": is_what, "where": is_where, "when": is_when, "how": is_how},
                    "is_not": {"what": isnot_what, "where": isnot_where,
                               "when": isnot_when, "how": isnot_how},
                    "problem_statement": stmt}
            return d, build_progress_html(d)

        def save_d3(d_data, actions, dt, owner, eff):
            d = dict(d_data)
            d[3] = {"actions": actions, "date": dt, "owner": owner, "effectiveness": eff}
            return d, build_progress_html(d)

        def save_d4(d_data, rc, escape, method):
            d = dict(d_data)
            d[4] = {"root_cause": rc, "escape_point": escape, "method": method}
            return d, build_progress_html(d)

        def save_d5(d_data, pca, actions):
            d = dict(d_data)
            d[5] = {"pca": pca, "actions": actions}
            return d, build_progress_html(d)

        def save_d6(d_data, status, verif, dt, by):
            d = dict(d_data)
            d[6] = {"status": status, "verification": verif, "date": dt, "verified_by": by}
            return d, build_progress_html(d)

        def save_d7(d_data, systemic, lessons, similar):
            d = dict(d_data)
            d[7] = {"systemic": systemic, "lessons": lessons, "similar": similar}
            return d, build_progress_html(d)

        def save_d8(d_data, recog, dt, by):
            d = dict(d_data)
            d[8] = {"recognition": recog, "closure_date": dt, "closed_by": by}
            return d, build_progress_html(d)

        d0_save.click(save_d0,
            inputs=[d_data_state, d0_title, d0_desc, d0_product, d0_customer, d0_date, d0_sev, d0_no],
            outputs=[d_data_state, progress_display])
        d1_save.click(save_d1, inputs=[d_data_state, d1_champion, d1_team],
            outputs=[d_data_state, progress_display])
        d2_save.click(save_d2,
            inputs=[d_data_state, d2_is_what, d2_is_where, d2_is_when, d2_is_how,
                    d2_isnot_what, d2_isnot_where, d2_isnot_when, d2_isnot_how, d2_stmt],
            outputs=[d_data_state, progress_display])
        d3_save.click(save_d3,
            inputs=[d_data_state, d3_actions, d3_date, d3_owner, d3_eff],
            outputs=[d_data_state, progress_display])
        d4_save.click(save_d4,
            inputs=[d_data_state, d4_rc, d4_escape, d4_method],
            outputs=[d_data_state, progress_display])
        d5_save.click(save_d5, inputs=[d_data_state, d5_pca, d5_actions],
            outputs=[d_data_state, progress_display])
        d6_save.click(save_d6,
            inputs=[d_data_state, d6_status, d6_verif, d6_date, d6_by],
            outputs=[d_data_state, progress_display])
        d7_save.click(save_d7,
            inputs=[d_data_state, d7_systemic, d7_lessons, d7_similar],
            outputs=[d_data_state, progress_display])
        d8_save.click(save_d8,
            inputs=[d_data_state, d8_recog, d8_date, d8_by],
            outputs=[d_data_state, progress_display])

        # ════════════════════════════════════════════════════════════════════
        # "Ask Expert →" button handlers
        # ════════════════════════════════════════════════════════════════════

        def make_ask_handler(discipline: str):
            def handler(d_data):
                question = build_expert_question(discipline, d_data)
                return question, gr.Tabs(selected=1)
            return handler

        for disc, btn in [
            ("D0", d0_expert), ("D1", d1_expert), ("D2", d2_expert),
            ("D3", d3_expert), ("D4", d4_expert), ("D5", d5_expert),
            ("D6", d6_expert), ("D7", d7_expert), ("D8", d8_expert),
        ]:
            btn.click(
                fn=make_ask_handler(disc),
                inputs=[d_data_state],
                outputs=[msg_input, tabs],
            )

        # ════════════════════════════════════════════════════════════════════
        # Reset + Download
        # ════════════════════════════════════════════════════════════════════

        def reset_all():
            return {}, build_progress_html({})
        reset_btn.click(reset_all, outputs=[d_data_state, progress_display])

        def generate_download(d_data):
            if not d_data:
                return gr.File(visible=False)
            text = build_report_text(d_data)
            report_no = d_data.get(0, {}).get("report_no", "8D_Report")
            path = f"/tmp/{report_no}.txt"
            with open(path, "w") as f:
                f.write(text)
            return gr.File(value=path, visible=True)
        download_btn.click(generate_download, inputs=[d_data_state], outputs=[download_file])

        # ════════════════════════════════════════════════════════════════════
        # Expert Q&A streaming handlers
        # ════════════════════════════════════════════════════════════════════

        submit_inputs  = [msg_input, expert_hist, use_rewrite]
        # Outputs: chatbot (shows history), msg_input (cleared), sources_panel
        submit_outputs = [chatbot, msg_input, sources_panel]

        def on_submit(message, history, use_rw):
            yield from rag_chat_stream(message, history, use_rw)

        # .then() syncs the gr.State with what chatbot holds after streaming completes
        msg_input.submit(
            fn=on_submit, inputs=submit_inputs, outputs=submit_outputs,
        ).then(fn=lambda h: h, inputs=[chatbot], outputs=[expert_hist])

        submit_btn.click(
            fn=on_submit, inputs=submit_inputs, outputs=submit_outputs,
        ).then(fn=lambda h: h, inputs=[chatbot], outputs=[expert_hist])

        clear_btn.click(
            fn=lambda: ([], [], "", "<p style='color:#8b949e'>Ask a question to see sources.</p>"),
            outputs=[chatbot, expert_hist, msg_input, sources_panel],
        )

    return demo


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8D Expert Workbench")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()

    print(f"\n{'─'*55}")
    print(f"  8D Expert Workbench")
    print(f"  http://localhost:{args.port}")
    print(f"  Tab 0: 8D Builder  |  Tab 1: Expert Q&A (streaming)")
    print(f"{'─'*55}\n")

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
        theme=DARK_THEME,
    )
