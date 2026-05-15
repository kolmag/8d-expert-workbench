# 8D Expert Workbench

An integrated quality engineering workbench combining a **guided 8D report builder** with a **production-grade RAG expert system** — built for quality engineers who need both structured problem-solving support and grounded domain expertise in a single interface.

---

## What It Does

The workbench has two tabs that work together:

**Tab 1 — 8D Builder**
Step-by-step guided form covering all nine 8D disciplines (D0–D8). Each discipline has:
- Structured input fields for the required content
- **AI Suggest** button — calls Claude Haiku for fast, context-aware draft suggestions
- **↗ Ask Expert** button — pre-fills the Expert Q&A tab with a context-aware question drawn from your current report state and switches tabs automatically

**Tab 2 — Expert Q&A**
Production RAG pipeline answering questions grounded in a 14-document CAPA/8D knowledge base. Every answer is retrieved, reranked, generated, and checked for groundedness before being shown.

**The integration:** Click "↗ Ask Expert" next to any discipline → the system reads your current problem description, product, and discipline-specific context → pre-fills a targeted expert question → switches to the Expert tab. You see the question, edit if needed, and ask. The answer comes from the KB, not from general LLM knowledge.

---

## Pipeline Architecture

```
User question
    ↓
Query rewriting       Claude Haiku — 3 alternative phrasings
    ↓
Retrieval             text-embedding-3-small + Chroma (top 30 candidates)
    ↓
Reranking             BAAI/bge-reranker-v2-m3 (local cross-encoder, top 5)
    ↓
Answer generation     GPT-4o-mini
    ↓
Groundedness check    Claude Haiku — strips ungrounded claims
    ↓
Answer + sources
```

Multi-LLM orchestration: GPT-4o-mini generates, Claude Haiku audits. Model diversity between generator and critic reduces self-leniency — a single model checking its own output has the same blind spots as when it generated it.

---

## Knowledge Base

14 documents covering CAPA and 8D methodology across multiple industries:

| Document | Coverage |
|---|---|
| 8D_problem_solving_methodology.md | Complete D0–D8 framework |
| CAPA_SOP_enriched.md | CAPA procedure and requirements |
| containment_decision_guide.md | D3 ICA decision logic + practitioner scenarios |
| root_cause_analysis.md | RCA tools and selection |
| rca_tool_selection_matrix.md | 5 Whys, Ishikawa, Is/Is Not, FTA |
| is_is_not_analysis.md | Is/Is Not methodology |
| fmea_basics.md | FMEA and control plan integration |
| effectiveness_verification_guide.md | D6 VoE methodology |
| control_plan_basics.md | Control plan structure |
| capa_edge_cases.md | Edge cases and boundary decisions |
| 8d_practitioner_scenarios.md | Real-world practitioner scenarios |
| 8D_report_example_automotive.md | Worked automotive example with FINDING anchors |
| 8D_report_example_semiconductor.md | Worked semiconductor example |
| multi-industry_CAPA_8D_compliance.md | ISO 9001, IATF 16949 compliance |

Each document is LLM-enriched at ingest time — Claude Haiku generates a headline, summary, and 3 practitioner queries per chunk. These are embedded alongside the original text, bridging formal SOP vocabulary with conversational question phrasing.

---

## Eval Results

197-question evaluation across 3 independent sources (developer, blind Gemini, blind ChatGPT):

| Run | Overall | Groundedness | Top Chunk |
|---|---|---|---|
| Baseline | 6.926 | 7.098 | 4.049 |
| + FINDING anchors | 6.774 | 6.928 | 4.048 |
| + Guardrails + Markdown chunking | 7.094 | 7.487 | 5.120 |
| + Synthetic queries + Containment content | **7.121** | **7.456** | 4.424 |

**Model benchmark** — same test set, same judge (Claude Sonnet 4.5):

| Stack | Overall | Checker | Est. cost/197q | Median latency |
|---|---|---|---|---|
| GPT-4o-mini + Claude Haiku | **7.121** | 0.671 | ~$2.50 | ~28s |
| Llama 3.3 70B full-stack (Groq) | 6.942 | 0.590 | ~$1.20 | ~32s |

Llama 3.3 70B is 52% cheaper and competitive on methodology questions. GPT-4o-mini wins significantly on compliance-heavy content (-0.920 on `compliance` category) — formal standards vocabulary favours GPT-4o-mini.

---

## Setup

```bash
# Clone
git clone https://github.com/kolmag/8d-expert-workbench.git
cd 8d-expert-workbench

# Install dependencies
uv sync
# This installs the shared RAG core from:
# https://github.com/kolmag/capa-8d-expert

# Copy environment variables
cp .env.example .env
# Add: OPENAI_API_KEY, ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# Ingest knowledge base (first run only)
uv run scripts/ingest.py

# Launch workbench
uv run workbench_app.py

# With public share link
uv run workbench_app.py --share
```

**Note on BGE reranker:** Requires PyTorch. On M1 Mac with 8GB RAM, runs via MPS but may OOM if other processes compete for memory. The app gracefully falls back to LLM reranker if BGE is unavailable — set `--reranker llm` to force the fallback.

---

## Diagnostics

```bash
# Embedding space visualisation — run after any KB change
uv run scripts/diagnostics/tsne_viz.py \
    --db_path ./chroma_db \
    --collection capa_8d_expert \
    --dims 2

# Cosine similarity analysis
uv run scripts/diagnostics/sc_viz.py \
    --db_path ./chroma_db \
    --collection capa_8d_expert
```

---

## Production Roadmap

See [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md) for the full five-phase transition path from this portfolio implementation to a production-ready enterprise system.

**Known gaps (portfolio → production):**

| Concern | Current | Production |
|---|---|---|
| State | `gr.State` (in-memory) | PostgreSQL + report_id |
| Vector store | Local Chroma | Qdrant Cloud / Pinecone |
| Reranker | Local BGE (OOM under load) | Dedicated GPU microservice |
| Auth | None | OAuth2/OIDC + RBAC |
| KB updates | Manual `--reset` | CI/CD auto-ingest |

---

## Related

- [capa-8d-expert](https://github.com/kolmag/capa-8d-expert) — reusable RAG core and standalone Expert Q&A app. This workbench depends on that package instead of duplicating the pipeline.

---

## License

MIT
