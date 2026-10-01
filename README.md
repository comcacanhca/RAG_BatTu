# 🔬 AlignRAG

> **Agentic RAG system for LLM alignment research**
> — arXiv → vector index → multi-step agent pipeline → cited answers

Answers comparison and taxonomy questions about alignment methods (DPO, PPO, RLHF, KTO, …) using a 3-step agentic pipeline built on top of hybrid retrieval.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│ Intent Classify │ → mode: comparison / taxonomy / general
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              ① Query Planner  (comparison/taxonomy)     │
│  LLM decomposes question into 2-3 targeted sub-queries  │
│  "DPO stability" · "PPO reward hacking" · "convergence" │
└──┬──────────────────────────────────────────────────────┘
   │ ×N parallel sub-queries
   ▼
┌────────────────────────────────────────────────────────┐
│                  Hybrid Retriever                      │
│  Dense FAISS (bge-large-en-v1.5)                       │
│  + Sparse BM25  →  RRF Fusion                          │
│  + Section Boost  (Method/Experiments ↑ for comparison │
│                    Abstract/Intro ↑ for taxonomy)      │
│  + bge-reranker-large                                  │
└────────────────────────┬───────────────────────────────┘
                         │ merge & dedup
                         ▼
            ┌────────────────────────┐
            │  ② Gap Detector        │  (comparison/taxonomy)
            │  Are all 5 dimensions  │
            │  covered? If not →     │
            │  supplement retrieval  │
            └────────────┬───────────┘
                         │
            ┌────────────▼───────────┐
            │  Memory Boost          │
            │  Jaccard match on past │
            │  queries → promote     │
            │  known-good papers     │
            └────────────┬───────────┘
                         │
            ┌────────────▼───────────┐
            │  Generator             │
            │  Qwen2.5-7B / API      │
            │  mode-specific prompt  │
            └────────────┬───────────┘
                         │
            ┌────────────▼───────────┐
            │  ③ Self-Critique       │  (comparison/taxonomy)
            │  Fix "Not reported"    │
            │  PASS or revise answer │
            └────────────┬───────────┘
                         │
            Answer + [N] Citations
```

---

## Demo

```bash
# 1. Build index — fetch 100 alignment papers from arXiv
python pipeline.py \
  --query "LLM alignment preference optimization DPO PPO RLHF" \
  --max_papers 100

# 2. Ask a comparison question (all 3 agents fire automatically)
python pipeline.py --query any --skip_fetch \
  --ask "Compare DPO vs PPO: training stability, annotation cost, and failure modes"

# 3. Launch Gradio UI with real-time agent trace panel
python app.py   # → http://localhost:7860
```

---

## Ablation: Agent Impact

Results on 9 comparison + taxonomy benchmark questions
(`python eval/run_eval.py --skip_general`):

| Version              | Missing Dims ↓ | Citation Coverage ↑ | Avg Citations | Revised |
|:---------------------|---------------:|--------------------:|--------------:|--------:|
| Base (pipeline only) |          0.320 |               0.481 |           3.1 |       — |
| + Query Planner      |          0.180 |               0.610 |           4.2 |       — |
| + Gap Agent          |          0.090 |               0.740 |           5.0 |       — |
| + Self-Critique      |          0.060 |               0.810 |           5.3 |     31% |

**Missing Dims**: avg `Not reported` occurrences / 5 comparison dimensions
**Citation Coverage**: sentences with `[N]` citation / total substantive sentences
**Revised**: fraction of answers rewritten by Self-Critique

---

## Features

| Component | What it does |
|:----------|:------------|
| **Intent Classifier** | Detects `comparison` / `taxonomy` / `table` / `general` from question keywords |
| **Query Planner** ① | LLM decomposes question → 2-3 sub-queries for parallel multi-faceted retrieval |
| **Hybrid Retriever** | FAISS + BM25 → RRF fusion → section-aware score boost → bge-reranker-large |
| **Section Boost** | Comparison: upweights Method/Experiments. Taxonomy: upweights Abstract/Intro |
| **Gap Detector** ② | Checks all 5 comparison dimensions are covered; fires supplement retrieval if not |
| **Memory Cache** | Cross-query Jaccard match on past citations → promotes known-good papers |
| **Generator** | Mode-specific prompts: comparison table, taxonomy families, table extraction |
| **Self-Critique** ③ | Finds fixable "Not reported" in context; revises answer or passes |

---

## Hardware

| Setup | Config |
|:------|:-------|
| 2 GPUs (recommended) | GPU 0: Embedding + Reranker (~3 GB) · GPU 1: LLM (~15 GB) |
| 1 GPU (24 GB) | Set all `device` fields to `cuda:0` in `configs/config.yaml` |
| CPU only | Remove CUDA devices; slow but functional for small corpora |

---

## Quick Config

```yaml
# configs/config.yaml

llm:
  backend: "local"                # "local" = HuggingFace Transformers
  model: "Qwen/Qwen2.5-7B-Instruct"
  # Switch to any OpenAI-compatible API:
  # backend: "api"
  # api_base: "http://localhost:11434/v1"   # Ollama
  # api_base: "https://api.deepseek.com/v1" # DeepSeek
  # model: "qwen2.5:7b"

agents:
  query_planner: true             # ① multi-query decomposition
  gap_detector: true              # ② evidence gap fill
  self_critique: true             # ③ answer revision
```

---

## Project Structure

```
alignrag/
├── pipeline.py                  # End-to-end pipeline (main entry)
├── app.py                       # Gradio UI with agent trace panel
├── qa_server.py                 # Rich CLI interactive server
│
├── src/
│   ├── arxiv_fetcher.py         # arXiv search + PDF download
│   ├── pdf_parser.py            # PyMuPDF + section detection + domain schema
│   ├── embedder.py              # bge-large-en-v1.5 encoder
│   ├── indexer.py               # FAISS + BM25 indexes
│   ├── retriever.py             # Hybrid retrieval + RRF + section boost + reranker
│   └── generator.py             # Generator + Query Planner + Gap Detector + Critique
│
├── eval/
│   ├── benchmark_questions.json # 11 comparison/taxonomy/general benchmark questions
│   └── run_eval.py              # 4-config ablation study
│
├── configs/config.yaml          # All hyperparameters + per-agent toggles
└── data/
    ├── faiss_index/             # Persisted FAISS vector index
    ├── query_memory.json        # Cross-query memory cache (auto-generated)
    └── eval_logs/               # Retrieval + generation diagnostics (JSONL)
```

---

## Stack

`bge-large-en-v1.5` · `bge-reranker-large` · `Qwen2.5-7B-Instruct` · `FAISS` · `BM25Okapi` · `PyMuPDF` · `LangChain text splitters` · `Gradio`

## License

This project is licensed under the MIT License.
