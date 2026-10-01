"""
Generator: 使用 HuggingFace Transformers 本地加载模型生成回答
- 默认: 本地加载 Qwen2.5-7B-Instruct (单卡 24GB 足够)
- 备选: 兼容 OpenAI API (可切换到 vLLM / DeepSeek / OpenAI 等)
"""

import json
import logging
import re
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Query Planning ──────────────────────────────────────────────────────────
_PLANNING_PROMPT = """You are a query planner for an LLM alignment research assistant.
Break the question into 2-3 focused search queries to retrieve paper evidence.

Mode: {mode}
Question: {question}

Rules:
- comparison: one query per method (e.g. "DPO training stability properties" + "PPO reward hacking instability")
- taxonomy: queries covering different method families (e.g. "RLHF PPO reward model" + "DPO KTO offline preference")
- Each query must be a short English search string (not a full sentence)
- Max 3 queries

Output ONLY a JSON array of strings, nothing else.
Example: ["query one", "query two", "query three"]"""

# ─── Gap Detection ────────────────────────────────────────────────────────────
_COMPARISON_DIMS = [
    "Training Stability",
    "Annotation Cost",
    "Inference Overhead",
    "Alignment Quality",
    "Failure Modes",
]

_GAP_DETECTOR_PROMPT = """You are a coverage checker for an LLM alignment research assistant.

Mode: {mode}
Question: {question}

Required dimensions to cover:
{dimensions}

Retrieved papers (title + snippet):
{doc_summary}

For each required dimension, check if any retrieved paper provides relevant evidence.
Output ONLY a valid JSON object (no other text):
{{"covered": <true/false>, "supplement_queries": ["query1", "query2"]}}

Rules:
- "covered": true only if ALL dimensions have at least partial evidence in the papers above
- "supplement_queries": 1-2 short English search strings targeting the most critical missing dimensions ([] if covered)"""

# ─── Self-Critique ────────────────────────────────────────────────────────────
_CRITIQUE_PROMPT = """Review this research answer for quality issues.

Mode: {mode}
Question: {question}

Paper context (evidence available):
{context}

Initial answer:
{answer}

Check for:
1. Sentences saying "Not reported" where the context above actually contains relevant evidence — fix with [N] citations
2. Factual claims missing [N] citations
3. Comparison mode: ALL 5 dimensions must have content (Training Stability / Annotation Cost / Inference Overhead / Alignment Quality / Failure Modes)
4. Taxonomy mode: every method family mentioned in the question must be covered

If the answer needs significant fixes: output a COMPLETE revised answer with [N] citations.
If the answer is already acceptable (only minor gaps): output exactly: PASS

Your response (PASS or complete revised answer with [N] citations):"""


def _parse_query_list(raw: str, fallback: str) -> list[str]:
    """从 LLM 输出中解析 JSON 列表，失败时返回原始问题作为唯一查询"""
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            queries = json.loads(match.group(0))
            if isinstance(queries, list):
                cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
                if cleaned:
                    return cleaned[:3]
        except (json.JSONDecodeError, ValueError):
            pass
    logger.warning(f"Query planner: failed to parse JSON from output: {raw[:120]!r}")
    return [fallback]


def _parse_gap_result(raw: str) -> dict:
    """从 LLM 输出中解析 Gap Detector 结果，失败时返回 covered=True（保守降级）"""
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "covered" in obj:
                return {
                    "covered": bool(obj.get("covered", True)),
                    "supplement_queries": [
                        q.strip() for q in obj.get("supplement_queries", [])
                        if isinstance(q, str) and q.strip()
                    ][:2],
                }
        except (json.JSONDecodeError, ValueError):
            pass
    logger.warning(f"Gap parser: failed to parse JSON from: {raw[:120]!r}")
    return {"covered": True, "supplement_queries": []}

# ============================================================
# System Prompt — 强制 [N] 引用格式 + few-shot 示范
# ============================================================
SYSTEM_PROMPT = """You are an expert AI research assistant. You answer questions based ONLY on the provided paper excerpts.

## STRICT RULES (violation = failure):

1. **ONLY use provided papers**: Do NOT use your own knowledge. Do NOT invent papers, authors, or arXiv IDs.
2. **Citation format**: Use ONLY numeric citations like [1], [2], [3] that match the provided excerpt numbers. NEVER use author-year format like "(Smith et al., 2023)".
3. **Every factual sentence MUST have a citation**: e.g., "Diffusion models generate images by denoising [1]."
4. **If evidence is insufficient**: Say "Based on the provided papers, I cannot find sufficient information to answer this question." Do NOT make up an answer.
5. **No References section needed**: The system will auto-append references. Just focus on the answer with inline [N] citations.

## Example of CORRECT output:

Progressive distillation reduces sampling steps by training a student model to match two teacher steps in one [1]. Schedule optimization methods search for better noise schedules to improve sample quality with fewer steps [3]. Cached sampling reuses intermediate features across denoising steps to reduce computation [2].

## Example of WRONG output (DO NOT do this):

Progressive distillation (Salimans & Ho, 2022) reduces sampling steps...
[This is WRONG because it uses author-year format instead of [N]]
"""

CONTEXT_TEMPLATE = """Here are {num_docs} paper excerpts (ONLY cite these using [1]-[{num_docs}]):

{context}

---
Question: {question}

Answer using ONLY the above excerpts. Every claim must have [N] citation. Do NOT use author-year format."""


# === 低置信度专用 prompt ===
LOW_CONFIDENCE_SYSTEM_PROMPT = """You are an expert AI research assistant. The retrieval system found very few relevant papers for this question.

## STRICT RULES:
1. Use ONLY numeric citations [1], [2], etc. matching the provided excerpt numbers.
2. NEVER use author-year format like "(Smith et al., 2023)".
3. NEVER invent papers, authors, or arXiv IDs not in the provided excerpts.
4. If the provided papers don't address the question, say so explicitly: "The provided papers do not directly address this question."
5. You may add brief general knowledge ONLY if clearly labeled as "(General knowledge, not from provided papers)".
6. No References section needed — the system auto-appends it.
"""

MEDIUM_CONFIDENCE_TEMPLATE = """Here are {num_docs} paper excerpts (some may have limited relevance, ONLY cite using [1]-[{num_docs}]):

{context}

---
Question: {question}

First assess whether these papers address the question. Answer with [N] citations where possible. Do NOT use author-year format. If papers are insufficient, state clearly."""


# ─────────────────────────────────────────────────────────────────────────────
# AlignRAG: 垂域专用 Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY_SYSTEM_PROMPT = """You are AlignRAG, a specialist assistant for LLM alignment and preference optimization research.

## TASK: Technique Taxonomy
Classify alignment methods from the provided papers into method families and produce a structured taxonomy.

## STRICT RULES:
1. Use ONLY numeric citations [1], [2], ... matching provided excerpt numbers.
2. Only include method families that appear in the provided papers.
3. For each family: Definition, When-to-use, Key differences from neighbors, Representative papers, Limitations.
4. Every factual claim must have a [N] citation.
5. No References section — the system auto-appends it."""

TAXONOMY_TEMPLATE = """Here are {num_docs} alignment paper excerpts:

{context}

---
Question: {question}

Produce a structured taxonomy. For each method family present in the papers:

## [Family Name]
**Definition**: (1-2 sentences) [N]
**When to use**: (conditions or use cases) [N]
**Key differences from neighbors**: (brief contrast) [N]
**Representative papers**: [N], [M]
**Known limitations**: [N]

Only include families supported by the provided papers. Every claim must cite [N]."""

COMPARISON_SYSTEM_PROMPT = """You are AlignRAG, a specialist assistant for LLM alignment research.

## TASK: Method Comparison
Compare alignment methods along 5 fixed dimensions with paper-backed evidence.

## STRICT RULES:
1. Use ONLY numeric citations [1], [2], ...
2. Always address ALL 5 dimensions: Training Stability, Annotation Cost, Inference Overhead, Alignment Quality, Failure Modes.
3. Use per-dimension bullets, not free-form prose.
4. If a dimension is not reported, state "Not reported" and cite the closest relevant paper [N].
5. End with a Recommendation citing [N]."""

COMPARISON_TEMPLATE = """Here are {num_docs} alignment paper excerpts:

{context}

---
Question: {question}

Compare the methods along these 5 dimensions using only the provided papers:

**Training Stability**: [findings with citations]
**Annotation Cost**: [findings with citations]
**Inference Overhead**: [findings with citations]
**Alignment Quality**: [findings with citations]
**Failure Modes**: [findings with citations]

**Recommendation**: (2-3 sentences on when to prefer which method, with citations like [1][2])"""

TABLE_SYSTEM_PROMPT = """You are AlignRAG, a specialist assistant for LLM alignment research.

## TASK: Paper-to-Table Extraction
Extract structured information from alignment papers into a standardized schema.

## STRICT RULES:
1. Use ONLY numeric citations [N] matching provided excerpt numbers.
2. Extract ALL 8 fields: Objective, Training Data, Preference Signal, Optimization Method, Training Stage, Evaluation, Key Conclusion, Limitations.
3. Use "N/A [N]" when a field is not reported in the paper.
4. Extract actual paper claims — do not paraphrase into opinion.
5. Create one table block per paper if multiple papers are relevant."""

TABLE_TEMPLATE = """Here are {num_docs} alignment paper excerpts:

{context}

---
Question: {question}

Extract paper information using this standardized schema. One block per paper:

### [Paper Title] [N]
| Field | Extracted Value |
|-------|----------------|
| Objective | |
| Training Data | |
| Preference Signal | |
| Optimization Method | |
| Training Stage | |
| Evaluation Benchmark | |
| Key Conclusion | |
| Limitations | |

Fill each field from the provided excerpts with inline [N] citations."""


def format_context(retrieved_docs: list[dict], max_docs: int = 8) -> tuple[str, list[dict]]:
    """
    将检索结果格式化为 LLM context。
    同一篇论文（arxiv_id 相同）的多个 chunk 合并为一个条目，避免引用列表出现重复论文。
    """
    # 按 arxiv_id 合并 chunks，保留先出现的（分数更高的）
    seen_arxiv = {}  # arxiv_id → index in merged list
    merged_docs = []  # [(doc_meta, [text1, text2, ...])]

    for doc in retrieved_docs:
        arxiv_id = doc.get("arxiv_id", "")
        if arxiv_id in seen_arxiv:
            # 同论文第二个 chunk，合并文本
            idx = seen_arxiv[arxiv_id]
            merged_docs[idx][1].append(doc["text"])
        else:
            seen_arxiv[arxiv_id] = len(merged_docs)
            merged_docs.append((doc, [doc["text"]]))

        if len(merged_docs) >= max_docs:
            break

    context_parts = []
    citations = []

    for i, (doc, texts) in enumerate(merged_docs):
        idx = i + 1
        authors_str = ", ".join(doc.get("authors", [])[:3])
        if len(doc.get("authors", [])) > 3:
            authors_str += " et al."

        combined_text = "\n\n".join(texts)
        header = f"[{idx}] Title: {doc['title']}\n    Authors: {authors_str}\n    arXiv: {doc['arxiv_id']}"
        context_parts.append(f"{header}\n    Excerpt: {combined_text}")

        citations.append({
            "index": idx,
            "title": doc["title"],
            "arxiv_id": doc["arxiv_id"],
            "authors": doc.get("authors", []),
            "pdf_url": doc.get("pdf_url", f"https://arxiv.org/pdf/{doc['arxiv_id']}"),
            "abs_url": f"https://arxiv.org/abs/{doc['arxiv_id']}",
            "page_numbers": doc.get("page_numbers", []),
        })

    formatted = "\n\n".join(context_parts)
    return formatted, citations


def _postprocess_answer(answer: str, num_docs: int) -> str:
    """
    后处理 LLM 输出:
    1. 将 [1, 3, 4, 5] 格式展开为 [1][3][4][5]
    2. 移除模型自行生成的 References 段
    3. 移除超范围的引用 [N] (N > num_docs)
    """
    # Step 1: 展开逗号分隔引用 [1, 3, 4, 5] → [1][3][4][5]
    def _expand_multi_cite(m):
        nums = re.findall(r"\d+", m.group(0))
        return "".join(f"[{n}]" for n in nums)

    answer = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)+\s*\]", _expand_multi_cite, answer)

    # Step 2: 移除模型自己生成的 References 段落
    answer = re.sub(
        r"\n\s*(?:#{1,3}\s*)?(?:References|参考文献)\s*\n[\s\S]*$",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()

    # Step 3: 移除超范围引用 [N] where N > num_docs or N == 0
    def _filter_citation(m):
        n = int(m.group(1))
        if 1 <= n <= num_docs:
            return m.group(0)
        logger.warning(f"Removed hallucinated citation [{n}] (valid range: [1]-[{num_docs}])")
        return ""

    answer = re.sub(r"\[(\d+)\]", _filter_citation, answer)

    # Step 4: 清理连续空格
    answer = re.sub(r"  +", " ", answer)

    return answer.strip()


class Generator:
    """
    HuggingFace Transformers 本地生成器
    
    直接加载模型到 GPU，不需要额外启动任何服务。
    Qwen2.5-7B-Instruct 大约占 14-15GB 显存 (float16)。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda:1",       # 默认用第 2 张卡，卡 0 给 embedding/reranker
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        torch_dtype: str = "float16",  # float16 / bfloat16 / auto
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # 延迟 import，避免没装 transformers 时报错
        from transformers import AutoTokenizer, AutoModelForCausalLM

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)

        logger.info(f"Loading LLM: {model_name} on {device} (dtype={torch_dtype})")
        logger.info("This may take 1-2 minutes on first run (downloading model)...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if device == "auto":
            # 多卡分布：GPU 0 留给 embedding/reranker，LLM 分布到其余卡
            max_memory = {i: "22000MiB" for i in range(torch.cuda.device_count())}
            max_memory[0] = "0MiB"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
        self.model.eval()

        logger.info(f"LLM loaded successfully! Device map: {device}")

    def generate(
        self,
        question: str,
        retrieved_docs: list[dict],
        max_citations: int = 8,
        confidence: str = "high",
        mode: str = "general",
    ) -> dict:
        """生成带引用的回答，根据 mode 和置信度选择 prompt 策略"""
        context_str, citations = format_context(retrieved_docs, max_docs=max_citations)
        num_docs = len(citations)

        # 垂域专用 mode 优先于置信度路由
        if mode == "taxonomy":
            sys_prompt = TAXONOMY_SYSTEM_PROMPT
            user_prompt = TAXONOMY_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
            logger.info("Mode: taxonomy")
        elif mode == "comparison":
            sys_prompt = COMPARISON_SYSTEM_PROMPT
            user_prompt = COMPARISON_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
            logger.info("Mode: comparison")
        elif mode == "table":
            sys_prompt = TABLE_SYSTEM_PROMPT
            user_prompt = TABLE_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
            logger.info("Mode: table")
        elif confidence == "low":
            sys_prompt = LOW_CONFIDENCE_SYSTEM_PROMPT
            user_prompt = MEDIUM_CONFIDENCE_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
            logger.warning("Low confidence retrieval — using cautious generation prompt")
        elif confidence == "medium":
            sys_prompt = SYSTEM_PROMPT
            user_prompt = MEDIUM_CONFIDENCE_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
            logger.info("Medium confidence — reminding LLM to assess relevance")
        else:
            sys_prompt = SYSTEM_PROMPT
            user_prompt = CONTEXT_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)

        logger.info(f"Generating answer for: {question[:80]}...")

        # 构造 chat messages
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # 用 tokenizer 的 chat template 格式化
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 只取新生成的 token（去掉 prompt 部分）
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"Error generating answer: {e}"

        # 后处理: 清理幻觉引用 + 去掉模型自己生成的 References 段
        answer = _postprocess_answer(answer, num_docs)

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "num_citations": len(citations),
        }

    def plan_queries(self, question: str, mode: str) -> list[str]:
        """把问题分解成 2-3 个子检索查询（temperature=0，确定性输出）"""
        prompt = _PLANNING_PROMPT.format(question=question, mode=mode)
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,          # temperature=0
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            raw = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            queries = _parse_query_list(raw, question)
            logger.info(f"Query planner (local): {queries}")
            return queries
        except Exception as e:
            logger.warning(f"Query planning failed (local): {e}")
            return [question]

    def check_gaps(self, question: str, mode: str, docs: list[dict]) -> dict:
        """
        插入点 2 — Gap Detector:
        检查检索结果是否覆盖所有必要维度，返回缺口情况和补充查询。
        失败时保守降级为 covered=True（不触发补充检索）。
        """
        seen, summaries = set(), []
        for doc in docs:
            aid = doc.get("arxiv_id", "")
            if aid not in seen:
                seen.add(aid)
                snippet = doc.get("text", "")[:150].replace("\n", " ").strip()
                summaries.append(f"- [{doc.get('title', 'Unknown')}]: {snippet}...")
            if len(summaries) >= 8:
                break
        doc_summary = "\n".join(summaries) or "(no papers retrieved)"

        if mode == "comparison":
            dims = "\n".join(f"  - {d}" for d in _COMPARISON_DIMS)
        else:
            dims = ("  - Definitions of each method family\n"
                    "  - Use cases and when to apply\n"
                    "  - Known limitations")

        prompt = _GAP_DETECTOR_PROMPT.format(
            mode=mode, question=question, dimensions=dims, doc_summary=doc_summary)
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=200, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            result = _parse_gap_result(raw)
            logger.info(f"Gap detector: covered={result['covered']}, "
                        f"supplement_queries={result['supplement_queries']}")
            return result
        except Exception as e:
            logger.warning(f"Gap detection failed: {e}")
            return {"covered": True, "supplement_queries": []}

    def critique(self, question: str, mode: str, context_str: str,
                 answer: str, num_docs: int) -> str | None:
        """
        插入点 3 — Self-Critique:
        检查答案质量，发现可改进的问题则返回修订版答案，否则返回 None（PASS）。
        """
        prompt = _CRITIQUE_PROMPT.format(
            mode=mode, question=question, context=context_str, answer=answer)
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            logger.info(f"Self-critique raw output (first 120 chars): {raw[:120]!r}")
            if raw.upper().startswith("PASS"):
                logger.info("Self-critique: PASS (no significant issues found)")
                return None
            revised = _postprocess_answer(raw, num_docs)
            logger.info("Self-critique: revised answer produced")
            return revised
        except Exception as e:
            logger.warning(f"Self-critique failed: {e}")
            return None

    def health_check(self) -> bool:
        """检查模型是否可用"""
        try:
            test_messages = [{"role": "user", "content": "Hi"}]
            text = self.tokenizer.apply_chat_template(
                test_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=5)
            logger.info("LLM health check passed!")
            return True
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False


class APIGenerator:
    """
    备选：通过 OpenAI-compatible API 调用
    适配 vLLM / Ollama / DeepSeek / OpenAI 等
    
    用法:
        gen = APIGenerator(api_base="http://localhost:11434/v1", model="qwen2.5:7b")  # Ollama
        gen = APIGenerator(api_base="https://api.deepseek.com/v1", api_key="sk-xxx")  # DeepSeek
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str = "not-needed",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        logger.info(f"APIGenerator initialized: model={model}, api_base={api_base}")

    def generate(self, question: str, retrieved_docs: list[dict],
                 max_citations: int = 8, confidence: str = "high",
                 mode: str = "general") -> dict:
        context_str, citations = format_context(retrieved_docs, max_docs=max_citations)
        num_docs = len(citations)

        if mode == "taxonomy":
            sys_prompt = TAXONOMY_SYSTEM_PROMPT
            user_prompt = TAXONOMY_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
        elif mode == "comparison":
            sys_prompt = COMPARISON_SYSTEM_PROMPT
            user_prompt = COMPARISON_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
        elif mode == "table":
            sys_prompt = TABLE_SYSTEM_PROMPT
            user_prompt = TABLE_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
        elif confidence == "low":
            sys_prompt = LOW_CONFIDENCE_SYSTEM_PROMPT
            user_prompt = MEDIUM_CONFIDENCE_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
        elif confidence == "medium":
            sys_prompt = SYSTEM_PROMPT
            user_prompt = MEDIUM_CONFIDENCE_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)
        else:
            sys_prompt = SYSTEM_PROMPT
            user_prompt = CONTEXT_TEMPLATE.format(
                context=context_str, question=question, num_docs=num_docs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            answer = f"Error: {e}"

        answer = _postprocess_answer(answer, num_docs)

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "num_citations": len(citations),
        }

    def plan_queries(self, question: str, mode: str) -> list[str]:
        """把问题分解成 2-3 个子检索查询（temperature=0）"""
        prompt = _PLANNING_PROMPT.format(question=question, mode=mode)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            queries = _parse_query_list(raw, question)
            logger.info(f"Query planner (API): {queries}")
            return queries
        except Exception as e:
            logger.warning(f"Query planning failed (API): {e}")
            return [question]

    def check_gaps(self, question: str, mode: str, docs: list[dict]) -> dict:
        """
        插入点 2 — Gap Detector:
        检查检索结果是否覆盖所有必要维度，返回缺口情况和补充查询。
        """
        seen, summaries = set(), []
        for doc in docs:
            aid = doc.get("arxiv_id", "")
            if aid not in seen:
                seen.add(aid)
                snippet = doc.get("text", "")[:150].replace("\n", " ").strip()
                summaries.append(f"- [{doc.get('title', 'Unknown')}]: {snippet}...")
            if len(summaries) >= 8:
                break
        doc_summary = "\n".join(summaries) or "(no papers retrieved)"

        if mode == "comparison":
            dims = "\n".join(f"  - {d}" for d in _COMPARISON_DIMS)
        else:
            dims = ("  - Definitions of each method family\n"
                    "  - Use cases and when to apply\n"
                    "  - Known limitations")

        prompt = _GAP_DETECTOR_PROMPT.format(
            mode=mode, question=question, dimensions=dims, doc_summary=doc_summary)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            result = _parse_gap_result(raw)
            logger.info(f"Gap detector: covered={result['covered']}, "
                        f"supplement_queries={result['supplement_queries']}")
            return result
        except Exception as e:
            logger.warning(f"Gap detection failed (API): {e}")
            return {"covered": True, "supplement_queries": []}

    def critique(self, question: str, mode: str, context_str: str,
                 answer: str, num_docs: int) -> str | None:
        """
        插入点 3 — Self-Critique:
        检查答案质量，发现可改进的问题则返回修订版答案，否则返回 None（PASS）。
        """
        prompt = _CRITIQUE_PROMPT.format(
            mode=mode, question=question, context=context_str, answer=answer)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            raw = response.choices[0].message.content.strip()
            if raw.upper().startswith("PASS"):
                logger.info("Self-critique: PASS (no significant issues found)")
                return None
            revised = _postprocess_answer(raw, num_docs)
            logger.info("Self-critique: revised answer produced")
            return revised
        except Exception as e:
            logger.warning(f"Self-critique failed (API): {e}")
            return None

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


def analyze_generation_quality(result: dict) -> dict:
    """
    #0 生成质量统计:
    - answer_tokens: 回答 token 数（按空格近似）
    - num_citations_used: 实际使用的引用数
    - uncited_sentences: 没有带引用的断言句数量
    - citation_coverage: 带引用句子占比
    """
    import re
    answer = result.get("answer", "")

    # token 数（空格近似）
    answer_tokens = len(answer.split())

    # 实际使用的引用编号
    cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
    num_citations_used = len(cited_nums)

    # 按句子切分，统计无引用的断言句
    sentences = re.split(r"[.!?]\s+", answer)
    uncited = 0
    transitional = re.compile(
        r"^(in summary|overall|in conclusion|to summarize|"
        r"therefore|thus|hence|additionally|moreover|furthermore|"
        r"references|based on)",
        re.IGNORECASE,
    )
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        if transitional.match(sent):
            continue
        if not re.search(r"\[\d+\]", sent):
            uncited += 1

    total_substantive = sum(1 for s in sentences if len(s.strip()) >= 15 and not transitional.match(s.strip()))
    coverage = 1.0 - (uncited / max(total_substantive, 1))

    stats = {
        "answer_tokens": answer_tokens,
        "num_citations_used": num_citations_used,
        "uncited_sentences": uncited,
        "citation_coverage": round(coverage, 3),
    }
    return stats


def format_answer_for_display(result: dict) -> str:
    """格式化答案用于终端显示"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"📝 Question: {result['question']}")

    # 显示检索置信度
    confidence = result.get("confidence", "unknown")
    top_score = result.get("retrieval_top_score", 0)
    margin = result.get("retrieval_margin", 0)
    num_rel = result.get("retrieval_num_relevant", 0)
    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
    lines.append(f"{conf_emoji} Retrieval confidence: {confidence} "
                 f"(top1={top_score:.2f}, margin={margin:.2f}, relevant={num_rel})")

    # 显示生成质量
    gen_stats = result.get("generation_stats", {})
    if gen_stats:
        lines.append(f"📊 Generation: {gen_stats.get('answer_tokens', 0)} tokens, "
                     f"{gen_stats.get('num_citations_used', 0)} citations used, "
                     f"coverage={gen_stats.get('citation_coverage', 0):.0%}, "
                     f"uncited={gen_stats.get('uncited_sentences', 0)}")

    lines.append("=" * 70)
    lines.append("")
    lines.append(result["answer"])
    lines.append("")
    lines.append("-" * 70)
    lines.append("📚 Paper References:")
    lines.append("-" * 70)

    # 只显示回答中实际引用过的论文；如果没有任何引用则显示全部
    used_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", result.get("answer", "")))
    show_citations = [c for c in result["citations"] if c["index"] in used_nums] if used_nums else result["citations"]

    for c in show_citations:
        authors = ", ".join(c["authors"][:2])
        if len(c["authors"]) > 2:
            authors += " et al."
        lines.append(f"  [{c['index']}] {c['title']}")
        lines.append(f"      Authors: {authors}")
        lines.append(f"      arXiv: {c['abs_url']}")
        lines.append(f"      PDF:   {c['pdf_url']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing local generator...")
    gen = Generator(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda:1")
    if gen.health_check():
        print("Local LLM is ready!")
