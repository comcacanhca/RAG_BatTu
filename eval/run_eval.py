"""
Ablation Evaluation — AlignRAG Pipeline
========================================
运行 4 个配置的消融实验，输出对比指标表格。

用法:
    python eval/run_eval.py --config configs/config.yaml
    python eval/run_eval.py --config configs/config.yaml --questions eval/benchmark_questions.json
    python eval/run_eval.py --skip_general   # 只评估 comparison/taxonomy 问题

指标说明:
    missing_dim_rate  — comparison 模式下 "Not reported" 出现次数 / 5 维度
    citation_coverage — 带 [N] 引用的句子占全部断言句比例
    avg_citations     — 回答中实际使用的论文数
    revised_rate      — 被 Self-Critique 修订的回答比例（仅 +all 配置有意义）
"""

import sys
import json
import re
import copy
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import RAGPipeline, load_config
from src.generator import analyze_generation_quality

logging.basicConfig(level=logging.WARNING)   # 消融时抑制 INFO 输出，只看结果
logger = logging.getLogger("eval")

# ─── 4 个消融配置 ─────────────────────────────────────────────────────────────
ABLATION_CONFIGS = [
    {
        "name": "Base (no agents)",
        "agents": {"query_planner": False, "gap_detector": False, "self_critique": False},
    },
    {
        "name": "+ Planner",
        "agents": {"query_planner": True,  "gap_detector": False, "self_critique": False},
    },
    {
        "name": "+ Gap Agent",
        "agents": {"query_planner": True,  "gap_detector": True,  "self_critique": False},
    },
    {
        "name": "+ Self-Critique (full)",
        "agents": {"query_planner": True,  "gap_detector": True,  "self_critique": True},
    },
]

# 5 个 comparison 维度（用于检测 "Not reported"）
COMPARISON_DIMS = [
    "Training Stability",
    "Annotation Cost",
    "Inference Overhead",
    "Alignment Quality",
    "Failure Modes",
]


# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def compute_missing_dim_rate(answer: str, mode: str) -> float:
    """comparison 模式: 统计 'Not reported' 出现次数，归一化到 [0, 1]"""
    if mode != "comparison":
        return float("nan")
    count = len(re.findall(r"not reported", answer, re.IGNORECASE))
    return round(count / len(COMPARISON_DIMS), 3)


def compute_family_coverage(answer: str, key_methods: list[str]) -> float:
    """taxonomy/comparison: 检查关键方法族在回答中出现的比例"""
    if not key_methods:
        return float("nan")
    found = sum(1 for m in key_methods if re.search(rf"\b{re.escape(m)}\b", answer, re.IGNORECASE))
    return round(found / len(key_methods), 3)


def evaluate_result(result: dict, question_meta: dict) -> dict:
    """计算单条结果的所有指标"""
    answer = result.get("answer", "")
    mode = question_meta["mode"]
    gen_stats = result.get("generation_stats") or analyze_generation_quality(result)

    return {
        "missing_dim_rate":   compute_missing_dim_rate(answer, mode),
        "citation_coverage":  gen_stats.get("citation_coverage", 0.0),
        "avg_citations":      gen_stats.get("num_citations_used", 0),
        "uncited_sentences":  gen_stats.get("uncited_sentences", 0),
        "answer_revised":     int(result.get("answer_revised", False)),
        "family_coverage":    compute_family_coverage(answer, question_meta.get("key_methods", [])),
        "answer_tokens":      gen_stats.get("answer_tokens", 0),
    }


# ─── 主评估循环 ───────────────────────────────────────────────────────────────

def run_ablation(pipeline: RAGPipeline, questions: list[dict],
                 skip_general: bool = False) -> dict:
    """
    对所有问题跑 4 个配置，返回结构:
    { config_name: { qid: metrics_dict } }
    """
    results = {}

    for abl in ABLATION_CONFIGS:
        name = abl["name"]
        print(f"\n{'─'*60}")
        print(f"Running: {name}")
        print(f"{'─'*60}")

        # 临时覆盖 config 中的 agents 设置
        pipeline.config["agents"] = copy.deepcopy(abl["agents"])

        config_results = {}
        for q in questions:
            if skip_general and q["mode"] == "general":
                continue
            qid = q["id"]
            print(f"  [{qid}] {q['question'][:70]}...")
            try:
                result = pipeline.query(q["question"])
                metrics = evaluate_result(result, q)
                config_results[qid] = metrics
                cov = metrics["citation_coverage"]
                mdr = metrics["missing_dim_rate"]
                print(f"    coverage={cov:.0%}  "
                      f"missing_dims={mdr if mdr == mdr else 'n/a'}  "
                      f"citations={metrics['avg_citations']}")
            except Exception as e:
                logger.error(f"  Error on {qid}: {e}")
                config_results[qid] = {}

        results[name] = config_results

    return results


def aggregate(results: dict, questions: list[dict]) -> dict:
    """按配置聚合指标（过滤掉 nan）"""
    qid_to_mode = {q["id"]: q["mode"] for q in questions}
    agg = {}
    for name, qmap in results.items():
        metrics_lists: dict[str, list] = {
            "missing_dim_rate": [], "citation_coverage": [],
            "avg_citations": [], "uncited_sentences": [],
            "answer_revised": [], "family_coverage": [],
        }
        for qid, m in qmap.items():
            for k in metrics_lists:
                v = m.get(k)
                if v is None:
                    continue
                # missing_dim_rate 只对 comparison 有意义
                if k == "missing_dim_rate" and qid_to_mode.get(qid) != "comparison":
                    continue
                if isinstance(v, float) and v != v:   # nan check
                    continue
                metrics_lists[k].append(v)

        def _mean(lst):
            return round(sum(lst) / len(lst), 3) if lst else "—"

        agg[name] = {k: _mean(v) for k, v in metrics_lists.items()}
    return agg


def print_table(agg: dict):
    """打印消融对比表格"""
    print("\n" + "=" * 90)
    print("ABLATION RESULTS")
    print("=" * 90)

    cols = ["missing_dim_rate ↓", "citation_coverage ↑", "avg_citations ↑",
            "uncited_sents ↓", "revised_rate"]
    keys = ["missing_dim_rate", "citation_coverage", "avg_citations",
            "uncited_sentences", "answer_revised"]
    col_w = 20

    header = f"{'Config':<30}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(header)
    print("─" * len(header))

    for name, m in agg.items():
        row = f"{name:<30}"
        for k in keys:
            v = m.get(k, "—")
            if isinstance(v, float):
                row += f"{v:>{col_w}.3f}"
            else:
                row += f"{str(v):>{col_w}}"
        print(row)

    print("=" * 90)
    print("\nNotes:")
    print("  missing_dim_rate: avg 'Not reported' / 5 dims (comparison Qs only)")
    print("  citation_coverage: fraction of sentences with [N] citation")
    print("  avg_citations: unique papers cited per answer")
    print("  uncited_sents: sentences missing citations")
    print("  revised_rate: fraction of answers revised by Self-Critique")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AlignRAG Ablation Eval")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--questions", default="eval/benchmark_questions.json")
    parser.add_argument("--skip_general", action="store_true",
                        help="Skip general-mode questions (focus on comparison/taxonomy)")
    parser.add_argument("--output", default=None,
                        help="Save full results to JSON (optional)")
    args = parser.parse_args()

    print("Loading pipeline...")
    config = load_config(args.config)
    pipeline = RAGPipeline(config)
    pipeline.init_models(skip_llm=False)
    pipeline.load_existing_index()
    print("Pipeline ready.\n")

    questions_path = PROJECT_ROOT / args.questions
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} benchmark questions.")

    results = run_ablation(pipeline, questions, skip_general=args.skip_general)
    agg = aggregate(results, questions)
    print_table(agg)

    if args.output:
        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"aggregated": agg, "per_question": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
