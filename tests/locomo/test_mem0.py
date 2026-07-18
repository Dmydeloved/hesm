"""
Mem0 Test Pipeline — 与 HESM test_pipeline.py 流程完全一致。

流程：
    构建/挂载 Mem0 记忆 → Retrieval → Answer Generation → Evaluate

【设计约束】
- 记忆路径：outputs/locomo/memory/mem0_{conv_id}/chroma/（与主实验一致）
  首次运行自动构建并写 .built 标记；后续跳过重建。
- 问题集：共用 tests/locomo/questions.json（与 HESM 测试相同题目）。
- 完全隔离：所有测试产物写到 tests/locomo/results/mem0/。
- 流程一致：Mem0.search → LLMAnswerGenerator → F1 / LLM-Judge(0/1) / Retrieval@K。

Usage (from d:/code/hesm):
    python -m tests.locomo.test_mem0
    python -m tests.locomo.test_mem0 --rebuild-memory   # 强制重建 Mem0 记忆
    python -m tests.locomo.test_mem0 --dry-run          # 跳过 LLM 调用，验证基础设施

Outputs (tests/locomo/results/mem0/):
    steps.jsonl   — 每道题检索/回答/评估的 JSONL 节点流
    answers.json  — 所有题目完整记录 + 指标
    metrics.json  — 聚合指标（与主实验相同 schema）
    test.log      — 文件日志（UTF-8）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ─── 路径常量 ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_TESTS_DIR     = Path(__file__).parent
_RESULTS_DIR   = _TESTS_DIR / "results" / "mem0"

# 问题集：与 HESM 测试共用
_QUESTIONS_FILE = _TESTS_DIR / "questions.json"

# Mem0 记忆存在主实验目录（与 run_main.py 路径一致）
_MAIN_MEMORY_ROOT = _PROJECT_ROOT / "outputs" / "locomo" / "memory"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── 日志 ─────────────────────────────────────────────────────────────────────
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_RESULTS_DIR / "test.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step Tracer
# ─────────────────────────────────────────────────────────────────────────────

class StepTracer:
    """
    向 JSONL 写入流水线步骤，每条记录：
      {"q_index": int, "step": str, "ts": float, "data": dict}

    step 取值："retrieval" | "answer" | "evaluation"
    （Mem0 不做 topic extraction，所以没有该步骤）
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        logger.info("步骤追踪 → %s", path)

    def log(self, q_index: int, step: str, data: dict[str, Any]) -> None:
        entry = {"q_index": q_index, "step": step, "ts": time.time(), "data": data}
        self._f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# 记忆构建 / 挂载
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_build_mem0(
    conv_id: str,
    hesm_cfg: dict[str, Any],
    force_rebuild: bool = False,
) -> Any:
    """
    返回已初始化的 Mem0Memory 实例。

    - 若 .built 标记存在且 force_rebuild=False → 直接挂载已有 Chroma
    - 否则 → 构建记忆，写 .built 标记
    """
    from experiments.locomo.data.loader import LoCoMoLoader
    from experiments.locomo.methods.mem0_adapter import Mem0Memory

    exp_cfg_path = (
        _PROJECT_ROOT / "experiments" / "locomo" / "config" / "experiment.yaml"
    )
    import yaml
    with open(exp_cfg_path, encoding="utf-8") as f:
        exp_cfg: dict[str, Any] = yaml.safe_load(f)

    mem0_section = exp_cfg.get("mem0", {})
    collection_prefix = mem0_section.get("collection_prefix", "mem0_locomo")

    mem0 = Mem0Memory(
        memory_root=_MAIN_MEMORY_ROOT,
        hesm_config=hesm_cfg,
        collection_prefix=collection_prefix,
    )

    marker = _MAIN_MEMORY_ROOT / f"mem0_{conv_id}.built"
    chroma_dir = _MAIN_MEMORY_ROOT / f"mem0_{conv_id}" / "chroma"

    already_built = marker.exists() and chroma_dir.exists() and not force_rebuild
    if already_built:
        logger.info("Mem0 记忆已存在，跳过构建 (marker: %s)", marker)
        logger.info("Chroma: %s", chroma_dir)
        # 只初始化 _memory 对象和 user_id，不重放 turns
        mem0._user_id = conv_id
        mem0._memory = mem0._create_mem0(conv_id)
        if mem0._memory is None:
            raise RuntimeError("Mem0 初始化失败，请检查 mem0ai 安装和 API 配置")
        return mem0

    # ── 构建 ──────────────────────────────────────────────────────────────
    dataset_path = _PROJECT_ROOT / exp_cfg["dataset"]["path"]
    loader = LoCoMoLoader(dataset_path)
    conversations = loader.load(max_conversations=1)
    if not conversations:
        raise RuntimeError(f"无法加载对话: {dataset_path}")

    conv = conversations[0]
    logger.info(
        "构建 Mem0 记忆: %s  (%d sessions, %d turns)",
        conv_id, len(conv.sessions), len(conv.all_turns),
    )
    t0 = time.time()
    mem0.reset()
    mem0.build_memory(
        conv_id=conv_id,
        sessions=conv.sessions,
        speaker_a=conv.speaker_a,
        speaker_b=conv.speaker_b,
    )
    elapsed = time.time() - t0

    # mem0._memory is None means _create_mem0 failed (mem0ai not installed or API error)
    if mem0._memory is None:
        raise RuntimeError(
            "Mem0 初始化失败：mem0ai 未安装或 API 配置有误。\n"
            "请运行: pip install mem0ai"
        )

    logger.info("Mem0 记忆构建完成: %.1fs", elapsed)

    # 只有真正成功才写标记，防止下次跳过构建但 _memory 仍为 None
    marker.write_text(
        json.dumps({"conv_id": conv_id, "elapsed_s": round(elapsed, 2)}),
        encoding="utf-8",
    )
    return mem0


# ─────────────────────────────────────────────────────────────────────────────
# 单题流水线
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_question(
    q_idx: int,
    total: int,
    item: dict[str, Any],
    conv_token_count: int,
    mem0: Any,
    answer_generator: Any,
    judge: Any,
    tracer: StepTracer,
    dry_run: bool,
) -> dict[str, Any]:
    """
    单题完整流水线：Retrieval → Answer → Evaluate。
    Mem0 无 TopicExtractor 步骤；直接调用 memory.search(question)。
    """
    from experiments.locomo.evaluation.f1 import compute_f1
    from experiments.locomo.evaluation.retrieval_metrics import compute_retrieval_metrics

    question     = item["question"]
    ground_truth = item["answer"]
    evidence_ids: list[str] = item.get("evidence", [])
    category: int = item.get("category", 0)

    logger.info("─" * 66)
    logger.info(
        "Q%d/%d  [cat=%d]  %s",
        q_idx + 1, total, category,
        question[:80] + ("…" if len(question) > 80 else ""),
    )
    logger.info("  期望证据: %s", evidence_ids)
    logger.info("  标准答案: %s", ground_truth[:100])

    # ── Step 1: Retrieval ────────────────────────────────────────────────
    t1 = time.time()
    if dry_run:
        from experiments.locomo.methods.base import RetrievalResult
        ret = RetrievalResult("(dry run)", [], 3, {"num_results": 0})
    else:
        ret = mem0.retrieve(question=question, top_k=5)

    t1_ms = (time.time() - t1) * 1000
    n_results = ret.raw_result.get("num_results", len(ret.retrieved_ids))
    hits = set(ret.retrieved_ids) & set(evidence_ids)

    logger.info(
        "  [1/3] Retrieval (%.0fms): results=%d  tokens=%d",
        t1_ms, n_results, ret.token_count,
    )
    logger.info("        retrieved_ids : %s", ret.retrieved_ids)
    logger.info("        命中证据: %s / %s", sorted(hits), evidence_ids)
    logger.info(
        "        context preview: %s",
        ret.context_text[:200].replace("\n", " ") if ret.context_text else "(空)",
    )

    tracer.log(q_idx, "retrieval", {
        "question":         question,
        "retrieved_ids":    ret.retrieved_ids,
        "evidence_ids":     evidence_ids,
        "hits":             sorted(hits),
        "num_results":      n_results,
        "retrieved_tokens": ret.token_count,
        "context_preview":  ret.context_text[:600],
        "latency_ms":       round(t1_ms, 1),
    })

    # ── Step 2: Answer Generation ────────────────────────────────────────
    t2 = time.time()
    if dry_run:
        prediction = "(dry run — no LLM call)"
    else:
        try:
            prediction = answer_generator.generate(
                question=question,
                context=ret.context_text,
                speaker_a=item.get("speaker_a", "Speaker A"),
                speaker_b=item.get("speaker_b", "Speaker B"),
            )
        except Exception as exc:
            logger.warning("  [2] Answer Generation FAILED: %s", exc)
            prediction = ""

    t2_ms = (time.time() - t2) * 1000
    logger.info(
        "  [2/3] Answer Generation (%.0fms):\n"
        "        预测: %s\n"
        "        标准: %s",
        t2_ms,
        prediction[:150],
        ground_truth[:150],
    )

    tracer.log(q_idx, "answer", {
        "question":       question,
        "ground_truth":   ground_truth,
        "prediction":     prediction,
        "context_tokens": ret.token_count,
        "latency_ms":     round(t2_ms, 1),
    })

    # ── Step 3: Evaluation ───────────────────────────────────────────────
    t3 = time.time()

    f1_res = compute_f1(prediction, ground_truth)

    if dry_run:
        judge_score = -1
    else:
        try:
            judge_score = judge.judge(
                question=question,
                ground_truth=ground_truth,
                prediction=prediction,
            )
        except Exception as exc:
            logger.warning("  Judge FAILED: %s", exc)
            judge_score = -1

    rm_by_k = compute_retrieval_metrics(
        ret.retrieved_ids, evidence_ids, k_values=[1, 3, 5]
    )
    ret_metrics_dict = {
        str(k): {
            "recall":    rm.recall,
            "precision": rm.precision,
            "f1":        rm.f1,
            "accuracy":  rm.accuracy,
        }
        for k, rm in rm_by_k.items()
    }

    compression = (
        conv_token_count / ret.token_count if ret.token_count > 0 else None
    )

    t3_ms = (time.time() - t3) * 1000
    judge_label = {1: "CORRECT ✓", 0: "WRONG ✗", -1: "FAILED"}.get(judge_score, "?")
    logger.info(
        "  [3/3] Evaluation (%.0fms): "
        "F1=%.3f  P=%.3f  R=%.3f  Judge=%s  "
        "Recall@1=%.2f  @3=%.2f  @5=%.2f  compress=%.0fx",
        t3_ms,
        f1_res["f1"], f1_res["precision"], f1_res["recall"],
        judge_label,
        rm_by_k[1].recall, rm_by_k[3].recall, rm_by_k[5].recall,
        compression,
    )

    tracer.log(q_idx, "evaluation", {
        "question":     question,
        "ground_truth": ground_truth,
        "prediction":   prediction,
        "f1":           f1_res["f1"],
        "f1_precision": f1_res["precision"],
        "f1_recall":    f1_res["recall"],
        "judge_score":  judge_score,
        "judge_label":  judge_label.split()[0],
        "retrieval_metrics": ret_metrics_dict,
        "total_conversation_tokens": conv_token_count,
        "compression_ratio": round(compression, 2) if compression is not None else None,
        "latency_ms":   round(t3_ms, 1),
    })

    total_ms = (time.time() - t1) * 1000
    return {
        # 标识
        "q_index":          q_idx,
        "original_q_index": item.get("q_index", q_idx),
        "category":         category,
        "conv_id":          item.get("conv_id", ""),
        # 问答文本
        "question":         question,
        "ground_truth":     ground_truth,
        "prediction":       prediction,
        # 检索
        "retrieved_ids":    ret.retrieved_ids,
        "retrieved_context": ret.context_text,
        "retrieved_tokens": ret.token_count,
        "evidence":         evidence_ids,
        # 答案质量
        "f1":               f1_res["f1"],
        "f1_precision":     f1_res["precision"],
        "f1_recall":        f1_res["recall"],
        "judge_score":      judge_score,
        # 检索质量
        "retrieval_metrics": ret_metrics_dict,
        # 效率
        "total_conversation_tokens": conv_token_count,
        "compression_ratio": round(compression, 2) if compression is not None else None,
        # 各步耗时 (ms)
        "latency_retrieval_ms": round(t1_ms, 1),
        "latency_answer_ms":    round(t2_ms, 1),
        "latency_eval_ms":      round(t3_ms, 1),
        "latency_total_ms":     round(total_ms, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    rebuild_memory: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """运行 Mem0 测试流水线，返回聚合指标 dict。"""
    import yaml
    from experiments.locomo.data.loader import LoCoMoLoader
    from experiments.locomo.evaluation.aggregator import aggregate
    from experiments.locomo.evaluation.judge import LLMJudge
    from experiments.locomo.evaluation.token_metrics import count_tokens
    from experiments.locomo.methods.base import LLMAnswerGenerator

    logger.info("=" * 66)
    logger.info("Mem0 Test Pipeline  |  方法=mem0  问题集=%s", _QUESTIONS_FILE.name)
    logger.info("=" * 66)

    # ── 加载配置 ────────────────────────────────────────────────────────
    with open(_PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        hesm_cfg: dict[str, Any] = yaml.safe_load(f)

    exp_cfg_path = (
        _PROJECT_ROOT / "experiments" / "locomo" / "config" / "experiment.yaml"
    )
    with open(exp_cfg_path, encoding="utf-8") as f:
        exp_cfg: dict[str, Any] = yaml.safe_load(f)

    # ── 加载问题集 ────────────────────────────────────────────────────────
    if not _QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"questions.json 不存在: {_QUESTIONS_FILE}\n"
            "请先运行: python -m tests.locomo.questions_builder"
        )

    with open(_QUESTIONS_FILE, encoding="utf-8") as f:
        q_data: dict[str, Any] = json.load(f)

    conv_id   = q_data["conv_id"]
    speaker_a = q_data["speaker_a"]
    speaker_b = q_data["speaker_b"]
    questions = q_data["questions"]

    for q in questions:
        q.setdefault("speaker_a", speaker_a)
        q.setdefault("speaker_b", speaker_b)

    from collections import Counter
    cat_dist = Counter(q["category"] for q in questions)
    logger.info(
        "问题集: %d 题  conv=%s  (%s & %s)",
        len(questions), conv_id, speaker_a, speaker_b,
    )
    for cat in sorted(cat_dist):
        logger.info("  Category %d: %d 题", cat, cat_dist[cat])

    # ── 构建或挂载 Mem0 记忆 ─────────────────────────────────────────────
    mem0 = _get_or_build_mem0(conv_id, hesm_cfg, force_rebuild=rebuild_memory)

    # ── 对话总 token（用于压缩比）────────────────────────────────────────
    dataset_path = _PROJECT_ROOT / exp_cfg["dataset"]["path"]
    loader = LoCoMoLoader(dataset_path)
    conversations = loader.load(max_conversations=1)
    conv = conversations[0]
    conv_token_count = count_tokens(conv.total_turn_text())
    logger.info(
        "对话总 tokens: %d  (%d sessions, %d turns)",
        conv_token_count, len(conv.sessions), len(conv.all_turns),
    )

    # ── 初始化共享组件 ──────────────────────────────────────────────────
    answer_generator = LLMAnswerGenerator(hesm_cfg)
    judge = LLMJudge(hesm_cfg)

    # ── 打开步骤追踪器 ──────────────────────────────────────────────────
    tracer = StepTracer(_RESULTS_DIR / "steps.jsonl")

    # ── 逐题执行 ─────────────────────────────────────────────────────────
    total = len(questions)
    logger.info("=" * 66)
    logger.info("开始执行 %d 道题 ...", total)
    logger.info("=" * 66)

    qa_records: list[dict[str, Any]] = []
    t_start = time.time()

    for q_idx, item in enumerate(questions):
        try:
            record = _run_one_question(
                q_idx=q_idx,
                total=total,
                item=item,
                conv_token_count=conv_token_count,
                mem0=mem0,
                answer_generator=answer_generator,
                judge=judge,
                tracer=tracer,
                dry_run=dry_run,
            )
            qa_records.append(record)
        except Exception as exc:
            logger.error("Q%d 异常: %s", q_idx + 1, exc, exc_info=True)

    tracer.close()
    elapsed = time.time() - t_start

    # ── 聚合指标 ─────────────────────────────────────────────────────────
    metrics = aggregate("mem0_test", qa_records)

    # ── 打印汇总 ─────────────────────────────────────────────────────────
    logger.info("=" * 66)
    logger.info(
        "完成  %d 题 / %.1fs (avg %.1fs/题)",
        total, elapsed, elapsed / max(total, 1),
    )
    logger.info("=" * 66)
    logger.info("  Avg F1               : %.4f", metrics.avg_f1)
    logger.info("  Avg Precision        : %.4f", metrics.avg_precision)
    logger.info("  Avg Recall           : %.4f", metrics.avg_recall)
    logger.info(
        "  Avg Judge Score      : %.4f  [0=WRONG, 1=CORRECT]",
        metrics.avg_judge_score,
    )
    logger.info("  Avg Retrieved Tokens : %.1f", metrics.avg_retrieved_tokens)
    logger.info("  Avg Compression      : %.1fx", metrics.avg_compression_ratio)
    for k in sorted(metrics.retrieval):
        km = metrics.retrieval[k]
        logger.info(
            "  Retrieval@%-2d         : recall=%.3f  prec=%.3f  f1=%.3f  acc=%.3f",
            k, km.recall, km.precision, km.f1, km.accuracy,
        )
    logger.info("  Per-category breakdown:")
    for cat in sorted(metrics.category_judge):
        logger.info(
            "    Cat %d (n=%2d) : judge=%.3f  f1=%.3f",
            cat,
            metrics.category_count.get(cat, 0),
            metrics.category_judge.get(cat, 0.0),
            metrics.category_f1.get(cat, 0.0),
        )

    # ── 保存产物 ─────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(_RESULTS_DIR / "answers.json", "w", encoding="utf-8") as f:
        json.dump(qa_records, f, ensure_ascii=False, indent=2)
    logger.info("答案  → %s", _RESULTS_DIR / "answers.json")

    metrics_dict = metrics.to_dict()
    metrics_dict["retrieval"] = {
        str(k): {"k": v.k, "recall": v.recall, "precision": v.precision,
                 "f1": v.f1, "accuracy": v.accuracy}
        for k, v in metrics.retrieval.items()
    }
    with open(_RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    logger.info("指标  → %s", _RESULTS_DIR / "metrics.json")
    logger.info("步骤  → %s", _RESULTS_DIR / "steps.jsonl")
    logger.info("日志  → %s", _RESULTS_DIR / "test.log")
    logger.info("=" * 66)

    return metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mem0 LoCoMo mini test pipeline",
    )
    p.add_argument(
        "--rebuild-memory",
        action="store_true",
        help="强制重建 Mem0 记忆（即使已有 .built 标记）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="跳过所有 LLM 调用，仅验证基础设施（检索路径、mem0 初始化）",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_test(
        rebuild_memory=args.rebuild_memory,
        dry_run=args.dry_run,
    )
