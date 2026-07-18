"""
HESM Test Pipeline — QA-only, 挂载已构建的主实验记忆。

流程：
    加载已有记忆 → Topic Extraction → Retrieval → Answer Generation → Evaluate

【设计约束】
- 记忆只读：直接挂载 outputs/locomo/memory/hesm_{conv_id}，不做任何写入/构建。
- 完全隔离：所有日志和产物写到 tests/locomo/results/，不污染主实验目录。
- 流程一致：Topic Extractor → HybridRetriever → LLMAnswerGenerator，
            指标：Token-F1 / LLM-Judge(0/1) / Retrieval@K(1,3,5)。

Usage (from d:/code/hesm):
    # 首次：生成问题集
    python -m tests.locomo.questions_builder

    # 运行测试
    python -m tests.locomo.test_pipeline

    # 基础设施验证（跳过所有 LLM 调用）
    python -m tests.locomo.test_pipeline --dry-run

Outputs (tests/locomo/results/):
    steps.jsonl   — 每道题的 4 个关键节点 JSONL 流
    answers.json  — 所有题目的完整问答记录 + 指标
    metrics.json  — 聚合指标（与主实验相同 schema）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ─── 路径设置 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR    = Path(__file__).parent
_RESULTS_DIR  = _TESTS_DIR / "results"

# 硬编码：主实验已构建的记忆根目录
_MAIN_MEMORY_ROOT = _PROJECT_ROOT / "outputs" / "locomo" / "memory"

# 测试输入
_QUESTIONS_FILE = _TESTS_DIR / "questions.json"

# 测试产物（全部写到 tests/locomo/results/）
_STEPS_FILE   = _RESULTS_DIR / "steps.jsonl"
_ANSWERS_FILE = _RESULTS_DIR / "answers.json"
_METRICS_FILE = _RESULTS_DIR / "metrics.json"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── 日志（输出到 stdout + tests/locomo/results/test.log）────────────────
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
# Step Tracer — 每个关键节点写一条 JSONL
# ─────────────────────────────────────────────────────────────────────────────

class StepTracer:
    """
    向 JSONL 文件写入结构化的流水线步骤记录。

    每条记录格式：
      {"q_index": int, "step": str, "ts": float, "data": dict}

    step 取值：
      "topic_extraction" | "retrieval" | "answer" | "evaluation"
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        logger.info("步骤追踪 → %s", path)

    def log(self, q_index: int, step: str, data: dict[str, Any]) -> None:
        entry = {
            "q_index": q_index,
            "step": step,
            "ts": time.time(),
            "data": data,
        }
        self._f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# 附加到已有记忆（只读挂载）
# ─────────────────────────────────────────────────────────────────────────────

def _attach_hesm_memory(conv_id: str, hesm_cfg: dict[str, Any]) -> Any:
    """
    打开主实验已构建的 HESM 存储，返回配置好 _retriever/_extractor 的 HESMMemory。

    存储路径（硬编码）：
        {_MAIN_MEMORY_ROOT}/hesm_{conv_id}/memory.sqlite3
        {_MAIN_MEMORY_ROOT}/hesm_{conv_id}/chroma/
    """
    from experiments.locomo.methods.hesm_adapter import HESMMemory

    memory_dir = _MAIN_MEMORY_ROOT / f"hesm_{conv_id}"
    db_path    = memory_dir / "memory.sqlite3"
    chroma_dir = memory_dir / "chroma"

    if not db_path.exists():
        raise FileNotFoundError(
            f"记忆文件不存在: {db_path}\n"
            f"请先运行主实验 (python -m experiments.locomo.run_main --methods hesm)"
        )

    logger.info("挂载已有记忆: %s", memory_dir)
    logger.info("  SQLite : %s (%.1f KB)", db_path, db_path.stat().st_size / 1024)
    logger.info("  Chroma : %s", chroma_dir)

    # 构造 HESMMemory 对象，然后调用内部方法附加到已有存储
    hesm_memory = HESMMemory(
        memory_root=_MAIN_MEMORY_ROOT,
        hesm_cfg=hesm_cfg,
        use_llm_summarizer=hesm_cfg.get("use_llm_summarizer", True),
        use_llm_reranker=hesm_cfg.get("use_llm_reranker", True),
        use_cache=True,
    )
    # 直接调用 _setup_components 打开已有 SQLite/Chroma，不触发任何构建逻辑
    hesm_memory._conv_id = conv_id
    hesm_memory._setup_components(conv_id)

    # 简单验证：读取 QA 计数
    try:
        qa_count = hesm_memory._storage._conn.execute(
            "SELECT COUNT(*) FROM qa_memory"
        ).fetchone()[0]
        logger.info("  已有 QA 记录: %d 条", qa_count)
    except Exception:
        pass

    return hesm_memory


# ─────────────────────────────────────────────────────────────────────────────
# 单题流水线
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_question(
    q_idx: int,
    total: int,
    item: dict[str, Any],
    conv_token_count: int,
    hesm_memory: Any,
    answer_generator: Any,
    judge: Any,
    tracer: StepTracer,
    dry_run: bool,
) -> dict[str, Any]:
    """
    单道题完整流水线：Topic Extraction → Retrieval → Answer → Evaluate。

    返回与主实验 aggregator 兼容的 QA record dict。
    """
    from experiments.locomo.evaluation.f1 import compute_f1
    from experiments.locomo.evaluation.retrieval_metrics import compute_retrieval_metrics
    # conv_token_count is passed in as a parameter — no need to import count_tokens here

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

    # ── Step 1: Topic Extraction ─────────────────────────────────────────
    t1 = time.time()
    if dry_run:
        q_topic: dict[str, Any] = {
            "topic": "dry_run",
            "core_entity": "dry_run",
            "intent": "dry_run",
            "entities": [],
        }
    else:
        extractor = hesm_memory.get_extractor()
        try:
            raw = extractor.extract(user_input=question)
            q_topic = (raw[0] if isinstance(raw, list) else raw) or {}
        except Exception as exc:
            logger.warning("  [1] Topic Extraction FAILED: %s", exc)
            q_topic = {}

    t1_ms = (time.time() - t1) * 1000
    logger.info(
        "  [1/4] Topic Extraction (%.0fms): topic=%r  entity=%r  intent=%r  entities=%s",
        t1_ms,
        q_topic.get("topic", ""),
        q_topic.get("core_entity", ""),
        q_topic.get("intent", ""),
        q_topic.get("entities", []),
    )
    tracer.log(q_idx, "topic_extraction", {
        "question": question,
        "topic":       q_topic.get("topic", ""),
        "core_entity": q_topic.get("core_entity", ""),
        "intent":      q_topic.get("intent", ""),
        "entities":    q_topic.get("entities", []),
        "latency_ms":  round(t1_ms, 1),
    })

    # ── Step 2: Retrieval ────────────────────────────────────────────────
    t2 = time.time()
    if dry_run:
        from experiments.locomo.methods.base import RetrievalResult
        ret = RetrievalResult("(dry run)", [], 5, {})
    else:
        ret = hesm_memory.retrieve(question=question, top_k=5)

    t2_ms = (time.time() - t2) * 1000
    raw_r  = ret.raw_result or {}
    n_exp  = len(raw_r.get("experiences", []))
    n_seg  = len(raw_r.get("segments", []))
    n_qa   = len(raw_r.get("qas", []))
    debug  = raw_r.get("debug", {})

    logger.info(
        "  [2/4] Retrieval (%.0fms): exp=%d  seg=%d  qa=%d  tokens=%d",
        t2_ms, n_exp, n_seg, n_qa, ret.token_count,
    )
    logger.info("        retrieved_ids : %s", ret.retrieved_ids)
    logger.info(
        "        context preview: %s",
        ret.context_text[:200].replace("\n", " ") if ret.context_text else "(空)",
    )
    # 计算命中情况
    hits = set(ret.retrieved_ids) & set(evidence_ids)
    logger.info(
        "        命中证据: %s / %s", sorted(hits), evidence_ids
    )
    tracer.log(q_idx, "retrieval", {
        "question":       question,
        "retrieved_ids":  ret.retrieved_ids,
        "evidence_ids":   evidence_ids,
        "hits":           sorted(hits),
        "n_experiences":  n_exp,
        "n_segments":     n_seg,
        "n_qas":          n_qa,
        "retrieved_tokens": ret.token_count,
        "context_preview":  ret.context_text[:600],
        "debug":            debug,
        "latency_ms":       round(t2_ms, 1),
    })

    # ── Step 3: Answer Generation ────────────────────────────────────────
    t3 = time.time()
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
            logger.warning("  [3] Answer Generation FAILED: %s", exc)
            prediction = ""

    t3_ms = (time.time() - t3) * 1000
    logger.info(
        "  [3/4] Answer Generation (%.0fms):\n"
        "        预测: %s\n"
        "        标准: %s",
        t3_ms,
        prediction[:150],
        ground_truth[:150],
    )
    tracer.log(q_idx, "answer", {
        "question":       question,
        "ground_truth":   ground_truth,
        "prediction":     prediction,
        "context_tokens": ret.token_count,
        "latency_ms":     round(t3_ms, 1),
    })

    # ── Step 4: Evaluation ───────────────────────────────────────────────
    t4 = time.time()

    # Token-level F1
    f1_res = compute_f1(prediction, ground_truth)

    # LLM Judge: 0=WRONG, 1=CORRECT, -1=failure
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

    # Retrieval@K=1,3,5
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

    # Compression ratio
    compression = (
        conv_token_count / ret.token_count
        if ret.token_count > 0
        else None
    )

    t4_ms = (time.time() - t4) * 1000
    judge_label = {1: "CORRECT ✓", 0: "WRONG ✗", -1: "FAILED"}.get(judge_score, "?")
    logger.info(
        "  [4/4] Evaluation (%.0fms): "
        "F1=%.3f  P=%.3f  R=%.3f  Judge=%s  "
        "Recall@1=%.2f  @3=%.2f  @5=%.2f  compress=%.0fx",
        t4_ms,
        f1_res["f1"], f1_res["precision"], f1_res["recall"],
        judge_label,
        rm_by_k[1].recall, rm_by_k[3].recall, rm_by_k[5].recall,
        compression,
    )
    tracer.log(q_idx, "evaluation", {
        "question":      question,
        "ground_truth":  ground_truth,
        "prediction":    prediction,
        "f1":            f1_res["f1"],
        "f1_precision":  f1_res["precision"],
        "f1_recall":     f1_res["recall"],
        "judge_score":   judge_score,
        "judge_label":   judge_label.split()[0],   # "CORRECT" / "WRONG" / "FAILED"
        "retrieval_metrics": ret_metrics_dict,
        "total_conversation_tokens": conv_token_count,
        "compression_ratio": round(compression, 2) if compression is not None else None,
        "latency_ms":    round(t4_ms, 1),
    })

    # ── 汇总 QA record（与 aggregator 兼容）──────────────────────────────
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
        # 各步骤耗时 (ms)
        "latency_topic_ms":    round(t1_ms, 1),
        "latency_retrieval_ms": round(t2_ms, 1),
        "latency_answer_ms":   round(t3_ms, 1),
        "latency_eval_ms":     round(t4_ms, 1),
        "latency_total_ms":    round(total_ms, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_test(dry_run: bool = False) -> dict[str, Any]:
    """
    运行完整测试流水线并返回聚合指标 dict。
    """
    import yaml
    from experiments.locomo.data.loader import LoCoMoLoader
    from experiments.locomo.evaluation.aggregator import aggregate
    from experiments.locomo.evaluation.judge import LLMJudge
    from experiments.locomo.evaluation.token_metrics import count_tokens
    from experiments.locomo.methods.base import LLMAnswerGenerator

    logger.info("=" * 66)
    logger.info("HESM Test Pipeline  |  记忆只读，QA 阶段独立执行")
    logger.info("=" * 66)

    # ── 加载配置 ────────────────────────────────────────────────────────
    with open(_PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        hesm_cfg_raw: dict[str, Any] = yaml.safe_load(f)

    exp_cfg_path = (
        _PROJECT_ROOT / "experiments" / "locomo" / "config" / "experiment.yaml"
    )
    with open(exp_cfg_path, encoding="utf-8") as f:
        exp_cfg: dict[str, Any] = yaml.safe_load(f)

    hesm_section: dict[str, Any] = exp_cfg.get("hesm", {})

    # ── 加载问题集 ───────────────────────────────────────────────────────
    if not _QUESTIONS_FILE.exists():
        logger.info("questions.json 不存在，先运行 questions_builder ...")
        from tests.locomo.questions_builder import build_questions
        build_questions()

    with open(_QUESTIONS_FILE, encoding="utf-8") as f:
        q_data: dict[str, Any] = json.load(f)

    conv_id   = q_data["conv_id"]
    speaker_a = q_data["speaker_a"]
    speaker_b = q_data["speaker_b"]
    questions = q_data["questions"]

    # 给每道题附加 speaker 信息（answer_generator 需要）
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

    # ── 挂载已有记忆（只读）─────────────────────────────────────────────
    hesm_memory = _attach_hesm_memory(conv_id, hesm_section)

    # ── 计算对话总 token 数（用于压缩比）───────────────────────────────
    dataset_path = _PROJECT_ROOT / exp_cfg["dataset"]["path"]
    loader = LoCoMoLoader(dataset_path)
    conversations = loader.load(max_conversations=1)
    conv = conversations[0]
    conv_token_count = count_tokens(conv.total_turn_text())
    logger.info(
        "对话总 tokens: %d  (conv=%s, %d sessions, %d turns)",
        conv_token_count, conv.conv_id,
        len(conv.sessions), len(conv.all_turns),
    )

    # ── 初始化共享组件 ───────────────────────────────────────────────────
    answer_generator = LLMAnswerGenerator(hesm_cfg_raw)
    judge = LLMJudge(hesm_cfg_raw)

    # ── 打开步骤追踪器 ───────────────────────────────────────────────────
    tracer = StepTracer(_STEPS_FILE)

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
                hesm_memory=hesm_memory,
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
    metrics = aggregate("hesm_test", qa_records)

    # ── 打印汇总 ─────────────────────────────────────────────────────────
    logger.info("=" * 66)
    logger.info("测试完成  %d 题 / %.1fs (avg %.1fs/题)", total, elapsed, elapsed / max(total, 1))
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

    with open(_ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_records, f, ensure_ascii=False, indent=2)
    logger.info("答案  → %s", _ANSWERS_FILE)

    # 序列化 metrics（把 dataclass key 转为 str）
    metrics_dict = metrics.to_dict()
    metrics_dict["retrieval"] = {
        str(k): {"k": v.k, "recall": v.recall, "precision": v.precision,
                 "f1": v.f1, "accuracy": v.accuracy}
        for k, v in metrics.retrieval.items()
    }
    with open(_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    logger.info("指标  → %s", _METRICS_FILE)
    logger.info("步骤  → %s", _STEPS_FILE)
    logger.info("日志  → %s", _RESULTS_DIR / "test.log")
    logger.info("=" * 66)

    return metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HESM LoCoMo mini test pipeline（QA only，记忆只读）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="跳过所有 LLM 调用，仅验证基础设施（挂载记忆、检索路径）",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_test(dry_run=args.dry_run)
