"""
Mini memory builder — 100 turns，验证 Experience/Segment 聚合效果。

用途：在修改 MemoryManager 路由逻辑后，用少量数据快速验证三层结构
是否从"1 QA : 1 Segment : 1 Experience"改善为有实质聚合。

输出目录：tests/locomo/results/mini_memory/hesm_mini_{timestamp}/
运行方式：
    cd d:/code/hesm
    python -m tests.locomo.build_mini_memory [--turns 100] [--threshold 0.82]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_FILE     = _PROJECT_ROOT / "data" / "locomo10.json"
_RESULTS_DIR   = _PROJECT_ROOT / "tests" / "locomo" / "results" / "mini_memory"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_mini")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_turns(n: int) -> tuple[str, list[dict]]:
    """Load first n turns from conv-26 (conversation index 0)."""
    from experiments.locomo.data.loader import LoCoMoLoader

    conversations = LoCoMoLoader(_DATA_FILE).load(max_conversations=1)
    if not conversations:
        raise RuntimeError(f"No conversations loaded from {_DATA_FILE}")

    conv = conversations[0]
    turns: list[dict] = []
    for turn in conv.all_turns:
        turns.append({
            "dia_id": turn.dia_id,
            "speaker": turn.speaker,
            "text": turn.text,
            "timestamp": turn.timestamp,
            "session_num": turn.session_num,
        })
        if len(turns) >= n:
            break
    return conv.conv_id, turns[:n]


def _print_stats(storage, label: str) -> dict:
    """Print layer stats and return as dict."""
    import sqlite3
    conn = sqlite3.connect(storage.db_path)
    conn.row_factory = sqlite3.Row

    n_qa  = conn.execute("SELECT COUNT(*) FROM qa_memory").fetchone()[0]
    n_seg = conn.execute("SELECT COUNT(*) FROM segment_memory").fetchone()[0]
    n_exp = conn.execute("SELECT COUNT(*) FROM experience_memory").fetchone()[0]

    # QA per segment distribution
    qa_per_seg = [
        r[0] for r in conn.execute(
            "SELECT json_array_length(qa_ids_json) FROM segment_memory"
        ).fetchall()
    ]
    # Segment per experience distribution
    seg_per_exp = [
        r[0] for r in conn.execute(
            "SELECT json_array_length(segment_ids_json) FROM experience_memory"
        ).fetchall()
    ]
    conn.close()

    avg_qa_per_seg  = sum(qa_per_seg)  / len(qa_per_seg)  if qa_per_seg  else 0
    avg_seg_per_exp = sum(seg_per_exp) / len(seg_per_exp) if seg_per_exp else 0

    singleton_segs = sum(1 for x in qa_per_seg  if x <= 1)
    singleton_exps = sum(1 for x in seg_per_exp if x <= 1)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  QA 条数         : {n_qa}")
    print(f"  Segment 数      : {n_seg}")
    print(f"  Experience 数   : {n_exp}")
    print(f"  QA/Segment 比   : {n_qa/n_seg:.1f}  (avg {avg_qa_per_seg:.1f}, 期望 >3)")
    print(f"  Seg/Experience  : {n_seg/n_exp:.1f}  (avg {avg_seg_per_exp:.1f}, 期望 >2)")
    print(f"  单条 Segment    : {singleton_segs}/{n_seg}  (越少越好)")
    print(f"  单条 Experience : {singleton_exps}/{n_exp}  (越少越好)")

    # Show all experiences
    import sqlite3 as _sq
    conn2 = _sq.connect(storage.db_path)
    conn2.row_factory = _sq.Row
    exps = conn2.execute(
        "SELECT experience_id, topic, core_entity, segment_ids_json FROM experience_memory ORDER BY created_at"
    ).fetchall()
    print(f"\n  Experience 列表 (共 {n_exp} 个):")
    for e in exps:
        segs = json.loads(e["segment_ids_json"] or "[]")
        print(f"    {e['experience_id'][-8:]}  topic={e['topic'][:20]:20s}  "
              f"entity={e['core_entity'][:12]:12s}  segs={len(segs)}")
    conn2.close()
    print()

    return {
        "n_qa": n_qa, "n_seg": n_seg, "n_exp": n_exp,
        "qa_seg_ratio": round(n_qa / n_seg, 2) if n_seg else 0,
        "seg_exp_ratio": round(n_seg / n_exp, 2) if n_exp else 0,
        "avg_qa_per_seg": round(avg_qa_per_seg, 2),
        "avg_seg_per_exp": round(avg_seg_per_exp, 2),
        "singleton_segs": singleton_segs,
        "singleton_exps": singleton_exps,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build HESM mini memory (100 turns)")
    parser.add_argument("--turns", type=int, default=100, help="Number of turns to ingest")
    parser.add_argument("--threshold", type=float, default=0.82, help="Experience similarity threshold")
    parser.add_argument("--min-segment-qas", type=int, default=2, help="Min QAs before segment can split")
    parser.add_argument("--clean", action="store_true", help="Delete existing mini memory before building")
    args = parser.parse_args()

    from memory.embedder import BailianEmbedder
    from memory.extractor import TopicExtractor
    from memory.manager import MemoryManager
    from memory.storage import MemoryStorage
    from memory.summarizer import TemplateSummarizer
    from memory.vector_store import ChromaVectorStore

    # Output dir
    out_dir = _RESULTS_DIR / "hesm_mini"
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
        logger.info("已清除旧记忆: %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path     = out_dir / "memory.sqlite3"
    chroma_path = out_dir / "chroma"

    logger.info("初始化组件 threshold=%.2f min_segment_qas=%d", args.threshold, args.min_segment_qas)
    storage      = MemoryStorage(db_path=str(db_path))
    vector_store = ChromaVectorStore(persist_path=str(chroma_path))
    embedder     = BailianEmbedder()
    extractor    = TopicExtractor()

    manager = MemoryManager(
        storage=storage,
        vector_store=vector_store,
        embedder=embedder,
        summarizer=TemplateSummarizer(),   # 不调 LLM，快速验证结构
        segment_summary_qa_threshold=5,
        experience_summary_segment_threshold=5,
        experience_similarity_threshold=args.threshold,
        min_segment_qas=args.min_segment_qas,
    )

    # Load turns
    conv_id, turns = _load_turns(args.turns)
    logger.info("加载 %d turns from %s", len(turns), conv_id)

    # Ingest
    recent: list[str] = []
    ok = 0
    _CTX = 3   # context window

    for i, t in enumerate(turns):
        if not t["text"].strip():
            continue
        user_input = f"[{t['speaker']}]: {t['text']}"
        window_turns = recent[-_CTX:]
        str_list = [json.dumps(turn, ensure_ascii=False) for turn in window_turns]
        context = "\n".join(str_list)

        try:
            extracted = extractor.extract(user_input=user_input, context=context)
            topic_records = extracted if isinstance(extracted, list) else [extracted]
            if not topic_records:
                raise ValueError("TopicExtractor returned no topic records")

            results = []
            for topic_record in topic_records:
                results.append(manager.add_qa(
                    topic_result=topic_record,
                    user_input=user_input,
                    assistant_output="",
                    tools=[{"dia_id": t["dia_id"]}] if t["dia_id"] else [],
                    timestamp=t["timestamp"],
                ))

            ok += 1
            result = results[-1]
            topic_for_log = topic_records[0].get("topic", "")[:18]
            logger.info(
                "[%3d/%d] %s  %-20s  action=%-16s  exp=...%s  seg=...%s",
                i + 1, len(turns),
                t["dia_id"],
                topic_for_log,
                result["action"],
                result["experience_id"][-6:],
                result["segment_id"][-6:],
            )
        except Exception as exc:
            logger.warning("[%3d/%d] %s 失败: %s", i + 1, len(turns), t["dia_id"], exc)
            continue

        temp_turn = {
            "user_input": user_input,
            "topic_extractor": topic_records[0] if len(topic_records) == 1 else topic_records,
        }
        recent.append(temp_turn)

    logger.info("写入完成 %d/%d turns", ok, len(turns))

    # Stats
    stats = _print_stats(storage, f"mini memory  turns={len(turns)}  threshold={args.threshold}")

    # Save stats
    stats_file = out_dir / "stats.json"
    stats["config"] = {
        "turns": len(turns),
        "threshold": args.threshold,
        "min_segment_qas": args.min_segment_qas,
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("统计已保存到 %s", stats_file)

    storage.commit()
    storage.close()


if __name__ == "__main__":
    main()


