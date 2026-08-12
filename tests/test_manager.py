import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hesm.embedder import HashingEmbedder
from hesm.manager import MemoryManager
from hesm.storage import MemoryStorage
from hesm.summarizer import LLMSummarizer, TemplateSummarizer
from hesm.vector_store import ChromaVectorStore


def build_cases() -> list[dict]:
    return [
        {
            "topic_result": {
                "topic": "agent记忆管理",
                "core_entity": "hesm记忆系统",
                "intent": "定义goal结构",
                "entities": ["hesm", "goal", "experience", "segment", "qa"],
                "confidence": 0.96,
                "reasoning": "The user is defining the core memory goal structure.",
            },
            "user_input": "我想把 qa、segment、experience 组织成一条 goal trace。",
            "assistant_output": "可以把 experience 作为 goal，segment 作为阶段，qa 作为原子证据。",
        },
        {
            "topic_result": {
                "topic": "agent记忆管理",
                "core_entity": "hesm记忆系统",
                "intent": "定义goal结构",
                "entities": ["experience", "goal", "summary"],
                "confidence": 0.95,
                "reasoning": "The user is refining how experience maps to goal and summary.",
            },
            "user_input": "experience 更像最终 goal，summary 需要体现已经完成了哪些任务。",
            "assistant_output": "可以把 experience summary 聚焦为 goal 当前完成度和剩余事项。",
        },
        {
            "topic_result": {
                "topic": "agent记忆管理",
                "core_entity": "hesm记忆系统",
                "intent": "定义goal结构",
                "entities": ["segment", "intent", "subgoal"],
                "confidence": 0.94,
                "reasoning": "The user is clarifying that segment should map to intent/subgoal.",
            },
            "user_input": "segment 我希望表示 goal 下的一段 intent，也就是一个 subgoal。",
            "assistant_output": "那 segment summary 就应围绕 intent 对应的阶段任务来写。",
        },
        {
            "topic_result": {
                "topic": "agent记忆管理",
                "core_entity": "hesm记忆系统",
                "intent": "设计summary机制",
                "entities": ["summary", "prompt", "llm"],
                "confidence": 0.93,
                "reasoning": "The user moved from structure definition into summary mechanism design.",
            },
            "user_input": "我不想大改结构，只想在 summary 阶段利用 LLM 做总结。",
            "assistant_output": "可以保留现有 schema，只新增 LLMSummarizer 和对应 prompt。",
        },
        {
            "topic_result": {
                "topic": "agent记忆管理",
                "core_entity": "hesm记忆系统",
                "intent": "设计summary机制",
                "entities": ["topic", "core_entity", "intent", "summary"],
                "confidence": 0.92,
                "reasoning": "The user is constraining the summary logic to follow current system semantics.",
            },
            "user_input": "summary 里要严格体现 topic+core_entity 表示 goal，segment 用 intent 表示。",
            "assistant_output": "那 prompt 里就要显式约束 goal 身份和 intent 对齐关系。",
        },
        {
            "topic_result": {
                "topic": "配置管理",
                "core_entity": "summarization配置",
                "intent": "配置summary参数",
                "entities": ["config", "summarization", "api_key", "model"],
                "confidence": 0.91,
                "reasoning": "The user switched to a different goal about configuration management.",
            },
            "user_input": "summary 的 api 和 model 等参数单独放到 config 里，不要复用别的配置。",
            "assistant_output": "可以新增 summarization 配置段，并让 LLMSummarizer 只读取这一段。",
        },
    ]


def build_summarizer():
    try:
        return LLMSummarizer()
    except ValueError as error:
        print("LLMSummarizer unavailable, fallback to TemplateSummarizer:")
        print(f"  {error}")
        return TemplateSummarizer()


def print_memory_report(storage: MemoryStorage, experience_ids: list[str], results: list[dict]) -> None:
    print("memory build results:")
    print(json.dumps(results, ensure_ascii=False, indent=4))

    unique_experience_ids = list(dict.fromkeys(experience_ids))
    experiences = storage.get_experiences(unique_experience_ids)

    print("experience summaries:")
    for experience in experiences:
        print(
            json.dumps(
                {
                    "experience_id": experience["experience_id"],
                    "topic": experience["topic"],
                    "core_entity": experience["core_entity"],
                    "intents_link": experience["intents_link"],
                    "segment_ids": experience["segment_ids"],
                    "summary": experience["summary"],
                },
                ensure_ascii=False,
                indent=4,
            )
        )
        segments = storage.get_segments(experience["segment_ids"])
        for segment in segments:
            qas = storage.get_qas(segment["qa_ids"])
            print(
                json.dumps(
                    {
                        "segment_id": segment["segment_id"],
                        "experience_id": segment["experience_id"],
                        "intent": segment["intent"],
                        "qa_count": len(segment["qa_ids"]),
                        "summary": segment["summary"],
                        "qa_inputs": [qa["user_input"] for qa in qas],
                    },
                    ensure_ascii=False,
                    indent=4,
                )
            )

    print("row counts:")
    print(
        json.dumps(
            {
                "qa_memory": storage.count_rows("qa_memory"),
                "segment_memory": storage.count_rows("segment_memory"),
                "experience_memory": storage.count_rows("experience_memory"),
                "runtime_state": storage.count_rows("runtime_state"),
            },
            ensure_ascii=False,
            indent=4,
        )
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        storage = MemoryStorage(Path(directory) / "topic_memory.sqlite3")
        vector_store = ChromaVectorStore(ephemeral=True)
        manager = MemoryManager(
            storage=storage,
            vector_store=vector_store,
            embedder=HashingEmbedder(),
            summarizer=build_summarizer(),
            segment_summary_qa_threshold=2,
            experience_summary_segment_threshold=2,
        )

        state_key = "test_manager"
        results = []
        experience_ids = []
        for case in build_cases():
            result = manager.add_qa(
                topic_result=case["topic_result"],
                user_input=case["user_input"],
                assistant_output=case["assistant_output"],
                state_key=state_key,
            )
            results.append(result)
            experience_ids.append(result["experience_id"])

        print_memory_report(storage, experience_ids, results)
        print(f"vector_memory: {vector_store.count()}")
        storage.close()
