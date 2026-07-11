import unittest
from types import SimpleNamespace

from memory.summarizer import LLMSummarizer
from prompts.topic_memory import (
    build_experience_summary_prompt,
    build_segment_summary_prompt,
)


def fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return fake_response(self.responses.pop(0))


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class SummarizerTests(unittest.TestCase):
    def test_build_segment_summary_prompt_contains_segment_and_qas(self):
        prompt = build_segment_summary_prompt(
            {
                "segment_id": "seg_1",
                "topic": "agent记忆管理",
                "core_entity": "hesm",
                "intent": "设计segment",
                "summary": "",
            },
            [
                {
                    "qa_id": "qa_1",
                    "user_input": "segment 是否必须",
                    "assistant_output": "需要看 goal 长度",
                }
            ],
        )
        self.assertIn("当前 Segment", prompt)
        self.assertIn("segment 是否必须", prompt)
        self.assertIn("agent记忆管理", prompt)

    def test_build_experience_summary_prompt_contains_experience_and_segments(self):
        prompt = build_experience_summary_prompt(
            {
                "experience_id": "exp_1",
                "topic": "agent记忆管理",
                "core_entity": "hesm",
                "summary": "",
            },
            [
                {
                    "segment_id": "seg_1",
                    "intent": "设计segment",
                    "summary": "子目标：判断 segment 是否保留",
                }
            ],
        )
        self.assertIn("当前 Experience", prompt)
        self.assertIn("设计segment", prompt)
        self.assertIn("hesm", prompt)

    def test_llm_summarizer_strips_code_fence_for_segment_summary(self):
        client = FakeClient(
            [
                "```text\n子目标：定义 segment 角色\n当前状态：进行中\n已完成：明确 segment 是 subgoal\n待处理：决定是否轻量化\n下一步：接入 LLM 总结\n```"
            ]
        )
        summarizer = LLMSummarizer(client=client, model="fake-model")
        summary = summarizer.summarize_segment(
            {
                "segment_id": "seg_1",
                "topic": "agent记忆管理",
                "core_entity": "hesm",
                "intent": "设计segment",
                "summary": "",
            },
            [
                {
                    "qa_id": "qa_1",
                    "user_input": "segment 是否必须",
                    "assistant_output": "保留轻量层",
                    "reasoning": "test",
                }
            ],
        )
        self.assertTrue(summary.startswith("子目标：定义 segment 角色"))
        prompt = client.chat.completions.calls[0]["messages"][0]["content"]
        self.assertIn("该 Segment 下的 QA 证据", prompt)

    def test_llm_summarizer_accepts_json_wrapped_experience_summary(self):
        client = FakeClient(
            [
                '{"summary": "目标：构建 hesm 的 goal trace\\n总体状态：进行中\\n已完成：明确 experience/segment/qa 分工\\n当前推进：为 summary 接入 LLM\\n待处理：补连续性判断\\n下一步：接入 manager"}'
            ]
        )
        summarizer = LLMSummarizer(client=client, model="fake-model")
        summary = summarizer.summarize_experience(
            {
                "experience_id": "exp_1",
                "topic": "agent记忆管理",
                "core_entity": "hesm",
                "summary": "",
            },
            [
                {
                    "segment_id": "seg_1",
                    "intent": "设计segment",
                    "summary": "子目标：判断 segment 是否保留",
                }
            ],
        )
        self.assertIn("目标：构建 hesm 的 goal trace", summary)
        prompt = client.chat.completions.calls[0]["messages"][0]["content"]
        self.assertIn("该 Experience 下的 Segment 证据", prompt)


if __name__ == "__main__":
    unittest.main()
