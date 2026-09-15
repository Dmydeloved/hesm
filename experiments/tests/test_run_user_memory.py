from types import SimpleNamespace

from experiments.run_user_memory import BENCHMARKS, stage_commands


def test_all_user_memory_benchmarks_define_a_complete_six_stage_flow():
    for benchmark, definition in BENCHMARKS.items():
        args = SimpleNamespace(
            benchmark=benchmark,
            scale="100k",
            eval_python="python",
            workers=1,
            llm_workers=2,
            top_k=20,
            judge_runs=1,
        )

        commands = stage_commands(args, "flow-check")

        assert len(commands) == 6
        assert [command[1].rsplit("_", 1)[-1] for command in commands] == [
            f"{stage}.py" for stage in definition["stages"]
        ]
        assert commands[0][1].endswith("_ingestion.py")
        assert commands[-1][1].endswith("_report.py")
