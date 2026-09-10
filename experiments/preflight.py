"""Check native HESM and OmniMemEval model configuration."""
import argparse
import json
import os

from experiments.settings import (
    load_env,
    read_settings,
    validate_omnimemeval_models,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--direct-network", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.direct_network:
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            os.environ.pop(key, None)
    load_env(args.env)
    validate_omnimemeval_models()
    settings = read_settings(args.config)
    native = settings["native_config"]
    configs = {
        "memory": native.get("topic_extraction") or {},
        "embedding": native.get("embedding") or {},
        "ANSWER": {key: os.environ.get(f"ANSWER_{key.upper()}", "") for key in ["model", "api_key", "base_url"]},
        "EVAL": {key: os.environ.get(f"EVAL_{key.upper()}", "") for key in ["model", "api_key", "base_url"]},
    }
    result = {"profile": settings["profile"], "config_fingerprint": settings["config_fingerprint"], "checks": []}
    from openai import OpenAI
    import httpx
    for stage, cfg in configs.items():
        try:
            if not all(str(cfg.get(key) or "").strip() for key in ["model", "api_key", "base_url"]):
                raise ValueError(f"Missing {stage} model/api_key/base_url")
            item = {"stage": stage, "requested_model": cfg["model"], "status": "configured"}
            if args.live:
                client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=45, max_retries=1,
                                http_client=httpx.Client(trust_env=not args.direct_network))
                if stage == "embedding":
                    response = client.embeddings.create(model=cfg["model"], input="HESM evaluation connectivity check")
                    item["dimensions"] = len(response.data[0].embedding)
                else:
                    response = client.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": "Reply with OK."}], max_tokens=8, temperature=0)
                    item["returned_model"] = response.model
                item["status"] = "ok"
            result["checks"].append(item)
        except Exception as exc:
            reason = str(exc)
            api_key = str(cfg.get("api_key") or "")
            if api_key:
                reason = reason.replace(api_key, "<redacted>")
            result["checks"].append({"stage": stage, "status": "failed", "error_type": type(exc).__name__, "error": reason[:300]})
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if any(item["status"] == "failed" for item in result["checks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
