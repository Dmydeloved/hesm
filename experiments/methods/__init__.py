from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import MemoryMethod
from .bm25 import BM25Method
from .dense_rag import DenseRAGMethod
from .full_context import FullContextMethod
from .hesm import HESMMethod


def build_method(name: str, output_dir: Path, config: dict[str, Any]) -> MemoryMethod:
    methods = {
        "bm25": BM25Method,
        "dense_rag": DenseRAGMethod,
        "full_context": FullContextMethod,
        "hesm": HESMMethod,
    }
    try:
        method_cls = methods[name]
    except KeyError as error:
        raise ValueError(f"Unknown method: {name}") from error
    return method_cls(output_dir=output_dir, config=config)


__all__ = ["MemoryMethod", "build_method"]

