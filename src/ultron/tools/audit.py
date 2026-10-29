from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class AuditLog:
    """Append-only JSONL record of every tool call."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")

    def write(self, **record: Any) -> None:
        record = {"ts": time.time(), **record}
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
