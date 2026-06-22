from pathlib import Path
from datetime import datetime, timezone
import json
import time
from typing import Any


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"deepresearch_dup_{ts}"


RUN_ID = _new_run_id()
LOG_FILE = LOG_DIR / f"{RUN_ID}.jsonl"


def log_tool_call(
    agent_id: str,
    tool_name: str,
    args: dict[str, Any],
    result: str,
) -> None:
    record = {
        "run_id": RUN_ID,
        "agent_id": agent_id,
        "tool_name": tool_name,
        "args": args,
        "result": result,
        "timestamp": time.time(),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")