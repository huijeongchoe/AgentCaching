import json
from pathlib import Path


DATA_DIR = Path("/NHNHOME/share/huijeongchoe/deep_research_bench")
QUERY_PATH = DATA_DIR / "query.jsonl"


def load_tasks(
    lang: str | None = None,
    topics: list[str] | None = None,
) -> list[dict]:
    """Load DeepResearch Bench tasks from /NHNHOME/share/huijeongchoe/deep_research_bench.

    Each task has: id, topic, language, prompt.
    Filter by language ('en'/'zh') and/or topic list if provided.
    """
    print("Loading DeepResearch Bench tasks...")
    raw = QUERY_PATH.read_text(encoding="utf-8")

    tasks = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        tasks.append(json.loads(line))

    # Apply filters
    if lang:
        tasks = [t for t in tasks if t.get("language") == lang]
    if topics:
        tasks = [t for t in tasks if t.get("topic") in topics]

    print(f"  Loaded {len(tasks)} tasks"
          + (f" (lang={lang})" if lang else "")
          + (f" topics={topics}" if topics else ""))
    return tasks