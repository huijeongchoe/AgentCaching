# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
import re
import urllib.request

from ddgs import DDGS
from langchain_core.tools import tool

from logger import log_tool_call

_current_agent_id: str = ""


def _set_current_agent(agent_id: str) -> None:
    global _current_agent_id
    _current_agent_id = agent_id


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        result = f"Search error: {e}"
        log_tool_call(_current_agent_id, "web_search", {"query": query}, result)
        return result

    if not results:
        output = "No results found."
    else:
        lines = []
        for r in results:
            lines.append(
                f"- {r['title']}\n  URL: {r['href']}\n  {r['body'][:200]}"
            )
        output = "\n\n".join(lines)

    log_tool_call(_current_agent_id, "web_search", {"query": query}, output[:500])
    return output


@tool
def fetch_url(url: str) -> str:
    """Fetch and extract text content from a URL."""    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        result = f"Fetch error: {e}"
        log_tool_call(_current_agent_id, "fetch_url", {"url": url}, result)
        return result

    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:3000]

    log_tool_call(_current_agent_id, "fetch_url", {"url": url}, text[:200])
    return text


TOOLS = [web_search, fetch_url]