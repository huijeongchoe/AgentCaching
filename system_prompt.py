# ---------------------------------------------------------------------------
# Prompts for the research agents
# ---------------------------------------------------------------------------


def build_system_prompt(agent_id: str, task_descriptions: str) -> str:
    """Build the system message for a single research agent."""
    return (
        f"You are research agent '{agent_id}'. "
        "You must research the following DeepResearch Bench queries "
        "by searching the web and reading relevant pages.\n\n"
        f"Your assigned tasks:\n{task_descriptions}\n\n"
        "Use web_search to find relevant pages, then fetch_url to read them. "
        "Be thorough — make multiple searches with different query phrasings "
        "to cover each topic comprehensively. "
        "When you have gathered enough information, respond with a summary "
        "of your findings (no more tool calls)."
    )


# Initial message that kicks off the workflow.
INITIAL_USER_MESSAGE = (
    "Research the assigned DeepResearch Bench queries. "
    "Use web_search and fetch_url to find and read relevant sources."
)
