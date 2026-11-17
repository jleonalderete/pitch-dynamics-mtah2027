"""Refusal to run steps that touch the challenge match or reserved data inside an AI agent."""

import os


def refuse_agent() -> None:
    """Exit when running inside an AI coding agent (the CLAUDECODE variable is set)."""
    if os.environ.get("CLAUDECODE"):
        raise SystemExit(
            "This step reads the challenge match or a reserved holdout, which must never reach "
            "an AI service: run it in your own terminal, outside the agent."
        )
