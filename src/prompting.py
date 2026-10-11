"""
Single place for the prompt template so SFT, both DPO runs, and generation
all format instructions the exact same way. If this drifts between stages
the comparisons stop being fair, so everything should import from here
instead of hardcoding the string somewhere else.
"""

from __future__ import annotations

PROMPT_TEMPLATE = "Instruction: {instruction}\nResponse: "


def format_prompt(instruction: str) -> str:
    # trailing space after "Response: " is on purpose, that's where
    # generation picks up from
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())