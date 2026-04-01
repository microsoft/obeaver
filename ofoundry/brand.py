"""Brand palette and Rich styles for ofoundry terminal output."""

from __future__ import annotations


BARK = "#1A0F05"
BROWN = "#6B3D1A"
AMBER = "#E8A828"
BIRCH = "#FDF5E6"
RIVER = "#2DD4A0"
BELLY = "#C8A474"
OUTLINE = "#2A1205"
TAIL = "#4A2810"

RULE_STYLE = AMBER
TITLE_STYLE = f"bold {AMBER}"
SUBTITLE_STYLE = f"italic {BELLY}"
LABEL_STYLE = f"bold {BIRCH}"
TEXT_STYLE = BIRCH
VALUE_STYLE = RIVER
PATH_STYLE = BELLY
COMMAND_STYLE = f"bold {AMBER}"
ARG_STYLE = f"bold {RIVER}"
MUTED_STYLE = f"dim {BELLY}"
SUCCESS_STYLE = f"bold {RIVER}"
WARNING_STYLE = f"bold {BARK} on {AMBER}"
ERROR_STYLE = f"bold {BIRCH} on {BROWN}"
PROMPT_USER_STYLE = f"bold {RIVER}"
PROMPT_ASSISTANT_STYLE = f"bold {AMBER}"
TABLE_HEADER_STYLE = f"bold {AMBER}"
TABLE_MODEL_STYLE = RIVER
TABLE_ENGINE_STYLE = BELLY
TABLE_TYPE_STYLE = BIRCH
URL_STYLE = f"underline {RIVER}"


def markup(text: object, style: str) -> str:
    return f"[{style}]{text}[/]"