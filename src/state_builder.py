"""State builder: Jigsaw row -> sanitized, pruned juror state (D1 contract, Section 4).

Contract:
    in : {"id", "comment_text", "toxic"}   <- the label NEVER enters state (leakage)
    out: {"content", "char_len", "truncated"}

Order matters: sanitize first, estimate ONCE on the sanitized text, then
warn (>320) and/or prune (>300) from that same pre-truncation estimate.
"""

import re
import warnings

TOKENS_PER_WORD = 1.3   # hermetic English estimate; no tokenizer load (Section 4)
WARN_TOKENS = 320       # English-root state budget (Section 2b)
PRUNE_TOKENS = 300       # head-truncate target, leaves headroom under the budget

_HTML_TAG = re.compile(r"<[^>]+>")
# Instruction-like substrings that must not reach the model (Section 5, failure mode 6).
_INSTRUCTION_LIKE = re.compile(
    r"(?i)(?:\bignore (?:all |previous |prior )?instructions\b"
    r"|\bsystem\s*:"
    r"|\b(?:disregard|you are now|new instructions?)\b)"
)


def estimate_tokens(text: str) -> int:
    """Hermetic token estimate; exact tokenizer count lives in @pytest.mark.live only."""
    return int(len(text.split()) * TOKENS_PER_WORD)


def sanitize(text: str) -> str:
    """Strip HTML tags and instruction-like substrings from untrusted text."""
    text = _HTML_TAG.sub(" ", text)
    text = _INSTRUCTION_LIKE.sub(" ", text)
    return " ".join(text.split())          # collapse whitespace


def prune(content: str, est: int) -> tuple[str, bool]:
    """Head-truncate on word boundaries when the pre-truncation estimate exceeds budget."""
    if est <= PRUNE_TOKENS:
        return content, False
    keep = int(PRUNE_TOKENS / TOKENS_PER_WORD)      # words that fit the prune target
    return " ".join(content.split()[:keep]), True


def build_state(row: dict, *, precompute: bool = True) -> dict:
    """Apply the D1 contract: facts in code, labels out (Sections 4 and 5).

    precompute=False is the M7 builder-off ablation (checklist): ship only the
    sanitized content, no precomputed facts - the ablation must not be hand-edited.
    """
    content = sanitize(row["comment_text"])
    est = estimate_tokens(content)          # ONE estimate, computed pre-truncation
    if est > WARN_TOKENS:
        warnings.warn(
            f"state estimate {est} tokens > budget {WARN_TOKENS}", stacklevel=2
        )
    content, truncated = prune(content, est)   # budget rule applies even in the ablation
    if not precompute:
        return {"content": content}           # builder-off: facts stripped, content kept
    return {"content": content, "char_len": len(content), "truncated": truncated}