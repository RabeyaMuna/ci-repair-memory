# baselines/utilities/chunking_logic.py

from __future__ import annotations

from typing import List, Tuple
import tiktoken
from tiktoken import encoding_for_model


def _get_encoder_from_model(model: str):
    """
    Return a tiktoken encoder:
    - If the exact model is known to tiktoken, use it.
    - Otherwise fall back to cl100k_base (good generic BPE for many OpenAI-like APIs).
    """
    name = (model or "").strip().lower()

    # OpenAI GPT-family models: use exact encoding if possible
    if name.startswith("gpt") or "gpt-" in name:
        try:
            return encoding_for_model(model)
        except Exception:
            pass

    # DeepSeek/Claude/unknown: generic BPE fallback
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count tokens using a safe encoder. If model is unknown, uses cl100k_base.
    """
    
    enc = _get_encoder_from_model(model)
    return len(enc.encode(text or "", disallowed_special=()))


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Backward-compatible alias for token counting. Uses safe fallback for unknown models.
    """
    return count_tokens(text, model=model)


def chunk_log_by_tokens(
    log_text: str,
    max_tokens: int = 60_000,
    overlap: int = 100,
    model: str = "cl100k_base",
) -> List[str]:
    """
    Chunk text by *token budget* while preserving line boundaries.

    - max_tokens: token target per chunk (approx).
    - overlap: number of *lines* to overlap between consecutive chunks.
      (Yes: overlap is line-based, not token-based.)
    - model: used only for token estimation; unknown models fall back to cl100k_base.

    Notes:
    - If a single line exceeds max_tokens by itself, it will be placed in its own chunk.
    - Overlap is applied after a chunk flush.
    """
    enc = _get_encoder_from_model(model)
    lines = (log_text or "").splitlines()

    chunks: List[str] = []
    current_lines: List[str] = []
    current_tokens = 0

    for line in lines:
        # token count per line (including a newline approximation to reduce overflow risk)
        line_tokens = len(enc.encode(line, disallowed_special=())) + 1

        # If adding this line exceeds the budget, flush current chunk first
        if current_lines and (current_tokens + line_tokens > max_tokens):
            chunks.append("\n".join(current_lines))

            # Apply overlap in *lines*
            if overlap > 0:
                overlap_lines = (
                    current_lines[-overlap:] if len(current_lines) >= overlap else current_lines
                )
                current_lines = list(overlap_lines)
                current_tokens = sum((len(enc.encode(l, disallowed_special=())) + 1) for l in current_lines)
            else:
                current_lines = []
                current_tokens = 0

        # If the line itself is huge and chunk is empty, still add it (single-line chunk)
        current_lines.append(line)
        current_tokens += line_tokens

        # Optional safety: if a single line made the chunk blow up massively, flush immediately
        if current_tokens > max_tokens and len(current_lines) == 1:
            chunks.append(current_lines[0])
            current_lines = []
            current_tokens = 0

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def chunk_lines_with_overlap(
    content: str,
    lines_per_chunk: int = 250,
    overlap: int = 50,
) -> List[Tuple[int, int, str]]:
    """
    Chunk by line counts (not tokens). Returns:
      [(start_line_1_based, end_line_1_based, chunk_text), ...]
    with 'overlap' lines shared between neighboring chunks.
    """
    lines = (content or "").splitlines()
    total = len(lines)
    chunks: List[Tuple[int, int, str]] = []

    i = 0
    while i < total:
        start = max(i - overlap, 0)
        end = min(i + lines_per_chunk, total)
        chunk = "\n".join(lines[start:end])
        chunks.append((start + 1, end, chunk))  # 1-based line numbers
        i += lines_per_chunk

    return chunks
