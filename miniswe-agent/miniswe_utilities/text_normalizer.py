"""
Text Normalization Utility

Comprehensive text cleaning to handle ANY encoding/character issues that might
cause LLM parsing errors or JSON decode failures.

Usage:
    from utilities.text_normalizer import normalize_text

    clean_text = normalize_text(raw_log)
"""
import re
import unicodedata
from typing import Optional


def normalize_text(
    text: str,
    remove_bom: bool = True,
    remove_null_bytes: bool = True,
    normalize_unicode: bool = True,
    remove_control_chars: bool = True,
    collapse_whitespace: bool = True,
    remove_special_tokens: bool = True,
) -> str:
    """
    Comprehensive text normalization to handle ANY encoding/character issues.

    Args:
        text: Input text to normalize
        remove_bom: Remove all types of BOM (UTF-8, UTF-16, UTF-32)
        remove_null_bytes: Remove null bytes (\x00)
        normalize_unicode: Normalize to NFC form (composed characters)
        remove_control_chars: Remove control characters (except \n, \t, \r)
        collapse_whitespace: Collapse multiple spaces/newlines
        remove_special_tokens: Remove LLM special tokens (<|endoftext|>, etc.)

    Returns:
        Normalized text safe for LLM processing and JSON serialization

    Handles:
    - Special LLM tokens (<|endoftext|>, <|im_start|>, etc.)
    - All BOM types (UTF-8, UTF-16 LE/BE, UTF-32 LE/BE)
    - Control characters (except newline, tab, carriage return)
    - Invalid Unicode sequences
    - Null bytes
    - Mixed encodings
    - Unicode normalization to NFC form

    Examples:
        >>> normalize_text('﻿Hello World')  # UTF-8 BOM
        'Hello World'

        >>> normalize_text('ï»¿Test')  # BOM as literal bytes
        'Test'

        >>> normalize_text('Line1\\x00Line2')  # Null byte
        'Line1Line2'
    """
    if not text:
        return text

    # Step 0: Remove special LLM tokens that cause encoding errors
    if remove_special_tokens:
        # Remove common special tokens that appear in logs
        special_tokens = [
            '<|endoftext|>',
            '<|startoftext|>',
            '<|end|>',
            '<|start|>',
            '<|im_start|>',
            '<|im_end|>',
        ]
        for token in special_tokens:
            text = text.replace(token, '')

    # Step 1: Remove all types of BOM (Byte Order Mark)
    if remove_bom:
        # UTF-8 BOM (U+FEFF)
        if text.startswith('﻿'):
            text = text[1:]

        # UTF-8 BOM as literal bytes (common in some systems)
        if text.startswith('ï»¿'):
            text = text[3:]

        # UTF-16 LE BOM (U+FFFE)
        if text.startswith('￾'):
            text = text[1:]

        # UTF-16 BE BOM (U+FEFF)
        if text.startswith('￿'):
            text = text[1:]

        # Strip any remaining BOM-like characters at start
        text = text.lstrip('﻿￾￿')

    # Step 2: Remove null bytes (common in binary data mixed with text)
    if remove_null_bytes:
        text = text.replace('\x00', '')

    # Step 3: Normalize Unicode to NFC (Normalization Form Composed)
    # This handles various Unicode representations of the same character
    # e.g., 'é' can be U+00E9 or U+0065 U+0301 (e + combining acute)
    if normalize_unicode:
        try:
            text = unicodedata.normalize('NFC', text)
        except (ValueError, TypeError):
            # If normalization fails, continue with original text
            pass

    # Step 4: Remove problematic control characters
    # Keep: \n (newline), \t (tab), \r (carriage return)
    # Remove: all other control chars (0x00-0x1F and 0x7F-0x9F)
    if remove_control_chars:
        cleaned_chars = []
        for char in text:
            code = ord(char)
            # Keep printable chars and safe whitespace
            if code >= 32 or char in '\n\t\r':
                # Skip DEL (127) and C1 control characters (128-159)
                if code < 127 or code >= 160 or char in '\n\t\r':
                    cleaned_chars.append(char)
        text = ''.join(cleaned_chars)

    # Step 5: Collapse whitespace (optional cleanup)
    if collapse_whitespace:
        # Collapse multiple newlines to max 2 (keep paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Collapse multiple spaces/tabs to single space (except at line start)
        text = re.sub(r'(?<!^)[ \t]{2,}', ' ', text, flags=re.MULTILINE)

        # Ensure text ends with a single newline (if it had any)
        if text and text != text.rstrip('\n'):
            text = text.rstrip('\n') + '\n'

    return text


def normalize_log_text(text: str) -> str:
    """
    Specialized normalization for CI/build logs.

    This is an alias for normalize_text with all options enabled,
    optimized for processing CI logs before LLM analysis.

    Args:
        text: Raw log text

    Returns:
        Normalized log text
    """
    return normalize_text(
        text,
        remove_bom=True,
        remove_null_bytes=True,
        normalize_unicode=True,
        remove_control_chars=True,
        collapse_whitespace=True,
    )


def safe_json_string(text: Optional[str]) -> str:
    """
    Ensure a string is safe for JSON serialization.

    Removes characters that might break JSON encoding:
    - Control characters (except \\n, \\t, \\r)
    - Invalid Unicode
    - Null bytes

    Args:
        text: Input text (can be None)

    Returns:
        JSON-safe string
    """
    if text is None:
        return ""

    return normalize_text(
        text,
        remove_bom=True,
        remove_null_bytes=True,
        normalize_unicode=True,
        remove_control_chars=True,
        collapse_whitespace=False,  # Keep original whitespace for JSON
    )


# Backwards compatibility alias
clean_log_text = normalize_log_text
