"""
Operational error handling & text sanitization tools.
"""
from __future__ import annotations

class ScanBusyError(Exception):
    """Raised when another scan is already in progress."""
    pass

def safe_error_text(exc: Exception | str, max_len: int = 300, mask_tokens: tuple[str, ...] = ()) -> str:
    msg = str(exc)
    for token in mask_tokens:
        if token and isinstance(token, str):
            msg = msg.replace(token, "*****")
    return msg[:max_len]