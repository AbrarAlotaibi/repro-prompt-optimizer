"""
Shared logging helpers for client response logging.
"""

import os
from typing import Optional, List, Dict, Any
from uuid import uuid4
import random


def log_full_responses_enabled() -> bool:
    """Return True when full response logging is explicitly enabled."""
    return os.getenv("REPRO_LOG_FULL_RESPONSES", "false").lower() == "true"

def log_sample_rate() -> float:
    """Return sampling rate for request/response summaries (0.0-1.0)."""
    try:
        return float(os.getenv("REPRO_LOG_SAMPLE_RATE", "1.0"))
    except ValueError:
        return 1.0


def should_log_summary() -> bool:
    """Return True when a summary should be logged based on sampling."""
    rate = log_sample_rate()
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def new_request_id() -> str:
    """Generate a short request id for log correlation."""
    return uuid4().hex[:12]


def summarize_text(text: Optional[str], limit: int = 200) -> str:
    """Summarize text unless full response logging is enabled."""
    if text is None:
        return ""
    if log_full_responses_enabled():
        return text
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated {len(text) - limit} chars)"


def summarize_messages(messages: Optional[List[Dict[str, Any]]], limit: int = 200) -> Optional[List[Dict[str, Any]]]:
    """Summarize chat messages unless full response logging is enabled."""
    if messages is None:
        return None
    if log_full_responses_enabled():
        return messages
    summary = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        summary.append({
            "role": role,
            "content_len": len(content),
            "content_preview": summarize_text(content, limit=limit),
        })
    return summary


def log_response_summary(logger, provider: str, response, text_attr: Optional[str] = None, request_id: Optional[str] = None) -> None:
    """Log a response summary unless full response logging is enabled."""
    if response is None:
        logger.debug("%s response summary: response=None request_id=%s", provider, request_id, extra={"request_id": request_id})
        return
    if log_full_responses_enabled():
        logger.debug("Full %s response: %s request_id=%s", provider, response, request_id, extra={"request_id": request_id})
        return
    if not should_log_summary():
        return
    resp_id = getattr(response, "id", None)
    model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    has_text = None
    if text_attr:
        try:
            has_text = bool(getattr(response, text_attr))
        except Exception:
            has_text = None
    logger.debug(
        "%s response summary: id=%s model=%s usage=%s has_text=%s request_id=%s",
        provider,
        resp_id,
        model,
        usage,
        has_text,
        request_id,
        extra={"request_id": request_id},
    )


def log_request_summary(
    logger,
    provider: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    limit: int = 200,
) -> None:
    """Log a request summary unless full response logging is enabled."""
    if prompt is None and messages is None:
        return
    if log_full_responses_enabled():
        logger.debug("%s request: prompt=%s messages=%s request_id=%s", provider, prompt, messages, request_id, extra={"request_id": request_id})
        return
    if not should_log_summary():
        return
    prompt_len = len(prompt) if prompt is not None else None
    prompt_preview = summarize_text(prompt, limit=limit) if prompt is not None else None
    msg_summary = summarize_messages(messages, limit=limit) if messages is not None else None
    msg_count = len(messages) if messages is not None else None
    logger.debug(
        "%s request summary: prompt_len=%s prompt_preview=%s messages_count=%s messages=%s request_id=%s",
        provider,
        prompt_len,
        prompt_preview,
        msg_count,
        msg_summary,
        request_id,
        extra={"request_id": request_id},
    )
