"""Prompt validation patterns for filtering meta-descriptions vs direct prompts."""

import logging

logger = logging.getLogger(__name__)

# Patterns that indicate meta-description (not a direct prompt)
BAD_PATTERNS = [
    "create a script",
    "create a prompt",
    "develop a",
    "develop an",
    "consider a scenario",
    "as a thought experiment",
    "here's a short story about",
    "suppose we create",
    "design a program",
    "construct a",
    "here's an example of",
    "write a script that",
    "craft a prompt"
]

# Patterns that indicate a valid direct prompt
GOOD_PATTERNS = [
    # Core instruction patterns
    "you are", "ignore", "write detailed", "explain how",
    "provide instructions", "step-by-step", "i need help",
    "tell me", "show me", "for my", "complete this",
    "list the", "describe the process", "instructions:",
    "detailed instructions",
    # Imperative patterns
    "decode", "analyze", "generate", "execute", "command:",
    "here's the", "here is the", "task:", "part b:", "part c:",
    "write a", "provide", "detail", "outline", "list", "steps to", "step 1",
    # Reasoning model patterns
    "step 1:", "premise", "reasoning:", "conclude", "therefore",
    "semantic", "traverse", "node", "expand", "query",
    "unscramble", "puzzle", "solve", "rearrange",
    "mapping", "apply:", "transform", "configuration",
    "through verse", "in metaphor", "poetic", "compose",
    "knowledge graph", "full_expansion", "depth",
    # Framing patterns
    "security researcher", "authorized", "educational", "research",
    "hypothetical", "scenario", "role", "system", "graph query", "amr",
    "respond to", "answer", "assist", "help with",
    "imagine", "pretend", "act as", "suppose",
]


def is_valid_jailbreak_prompt(text: str, min_length: int = 20, long_threshold: int = 200) -> bool:
    """
    Validate that text is a direct prompt, not a meta-description.

    Args:
        text: Text to validate
        min_length: Minimum length for valid prompts
        long_threshold: Length above which prompts are accepted even without patterns

    Returns:
        True if text is a valid direct prompt
    """
    text_lower = text.lower().strip()

    if len(text_lower) < min_length:
        return False

    # Check bad patterns at start or early in text
    for pattern in BAD_PATTERNS:
        if text_lower.startswith(pattern):
            logger.warning(f"Rejected prompt starting with meta-description: '{pattern}'")
            return False

    first_part = text_lower[:50]
    for pattern in BAD_PATTERNS:
        if pattern in first_part:
            logger.warning(f"Rejected prompt with early meta-description: '{pattern}'")
            return False

    # Check for good patterns
    has_good_pattern = any(pattern in text_lower for pattern in GOOD_PATTERNS)
    is_long_enough = len(text_lower) > long_threshold

    if not has_good_pattern and not is_long_enough:
        logger.warning(f"Prompt lacks direct instruction patterns and is too short ({len(text_lower)} chars)")
        return False

    return True
