"""Strategy family classification utilities."""

from typing import Dict, List


STRATEGY_FAMILIES: Dict[str, List[str]] = {
    "hierarchical": [
        "hierarchical",
        "hierarchical:steps",
        "hierarchical:roleplay",
        "hierarchical:wrapper",
        "hierarchical:questions",
        "hierarchical_roleplay",
    ],
    "graph": ["graph_attack", "graph_attack_semantic", "graph_attack_semantic_universal", "graph_traversal"],
    "persona": ["roleplay", "persona", "character", "developer_mode", "roleplay_persona", "roleplay_persona_universal"],
    "injection": [
        "prefix_injection",
        "prefix_injection_universal",
        "suffix_injection",
        "prompt_injection",
        "skeleton_key",
        "skeleton_key_universal",
        "policy_puppetry",
        "policy_puppetry_universal",
    ],
    "obfuscation": [
        "token_smuggling",
        "base64",
        "cipher",
        "encoding",
        "encrypt",
        "obscure",
        "cipher_encoding",
        "cipher_encoding_universal",
        "ascii_art_attack",
        "flip_attack",
        "code_chameleon",
    ],
    "framing": [
        "academic",
        "research",
        "educational",
        "fictional",
        "hypothetical",
        "story",
        "authority_framing",
        "adversarial_poetry",
        "adversarial_poetry_universal",
        "task_deflection",
        "reverse_psychology",
        "logic_trap",
        "logic_trap_universal",
    ],
    "technical": [
        "technical_wrapper",
        "code_wrapper_universal",
        "json_injection",
        "xml_injection",
        "code",
        "programming",
        "output_format_exploit",
        "output_format_universal",
        "code_context",
    ],
    "word_level": [
        "word_level_combined_mutation",
        "word_level_sentence_reordering",
        "word_level_synonym_substitution",
        "word_level",
        "synonym",
        "reorder",
    ],
    "semantic": [
        "semantic_drift_hybrid",
        "semantic_drift_chained",
        "semantic_drift_persona",
        "semantic_drift",
        "semantic",
        "drift",
        "context",
        "context_dilution",
        "context_dilution_universal",
        "cot_dilution",
    ],
    "inception": [
        "deep_inception",
        "deep_inception_universal",
        "virtualization_attack",
        "virtualization_universal",
        "nested_scenario",
    ],
    "progressive": [
        "crescendo_attack",
        "crescendo_universal",
        "many_shot_icl",
        "many_shot_universal",
        "cognitive_load",
        "cognitive_overload_universal",
    ],
    "temporal": [
        "past_tense_reframe",
        "past_tense_universal",
        "future_tense_speculation",
        "competing_objectives",
    ],
    "language": ["language_pivot", "multilingual_switching"],
    "other": [
        "bad_likert_judge",
        "dataset_fallback",
        "hcot_hijacking",
        "hcot_hijacking_universal",
        "mousetrap_mapping",
        "mousetrap_mapping_universal",
        "puzzled_word_puzzle",
        "refusal_suppression",
        "refusal_suppression_universal",
    ],
}


def get_strategy_family(strategy: str) -> str:
    """Map a strategy name to a family label."""
    if not strategy:
        return "unknown"

    strategy_lower = strategy.lower()

    for family, members in STRATEGY_FAMILIES.items():
        for member in members:
            if member in strategy_lower:
                return family

    if ":" in strategy:
        return strategy.split(":")[0]
    return strategy
