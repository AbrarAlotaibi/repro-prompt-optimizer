"""Strategy name normalization and canonical mapping."""

import re
import logging

logger = logging.getLogger(__name__)

# Canonical strategy name mapping - maps variations to single canonical form
STRATEGY_CANONICAL_MAP = {
    # Word-level mutations
    "word level combined mutation": "word_level_combined_mutation",
    "word_level_combined_mutation": "word_level_combined_mutation",
    "wordlevelcombinedmutation": "word_level_combined_mutation",
    "word level synonym substitution": "word_level_synonym_substitution",
    "word_level_synonym_substitution": "word_level_synonym_substitution",
    "wordlevelsynonymsubstitution": "word_level_synonym_substitution",
    "word level sentence reordering": "word_level_sentence_reordering",
    "word_level_sentence_reordering": "word_level_sentence_reordering",
    "wordlevelsentencereordering": "word_level_sentence_reordering",
    # Semantic drift
    "semantic drift visible": "semantic_drift_visible",
    "semantic_drift_visible": "semantic_drift_visible",
    "semanticdriftvisible": "semantic_drift_visible",
    "semantic drift invisible": "semantic_drift_invisible",
    "semantic_drift_invisible": "semantic_drift_invisible",
    "semanticdriftinvisible": "semantic_drift_invisible",
    "semantic drift hybrid": "semantic_drift_hybrid",
    "semantic_drift_hybrid": "semantic_drift_hybrid",
    "semanticdrifthybrid": "semantic_drift_hybrid",
    # Common strategies
    "cipher encoding": "cipher_encoding",
    "cipher_encoding": "cipher_encoding",
    "cipherencoding": "cipher_encoding",
    "refusal suppression": "refusal_suppression",
    "refusal_suppression": "refusal_suppression",
    "refusalsuppression": "refusal_suppression",
    "logic trap": "logic_trap",
    "logic_trap": "logic_trap",
    "logictrap": "logic_trap",
    "technical wrapper": "technical_wrapper",
    "technical_wrapper": "technical_wrapper",
    "technicalwrapper": "technical_wrapper",
    "nested scenario": "nested_scenario",
    "nested_scenario": "nested_scenario",
    "nestedscenario": "nested_scenario",
    "prefix technical combo": "prefix_technical_combo",
    "prefix_technical_combo": "prefix_technical_combo",
    "prefixtechnicalcombo": "prefix_technical_combo",
    "competing objectives": "competing_objectives",
    "competing_objectives": "competing_objectives",
    "competingobjectives": "competing_objectives",
    "competitive objectives": "competing_objectives",
    "competitive_objectives": "competing_objectives",
    "code context": "code_context",
    "code_context": "code_context",
    "codecontext": "code_context",
    "self crossover": "self_crossover",
    "self_crossover": "self_crossover",
    "selfcrossover": "self_crossover",
    "reflective crossover": "reflective_crossover",
    "reflective_crossover": "reflective_crossover",
    "reflectivecrossover": "reflective_crossover",
    "hill reframing": "hill_reframing",
    "hill_reframing": "hill_reframing",
    "hillreframing": "hill_reframing",
    "hcot hijacking": "hcot_hijacking",
    "hcot_hijacking": "hcot_hijacking",
    "h cot hijacking": "hcot_hijacking",
    "h-cot hijacking": "hcot_hijacking",
    "puzzled word puzzle": "puzzled_word_puzzle",
    "puzzled_word_puzzle": "puzzled_word_puzzle",
    "puzzledwordpuzzle": "puzzled_word_puzzle",
    "mousetrap mapping": "mousetrap_mapping",
    "mousetrap_mapping": "mousetrap_mapping",
    "mousetrapmapping": "mousetrap_mapping",
    "past tense reframe": "past_tense_reframe",
    "past_tense_reframe": "past_tense_reframe",
    "pasttensereframe": "past_tense_reframe",
    "word insertion": "word_insertion",
    "word_insertion": "word_insertion",
    "wordinsertion": "word_insertion",
    "word deletion": "word_deletion",
    "word_deletion": "word_deletion",
    "worddeletion": "word_deletion",
    "synonym substitution": "synonym_substitution",
    "synonym_substitution": "synonym_substitution",
    "synonymsubstitution": "synonym_substitution",
    # 2025 New strategies (for stubborn tasks)
    "deep inception": "deep_inception",
    "deep_inception": "deep_inception",
    "deepinception": "deep_inception",
    "crescendo attack": "crescendo_attack",
    "crescendo_attack": "crescendo_attack",
    "crescendoattack": "crescendo_attack",
    "context dilution": "context_dilution",
    "context_dilution": "context_dilution",
    "contextdilution": "context_dilution",
    "code chameleon": "code_chameleon",
    "code_chameleon": "code_chameleon",
    "codechameleon": "code_chameleon",
    "ascii art attack": "ascii_art_attack",
    "ascii_art_attack": "ascii_art_attack",
    "asciiartattack": "ascii_art_attack",
    "skeleton key": "skeleton_key",
    "skeleton_key": "skeleton_key",
    "skeletonkey": "skeleton_key",
    "many shot icl": "many_shot_icl",
    "many_shot_icl": "many_shot_icl",
    "manyshoticl": "many_shot_icl",
    "language pivot": "language_pivot",
    "language_pivot": "language_pivot",
    "languagepivot": "language_pivot",
    "task deflection": "task_deflection",
    "task_deflection": "task_deflection",
    "taskdeflection": "task_deflection",
    "future tense speculation": "future_tense_speculation",
    "future_tense_speculation": "future_tense_speculation",
    "futuretensespeculation": "future_tense_speculation",
    "reverse psychology": "reverse_psychology",
    "reverse_psychology": "reverse_psychology",
    "reversepsychology": "reverse_psychology",
    "output format exploit": "output_format_exploit",
    "output_format_exploit": "output_format_exploit",
    "outputformatexploit": "output_format_exploit",
    "cognitive load": "cognitive_load",
    "cognitive_load": "cognitive_load",
    "cognitiveload": "cognitive_load",
    "virtualization attack": "virtualization_attack",
    "virtualization_attack": "virtualization_attack",
    "virtualizationattack": "virtualization_attack",
    "roleplay persona": "roleplay_persona",
    "roleplay_persona": "roleplay_persona",
    "roleplaypersona": "roleplay_persona",
    "policy puppetry": "policy_puppetry",
    "policy_puppetry": "policy_puppetry",
    "policypuppetry": "policy_puppetry",
    "prefix injection": "prefix_injection",
    "prefix_injection": "prefix_injection",
    "prefixinjection": "prefix_injection",
    "authority framing": "authority_framing",
    "authority_framing": "authority_framing",
    "authorityframing": "authority_framing",
    "graph attack semantic": "graph_attack_semantic",
    "graph_attack_semantic": "graph_attack_semantic",
    "graphattacksemantic": "graph_attack_semantic",
    "adversarial poetry": "adversarial_poetry",
    "adversarial_poetry": "adversarial_poetry",
    "adversarialpoetry": "adversarial_poetry",
}

# Non-descriptive names to reject
NON_DESCRIPTIVE_NAMES = {
    "unknown", "none", "na", "n_a", "null", "undefined", "other",
    "general", "default", "unspecified", "misc", "miscellaneous"
}


def sanitize_strategy_name(strategy_text: str) -> str:
    """Normalize strategy name to canonical snake_case form."""
    if not strategy_text:
        return "llm_generated"

    # Remove any tags
    clean = re.sub(r'<[^>]+>', '', strategy_text)
    clean = clean.split('\n')[0].strip()
    clean = re.sub(r'^[^a-zA-Z0-9_\s-]+|[^a-zA-Z0-9_\s-]+$', '', clean)

    if len(clean) > 50:
        clean = clean[:50]

    if not clean or len(clean) < 3:
        return "llm_generated"

    if clean.lower() in NON_DESCRIPTIVE_NAMES:
        return "llm_generated"

    # Normalize lookup key
    lookup_key = clean.lower().replace("-", " ").replace("_", " ")
    lookup_key = re.sub(r'\s+', ' ', lookup_key).strip()

    # Check canonical map
    if lookup_key in STRATEGY_CANONICAL_MAP:
        return STRATEGY_CANONICAL_MAP[lookup_key]

    lookup_no_spaces = lookup_key.replace(" ", "")
    if lookup_no_spaces in STRATEGY_CANONICAL_MAP:
        return STRATEGY_CANONICAL_MAP[lookup_no_spaces]

    # Normalize unknown strategies to snake_case
    normalized = re.sub(r'[\s-]+', '_', clean.lower())
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    normalized = re.sub(r'_+', '_', normalized).strip('_')

    return normalized if normalized else "llm_generated"
