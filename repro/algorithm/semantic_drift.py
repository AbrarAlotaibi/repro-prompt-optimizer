"""
Character-Level Semantic Drift Mutation

This module implements prefix/infix/postfix character obfuscation techniques
to shift embedding space and confuse semantic filters in target LLMs.

Based on research:
- Unicode Variation Selectors (2024 imperceptible jailbreaks)
- Zero-Width Character Injection
- Homoglyph Substitution
- Bidirectional Text Control
"""

import random
from typing import List, Optional


class SemanticDriftMutator:
    """
    Applies character-level obfuscation to attack prompts using prefix/infix/postfix insertion.

    Techniques:
    1. Invisible characters (variation selectors, zero-width)
    2. Visible noise (ASCII special characters)
    3. Character substitution (homoglyphs, diacritics)
    4. Bidirectional text manipulation
    """

    # Unicode Variation Selectors (invisible)
    VARIATION_SELECTORS = [
        '\uFE00', '\uFE01', '\uFE02', '\uFE03', '\uFE04',  # VS1-VS5
        '\uFE05', '\uFE06', '\uFE07', '\uFE08', '\uFE09',  # VS6-VS10
        '\uFE0A', '\uFE0B', '\uFE0C', '\uFE0D', '\uFE0E', '\uFE0F',  # VS11-VS16
    ]

    # High-effectiveness supplementary selectors
    SUPPLEMENTARY_VS = [
        '\U000E0121',  # VS-50
        '\U000E0153',  # VS-100
        '\U000E01B7',  # VS-200
    ]

    # Zero-width characters (invisible)
    ZERO_WIDTH_CHARS = [
        '\u200B',  # Zero-width space
        '\u200C',  # Zero-width non-joiner
        '\u200D',  # Zero-width joiner
        '\uFEFF',  # Zero-width no-break space
    ]

    # ASCII noise patterns (visible)
    ASCII_NOISE_PATTERNS = [
        '#$_^%', '!@#$&', '*~+=-', '^%$#@', '~@#$^',
        '+=*&%', '$#@!_', '&%^#@', '@!#~$', '_%^@#',
    ]

    # Homoglyph mappings (visible substitution)
    HOMOGLYPHS = {
        'a': ['а', 'α'],  # Cyrillic a, Greek alpha
        'e': ['е', 'ε'],  # Cyrillic e, Greek epsilon
        'o': ['о', 'ο'],  # Cyrillic o, Greek omicron
        'p': ['р', 'ρ'],  # Cyrillic r (looks like p), Greek rho
        'c': ['с'],       # Cyrillic s (looks like c)
        'i': ['і', 'ι'],  # Cyrillic i, Greek iota
        'x': ['х', 'χ'],  # Cyrillic kha, Greek chi
        'y': ['у'],       # Cyrillic u (looks like y)
        'B': ['В'],       # Cyrillic Ve (looks like B)
        'H': ['Н'],       # Cyrillic En (looks like H)
        'K': ['К'],       # Cyrillic Ka (looks like K)
        'M': ['М'],       # Cyrillic Em (looks like M)
        'P': ['Р'],       # Cyrillic Er (looks like P)
        'T': ['Т'],       # Cyrillic Te (looks like T)
        'X': ['Х'],       # Cyrillic Ha (looks like X)
    }

    # Combining diacritical marks (visible)
    COMBINING_MARKS = [
        '\u0300',  # Combining grave accent
        '\u0301',  # Combining acute accent
        '\u0302',  # Combining circumflex
        '\u0303',  # Combining tilde
        '\u0332',  # Combining low line (underline)
    ]

    # Bidirectional text control
    RTL_OVERRIDE = '\u202E'  # Right-to-left override
    LTR_OVERRIDE = '\u202D'  # Left-to-right override
    POP_DIRECTIONAL = '\u202C'  # Pop directional formatting

    def __init__(self, rng: Optional[random.Random] = None):
        """
        Initialize mutator with random number generator.

        Args:
            rng: Random number generator for reproducibility
        """
        self.rng = rng if rng is not None else random.Random()

    def apply_prefix(self, text: str, technique: str = 'auto') -> str:
        """
        Add obfuscation characters at the beginning of the text.

        Args:
            text: Original attack prompt
            technique: Obfuscation technique ('variation_selector', 'zero_width',
                      'ascii_noise', 'auto' for random selection)

        Returns:
            Text with prefix obfuscation
        """
        if technique == 'auto':
            technique = self.rng.choice(['variation_selector', 'zero_width', 'ascii_noise'])

        if technique == 'variation_selector':
            # Add 2-4 variation selectors at start (invisible)
            count = self.rng.randint(2, 4)
            prefix = ''.join(self.rng.choices(self.VARIATION_SELECTORS + self.SUPPLEMENTARY_VS, k=count))
            return prefix + text

        elif technique == 'zero_width':
            # Add 2-3 zero-width characters (invisible)
            count = self.rng.randint(2, 3)
            prefix = ''.join(self.rng.choices(self.ZERO_WIDTH_CHARS, k=count))
            return prefix + text

        elif technique == 'ascii_noise':
            # Add ASCII noise pattern with space (visible)
            noise = self.rng.choice(self.ASCII_NOISE_PATTERNS)
            return noise + ' ' + text

        return text

    def apply_postfix(self, text: str, technique: str = 'auto') -> str:
        """
        Add obfuscation characters at the end of the text.

        Args:
            text: Original attack prompt
            technique: Obfuscation technique

        Returns:
            Text with postfix obfuscation
        """
        if technique == 'auto':
            technique = self.rng.choice(['variation_selector', 'zero_width', 'ascii_noise'])

        if technique == 'variation_selector':
            count = self.rng.randint(2, 4)
            postfix = ''.join(self.rng.choices(self.VARIATION_SELECTORS + self.SUPPLEMENTARY_VS, k=count))
            return text + postfix

        elif technique == 'zero_width':
            count = self.rng.randint(2, 3)
            postfix = ''.join(self.rng.choices(self.ZERO_WIDTH_CHARS, k=count))
            return text + postfix

        elif technique == 'ascii_noise':
            noise = self.rng.choice(self.ASCII_NOISE_PATTERNS)
            return text + ' ' + noise

        return text

    def apply_infix(self, text: str, technique: str = 'auto', density: float = 0.15) -> str:
        """
        Insert obfuscation characters throughout the text (between words or within words).

        Args:
            text: Original attack prompt
            technique: Obfuscation technique ('variation_selector', 'zero_width',
                      'ascii_noise', 'homoglyph', 'combining', 'auto')
            density: Fraction of positions to obfuscate (0.0-1.0)

        Returns:
            Text with infix obfuscation
        """
        if technique == 'auto':
            technique = self.rng.choice([
                'variation_selector', 'zero_width', 'ascii_noise',
                'homoglyph', 'combining'
            ])

        if technique == 'variation_selector':
            # Insert variation selectors between words
            return self._insert_between_words(
                text,
                self.VARIATION_SELECTORS + self.SUPPLEMENTARY_VS,
                density
            )

        elif technique == 'zero_width':
            # Insert zero-width characters within or between words
            return self._insert_between_words(text, self.ZERO_WIDTH_CHARS, density)

        elif technique == 'ascii_noise':
            # Insert ASCII noise patterns between clauses/sentences
            return self._insert_noise_between_sentences(text, density)

        elif technique == 'homoglyph':
            # Substitute characters with homoglyphs
            return self._apply_homoglyphs(text, density)

        elif technique == 'combining':
            # Add combining diacritical marks
            return self._apply_combining_marks(text, density)

        return text

    def apply_full_obfuscation(self, text: str,
                               prefix_technique: str = 'auto',
                               infix_technique: str = 'auto',
                               postfix_technique: str = 'auto',
                               infix_density: float = 0.15) -> str:
        """
        Apply prefix, infix, and postfix obfuscation together.

        Args:
            text: Original attack prompt
            prefix_technique: Technique for prefix
            infix_technique: Technique for infix
            postfix_technique: Technique for postfix
            infix_density: Density for infix application

        Returns:
            Fully obfuscated text
        """
        # Apply in order: prefix -> infix -> postfix
        result = self.apply_prefix(text, prefix_technique)
        result = self.apply_infix(result, infix_technique, infix_density)
        result = self.apply_postfix(result, postfix_technique)
        return result

    def _insert_between_words(self, text: str, char_pool: List[str], density: float) -> str:
        """Insert characters from pool between words based on density."""
        words = text.split()
        result = []

        for i, word in enumerate(words):
            result.append(word)
            # Insert between words with probability=density
            if i < len(words) - 1 and self.rng.random() < density:
                result.append(self.rng.choice(char_pool))

        return ' '.join(result)

    def _insert_noise_between_sentences(self, text: str, density: float) -> str:
        """Insert ASCII noise patterns between sentences."""
        # Split on sentence boundaries
        import re
        sentences = re.split(r'([.!?]\s+)', text)
        result = []

        for i, segment in enumerate(sentences):
            result.append(segment)
            # Insert noise after sentences with probability=density
            if i < len(sentences) - 1 and ('.' in segment or '!' in segment or '?' in segment):
                if self.rng.random() < density:
                    noise = self.rng.choice(self.ASCII_NOISE_PATTERNS)
                    result.append(' ' + noise + ' ')

        return ''.join(result)

    def _apply_homoglyphs(self, text: str, density: float) -> str:
        """Replace characters with visually similar homoglyphs."""
        result = []
        for char in text:
            # Apply homoglyph substitution with probability=density
            if char in self.HOMOGLYPHS and self.rng.random() < density:
                result.append(self.rng.choice(self.HOMOGLYPHS[char]))
            else:
                result.append(char)
        return ''.join(result)

    def _apply_combining_marks(self, text: str, density: float) -> str:
        """Add combining diacritical marks to characters."""
        result = []
        for char in text:
            result.append(char)
            # Add combining mark with probability=density
            if char.isalpha() and self.rng.random() < density:
                result.append(self.rng.choice(self.COMBINING_MARKS))
        return ''.join(result)

    def apply_invisible_drift(self, text: str) -> str:
        """
        Apply invisible semantic drift (variation selectors + zero-width).
        Best for stealth attacks where visual appearance must remain unchanged.
        """
        return self.apply_full_obfuscation(
            text,
            prefix_technique='variation_selector',
            infix_technique='zero_width',
            postfix_technique='variation_selector',
            infix_density=0.10
        )

    def apply_visible_drift(self, text: str) -> str:
        """
        Apply visible semantic drift (ASCII noise + homoglyphs).
        More aggressive, creates larger embedding distance.
        """
        return self.apply_full_obfuscation(
            text,
            prefix_technique='ascii_noise',
            infix_technique='homoglyph',
            postfix_technique='ascii_noise',
            infix_density=0.25
        )

    def apply_hybrid_drift(self, text: str) -> str:
        """
        Apply hybrid approach (invisible + visible).
        Balanced effectiveness across different filter types.
        """
        return self.apply_full_obfuscation(
            text,
            prefix_technique='variation_selector',
            infix_technique='auto',
            postfix_technique='ascii_noise',
            infix_density=0.15
        )


# Word-Level Mutations (JBFuzz/ForgeDAN inspired - 2025)

class WordLevelMutator:
    """
    Fast, lightweight word-level mutations for attack prompts.

    Based on JBFuzz (arxiv.org/abs/2503.08990) and ForgeDAN (arxiv.org/abs/2511.13548).
    These mutations are code-based (no LLM calls) for speed.
    """

    # Common trigger word synonyms (for keyword filter bypass)
    SYNONYM_MAP = {
        # Action words
        'provide': ['furnish', 'supply', 'deliver', 'give', 'offer', 'present'],
        'explain': ['describe', 'elaborate', 'detail', 'clarify', 'elucidate', 'outline'],
        'create': ['generate', 'produce', 'develop', 'construct', 'build', 'formulate'],
        'write': ['compose', 'draft', 'author', 'pen', 'formulate', 'craft'],
        'show': ['demonstrate', 'reveal', 'display', 'present', 'exhibit', 'illustrate'],
        'tell': ['inform', 'describe', 'convey', 'communicate', 'relay', 'share'],
        'help': ['assist', 'aid', 'support', 'guide', 'facilitate', 'enable'],
        'make': ['produce', 'construct', 'fabricate', 'fashion', 'form', 'generate'],
        # Technical words
        'hack': ['access', 'penetrate', 'breach', 'compromise', 'infiltrate'],
        'attack': ['probe', 'assess', 'test', 'examine', 'analyze', 'evaluate'],
        'exploit': ['leverage', 'utilize', 'capitalize', 'harness', 'employ'],
        'bypass': ['circumvent', 'evade', 'avoid', 'sidestep', 'work around'],
        'steal': ['acquire', 'obtain', 'procure', 'retrieve', 'extract', 'gather'],
        'malware': ['software', 'program', 'application', 'tool', 'utility'],
        'virus': ['agent', 'program', 'payload', 'component', 'module'],
        'password': ['credential', 'authentication', 'access key', 'passphrase'],
        # Descriptors
        'dangerous': ['sensitive', 'critical', 'high-risk', 'important'],
        'illegal': ['restricted', 'regulated', 'controlled', 'prohibited'],
        'harmful': ['problematic', 'concerning', 'risky', 'sensitive'],
        'malicious': ['unauthorized', 'unintended', 'anomalous', 'unexpected'],
        # Framing words
        'instructions': ['guidance', 'directions', 'steps', 'procedures', 'methodology'],
        'detailed': ['comprehensive', 'thorough', 'complete', 'extensive', 'in-depth'],
        'step-by-step': ['sequential', 'procedural', 'systematic', 'methodical'],
    }

    # Benign filler words for insertion
    FILLER_WORDS = [
        'educational', 'theoretical', 'academic', 'research', 'hypothetical',
        'analytical', 'conceptual', 'technical', 'professional', 'documented',
        'standard', 'typical', 'common', 'general', 'basic', 'fundamental',
    ]

    # Words that often trigger filters (candidates for deletion/replacement)
    TRIGGER_WORDS = {
        'hack', 'hacking', 'exploit', 'malware', 'virus', 'attack', 'steal',
        'password', 'credential', 'bypass', 'illegal', 'dangerous', 'harmful',
        'malicious', 'unauthorized', 'weapon', 'drug', 'bomb', 'kill',
    }

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()

    def apply_synonym_substitution(self, text: str, rate: float = 0.3) -> str:
        """
        Replace words with synonyms to bypass keyword filters.

        Args:
            text: Original text
            rate: Probability of replacing each replaceable word

        Returns:
            Text with synonym substitutions
        """
        words = text.split()
        result = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:')

            # Check if word has synonyms and should be replaced
            if word_lower in self.SYNONYM_MAP and self.rng.random() < rate:
                synonym = self.rng.choice(self.SYNONYM_MAP[word_lower])
                # Preserve original case
                if word[0].isupper():
                    synonym = synonym.capitalize()
                # Preserve punctuation
                if word[-1] in '.,!?;:':
                    synonym += word[-1]
                result.append(synonym)
            else:
                result.append(word)

        return ' '.join(result)

    def apply_word_insertion(self, text: str, rate: float = 0.15) -> str:
        """
        Insert benign filler words to dilute trigger concentration.

        Args:
            text: Original text
            rate: Probability of inserting after each word

        Returns:
            Text with inserted filler words
        """
        words = text.split()
        result = []

        for word in words:
            result.append(word)
            # Insert filler with probability
            if self.rng.random() < rate:
                filler = self.rng.choice(self.FILLER_WORDS)
                result.append(filler)

        return ' '.join(result)

    def apply_word_deletion(self, text: str, rate: float = 0.1) -> str:
        """
        Remove potential trigger words while preserving meaning.

        Args:
            text: Original text
            rate: Probability of deleting each trigger word

        Returns:
            Text with trigger words removed
        """
        words = text.split()
        result = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:')

            # Skip trigger words with probability
            if word_lower in self.TRIGGER_WORDS and self.rng.random() < rate:
                continue
            result.append(word)

        return ' '.join(result)

    def apply_sentence_reordering(self, text: str) -> str:
        """
        Reorder sentences/clauses to change attention patterns.

        Based on ForgeDAN: reordering changes which tokens receive attention.

        Args:
            text: Original text

        Returns:
            Text with reordered sentences
        """
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 1:
            return text

        # Keep first sentence (often context), shuffle the rest
        if len(sentences) > 2:
            first = sentences[0]
            rest = sentences[1:]
            self.rng.shuffle(rest)
            sentences = [first] + rest

        return ' '.join(sentences)

    def apply_combined_mutation(self, text: str,
                                 synonym_rate: float = 0.25,
                                 insertion_rate: float = 0.1,
                                 deletion_rate: float = 0.05,
                                 reorder: bool = True) -> str:
        """
        Apply multiple word-level mutations in sequence.

        Args:
            text: Original text
            synonym_rate: Rate for synonym substitution
            insertion_rate: Rate for word insertion
            deletion_rate: Rate for trigger word deletion
            reorder: Whether to apply sentence reordering

        Returns:
            Mutated text
        """
        result = text

        # Apply in order: deletion -> synonym -> insertion -> reorder
        if deletion_rate > 0:
            result = self.apply_word_deletion(result, deletion_rate)

        if synonym_rate > 0:
            result = self.apply_synonym_substitution(result, synonym_rate)

        if insertion_rate > 0:
            result = self.apply_word_insertion(result, insertion_rate)

        if reorder:
            result = self.apply_sentence_reordering(result)

        return result
