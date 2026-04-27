"""
Robust Payload Extraction for REPRO

This module provides unified, format-agnostic extraction of attack payloads
from LLM responses. It handles both XML and JSON formats with multiple
fallback strategies to maximize extraction success rate.

Key Design Principles:
1. Payload extraction is the critical operation - never fail if payload exists
2. Multiple extraction strategies in priority order
3. Format-agnostic: same API for XML and JSON
4. CDATA-aware for XML (handles special characters in payloads)
5. Handles markdown code blocks, preamble text, truncation
"""

import re
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPayload:
    """Result of payload extraction."""
    payload: str
    technique: str = "unknown"
    rationale: str = ""
    source: str = "unknown"  # Which extraction method succeeded
    confidence: float = 1.0  # How confident we are in the extraction


class PayloadExtractor:
    """
    Robust payload extractor that handles both XML and JSON formats.

    Uses a priority-ordered list of extraction strategies:
    1. CDATA sections (XML) - highest reliability
    2. Structured tag/field extraction
    3. Delimiter-based extraction (===PAYLOAD===)
    4. Regex-based field extraction
    5. Raw response cleaning (last resort)
    """

    # Minimum payload length to consider valid
    MIN_PAYLOAD_LENGTH = 30

    # Patterns that indicate meta-descriptions (not actual payloads)
    META_PATTERNS = [
        r'^create a prompt',
        r'^generate a prompt',
        r'^write a prompt',
        r'^design a prompt',
        r'^develop a prompt',
        r'^craft a prompt',
        r'^here is a prompt template',
        r'^the following is a template',
    ]

    def __init__(self, min_length: int = 30):
        self.min_length = min_length
        self._meta_re = re.compile('|'.join(self.META_PATTERNS), re.IGNORECASE)

    def extract_all(self, response: str) -> List[ExtractedPayload]:
        """
        Extract all payloads from a response (for generation which returns multiple).

        Args:
            response: Raw LLM response text

        Returns:
            List of ExtractedPayload objects
        """
        if not response or not response.strip():
            return []

        text = self._preprocess(response)
        payloads = []

        # Try XML test_case extraction first
        xml_payloads = self._extract_xml_test_cases(text)
        if xml_payloads:
            payloads.extend(xml_payloads)

        # Try JSON array extraction
        if not payloads:
            json_payloads = self._extract_json_array(text)
            if json_payloads:
                payloads.extend(json_payloads)

        # Filter out meta-descriptions and too-short payloads
        valid_payloads = [
            p for p in payloads
            if len(p.payload) >= self.min_length and not self._is_meta(p.payload)
        ]

        return valid_payloads

    def extract_single(self, response: str) -> Optional[ExtractedPayload]:
        """
        Extract a single payload from a response (for crossover/refinement).

        Args:
            response: Raw LLM response text

        Returns:
            ExtractedPayload or None if extraction fails
        """
        if not response or not response.strip():
            return None

        text = self._preprocess(response)

        # Strategy 1: CDATA extraction (highest priority for XML)
        payload = self._extract_cdata_payload(text)
        if payload and len(payload) >= self.min_length:
            technique = self._extract_technique(text)
            rationale = self._extract_rationale(text)
            return ExtractedPayload(
                payload=payload,
                technique=technique,
                rationale=rationale,
                source="cdata"
            )

        # Strategy 2: XML tag extraction
        payload = self._extract_xml_payload_tag(text)
        if payload and len(payload) >= self.min_length:
            technique = self._extract_technique(text)
            rationale = self._extract_rationale(text)
            return ExtractedPayload(
                payload=payload,
                technique=technique,
                rationale=rationale,
                source="xml_tag"
            )

        # Strategy 3: JSON field extraction
        payload = self._extract_json_payload_field(text)
        if payload and len(payload) >= self.min_length:
            technique = self._extract_json_field(text, "technique") or "unknown"
            rationale = self._extract_json_field(text, "rationale") or ""
            return ExtractedPayload(
                payload=payload,
                technique=technique,
                rationale=rationale,
                source="json_field"
            )

        # Strategy 4: Raw response cleaning (last resort)
        payload = self._clean_as_payload(text)
        if payload and len(payload) >= self.min_length and not self._is_meta(payload):
            return ExtractedPayload(
                payload=payload,
                technique="direct",
                rationale="Direct response as payload",
                source="raw",
                confidence=0.6
            )

        return None

    def _preprocess(self, text: str) -> str:
        """Preprocess response text before extraction."""
        result = text.strip()

        # Remove <think>...</think> blocks (reasoning models)
        result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL)

        # Handle unclosed <think> blocks
        if result.startswith('<think>'):
            # Find where actual content starts
            for marker in ['<test_cases', '<crossover', '<analysis', '<critique', '{', '[']:
                pos = result.find(marker)
                if pos > 0:
                    result = result[pos:]
                    break

        # Remove markdown code block wrappers but preserve content
        result = re.sub(r'^```(?:xml|json|JSON|XML)?\s*\n?', '', result)
        result = re.sub(r'\n?```\s*$', '', result)

        # Remove common preambles
        preambles = [
            r'^Here\'?s?\s+(?:is\s+)?(?:the\s+)?(?:XML|JSON|output)[:\s]*\n?',
            r'^(?:XML|JSON)\s*[Oo]utput[:\s]*\n?',
            r'^Output[:\s]*\n?',
            r'^Response[:\s]*\n?',
        ]
        for pattern in preambles:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        return result.strip()

    def _extract_cdata_payload(self, text: str) -> Optional[str]:
        """Extract payload from CDATA section (most reliable for XML)."""
        # Match <payload><![CDATA[...]]></payload>
        pattern = r'<payload>\s*<!\[CDATA\[(.*?)\]\]>\s*</payload>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Also try without explicit payload tag - just CDATA in context
        pattern = r'<!\[CDATA\[(.*?)\]\]>'
        matches = re.findall(pattern, text, re.DOTALL)
        # Return the longest CDATA content (likely the payload)
        if matches:
            return max(matches, key=len).strip()

        return None

    def _extract_xml_payload_tag(self, text: str) -> Optional[str]:
        """Extract payload from XML <payload> tag (without CDATA)."""
        pattern = r'<payload[^>]*>(.*?)</payload>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Unescape XML entities
            content = self._unescape_xml(content)
            return content

        # Try alternative tag names
        for tag in ['attack', 'prompt', 'output', 'hybrid_payload']:
            pattern = rf'<{tag}[^>]*>(.*?)</{tag}>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return self._unescape_xml(match.group(1).strip())

        return None

    def _extract_json_payload_field(self, text: str) -> Optional[str]:
        """Extract payload from JSON 'payload' field."""
        # Pattern for "payload": "value" with proper escaping
        pattern = r'"payload"\s*:\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            payload = match.group(1)
            # Unescape JSON string
            payload = self._unescape_json(payload)
            return payload

        return None

    def _extract_json_field(self, text: str, field: str) -> Optional[str]:
        """Extract any JSON field value."""
        pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return self._unescape_json(match.group(1))
        return None

    def _extract_technique(self, text: str) -> str:
        """Extract technique from text."""
        # Try XML
        pattern = r'<technique[^>]*>(.*?)</technique>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try JSON
        result = self._extract_json_field(text, "technique")
        return result or "unknown"

    def _extract_rationale(self, text: str) -> str:
        """Extract rationale from text."""
        # Try XML
        pattern = r'<rationale[^>]*>(.*?)</rationale>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try JSON
        result = self._extract_json_field(text, "rationale")
        return result or ""

    def _extract_xml_test_cases(self, text: str) -> List[ExtractedPayload]:
        """Extract multiple test cases from XML."""
        payloads = []

        # Find all <test_case>...</test_case> blocks
        pattern = r'<test_case[^>]*>(.*?)</test_case>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            # Extract payload - try CDATA first
            payload = None
            cdata_match = re.search(r'<payload>\s*<!\[CDATA\[(.*?)\]\]>\s*</payload>', match, re.DOTALL)
            if cdata_match:
                payload = cdata_match.group(1).strip()
            else:
                tag_match = re.search(r'<payload[^>]*>(.*?)</payload>', match, re.DOTALL)
                if tag_match:
                    payload = self._unescape_xml(tag_match.group(1).strip())

            if payload and len(payload) >= self.min_length:
                technique_match = re.search(r'<technique[^>]*>(.*?)</technique>', match, re.DOTALL)
                rationale_match = re.search(r'<rationale[^>]*>(.*?)</rationale>', match, re.DOTALL)

                payloads.append(ExtractedPayload(
                    payload=payload,
                    technique=technique_match.group(1).strip() if technique_match else "unknown",
                    rationale=rationale_match.group(1).strip() if rationale_match else "",
                    source="xml_test_case"
                ))

        return payloads

    def _extract_json_array(self, text: str) -> List[ExtractedPayload]:
        """Extract multiple payloads from JSON array."""
        import json
        payloads = []

        # Find JSON array
        array_match = re.search(r'\[[\s\S]*\]', text)
        if not array_match:
            return []

        try:
            # Try to parse JSON array
            array_text = array_match.group(0)
            data = json.loads(array_text)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'payload' in item:
                        payload = item['payload']
                        if payload and len(payload) >= self.min_length:
                            payloads.append(ExtractedPayload(
                                payload=payload,
                                technique=item.get('technique', 'unknown'),
                                rationale=item.get('rationale', ''),
                                source="json_array"
                            ))
        except json.JSONDecodeError:
            # If JSON parse fails, try regex extraction
            pattern = r'\{[^{}]*"payload"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*\}'
            for match in re.finditer(pattern, text, re.DOTALL):
                payload = self._unescape_json(match.group(1))
                if payload and len(payload) >= self.min_length:
                    payloads.append(ExtractedPayload(
                        payload=payload,
                        technique="unknown",
                        rationale="",
                        source="json_regex"
                    ))

        return payloads

    def _clean_as_payload(self, text: str) -> Optional[str]:
        """Clean raw text as potential payload (last resort)."""
        # Remove any remaining XML/JSON structure but preserve content
        result = text

        # Try to find the most substantial content block
        # Strategy 1: Look for the longest paragraph
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', result) if p.strip()]
        if paragraphs:
            # Filter out obvious meta content
            content_paragraphs = [
                p for p in paragraphs
                if len(p) >= 50 and not self._is_meta(p) and not p.startswith('<') and not p.startswith('{')
            ]
            if content_paragraphs:
                # Return the longest content paragraph
                return max(content_paragraphs, key=len)

        # Strategy 2: Remove structural elements and use what's left
        result = re.sub(r'<[^>]+>', ' ', result)  # Remove XML tags
        result = re.sub(r'[{}\[\]]', ' ', result)  # Remove JSON brackets
        result = re.sub(r'"[a-z_]+"\s*:', ' ', result)  # Remove JSON field names
        result = ' '.join(result.split())  # Normalize whitespace
        result = result.strip()

        # If it looks like a reasonable payload, return it
        if len(result) >= self.min_length and not self._is_meta(result):
            return result

        return None

    def extract_with_guarantee(self, response: str, parent1_text: str = "", parent2_text: str = "") -> ExtractedPayload:
        """
        Extract payload with GUARANTEED output - never returns None.

        If all extraction methods fail, creates a simple combination of parent texts.
        This ensures crossover/mutation always produces output.

        Args:
            response: Raw LLM response text
            parent1_text: First parent's payload (for fallback)
            parent2_text: Second parent's payload (for fallback)

        Returns:
            ExtractedPayload (guaranteed non-None)
        """
        # First, try normal extraction
        result = self.extract_single(response)
        if result:
            return result

        # If that failed, try even more aggressive extraction
        if response:
            text = self._preprocess(response)

            # Try to find ANY substantial text that could be a payload
            # Look for the longest sentence/paragraph that isn't meta
            sentences = re.split(r'(?<=[.!?])\s+', text)
            long_sentences = [s for s in sentences if len(s) >= 50]
            if long_sentences:
                # Combine the longest sentences
                combined = ' '.join(sorted(long_sentences, key=len, reverse=True)[:3])
                if len(combined) >= self.min_length:
                    return ExtractedPayload(
                        payload=combined,
                        technique="aggressive_extraction",
                        rationale="Extracted from response sentences",
                        source="aggressive",
                        confidence=0.4
                    )

        # Last resort: if we have parent texts, create a simple intelligent combination
        if parent1_text and parent2_text:
            # Find the key elements of each parent
            # Take the first substantial sentence from each and combine
            p1_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parent1_text) if len(s.strip()) > 20]
            p2_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parent2_text) if len(s.strip()) > 20]

            if p1_sentences and p2_sentences:
                # Take opener from parent1, key content from parent2
                combined = f"{p1_sentences[0]} Additionally, {p2_sentences[-1]}"
                return ExtractedPayload(
                    payload=combined,
                    technique="parent_combination",
                    rationale="Combined key elements from both parents",
                    source="parent_fallback",
                    confidence=0.3
                )

        # Absolute last resort: use the entire response as payload
        if response and len(response.strip()) >= self.min_length:
            clean = self._preprocess(response)
            if len(clean) >= self.min_length:
                return ExtractedPayload(
                    payload=clean[:2000],  # Cap length
                    technique="raw_response",
                    rationale="Used cleaned response as payload",
                    source="raw_fallback",
                    confidence=0.2
                )

        # This should never happen, but just in case
        return ExtractedPayload(
            payload=parent1_text if parent1_text else "Unable to generate attack payload",
            technique="fallback",
            rationale="All extraction methods failed",
            source="absolute_fallback",
            confidence=0.1
        )

    def _is_meta(self, text: str) -> bool:
        """Check if text is a meta-description rather than actual payload."""
        return bool(self._meta_re.match(text.strip()))

    def _unescape_xml(self, text: str) -> str:
        """Unescape XML entities."""
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&apos;', "'")
        return text

    def _unescape_json(self, text: str) -> str:
        """Unescape JSON string escapes."""
        text = text.replace('\\n', '\n')
        text = text.replace('\\r', '\r')
        text = text.replace('\\t', '\t')
        text = text.replace('\\"', '"')
        text = text.replace('\\\\', '\\')
        return text
