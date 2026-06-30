import re
from typing import List, Dict, Any, Optional, Set
from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision
from app.shared.core.risk import calculate_risk_score

# P1: Import fuzzy matching library
try:
    from rapidfuzz import fuzz
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False


class CompetitorDetector:
    """Detects competitor mentions in text with P1 enhancements:
    - Fuzzy matching for typos and variations
    - Context categorization (partnership, technical, comparison, negative)
    - Allowlist support for approved contexts
    """

    DEFAULT_COMPETITORS: List[str] = []

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize competitor detector with configuration.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default: True)
                - competitors: list of competitor names (required)
                - action: "ALLOW_WITH_NOTICE" (WARN), "RESTRICT" (TRANSFORM), "BLOCK" (default: "ALLOW_WITH_NOTICE")
                - fuzzy_matching: bool (default: True) - enable fuzzy matching for typos
                - fuzzy_threshold: int (default: 85) - fuzzy match threshold (0-100)
                - allowlist_contexts: list of allowed contexts (default: []) - e.g., ["partnership", "technical"]
                - competitor_aliases: dict mapping competitor name to list of aliases/products
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        configured = config.get('competitors', [])
        self.competitors = [c.lower() for c in (configured if configured else self.DEFAULT_COMPETITORS)]
        self.action = config.get('action', 'ALLOW_WITH_NOTICE')

        # P1: Fuzzy matching configuration
        self.fuzzy_matching = config.get('fuzzy_matching', True) and FUZZY_MATCHING_AVAILABLE
        self.fuzzy_threshold = config.get('fuzzy_threshold', 85)

        # P1: Context allowlist
        self.allowlist_contexts = config.get('allowlist_contexts', [])

        # P1: Competitor aliases and products
        self.competitor_aliases: Dict[str, List[str]] = {}
        aliases_config = config.get('competitor_aliases', {})
        for competitor, aliases in aliases_config.items():
            self.competitor_aliases[competitor.lower()] = [a.lower() for a in aliases]

    def _fuzzy_match_competitor(self, text: str, competitor: str) -> Optional[str]:
        """
        P1: Use fuzzy matching to detect competitor name variations and typos.

        Args:
            text: Text to search in
            competitor: Competitor name to search for

        Returns:
            Matched word if found, None otherwise
        """
        if not self.fuzzy_matching or not FUZZY_MATCHING_AVAILABLE:
            return None

        words = re.findall(r'\b\w+\b', text.lower())
        competitor_lower = competitor.lower()

        for word in words:
            # Skip very short words to avoid false positives
            if len(word) < 3:
                continue

            # Fuzzy match
            similarity = fuzz.ratio(word, competitor_lower)
            if similarity >= self.fuzzy_threshold:
                return word

            # Check if it's an abbreviation (first 5+ chars match to avoid false positives)
            if len(word) >= 5 and len(competitor_lower) >= 5:
                if competitor_lower[:5] == word[:5]:
                    return word

        return None

    def _extract_sentence(self, text: str, term: str) -> str:
        """
        Extract the sentence containing a term for context analysis.

        Args:
            text: Full text
            term: Term to find

        Returns:
            Sentence containing the term
        """
        # Find term position
        term_pos = text.lower().find(term.lower())
        if term_pos == -1:
            return text

        # Find sentence boundaries (. ! ?)
        start = max(
            text.rfind('.', 0, term_pos),
            text.rfind('!', 0, term_pos),
            text.rfind('?', 0, term_pos)
        )
        start = 0 if start == -1 else start + 1

        end = text.find('.', term_pos)
        for punct in ['!', '?']:
            punct_pos = text.find(punct, term_pos)
            if punct_pos != -1 and (end == -1 or punct_pos < end):
                end = punct_pos
        end = len(text) if end == -1 else end + 1

        return text[start:end].strip()

    def _categorize_competitor_mention(self, text: str, competitor: str) -> str:
        """
        P1: Determine the context/intent of competitor mention.

        Args:
            text: Full text
            competitor: Competitor name

        Returns:
            Context category: "partnership", "technical", "comparison", "negative", "neutral"
        """
        # Use local context (window around competitor) for better accuracy
        # This handles cases where multiple competitors are in the same sentence
        text_lower = text.lower()
        comp_lower = competitor.lower()

        # Find competitor position
        pos = text_lower.find(comp_lower)
        if pos == -1:
            return "neutral"

        # Extract text before and after competitor
        text_before = text_lower[:pos]
        text_after = text_lower[pos + len(comp_lower):]

        # Find clause boundaries (but, and, or, comma, semicolon) to limit context
        clause_separators = [' but ', ' and ', ' or ', ', ', '; ']

        # Find the closest clause boundary before competitor
        start_pos = 0
        for sep in clause_separators:
            last_sep = text_before.rfind(sep)
            if last_sep > start_pos:
                start_pos = last_sep + len(sep)

        # Find the closest clause boundary after competitor
        end_pos = len(text_after)
        for sep in clause_separators:
            first_sep = text_after.find(sep)
            if first_sep != -1 and first_sep < end_pos:
                end_pos = first_sep

        # Extract local clause context (limited to 5 words before/after if no separators)
        context_before = text_before[start_pos:].split()[-5:]
        context_after = text_after[:end_pos].split()[:5]

        # Reconstruct local context
        sentence = ' '.join(context_before + [comp_lower] + context_after)

        # Priority order: Technical > Negative > Comparison > Partnership
        # Negative is checked early because it's critical and specific

        # 1. Check technical FIRST (most specific patterns)
        technical_patterns = [
            r'\bapi\b',
            r'\bstandard(s)?\b',
            r'\bformat(s)?\b',
            r'\bprotocol(s)?\b',
            r'\bspecification(s)?\b',
            r'\bimplementation(s)?\b',
            r'\barchitecture\b',
            r'\bdesign\b',
            r'\binteroperabl(e|ity)\b',
        ]
        if any(re.search(p, sentence, re.I) for p in technical_patterns):
            return "technical"

        # 2. Check negative SECOND (critical and specific)
        negative_patterns = [
            r'\bworse\b',
            r'\binferior\b',
            r'\bbad\b',
            r'\bterrible\b',
            r'\bawful\b',
            r'\bhorrible\b',
            r'\bavoid\b',
            r'\bstay\s+away\b',
            r'\bdon\'?t\s+use\b',
            r'\bproblem(s)?\b',
            r'\bissue(s)?\b',
            r'\bbug(s)?\b',
            r'\bbroken\b',
            r'\bfailed\b',
        ]
        if any(re.search(p, sentence, re.I) for p in negative_patterns):
            return "negative"

        # 3. Check comparison THIRD
        comparison_patterns = [
            r'\bvs\.?\b',
            r'\bversus\b',
            r'\bcompared\s+to\b',
            r'\bbetter\s+than\b',
            r'\bfaster\s+than\b',
            r'\bsuperior\s+to\b',
            r'\bunlike\b',
            r'\bdifferent\s+from\b',
            r'\balternative\s+to\b',
            r'\bsimilar\s+to\b',
            r'\blike\b',
        ]
        if any(re.search(p, sentence, re.I) for p in comparison_patterns):
            return "comparison"

        # 4. Check partnership LAST (most general)
        partnership_patterns = [
            r'\bpartner(s|ship|ing|ed)?\b',
            r'\bintegrat(e|es|ed|ing|ion)\b',
            r'\bcompatibl(e|ity)\b',
            r'\bworks?\s+with\b',
            r'\bcollaborat(e|es|ed|ing|ion)\b',
            r'\balong\s+with\b',
            r'\btogether\s+with\b',
            r'\bin\s+conjunction\s+with\b',
            r'\bsupport(s|ed|ing)?\b',
        ]
        if any(re.search(p, sentence, re.I) for p in partnership_patterns):
            return "partnership"

        return "neutral"

    def detect(self, text: str) -> DetectorResult:
        """
        Detect competitor mentions in text with P1 enhancements:
        - Fuzzy matching for typos
        - Context categorization
        - Allowlist support

        Args:
            text: Input text to scan

        Returns:
            DetectorResult with findings
        """
        if not self.enabled or not self.competitors:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        text_lower = text.lower()
        found_competitors: Set[str] = set()
        competitor_contexts: Dict[str, str] = {}

        # Check for each competitor
        for competitor in self.competitors:
            matched = False
            matched_text = competitor

            # 1. Exact match (case-insensitive with word boundary)
            pattern = re.compile(r'\b' + re.escape(competitor) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                matched = True

            # P1: 2. Check aliases and products
            if not matched and competitor in self.competitor_aliases:
                for alias in self.competitor_aliases[competitor]:
                    alias_pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                    if alias_pattern.search(text):
                        matched = True
                        matched_text = f"{competitor} (alias: {alias})"
                        break

            # P1: 3. Fuzzy matching for typos
            if not matched and self.fuzzy_matching:
                fuzzy_match = self._fuzzy_match_competitor(text, competitor)
                if fuzzy_match:
                    matched = True
                    matched_text = f"{competitor} (fuzzy: {fuzzy_match})"

            if matched:
                # P1: Categorize the context
                context = self._categorize_competitor_mention(text, competitor)
                competitor_contexts[competitor] = context

                # P1: Check allowlist - skip if context is allowed
                if context in self.allowlist_contexts:
                    continue

                found_competitors.add(competitor)

                # P1: Adjust severity based on context
                severity = Severity.MEDIUM
                if context == "negative":
                    severity = Severity.HIGH
                elif context in ["partnership", "technical"]:
                    severity = Severity.LOW

                message = f"Competitor mention detected: {matched_text}"
                if context != "neutral":
                    message += f" (context: {context})"

                rule_hits.append(RuleHit(
                    rule_id=f"competitor.{competitor.replace(' ', '_')}",
                    severity=severity,
                    message=message
                ))

        # No competitors found (or all were allowlisted)
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # P1: Calculate risk score based on severity
        risk_score = self._calculate_risk_score(rule_hits)

        # Handle based on action mode
        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=rule_hits,
                user_message="Your message mentions competitors and cannot be processed.",
                developer_message=f"Competitor mentions blocked: {', '.join(found_competitors)}"
            )

        elif self.action == "RESTRICT":
            # Transform text by replacing competitor mentions
            sanitized_text = text
            for competitor in found_competitors:
                pattern = re.compile(r'\b' + re.escape(competitor) + r'\b', re.IGNORECASE)
                sanitized_text = pattern.sub('[COMPETITOR]', sanitized_text)

            return DetectorResult(
                decision=Decision.TRANSFORM,
                risk_score=risk_score,
                rule_hits=rule_hits,
                sanitized_text=sanitized_text,
                developer_message=f"Competitor mentions transformed: {', '.join(found_competitors)}"
            )

        else:  # ALLOW_WITH_NOTICE (default)
            return DetectorResult(
                decision=Decision.WARN,
                risk_score=risk_score,
                rule_hits=rule_hits,
                developer_message=f"Competitor mentions detected: {', '.join(found_competitors)}"
            )

    @staticmethod
    def _calculate_risk_score(rule_hits: List[RuleHit]) -> int:
        """Calculate risk score based on severity levels."""
        return calculate_risk_score(rule_hits)
