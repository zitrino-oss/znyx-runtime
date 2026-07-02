from typing import List
from znyx_core.core.models import Decision, DetectorResult, RuleHit
from znyx_core.core.labels import DECISION_PRECEDENCE


class DecisionAggregator:
    """Aggregates detector results into a final decision"""

    # Canonical "worst wins" ranking (BLOCK > REDACT > TRANSFORM > WARN > ALLOW),
    # defined once in core.labels so the aggregator, trace UI, and docs agree.
    DECISION_PRIORITY = DECISION_PRECEDENCE

    @staticmethod
    def aggregate(results: List[DetectorResult]) -> DetectorResult:
        """
        Aggregate multiple detector results into a single result.
        Uses highest priority decision and combines all rule hits.
        """
        if not results:
            return DetectorResult(
                decision=Decision.ALLOW,
                risk_score=0,
                rule_hits=[],
                sanitized_text=None,
                user_message=None,
                developer_message=None
            )

        # Combine all rule hits
        all_rule_hits: List[RuleHit] = []
        for result in results:
            all_rule_hits.extend(result.rule_hits)

        # Calculate max risk score
        max_risk_score = max((r.risk_score for r in results), default=0)

        # Find highest priority decision
        highest_decision = Decision.ALLOW
        highest_priority = 0
        selected_result = None

        for result in results:
            if result.decision:
                priority = DecisionAggregator.DECISION_PRIORITY.get(result.decision, 0)
                if priority > highest_priority:
                    highest_priority = priority
                    highest_decision = result.decision
                    selected_result = result

        # Use sanitized/transformed text and messages from the selected result
        sanitized_text = selected_result.sanitized_text if selected_result else None
        user_message = selected_result.user_message if selected_result else None
        developer_message = selected_result.developer_message if selected_result else None

        # If we have multiple results with text transformations, use the last one
        # (this handles chaining of transformations)
        for result in results:
            if result.sanitized_text and result.decision in [Decision.REDACT, Decision.TRANSFORM]:
                sanitized_text = result.sanitized_text

        return DetectorResult(
            decision=highest_decision,
            risk_score=max_risk_score,
            rule_hits=all_rule_hits,
            sanitized_text=sanitized_text,
            user_message=user_message,
            developer_message=developer_message
        )
