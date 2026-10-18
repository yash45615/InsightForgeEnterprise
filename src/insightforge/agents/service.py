from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from insightforge.agents.retrieval_tool import KnowledgeTool
from insightforge.agents.tools import (
    get_business_overview,
    get_campaign_metrics,
    get_feedback_summary,
)
from insightforge.evaluation.safety import inspect_user_input

logger = logging.getLogger(__name__)

ALLOWED_TOPICS = {
    "overview",
    "campaigns",
    "feedback",
    "campaign_guidance",
    "sentiment_guidance",
}


class InvestigationService:
    """Investigate approved business topics using safe, predefined tools."""

    ALLOWED_TOPICS = ALLOWED_TOPICS

    def __init__(self) -> None:
        self.knowledge = KnowledgeTool()

    def investigate(self, topic: str, db: Session) -> dict[str, object]:
        """Run a business investigation or retrieve relevant knowledge."""
        normalized_topic = topic.strip().lower()

        safety_result = inspect_user_input(topic)
        safety_flags = (
            safety_result.get("flags", []) if isinstance(safety_result, dict) else []
        )

        is_safe = (
            safety_result.get("safe", True)
            and safety_result.get("allowed", True)
            and not safety_flags
            if isinstance(safety_result, dict)
            else not safety_flags
        )

        if not is_safe:
            flag_text = ", ".join(str(flag) for flag in safety_flags)
            if not flag_text:
                flag_text = "unsafe_input"

            logger.warning("agent_request_rejected reason=unsafe_input")
            raise ValueError(f"Request rejected by the input safety check: {flag_text}")

        if normalized_topic not in self.ALLOWED_TOPICS:
            logger.warning("agent_request_rejected reason=unsupported_topic")
            raise ValueError(
                "Unsupported topic. Choose overview, campaigns, feedback, "
                "campaign_guidance, or sentiment_guidance."
            )

        trace: list[dict[str, object]] = []
        evidence: dict[str, object] = {}

        if normalized_topic == "overview":

            def operation() -> Any:
                return get_business_overview(db)

            tool_name = "business_overview"
            evidence_key = "business_overview"

        elif normalized_topic == "campaigns":

            def operation() -> Any:
                return get_campaign_metrics(db)

            tool_name = "campaign_metrics"
            evidence_key = "campaign_metrics"

        elif normalized_topic == "feedback":

            def operation() -> Any:
                return get_feedback_summary(db)

            tool_name = "feedback_summary"
            evidence_key = "feedback_summary"

        else:
            if normalized_topic == "campaign_guidance":
                query = "campaign metrics and how to interpret CTR conversion rate"
            else:
                query = "sentiment evaluation macro F1 confusion matrix"

            def operation() -> Any:
                return self.knowledge.search(query, top_k=3)

            tool_name = "knowledge_answer"
            evidence_key = "knowledge"

        succeeded, result = self._execute_tool(
            tool_name=tool_name,
            operation=operation,
            trace=trace,
        )

        if not succeeded:
            return {
                "topic": normalized_topic,
                "status": "failed",
                "summary": "The investigation could not be completed.",
                "evidence": evidence,
                "trace": trace,
                "safety": safety_result,
                "limitations": [
                    "Tool execution failed; internal error details are withheld.",
                    "Only predefined, allowlisted tools are executed.",
                    "Results depend on the available data and local knowledge base.",
                ],
            }

        if not isinstance(result, dict):
            return {
                "topic": normalized_topic,
                "status": "failed",
                "summary": "The investigation returned an unexpected result.",
                "evidence": evidence,
                "trace": trace,
                "safety": safety_result,
                "limitations": [
                    "The tool did not return a structured result.",
                ],
            }

        if tool_name == "knowledge_answer":
            knowledge_status = str(result.get("status", "unknown"))

            if trace:
                trace[-1]["tool"] = "knowledge_answer"
                trace[-1]["status"] = knowledge_status

            return {
                "topic": normalized_topic,
                "status": knowledge_status,
                "query": result.get("query", ""),
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "evidence": {"knowledge": result},
                "trace": trace,
                "safety": safety_result,
                "limitations": [
                    "Knowledge answers depend on retrieved local documentation.",
                    "Citation matching does not independently prove that every "
                    "answer claim is supported.",
                    "The investigation uses predefined routing rather than "
                    "autonomous LLM planning.",
                ],
            }

        evidence[evidence_key] = result

        return {
            "topic": normalized_topic,
            "status": "completed",
            "summary": self._build_summary(normalized_topic, result),
            "evidence": evidence,
            "trace": trace,
            "safety": safety_result,
            "limitations": [
                "The investigation uses predefined, allowlisted tools.",
                "Results depend on the available synthetic dataset.",
                "Descriptive metrics do not establish causal impact.",
            ],
        }

    def _execute_tool(
        self,
        tool_name: str,
        operation: Callable[[], Any],
        trace: list[dict[str, object]],
    ) -> tuple[bool, Any]:
        """Execute an approved tool with timing and safe structured logs."""
        started = perf_counter()

        try:
            result = operation()
        except Exception as exc:
            duration_ms = round((perf_counter() - started) * 1000, 2)

            trace.append(
                {
                    "tool": tool_name,
                    "status": "failed",
                    "duration_ms": duration_ms,
                    "error": "Tool execution failed. Internal details withheld.",
                }
            )

            logger.error(
                "agent_tool_failed tool=%s status=failed "
                "duration_ms=%.2f error_type=%s",
                tool_name,
                duration_ms,
                type(exc).__name__,
            )
            return False, None

        duration_ms = round((perf_counter() - started) * 1000, 2)

        trace.append(
            {
                "tool": tool_name,
                "status": "success",
                "duration_ms": duration_ms,
            }
        )

        logger.info(
            "agent_tool_completed tool=%s status=success duration_ms=%.2f",
            tool_name,
            duration_ms,
        )

        return True, result

    @staticmethod
    def _build_summary(topic: str, result: dict[str, Any]) -> str:
        """Build a short summary from business tool output."""
        if topic == "overview":
            customers = result.get("total_customers", "unknown")
            transactions = result.get("total_transactions", "unknown")
            revenue = result.get("completed_revenue", "unknown")

            return (
                "Business overview retrieved. "
                f"Customers: {customers}; transactions: {transactions}; "
                f"completed revenue: {revenue}."
            )

        if topic == "campaigns":
            campaign_count = result.get("campaign_count", "unknown")

            return (
                "Campaign metrics retrieved. "
                f"Campaign count: {campaign_count}. "
                "Review the evidence for engagement and conversion metrics."
            )

        if topic == "feedback":
            feedback_count = result.get("total_feedback", "unknown")

            return (
                "Feedback summary retrieved. "
                f"Feedback records: {feedback_count}. "
                "Review the evidence for sentiment and rating distributions."
            )

        return "The investigation completed."
