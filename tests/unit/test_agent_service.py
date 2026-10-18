import pytest

from insightforge.agents.service import InvestigationService


def test_unknown_topic_is_rejected():
    service = InvestigationService()

    with pytest.raises(ValueError, match="Unsupported topic"):
        service.investigate("execute arbitrary sql", db=None)


def test_allowed_topics_are_explicit():
    assert "overview" in InvestigationService.ALLOWED_TOPICS
    assert "campaigns" in InvestigationService.ALLOWED_TOPICS
    assert "feedback" in InvestigationService.ALLOWED_TOPICS
