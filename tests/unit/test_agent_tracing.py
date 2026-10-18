from unittest.mock import Mock

from insightforge.agents.service import InvestigationService


def test_execute_tool_records_success_and_duration() -> None:
    service = InvestigationService()
    trace: list[dict[str, object]] = []
    operation = Mock(return_value={"customers": 10})

    succeeded, result = service._execute_tool(
        "business_overview",
        operation,
        trace,
    )

    assert succeeded is True
    assert result == {"customers": 10}
    assert len(trace) == 1
    assert trace[0]["tool"] == "business_overview"
    assert trace[0]["status"] == "success"
    assert isinstance(trace[0]["duration_ms"], float)
    assert trace[0]["duration_ms"] >= 0
    operation.assert_called_once()


def test_execute_tool_handles_failure_without_exposing_exception() -> None:
    service = InvestigationService()
    trace: list[dict[str, object]] = []
    operation = Mock(side_effect=RuntimeError("sensitive database detail"))

    succeeded, result = service._execute_tool(
        "business_overview",
        operation,
        trace,
    )

    assert succeeded is False
    assert result is None
    assert len(trace) == 1
    assert trace[0]["tool"] == "business_overview"
    assert trace[0]["status"] == "failed"
    assert trace[0]["duration_ms"] >= 0
    assert "sensitive database detail" not in str(trace)


def test_execute_tool_records_each_call_separately() -> None:
    service = InvestigationService()
    trace: list[dict[str, object]] = []

    first_ok, first_result = service._execute_tool(
        "first_tool",
        lambda: "first",
        trace,
    )
    second_ok, second_result = service._execute_tool(
        "second_tool",
        lambda: "second",
        trace,
    )

    assert first_ok and second_ok
    assert first_result == "first"
    assert second_result == "second"
    assert [entry["tool"] for entry in trace] == [
        "first_tool",
        "second_tool",
    ]
    assert all(entry["status"] == "success" for entry in trace)


def test_successful_tool_execution_is_logged(caplog) -> None:
    import logging

    service = InvestigationService()
    trace: list[dict[str, object]] = []

    with caplog.at_level(
        logging.INFO,
        logger="insightforge.agents.service",
    ):
        succeeded, result = service._execute_tool(
            "campaign_metrics",
            lambda: {"campaigns": 3},
            trace,
        )

    assert succeeded is True
    assert result == {"campaigns": 3}
    assert "agent_tool_completed" in caplog.text
    assert "campaign_metrics" in caplog.text


def test_failed_tool_logs_exception_type_not_message(caplog) -> None:
    import logging

    service = InvestigationService()
    trace: list[dict[str, object]] = []

    with caplog.at_level(
        logging.ERROR,
        logger="insightforge.agents.service",
    ):
        succeeded, result = service._execute_tool(
            "business_overview",
            lambda: (_ for _ in ()).throw(RuntimeError("private database details")),
            trace,
        )

    assert succeeded is False
    assert result is None
    assert "agent_tool_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "private database details" not in caplog.text
