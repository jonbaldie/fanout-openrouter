from fanout_openrouter.openrouter_client import OpenRouterError
from fanout_openrouter.orchestrator import (
    UpstreamClientError,
    _shared_upstream_client_error,
)


def test_shared_upstream_client_error_none_when_empty():
    assert _shared_upstream_client_error([]) is None


def test_shared_upstream_client_error_none_when_has_none():
    assert (
        _shared_upstream_client_error(
            [
                OpenRouterError(
                    "test",
                    status_code=400,
                    retryable=False,
                    upstream_message="msg",
                    upstream_code="code",
                ),
                None,
            ]
        )
        is None
    )


def test_shared_upstream_client_error_none_when_retryable():
    assert (
        _shared_upstream_client_error(
            [
                OpenRouterError(
                    "test",
                    status_code=400,
                    retryable=True,
                    upstream_message="msg",
                    upstream_code="code",
                )
            ]
        )
        is None
    )


def test_shared_upstream_client_error_none_when_500():
    assert (
        _shared_upstream_client_error(
            [
                OpenRouterError(
                    "test",
                    status_code=500,
                    retryable=False,
                    upstream_message="msg",
                    upstream_code="code",
                )
            ]
        )
        is None
    )


def test_shared_upstream_client_error_none_when_mismatched_status():
    assert (
        _shared_upstream_client_error(
            [
                OpenRouterError(
                    "test",
                    status_code=400,
                    retryable=False,
                    upstream_message="msg",
                    upstream_code="code",
                ),
                OpenRouterError(
                    "test",
                    status_code=401,
                    retryable=False,
                    upstream_message="msg",
                    upstream_code="code",
                ),
            ]
        )
        is None
    )


def test_shared_upstream_client_error_none_when_mismatched_message():
    assert (
        _shared_upstream_client_error(
            [
                OpenRouterError(
                    "test",
                    status_code=400,
                    retryable=False,
                    upstream_message="msg1",
                    upstream_code="code",
                ),
                OpenRouterError(
                    "test",
                    status_code=400,
                    retryable=False,
                    upstream_message="msg2",
                    upstream_code="code",
                ),
            ]
        )
        is None
    )


def test_shared_upstream_client_error_success():
    exc = _shared_upstream_client_error(
        [
            OpenRouterError(
                "test",
                status_code=400,
                retryable=False,
                upstream_message="msg",
                upstream_code="code",
            ),
            OpenRouterError(
                "test",
                status_code=400,
                retryable=False,
                upstream_message="msg",
                upstream_code="code",
            ),
        ]
    )
    assert isinstance(exc, UpstreamClientError)
    assert exc.status_code == 400
    assert exc.message == "msg"
    assert exc.code == "code"
