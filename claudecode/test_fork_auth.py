"""Tests for the Visionvolve fork: keyless auth + fail-closed behaviour."""

import json
import os
import stat
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from claudecode import workload_identity
from claudecode.claude_api_client import ClaudeAPIClient
from claudecode.findings_filter import ClaudeAPIError, FindingsFilter
from claudecode.github_action_audit import SimpleClaudeRunner

CRED_VARS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID",
    "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_PROFILE",
    "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    # Keep the SDK away from any real ~/.config/anthropic on the test machine.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    workload_identity._active = None
    yield
    if workload_identity._active is not None:
        workload_identity._active.stop()
    workload_identity._active = None


def _federation_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_test")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "00000000-0000-0000-0000-000000000000")
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.example/token?x=1")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runner-request-token")


class _OIDC:
    """Fake GitHub OIDC endpoint returning a new JWT on each call."""

    def __init__(self):
        self.calls = 0

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        assert params == {"audience": "https://api.anthropic.com"}
        assert headers["Authorization"] == "bearer runner-request-token"
        return Mock(status_code=200, json=lambda: {"value": f"jwt-{self.calls}"})


class TestAuthModes:
    def test_auth_token_uses_bearer_not_x_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-test")
        client = ClaudeAPIClient()
        assert client.auth_mode == "bearer-token"
        headers = client.client.auth_headers
        assert headers.get("Authorization") == "Bearer sk-ant-oat01-test"
        assert "X-Api-Key" not in headers and "x-api-key" not in headers

    def test_api_key_path_unchanged(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
        client = ClaudeAPIClient()
        assert client.auth_mode == "api-key"

    def test_no_credentials_raises(self):
        with pytest.raises(ValueError, match="No Anthropic credentials"):
            ClaudeAPIClient()

    def test_describe_auth_mode(self, monkeypatch):
        assert workload_identity.describe_auth_mode() == "none"
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
        assert workload_identity.describe_auth_mode() == "bearer-token"


class TestWorkloadIdentity:
    def test_setup_writes_token_profile_and_env(self, monkeypatch):
        _federation_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # empty would shadow federation
        oidc = _OIDC()
        with patch("claudecode.workload_identity.requests.get", oidc):
            wi = workload_identity.setup_from_env()
        assert "ANTHROPIC_API_KEY" not in os.environ
        token_file = Path(os.environ["ANTHROPIC_IDENTITY_TOKEN_FILE"])
        assert token_file.read_text() == "jwt-1"
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        profile = Path(os.environ["ANTHROPIC_CONFIG_DIR"]) / "configs" / "default.json"
        assert json.loads(profile.read_text())["authentication"]["type"] == "oidc_federation"
        assert os.environ["ANTHROPIC_PROFILE"] == "default"
        assert workload_identity.describe_auth_mode() == "workload-identity-federation"
        with patch("claudecode.workload_identity.requests.get", oidc):
            workload_identity.refresh_if_active()
        assert token_file.read_text() == "jwt-2"
        wi.stop()
        assert not token_file.exists()
        workload_identity._active = None

    def test_static_key_alongside_federation_is_rejected(self, monkeypatch):
        _federation_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-real")
        with patch("claudecode.workload_identity.requests.get", _OIDC()):
            with pytest.raises(workload_identity.WorkloadIdentityError, match="shadow"):
                workload_identity.setup_from_env()

    def test_missing_oidc_fails(self, monkeypatch):
        _federation_env(monkeypatch)
        monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_URL")
        with pytest.raises(workload_identity.WorkloadIdentityError, match="id-token: write"):
            workload_identity.setup_from_env()

    def test_sdk_exchanges_jwt_and_sends_bearer(self, monkeypatch):
        """End to end through the real SDK credential chain with a mock transport."""
        _federation_env(monkeypatch)
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path == "/v1/oauth/token":
                body = json.loads(request.content)
                assert body["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
                assert body["federation_rule_id"] == "fdrl_test"
                assert body["service_account_id"] == "svac_test"
                assert body["workspace_id"] == "wrkspc_test"
                assert body["assertion"].startswith("jwt-")
                return httpx.Response(200, json={
                    "access_token": "sk-ant-oat01-minted", "token_type": "Bearer",
                    "expires_in": 600, "scope": "workspace:inference"})
            assert request.headers.get("authorization") == "Bearer sk-ant-oat01-minted"
            assert "x-api-key" not in request.headers
            return httpx.Response(200, json={
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client
        # Token exchange: if the SDK's credential provider builds its own client,
        # route that through the mock transport too (subclass keeps isinstance working).
        class _MockClient(real_client):
            def __init__(self, *a, **kw):
                kw["transport"] = transport
                super().__init__(*a, **kw)
        monkeypatch.setattr(httpx, "Client", _MockClient)
        # Messages: inject the mock transport into the Anthropic client itself.
        import anthropic
        from claudecode import claude_api_client as cac
        monkeypatch.setattr(cac, "Anthropic", lambda **kw: anthropic.Anthropic(
            http_client=_MockClient(), **kw))
        with patch("claudecode.workload_identity.requests.get", _OIDC()):
            client = ClaudeAPIClient(model="claude-test")
            assert client.auth_mode == "workload-identity-federation"
            ok, err = client.validate_api_access()
        assert ok, err
        paths = [r.url.path for r in seen]
        assert "/v1/oauth/token" in paths and "/v1/messages" in paths


class TestFailClosed:
    def _runner_result(self, stdout):
        return Mock(returncode=0, stdout=stdout, stderr="")

    @patch("subprocess.run")
    def test_cli_error_result_fails(self, mock_run, tmp_path):
        mock_run.return_value = self._runner_result(json.dumps({
            "type": "result", "subtype": "success", "is_error": True,
            "result": "Invalid API key · Please run /login"}))
        ok, err, _ = SimpleClaudeRunner().run_security_audit(tmp_path, "prompt")
        assert not ok and "Invalid API key" in err

    @patch("subprocess.run")
    def test_unparseable_review_fails(self, mock_run, tmp_path):
        mock_run.return_value = self._runner_result(json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "I could not complete the review."}))
        ok, err, _ = SimpleClaudeRunner().run_security_audit(tmp_path, "prompt")
        assert not ok and "no parseable security review" in err

    @patch("subprocess.run")
    def test_explicit_no_findings_succeeds(self, mock_run, tmp_path):
        mock_run.return_value = self._runner_result(json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": json.dumps({"findings": [], "analysis_summary": {"review_completed": True}})}))
        ok, err, res = SimpleClaudeRunner().run_security_audit(tmp_path, "prompt")
        assert ok and res["findings"] == []

    def test_filter_validation_failure_raises(self):
        with patch("claudecode.findings_filter.ClaudeAPIClient") as cls:
            cls.return_value.validate_api_access.return_value = (False, "401 authentication_error")
            with pytest.raises(ClaudeAPIError, match="401"):
                FindingsFilter(use_claude_filtering=True, api_key="x")

    def test_filter_per_finding_api_failure_raises(self):
        with patch("claudecode.findings_filter.ClaudeAPIClient") as cls:
            cls.return_value.validate_api_access.return_value = (True, "")
            cls.return_value.analyze_single_finding.return_value = (False, {}, "401 authentication_error")
            f = FindingsFilter(use_claude_filtering=True, api_key="x")
            finding = {"file": "a.py", "line": 1, "severity": "HIGH", "category": "sql_injection",
                       "description": "SQL injection via user input", "exploit_scenario": "x"}
            with pytest.raises(ClaudeAPIError, match="401"):
                f.filter_findings([finding])


class TestCliIsolation:
    @patch("subprocess.run")
    def test_cli_gets_fresh_jwt_and_no_oidc_request_creds(self, mock_run, monkeypatch, tmp_path):
        _federation_env(monkeypatch)
        oidc = _OIDC()
        with patch("claudecode.workload_identity.requests.get", oidc):
            workload_identity.setup_from_env()
            calls_after_setup = oidc.calls
            mock_run.return_value = Mock(returncode=0, stderr="", stdout=json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "result": json.dumps({"findings": []})}))
            ok, _, _ = SimpleClaudeRunner().run_security_audit(tmp_path, "prompt")
        assert ok
        assert oidc.calls == calls_after_setup + 1  # refreshed right before the CLI
        env = mock_run.call_args.kwargs["env"]
        assert "ACTIONS_ID_TOKEN_REQUEST_URL" not in env
        assert "ACTIONS_ID_TOKEN_REQUEST_TOKEN" not in env
        assert env["ANTHROPIC_IDENTITY_TOKEN_FILE"] == os.environ["ANTHROPIC_IDENTITY_TOKEN_FILE"]
