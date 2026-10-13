"""Visionvolve fork: credentials never leak into output, logs or exception text,
and federation material is deleted after the run."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import anthropic
import httpx
import pytest
import requests

from claudecode import workload_identity
from claudecode.workload_identity import redact_exception, register_secret, scrub

SENTINEL_JWT = "eyJSENTINEL-JWT.c2VudGluZWw.SENTINELsig_0123456789"
SENTINEL_TOKEN = "sk-ant-oat01-SENTINEL-ACCESS-TOKEN-0123456789"

CRED_VARS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID",
    "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_PROFILE", "CLAUDECODE_MASK_FD",
    "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ENABLE_CLAUDE_FILTERING",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    workload_identity._active = None
    workload_identity._secrets.clear()
    yield
    workload_identity.cleanup()
    workload_identity._secrets.clear()


def _oidc_get(url, params=None, headers=None, timeout=None):
    return Mock(status_code=200, json=lambda: {"value": SENTINEL_JWT})


def _api_error(cls, status, sentinel):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages",
                            headers={"Authorization": f"Bearer {sentinel}"},
                            content=json.dumps({"assertion": sentinel}).encode())
    body = {"type": "error", "error": {"type": "authentication_error", "message": f"echo {sentinel}"}}
    response = httpx.Response(status, request=request, json=body,
                              headers={"request-id": "req_test123"})
    return cls(f"Error code: {status} - {body} {sentinel}", response=response, body=body)


class TestRedaction:
    def test_anthropic_status_error_keeps_only_status_type_request_id(self):
        e = _api_error(anthropic.AuthenticationError, 401, SENTINEL_TOKEN)
        text = redact_exception(e)
        assert SENTINEL_TOKEN not in text
        assert "HTTP 401" in text and "type=authentication_error" in text
        assert "echo" not in text  # no server message, no request body

    def test_requests_http_error(self):
        resp = Mock(status_code=403)
        e = requests.HTTPError(f"403 for url https://x?token={SENTINEL_JWT}", response=resp)
        text = redact_exception(e)
        assert SENTINEL_JWT not in text and text == "HTTPError: HTTP 403"

    def test_generic_exception_is_type_name_only(self):
        assert redact_exception(RuntimeError(f"boom {SENTINEL_JWT}")) == "RuntimeError"

    def test_own_exception_message_is_scrubbed(self):
        register_secret(SENTINEL_JWT)
        e = workload_identity.WorkloadIdentityError(f"oops {SENTINEL_JWT}")
        assert SENTINEL_JWT not in redact_exception(e)

    def test_scrub_covers_registered_and_env_secrets(self, monkeypatch):
        register_secret(SENTINEL_JWT)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SENTINEL_TOKEN)
        out = scrub(f"a {SENTINEL_JWT} b {SENTINEL_TOKEN} c")
        assert SENTINEL_JWT not in out and SENTINEL_TOKEN not in out


class TestMasking:
    def test_jwt_masked_on_mask_fd_immediately(self, monkeypatch):
        r, w = os.pipe()
        monkeypatch.setenv("CLAUDECODE_MASK_FD", str(w))
        register_secret(SENTINEL_JWT)
        os.close(w)
        with os.fdopen(r) as f:
            assert f.read() == f"::add-mask::{SENTINEL_JWT}\n"


def _federation_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_test")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "00000000-0000-0000-0000-000000000000")
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_test")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.example/token?x=1")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runner-request-token-0123456789")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("PR_NUMBER", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_test_token_0123456789")
    monkeypatch.setenv("REPO_PATH", str(tmp_path))
    gh_out = tmp_path / "gh_output"
    gh_sum = tmp_path / "gh_summary"
    gh_out.write_text("")
    gh_sum.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh_sum))
    return gh_out, gh_sum


def _github_client():
    client = MagicMock()
    client.get_pr_data.return_value = {
        "number": 1, "title": "t", "body": "", "user": "u", "created_at": "", "updated_at": "",
        "state": "open", "head": {"ref": "h", "sha": "s", "repo": {"full_name": "o/r"}},
        "base": {"ref": "b", "sha": "s"}, "files": [], "additions": 0, "deletions": 0,
        "changed_files": 0}
    client.get_pr_diff.return_value = ""
    client._is_excluded.return_value = False
    return client


def _run_main(capsys):
    from claudecode import github_action_audit
    with pytest.raises(SystemExit) as exc:
        github_action_audit.main()
    out, err = capsys.readouterr()
    return exc.value.code, out, err


def _assert_clean(texts, runner_temp):
    for name, text in texts.items():
        assert SENTINEL_JWT not in text, f"JWT leaked into {name}"
        assert SENTINEL_TOKEN not in text, f"access token leaked into {name}"
    assert not (Path(runner_temp) / "claude-workload-identity").exists(), "federation files left behind"
    for var in ("ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_PROFILE"):
        assert var not in os.environ


class TestEndToEndNoLeak:
    def test_cli_error_echoing_secrets_is_scrubbed_and_files_deleted(
            self, monkeypatch, tmp_path, capsys, caplog):
        caplog.set_level(logging.DEBUG)
        gh_out, gh_sum = _federation_env(monkeypatch, tmp_path)
        seen_files = []

        def fake_run(cmd, **kw):
            if cmd[:2] == ["claude", "--version"]:
                return Mock(returncode=0, stdout="2.1.296", stderr="")
            env = kw["env"]
            assert "ACTIONS_ID_TOKEN_REQUEST_TOKEN" not in env
            seen_files.append(env["ANTHROPIC_IDENTITY_TOKEN_FILE"])
            assert Path(env["ANTHROPIC_IDENTITY_TOKEN_FILE"]).read_text() == SENTINEL_JWT
            # A misbehaving CLI that echoes both credentials into its error output.
            return Mock(returncode=1, stderr=f"auth failed {SENTINEL_JWT}",
                        stdout=json.dumps({"type": "result", "is_error": True,
                                           "result": f"401 {SENTINEL_TOKEN}"}))

        r, w = os.pipe()
        monkeypatch.setenv("CLAUDECODE_MASK_FD", str(w))
        with patch("claudecode.workload_identity.requests.get", _oidc_get), \
                patch("claudecode.github_action_audit.GitHubActionClient", return_value=_github_client()), \
                patch("subprocess.run", side_effect=fake_run), \
                patch("time.sleep"):
            workload_identity.register_secret(SENTINEL_TOKEN)  # as if the CLI's token were known
            code, out, err = _run_main(capsys)
        os.close(w)
        with os.fdopen(r) as f:
            masks = f.read()

        assert code != 0
        assert json.loads(out)["error"]  # failure is reported, with secrets replaced
        assert "***" in out
        assert seen_files and not Path(seen_files[0]).exists()
        assert f"::add-mask::{SENTINEL_JWT}" in masks  # the only place the JWT may appear
        _assert_clean({"stdout": out, "stderr": err, "logs": caplog.text,
                       "GITHUB_OUTPUT": gh_out.read_text(),
                       "GITHUB_STEP_SUMMARY": gh_sum.read_text()}, os.environ["RUNNER_TEMP"])

    def test_sdk_auth_error_is_redacted_and_files_deleted(self, monkeypatch, tmp_path, capsys, caplog):
        caplog.set_level(logging.DEBUG)
        gh_out, gh_sum = _federation_env(monkeypatch, tmp_path)
        monkeypatch.setenv("ENABLE_CLAUDE_FILTERING", "true")
        sdk = MagicMock()
        sdk.messages.create.side_effect = _api_error(anthropic.AuthenticationError, 401, SENTINEL_TOKEN)
        with patch("claudecode.workload_identity.requests.get", _oidc_get), \
                patch("claudecode.claude_api_client.Anthropic", return_value=sdk), \
                patch("claudecode.github_action_audit.GitHubActionClient", return_value=_github_client()):
            code, out, err = _run_main(capsys)
        assert code != 0
        error = json.loads(out)["error"]
        assert "HTTP 401" in error and "authentication_error" in error
        _assert_clean({"stdout": out, "stderr": err, "logs": caplog.text,
                       "GITHUB_OUTPUT": gh_out.read_text(),
                       "GITHUB_STEP_SUMMARY": gh_sum.read_text()}, os.environ["RUNNER_TEMP"])

    def test_oidc_fetch_failure_does_not_leak_request_token(self, monkeypatch, tmp_path, capsys, caplog):
        _federation_env(monkeypatch, tmp_path)

        def failing_get(url, params=None, headers=None, timeout=None):
            raise requests.ConnectionError(f"failed {headers['Authorization']} {SENTINEL_JWT}")

        with patch("claudecode.workload_identity.requests.get", failing_get):
            code, out, err = _run_main(capsys)
        assert code != 0
        assert "runner-request-token" not in out + err + caplog.text
        _assert_clean({"stdout": out, "stderr": err, "logs": caplog.text}, os.environ["RUNNER_TEMP"])
