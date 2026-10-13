"""Workload Identity Federation (keyless auth) for GitHub Actions.

Visionvolve fork addition. Mirrors anthropics/claude-code-action's
``base-action/src/workload-identity.ts``:

* fetch a GitHub Actions OIDC token (JWT) for the configured audience,
* write it to a 0600 file and point both the Claude Code CLI and the Python SDK
  at it via ``ANTHROPIC_IDENTITY_TOKEN_FILE``,
* write a minimal ``oidc_federation`` profile so the SDK/CLI share an on-disk
  credentials cache (the remaining fields are gap-filled from the
  ``ANTHROPIC_FEDERATION_RULE_ID`` / ``_ORGANIZATION_ID`` /
  ``_SERVICE_ACCOUNT_ID`` / ``_WORKSPACE_ID`` env vars),
* keep the JWT file fresh in the background so access tokens can be re-minted
  for runs longer than the access-token lifetime.

GitHub OIDC JWTs carry a ``jti``; Anthropic accepts each ``jti`` once. ``refresh()``
is called before every new consumer (SDK client, CLI subprocess) so two
consumers never exchange the same JWT.

The JWT and the exchanged access token are never printed.
"""

import atexit
import json
import os
import shutil
import threading
from typing import Optional

import requests

from claudecode.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OIDC_AUDIENCE = "https://api.anthropic.com"
REFRESH_INTERVAL_SECONDS = 4 * 60

# Credential env vars that take precedence over federation in the SDK/CLI chain.
# An exported-but-empty value still wins its precedence slot, so unset them.
SHADOWING_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


class WorkloadIdentityError(RuntimeError):
    """Raised when the GitHub OIDC token cannot be obtained. Messages never carry secrets."""


# --- Secret hygiene ---------------------------------------------------------
# Every credential value this process learns is registered here, so it can be
# scrubbed from any text that leaves the process (results JSON, logs).
_secrets: set = set()
_MASK_FD_ENV = "CLAUDECODE_MASK_FD"


def register_secret(value: str) -> None:
    """Remember a secret for scrub() and mask it in the GitHub Actions log.

    Defence in depth: the value is never printed. The step redirects stdout and
    stderr to files, so the ``::add-mask::`` command goes to a dedicated fd that
    action.yml connects to the step's real stdout (``CLAUDECODE_MASK_FD``).
    JWTs and Anthropic tokens are single-line base64url/ASCII, so masking the raw
    value also covers JSON-escaped forms (nothing in them needs escaping).
    """
    if not value:
        return
    _secrets.add(value)
    fd = os.environ.get(_MASK_FD_ENV, "")
    if fd.isdigit():
        try:
            os.write(int(fd), f"::add-mask::{value}\n".encode())
        except OSError:
            pass


def scrub(text: str) -> str:
    """Replace every registered secret (and env credentials) in ``text`` with ***."""
    if not text:
        return text
    values = set(_secrets)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(var, "")
        if len(v) >= 8:
            values.add(v)
    for v in sorted(values, key=len, reverse=True):
        text = text.replace(v, "***")
    return text


def redact_exception(e: BaseException) -> str:
    """A safe one-line description of an exception.

    Never ``str(e)`` for HTTP/SDK errors: those can embed request bodies or
    headers. Keeps the HTTP status, Anthropic's error type and request id only.
    """
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    if status is None:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
    if status is not None:
        err_type = ""
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            inner = body.get("error") if isinstance(body.get("error"), dict) else body
            err_type = str(inner.get("type", "")) if isinstance(inner, dict) else ""
        req_id = getattr(e, "request_id", None) or ""
        parts = [f"{name}: HTTP {status}"]
        if err_type:
            parts.append(f"type={err_type}")
        if req_id:
            parts.append(f"request_id={req_id}")
        return " ".join(parts)
    if type(e).__module__.startswith("claudecode"):
        # Our own exception types; their messages are built from redacted parts
        # and scrubbed again here.
        return scrub(f"{name}: {e}")[:2000]
    return name


def is_configured() -> bool:
    """Federation is requested when the rule and organization IDs are set."""
    return bool(
        os.environ.get("ANTHROPIC_FEDERATION_RULE_ID", "").strip()
        and os.environ.get("ANTHROPIC_ORGANIZATION_ID", "").strip()
    )


def _fetch_github_oidc_token(audience: str) -> str:
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
    bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
    if not url or not bearer:
        raise WorkloadIdentityError(
            "GitHub OIDC token is not available to this job "
            "(ACTIONS_ID_TOKEN_REQUEST_URL/_TOKEN unset). Add `id-token: write` "
            "to the job permissions; fork and Dependabot PRs never get one."
        )
    last_error = ""
    for _attempt in range(3):
        try:
            resp = requests.get(
                url,
                params={"audience": audience},
                headers={"Authorization": f"bearer {bearer}"},
                timeout=30,
            )
            if resp.status_code == 200:
                value = resp.json().get("value", "")
                if value:
                    return value
                last_error = "response had no token value"
            else:
                last_error = f"HTTP {resp.status_code}"
        except (requests.RequestException, ValueError) as e:
            last_error = type(e).__name__
    raise WorkloadIdentityError(f"Failed to fetch GitHub OIDC token: {last_error}")


class WorkloadIdentity:
    """Handle for an active federation setup."""

    def __init__(self, base_dir: str, audience: str = DEFAULT_OIDC_AUDIENCE):
        self.audience = audience
        self.token_dir = os.path.join(base_dir, "claude-workload-identity")
        self.token_file = os.path.join(self.token_dir, "identity-token")
        self.config_dir = os.path.join(self.token_dir, "config")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def refresh(self) -> None:
        """Write a fresh GitHub OIDC JWT (new ``jti``) to the token file."""
        token = _fetch_github_oidc_token(self.audience)
        register_secret(token)  # mask before anything else can happen with it
        with self._lock:
            if self._stop.is_set():
                return  # never recreate the file after cleanup
            os.makedirs(self.token_dir, mode=0o700, exist_ok=True)
            tmp = self.token_file + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(token)
            os.replace(tmp, self.token_file)

    def _write_profile(self) -> None:
        configs = os.path.join(self.config_dir, "configs")
        os.makedirs(configs, mode=0o700, exist_ok=True)
        path = os.path.join(configs, "default.json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"version": "1.0", "authentication": {"type": "oidc_federation"}}, f)

    def start(self) -> "WorkloadIdentity":
        for var in SHADOWING_VARS:
            if var in os.environ:
                # Never silently fall back to a static credential.
                if os.environ[var].strip():
                    raise WorkloadIdentityError(
                        f"{var} is set alongside the federation inputs; it would shadow "
                        "federation. Configure exactly one auth method."
                    )
                del os.environ[var]
        self.refresh()
        self._write_profile()
        os.environ["ANTHROPIC_IDENTITY_TOKEN_FILE"] = self.token_file
        os.environ["ANTHROPIC_CONFIG_DIR"] = self.config_dir
        os.environ["ANTHROPIC_PROFILE"] = "default"
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Workload identity federation configured (rule %s, workspace %s)",
            os.environ.get("ANTHROPIC_FEDERATION_RULE_ID"),
            os.environ.get("ANTHROPIC_WORKSPACE_ID") or "<rule default>",
        )
        return self

    def _refresh_loop(self) -> None:
        while not self._stop.wait(REFRESH_INTERVAL_SECONDS):
            try:
                self.refresh()
            except Exception as e:  # keep going; the next exchange will surface it
                logger.warning(f"Failed to refresh GitHub OIDC identity token: {redact_exception(e)}")

    def stop(self) -> None:
        """Stop refreshing and delete the JWT file, the profile and the SDK/CLI
        credentials cache (``config/credentials/``, which holds the exchanged
        access token). Idempotent."""
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        with self._lock:
            shutil.rmtree(self.token_dir, ignore_errors=True)
        for var in ("ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_PROFILE"):
            os.environ.pop(var, None)


_active: Optional[WorkloadIdentity] = None


def setup_from_env() -> Optional[WorkloadIdentity]:
    """Start federation if configured. Idempotent; returns the active handle."""
    global _active
    if _active is not None:
        return _active
    if not is_configured():
        return None
    base = os.environ.get("RUNNER_TEMP") or "/tmp"
    audience = os.environ.get("ANTHROPIC_OIDC_AUDIENCE", "").strip() or DEFAULT_OIDC_AUDIENCE
    handle = WorkloadIdentity(base, audience)
    _active = handle
    atexit.register(cleanup)  # fallback; callers also call cleanup() in finally
    try:
        handle.start()
    except BaseException:
        cleanup()
        raise
    return _active


def cleanup() -> None:
    """Delete all federation material. Safe to call more than once."""
    global _active
    handle, _active = _active, None
    if handle is not None:
        handle.stop()


def refresh_if_active() -> None:
    """Give the next consumer a JWT nobody has exchanged yet."""
    if _active is not None:
        _active.refresh()


# Note: the federated Anthropic access token never reaches this code. The SDK and
# the Claude Code CLI mint it internally and keep it in memory or in their cache
# under ANTHROPIC_CONFIG_DIR, which cleanup() deletes. So there is nothing to mask
# for it here; scrub() still covers ANTHROPIC_AUTH_TOKEN in bearer-token mode.

# The Claude Code CLI reads PR-controlled content. It needs the identity token
# file, not the runner's ability to mint new OIDC tokens for arbitrary audiences.
CHILD_ENV_DENYLIST = ("ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN", _MASK_FD_ENV)


def child_env() -> dict:
    """Environment for the Claude Code subprocess, without the OIDC request credentials."""
    return {k: v for k, v in os.environ.items() if k not in CHILD_ENV_DENYLIST}


def secret_dir_denies() -> list:
    """Claude Code --disallowed-tools rules that block reading the federation
    directory. Empty when federation is not active."""
    if _active is None:
        return []
    d = _active.token_dir.rstrip("/")
    # '//' prefix = absolute path in Claude Code permission rules.
    # Bash is not allow-listed for the CLI (non-interactive mode denies it), and
    # reads outside the working directory need permission it cannot get; these
    # rules make the federation directory explicitly off-limits on top of that.
    return [f"Read(/{d}/**)", f"Grep(/{d}/**)", f"Glob(/{d}/**)"]


def describe_auth_mode() -> str:
    """Which credential the SDK/CLI will use. Never returns a secret."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "api-key"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip():
        return "bearer-token"
    if is_configured() and os.environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE"):
        return "workload-identity-federation"
    return "none"
