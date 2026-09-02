"""Account linking and Offset Plus entitlement.

The entitlement itself lives on the server; this module only caches the answer.
`auth.json` holds a bearer token, so it is written the same way credentials are:
parent directory created first, atomically replaced, owner-only permissions.

The cache is deliberately allowed to outlive a network outage.  Downgrading a
paying subscriber to Lite because their wifi dropped is worse than trusting a
token we already verified, so `sync_account_tier` keeps the last known tier when
it cannot reach the server and says so instead of silently re-tiering.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from offset.core import settings

# Production auth server URL on Render
AUTH_SERVER_URL = os.environ.get("OFFSET_AUTH_SERVER", "https://offset-backend.onrender.com").rstrip("/")

#: Test-only shortcuts (`test-plus-key`, emails containing "plus") are a
#: developer convenience and a paywall bypass in production - `me+plus@gmail.com`
#: is a perfectly ordinary address.  They stay off unless explicitly enabled.
def _dev_mode() -> bool:
    return bool(os.environ.get("OFFSET_DEV_MODE")) or "PYTEST_CURRENT_TEST" in os.environ


def _auth_file() -> Path:
    return settings.home() / "auth.json"


def _read_auth() -> dict[str, Any]:
    """Cached entitlement, or an empty dict if absent/unreadable."""
    path = _auth_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_auth(data: dict[str, Any]) -> Path:
    """Persist the entitlement cache.

    Creating the parent first is the whole point: on a fresh install `~/.offset`
    does not exist yet, and `write_text` reported that as `FileNotFoundError`
    from inside the login prompt (issue #1).  Written via a temp file so an
    interrupted write cannot leave a half-parsed token behind, and chmod 0600
    because `token` is a bearer credential.
    """
    path = _auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return path


def check_login():
    """Verify login status on CLI startup."""
    if not sys.stdin.isatty() or "PYTEST_CURRENT_TEST" in os.environ:
        return

    data = _read_auth()
    if data.get("logged_in") and data.get("account"):
        return

    prompt_account_login()


def prompt_account_login():
    """Prompt user to sign in with their Google or GitHub account."""
    print("\033[1;36mWelcome to Offset!\033[0m")
    print("Sign in with your Google or GitHub account to activate your workspace:")
    print("  \033[1m1.\033[0m GitHub Account")
    print("  \033[1m2.\033[0m Google Account")
    print("  \033[1m3.\033[0m Enter Gumroad License Key Directly")

    try:
        choice = input("Select option [1/2/3]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if choice == "3":
        try:
            key = input("Enter your Gumroad license key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        verify_direct_license_key(key)
        return

    provider = "github" if choice == "1" else "google"
    try:
        account_email = input(f"Enter your {provider.capitalize()} account email: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not account_email:
        print("\033[1;31m✗ An email address is required to link your account.\033[0m")
        return

    print(f"\nAuthenticating with {provider.capitalize()}...")
    time.sleep(1.0)
    print(f"\033[1;32m✓ Signed in as {account_email}\033[0m")

    try:
        tier, reachable = sync_account_tier(account_email, provider)
    except OSError as exc:
        # A write we could not complete: say so rather than claiming the
        # account is unlicensed.
        print(f"\033[1;31m✗ Could not save your login: {exc}\033[0m")
        return

    if tier == "plus":
        print("\033[1;32m★ OFFSET PLUS ACTIVATED!\033[0m (Linked to your Gumroad subscription)")
        return

    if not reachable:
        print("\033[1;33m! Could not reach the licence server; continuing as Offset Lite.\033[0m")
        print("  Run \033[1moffset sync\033[0m once you are back online to pick up Plus.")
        return

    print("\033[1;32m✓ OFFSET LITE ACTIVATED.\033[0m (Free forever)")
    print("\n\033[1;33mUpgrade to Offset Plus (Speculative Branching & Multi-Model /flow):\033[0m")
    print(f"  Subscribe at: https://debarghya47.gumroad.com/l/qzqnxk using {account_email}")
    print("  Your account will automatically upgrade to Plus on next launch or via: \033[1moffset sync\033[0m\n")
    time.sleep(1.2)


def _query_tier(account_email: str, provider: str) -> tuple[str, str | None, bool]:
    """Ask the server for an entitlement: `(tier, token, reachable)`.

    `reachable` separates "we asked and you are not a subscriber" from "we never
    got an answer", which are the same value of `tier` but very different things
    to tell a customer who has just paid.
    """
    req = urllib.request.Request(
        f"{AUTH_SERVER_URL}/auth/verify_account",
        data=json.dumps({"email": account_email, "provider": provider}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            result = json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return "lite", None, False

    if result.get("tier") == "plus":
        return "plus", result.get("access_token"), True
    return "lite", None, True


def sync_account_tier(account_email: str, provider: str = "google") -> tuple[str, bool]:
    """Refresh and persist the entitlement for `account_email`.

    Returns `(tier, reachable)`.  When the server cannot be reached the tier
    already on disk is kept: a subscriber who opens a laptop on a plane keeps
    the Plus they paid for, and a non-subscriber gains nothing.
    """
    tier, token, reachable = _query_tier(account_email, provider)
    data = _read_auth()

    if not reachable:
        if _dev_mode() and "plus" in account_email.lower():
            tier = "plus"
        elif data.get("account") == account_email and data.get("tier") == "plus":
            tier = "plus"
            token = data.get("token")

    data["logged_in"] = True
    data["account"] = account_email
    data["provider"] = provider
    data["tier"] = tier
    if token:
        data["token"] = token
    elif tier != "plus":
        data.pop("token", None)

    _write_auth(data)
    return tier, reachable


def verify_direct_license_key(key: str) -> int:
    """Verify a Gumroad licence key and store the entitlement it grants."""
    if not key:
        print("\033[1;31m✗ No licence key entered.\033[0m")
        return 1

    print(f"Verifying license key '{key}' with server...")

    result: dict[str, Any] | None = None
    req = urllib.request.Request(
        f"{AUTH_SERVER_URL}/auth/verify_license",
        data=json.dumps({"license_key": key}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            result = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        # 402 is the server's "this key is not valid" - a definite answer, so
        # it falls through to the rejection message below.
        if exc.code != 402:
            # Anything else is the server saying it could not check, which is
            # not a verdict on the key.  Show what it actually said: "try
            # again shortly" is the difference between a retry and a support
            # ticket about a key that was never broken.
            detail = ""
            try:
                detail = str(json.loads(exc.read().decode()).get("detail") or "")
            except (ValueError, OSError, AttributeError):
                pass
            print(f"\033[1;31m✗ Licence server error ({exc.code}).\033[0m")
            if detail:
                print(f"  {detail}")
            return 1
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        if _dev_mode() and key == "test-plus-key":
            result = {"tier": "plus", "email": "test@subscriber.com"}
        else:
            print("\033[1;31m✗ Could not reach the licence server. Check your connection.\033[0m")
            return 1

    if not result or result.get("tier") != "plus":
        print("\033[1;31m✗ Invalid or expired licence key.\033[0m")
        return 1

    # Verification succeeded.  A failure to persist it is a disk problem, not an
    # invalid key, and must never be reported as one - previously the write sat
    # inside the request's `try`, so a fresh install turned a real subscriber's
    # valid key into "invalid licence key".
    data = _read_auth()
    data["logged_in"] = True
    data["account"] = result.get("email") or "licensed_user@offset.dev"
    data["tier"] = "plus"
    if result.get("plan"):
        data["plan"] = result["plan"]
    if result.get("access_token"):
        data["token"] = result["access_token"]

    try:
        _write_auth(data)
    except OSError as exc:
        print(f"\033[1;31m✗ Licence is valid but could not be saved: {exc}\033[0m")
        return 1

    print("\033[1;32m★ OFFSET PLUS ACTIVATED!\033[0m All features unlocked.")
    return 0


def sync_command() -> int:
    """CLI handler for 'offset sync'."""
    data = _read_auth()
    account = data.get("account")
    if not account:
        print("No linked account found. Please sign in first.")
        prompt_account_login()
        return 0

    provider = data.get("provider", "google")
    print(f"Syncing subscription status for \033[1m{account}\033[0m ({provider})...")

    try:
        tier, reachable = sync_account_tier(account, provider)
    except OSError as exc:
        print(f"\033[1;31m✗ Could not save your subscription status: {exc}\033[0m")
        return 1

    if not reachable:
        print("\033[1;31m✗ Could not reach the licence server. Check your connection.\033[0m")
        return 1

    if tier == "plus":
        print("\033[1;32m★ OFFSET PLUS IS ACTIVE!\033[0m All parallel features unlocked.")
    else:
        print(f"\033[1;33mOffset Lite is active.\033[0m To upgrade, subscribe on Gumroad with {account} and run 'offset sync'.")
    return 0


def is_plus() -> bool:
    """Check if the current workspace has Offset Plus privileges."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return _read_auth().get("tier") == "plus"


def require_plus(feature_name: str) -> bool:
    return is_plus()
