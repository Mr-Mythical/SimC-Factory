"""Load local AWS environment settings and saved dashboard credentials."""

import json
import os

from web.config import PROJECT_ROOT

_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
_CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, "web", ".credentials.json")

_ALLOWED_CREDENTIAL_KEYS = {
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region",
}


def load() -> dict[str, str]:
    """Return dashboard-saved AWS credentials, if present."""
    if not os.path.exists(_CREDENTIALS_FILE):
        return {}
    try:
        with open(_CREDENTIALS_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: str(value)
        for key, value in raw.items()
        if key in _ALLOWED_CREDENTIAL_KEYS and str(value).strip()
    }


def save(creds: dict[str, str]) -> None:
    """Persist dashboard AWS credentials locally, or clear them when empty."""
    filtered = {
        key: str(value).strip()
        for key, value in creds.items()
        if key in _ALLOWED_CREDENTIAL_KEYS and str(value).strip()
    }

    if not filtered:
        try:
            os.remove(_CREDENTIALS_FILE)
        except FileNotFoundError:
            pass
        return

    os.makedirs(os.path.dirname(_CREDENTIALS_FILE), exist_ok=True)
    with open(_CREDENTIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)
        f.write("\n")

    if os.name != "nt":
        try:
            os.chmod(_CREDENTIALS_FILE, 0o600)
        except OSError:
            pass


def get_aws_env() -> dict[str, str]:
    """Translate saved dashboard credentials into AWS environment variables."""
    creds = load()
    env: dict[str, str] = {}
    if creds.get("aws_access_key_id"):
        env["AWS_ACCESS_KEY_ID"] = creds["aws_access_key_id"]
    if creds.get("aws_secret_access_key"):
        env["AWS_SECRET_ACCESS_KEY"] = creds["aws_secret_access_key"]
    if creds.get("aws_session_token"):
        env["AWS_SESSION_TOKEN"] = creds["aws_session_token"]
    if creds.get("aws_region"):
        env["AWS_REGION"] = creds["aws_region"]
        env["AWS_DEFAULT_REGION"] = creds["aws_region"]
    return env


def load_dotenv() -> None:
    """Parse .env file and set any vars not already in os.environ."""
    if not os.path.exists(_ENV_FILE):
        return
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value

    _configure_signing_helper()


def _configure_signing_helper() -> None:
    """If Roles Anywhere env vars are set, configure AWS credential process.

    Sets AWS_SHARED_CREDENTIALS_FILE to a generated temp file that uses
    aws_signing_helper as credential_process, so boto3/AWS CLI in
    subprocesses automatically get temporary credentials.
    """
    signing_helper = os.environ.get("AWS_SIGNING_HELPER", "")
    trust_anchor_arn = os.environ.get("ROLES_ANYWHERE_TRUST_ANCHOR_ARN", "")
    profile_arn = os.environ.get("ROLES_ANYWHERE_PROFILE_ARN", "")
    role_arn = os.environ.get("ROLES_ANYWHERE_ROLE_ARN", "")
    cert_path = os.environ.get("ROLES_ANYWHERE_CERT", "")
    key_path = os.environ.get("ROLES_ANYWHERE_KEY", "")

    # All six must be set for Roles Anywhere to work
    if not all([signing_helper, trust_anchor_arn, profile_arn, role_arn, cert_path, key_path]):
        return

    # Resolve relative paths from project root
    if not os.path.isabs(cert_path):
        cert_path = os.path.join(PROJECT_ROOT, cert_path)
    if not os.path.isabs(key_path):
        key_path = os.path.join(PROJECT_ROOT, key_path)
    if not os.path.isabs(signing_helper):
        signing_helper = os.path.join(PROJECT_ROOT, signing_helper)

    # Build the credential_process command
    credential_process = (
        f'"{signing_helper}" credential-process'
        f" --trust-anchor-arn {trust_anchor_arn}"
        f" --profile-arn {profile_arn}"
        f" --role-arn {role_arn}"
        f' --certificate "{cert_path}"'
        f' --private-key "{key_path}"'
    )

    # Write an AWS config file that uses this credential process
    config_dir = os.path.join(PROJECT_ROOT, ".aws")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config")

    region = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")

    with open(config_path, "w") as f:
        f.write("[default]\n")
        f.write(f"region = {region}\n")
        f.write(f"credential_process = {credential_process}\n")

    # Point boto3/AWS CLI in subprocesses to this config
    os.environ["AWS_CONFIG_FILE"] = config_path
    # Clear any conflicting env vars so credential_process takes precedence
    for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        os.environ.pop(key, None)
