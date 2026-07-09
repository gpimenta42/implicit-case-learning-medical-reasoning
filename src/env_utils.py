from __future__ import annotations

import os
from pathlib import Path


def _fallback_load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_file(path: Path) -> None:
    """Load environment variables from a .env file without overriding exports."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:
        _fallback_load_env_file(path)


def load_env(project_root: Path) -> None:
    for path in [project_root / ".env", Path.home() / ".env"]:
        load_env_file(path)


def load_env_from_cwd_and_home() -> None:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        load_env_file(candidate / ".env")
    load_env_file(Path.home() / ".env")
