# src/utils/helpers.py

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file with UTF-8 encoding."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env(key: str, default: str = None) -> str:
    """Safely retrieve environment variable."""
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


def ensure_dirs(config: dict) -> None:
    """Create necessary data directories if they don't exist."""
    for key, path in config.get("paths", {}).items():
        Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Basic text cleaning for legal documents."""
    import re
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text