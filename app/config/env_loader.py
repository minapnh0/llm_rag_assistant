"""
env_loader.py

Optional early-stage .env loader if you need to load env vars
before initializing Pydantic (e.g., for logger, cloud client).
"""

import os
from dotenv import load_dotenv
from pathlib import Path


def load_env():
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    print(f"Loaded environment from: {env_path}")