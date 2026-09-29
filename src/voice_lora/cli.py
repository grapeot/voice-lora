"""Shared argument handling for the stage scripts."""
from __future__ import annotations

import argparse

from .config import Config, load_config


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="local/config.yaml", help="path to the YAML config (default: local/config.yaml)")
    return ap


def config_from(args: argparse.Namespace) -> Config:
    return load_config(args.config)
