"""Load the YAML config; every stage reads its settings from here instead of hard-coded paths."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    base_dir: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, p: str | os.PathLike) -> Path:
        """Resolve a path from the config relative to the config file's directory."""
        p = Path(os.path.expanduser(str(p)))
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def glob(self, patterns: list[str]) -> list[Path]:
        out: list[Path] = []
        for pattern in patterns:
            out.extend(Path(x) for x in sorted(glob.glob(str(self.path(pattern)))))
        return out

    @property
    def workdir(self) -> Path:
        d = self.path(self.raw.get("workdir", "local/runs/default"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def work(self, *parts: str) -> Path:
        """A path inside the workdir; parent directories are created."""
        p = self.workdir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def instruction(self) -> str:
        author = self.raw.get("author", {})
        template = author.get("instruction", "把下面这段文字改写成{name}的文风，内容和顺序不变。")
        return template.format(name=author.get("name", "作者"))


def load_config(path: str | os.PathLike) -> Config:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(raw=raw, base_dir=path.parent)
