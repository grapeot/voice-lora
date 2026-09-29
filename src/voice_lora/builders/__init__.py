"""Dataset builders. Each one turns paragraph units into AI-side texts and then into SFT examples;
training, evaluation and serving are shared by all of them."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType

BUILDERS = {
    "paragraph_rewrite": "voice_lora.builders.paragraph_rewrite",
    "outline_regen": "voice_lora.builders.outline_regen",
}


def get_builder(name: str) -> ModuleType:
    if name not in BUILDERS:
        raise ValueError(f"unknown builder {name!r}; choose from {list(BUILDERS)}")
    return import_module(BUILDERS[name])
