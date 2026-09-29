"""Dataset builders. Each one turns paragraph units into AI-side texts and then into SFT examples;
training, evaluation and serving are shared by all of them."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType

BUILDERS = {
    "paragraph_rewrite": "voice_lora.builders.paragraph_rewrite",
}
# Tried and not adopted (see voice_lora/experimental/__init__.py); usable only with builder.allow_experimental.
EXPERIMENTAL = {
    "outline_regen": "voice_lora.experimental.outline_regen",
}


def get_builder(name: str, allow_experimental: bool = False) -> ModuleType:
    if name in EXPERIMENTAL:
        if not allow_experimental:
            raise ValueError(f"builder {name!r} is experimental and not recommended (see docs/v2_experiment.md); "
                             "set builder.allow_experimental: true to reproduce it anyway")
        return import_module(EXPERIMENTAL[name])
    if name not in BUILDERS:
        raise ValueError(f"unknown builder {name!r}; choose from {list(BUILDERS)}")
    return import_module(BUILDERS[name])
