"""v2 dataset builder (not implemented yet): structure-level voice.

Planned: cut each of the author's articles into sections (by heading, or windows of 3-6 paragraphs),
extract an outline plus a fact list from each section, have an AI write that section from the outline
alone, and train on (AI section -> original section). Unlike paragraph_rewrite, the AI side then owns the
paragraph structure and argument flow inside the section, which is what real AI-drafted articles look
like. See skills/voice-lora/references/builders.md.
"""
from __future__ import annotations


def make_jobs(*_args, **_kwargs):
    raise NotImplementedError("outline_regen is the planned v2 builder; see docs/rfc.md §v2")


def make_worker(*_args, **_kwargs):
    raise NotImplementedError("outline_regen is the planned v2 builder; see docs/rfc.md §v2")


def build(*_args, **_kwargs):
    raise NotImplementedError("outline_regen is the planned v2 builder; see docs/rfc.md §v2")
