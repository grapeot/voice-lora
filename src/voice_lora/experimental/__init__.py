"""Tried and not adopted. Kept so the results stay reproducible, not as options to pick.

- outline_regen (v2, section-level: an AI rewrites each section from an extracted outline). Scored higher than v1
  on its own held-out sections, but on real AI articles it invented quotes and details, dropped arguments and
  pushed the prose toward AI formatting; the classifier could not see any of it. See docs/v2_experiment.md.
  Using it needs `builder.allow_experimental: true` in the config.
"""
