# -*- coding: utf-8 -*-
"""Hypothesis profile configuration for hyperagents tests."""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=500,
)

settings.register_profile(
    "default",
    max_examples=200,
)

settings.load_profile("default")
