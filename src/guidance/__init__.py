"""Guidance generators: turn structured occupancy into human-friendly directions."""

from .base import BaseGuidance, GuidanceResult
from .llm import OllamaGuidance
from .rule_based import RuleBasedGuidance

__all__ = ["BaseGuidance", "GuidanceResult", "RuleBasedGuidance", "OllamaGuidance"]
