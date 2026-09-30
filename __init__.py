"""Ticket triage prototype: local Naive-Bayes model + public LLM API engine."""

from .labels import CATEGORIES, URGENCIES, ROUTING
from .model_local import TriageModel, train_local_model

__all__ = ["CATEGORIES", "URGENCIES", "ROUTING", "TriageModel", "train_local_model"]
__version__ = "0.1.0"
