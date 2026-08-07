"""Lightweight task tagging for model routing."""

from .embedding import HashEmbedder, SentenceTransformerEmbedder
from .data_cleaning import CleaningConfig, clean_records
from .model import TaskModel
from .schema import TrainingExample
from .tagger import TaskTagger
from .task_ir import TaskIR

__all__ = [
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "CleaningConfig",
    "clean_records",
    "TaskIR",
    "TaskModel",
    "TaskTagger",
    "TrainingExample",
]
__version__ = "0.2.0"
