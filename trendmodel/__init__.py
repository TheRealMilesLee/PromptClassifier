"""Lightweight task tagging for model routing."""

from .embedding import HashEmbedder, SentenceTransformerEmbedder
from .data_cleaning import CleaningConfig, clean_records
from .model import TaskModel
from .schema import TrainingExample
from .tagger import TaskTagger
from .task_ast import TaskAST
from .task_ir import TaskIR

__all__ = [
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "CleaningConfig",
    "clean_records",
    "TaskIR",
    "TaskAST",
    "TaskModel",
    "TaskTagger",
    "TrainingExample",
]
__version__ = "0.5.0"
