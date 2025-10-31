"""Summarization module for MiniRAG."""

from minirag.summarization.bart_summarizer import (
    BARTSummarizer,
    TRANSFORMERS_AVAILABLE,
)

__all__ = ["BARTSummarizer", "TRANSFORMERS_AVAILABLE"]
