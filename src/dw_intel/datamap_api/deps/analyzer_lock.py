from functools import lru_cache
import os

from fastapi import Depends
from quack.datamap_api.configuration.api import ApiConfiguration, get_api_configuration
from quack.shared.analyser import DataMapSchemaAnalyzer
import threading

configuration = get_api_configuration()


@lru_cache(maxsize=1)
def get_old_analyser() -> DataMapSchemaAnalyzer:
    """Get the DataMapSchemaAnalyzer instance."""
    return DataMapSchemaAnalyzer(configuration)

# Initialize analyzer with thread safety
_analyzer = None
_analyzer_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_analyzer() -> DataMapSchemaAnalyzer:
    """Thread-safe getter for the analyzer instance."""
    with _analyzer_lock:
        return _analyzer

@lru_cache(maxsize=1)
def set_analyzer(new_analyzer: DataMapSchemaAnalyzer) -> None:
    """Thread-safe setter for the analyzer instance."""
    with _analyzer_lock:
        global _analyzer
        _analyzer = new_analyzer