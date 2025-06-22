from functools import lru_cache
import os
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import Depends
from dw_intel.datamap_api.configuration.api import ApiConfiguration, get_api_configuration
from dw_intel.shared.analyser import DataMapSchemaAnalyzer
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