# src/utils.py
# ============================================================
# Utility Functions & Logging Setup
# ============================================================
# This module provides:
#   1. A configured logger for the entire project
#   2. A timing decorator to measure function execution time
#   3. Safe file I/O helpers
#
# WHY centralized logging?
#   - Consistent format across all modules
#   - Easy to switch between DEBUG/INFO levels
#   - No print() statements anywhere in the project
# ============================================================

import logging
import time
import functools
from pathlib import Path
from typing import Any


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def get_logger(name: str) -> logging.Logger:
    """
    Create a configured logger for a module.

    Usage:
        from src.utils import get_logger
        logger = get_logger(__name__)
        logger.info("Loading documents...")

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Only add handler if the logger doesn't already have one
    # (prevents duplicate log lines when modules are imported multiple times)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# ============================================================
# TIMING DECORATOR
# ============================================================

def timer(func):
    """
    Decorator that logs the execution time of a function.

    Usage:
        @timer
        def slow_function():
            time.sleep(2)

    Output:
        "slow_function completed in 2.01s"
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapper


# ============================================================
# FILE I/O HELPERS
# ============================================================

def ensure_directory(path: Path) -> Path:
    """
    Create a directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_file(filepath: Path) -> str:
    """
    Safely read a text file and return its contents.

    Args:
        filepath: Path to the text file.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return filepath.read_text(encoding="utf-8").strip()
