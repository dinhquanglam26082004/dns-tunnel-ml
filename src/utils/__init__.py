"""
Utilities module for dns-tunnel-ml project.
"""

from .logging_setup import (
    PrintToLog,
    UTF8LogFormatter,
    get_logger,
    log_section,
    safe_log,
    setup_logger,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "safe_log",
    "log_section",
    "UTF8LogFormatter",
    "PrintToLog",
]
