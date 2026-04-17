#!/usr/bin/env python3
"""
Logging setup module with Windows Unicode support.

Giải quyết UnicodeEncodeError trên Windows PowerShell:
- Tự động detect platform và encoding
- StreamHandler dùng UTF-8 trên Windows
- Safe character replacement cho special symbols
- Format chuẩn toàn project
"""

import io
import logging
import logging.handlers
import platform
import sys
from pathlib import Path
from typing import Optional


# Global logger cache để tránh duplicate
_loggers = {}


def _setup_stdout_encoding() -> None:
    """
    Fix stdout encoding trên Windows PowerShell.
    
    Vấn đề: Windows PowerShell default dùng cp1252, không support UTF-8 characters
    Giải pháp: Wrap stdout.buffer với UTF-8 encoding wrapper
    """
    if platform.system() == "Windows":
        try:
            # Nếu stdout không phải là text wrapper, wrap nó
            if not isinstance(sys.stdout, io.TextIOWrapper):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
            elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
                # Nếu đã là TextIOWrapper nhưng dùng encoding khác, replace
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
        except Exception as e:
            # Fallback nếu không thể wrap
            print(f"Warning: Could not set UTF-8 encoding: {e}", file=sys.stderr)


def _safe_replace_chars(message: str) -> str:
    """
    Thay thế special Unicode characters bằng ASCII fallback.
    
    Nếu terminal không support UTF-8, replace các ký tự đặc biệt:
    - ✓ → [OK]
    - ✗ → [FAIL]
    - → → ->
    - ≈ → ~
    - … → ...
    
    Args:
        message: Message cần replace
        
    Returns:
        Message với characters đã replace
    """
    replacements = {
        "✓": "[OK]",
        "✗": "[FAIL]",
        "→": "->",
        "≈": "~",
        "…": "...",
        "◆": "*",
        "■": "=",
        "●": "o",
        "◇": "o",
    }
    
    result = message
    for unicode_char, ascii_char in replacements.items():
        result = result.replace(unicode_char, ascii_char)
    
    return result


class UTF8LogFormatter(logging.Formatter):
    """
    Custom formatter với fallback cho UTF-8 characters.
    
    - Cố gắng thêm UTF-8 characters (✓, etc.)
    - Nếu lỗi, fallback sang ASCII
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record với safe Unicode handling."""
        try:
            result = super().format(record)
            # Test encoding to stdout
            result.encode(sys.stdout.encoding or "utf-8")
            return result
        except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
            # Fallback: replace special characters
            original = super().format(record)
            return _safe_replace_chars(original)


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup logger với file + console handlers, UTF-8 support cho Windows.
    
    Args:
        name: Logger name (thường là __name__)
        log_dir: Directory để lưu log files (nếu None, chỉ dùng console)
        level: Log level cho console
        file_level: Log level cho file (thường >= DEBUG)
        
    Returns:
        Configured logger instance
        
    Example:
        logger = setup_logger(__name__, log_dir=Path("logs"))
        logger.info("Test message")
    """
    # Check cache trước
    if name in _loggers:
        return _loggers[name]
    
    # Setup encoding trên Windows
    _setup_stdout_encoding()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = UTF8LogFormatter(fmt, date_fmt)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (nếu log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{name}.log"
        
        # File handler dùng UTF-8 encoding
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",  # ← Force UTF-8 cho file
            )
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(fmt, date_fmt)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    # Cache
    _loggers[name] = logger
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger từ cache hoặc create mới.
    
    Dùng hàm này để lấy logger trong các module khác.
    
    Args:
        name: Logger name (thường là __name__)
        
    Returns:
        Logger instance
        
    Example:
        from src.utils.logging_setup import get_logger
        
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    if name not in _loggers:
        # Auto-create logger nếu chưa tồn tại
        return setup_logger(name)
    
    return _loggers[name]


def safe_log(
    logger: logging.Logger,
    message: str,
    level: str = "info",
) -> None:
    """
    Log message với safe Unicode character handling.
    
    Thay thế special Unicode characters bằng ASCII fallback
    nếu cần thiết.
    
    Args:
        logger: Logger instance
        message: Message để log (có thể chứa UTF-8 characters)
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        
    Example:
        safe_log(logger, "✓ Training completed")
        safe_log(logger, "✗ Error occurred", level="error")
    """
    # Try log ngay
    try:
        log_func = getattr(logger, level.lower())
        log_func(message)
    except UnicodeEncodeError:
        # Fallback: replace characters
        safe_message = _safe_replace_chars(message)
        log_func = getattr(logger, level.lower())
        log_func(safe_message)


def log_section(
    logger: logging.Logger,
    title: str,
    length: int = 70,
) -> None:
    """
    Log một section header với dashes.
    
    Args:
        logger: Logger instance
        title: Section title
        length: Độ dài của line (default: 70)
        
    Example:
        log_section(logger, "MODEL TRAINING")
        # Output: ======================================================================
        #         MODEL TRAINING
        #         ======================================================================
    """
    dash = "=" * length
    safe_log(logger, dash)
    safe_log(logger, title.center(length))
    safe_log(logger, dash)


# ============================================================================
# Compat wrapper cho old-style print-to-log
# ============================================================================

class PrintToLog:
    """
    Context manager để redirect print() calls sang logger.
    
    Dùng khi có legacy code dùng print() thay vì logger.
    
    Example:
        logger = setup_logger(__name__)
        with PrintToLog(logger):
            print("This goes to logger")
    """
    
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self.old_stdout = None
    
    def __enter__(self):
        self.old_stdout = sys.stdout
        
        class LogWriter:
            def __init__(self, log, level):
                self.logger = log
                self.level = level
            
            def write(self, msg):
                if msg.strip():
                    self.logger.log(self.level, msg.strip())
            
            def flush(self):
                pass
        
        sys.stdout = LogWriter(self.logger, self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout


if __name__ == "__main__":
    # Test script
    print("Testing logging setup...")
    
    logger = setup_logger("test", log_dir=Path("logs"))
    
    logger.info("✓ UTF-8 logging works")
    logger.debug("→ Debug message")
    logger.warning("⚠ Warning message")
    logger.error("✗ Error message")
    
    safe_log(logger, "✓ Safe log function works")
    
    print("Test completed - check logs/ directory")
