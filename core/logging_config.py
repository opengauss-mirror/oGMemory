# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Structured logging configuration for ContextEngine.

Provides centralized logging setup with:
- Structured JSON output for production
- Colored console output for development
- Context-aware loggers (account_id, trace_id injection)
"""

import logging
import sys
from typing import Optional

from core.models import RequestContext


NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "pyagfs",
    "urllib3",
)


class ContextFilter(logging.Filter):
    """Inject RequestContext into log records."""

    def __init__(self, ctx: Optional[RequestContext] = None):
        super().__init__()
        self.ctx = ctx

    def filter(self, record):
        # Add context fields if available
        if hasattr(self, 'ctx') and self.ctx:
            record.account_id = getattr(self.ctx, 'account_id', 'N/A')
            record.user_id = getattr(self.ctx, 'user_id', 'N/A')
            record.agent_id = getattr(self.ctx, 'agent_id', 'N/A')
            record.trace_id = getattr(self.ctx, 'trace_id', 'N/A')
        else:
            record.account_id = 'N/A'
            record.user_id = 'N/A'
            record.agent_id = 'N/A'
            record.trace_id = 'N/A'
        return True


class ContextLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that injects context into every log message.

    Thread-safe alternative to adding filters directly to loggers.
    """

    def __init__(self, logger: logging.Logger, ctx: RequestContext):
        super().__init__(logger, {})
        self.ctx = ctx

    def process(self, msg, kwargs):
        # Add context to extra dict for formatters
        kwargs.setdefault('extra', {})
        kwargs['extra']['account_id'] = self.ctx.account_id
        kwargs['extra']['user_id'] = self.ctx.user_id
        kwargs['extra']['agent_id'] = self.ctx.agent_id
        kwargs['extra']['trace_id'] = self.ctx.trace_id
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None
) -> None:
    """Configure logging for ContextEngine.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output structured JSON (for production)
        log_file: Optional file path for log output
    """
    handlers = []

    # Console handler
    if json_output:
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s %(account_id)s %(trace_id)s'
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
        except ImportError:
            # Fallback if pythonjsonlogger not available
            formatter = logging.Formatter(
                '%(asctime)s [%(account_id)s/%(trace_id)s] %(name)s %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(account_id)s/%(trace_id)s] %(name)s %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)

    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = handlers

    configure_third_party_loggers()


def configure_third_party_loggers(level: int = logging.WARNING) -> None:
    """Keep routine third-party client chatter out of INFO logs."""
    for logger_name in NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with context filtering capability.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def with_context(logger: logging.Logger, ctx: RequestContext) -> ContextLoggerAdapter:
    """Add RequestContext to a logger for this scope.

    Args:
        logger: Base logger
        ctx: RequestContext with account/user/agent info

    Returns:
        ContextLoggerAdapter that injects context into every log message.
        Thread-safe and does not accumulate filters.
    """
    return ContextLoggerAdapter(logger, ctx)
