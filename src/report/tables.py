"""Pure-function table and metric formatting utilities for report generation.

All functions are stateless: they accept primitive values and return formatted
strings suitable for embedding in Markdown report sections.
"""

from __future__ import annotations


def format_ratio(value: float, decimals: int = 4) -> str:
    """Format a ratio-valued metric to *decimals* decimal places.

    >>> format_ratio(0.87273)
    '0.8727'
    """
    return f"{value:.{decimals}f}"


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format an aggregate metric as ``mean +/- std``.

    >>> format_mean_std(0.8727, 0.0031)
    '0.8727 +/- 0.0031'
    """
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def format_param_count(n: int) -> str:
    """Format an integer parameter count with comma separators.

    >>> format_param_count(5197975)
    '5,197,975'
    """
    return f"{n:,}"


def format_time_seconds(value: float) -> str:
    """Format a duration in seconds to 2 decimal places.

    >>> format_time_seconds(60.6234)
    '60.62'
    """
    return f"{value:.2f}"


def format_ms_per_example(value: float) -> str:
    """Format a per-example latency in milliseconds to 4 decimal places.

    >>> format_ms_per_example(0.056312)
    '0.0563'
    """
    return f"{value:.4f}"


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a pipe-delimited Markdown table from *headers* and *rows*.

    Each row must have the same number of elements as *headers*.
    Columns are right-padded to the widest cell for readability.

    Returns the complete table as a single string (no trailing newline).
    """
    if not headers:
        return ""
    n_cols = len(headers)
    for i, row in enumerate(rows):
        if len(row) != n_cols:
            raise ValueError(f"Row {i} has {len(row)} columns but headers have {n_cols}")

    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def _pad_row(cells: list[str]) -> str:
        padded = [cell.ljust(col_widths[j]) for j, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    lines: list[str] = [_pad_row(headers)]
    separator_cells = ["-" * col_widths[j] for j in range(n_cols)]
    lines.append("| " + " | ".join(separator_cells) + " |")
    for row in rows:
        lines.append(_pad_row(row))

    return "\n".join(lines)
