"""Tests for src.report.tables — pure formatting functions.

Covers decimal places, mean +/- std format, comma-separated integers,
time formatting, and markdown table building (including edge cases).
"""

from __future__ import annotations

import pytest

from src.report.tables import (
    build_markdown_table,
    format_mean_std,
    format_ms_per_example,
    format_param_count,
    format_ratio,
    format_time_seconds,
)


class TestFormatRatio:
    def test_default_four_decimals(self) -> None:
        assert format_ratio(0.87273) == "0.8727"

    def test_exact_value(self) -> None:
        assert format_ratio(0.5) == "0.5000"

    def test_zero(self) -> None:
        assert format_ratio(0.0) == "0.0000"

    def test_one(self) -> None:
        assert format_ratio(1.0) == "1.0000"

    def test_custom_decimals(self) -> None:
        assert format_ratio(0.5, decimals=2) == "0.50"

    def test_very_small(self) -> None:
        assert format_ratio(0.00001) == "0.0000"

    def test_negative(self) -> None:
        assert format_ratio(-0.1234) == "-0.1234"


class TestFormatMeanStd:
    def test_default(self) -> None:
        assert format_mean_std(0.8727, 0.0031) == "0.8727 +/- 0.0031"

    def test_custom_decimals(self) -> None:
        assert format_mean_std(60.62, 10.0, decimals=2) == "60.62 +/- 10.00"

    def test_zero_std(self) -> None:
        assert format_mean_std(0.9, 0.0) == "0.9000 +/- 0.0000"


class TestFormatParamCount:
    def test_large_number(self) -> None:
        assert format_param_count(5197975) == "5,197,975"

    def test_small_number(self) -> None:
        assert format_param_count(100) == "100"

    def test_zero(self) -> None:
        assert format_param_count(0) == "0"

    def test_one_million(self) -> None:
        assert format_param_count(1000000) == "1,000,000"


class TestFormatTimeSeconds:
    def test_normal(self) -> None:
        assert format_time_seconds(60.6234) == "60.62"

    def test_zero(self) -> None:
        assert format_time_seconds(0.0) == "0.00"

    def test_large(self) -> None:
        assert format_time_seconds(3600.5) == "3600.50"


class TestFormatMsPerExample:
    def test_normal(self) -> None:
        assert format_ms_per_example(0.056312) == "0.0563"

    def test_zero(self) -> None:
        assert format_ms_per_example(0.0) == "0.0000"


class TestBuildMarkdownTable:
    def test_simple_table(self) -> None:
        result = build_markdown_table(["A", "B"], [["1", "2"], ["3", "4"]])
        lines = result.split("\n")
        assert len(lines) == 4
        assert "| A" in lines[0]
        assert "| ---" in lines[1] or "|" in lines[1]
        assert "| 1" in lines[2]
        assert "| 3" in lines[3]

    def test_empty_headers(self) -> None:
        assert build_markdown_table([], []) == ""

    def test_column_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Row 0 has 1 columns"):
            build_markdown_table(["A", "B"], [["only_one"]])

    def test_column_alignment_padding(self) -> None:
        result = build_markdown_table(["Model", "Score"], [["MLP", "0.8727"]])
        assert "MLP" in result
        assert "0.8727" in result

    def test_no_trailing_newline(self) -> None:
        result = build_markdown_table(["X"], [["y"]])
        assert not result.endswith("\n")
