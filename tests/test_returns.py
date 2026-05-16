"""Tests for honey.data.returns — Returns class."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from honey.data.returns import Returns


@pytest.fixture
def returns_df() -> pl.DataFrame:
    """Small returns DataFrame for testing."""
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    return pl.DataFrame({"date": dates, "A": [0.01 * ((-1) ** i) for i in range(100)], "B": [0.005] * 100})


@pytest.fixture
def valid_df(returns_df: pl.DataFrame) -> pl.DataFrame:
    """Validity mask: A valid for first 50 rows, B always valid."""
    dates = returns_df["date"].to_list()
    return pl.DataFrame({"date": dates, "A": [i < 50 for i in range(100)], "B": [True] * 100})


class TestReturnsInit:
    """Tests for Returns initialisation."""

    def test_creates_returns(self, returns_df: pl.DataFrame) -> None:
        """Returns wraps a DataFrame with default date_col and no validity mask."""
        r = Returns(df=returns_df)
        assert r.df is returns_df
        assert r.date_col == "date"
        assert r._valid is None

    def test_custom_date_col(self, returns_df: pl.DataFrame) -> None:
        """A custom date column name is stored correctly."""
        df = returns_df.rename({"date": "dt"})
        r = Returns(df=df, date_col="dt")
        assert r.date_col == "dt"


class TestWinsorise:
    """Tests for Returns.winsorise."""

    def test_returns_new_instance(self, returns_df: pl.DataFrame) -> None:
        """Winsorise returns a new Returns instance with the same date_col."""
        r = Returns(df=returns_df)
        result = r.winsorise(window=10, n_sigma=2.0)
        assert isinstance(result, Returns)
        assert result is not r
        assert result.date_col == "date"

    def test_preserves_valid(self, returns_df: pl.DataFrame, valid_df: pl.DataFrame) -> None:
        """Winsorise carries the validity mask through."""
        r = Returns(df=returns_df, _valid=valid_df)
        result = r.winsorise(window=10)
        assert result._valid is valid_df


class TestVolAdjust:
    """Tests for Returns.vol_adjust."""

    def test_returns_new_instance(self, returns_df: pl.DataFrame) -> None:
        """Vol-adjust returns a new Returns instance."""
        r = Returns(df=returns_df)
        result = r.vol_adjust(window=10)
        assert isinstance(result, Returns)
        assert result is not r

    def test_preserves_valid(self, returns_df: pl.DataFrame, valid_df: pl.DataFrame) -> None:
        """Vol-adjust carries the validity mask through."""
        r = Returns(df=returns_df, _valid=valid_df)
        result = r.vol_adjust(window=10)
        assert result._valid is valid_df


class TestMask:
    """Tests for Returns.mask."""

    def test_no_valid_returns_copy(self, returns_df: pl.DataFrame) -> None:
        """Mask without a validity mask returns a copy with same shape."""
        r = Returns(df=returns_df)
        result = r.mask()
        assert isinstance(result, Returns)
        assert result._valid is None
        assert result.df.shape == returns_df.shape

    def test_applies_mask(self, returns_df: pl.DataFrame, valid_df: pl.DataFrame) -> None:
        """Mask nulls values outside valid intervals."""
        r = Returns(df=returns_df, _valid=valid_df)
        result = r.mask()
        # A should be null after row 50
        a_values = result.df["A"].to_list()
        assert a_values[0] is not None
        assert a_values[50] is None
        assert a_values[99] is None
        # B should remain non-null everywhere
        assert result.df["B"].null_count() == 0
