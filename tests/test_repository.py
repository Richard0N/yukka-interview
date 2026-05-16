"""Tests for honey.data.repository — Asset, Repository ABC, and YukkaRepository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from honey.data.repository import Asset, Repository
from honey.data.yukka_repository import YukkaRepository

AAPL = Asset(name="Apple Inc.", isin="US0378331005", yukka_id="company:apple", ric="AAPL.O")


class TestAsset:
    """Unit tests for the Asset dataclass."""

    def test_fields(self):
        """Verify ric and name are stored correctly."""
        asset = Asset(name="Apple Inc.", isin="US0378331005", yukka_id="company:apple", ric="AAPL.O")
        assert asset.ric == "AAPL.O"
        assert asset.name == "Apple Inc."

    def test_frozen(self):
        """Verify that frozen=True prevents attribute mutation."""
        asset = Asset(name="Apple Inc.", isin="US0378331005", yukka_id="company:apple", ric="AAPL.O")
        with pytest.raises(AttributeError):
            asset.ric = "MSFT.O"  # type: ignore[misc]

    def test_equality(self):
        """Verify value-based equality semantics."""
        assert Asset(name="Y", isin="I", yukka_id="U", ric="X") == Asset(name="Y", isin="I", yukka_id="U", ric="X")
        assert Asset(name="Y", isin="I", yukka_id="U", ric="X") != Asset(name="Y", isin="I", yukka_id="U", ric="Z")


class TestRepositoryABC:
    """Unit tests for the Repository abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Verify that Repository cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Repository()


class _NoopRepository(Repository):
    """Minimal concrete subclass used to test Repository."""

    def assets(self, **kwargs):
        return []

    def prices(self, assets=None, **kwargs):
        return pl.DataFrame(schema={"date": pl.Date})

    def returns(self, assets=None, **kwargs):
        return pl.DataFrame(schema={"date": pl.Date})


@dataclass
class _FakeIndex:
    """Duck-type substitute for yukka.data.Index used in tests."""

    name: str
    frame: pl.DataFrame
    membership_intervals: pl.DataFrame

    @property
    def assets(self) -> list[Asset]:
        return [
            Asset(name=row["name"], isin=row["isin"], yukka_id=row["yukka_id"], ric=row["ric"])
            for row in self.frame.unique(subset=["isin"], keep="first").iter_rows(named=True)
        ]


_EMPTY_INDEX = _FakeIndex(
    name="test",
    frame=pl.DataFrame(schema={"name": pl.Utf8, "isin": pl.Utf8, "yukka_id": pl.Utf8, "ric": pl.Utf8}),
    membership_intervals=pl.DataFrame(schema={"ric": pl.Utf8, "start_date": pl.Date, "end_date": pl.Date}),
)


def _make_fake_index(*rows: dict, name: str = "test") -> _FakeIndex:
    """Return a _FakeIndex with the given rows as its frame."""
    df = pl.DataFrame(rows)
    membership_intervals = pl.DataFrame(
        schema={"ric": pl.Utf8, "start_date": pl.Date, "end_date": pl.Date},
    )
    return _FakeIndex(name=name, frame=df, membership_intervals=membership_intervals)


class TestYukkaRepository:
    """Unit tests for YukkaRepository."""

    def test_assets_maps_columns(self):
        """Verify that index DataFrame columns map to Asset fields."""
        idx = _make_fake_index(
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
            },
        )
        repo = YukkaRepository(index=idx)
        assets = repo.assets
        assert len(assets) == 1
        assert assets[0] == AAPL

    def test_assets_deduplicates_on_isin(self):
        """Verify that duplicate ISINs within an index are deduplicated."""
        row = {
            "name": "Apple Inc.",
            "isin": "US0378331005",
            "yukka_id": "company:apple",
            "ric": "AAPL.O",
        }
        repo = YukkaRepository(index=_make_fake_index(row, row, name="test"))
        assert len(repo.assets) == 1

    def test_assets_returns_empty_for_empty_index(self):
        """assets() returns an empty list when the index frame has no rows."""
        repo = YukkaRepository(index=_EMPTY_INDEX)
        assert repo.assets == []

    def test_assets_returns_same_content_on_repeated_calls(self):
        """assets() returns equal content on repeated calls."""
        idx = _make_fake_index(
            {"name": "X", "isin": "I", "yukka_id": "U", "ric": "R"},
        )
        repo = YukkaRepository(index=idx)
        assert repo.assets == repo.assets


MSFT = Asset(name="Microsoft Corp.", isin="US5949181045", yukka_id="company:microsoft", ric="MSFT.O")
MISSING = Asset(name="Ghost Corp.", isin="XX0000000000", yukka_id="company:ghost", ric="GHOST.X")


@pytest.fixture
def prices_parquet(tmp_path: Path) -> Path:
    """Create a temporary parquet with known price data."""
    import datetime

    df = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
            "AAPL.O": [150.0, 151.0],
            "MSFT.O": [350.0, 352.0],
        }
    )
    df.write_parquet(tmp_path / "prices_all.parquet")
    return tmp_path


class TestYukkaRepositoryPrices:
    """Unit tests for YukkaRepository.prices."""

    def test_default_ric_columns(self, prices_parquet: Path):
        """Default id_col='ric' keeps RIC column names."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df = repo.prices([AAPL, MSFT])
        assert "AAPL.O" in df.columns
        assert "MSFT.O" in df.columns

    def test_isin_columns(self, prices_parquet: Path):
        """id_col='isin' renames columns to ISIN values."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet, id_col="isin")
        df = repo.prices([AAPL, MSFT])
        assert "US0378331005" in df.columns
        assert "US5949181045" in df.columns
        assert "AAPL.O" not in df.columns

    def test_yukka_id_columns(self, prices_parquet: Path):
        """id_col='yukka_id' renames columns to yukka_id values."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet, id_col="yukka_id")
        df = repo.prices([AAPL, MSFT])
        assert "company:apple" in df.columns
        assert "company:microsoft" in df.columns

    def test_date_column_always_present(self, prices_parquet: Path):
        """The date column is always present regardless of id_col."""
        for id_col in ("ric", "isin", "yukka_id"):
            repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet, id_col=id_col)
            df = repo.prices([AAPL])
            assert "date" in df.columns

    def test_missing_ric_silently_skipped(self, prices_parquet: Path):
        """Assets whose RIC is not in the parquet are silently skipped."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df = repo.prices([AAPL, MISSING])
        assert "AAPL.O" in df.columns
        assert "GHOST.X" not in df.columns
        assert df.shape == (2, 2)  # date + AAPL.O only

    def test_duplicate_id_col_deduplicates(self, prices_parquet: Path):
        """Assets sharing the same target id_col keep only the first occurrence."""
        aapl_dup = Asset(name="Apple Dup", isin="US0378331005", yukka_id="company:apple", ric="MSFT.O")
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet, id_col="yukka_id")
        df = repo.prices([AAPL, aapl_dup])
        assert df.columns.count("company:apple") == 1
        assert df.shape == (2, 2)  # date + one company:apple column


class TestYukkaRepositoryPricesDateFiltering:
    """Tests for date_from / date_to on prices()."""

    def test_date_from_filters_rows(self, prices_parquet: Path):
        """date_from excludes rows before the given date."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df = repo.prices([AAPL], date_from="2024-01-02")
        assert df.shape[0] == 1
        import datetime

        assert df["date"][0] == datetime.date(2024, 1, 2)

    def test_date_to_filters_rows(self, prices_parquet: Path):
        """date_to excludes rows after the given date."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df = repo.prices([AAPL], date_to="2024-01-01")
        assert df.shape[0] == 1
        import datetime

        assert df["date"][0] == datetime.date(2024, 1, 1)

    def test_date_range(self, masked_prices_parquet: Path):
        """date_from and date_to together select a sub-range."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=masked_prices_parquet)
        df = repo.prices([AAPL], date_from="2024-01-02", date_to="2024-01-03")
        assert df.shape[0] == 2

    def test_defaults_use_repo_date_from(self, prices_parquet: Path):
        """Without kwargs, repo-level date_from is applied."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet, date_from="2024-01-02")
        df = repo.prices([AAPL])
        assert df.shape[0] == 1


# ---------------------------------------------------------------------------
# Membership masking tests
# ---------------------------------------------------------------------------


def _make_membership_index(*rows: dict, name: str = "test") -> _FakeIndex:
    """Return a _FakeIndex with membership interval data."""
    df = pl.DataFrame(rows)
    membership_intervals = df.select(
        pl.col("ric"),
        pl.col("start_date").cast(pl.Date),
        pl.col("end_date").cast(pl.Date),
    )
    return _FakeIndex(name=name, frame=df, membership_intervals=membership_intervals)


@pytest.fixture
def masked_prices_parquet(tmp_path: Path) -> Path:
    """Create a temporary parquet with price data spanning several days."""
    import datetime

    df = pl.DataFrame(
        {
            "date": [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 1, 2),
                datetime.date(2024, 1, 3),
                datetime.date(2024, 1, 4),
            ],
            "AAPL.O": [150.0, 151.0, 152.0, 153.0],
            "MSFT.O": [350.0, 352.0, 354.0, 356.0],
        }
    )
    df.write_parquet(tmp_path / "prices_all.parquet")
    return tmp_path


class TestMembershipMaskingPrices:
    """Tests for membership masking in prices()."""

    def test_prices_returns_unmasked_when_intervals_empty(self, prices_parquet: Path):
        """Prices are returned unmasked when index has no membership intervals."""
        idx = _make_fake_index(
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
            },
        )
        repo = YukkaRepository(index=idx, cache_dir=prices_parquet)
        df = repo.prices([AAPL], mask=True)
        assert df["AAPL.O"].null_count() == 0

    def test_prices_warns_for_ric_not_in_intervals(self, masked_prices_parquet: Path, caplog):
        """Warning is logged when an asset RIC is absent from membership intervals."""
        import logging

        loader = _make_membership_index(
            {
                "name": "Microsoft Corp.",
                "isin": "US5949181045",
                "yukka_id": "company:microsoft",
                "ric": "MSFT.O",
                "start_date": "2024-01-01",
                "end_date": "2024-01-04",
            },
        )
        repo = YukkaRepository(index=loader, cache_dir=masked_prices_parquet)
        with caplog.at_level(logging.WARNING):
            df = repo.prices([AAPL, MSFT], mask=True)
        assert "AAPL.O" in caplog.text
        assert df["AAPL.O"].null_count() == 0

    def test_prices_mask_nulls_outside_interval(self, masked_prices_parquet: Path):
        """Prices outside the membership interval are nulled."""
        loader = _make_membership_index(
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
                "start_date": "2024-01-02",
                "end_date": "2024-01-03",
            },
            {
                "name": "Microsoft Corp.",
                "isin": "US5949181045",
                "yukka_id": "company:microsoft",
                "ric": "MSFT.O",
                "start_date": "2024-01-01",
                "end_date": "2024-01-04",
            },
        )
        repo = YukkaRepository(index=loader, cache_dir=masked_prices_parquet)
        df = repo.prices([AAPL, MSFT], mask=True)
        aapl_vals = df["AAPL.O"].to_list()
        # Jan 1 and Jan 4 are outside AAPL's interval
        assert aapl_vals[0] is None
        assert aapl_vals[1] == 151.0
        assert aapl_vals[2] == 152.0
        assert aapl_vals[3] is None
        # MSFT covers all dates
        assert df["MSFT.O"].null_count() == 0

    def test_prices_mask_false_returns_unfiltered(self, masked_prices_parquet: Path):
        """mask=False returns all data without nulling."""
        loader = _make_membership_index(
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
                "start_date": "2024-01-02",
                "end_date": "2024-01-03",
            },
        )
        repo = YukkaRepository(index=loader, cache_dir=masked_prices_parquet)
        df = repo.prices([AAPL], mask=False)
        assert df["AAPL.O"].null_count() == 0

    def test_prices_multi_interval(self, masked_prices_parquet: Path):
        """Company that leaves and re-enters keeps both intervals."""
        loader = _make_membership_index(
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
            },
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
                "start_date": "2024-01-03",
                "end_date": "2024-01-04",
            },
        )
        repo = YukkaRepository(index=loader, cache_dir=masked_prices_parquet)
        df = repo.prices([AAPL], mask=True)
        vals = df["AAPL.O"].to_list()
        assert vals[0] == 150.0  # Jan 1 in first interval
        assert vals[1] is None  # Jan 2 gap
        assert vals[2] == 152.0  # Jan 3 in second interval
        assert vals[3] == 153.0  # Jan 4 in second interval

    def test_prices_empty_index_skips_masking(self, masked_prices_parquet: Path):
        """Empty membership intervals means masking has no effect even if mask=True."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=masked_prices_parquet)
        df = repo.prices([AAPL], mask=True)
        assert df["AAPL.O"].null_count() == 0


# ---------------------------------------------------------------------------
# Returns tests
# ---------------------------------------------------------------------------


class TestYukkaRepositoryReturns:
    """Tests for YukkaRepository.returns."""

    def test_returns_basic(self, prices_parquet: Path):
        """returns() produces a DataFrame with expected shape and columns."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df = repo.returns([AAPL, MSFT])
        assert "date" in df.columns
        assert "AAPL.O" in df.columns
        assert "MSFT.O" in df.columns
        assert df.shape[0] == 2

    def test_returns_first_row_null(self, prices_parquet: Path):
        """First row is null because of the h-period shift."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df = repo.returns([AAPL])
        assert df["AAPL.O"][0] is None

    def test_returns_values(self, prices_parquet: Path):
        """Returns are price[t] / price[t-h] - 1."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df = repo.returns([AAPL])
        expected = 151.0 / 150.0 - 1
        assert abs(df["AAPL.O"][1] - expected) < 1e-10

    def test_returns_with_masking(self, masked_prices_parquet: Path):
        """Returns outside the membership interval are nulled (id_col='ric' path)."""
        loader = _make_membership_index(
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
                "start_date": "2024-01-02",
                "end_date": "2024-01-03",
            },
        )
        repo = YukkaRepository(index=loader, cache_dir=masked_prices_parquet)
        df = repo.returns([AAPL])
        non_null = df["AAPL.O"].drop_nulls()
        assert len(non_null) == 1

    def test_returns_with_non_ric_id_col(self, masked_prices_parquet: Path):
        """id_col='isin' columns are correctly resolved to RICs for masking."""
        loader = _make_membership_index(
            {
                "name": "Apple Inc.",
                "isin": "US0378331005",
                "yukka_id": "company:apple",
                "ric": "AAPL.O",
                "start_date": "2024-01-01",
                "end_date": "2024-01-04",
            },
        )
        repo = YukkaRepository(index=loader, cache_dir=masked_prices_parquet, id_col="isin")
        df = repo.returns([AAPL])
        assert "US0378331005" in df.columns


# ---------------------------------------------------------------------------
# Integration test: STOXX 600 constituent cap
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStoxx600ConstituentCap:
    """Verify that masking keeps the STOXX 600 within its expected size."""

    def test_masked_prices_never_exceed_660_constituents(self):
        """After masking, no date should have more than 660 non-null price columns."""
        from yukka.data import Index

        repo = YukkaRepository(index=Index.STOXX600)
        assets = repo.assets
        df = repo.prices(assets, mask=True)

        # Count non-null values per row (exclude the date column)
        asset_cols = [c for c in df.columns if c != "date"]
        non_null_counts = df.select(
            pl.col("date"),
            pl.sum_horizontal(pl.col(c).is_not_null() for c in asset_cols).alias("n_constituents"),
        )

        max_constituents = non_null_counts["n_constituents"].max()
        worst_date = non_null_counts.filter(pl.col("n_constituents") == max_constituents)["date"][0]

        assert max_constituents <= 660, (
            f"Found {max_constituents} constituents with price data on {worst_date}, expected at most 660 (600 + 10%)"
        )

    def test_rank_masked_prices_never_exceed_range_size(self):
        """After rank masking with (1, 50), no date should have more than 50 non-null columns."""
        from yukka.data import Index

        repo = YukkaRepository(index=Index.STOXX600)
        assets = repo.assets
        df = repo.prices(assets, mask=True, rank_range=(1, 50))

        asset_cols = [c for c in df.columns if c != "date"]
        non_null_counts = df.select(
            pl.col("date"),
            pl.sum_horizontal(pl.col(c).is_not_null() for c in asset_cols).alias("n_constituents"),
        )

        max_constituents = non_null_counts["n_constituents"].max()
        worst_date = non_null_counts.filter(pl.col("n_constituents") == max_constituents)["date"][0]

        assert max_constituents <= 50, (
            f"Found {max_constituents} constituents with price data on {worst_date}, expected at most 50"
        )


# ---------------------------------------------------------------------------
# Rank masking unit tests
# ---------------------------------------------------------------------------


class TestRankMaskingPrices:
    """Unit tests for rank-based masking in prices()."""

    def test_rank_masking_allows_reentry(self, tmp_path: Path):
        """A company can enter and leave the rank range across review periods."""
        import datetime

        # Price data: 6 trading days
        dates = [datetime.date(2024, 1, d) for d in range(1, 7)]
        prices_df = pl.DataFrame({"date": dates, "A.X": [100.0] * 6, "B.X": [200.0] * 6})
        prices_df.write_parquet(tmp_path / "prices_all.parquet")

        # Ranks: 3 review dates — A.X toggles in/out of top-1
        ranks_df = pl.DataFrame(
            {
                "review_date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 3), datetime.date(2024, 1, 5)],
                "A.X": [1, 2, 1],  # in, out, in
                "B.X": [2, 1, 2],
            }
        )
        ranks_df.write_parquet(tmp_path / "ranks_wide.parquet")

        a = Asset(name="A", isin="IA", yukka_id="a", ric="A.X")
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=tmp_path)
        df = repo.prices([a], mask=True, rank_range=(1, 1))
        vals = df["A.X"].to_list()
        # Days 1-2: rank=1 (in), Days 3-4: rank=2 (out), Days 5-6: rank=1 (in)
        assert vals[0] == 100.0  # Jan 1
        assert vals[1] == 100.0  # Jan 2
        assert vals[2] is None  # Jan 3
        assert vals[3] is None  # Jan 4
        assert vals[4] == 100.0  # Jan 5
        assert vals[5] == 100.0  # Jan 6

    def test_rank_range_none_preserves_behavior(self, prices_parquet: Path):
        """rank_range=None (default) does not alter results."""
        repo = YukkaRepository(index=_EMPTY_INDEX, cache_dir=prices_parquet)
        df_default = repo.prices([AAPL, MSFT])
        df_explicit = repo.prices([AAPL, MSFT], rank_range=None)
        assert df_default.equals(df_explicit)
