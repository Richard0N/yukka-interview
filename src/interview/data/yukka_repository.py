"""Concrete repository backed by the Yukka SDK."""

from __future__ import annotations

import contextlib
import datetime
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from yukka import Session as YukkaSession

# Index enum initialization triggers constituent-loading prints from the SDK; suppress them.
with contextlib.redirect_stdout(io.StringIO()):
    from yukka.data import Asset, Index

from .config import CACHE_DIR
from .repository import Repository
from .returns import Returns

logger = logging.getLogger(__name__)


def _load_rank_mask(
    target_dates: pl.Series,
    ric_columns: list[str],
    rank_range: tuple[int, int],
    ranks_path: Path,
) -> pl.DataFrame:
    """Build a boolean mask (wide format) for the given dates and RICs.

    For each target date, finds the most recent review_date <= date via an asof
    join, then checks whether each RIC's rank falls within *rank_range* (inclusive).
    """
    lo, hi = rank_range
    ranks = pl.read_parquet(ranks_path).rename({"review_date": "date"}).sort("date")

    # Keep only the RIC columns that appear in the ranks file
    available = set(ranks.columns) - {"date"}
    ric_cols_in_ranks = [r for r in ric_columns if r in available]
    missing = [r for r in ric_columns if r not in available]

    dates_df = pl.DataFrame({"date": target_dates}).sort("date")

    if ric_cols_in_ranks:
        joined = dates_df.join_asof(
            ranks.select(["date", *ric_cols_in_ranks]),
            on="date",
            strategy="backward",
        )
        mask_exprs = [
            (pl.col(r).is_not_null() & (pl.col(r) >= lo) & (pl.col(r) <= hi)).alias(r) for r in ric_cols_in_ranks
        ]
        result = joined.select("date", *mask_exprs)
    else:
        result = dates_df.clone()

    # RICs not in the ranks file get False (excluded)
    for r in missing:
        result = result.with_columns(pl.lit(False).alias(r))

    return result


@dataclass
class YukkaRepository(Repository):
    """Concrete repository backed by the Yukka SDK.

    Can be used as a context manager to keep a single ``YukkaSession`` open
    across multiple data calls::

        with YukkaRepository() as repo:
            df = repo.prices()
    """

    index: Index = Index.STOXX600
    id_col: str = "ric"
    date_from: str = "2015-01-02"
    date_to: str | None = None
    cache_dir: Path = CACHE_DIR
    _session: YukkaSession | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> YukkaRepository:
        """Open a shared YukkaSession for the duration of the context."""
        self._session = YukkaSession().__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        """Close the shared YukkaSession."""
        if self._session is not None:
            self._session.__exit__(*args)
            self._session = None

    @property
    def assets(self) -> list[Asset]:
        """Return a list of all assets in the configured index."""
        return self.index.assets

    def _get_membership_intervals(self) -> pl.DataFrame:
        """Return membership intervals from the configured index(es)."""
        if self.index is None:
            return pl.concat([idx.membership_intervals for idx in Index]).unique()
        return self.index.membership_intervals

    def _mask_prices(
        self,
        df: pl.DataFrame,
        asset_columns: list[str],
        ric_columns: list[str],
        rank_range: tuple[int, int] | None = None,
    ) -> pl.DataFrame:
        """Null out price values outside membership intervals (wide format).

        If *rank_range* is given, also null out columns whose market-cap rank
        falls outside the range at each date.
        """
        intervals = self._get_membership_intervals()
        if not intervals.is_empty():
            constituent_rics = set(intervals["ric"].to_list())
            exprs: list[pl.Expr] = []
            for col_name, ric in zip(asset_columns, ric_columns, strict=True):
                if ric not in constituent_rics:
                    logger.warning("%s is not a constituent of any configured index — masking has no effect", ric)
                    continue
                ric_intervals = intervals.filter(pl.col("ric") == ric)
                clauses = [
                    (pl.col("date") >= row["start_date"]) & (pl.col("date") <= row["end_date"])
                    for row in ric_intervals.iter_rows(named=True)
                ]
                condition = clauses[0]
                for clause in clauses[1:]:
                    condition = condition | clause
                exprs.append(pl.when(condition).then(pl.col(col_name)).otherwise(None).alias(col_name))
            if exprs:
                df = df.with_columns(exprs)

        if rank_range is not None:
            rank_mask = _load_rank_mask(df["date"], ric_columns, rank_range, self.cache_dir / "ranks_wide.parquet")
            # Map RIC mask columns to asset_columns (they may differ when id_col != "ric")
            rank_exprs = []
            for col_name, ric in zip(asset_columns, ric_columns, strict=True):
                if ric in rank_mask.columns:
                    rank_exprs.append(
                        pl.when(pl.col(f"_rank_{ric}")).then(pl.col(col_name)).otherwise(None).alias(col_name)
                    )
            if rank_exprs:
                # Join rank mask onto df by date, with prefixed columns to avoid collisions
                rename_map = {r: f"_rank_{r}" for r in ric_columns if r in rank_mask.columns}
                rank_mask = rank_mask.rename(rename_map)
                df = df.join(rank_mask, on="date", how="left").with_columns(rank_exprs)
                df = df.drop([f"_rank_{r}" for r in ric_columns if r in rename_map])

        return df

    def _resolve_dates(self, kwargs: dict) -> tuple[datetime.date, datetime.date | None]:
        """Return (date_from, date_to) from kwargs, falling back to instance defaults."""
        date_from = datetime.date.fromisoformat(kwargs.get("date_from", self.date_from))
        raw_to = kwargs.get("date_to", self.date_to)
        date_to = datetime.date.fromisoformat(raw_to) if raw_to is not None else None
        return date_from, date_to

    @staticmethod
    def _filter_dates(df: pl.DataFrame, date_from: datetime.date, date_to: datetime.date | None) -> pl.DataFrame:
        """Filter a DataFrame with a ``date`` column to the given range."""
        pred = pl.col("date") >= date_from
        if date_to is not None:
            pred = pred & (pl.col("date") <= date_to)
        return df.filter(pred)

    def prices(self, assets: list[Asset] | None = None, **kwargs) -> pl.DataFrame:
        """Load raw price data for the given assets.

        Parameters
        ----------
        mask:
            If ``True`` (default), null out prices outside membership intervals.
        rank_range:
            Optional ``(lo, hi)`` tuple. When set, only companies whose
            market-cap rank falls within the range (inclusive) have data;
            others are nulled out.
        date_from:
            Override the repository-level start date (ISO format string).
        date_to:
            Override the repository-level end date (ISO format string or ``None``).
        """
        assets = assets or self.assets
        mask: bool = kwargs.get("mask", True)
        rank_range: tuple[int, int] | None = kwargs.get("rank_range")
        date_from, date_to = self._resolve_dates(kwargs)
        scanner = pl.scan_parquet(self.cache_dir / "prices_all.parquet")
        parquet_columns = set(scanner.collect_schema().names())

        # Deduplicate by the target id_col to avoid duplicate column names.
        seen_ids: set[str] = set()
        unique_assets: list[Asset] = []
        for a in assets:
            target_id = getattr(a, self.id_col)
            if a.ric in parquet_columns and target_id not in seen_ids:
                seen_ids.add(target_id)
                unique_assets.append(a)

        present_rics = [a.ric for a in unique_assets]
        df: pl.DataFrame = scanner.select(["date", *present_rics]).collect()  # type: ignore[assignment]
        df = self._filter_dates(df, date_from, date_to)

        if mask:
            df = self._mask_prices(df, present_rics, present_rics, rank_range=rank_range)

        if self.id_col != "ric":
            ric_to_id = {a.ric: getattr(a, self.id_col) for a in unique_assets}
            df = df.rename({ric: ric_to_id[ric] for ric in present_rics})

        return df

    def returns(self, assets: list[Asset] | None = None, **kwargs) -> pl.DataFrame:
        """Compute backward-looking returns from price data (no look-ahead bias).

        Parameters
        ----------
        h:
            Return horizon in periods (default 1). Uses price[t] / price[t-h] - 1.
        mask:
            If ``True`` (default), null out returns outside membership intervals.
        rank_range:
            Optional ``(lo, hi)`` tuple forwarded to masking.
        date_from, date_to:
            Forwarded to prices().
        """
        assets = assets or self.assets
        mask: bool = kwargs.pop("mask", True)
        h: int = kwargs.pop("h", 1)
        rank_range: tuple[int, int] | None = kwargs.pop("rank_range", None)
        prices = self.prices(assets, mask=False, **kwargs)
        prices = prices.unique(subset=["date"], keep="last").sort("date")
        entity_cols = [c for c in prices.columns if c != "date"]
        df = prices.select(
            "date",
            *[(pl.col(c) / pl.col(c).shift(h) - 1).alias(c) for c in entity_cols],
        )

        valid = None
        if mask:
            if self.id_col != "ric":
                id_to_ric = {getattr(a, self.id_col): a.ric for a in assets}
                ric_cols = [id_to_ric.get(c, c) for c in entity_cols]
            else:
                ric_cols = list(entity_cols)
            masked_prices = self._mask_prices(prices, entity_cols, ric_cols, rank_range=rank_range)
            valid = masked_prices.select(
                "date",
                *[(pl.col(c).is_not_null() & pl.col(c).shift(h).is_not_null()).alias(c) for c in entity_cols],
            )

        return Returns(df=df, date_col="date", _valid=valid).mask().df
