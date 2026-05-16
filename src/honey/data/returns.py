"""Return computation and preprocessing utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl
from jquantstats import Data


@dataclass
class Returns:
    """Computed returns with preprocessing methods.

    Examples:
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"date": ["2020-01-01", "2020-01-02"], "AAPL": [0.01, -0.02]})
    >>> r = Returns(df=df, date_col="date")
    >>> r.df.shape
    (2, 2)
    >>> r.mask().df.shape
    (2, 2)
    """

    df: pl.DataFrame
    date_col: str = "date"
    _valid: pl.DataFrame | None = field(default=None, repr=False)

    def _to_data(self) -> Data:
        """Wrap returns in a jquantstats Data object."""
        return Data.from_returns(returns=self.df, date_col=self.date_col)

    def winsorise(self, window: int = 60, n_sigma: float = 3.0) -> Returns:
        """Winsorise returns by clipping outliers beyond *n_sigma* in a rolling window."""
        result = self._to_data().utils.winsorise(window=window, n_sigma=n_sigma)
        return Returns(df=result, date_col=self.date_col, _valid=self._valid)

    def vol_adjust(
        self,
        window: int = 60,
        vol_estimator: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> Returns:
        """Volatility-adjust returns, optionally with a custom *vol_estimator*."""
        result = self._to_data().utils.to_volatility_adjusted_returns(
            window=window,
            vol_estimator=vol_estimator,
        )
        return Returns(df=result, date_col=self.date_col, _valid=self._valid)

    def mask(self) -> Returns:
        """Apply membership mask, nulling values outside valid intervals."""
        if self._valid is None:
            return Returns(df=self.df, date_col=self.date_col)
        entity_cols = [c for c in self.df.columns if c != self.date_col]
        valid_renamed = self._valid.select(
            self.date_col,
            *[pl.col(c).alias(f"__v_{c}") for c in entity_cols],
        )
        masked_df = self.df.join(valid_renamed, on=self.date_col, how="left").select(
            self.date_col,
            *[pl.when(pl.col(f"__v_{c}")).then(pl.col(c)).otherwise(None).alias(c) for c in entity_cols],
        )
        return Returns(df=masked_df, date_col=self.date_col)
