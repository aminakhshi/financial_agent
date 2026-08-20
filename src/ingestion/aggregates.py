"""Sector aggregate series computed from index constituents.

Method (equal_weight): a chained equal-weighted return index per GICS sector.
For each bar the sector return is the mean of member returns (point-in-time
membership), and the level is chained: level_t = level_{t-1} * (1 + r_t),
starting at 100. Chaining means membership additions/removals never cause a
level jump — the series is a stable training target that survives churn.

Open/high/low are approximated from the mean member open/high/low returns
relative to the previous close; volume is the sum of member volumes. Series are
stored as synthetic symbols (SECT_ENRG, SECT_INFT, ...) through the normal
market_data upsert, at both 1d and 1h timeframes.

Incremental runs recompute a trailing safety window (default 7 days) anchored
on the last stored synthetic bar before the window, so late-arriving
constituent bars are absorbed and an incremental run equals a full recompute.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

BASE_LEVEL = 100.0

GICS_SECTOR_CODES = {
    "energy": "ENRG",
    "materials": "MATR",
    "industrials": "INDU",
    "consumer discretionary": "COND",
    "consumer staples": "CONS",
    "health care": "HLTH",
    "financials": "FINL",
    "information technology": "INFT",
    "communication services": "TELS",
    "utilities": "UTIL",
    "real estate": "RELS",
}


def sector_symbol(sector_name: str) -> Optional[str]:
    code = GICS_SECTOR_CODES.get(str(sector_name).strip().lower())
    return f"SECT_{code}" if code else None


def sector_symbols() -> List[str]:
    return [f"SECT_{code}" for code in GICS_SECTOR_CODES.values()]


class SectorAggregateService:
    def __init__(self, db_manager, index_symbol: str = "^GSPC", recompute_days: int = 7):
        self.db = db_manager
        self.index_symbol = index_symbol
        self.recompute_days = int(recompute_days)

    # ------------------------------------------------------------------

    def _sector_groups(self) -> Dict[str, List[str]]:
        """synthetic symbol -> constituent symbols, from instruments.sector."""
        instruments = self.db.get_instruments(kind="equity")
        groups: Dict[str, List[str]] = {}
        if instruments.empty or "sector" not in instruments.columns:
            return groups
        for _, row in instruments.dropna(subset=["sector"]).iterrows():
            synthetic = sector_symbol(row["sector"])
            if synthetic:
                groups.setdefault(synthetic, []).append(str(row["symbol"]).upper())
        return groups

    def _load_bars(self, symbols: List[str], timeframe: str, start: Optional[datetime]) -> pd.DataFrame:
        from sqlalchemy import select

        MarketData = self.db.MarketData
        stmt = select(
            MarketData.symbol,
            MarketData.timestamp,
            MarketData.open_price.label("open"),
            MarketData.high_price.label("high"),
            MarketData.low_price.label("low"),
            MarketData.close_price.label("close"),
            MarketData.volume,
        ).where(
            MarketData.timeframe == timeframe,
            MarketData.symbol.in_(symbols),
        )
        if start is not None:
            stmt = stmt.where(MarketData.timestamp >= pd.to_datetime(start, utc=True).to_pydatetime())
        with self.db.engine.connect() as conn:
            frame = pd.read_sql(stmt, conn)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame["symbol"] = frame["symbol"].astype(str).str.upper()
        return frame

    def _membership_mask(self, index: pd.DatetimeIndex, symbols: List[str]) -> pd.DataFrame:
        intervals = self.db.get_membership_intervals(self.index_symbol)
        mask = pd.DataFrame(False, index=index, columns=symbols)
        if intervals.empty:
            # No membership info: treat every symbol with data as a member.
            return pd.DataFrame(True, index=index, columns=symbols)
        dates = pd.Series(index.tz_convert("UTC").date, index=index)
        for _, row in intervals.iterrows():
            symbol = str(row["symbol"]).upper()
            if symbol not in mask.columns:
                continue
            selector = dates >= row["effective_from"].date()
            if pd.notna(row["effective_to"]):
                selector &= dates < row["effective_to"].date()
            mask.loc[selector, symbol] = True
        return mask

    def _anchor(self, synthetic: str, timeframe: str, before: datetime) -> Tuple[float, Optional[pd.Timestamp]]:
        """Last stored synthetic bar strictly before `before` (level, timestamp)."""
        history = self.db.get_market_history(
            symbols=[synthetic],
            end=pd.to_datetime(before, utc=True) - timedelta(microseconds=1),
            timeframe=timeframe,
            limit_rows=1,
        )
        if history.empty:
            return BASE_LEVEL, None
        return float(history.iloc[0]["close_price"]), pd.to_datetime(history.iloc[0]["timestamp"], utc=True)

    # ------------------------------------------------------------------

    def recompute(self, timeframe: str = "1d", full: bool = False) -> dict:
        started_at = datetime.now(timezone.utc)
        groups = self._sector_groups()
        if not groups:
            return {
                "status": "skipped",
                "timeframe": timeframe,
                "message": (
                    "No GICS sector information is stored yet; run a membership refresh "
                    "before computing sector aggregates."
                ),
                "rows_stored": 0,
            }

        all_members = sorted({symbol for members in groups.values() for symbol in members})

        window_start: Optional[datetime] = None
        if not full:
            stored = self.db.get_watermarks(symbols=list(groups), timeframe=timeframe)
            if not stored.empty:
                window_start = (
                    stored["max_timestamp"].min() - timedelta(days=self.recompute_days)
                ).to_pydatetime()

        load_start = None
        if window_start is not None:
            # Buffer so the first recomputed bar has a previous close for returns.
            load_start = window_start - timedelta(days=self.recompute_days)

        bars = self._load_bars(all_members, timeframe, start=load_start)
        if bars.empty:
            return {
                "status": "no_data",
                "timeframe": timeframe,
                "message": f"No constituent {timeframe} bars available for sector aggregation.",
                "rows_stored": 0,
            }

        close_pivot = bars.pivot_table(index="timestamp", columns="symbol", values="close")
        open_pivot = bars.pivot_table(index="timestamp", columns="symbol", values="open")
        high_pivot = bars.pivot_table(index="timestamp", columns="symbol", values="high")
        low_pivot = bars.pivot_table(index="timestamp", columns="symbol", values="low")
        volume_pivot = bars.pivot_table(index="timestamp", columns="symbol", values="volume")

        close_ff = close_pivot.ffill()
        prev_close = close_ff.shift()
        returns = close_ff / prev_close - 1.0
        open_ratio = open_pivot / prev_close - 1.0
        high_ratio = high_pivot / prev_close - 1.0
        low_ratio = low_pivot / prev_close - 1.0

        mask = self._membership_mask(close_pivot.index, list(close_pivot.columns))
        # Only bars where the symbol actually traded contribute.
        contribution_mask = mask & close_pivot.notna()

        output_frames: List[pd.DataFrame] = []
        per_sector: Dict[str, int] = {}
        for synthetic, members in sorted(groups.items()):
            columns = [symbol for symbol in members if symbol in close_pivot.columns]
            if not columns:
                continue
            sector_mask = contribution_mask[columns]
            member_count = sector_mask.sum(axis=1)

            sector_return = returns[columns].where(sector_mask).mean(axis=1)
            sector_open = open_ratio[columns].where(sector_mask).mean(axis=1)
            sector_high = high_ratio[columns].where(sector_mask).mean(axis=1)
            sector_low = low_ratio[columns].where(sector_mask).mean(axis=1)
            sector_volume = volume_pivot[columns].where(sector_mask).sum(axis=1)

            anchor_level, anchor_ts = (BASE_LEVEL, None)
            if window_start is not None:
                anchor_level, anchor_ts = self._anchor(synthetic, timeframe, before=window_start)

            timestamps = close_pivot.index
            selector = member_count > 0
            if anchor_ts is not None:
                selector &= timestamps > anchor_ts
            elif window_start is not None:
                selector &= timestamps >= pd.to_datetime(window_start, utc=True)
            selected = timestamps[selector]
            if selected.empty:
                continue

            chained = anchor_level * (1.0 + sector_return.loc[selected].fillna(0.0)).cumprod()
            prev_level = chained.shift()
            prev_level.iloc[0] = anchor_level

            open_level = prev_level * (1.0 + sector_open.loc[selected].fillna(0.0))
            high_level = prev_level * (1.0 + sector_high.loc[selected].fillna(0.0))
            low_level = prev_level * (1.0 + sector_low.loc[selected].fillna(0.0))
            # The OHLC approximation can be internally inconsistent; clamp.
            high_level = pd.concat([high_level, open_level, chained], axis=1).max(axis=1)
            low_level = pd.concat([low_level, open_level, chained], axis=1).min(axis=1)

            frame = pd.DataFrame(
                {
                    "symbol": synthetic,
                    "timestamp": selected,
                    "open": open_level.values,
                    "high": high_level.values,
                    "low": low_level.values,
                    "close": chained.values,
                    "volume": sector_volume.loc[selected].fillna(0.0).values,
                    "timeframe": timeframe,
                }
            )
            output_frames.append(frame)
            per_sector[synthetic] = len(frame)

        if not output_frames:
            return {
                "status": "no_data",
                "timeframe": timeframe,
                "message": "No sector aggregate bars were produced.",
                "rows_stored": 0,
            }

        output = pd.concat(output_frames, ignore_index=True)
        rows_stored = self.db.insert_market_data(output)

        self.db.upsert_instruments(
            [
                {
                    "symbol": synthetic,
                    "name": f"Equal-weighted {sector.title()} sector index",
                    "sector": sector.title(),
                    "kind": "synthetic",
                    "active": True,
                }
                for sector, code in GICS_SECTOR_CODES.items()
                for synthetic in [f"SECT_{code}"]
                if synthetic in per_sector
            ]
        )

        duration_s = (datetime.now(timezone.utc) - started_at).total_seconds()
        return {
            "status": "ok",
            "timeframe": timeframe,
            "full": bool(full or window_start is None),
            "rows_stored": int(rows_stored),
            "rows_by_sector": per_sector,
            "duration_s": round(duration_s, 2),
            "message": (
                f"Stored {rows_stored} sector aggregate bars across {len(per_sector)} sectors "
                f"({timeframe}, {'full' if full or window_start is None else 'incremental'} recompute)."
            ),
        }
