import numpy as np
import pandas as pd
import pytest

from ingestion.yfinance_provider import YFinanceProvider

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _multiindex_download_frame():
    """Shape of yf.download(group_by='ticker') for two tickers; BBB has two
    pre-listing all-NaN rows."""
    index = pd.date_range("2024-01-02", periods=5, freq="D", name="Date")
    columns = pd.MultiIndex.from_product([["AAA", "BBB"], FIELDS])
    frame = pd.DataFrame(index=index, columns=columns, dtype=float)
    for i, field in enumerate(FIELDS):
        frame[("AAA", field)] = [10 + i, 11 + i, 12 + i, 13 + i, 14 + i]
        frame[("BBB", field)] = [np.nan, np.nan, 20 + i, 21 + i, 22 + i]
    return frame


def _single_history_frame():
    """Shape of yf.Ticker(...).history() (tz-aware index, plain columns)."""
    index = pd.date_range("2024-01-02", periods=3, freq="D", tz="America/New_York", name="Date")
    return pd.DataFrame(
        {field: [30.0 + i, 31.0 + i, 32.0 + i] for i, field in enumerate(FIELDS)},
        index=index,
    )


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr("ingestion.yfinance_provider.time.sleep", lambda seconds: None)
    return YFinanceProvider(batch_size=25, inter_batch_delay_s=0, max_retries=1)


def test_multiindex_frame_normalized(provider, monkeypatch):
    monkeypatch.setattr(
        "ingestion.yfinance_provider.yf.download",
        lambda tickers, **kwargs: _multiindex_download_frame(),
    )
    result = provider.fetch_bars(["AAA", "BBB"], interval="1d", period="5d")
    frame = result.frame

    assert list(frame.columns) == [
        "symbol", "timestamp", "open", "high", "low", "close", "volume", "timeframe",
    ]
    assert sorted(result.report.ok) == ["AAA", "BBB"]
    assert not result.report.failed
    assert frame["timestamp"].dt.tz is not None  # tz-aware UTC
    assert set(frame["timeframe"]) == {"1d"}
    # BBB's two pre-listing NaN rows dropped.
    assert len(frame[frame["symbol"] == "AAA"]) == 5
    assert len(frame[frame["symbol"] == "BBB"]) == 3


def test_single_symbol_fallback_and_empty(provider, monkeypatch):
    # Batch download only returns AAA; CCC falls back to Ticker.history and DDD is empty.
    def fake_download(tickers, **kwargs):
        frame = _multiindex_download_frame()
        return frame[["AAA"]]

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            if self.symbol == "CCC":
                return _single_history_frame()
            return pd.DataFrame()

    monkeypatch.setattr("ingestion.yfinance_provider.yf.download", fake_download)
    monkeypatch.setattr("ingestion.yfinance_provider.yf.Ticker", FakeTicker)

    result = provider.fetch_bars(["AAA", "CCC", "DDD"], interval="1d", period="5d")
    assert sorted(result.report.ok) == ["AAA", "CCC"]
    assert result.report.empty == ["DDD"]
    ccc = result.frame[result.frame["symbol"] == "CCC"]
    assert len(ccc) == 3
    assert str(ccc["timestamp"].dt.tz) == "UTC"


def test_retry_then_failure_is_reported(provider, monkeypatch):
    calls = {"n": 0}

    def flaky_download(tickers, **kwargs):
        calls["n"] += 1
        raise ConnectionError("boom")

    class FailingTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            raise ConnectionError("boom")

    monkeypatch.setattr("ingestion.yfinance_provider.yf.download", flaky_download)
    monkeypatch.setattr("ingestion.yfinance_provider.yf.Ticker", FailingTicker)
    provider.max_retries = 2

    result = provider.fetch_bars(["AAA"], interval="1d", period="5d")
    assert result.frame.empty
    assert "AAA" in result.report.failed
    assert calls["n"] == 2  # batch retried


def test_intraday_start_clamped(provider):
    kwargs = provider._history_request_kwargs(period=None, interval="1h", start="2000-01-01", end=None)
    clamped = pd.Timestamp(kwargs["start"])
    assert clamped > pd.Timestamp("2000-01-01")
    assert kwargs["auto_adjust"] is True

    # Daily requests are never clamped.
    kwargs_daily = provider._history_request_kwargs(period=None, interval="1d", start="2000-01-01", end=None)
    assert kwargs_daily["start"] == "2000-01-01"
