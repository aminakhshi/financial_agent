import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# non-interactive backend to prevent issues with dashboard when running in the background.
matplotlib.use("Agg")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.settings import (
    DATABASE_CONFIG,
    MARKET_CONFIG,
    should_use_database_config,
    sqlite_fallback_enabled,
)
from src.data.database import DatabaseManager


class FinancialDashboard:
    """Dashboard helpers for market history, predictions, and coverage."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def load_data(self, symbol, timeframe="1h", rows=168, pred_limit=24, use_predictions=True):
        market_data = self.db_manager.get_latest_data(
            symbol,
            limit_rows=rows,
            timeframe=timeframe,
            ascending=True,
        )

        predictions = pd.DataFrame()
        if use_predictions:
            predictions = self.db_manager.get_latest_predictions(
                symbol,
                limit_rows=pred_limit,
                timeframe=timeframe,
                ascending=True,
            )

        return market_data, predictions

    def load_market_coverage(self, symbols=None, timeframe=None):
        return self.db_manager.get_market_coverage(symbols=symbols, timeframe=timeframe)

    def load_prediction_coverage(self, symbols=None, timeframe=None):
        return self.db_manager.get_prediction_coverage(symbols=symbols, timeframe=timeframe)

    def list_stored_symbols(self, timeframe=None):
        return self.db_manager.get_available_symbols(timeframe=timeframe)

    def create_price_chart(self, market_data, predictions, symbol, timeframe):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=market_data["timestamp"],
                y=market_data["close_price"],
                mode="lines",
                name=f"{timeframe} close",
                line=dict(color="dodgerblue", width=2),
            )
        )

        if not predictions.empty:
            fig.add_trace(
                go.Scatter(
                    x=predictions["prediction_timestamp"],
                    y=predictions["predicted_price"],
                    mode="markers+lines",
                    name="Predicted price",
                    line=dict(color="teal", width=2, dash="dash"),
                    marker=dict(size=8, symbol="x-thin"),
                )
            )

        fig.update_layout(
            title=f"{symbol} price history",
            xaxis_title="Timestamp",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def create_technical_indicators_chart(self, market_data, symbol):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=market_data["timestamp"],
                y=market_data["close_price"],
                name="Close price",
                line=dict(color="lightskyblue", width=1),
            )
        )
        if "sma_20" in market_data.columns and market_data["sma_20"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=market_data["timestamp"],
                    y=market_data["sma_20"],
                    name="20-period SMA",
                    line=dict(color="orange"),
                )
            )

        if "bollinger_upper" in market_data.columns and market_data["bollinger_upper"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=market_data["timestamp"],
                    y=market_data["bollinger_upper"],
                    name="Bollinger upper",
                    line=dict(color="gray", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=market_data["timestamp"],
                    y=market_data["bollinger_lower"],
                    name="Bollinger lower",
                    line=dict(color="gray", dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(128,128,128,0.1)",
                )
            )

        fig.update_layout(
            title=f"{symbol} technical indicators",
            xaxis_title="Timestamp",
            yaxis_title="Price (USD)",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def calculate_metrics(self, market_data, predictions):
        if market_data.empty:
            return {
                "current_price": 0.0,
                "price_change_pct": 0.0,
                "avg_volume": 0.0,
                "prediction_accuracy": 0.0,
                "latest_prediction": None,
            }

        current_price = float(market_data["close_price"].iloc[-1])
        first_price = float(market_data["close_price"].iloc[0])
        price_change_pct = ((current_price - first_price) / first_price * 100.0) if first_price else 0.0
        avg_volume = float(market_data["volume"].mean())

        accuracy = 0.0
        if not predictions.empty and "actual_price" in predictions.columns:
            predictions_with_actual = predictions.dropna(subset=["actual_price"])
            if not predictions_with_actual.empty:
                mape = np.mean(
                    np.abs(
                        (
                            predictions_with_actual["actual_price"] - predictions_with_actual["predicted_price"]
                        )
                        / predictions_with_actual["actual_price"]
                    )
                ) * 100.0
                accuracy = max(0.0, 100.0 - float(mape))

        latest_prediction = None
        if not predictions.empty:
            latest_prediction = float(predictions["predicted_price"].iloc[-1])

        return {
            "current_price": current_price,
            "price_change_pct": price_change_pct,
            "avg_volume": avg_volume,
            "prediction_accuracy": accuracy,
            "latest_prediction": latest_prediction,
        }


def _resolve_universe_symbols(universe_name, stored_symbols, timeframe):
    mapping = {
        "Stored symbols": stored_symbols,
        "Default watchlist": MARKET_CONFIG.get("default_symbols", []),
        "S&P 500": MARKET_CONFIG.get("sp500_symbols", []),
        "NASDAQ": MARKET_CONFIG.get("nasdaq_symbols", []),
    }
    selected = mapping.get(universe_name, stored_symbols)
    if universe_name == "Stored symbols":
        return selected
    return sorted({str(symbol).strip().upper() for symbol in selected if str(symbol).strip()})


def _format_timestamp(value):
    if value is None or pd.isna(value):
        return "n/a"
    return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d %H:%M UTC")


def main():
    st.set_page_config(
        page_title="Financial Market Monitor",
        layout="wide",
    )

    st.title("Financial Market Monitor")
    st.caption("Inspect stored market history, model coverage, and recent predictions from the local SQL database.")

    @st.cache_resource
    def get_db_manager():
        sqlite_fallback = sqlite_fallback_enabled(default=True)
        db_config = DATABASE_CONFIG if should_use_database_config() else None
        db_manager = DatabaseManager(db_config, use_sqlite_fallback=sqlite_fallback)
        db_manager.create_tables()
        return db_manager

    db_manager = get_db_manager()
    dashboard = FinancialDashboard(db_manager)

    st.sidebar.header("Filters")
    timeframe = st.sidebar.selectbox("Stored data timeframe", ["1h", "1d"], index=0)
    universe_name = st.sidebar.selectbox(
        "Universe",
        ["Stored symbols", "Default watchlist", "S&P 500", "NASDAQ"],
        index=0,
    )
    show_stored_only = st.sidebar.checkbox("Only show symbols with stored data", value=True)
    rows_to_plot = st.sidebar.slider("Rows to plot", min_value=24, max_value=5000, value=336, step=24)
    prediction_rows = st.sidebar.slider("Prediction rows", min_value=1, max_value=200, value=48)
    symbol_filter = st.sidebar.text_input("Filter symbols", "").strip().upper()
    auto_refresh = st.sidebar.checkbox("Auto refresh every 30 seconds", value=False)

    stored_symbols = dashboard.list_stored_symbols(timeframe=timeframe)
    universe_symbols = _resolve_universe_symbols(universe_name, stored_symbols, timeframe)
    if show_stored_only:
        stored_set = set(stored_symbols)
        universe_symbols = [symbol for symbol in universe_symbols if symbol in stored_set]
    if symbol_filter:
        universe_symbols = [symbol for symbol in universe_symbols if symbol_filter in symbol]

    if not universe_symbols:
        st.warning("No symbols are available for the current filters.")
        return

    selected_symbol = st.sidebar.selectbox("Symbol", universe_symbols, index=0)

    market_coverage = dashboard.load_market_coverage(
        symbols=None if universe_name == "Stored symbols" else universe_symbols,
        timeframe=timeframe,
    )
    prediction_coverage = dashboard.load_prediction_coverage(
        symbols=None if universe_name == "Stored symbols" else universe_symbols,
        timeframe=timeframe,
    )

    market_data, predictions = dashboard.load_data(
        selected_symbol,
        timeframe=timeframe,
        rows=rows_to_plot,
        pred_limit=prediction_rows,
        use_predictions=True,
    )

    if market_data.empty:
        st.warning(
            f"No stored {timeframe} bars are available for {selected_symbol}. Run the relevant backfill first."
        )
        return

    metrics = dashboard.calculate_metrics(market_data, predictions)
    selected_market_coverage = market_coverage[market_coverage["symbol"] == selected_symbol]
    selected_prediction_coverage = prediction_coverage[prediction_coverage["symbol"] == selected_symbol]

    tabs = st.tabs(["Overview", "Coverage", "Recent Rows"])

    with tabs[0]:
        st.subheader(f"{selected_symbol} overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current price", f"${metrics['current_price']:.2f}")
        col2.metric("Change in view", f"{metrics['price_change_pct']:.2f}%")
        col3.metric("Average volume", f"{metrics['avg_volume'] / 1_000_000:.2f}M")
        col4.metric(
            "Latest prediction",
            f"${metrics['latest_prediction']:.2f}" if metrics["latest_prediction"] is not None else "n/a",
        )
        col5.metric("Prediction accuracy", f"{metrics['prediction_accuracy']:.2f}%")

        coverage_col1, coverage_col2, coverage_col3 = st.columns(3)
        if not selected_market_coverage.empty:
            coverage_row = selected_market_coverage.iloc[0]
            coverage_col1.metric("Stored rows", int(coverage_row["row_count"]))
            coverage_col2.metric("First bar", _format_timestamp(coverage_row["first_timestamp"]))
            coverage_col3.metric("Last bar", _format_timestamp(coverage_row["last_timestamp"]))
        else:
            coverage_col1.metric("Stored rows", 0)
            coverage_col2.metric("First bar", "n/a")
            coverage_col3.metric("Last bar", "n/a")

        if predictions.empty:
            st.info("No prediction history is stored yet for this symbol.")

        st.plotly_chart(
            dashboard.create_price_chart(market_data, predictions, selected_symbol, timeframe),
            use_container_width=True,
        )
        st.plotly_chart(
            dashboard.create_technical_indicators_chart(market_data, selected_symbol),
            use_container_width=True,
        )

        if not selected_prediction_coverage.empty:
            pred_row = selected_prediction_coverage.iloc[0]
            st.caption(
                "Prediction coverage: "
                f"{int(pred_row['prediction_count'])} stored, "
                f"{int(pred_row['evaluated_count'])} evaluated, "
                f"{int(pred_row['pending_actual_count'])} waiting for realized prices."
            )

    with tabs[1]:
        st.subheader("Universe coverage")
        requested_symbol_count = len(universe_symbols)
        stored_symbol_count = int(market_coverage["symbol"].nunique()) if not market_coverage.empty else 0
        predicted_symbol_count = int(prediction_coverage["symbol"].nunique()) if not prediction_coverage.empty else 0
        total_rows = int(market_coverage["row_count"].sum()) if not market_coverage.empty else 0

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Requested symbols", requested_symbol_count)
        summary_col2.metric("Symbols with stored bars", stored_symbol_count)
        summary_col3.metric("Symbols with predictions", predicted_symbol_count)
        summary_col4.metric("Stored rows", total_rows)

        if not market_coverage.empty:
            st.caption(
                f"Stored {timeframe} coverage spans "
                f"{_format_timestamp(market_coverage['first_timestamp'].min())} "
                f"through {_format_timestamp(market_coverage['last_timestamp'].max())}."
            )
            coverage_table = market_coverage.copy()
            coverage_table["first_timestamp"] = coverage_table["first_timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
            coverage_table["last_timestamp"] = coverage_table["last_timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
            st.dataframe(coverage_table, use_container_width=True, hide_index=True)
        else:
            st.info("No market coverage is stored for this timeframe yet.")

        st.subheader("Prediction coverage")
        if not prediction_coverage.empty:
            prediction_table = prediction_coverage.copy()
            prediction_table["first_prediction_timestamp"] = prediction_table["first_prediction_timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            prediction_table["last_prediction_timestamp"] = prediction_table["last_prediction_timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            prediction_table["coverage_pct"] = prediction_table["coverage_pct"].map(lambda value: f"{value:.2f}%")
            st.dataframe(prediction_table, use_container_width=True, hide_index=True)
        else:
            st.info("No prediction coverage is stored yet.")

    with tabs[2]:
        st.subheader("Recent rows")
        st.caption(f"Showing the most recent stored {timeframe} rows for {selected_symbol}.")
        market_preview = market_data.copy()
        market_preview["timestamp"] = market_preview["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        st.dataframe(market_preview.tail(50), use_container_width=True, hide_index=True)

        if not predictions.empty:
            prediction_preview = predictions.copy()
            prediction_preview["prediction_timestamp"] = prediction_preview["prediction_timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            if "created_at" in prediction_preview.columns:
                prediction_preview["created_at"] = prediction_preview["created_at"].dt.strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            st.caption("Recent prediction rows")
            st.dataframe(prediction_preview.tail(50), use_container_width=True, hide_index=True)

    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
