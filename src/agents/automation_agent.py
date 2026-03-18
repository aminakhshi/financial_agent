import asyncio
import os
from datetime import datetime, timedelta
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from crewai import Agent

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging

    logger = logging.getLogger(__name__)

try:
    from langchain.llms.base import LLM
except ImportError:
    from langchain_core.language_models.llms import LLM


class AutomationAgent:
    """Deterministic market pipeline orchestration for collection, training, and reporting."""

    DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    def __init__(self, config, db_manager, data_collector, ml_predictor):
        self.config = config
        self.db_manager = db_manager
        self.data_collector = data_collector
        self.ml_predictor = ml_predictor
        self.crewai_enabled = os.getenv("DISABLE_CREWAI", "false").strip().lower() != "true"
        configured_defaults = self.config["MARKET_CONFIG"].get("default_symbols", self.DEFAULT_SYMBOLS)
        self.default_symbols = self._resolve_symbols(configured_defaults)
        self.exchange_lookup = self._build_exchange_lookup()
        self.llm = self._build_llm()

        if self.crewai_enabled:
            self.setup_crew_agents()
        else:
            logger.info("CrewAI orchestration disabled. Using deterministic pipeline mode.")

    def _build_llm(self):
        try:
            from langchain_openai import ChatOpenAI

            model_name = self.config["LLM_CONFIG"]["model_name"]
            if model_name.startswith("ollama/"):
                model_name = model_name.split("ollama/", 1)[1]

            base_url = self.config["LLM_CONFIG"]["base_url"].rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"

            openai_api_key = (
                self.config["API_KEYS"].get("OPENAI_API_KEY")
                or self.config["API_KEYS"].get("OPENAI")
                or os.getenv("OPENAI_API_KEY")
                or "testapikey"
            )

            llm = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                openai_api_key=openai_api_key,
                timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "120")),
                temperature=self.config["LLM_CONFIG"].get("temperature", 0.1),
            )
            logger.info(f"Connected model {model_name} using {base_url}.")
            return llm
        except Exception as openai_error:
            try:
                from langchain_ollama import ChatOllama

                llm = ChatOllama(
                    model=self.config["LLM_CONFIG"]["model_name"].replace("ollama/", ""),
                    base_url=self.config["LLM_CONFIG"]["base_url"],
                    request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "120")),
                )
                logger.info(
                    f"Connected fallback model {self.config['LLM_CONFIG']['model_name']} using "
                    f"{self.config['LLM_CONFIG']['base_url']}."
                )
                return llm
            except Exception as ollama_error:
                logger.warning(
                    "LLM connection is unavailable. The pipeline will continue without LLM-backed summaries. "
                    f"OpenAI-compatible error: {openai_error}. Ollama error: {ollama_error}."
                )

                class SimpleMockLLM(LLM):
                    model_name: str = "mock-llm"
                    provider: str = "mock-provider"

                    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                        return "LLM output is not available in this environment."

                    @property
                    def _llm_type(self) -> str:
                        return "simple_mock"

                    @property
                    def _identifying_params(self) -> Dict[str, Any]:
                        return {"model_name": self.model_name, "provider": self.provider}

                return SimpleMockLLM()

    def _build_exchange_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for market_name, config_key in (("SP500", "sp500_symbols"), ("NASDAQ", "nasdaq_symbols")):
            for symbol in self.config["MARKET_CONFIG"].get(config_key, []):
                lookup.setdefault(symbol.upper(), market_name)
        return lookup

    def _resolve_symbols(self, symbols: Optional[Iterable[str]] = None) -> List[str]:
        source = symbols or self.DEFAULT_SYMBOLS
        unique_symbols: List[str] = []
        seen = set()
        for symbol in source:
            cleaned = str(symbol).strip().upper()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_symbols.append(cleaned)
        return unique_symbols

    def _resolve_market_universe(
        self,
        universe: str = "default",
        symbols: Optional[Iterable[str]] = None,
    ) -> List[str]:
        if symbols:
            resolved = self._resolve_symbols(symbols)
            if resolved:
                return resolved

        normalized_universe = (universe or "default").strip().lower()
        configured_sp500 = self.config["MARKET_CONFIG"].get("sp500_symbols", [])
        configured_nasdaq = self.config["MARKET_CONFIG"].get("nasdaq_symbols", [])
        universe_map = {
            "default": self.default_symbols,
            "watchlist": self.default_symbols,
            "sp500": configured_sp500,
            "s&p500": configured_sp500,
            "sp500_full": configured_sp500,
            "nasdaq": configured_nasdaq,
            "all": configured_sp500 + configured_nasdaq,
            "configured": configured_sp500 + configured_nasdaq,
            "configured_all": configured_sp500 + configured_nasdaq,
        }

        if normalized_universe not in universe_map:
            raise ValueError(
                "Unsupported universe. Use one of: default, sp500, nasdaq, all, configured."
            )

        resolved = self._resolve_symbols(universe_map[normalized_universe])
        if not resolved:
            raise ValueError(f"No symbols are configured for universe '{normalized_universe}'.")
        return resolved

    def _model_factory(self):
        return self.ml_predictor.__class__(self.config)

    def _normalize_market_frame(self, market_data: pd.DataFrame) -> pd.DataFrame:
        if market_data.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rename_map = {
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
        }
        normalized = market_data.rename(columns=rename_map).copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
        normalized = normalized.sort_values("timestamp").reset_index(drop=True)
        return normalized[["timestamp", "open", "high", "low", "close", "volume"]]

    def _next_prediction_timestamp(self, latest_timestamp, interval: str = "1h") -> pd.Timestamp:
        latest_timestamp = pd.to_datetime(latest_timestamp, utc=True)
        if interval.endswith("h"):
            return latest_timestamp + timedelta(hours=int(interval[:-1] or "1"))
        if interval.endswith("d"):
            return latest_timestamp + timedelta(days=int(interval[:-1] or "1"))
        raise ValueError(f"Unsupported interval: {interval}")

    def _default_training_period(self, interval: str) -> str:
        if interval.endswith("d"):
            return self.config["MARKET_CONFIG"].get("daily_model_training_period", "10y")
        return self.config["MARKET_CONFIG"].get("hourly_model_training_period", "6mo")

    def _default_refresh_period(self, interval: str) -> str:
        if interval.endswith("d"):
            return self.config["MARKET_CONFIG"].get("daily_prediction_refresh_period", "1y")
        return self.config["MARKET_CONFIG"].get("hourly_prediction_refresh_period", "5d")

    def _history_limit_for_interval(self, interval: str) -> int:
        sequence_length = self.ml_predictor.get_sequence_length(interval)
        if interval.endswith("d"):
            return max(sequence_length * 8, 252 * 10)
        return max(sequence_length * 8, int(self.config["MARKET_CONFIG"].get("lookback_days", 365)) * 24)

    def _recent_limit_for_interval(self, interval: str) -> int:
        sequence_length = self.ml_predictor.get_sequence_length(interval)
        if interval.endswith("d"):
            return max(sequence_length + 90, 320)
        return max(sequence_length + 60, 240)

    def _calculate_confidence(self, metadata: Dict[str, Any], current_price: float, predicted_price: float) -> float:
        rmse = metadata.get("test_rmse") or metadata.get("train_rmse")
        if rmse is None or current_price <= 0:
            baseline = 55.0
        else:
            error_ratio = abs(float(rmse)) / max(abs(current_price), 1e-9)
            baseline = 90.0 - min(55.0, error_ratio * 1800.0)

        move_ratio = abs(predicted_price - current_price) / max(abs(current_price), 1e-9)
        confidence = baseline - min(15.0, move_ratio * 250.0)
        return round(max(10.0, min(95.0, confidence)), 2)

    def _calibrate_prediction(
        self,
        recent_frame: pd.DataFrame,
        current_price: float,
        raw_predicted_price: float,
        interval: str,
    ) -> Dict[str, float]:
        raw_change_pct = ((raw_predicted_price - current_price) / current_price) * 100 if current_price else 0.0
        recent_returns = recent_frame["close"].pct_change().dropna()
        realized_volatility_pct = float(recent_returns.tail(20).std() * 100.0) if not recent_returns.empty else 1.0

        if interval.endswith("d"):
            min_cap_pct, max_cap_pct, smoothing = 2.5, 8.0, 0.55
        else:
            min_cap_pct, max_cap_pct, smoothing = 1.0, 6.0, 0.65

        cap_pct = min(max_cap_pct, max(min_cap_pct, realized_volatility_pct * 3.0))
        clipped_change_pct = max(-cap_pct, min(cap_pct, raw_change_pct))
        calibrated_change_pct = clipped_change_pct * smoothing
        calibrated_price = current_price * (1.0 + calibrated_change_pct / 100.0)

        return {
            "raw_predicted_price": float(raw_predicted_price),
            "raw_predicted_change_pct": round(float(raw_change_pct), 4),
            "predicted_price": float(calibrated_price),
            "predicted_change_pct": round(float(calibrated_change_pct), 4),
            "volatility_cap_pct": round(float(cap_pct), 4),
        }

    def _format_prediction_message(self, prediction: Dict[str, Any]) -> str:
        direction = prediction["direction"]
        return (
            f"{prediction['symbol']} is trading at ${prediction['current_price']:.2f}. "
            f"The next {prediction['interval']} model estimate is ${prediction['predicted_price']:.2f}, "
            f"{direction} {abs(prediction['predicted_change_pct']):.2f}%. "
            f"Confidence is {prediction['confidence_score']:.1f}%."
        )

    def _format_report_message(self, items: List[Dict[str, Any]], timestamp: str) -> str:
        if not items:
            return "No market data is available for the requested symbols."

        top_mover = max(items, key=lambda item: abs(item.get("predicted_change_pct", 0.0)))
        return (
            f"Market summary generated at {timestamp}. "
            f"Tracked {len(items)} symbols. "
            f"The largest projected move is {top_mover['symbol']} at {top_mover['predicted_change_pct']:+.2f}%."
        )

    def _build_status_message(self, action: str, symbols: List[str], failures: List[Dict[str, Any]]) -> str:
        if failures:
            return (
                f"{action} completed with partial results for {len(symbols)} symbols. "
                f"{len(failures)} symbol runs need attention."
            )
        return f"{action} completed for {len(symbols)} symbols."

    def _clean_json_value(self, value):
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if pd.isna(value):
            return None
        return value

    def _serialize_market_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        serialized = {key: self._clean_json_value(value) for key, value in row.items()}
        if serialized.get("timestamp") is not None:
            serialized["timestamp"] = pd.to_datetime(serialized["timestamp"], utc=True).isoformat()
        if serialized.get("created_at") is not None:
            serialized["created_at"] = pd.to_datetime(serialized["created_at"], utc=True).isoformat()
        return serialized

    def _serialize_prediction_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        serialized = {key: self._clean_json_value(value) for key, value in row.items()}
        if serialized.get("prediction_timestamp") is not None:
            serialized["prediction_timestamp"] = pd.to_datetime(
                serialized["prediction_timestamp"],
                utc=True,
            ).isoformat()
        if serialized.get("created_at") is not None:
            serialized["created_at"] = pd.to_datetime(serialized["created_at"], utc=True).isoformat()
        return serialized

    def _summarize_prediction_metrics(self, evaluated: pd.DataFrame) -> Dict[str, Optional[float]]:
        if evaluated.empty:
            return {
                "mae": None,
                "rmse": None,
                "mape": None,
                "accuracy_pct": None,
            }

        absolute_percentage_error = evaluated["absolute_percentage_error"].dropna()
        mape = float(absolute_percentage_error.mean()) if not absolute_percentage_error.empty else None
        accuracy_pct = None if mape is None else max(0.0, 100.0 - mape)
        return {
            "mae": round(float(evaluated["absolute_error"].mean()), 6),
            "rmse": round(float(sqrt((evaluated["error"] ** 2).mean())), 6),
            "mape": None if mape is None else round(mape, 6),
            "accuracy_pct": None if accuracy_pct is None else round(accuracy_pct, 6),
        }

    def _serialize_prediction_for_report(self, prediction: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        predicted_change_pct = prediction.get("predicted_change_pct")
        if predicted_change_pct is None:
            predicted_price = float(prediction["predicted_price"])
            predicted_change_pct = ((predicted_price - current_price) / current_price) * 100 if current_price else 0.0

        return {
            "prediction_timestamp": pd.to_datetime(prediction["prediction_timestamp"], utc=True).isoformat(),
            "predicted_price": float(prediction["predicted_price"]),
            "confidence_score": float(prediction["confidence_score"]),
            "predicted_change_pct": round(float(predicted_change_pct), 4),
        }

    def _prediction_is_fresh(
        self,
        prediction_timestamp: Any,
        latest_market_timestamp: Any,
    ) -> bool:
        prediction_ts = pd.to_datetime(prediction_timestamp, utc=True)
        market_ts = pd.to_datetime(latest_market_timestamp, utc=True)
        return prediction_ts > market_ts

    def setup_crew_agents(self):
        """Keep CrewAI agents available for future use without driving the critical path."""
        self.data_agent = Agent(
            role="Data Collection Specialist",
            goal="Collect and validate financial market data from various sources",
            backstory=(
                "You specialize in financial market data quality and keep market records current, "
                "complete, and consistent."
            ),
            llm=self.llm,
            verbose=True,
        )
        self.ml_agent = Agent(
            role="Machine Learning Engineer",
            goal="Train and evaluate time-series forecasting models",
            backstory=(
                "You focus on model quality, reproducibility, and clear model performance reporting."
            ),
            llm=self.llm,
            verbose=True,
        )
        self.prediction_agent = Agent(
            role="Market Prediction Analyst",
            goal="Summarize model outputs and risk signals",
            backstory=(
                "You translate short-term market model outputs into concise operational summaries."
            ),
            llm=self.llm,
            verbose=True,
        )

    def collect_market_data(
        self,
        symbols: Optional[Iterable[str]] = None,
        period: str = "5d",
        interval: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        symbols = self._resolve_symbols(symbols)
        range_label = f"start={start}, end={end or 'latest'}" if start or end else f"period {period}"
        logger.info(f"Collecting {interval} market data for {', '.join(symbols)} using {range_label}.")

        total_rows = 0
        total_symbols = set()
        rows_by_symbol: Dict[str, int] = {}
        total_actuals_updated = 0
        batch_count = 0

        for batch_data in self.data_collector.iter_yfinance_data_batches(
            symbols,
            period=period,
            interval=interval,
            start=start,
            end=end,
            batch_size=batch_size,
        ):
            if batch_data.empty:
                continue

            market_data = batch_data.copy()
            market_data["symbol"] = market_data["symbol"].astype(str).str.upper()
            market_data["exchange"] = market_data["symbol"].map(self.exchange_lookup).fillna("US")
            market_data["timeframe"] = interval
            self.db_manager.insert_market_data(market_data)

            batch_symbols = sorted({str(symbol).upper() for symbol in market_data["symbol"].tolist()})
            batch_actuals_updated = self.db_manager.sync_prediction_actuals(batch_symbols, timeframe=interval)

            batch_rows_by_symbol = {
                symbol: int(count) for symbol, count in market_data.groupby("symbol").size().to_dict().items()
            }
            for symbol, count in batch_rows_by_symbol.items():
                rows_by_symbol[symbol] = rows_by_symbol.get(symbol, 0) + count
                total_symbols.add(symbol)

            total_rows += int(len(market_data))
            total_actuals_updated += int(batch_actuals_updated)
            batch_count += 1
            logger.info(
                "Stored provider batch {}: {} rows across {} symbols. Cumulative rows: {}.",
                batch_count,
                len(market_data),
                len(batch_rows_by_symbol),
                total_rows,
            )

        if total_rows == 0:
            return {
                "status": "no_data",
                "symbols": symbols,
                "rows_collected": 0,
                "timeframe": interval,
                "message": "No market data was returned by the provider.",
                "timestamp": datetime.utcnow().isoformat(),
            }

        message = (
            f"Collected {total_rows} rows across {len(rows_by_symbol)} symbols in {batch_count} provider batches. "
            f"Updated {total_actuals_updated} prediction records with realized prices."
        )
        return {
            "status": "ok",
            "symbols": symbols,
            "period": period,
            "interval": interval,
            "start": start,
            "end": end,
            "rows_collected": int(total_rows),
            "rows_by_symbol": rows_by_symbol,
            "actuals_updated": int(total_actuals_updated),
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def collect_market_universe(
        self,
        universe: str = "all",
        period: str = "1mo",
        interval: str = "1h",
        symbols: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        resolved_symbols = self._resolve_market_universe(universe=universe, symbols=symbols)
        result = self.collect_market_data(
            resolved_symbols,
            period=period,
            interval=interval,
            start=start,
            end=end,
            batch_size=batch_size,
        )
        result["universe"] = (universe or "custom").strip().lower()
        result["requested_symbol_count"] = len(resolved_symbols)
        result["stored_symbol_count"] = len(result.get("rows_by_symbol", {}))
        result["message"] = (
            f"Collected {result['rows_collected']} rows for {result['stored_symbol_count']} of "
            f"{len(resolved_symbols)} configured symbols in the {result['universe']} universe. "
            f"Updated {result['actuals_updated']} prediction records with realized prices."
        )
        return result

    def backfill_daily_history(
        self,
        universe: str = "sp500",
        symbols: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        start = start or self.config["MARKET_CONFIG"].get("sp500_daily_backfill_start", "1991-01-01")
        result = self.collect_market_universe(
            universe=universe,
            symbols=symbols,
            interval="1d",
            start=start,
            end=end,
            batch_size=batch_size,
        )
        result["backfill_type"] = "daily"
        result["message"] = (
            f"Stored daily backfill rows from {start} through {end or 'latest'} for "
            f"{result['stored_symbol_count']} symbols."
        )
        return result

    def backfill_hourly_history(
        self,
        universe: str = "sp500",
        symbols: Optional[Iterable[str]] = None,
        period: Optional[str] = None,
        end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        period = period or self.config["MARKET_CONFIG"].get("sp500_hourly_backfill_period", "6mo")
        result = self.collect_market_universe(
            universe=universe,
            symbols=symbols,
            period=period,
            interval="1h",
            end=end,
            batch_size=batch_size,
        )
        result["backfill_type"] = "hourly"
        result["message"] = (
            f"Stored hourly backfill rows for the last {period} for {result['stored_symbol_count']} symbols."
        )
        return result

    def backfill_sp500_history(
        self,
        daily_start: Optional[str] = None,
        daily_end: Optional[str] = None,
        hourly_period: Optional[str] = None,
        hourly_end: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        daily = self.backfill_daily_history(
            universe="sp500",
            start=daily_start,
            end=daily_end,
            batch_size=batch_size,
        )
        hourly = self.backfill_hourly_history(
            universe="sp500",
            period=hourly_period,
            end=hourly_end,
            batch_size=batch_size,
        )
        return {
            "universe": "sp500",
            "daily": daily,
            "hourly": hourly,
            "message": "Completed S&P 500 daily and hourly history backfills.",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def train_model(
        self,
        symbol: str,
        history_period: str = "6mo",
        interval: str = "1h",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        effective_history_period = (
            self._default_training_period(interval) if history_period == "6mo" and interval.endswith("d") else history_period
        )
        min_rows = self.ml_predictor.get_sequence_length(interval) + 24
        history_limit = self._history_limit_for_interval(interval)
        market_data = self.db_manager.get_latest_data(
            symbol,
            limit_rows=history_limit,
            timeframe=interval,
            ascending=True,
        )

        if force_refresh or market_data.empty or len(market_data) < min_rows:
            self.collect_market_data([symbol], period=effective_history_period, interval=interval)
            market_data = self.db_manager.get_latest_data(
                symbol,
                limit_rows=history_limit,
                timeframe=interval,
                ascending=True,
            )

        if market_data.empty or len(market_data) < min_rows:
            raise ValueError(f"Not enough market history is available to train {symbol}.")

        predictor = self._model_factory()
        training_frame = self._normalize_market_frame(market_data)
        metrics = predictor.train(training_frame, symbol, interval=interval)
        message = (
            f"Trained the {symbol} {interval} model on {metrics['training_rows']} rows. "
            f"Test accuracy is {metrics.get('test_accuracy_pct', 0.0):.2f}%."
        )
        logger.info(message)
        return {
            "symbol": symbol,
            "interval": interval,
            "training_rows": metrics["training_rows"],
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_mae": metrics["train_mae"],
            "test_mae": metrics["test_mae"],
            "train_mape": metrics.get("train_mape"),
            "test_mape": metrics.get("test_mape"),
            "test_accuracy_pct": metrics.get("test_accuracy_pct"),
            "directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
            "model_version": metrics["model_version"],
            "trained_at": metrics["trained_at"],
            "message": message,
        }

    def train_models(
        self,
        symbols: Optional[Iterable[str]] = None,
        history_period: str = "6mo",
        interval: str = "1h",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        symbols = self._resolve_symbols(symbols)
        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for symbol in symbols:
            try:
                completed.append(
                    self.train_model(
                        symbol,
                        history_period=history_period,
                        interval=interval,
                        force_refresh=force_refresh,
                    )
                )
            except Exception as exc:
                failed.append({"symbol": symbol, "error": str(exc)})
                logger.error(f"Training failed for {symbol}: {exc}")

        return {
            "symbols": symbols,
            "completed": completed,
            "failed": failed,
            "message": self._build_status_message("Model training", symbols, failed),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def monitor_model_health(
        self,
        symbol: str,
        interval: str = "1d",
        auto_fine_tune: bool = True,
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        predictor = self._model_factory()
        settings = predictor.get_interval_settings(interval)
        if not predictor.has_model(symbol, interval):
            return {
                "symbol": symbol,
                "interval": interval,
                "action": "skipped",
                "degradation_streak": 0,
                "message": "No trained model is available yet for monitoring.",
            }

        self.db_manager.sync_prediction_actuals([symbol], timeframe=interval)
        history = self.db_manager.get_prediction_history(
            symbols=[symbol],
            timeframe=interval,
            limit_rows=int(self.config["MODEL_CONFIG"].get("monitoring", {}).get("lookback_evaluations", 12)),
            ascending=False,
            only_evaluated=True,
        )
        if history.empty:
            return {
                "symbol": symbol,
                "interval": interval,
                "action": "skipped",
                "degradation_streak": 0,
                "message": "No evaluated predictions are available for monitoring.",
            }

        history = history.sort_values("prediction_timestamp", ascending=False).reset_index(drop=True)
        history["absolute_percentage_error"] = (
            (history["predicted_price"] - history["actual_price"]).abs()
            / history["actual_price"].abs().replace(0, pd.NA)
        ) * 100.0
        history["accuracy_pct"] = 100.0 - history["absolute_percentage_error"]

        metadata = predictor.load_metadata(symbol, interval)
        baseline_accuracy = metadata.get("monitoring_baseline_accuracy_pct") or metadata.get("test_accuracy_pct")
        accuracy_floor = float(settings["accuracy_floor_pct"])
        allowed_drop = float(settings["allowed_accuracy_drop_pct"])
        threshold = accuracy_floor if baseline_accuracy is None else min(float(baseline_accuracy) - allowed_drop, accuracy_floor)

        degradation_streak = 0
        for _, row in history.iterrows():
            accuracy_pct = row.get("accuracy_pct")
            if pd.isna(accuracy_pct) or float(accuracy_pct) >= threshold:
                break
            degradation_streak += 1

        latest_row = history.iloc[0]
        latest_accuracy = None if pd.isna(latest_row["accuracy_pct"]) else float(latest_row["accuracy_pct"])
        latest_mape = None if pd.isna(latest_row["absolute_percentage_error"]) else float(latest_row["absolute_percentage_error"])
        action = "observed"
        note = (
            f"Threshold {threshold:.2f}%, latest accuracy {latest_accuracy:.2f}%."
            if latest_accuracy is not None
            else f"Threshold {threshold:.2f}%."
        )

        cooldown_predictions = int(self.config["MODEL_CONFIG"].get("monitoring", {}).get("cooldown_predictions", 2))
        last_monitor_streak = int(metadata.get("last_monitor_streak", 0))
        recent_monitor_events = self.db_manager.get_monitor_history(
            symbol=symbol,
            timeframe=interval,
            limit_rows=cooldown_predictions,
        )
        recently_fine_tuned = (
            not recent_monitor_events.empty
            and (recent_monitor_events["action"] == "fine_tuned").any()
        )
        can_fine_tune = (
            auto_fine_tune
            and degradation_streak >= int(settings["consecutive_drop_limit"])
            and degradation_streak > last_monitor_streak
            and not recently_fine_tuned
        )

        if can_fine_tune:
            market_data = self.db_manager.get_latest_data(
                symbol,
                limit_rows=self._history_limit_for_interval(interval),
                timeframe=interval,
                ascending=True,
            )
            training_frame = self._normalize_market_frame(market_data)
            tune_metrics = predictor.fine_tune(training_frame, symbol, interval=interval)
            action = "fine_tuned"
            note = (
                f"Fine-tuned after {degradation_streak} consecutive weak evaluations. "
                f"Updated accuracy target is {tune_metrics.get('test_accuracy_pct', 0.0):.2f}%."
            )
            metadata = predictor.load_metadata(symbol, interval)
            metadata["last_monitor_streak"] = 0
            metadata["last_monitor_accuracy_pct"] = latest_accuracy
            predictor.save_metadata(symbol, interval, metadata)
        else:
            if degradation_streak >= int(settings["consecutive_drop_limit"]):
                action = "cooldown"
            metadata["last_monitor_streak"] = degradation_streak
            metadata["last_monitor_accuracy_pct"] = latest_accuracy
            predictor.save_metadata(symbol, interval, metadata)

        self.db_manager.insert_monitor_event(
            {
                "symbol": symbol,
                "timeframe": interval,
                "prediction_timestamp": latest_row["prediction_timestamp"],
                "observed_accuracy_pct": latest_accuracy,
                "observed_mape": latest_mape,
                "degradation_streak": degradation_streak,
                "action": action,
                "model_version": metadata.get("model_version"),
                "note": note,
            }
        )

        if action == "cooldown" and degradation_streak <= cooldown_predictions:
            action = "observed"

        return {
            "symbol": symbol,
            "interval": interval,
            "action": action,
            "degradation_streak": degradation_streak,
            "latest_accuracy_pct": latest_accuracy,
            "threshold_accuracy_pct": round(threshold, 6),
            "message": note,
        }

    def generate_prediction(
        self,
        symbol: str,
        interval: str = "1h",
        refresh_period: str = "5d",
        force_refresh: bool = False,
        auto_train: bool = True,
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        effective_refresh_period = (
            self._default_refresh_period(interval) if refresh_period == "5d" and interval.endswith("d") else refresh_period
        )
        recent_limit = self._recent_limit_for_interval(interval)

        if force_refresh:
            self.collect_market_data([symbol], period=effective_refresh_period, interval=interval)

        market_data = self.db_manager.get_latest_data(
            symbol,
            limit_rows=recent_limit,
            timeframe=interval,
            ascending=True,
        )
        required_rows = self.ml_predictor.get_sequence_length(interval) + 5
        if market_data.empty or len(market_data) < required_rows:
            self.collect_market_data([symbol], period=effective_refresh_period, interval=interval)
            market_data = self.db_manager.get_latest_data(
                symbol,
                limit_rows=recent_limit,
                timeframe=interval,
                ascending=True,
            )

        if market_data.empty:
            raise ValueError(f"No market data is available for {symbol}.")

        predictor = self._model_factory()
        if not predictor.has_model(symbol, interval):
            if not auto_train:
                raise FileNotFoundError(f"No trained model is available for {symbol}.")
            self.train_model(
                symbol,
                history_period=self._default_training_period(interval),
                interval=interval,
                force_refresh=False,
            )

        monitoring = self.monitor_model_health(symbol, interval=interval, auto_fine_tune=auto_train)

        predictor.load_model(symbol, interval)
        metadata = predictor.load_metadata(symbol, interval)
        recent_frame = self._normalize_market_frame(market_data)

        current_price = float(recent_frame["close"].iloc[-1])
        raw_predicted_price = float(predictor.predict(recent_frame, interval=interval))
        calibrated_prediction = self._calibrate_prediction(
            recent_frame=recent_frame,
            current_price=current_price,
            raw_predicted_price=raw_predicted_price,
            interval=interval,
        )
        predicted_price = calibrated_prediction["predicted_price"]
        predicted_change_pct = calibrated_prediction["predicted_change_pct"]
        prediction_timestamp = self._next_prediction_timestamp(recent_frame["timestamp"].iloc[-1], interval)
        confidence_score = self._calculate_confidence(metadata, current_price, predicted_price)
        direction = "up" if predicted_price > current_price else "down" if predicted_price < current_price else "flat"
        model_version = metadata.get("model_version", "lstm")

        record = {
            "symbol": symbol,
            "timeframe": interval,
            "prediction_timestamp": prediction_timestamp,
            "predicted_price": predicted_price,
            "confidence_score": confidence_score,
            "model_version": model_version,
        }
        self.db_manager.upsert_prediction_results([record])
        self.db_manager.sync_prediction_actuals([symbol], timeframe=interval)

        response = {
            "symbol": symbol,
            "interval": interval,
            "prediction_timestamp": prediction_timestamp.isoformat(),
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": round(predicted_change_pct, 4),
            "raw_predicted_price": calibrated_prediction["raw_predicted_price"],
            "raw_predicted_change_pct": calibrated_prediction["raw_predicted_change_pct"],
            "volatility_cap_pct": calibrated_prediction["volatility_cap_pct"],
            "direction": direction,
            "confidence_score": confidence_score,
            "model_version": model_version,
            "monitoring": monitoring,
        }
        response["message"] = self._format_prediction_message(response)
        logger.info(response["message"])
        return response

    def generate_predictions(
        self,
        symbols: Optional[Iterable[str]] = None,
        interval: str = "1h",
        refresh_period: str = "5d",
        force_refresh: bool = False,
        auto_train: bool = True,
    ) -> Dict[str, Any]:
        symbols = self._resolve_symbols(symbols)
        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for symbol in symbols:
            try:
                completed.append(
                    self.generate_prediction(
                        symbol,
                        interval=interval,
                        refresh_period=refresh_period,
                        force_refresh=force_refresh,
                        auto_train=auto_train,
                    )
                )
            except Exception as exc:
                failed.append({"symbol": symbol, "error": str(exc)})
                logger.error(f"Prediction failed for {symbol}: {exc}")

        return {
            "symbols": symbols,
            "completed": completed,
            "failed": failed,
            "message": self._build_status_message("Prediction generation", symbols, failed),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_latest_predictions(self, symbol: str, limit_rows: int = 24, interval: Optional[str] = None) -> Dict[str, Any]:
        symbol = symbol.upper()
        predictions = self.db_manager.get_latest_predictions(
            symbol,
            limit_rows=limit_rows,
            timeframe=interval,
            ascending=False,
        )
        if predictions.empty:
            return {
                "symbol": symbol,
                "interval": interval,
                "predictions": [],
                "message": f"No prediction history is available for {symbol}.",
            }

        rows = []
        for record in predictions.to_dict(orient="records"):
            cleaned = {key: self._clean_json_value(value) for key, value in record.items()}
            cleaned["prediction_timestamp"] = pd.to_datetime(cleaned["prediction_timestamp"], utc=True).isoformat()
            if cleaned.get("created_at") is not None:
                cleaned["created_at"] = pd.to_datetime(cleaned["created_at"], utc=True).isoformat()
            rows.append(cleaned)
        return {
            "symbol": symbol,
            "interval": interval,
            "predictions": rows,
            "message": f"Retrieved {len(rows)} prediction rows for {symbol}.",
        }

    def get_market_history(
        self,
        symbols: Optional[Iterable[str]] = None,
        universe: str = "configured",
        start: Optional[str] = None,
        end: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit_rows: int = 1000,
        ascending: bool = False,
    ) -> Dict[str, Any]:
        normalized_universe = (universe or "").strip().lower()
        resolved_symbols = None
        if symbols or normalized_universe not in {"", "database", "stored"}:
            resolved_symbols = self._resolve_market_universe(universe=universe, symbols=symbols)
        market_history = self.db_manager.get_market_history(
            symbols=resolved_symbols,
            start=start,
            end=end,
            exchange=exchange,
            timeframe=timeframe,
            limit_rows=limit_rows,
            ascending=ascending,
        )

        rows = [self._serialize_market_row(row) for row in market_history.to_dict(orient="records")]
        available_symbols = (
            sorted({str(symbol).upper() for symbol in market_history["symbol"].tolist()})
            if not market_history.empty
            else (resolved_symbols or [])
        )
        return {
            "symbols": available_symbols,
            "universe": None if symbols else (universe or None),
            "row_count": len(rows),
            "rows": rows,
            "message": f"Retrieved {len(rows)} market history rows from SQL storage.",
        }

    def evaluate_predictions(
        self,
        symbols: Optional[Iterable[str]] = None,
        universe: str = "configured",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: Optional[str] = None,
        limit_rows: int = 1000,
        sync_actuals: bool = True,
    ) -> Dict[str, Any]:
        normalized_universe = (universe or "").strip().lower()
        resolved_symbols = None
        if symbols or normalized_universe not in {"", "database", "stored"}:
            resolved_symbols = self._resolve_market_universe(universe=universe, symbols=symbols)
        if sync_actuals:
            self.db_manager.sync_prediction_actuals(resolved_symbols, timeframe=interval)

        prediction_history = self.db_manager.get_prediction_history(
            symbols=resolved_symbols,
            start=start,
            end=end,
            timeframe=interval,
            limit_rows=limit_rows,
            ascending=False,
            only_evaluated=False,
        )

        if prediction_history.empty:
            return {
                "symbols": resolved_symbols or [],
                "universe": None if symbols else (universe or None),
                "interval": interval,
                "prediction_count": 0,
                "evaluated_count": 0,
                "pending_actual_count": 0,
                "metrics": self._summarize_prediction_metrics(pd.DataFrame()),
                "by_symbol": [],
                "rows": [],
                "message": "No prediction history is available for evaluation.",
            }

        evaluated = prediction_history.dropna(subset=["actual_price"]).copy()
        if not evaluated.empty:
            evaluated["error"] = evaluated["predicted_price"] - evaluated["actual_price"]
            evaluated["absolute_error"] = evaluated["error"].abs()
            actual_denominator = evaluated["actual_price"].abs().replace(0, pd.NA)
            evaluated["absolute_percentage_error"] = (
                evaluated["absolute_error"] / actual_denominator
            ) * 100.0

        metrics = self._summarize_prediction_metrics(evaluated)
        rows = []
        for row in prediction_history.to_dict(orient="records"):
            serialized = self._serialize_prediction_row(row)
            if serialized.get("actual_price") is not None:
                error = float(serialized["predicted_price"]) - float(serialized["actual_price"])
                serialized["error"] = round(error, 6)
                serialized["absolute_error"] = round(abs(error), 6)
                if float(serialized["actual_price"]) != 0:
                    serialized["absolute_percentage_error"] = round(
                        abs(error) / abs(float(serialized["actual_price"])) * 100.0,
                        6,
                    )
                else:
                    serialized["absolute_percentage_error"] = None
            else:
                serialized["error"] = None
                serialized["absolute_error"] = None
                serialized["absolute_percentage_error"] = None
            rows.append(serialized)

        by_symbol = []
        for symbol, group in prediction_history.groupby("symbol"):
            group_evaluated = group.dropna(subset=["actual_price"]).copy()
            if not group_evaluated.empty:
                group_evaluated["error"] = group_evaluated["predicted_price"] - group_evaluated["actual_price"]
                group_evaluated["absolute_error"] = group_evaluated["error"].abs()
                group_denominator = group_evaluated["actual_price"].abs().replace(0, pd.NA)
                group_evaluated["absolute_percentage_error"] = (
                    group_evaluated["absolute_error"] / group_denominator
                ) * 100.0
            symbol_metrics = self._summarize_prediction_metrics(group_evaluated)
            by_symbol.append(
                {
                    "symbol": symbol,
                    "prediction_count": int(len(group)),
                    "evaluated_count": int(len(group_evaluated)),
                    "pending_actual_count": int(len(group) - len(group_evaluated)),
                    **symbol_metrics,
                }
            )

        pending_count = int(len(prediction_history) - len(evaluated))
        return {
            "symbols": sorted({str(symbol).upper() for symbol in prediction_history["symbol"].tolist()}),
            "universe": None if symbols else (universe or None),
            "interval": interval,
            "prediction_count": int(len(prediction_history)),
            "evaluated_count": int(len(evaluated)),
            "pending_actual_count": pending_count,
            "metrics": metrics,
            "by_symbol": by_symbol,
            "rows": rows,
            "message": (
                f"Evaluated {len(evaluated)} predictions with realized prices. "
                f"{pending_count} predictions are still waiting for future market bars."
            ),
        }

    def get_data_coverage(
        self,
        symbols: Optional[Iterable[str]] = None,
        universe: str = "configured",
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_universe = (universe or "").strip().lower()
        resolved_symbols = None
        if symbols or normalized_universe not in {"", "database", "stored"}:
            resolved_symbols = self._resolve_market_universe(universe=universe, symbols=symbols)

        market_coverage = self.db_manager.get_market_coverage(
            symbols=resolved_symbols,
            exchange=exchange,
            timeframe=timeframe,
        )
        prediction_coverage = self.db_manager.get_prediction_coverage(symbols=resolved_symbols, timeframe=timeframe)

        target_symbols = resolved_symbols or sorted(
            set(market_coverage["symbol"].tolist()) | set(prediction_coverage["symbol"].tolist())
        )
        stored_market_symbols = sorted(set(market_coverage["symbol"].tolist())) if not market_coverage.empty else []
        stored_prediction_symbols = (
            sorted(set(prediction_coverage["symbol"].tolist())) if not prediction_coverage.empty else []
        )

        timeframe_summary = []
        if not market_coverage.empty:
            for current_timeframe, group in market_coverage.groupby("timeframe"):
                timeframe_summary.append(
                    {
                        "timeframe": current_timeframe,
                        "row_count": int(group["row_count"].sum()),
                        "symbol_count": int(group["symbol"].nunique()),
                        "first_timestamp": group["first_timestamp"].min().isoformat(),
                        "last_timestamp": group["last_timestamp"].max().isoformat(),
                    }
                )

        return {
            "symbols": target_symbols,
            "universe": None if symbols else (universe or None),
            "requested_symbol_count": len(target_symbols),
            "market_symbol_count": len(stored_market_symbols),
            "prediction_symbol_count": len(stored_prediction_symbols),
            "symbols_without_market_data": [symbol for symbol in target_symbols if symbol not in stored_market_symbols],
            "symbols_without_predictions": [
                symbol for symbol in target_symbols if symbol not in stored_prediction_symbols
            ],
            "timeframe_summary": timeframe_summary,
            "market_coverage": [
                {
                    "symbol": row["symbol"],
                    "exchange": row["exchange"],
                    "timeframe": row["timeframe"],
                    "row_count": int(row["row_count"]),
                    "first_timestamp": pd.to_datetime(row["first_timestamp"], utc=True).isoformat(),
                    "last_timestamp": pd.to_datetime(row["last_timestamp"], utc=True).isoformat(),
                }
                for row in market_coverage.to_dict(orient="records")
            ],
            "prediction_coverage": [
                {
                    "symbol": row["symbol"],
                    "timeframe": row.get("timeframe"),
                    "prediction_count": int(row["prediction_count"]),
                    "evaluated_count": int(row["evaluated_count"]),
                    "pending_actual_count": int(row["pending_actual_count"]),
                    "coverage_pct": round(float(row["coverage_pct"]), 4),
                    "first_prediction_timestamp": pd.to_datetime(
                        row["first_prediction_timestamp"],
                        utc=True,
                    ).isoformat(),
                    "last_prediction_timestamp": pd.to_datetime(
                        row["last_prediction_timestamp"],
                        utc=True,
                    ).isoformat(),
                }
                for row in prediction_coverage.to_dict(orient="records")
            ],
            "message": "Retrieved market and prediction coverage from SQL storage.",
        }

    def build_market_report(
        self,
        symbols: Optional[Iterable[str]] = None,
        interval: str = "1h",
        refresh_period: str = "5d",
        force_refresh: bool = False,
        auto_predict: bool = True,
        auto_train: bool = True,
    ) -> Dict[str, Any]:
        symbols = self._resolve_symbols(symbols)
        items: List[Dict[str, Any]] = []

        for symbol in symbols:
            market_data = self.db_manager.get_latest_data(
                symbol,
                limit_rows=48,
                timeframe=interval,
                ascending=True,
            )
            prediction_payload: Optional[Dict[str, Any]] = None
            prediction_error: Optional[str] = None

            if market_data.empty and auto_predict:
                try:
                    prediction_payload = self.generate_prediction(
                        symbol,
                        interval=interval,
                        refresh_period=refresh_period,
                        force_refresh=force_refresh,
                        auto_train=auto_train,
                    )
                except Exception as exc:
                    prediction_error = str(exc)
                    logger.error(f"Prediction refresh failed for {symbol} during report generation: {exc}")
                market_data = self.db_manager.get_latest_data(
                    symbol,
                    limit_rows=48,
                    timeframe=interval,
                    ascending=True,
                )

            if market_data.empty:
                continue

            recent_frame = self._normalize_market_frame(market_data)
            latest_predictions = self.db_manager.get_latest_predictions(
                symbol,
                limit_rows=1,
                timeframe=interval,
                ascending=False,
            )
            latest_market_timestamp = recent_frame["timestamp"].iloc[-1]

            if auto_predict:
                needs_prediction = latest_predictions.empty
                if not needs_prediction:
                    needs_prediction = not self._prediction_is_fresh(
                        latest_predictions.iloc[0]["prediction_timestamp"],
                        latest_market_timestamp,
                    )

                if needs_prediction:
                    try:
                        prediction_payload = self.generate_prediction(
                            symbol,
                            interval=interval,
                            refresh_period=refresh_period,
                            force_refresh=force_refresh,
                            auto_train=auto_train,
                        )
                        latest_predictions = self.db_manager.get_latest_predictions(
                            symbol,
                            limit_rows=1,
                            timeframe=interval,
                            ascending=False,
                        )
                        prediction_error = None
                    except Exception as exc:
                        prediction_error = str(exc)
                        logger.error(f"Prediction refresh failed for {symbol} during report generation: {exc}")

            current_price = float(recent_frame["close"].iloc[-1])
            price_change_24h = 0.0
            if len(recent_frame) > 1 and recent_frame["close"].iloc[0] != 0:
                price_change_24h = ((current_price - float(recent_frame["close"].iloc[0])) / float(recent_frame["close"].iloc[0])) * 100

            item = {
                "symbol": symbol,
                "current_price": current_price,
                "price_change_24h": round(price_change_24h, 4),
                "avg_volume_24h": float(recent_frame["volume"].tail(24).mean()),
            }

            if prediction_payload is not None:
                item.update(self._serialize_prediction_for_report(prediction_payload, current_price))
                item["message"] = (
                    f"{symbol} closed the latest hour at ${current_price:.2f}. "
                    f"The latest report estimate is ${item['predicted_price']:.2f} "
                    f"with {item['confidence_score']:.1f}% confidence."
                )
            elif not latest_predictions.empty:
                latest_prediction = latest_predictions.iloc[0]
                item.update(
                    self._serialize_prediction_for_report(
                        {
                            "prediction_timestamp": latest_prediction["prediction_timestamp"],
                            "predicted_price": latest_prediction["predicted_price"],
                            "confidence_score": latest_prediction["confidence_score"],
                        },
                        current_price,
                    )
                )
                item["message"] = (
                    f"{symbol} closed the latest hour at ${current_price:.2f}. "
                    f"The most recent model estimate is ${item['predicted_price']:.2f} "
                    f"with {item['confidence_score']:.1f}% confidence."
                )
            else:
                item["predicted_price"] = None
                item["confidence_score"] = None
                item["predicted_change_pct"] = 0.0
                if prediction_error:
                    item["message"] = (
                        f"{symbol} closed the latest hour at ${current_price:.2f}. "
                        f"A fresh model estimate is not available yet: {prediction_error}"
                    )
                else:
                    item["message"] = (
                        f"{symbol} closed the latest hour at ${current_price:.2f}. "
                        "No model estimate is stored yet."
                    )
            items.append(item)

        generated_at = datetime.utcnow().isoformat()
        return {
            "generated_at": generated_at,
            "symbols": symbols,
            "items": items,
            "message": self._format_report_message(items, generated_at),
        }

    async def run_full_pipeline(
        self,
        symbols: Optional[Iterable[str]] = None,
        history_period: str = "6mo",
        interval: str = "1h",
    ) -> Dict[str, Any]:
        symbols = self._resolve_symbols(symbols)
        logger.info(f"Starting full pipeline for {', '.join(symbols)}.")

        collection = self.collect_market_data(symbols, period=history_period, interval=interval)
        training = self.train_models(symbols, history_period=history_period, interval=interval, force_refresh=False)
        predictions = self.generate_predictions(
            symbols,
            interval=interval,
            refresh_period="5d",
            force_refresh=False,
            auto_train=False,
        )
        report = self.build_market_report(symbols, interval=interval, refresh_period="5d", auto_predict=False)

        return {
            "symbols": symbols,
            "data_collection": collection,
            "model_training": training,
            "predictions": predictions,
            "report": report,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "The full market pipeline completed.",
        }

    def run_hourly_update(
        self,
        symbols: Optional[Iterable[str]] = None,
        refresh_period: str = "5d",
        interval: str = "1h",
    ) -> Dict[str, Any]:
        symbols = self._resolve_symbols(symbols)
        logger.info(f"Running hourly update for {', '.join(symbols)}.")

        collection = self.collect_market_data(symbols, period=refresh_period, interval=interval)
        predictions = self.generate_predictions(
            symbols,
            interval=interval,
            refresh_period=refresh_period,
            force_refresh=False,
            auto_train=True,
        )
        report = self.build_market_report(symbols, interval=interval, refresh_period=refresh_period, auto_predict=False)

        return {
            "symbols": symbols,
            "data_collection": collection,
            "predictions": predictions,
            "report": report,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "The hourly market update completed.",
        }

    def run_daily_market_update(self) -> Dict[str, Any]:
        universe = os.getenv("DAILY_MARKET_UPDATE_UNIVERSE", "sp500").strip() or "sp500"
        period = os.getenv("DAILY_MARKET_UPDATE_PERIOD", "7d").strip() or "7d"
        batch_size_raw = os.getenv("DAILY_MARKET_UPDATE_BATCH_SIZE", "25").strip() or "25"
        try:
            batch_size = max(int(batch_size_raw), 1)
        except ValueError:
            logger.warning(
                "Invalid DAILY_MARKET_UPDATE_BATCH_SIZE='{}'. Falling back to 25.",
                batch_size_raw,
            )
            batch_size = 25

        logger.info(
            "Running scheduled daily market update for universe '{}' using period {} and batch size {}.",
            universe,
            period,
            batch_size,
        )
        result = self.collect_market_universe(
            universe=universe,
            period=period,
            interval="1d",
            batch_size=batch_size,
        )
        logger.info(
            "Daily market update stored {} rows across {} symbols.",
            result.get("rows_collected", 0),
            result.get("stored_symbol_count", 0),
        )
        return result

    def schedule_operations(self):
        """Schedule regular operations."""
        import schedule
        import time

        schedule.every().hour.do(self.run_hourly_update)
        schedule.every().day.at("02:00").do(lambda: asyncio.run(self.run_full_pipeline()))

        if os.getenv("ENABLE_DAILY_MARKET_UPDATE", "false").strip().lower() in {"1", "true", "yes", "on"}:
            update_time = os.getenv("DAILY_MARKET_UPDATE_TIME", "18:30").strip() or "18:30"
            schedule.every().day.at(update_time).do(self.run_daily_market_update)
            logger.info(
                "Scheduled daily universe refresh at {} for universe '{}'.",
                update_time,
                os.getenv("DAILY_MARKET_UPDATE_UNIVERSE", "sp500").strip() or "sp500",
            )
        else:
            logger.info(
                "Scheduled daily universe refresh is disabled. "
                "Enable it with ENABLE_DAILY_MARKET_UPDATE=true if you want ongoing 1d updates."
            )

        logger.info("Scheduler started. Waiting for the next market update window.")
        while True:
            schedule.run_pending()
            time.sleep(60)
