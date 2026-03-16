import asyncio
import os
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
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
        self.default_symbols = self._resolve_symbols(self.DEFAULT_SYMBOLS)
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
    ) -> Dict[str, Any]:
        symbols = self._resolve_symbols(symbols)
        logger.info(f"Collecting {interval} market data for {', '.join(symbols)} using period {period}.")

        market_data = self.data_collector.fetch_yfinance_data(symbols, period=period, interval=interval)
        if market_data.empty:
            return {
                "status": "no_data",
                "symbols": symbols,
                "rows_collected": 0,
                "message": "No market data was returned by the provider.",
                "timestamp": datetime.utcnow().isoformat(),
            }

        market_data = market_data.copy()
        market_data["symbol"] = market_data["symbol"].astype(str).str.upper()
        market_data["exchange"] = market_data["symbol"].map(self.exchange_lookup).fillna("US")
        self.db_manager.insert_market_data(market_data)
        actuals_updated = self.db_manager.sync_prediction_actuals(symbols)
        rows_by_symbol = {
            symbol: int(count) for symbol, count in market_data.groupby("symbol").size().to_dict().items()
        }

        message = (
            f"Collected {len(market_data)} rows across {len(rows_by_symbol)} symbols. "
            f"Updated {actuals_updated} prediction records with realized prices."
        )
        return {
            "status": "ok",
            "symbols": symbols,
            "period": period,
            "interval": interval,
            "rows_collected": int(len(market_data)),
            "rows_by_symbol": rows_by_symbol,
            "actuals_updated": int(actuals_updated),
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def collect_market_universe(
        self,
        universe: str = "all",
        period: str = "1mo",
        interval: str = "1h",
        symbols: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        resolved_symbols = self._resolve_market_universe(universe=universe, symbols=symbols)
        result = self.collect_market_data(resolved_symbols, period=period, interval=interval)
        result["universe"] = (universe or "custom").strip().lower()
        result["requested_symbol_count"] = len(resolved_symbols)
        result["stored_symbol_count"] = len(result.get("rows_by_symbol", {}))
        result["message"] = (
            f"Collected {result['rows_collected']} rows for {result['stored_symbol_count']} of "
            f"{len(resolved_symbols)} configured symbols in the {result['universe']} universe. "
            f"Updated {result['actuals_updated']} prediction records with realized prices."
        )
        return result

    def train_model(
        self,
        symbol: str,
        history_period: str = "6mo",
        interval: str = "1h",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        min_rows = self.ml_predictor.sequence_length + 24
        history_limit = max(int(self.config["MARKET_CONFIG"].get("lookback_days", 365)) * 24, min_rows * 2)
        market_data = self.db_manager.get_latest_data(symbol, limit_rows=history_limit, ascending=True)

        if force_refresh or market_data.empty or len(market_data) < min_rows:
            self.collect_market_data([symbol], period=history_period, interval=interval)
            market_data = self.db_manager.get_latest_data(symbol, limit_rows=history_limit, ascending=True)

        if market_data.empty or len(market_data) < min_rows:
            raise ValueError(f"Not enough market history is available to train {symbol}.")

        predictor = self._model_factory()
        training_frame = self._normalize_market_frame(market_data)
        metrics = predictor.train(training_frame, symbol)
        message = (
            f"Trained the {symbol} model on {metrics['training_rows']} hourly rows. "
            f"Test RMSE is {metrics['test_rmse']:.4f}."
        )
        logger.info(message)
        return {
            "symbol": symbol,
            "training_rows": metrics["training_rows"],
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_mae": metrics["train_mae"],
            "test_mae": metrics["test_mae"],
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

    def generate_prediction(
        self,
        symbol: str,
        interval: str = "1h",
        refresh_period: str = "5d",
        force_refresh: bool = False,
        auto_train: bool = True,
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        recent_limit = max(self.ml_predictor.sequence_length + 50, 200)

        if force_refresh:
            self.collect_market_data([symbol], period=refresh_period, interval=interval)

        market_data = self.db_manager.get_latest_data(symbol, limit_rows=recent_limit, ascending=True)
        if market_data.empty or len(market_data) < self.ml_predictor.sequence_length + 5:
            self.collect_market_data([symbol], period=refresh_period, interval=interval)
            market_data = self.db_manager.get_latest_data(symbol, limit_rows=recent_limit, ascending=True)

        if market_data.empty:
            raise ValueError(f"No market data is available for {symbol}.")

        model_path = Path(f"models/saved/{symbol}/lstm_model.h5")
        if not model_path.exists():
            if not auto_train:
                raise FileNotFoundError(f"No trained model is available for {symbol}.")
            self.train_model(symbol, history_period="6mo", interval=interval, force_refresh=False)

        predictor = self._model_factory()
        predictor.load_model(symbol)
        metadata = predictor.load_metadata(symbol)
        recent_frame = self._normalize_market_frame(market_data)

        current_price = float(recent_frame["close"].iloc[-1])
        predicted_price = float(predictor.predict(recent_frame))
        predicted_change_pct = ((predicted_price - current_price) / current_price) * 100 if current_price else 0.0
        prediction_timestamp = self._next_prediction_timestamp(recent_frame["timestamp"].iloc[-1], interval)
        confidence_score = self._calculate_confidence(metadata, current_price, predicted_price)
        direction = "up" if predicted_price > current_price else "down" if predicted_price < current_price else "flat"
        model_version = metadata.get("model_version", "lstm")

        record = {
            "symbol": symbol,
            "prediction_timestamp": prediction_timestamp,
            "predicted_price": predicted_price,
            "confidence_score": confidence_score,
            "model_version": model_version,
        }
        self.db_manager.upsert_prediction_results([record])
        self.db_manager.sync_prediction_actuals([symbol])

        response = {
            "symbol": symbol,
            "interval": interval,
            "prediction_timestamp": prediction_timestamp.isoformat(),
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": round(predicted_change_pct, 4),
            "direction": direction,
            "confidence_score": confidence_score,
            "model_version": model_version,
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

    def get_latest_predictions(self, symbol: str, limit_rows: int = 24) -> Dict[str, Any]:
        symbol = symbol.upper()
        predictions = self.db_manager.get_latest_predictions(symbol, limit_rows=limit_rows, ascending=False)
        if predictions.empty:
            return {
                "symbol": symbol,
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
        limit_rows: int = 1000,
        sync_actuals: bool = True,
    ) -> Dict[str, Any]:
        normalized_universe = (universe or "").strip().lower()
        resolved_symbols = None
        if symbols or normalized_universe not in {"", "database", "stored"}:
            resolved_symbols = self._resolve_market_universe(universe=universe, symbols=symbols)
        if sync_actuals:
            self.db_manager.sync_prediction_actuals(resolved_symbols)

        prediction_history = self.db_manager.get_prediction_history(
            symbols=resolved_symbols,
            start=start,
            end=end,
            limit_rows=limit_rows,
            ascending=False,
            only_evaluated=False,
        )

        if prediction_history.empty:
            return {
                "symbols": resolved_symbols or [],
                "universe": None if symbols else (universe or None),
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
            market_data = self.db_manager.get_latest_data(symbol, limit_rows=48, ascending=True)
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
                market_data = self.db_manager.get_latest_data(symbol, limit_rows=48, ascending=True)

            if market_data.empty:
                continue

            recent_frame = self._normalize_market_frame(market_data)
            latest_predictions = self.db_manager.get_latest_predictions(symbol, limit_rows=1, ascending=False)
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

    def schedule_operations(self):
        """Schedule regular operations."""
        import schedule
        import time

        schedule.every().hour.do(self.run_hourly_update)
        schedule.every().day.at("02:00").do(lambda: asyncio.run(self.run_full_pipeline()))

        logger.info("Scheduler started. Waiting for the next market update window.")
        while True:
            schedule.run_pending()
            time.sleep(60)
