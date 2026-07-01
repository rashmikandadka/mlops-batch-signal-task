"""
run.py - Minimal MLOps-style batch job.

Loads OHLCV data, computes a rolling mean on 'close', generates a binary
signal (close > rolling_mean), and writes structured metrics + logs.

Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers on re-entry

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_config(config_path: str, logger: logging.Logger) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid config structure: expected a YAML mapping/dict")

    required_fields = ["seed", "window", "version"]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Config missing required field(s): {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config field 'seed' must be an integer")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("Config field 'window' must be a positive integer")
    if not isinstance(config["version"], str):
        raise ValueError("Config field 'version' must be a string")

    logger.info(
        "Config loaded + validated | seed=%s window=%s version=%s",
        config["seed"], config["window"], config["version"],
    )
    return config


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format in {input_path}: {e}")
    except FileNotFoundError:
        raise ValueError(f"Missing input file: {input_path}")

    if df.empty:
        raise ValueError(f"Input file has no rows: {input_path}")

    if "close" not in df.columns:
        raise ValueError("Required column 'close' not found in input data")

    if df["close"].isnull().all():
        raise ValueError("Column 'close' contains no valid data")

    logger.info("Rows loaded: %d", len(df))
    return df


def process(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    # min_periods=1 keeps the calculation deterministic and avoids NaNs
    # for the first (window - 1) rows; those rows use a partial window mean.
    logger.info("Computing rolling mean on 'close' (window=%d, min_periods=1)", window)
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=1).mean()

    logger.info("Generating binary signal (1 if close > rolling_mean else 0)")
    df["signal"] = (df["close"] > df["rolling_mean"]).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description="Minimal MLOps batch job")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    start_time = time.perf_counter()
    logger.info("Job start")

    version = "v1"  # fallback if config fails to load before we know the real version

    try:
        config = load_config(args.config, logger)
        version = config["version"]

        np.random.seed(config["seed"])
        logger.info("Random seed set to %d", config["seed"])

        df = load_dataset(args.input, logger)
        df = process(df, config["window"], logger)

        rows_processed = int(len(df))
        signal_rate = round(float(df["signal"].mean()), 4)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": latency_ms,
            "seed": config["seed"],
            "status": "success",
        }

        write_metrics(args.output, metrics)
        logger.info("Metrics summary: %s", json.dumps(metrics))
        logger.info("Job end | status=success")

        print(json.dumps(metrics))
        sys.exit(0)

    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        error_payload = {
            "version": version,
            "status": "error",
            "error_message": str(e),
        }
        write_metrics(args.output, error_payload)
        logger.exception("Job failed: %s", e)
        logger.info("Job end | status=error")

        print(json.dumps(error_payload))
        sys.exit(1)


if __name__ == "__main__":
    main()
