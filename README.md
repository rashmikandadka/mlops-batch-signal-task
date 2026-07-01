# T0 - MLOps Batch Job (signal generation from OHLCV data)

A minimal, reproducible MLOps-style batch job that computes a rolling mean on
`close` price and generates a binary trading signal, with structured metrics
and logging, packaged for local and Docker execution.

## What it does

1. Loads and validates a YAML config (`seed`, `window`, `version`)
2. Loads and validates the input OHLCV CSV (checks file exists, is non-empty,
   is valid CSV, and contains a `close` column)
3. Computes a rolling mean on `close` using the configured `window`
   (`min_periods=1`, so the first `window - 1` rows use a partial-window mean
   instead of `NaN` — this keeps the run fully deterministic with no gaps)
4. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
5. Writes structured metrics to `metrics.json` and detailed logs to `run.log`
6. Prints the final metrics JSON to stdout and exits `0` on success / non-zero
   on failure

## Project structure

```
.
├── run.py
├── config.yaml
├── data.csv
├── requirements.txt
├── Dockerfile
├── README.md
├── metrics.json      # sample output from a successful run
└── run.log            # sample log from a successful run
```

## Local run instructions

```bash
pip install -r requirements.txt

python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

No paths are hard-coded — all four arguments are required and can point
anywhere on disk.

## Docker build/run instructions

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container includes `data.csv` and `config.yaml`, runs the job with the
exact CLI shown above, prints the final `metrics.json` contents to stdout,
and exits with the job's real exit code (0 = success, non-zero = failure).

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 26.9,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Missing input file: data.csv"
}
```

## Design notes

- **Determinism**: `numpy.random.seed()` is set from config immediately after
  config validation. All data transforms are deterministic pandas/numpy
  operations, so `rows_processed` and `signal_rate` are bit-identical across
  runs given the same input and config. Only `latency_ms` naturally varies
  run to run since it measures wall-clock execution time.
- **First `window - 1` rows**: handled via `rolling(window, min_periods=1)`
  rather than allowing `NaN`s, so every row gets a valid signal and no rows
  are silently dropped from `rows_processed` or `signal_rate`.
- **Validation**: config and dataset are validated before any processing;
  clear, specific `ValueError`s are raised and caught centrally in `main()`,
  which always writes a `metrics.json` (success or error shape) before
  exiting.
- **Logging**: every stage (job start, config validation, rows loaded,
  rolling mean, signal generation, metrics summary, job end/status, and any
  exception) is logged with timestamps to both `run.log` and stdout.
