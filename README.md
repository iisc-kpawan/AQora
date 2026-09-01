# AQora

AQora is a conversational system for reproducible global air-quality analytics.
It translates natural-language questions into validated analytical plans while
delegating numerical work to deterministic functions or explicitly selected
forecasting models.

This repository is the code artifact accompanying the paper *AQora: An Agentic
AI System for Global Air Quality Analytics*. It contains the application,
analytical tools, questions-only benchmark, portable tests, and a generic
evaluation client. Observation data and operational infrastructure are not
included.

## Design

```text
question -> planner -> validated query specification -> analytical tool
         -> deterministic result -> explanation + optional visualization
```

The language model plans and explains; it does not calculate reported air-
quality values. The implementation includes querying, descriptive and temporal
analytics, comparisons and rankings, quality-aware analysis, AQI standards,
visualization, forecasting, imputation assessment, uncertainty analysis, and
multi-criteria analytical workflows.

## Repository contents

- `src/aqa/`: application, API, conversational planner, analytics, and UI
- `benchmark/aqa_eval_300_questions.csv`: the 300 published questions, with no
  labels, answers, tool annotations, or internal metadata
- `scripts/run_questions_benchmark.py`: provider/model-neutral evaluation client
- `tests/`: data-independent unit and smoke tests
- `docs/`: public architecture, evaluation, and data-availability notes

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Run the portable tests:

```bash
.venv/bin/pytest -q
```

## Running the application

AQora expects a deployment-specific configuration supplied outside version
control. Set `AQA_CONFIG` to that file and start the API:

```bash
export AQA_CONFIG=/path/to/private/configuration.yaml
.venv/bin/aqa-serve
```

The listening address can be configured with `AQA_PORT`. LLM credentials, when
used, must be supplied through environment variables and must never be committed.
The deterministic planner remains available when a configured model service is
unavailable.

## Benchmark

The published benchmark intentionally contains only one column, `question`.
Run it against a configured AQora instance with:

```bash
.venv/bin/python scripts/run_questions_benchmark.py \
  --input benchmark/aqa_eval_300_questions.csv \
  --base-url http://127.0.0.1:8000 \
  --runs 3
```

Generated responses and timing records are local artifacts and are ignored by
Git. See `docs/EVALUATION.md` for metric definitions and reporting guidance.

## Data availability

No observations, station metadata, database files, partitions, credentials, or
deployment paths are distributed in this repository. See
`docs/DATA_AVAILABILITY.md` for the public-data citation and artifact boundary.

## Safety and reproducibility

- Numerical results are produced by explicit analytical code.
- Query specifications are validated before execution.
- Forecasts are labeled as model estimates rather than measurements.
- Quality, coverage, and value-lineage constraints are represented explicitly.
- Portable tests use generated in-memory values and do not require the research
  corpus.

## Citation

Citation metadata will be added after the anonymous review period. During
double-blind review, use the anonymous artifact URL supplied with the paper.
