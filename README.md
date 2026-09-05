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

## Live demo

A running instance is available for reviewers at
[https://angler-unstamped-antiquity.ngrok-free.dev](https://angler-unstamped-antiquity.ngrok-free.dev).
It serves the same code as this repository, over the full global dataset
described below.

### Data coverage at evaluation time

Ingestion and development continue on an ongoing basis, so as of this
evaluation different sources reflect data through different dates. India
(CPCB) and 48 other datasets are current through 31 May 2026. Six datasets
continued ingesting further at evaluation time and are current through 28
August 2026: China (CNEMC), the United States (EPA), Mexico (SINAICA), the
United Kingdom (AURN), Germany (EEA_DE), and France (EEA_FR). A question that
names a date beyond a dataset's own evaluation-time coverage receives an
explicit response stating the loaded date range, rather than an
extrapolated or approximate answer.

### Data-quality notes

Confirmed directly against the raw ingested data:
- The United States (EPA) O3 series has no valid values for June, July, or
  August 2026 (every row is flagged `missing_sparse_series`); other EPA
  pollutants and months are unaffected.
- India (CPCB) has no directly measured (`value_status=observed`) values for
  January through May 2026 across all six monitored pollutants -- every row
  in that period is either `imputed_legacy_unrestricted` or
  `invalid_negative_imputation`. December 2025 and earlier has normal
  observed coverage. A directly-measured-only question in that window (for
  example, an observed-only versus imputed comparison) states plainly that
  no directly measured values are available for that period, rather than
  presenting a filled value as if it were measured; questions that accept
  policy-approved filled values are unaffected.

Both notes reflect the system's data-quality safeguards working precisely as
intended: an explicit, accurate account of what is actually loaded, rather
than an approximate or fabricated answer. All other datasets and pollutants
are current.

## Design

```text
question -> planner -> validated query specification -> analytical tool
         -> deterministic result -> explanation + optional visualization
```

The language model plans and explains; it does not calculate reported air-
quality values. The implementation includes querying, descriptive and temporal
analytics, comparisons and rankings, quality-aware analysis, AQI standards,
visualization, forecasting, imputation assessment, uncertainty analysis, and
multi-criteria analytical workflows. Compound operators include cross-pollutant
relative variability, matched-season contrasts, paired-month year-over-year
attribution, historical-percentile burden, and same-issue multi-horizon
forecast comparison.

Internal result payloads (for example an `analysis.operator` field such as
`forecast_readiness_portfolio_v1`) carry short, versioned code-level labels for
each analytical tool. These labels are development identifiers, not paper
terminology: they may be renamed, split, or re-versioned as implementation
continues, independent of how the same capability is described in the paper.
This document and the paper always use the descriptive names; only raw API
responses or logs expose the internal literals.

## Repository contents

- `src/aqa/`: application, API, conversational planner, analytics, and UI
- `benchmark/aqa_eval_300_questions.csv`: the 300 published questions, with no
  labels, answers, tool annotations, or internal metadata
- `benchmark/aqa_eval_10_representative_questions.csv`: the 10 representative
  questions (Table 2 in the paper) used for the Table 1 quantitative
  evaluation, labeled by difficulty level
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

`benchmark/aqa_eval_10_representative_questions.csv` holds the 10 representative
questions (two per difficulty level, L1-L5) used for the paper's Table 1
quantitative evaluation across multiple local and cloud-based LLMs; see Table 2
of the paper for the same questions with commentary. Run them the same way,
pointing `--input` at that file instead.

## Data availability

No observations, station metadata, database files, partitions, credentials, or
deployment paths are distributed in this repository. See
`docs/DATA_AVAILABILITY.md` for the public-data citation and artifact boundary.

The ingested observation corpus and the operational lakehouse currently reside
only on the authors' server and are not yet published. They will be uploaded
for public access after the paper's review period concludes.

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
