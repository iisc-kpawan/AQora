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

Known limitations, both confirmed against the raw ingested data rather than
assumed:
- The India (CPCB) source has been unreachable from the hosting server since
  2026-05-31, so its data is frozen at that date. Questions that name an India
  date after 2026-05-31 (as in a few of the paper's Table 2 questions) will
  return an explicit "outside loaded scope" response rather than a computed
  answer.
- The United States (EPA) O3 series has no valid values at all for June,
  July, or August 2026 (every row is flagged `missing_sparse_series`).
  Questions about US O3 in that window will likewise return an explicit
  limitation rather than a computed answer. Other EPA pollutants and other
  months are unaffected.

Both reflect the system's data-quality safeguards working as intended -- an
honest "not available" rather than a fabricated number -- not a computation
error. All other datasets and pollutants are current.

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
