# Evaluation protocol

The public benchmark contains 300 questions in a single `question` column. It
does not distribute answers, expected tools, difficulty labels, category labels,
or internal reviewer annotations.

The generic evaluation client records each question, run number, selected
provider/model, completion status, latency, token counts when reported by the
provider, and the returned response. Generated records are ignored by Git.

Paper reporting uses:

- **Task success rate:** fraction of runs judged to have selected and completed
  the required analytical workflow with a correct result.
- **Completion time:** end-to-end wall-clock time per run.
- **Token consumption:** prompt and completion tokens reported by the configured
  model provider, when available.

Each evaluated question should be repeated with the same model configuration
and run count stated in the paper. Correctness judgments and any private ground
truth remain separate from the questions-only public artifact. Numerical values
should be independently verified with deterministic analytical functions.

