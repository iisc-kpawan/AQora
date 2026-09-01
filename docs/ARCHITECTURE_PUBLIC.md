# Public architecture overview

AQora separates language-model reasoning from numerical analysis. A request is
converted into a typed query specification, validated against the capability
contract, executed by a deterministic analytical tool, and returned as a
structured result with a natural-language explanation. Tables and figures are
generated only when supported by the result shape.

The implementation exposes tools for metadata discovery, aggregation,
statistics, comparison, ranking, temporal analysis, data-quality assessment,
AQI analysis, forecasting, imputation assessment, uncertainty analysis, and
visualization. Forecasting tools invoke explicit models and report model and
input-quality information with their outputs.

The research data service is connected through deployment configuration and is
not included in this artifact. This public repository does not document or
encode production hosts, credentials, physical storage locations, or
operational data inventories.

