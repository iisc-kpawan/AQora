from __future__ import annotations

import csv
import io
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .agent import ConversationalAgent
from .analytics import AnalyticsEngine
from .aqi import DEFAULT_STANDARD, STANDARDS, calculate_aqi
from .errors import AQAError, ValidationError
from .forecasting import forecast_series
from .queryspec import QuerySpec
from .lakehouse_catalog import LakehouseCatalog


class ChatContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    locations: list[str] | None = None


class ChatRequest(BaseModel):
    question: str
    provider: str | None = None
    model: str | None = None
    context: ChatContext | None = None


class ForecastRequest(BaseModel):
    query: QuerySpec
    horizon: int = 24
    model: str = "auto"


class AQIRequest(BaseModel):
    concentrations: dict[str, float]
    standard_id: str


class ForecastAQIRequest(BaseModel):
    query: QuerySpec
    horizon: int = 24
    standard_id: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="AQora", version="0.1.0")
    analytics = AnalyticsEngine()
    agent = ConversationalAgent()
    static = Path(__file__).with_name("static")

    @app.exception_handler(AQAError)
    async def aqa_error(_: Request, exc: AQAError):
        return JSONResponse(status_code=422, content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}})

    @app.get("/api/health")
    def health():
        datasets = LakehouseCatalog().datasets
        return {
            "status": "ok", "datasets": [item.dataset for item in datasets],
            "countries": len(datasets), "scope": "global_lakehouse",
        }

    @app.get("/api/data_coverage")
    def data_coverage():
        datasets = LakehouseCatalog().datasets
        rows = [
            {
                "dataset": item.dataset, "country_code": item.country_code, "country_name": item.country_name,
                "available_start": item.available_start.isoformat(), "available_end": item.available_end.isoformat(),
            }
            for item in datasets
        ]
        rows.sort(key=lambda row: row["country_name"])
        return {"rows": rows}

    @app.post("/api/query")
    def query(spec: QuerySpec):
        return analytics.execute(spec)

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        context = request.context.model_dump(mode="json") if request.context else None
        return agent.ask(request.question, provider=request.provider, model=request.model, context=context)

    @app.get("/api/models")
    def models():
        llm_config = agent.config["llm"]
        available: list[str] = []
        try:
            response = httpx.get(llm_config["ollama_base_url"].rstrip("/") + "/api/tags", timeout=2.0)
            response.raise_for_status()
            available = sorted(item["name"] for item in response.json().get("models", []))
        except httpx.HTTPError:
            pass
        return {
            "provider": llm_config["default_provider"],
            "default_model": llm_config["default_model"],
            "available_models": available,
        }

    @app.post("/api/forecast")
    def forecast(request: ForecastRequest):
        if request.query.dataset.value == "ALL":
            raise ValidationError("Forecast one country dataset at a time so a single time series and AQI context are unambiguous")
        if len(request.query.locations) != 1 or len(request.query.pollutants) != 1:
            raise ValidationError("Concentration forecasting requires exactly one location and one pollutant")
        if request.query.time_range.end is not None:
            issue = request.query.time_range.end
        else:
            issue = LakehouseCatalog().by_dataset(request.query.dataset.value).available_end
        model_text = ""
        if request.model == "linear_trend": model_text = " using linear trend"
        if request.model == "seasonal_naive": model_text = " using seasonal naive only"
        question = (
            f"Forecast {request.query.locations[0]} {request.query.pollutants[0]} for the next {request.horizon} hours "
            f"using data through {issue:%d %B %Y %H:%M}{model_text}."
        )
        safe = agent.ask(question)
        if safe["kind"] == "limitation":
            raise ValidationError(safe["answer"])
        forecast_payload = {
            "model": safe["analysis"]["model"], "horizon": request.horizon, "values": safe["data"],
            "selection_scores": safe["analysis"]["selection_scores_observed_mae"],
            "metrics": safe["analysis"]["metrics"], "baseline_metrics": safe["analysis"]["baseline_metrics"],
            "input_quality": safe["analysis"]["input_quality"], "interval_level": safe["analysis"]["interval_level"],
            "disclaimer": "Forecasts are model estimates, not measurements.",
        }
        return {"query": request.query.model_dump(mode="json"), "forecast": forecast_payload, "conversational_result": safe}

    @app.get("/api/aqi/standards")
    def standards():
        return {"defaults": DEFAULT_STANDARD, "standards": [asdict(item) for item in STANDARDS.values()]}

    @app.post("/api/aqi")
    def aqi(request: AQIRequest):
        return calculate_aqi(request.concentrations, request.standard_id)

    @app.post("/api/forecast/aqi")
    def forecast_aqi(request: ForecastAQIRequest):
        if request.query.dataset.value == "ALL":
            raise ValidationError("Forecast AQI requires CPCB or CNEMC because each country uses a different default AQI standard")
        if len(request.query.locations) != 1:
            raise ValidationError("Forecast AQI requires exactly one location")
        standard_id = request.standard_id or DEFAULT_STANDARD[request.query.dataset.value]
        # Anchor the forecast context to the dataset's actual current data
        # (or an explicit issue time, if the query already named one) rather
        # than reusing whatever historical period the base chat question
        # happened to query -- otherwise this silently forecasts starting
        # right after that old window instead of the next real 24 hours,
        # matching the same issue-time convention /api/forecast already uses.
        if request.query.time_range.end is not None:
            issue = request.query.time_range.end
        else:
            issue = LakehouseCatalog().by_dataset(request.query.dataset.value).available_end
        context_hours = int(analytics.config.get("forecasting", {}).get("context_hours", 168))
        context_start = issue - timedelta(hours=context_hours - 1)
        pollutant_forecasts = {}
        model_scores = {}
        for pollutant in ("PM25", "PM10", "NO2", "SO2", "CO", "O3"):
            raw = request.query.model_dump(mode="json")
            raw.update({
                "pollutants": [pollutant], "resolution": "hourly", "metrics": ["mean"], "limit": 10000,
                "operation": "aggregate",
                "time_range": {"start": context_start.isoformat(), "end": issue.isoformat()},
            })
            result = analytics.execute(QuerySpec.model_validate(raw))
            predicted = forecast_series(result["data"], horizon=request.horizon)
            pollutant_forecasts[pollutant] = predicted.values
            model_scores[pollutant] = {"model": predicted.model, "selection_scores": predicted.selection_scores}
        output = []
        for index in range(request.horizon):
            concentrations = {name: values[index]["forecast"] for name, values in pollutant_forecasts.items()}
            output.append({"timestamp": next(iter(pollutant_forecasts.values()))[index]["timestamp"], "concentrations": concentrations, **calculate_aqi(concentrations, standard_id)})
        return {"standard_id": standard_id, "horizon": request.horizon, "models": model_scores, "values": output, "disclaimer": "Forecast AQI values are model estimates, not measurements."}

    @app.post("/api/export/csv")
    def export_csv(spec: QuerySpec):
        result = analytics.execute(spec)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(result["data"][0]))
        writer.writeheader(); writer.writerows(result["data"])
        return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=aqa-result.csv"})

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("aqa.api:app", host="0.0.0.0", port=int(os.environ.get("AQA_PORT", "8000")))
