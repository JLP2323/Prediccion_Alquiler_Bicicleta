"""
VeloPredict Analytics - Backend FastAPI
Proyecto: Predicción de Demanda de Bicicletas (Bike Sharing)
Autora:  Julissa Lescano 
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="VeloPredict Analytics API",
    description="API para prediccion y analisis de demanda de bicicletas compartidas.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
state: dict = {}


def load_and_train(data_path: str = "hour.csv") -> dict:
    """Loads CSV, trains a RandomForest pipeline and returns model + stats."""
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Feature engineering
    df["sin_hr"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["cos_hr"] = np.cos(2 * np.pi * df["hr"] / 24)

    # KPI stats
    total_rentals = int(df["cnt"].sum())
    avg_hourly = round(float(df["cnt"].mean()), 2)
    casual_pct = round(float(df["casual"].sum() / df["cnt"].sum() * 100), 1)
    registered_pct = round(100 - casual_pct, 1)

    # Temp denormalized: dataset range -8C to 41C  (t_C = t_norm * 47 - 8)
    avg_temp_c = round(float(df["temp"].mean()) * 47 - 8, 1)
    avg_atemp_c = round(float(df["atemp"].mean()) * 50 - 16, 1)

    # Hourly demand: workingday vs weekend
    hourly_working = {
        int(k): round(float(v), 1)
        for k, v in df[df["workingday"] == 1].groupby("hr")["cnt"].mean().items()
    }
    hourly_weekend = {
        int(k): round(float(v), 1)
        for k, v in df[df["workingday"] == 0].groupby("hr")["cnt"].mean().items()
    }

    # Weather impact
    weather_labels = {1: "Despejado", 2: "Nublado", 3: "Lluvia ligera", 4: "Lluvia intensa"}
    weather_impact = {
        weather_labels.get(int(k), str(k)): round(float(v), 1)
        for k, v in df.groupby("weathersit")["cnt"].mean().items()
    }

    # Monthly avg
    monthly_avg = {
        int(k): round(float(v), 1)
        for k, v in df.groupby("mnth")["cnt"].mean().items()
    }

    # Seasonal avg
    season_labels = {1: "Invierno", 2: "Primavera", 3: "Verano", 4: "Otono"}
    seasonal_avg = {
        season_labels.get(int(k), str(k)): round(float(v), 1)
        for k, v in df.groupby("season")["cnt"].mean().items()
    }

    # Train ML model
    logger.info("Training RandomForest model...")
    NUMERIC_FEATURES = ["temp", "atemp", "hum", "windspeed", "sin_hr", "cos_hr"]
    CATEGORICAL_FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]

    X = df.drop(["instant", "dteday", "casual", "registered", "hr", "cnt"], axis=1, errors="ignore")
    y = df["cnt"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
    r2 = round(float(r2_score(y_test, y_pred)), 4)

    logger.info(f"Model trained MAE={mae}, RMSE={rmse}, R2={r2}")

    return {
        "pipeline": pipeline,
        "stats": {
            "total_rentals": total_rentals,
            "avg_hourly": avg_hourly,
            "casual_pct": casual_pct,
            "registered_pct": registered_pct,
            "avg_temp_c": avg_temp_c,
            "avg_atemp_c": avg_atemp_c,
            "model_metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        },
        "charts": {
            "hourly_working": hourly_working,
            "hourly_weekend": hourly_weekend,
            "weather_impact": weather_impact,
            "monthly_avg": monthly_avg,
            "seasonal_avg": seasonal_avg,
        },
    }


@app.on_event("startup")
async def startup_event():
    global state
    data_path = BASE_DIR / "hour.csv"
    if not data_path.exists():
        logger.error("hour.csv not found!")
        return
    result = load_and_train(str(data_path))
    state["pipeline"] = result["pipeline"]
    state["stats"] = result["stats"]
    state["charts"] = result["charts"]
    logger.info("VeloPredict Analytics ready")


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    hr: int = Field(..., ge=0, le=23)
    temp_c: float = Field(..., ge=-8, le=41)
    atemp_c: float = Field(..., ge=-16, le=50)
    hum: float = Field(..., ge=0, le=100)
    windspeed_kmh: float = Field(..., ge=0, le=100)
    season: int = Field(..., ge=1, le=4)
    mnth: int = Field(..., ge=1, le=12)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    yr: int = Field(1, ge=0, le=1)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"), media_type="text/html")


@app.get("/api/stats")
async def get_stats():
    if not state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return JSONResponse(content={
        "stats": state["stats"],
        "charts": state["charts"],
    })


@app.post("/api/predict")
async def predict(payload: PredictRequest):
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Normalize inputs to model scale
    temp_norm = max(0.0, min(1.0, (payload.temp_c + 8) / 47))
    atemp_norm = max(0.0, min(1.0, (payload.atemp_c + 16) / 50))
    hum_norm = payload.hum / 100
    wind_norm = max(0.0, min(1.0, payload.windspeed_kmh / 67))

    sin_hr = float(np.sin(2 * np.pi * payload.hr / 24))
    cos_hr = float(np.cos(2 * np.pi * payload.hr / 24))

    features = pd.DataFrame([{
        "season": payload.season,
        "yr": payload.yr,
        "mnth": payload.mnth,
        "holiday": payload.holiday,
        "weekday": payload.weekday,
        "workingday": payload.workingday,
        "weathersit": payload.weathersit,
        "temp": temp_norm,
        "atemp": atemp_norm,
        "hum": hum_norm,
        "windspeed": wind_norm,
        "sin_hr": sin_hr,
        "cos_hr": cos_hr,
    }])

    pipeline: Pipeline = state["pipeline"]
    raw_prediction = float(pipeline.predict(features)[0])
    cnt = max(0, round(raw_prediction))

    casual_est = max(0, round(cnt * 0.188))
    registered_est = max(0, cnt - casual_est)

    if cnt < 100:
        label = "Demanda Baja"
        label_color = "green"
    elif cnt < 300:
        label = "Demanda Moderada"
        label_color = "amber"
    else:
        label = "Alta Demanda"
        label_color = "red"

    return JSONResponse(content={
        "cnt": cnt,
        "casual_est": casual_est,
        "registered_est": registered_est,
        "demand_label": label,
        "demand_color": label_color,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": "pipeline" in state}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
