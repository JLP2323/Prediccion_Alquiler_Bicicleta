# 🚲 VeloPredict Analytics — Predicción de Demanda de Bicicletas

> Proyecto académico de Machine Learning con dashboard web interactivo.
> **Autores:** Eduardo Quinteros · Julissa Lescano · Dayannara · Ledesma

---

## 📋 Descripción

**VeloPredict Analytics** predice la demanda horaria de bicicletas compartidas usando el dataset *Bike Sharing* (`hour.csv`) de Kaggle. El sistema combina un pipeline de Machine Learning (Random Forest, Gradient Boosting, Red Neuronal) con una **API REST en FastAPI** y un dashboard web interactivo.

> *¿Cómo influyen las variables climáticas y de calendario en la demanda horaria de bicicletas?*

---

## 🗂️ Estructura del repositorio

```
Prediccion_Alquiler_Bicicleta/
├── app.py                    # 🌐 Backend FastAPI (VeloPredict API)
├── main..py                  # 🤖 Pipeline ML completo (EDA, entrenamiento, evaluación)
├── predict_interactive.py    # 💬 Predicción interactiva por consola
├── hour.csv                  # 📊 Dataset (Bike Sharing - Kaggle)
├── requirements.txt          # Dependencias ML (TensorFlow, sklearn, etc.)
├── requirements_web.txt      # Dependencias web (FastAPI, uvicorn)
├── templates/
│   └── index.html            # Frontend del dashboard
├── static/
│   ├── css/                  # Estilos
│   └── js/app.js             # Lógica del dashboard
├── models/                   # (generado) Modelos entrenados
└── plots/                    # (generado) Gráficos EDA
```

---

## 🛠️ Tecnologías

| Capa | Tecnología |
|---|---|
| Machine Learning | scikit-learn · RandomForest · GradientBoosting · Keras/TensorFlow |
| API Backend | FastAPI · Uvicorn |
| Frontend | HTML · CSS · JavaScript (Vanilla) |
| Data | Pandas · NumPy |
| Dataset | Bike Sharing Dataset (UCI / Kaggle) |

---

## ⚙️ Instalación

### 1. Clonar el repositorio
```powershell
git clone https://github.com/TU_USUARIO/Prediccion_Alquiler_Bicicleta.git
cd Prediccion_Alquiler_Bicicleta
```

### 2. Crear entorno virtual
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

### 3. Instalar dependencias

**Para la app web (FastAPI):**
```powershell
pip install -r requirements_web.txt
```

**Para el pipeline ML completo:**
```powershell
pip install -r requirements.txt
```

### 4. Dataset
Descarga `hour.csv` desde [Kaggle - Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) y colócalo en la **raíz del proyecto**.

---

## 🚀 Ejecución

### 🌐 App Web — Dashboard FastAPI
```powershell
& .\.venv\Scripts\Activate.ps1
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Abre tu navegador en: **http://127.0.0.1:8000**

### 🤖 Pipeline ML completo
```powershell
python .\main..py
```
Genera modelos en `models/` y gráficos en `plots/`.

### 💬 Predicción interactiva por consola
```powershell
python .\predict_interactive.py
```

---

## 🔌 Endpoints de la API

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Dashboard web |
| `/api/stats` | GET | KPIs y datos para gráficos |
| `/api/predict` | POST | Predicción de demanda horaria |
| `/api/health` | GET | Estado del servidor |
| `/docs` | GET | Documentación Swagger interactiva |

### Ejemplo de predicción
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"hr":8,"temp_c":18.0,"atemp_c":17.0,"hum":60.0,"windspeed_kmh":15.0,"season":2,"mnth":5,"holiday":0,"weekday":2,"workingday":1,"weathersit":1,"yr":1}'
```

**Respuesta:**
```json
{
  "cnt": 342,
  "casual_est": 64,
  "registered_est": 278,
  "demand_label": "Alta Demanda",
  "demand_color": "red"
}
```

---

## 📊 Metodología ML

1. **EDA** — Estadísticos descriptivos, correlaciones, patrones temporales.
2. **Feature Engineering** — Codificación cíclica de hora: `sin_hr = sin(2π·hr/24)`, `cos_hr = cos(2π·hr/24)`.
3. **Preprocesado** — `ColumnTransformer`: `StandardScaler` (numéricas) + `OneHotEncoder(drop='first')` (categóricas).
4. **Modelos entrenados:**
   - `LinearRegression` — baseline explicable
   - `RandomForestRegressor` — ensemble no lineal
   - `GradientBoostingRegressor` — ensemble boosting
   - Red Neuronal Sequential (Keras) — interacciones complejas
5. **Ajuste** — `GridSearchCV` para RF y GB.
6. **Evaluación** — MAE, RMSE, R² en test 80/20.

### Métricas de referencia (Random Forest)

| Métrica | Valor aprox. |
|---|---|
| MAE | ~30–40 bicicletas |
| RMSE | ~50–65 bicicletas |
| R² | ~0.93+ |

---

## 📦 Dataset

- **Fuente:** [UCI / Kaggle — Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
- **Archivo:** `hour.csv` — 17,379 registros horarios (2011–2012)
- **Variable objetivo:** `cnt` (total de alquileres por hora)

---

## 👥 Autores

| Nombre |
|---|
| Eduardo Quinteros |
| Julissa Lescano |
| Dayannara |
| Ledesma |

---

## 📄 Licencia

Proyecto académico — uso educativo.
