"""
Proyecto: Predicción de Demanda de Bicicletas - Bike Sharing
Autor: Julissa Lescano
Asignatura: Modelos y Simulación
Dataset: Bike Sharing Dataset (Kaggle hour.csv)
"""

import logging
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
def setup_logging():
  """Configura el sistema de logging para el proyecto"""
  log_dir = 'console'
  os.makedirs(log_dir, exist_ok=True)

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  log_file = f'{log_dir}/training_{timestamp}.log'

  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
      logging.FileHandler(log_file, encoding='utf-8'),
      logging.StreamHandler()
    ]
  )
  return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# CLASE PRINCIPAL PARA EL MODELO
# ============================================================================
class BikeShareModel:
  """
  Clase para manejar el pipeline completo de Machine Learning
  para predecir la demanda de bicicletas usando datos horarios
  """

  def __init__(self, data_path):
    """
    Inicializa el modelo

    Parameters:
    -----------
    data_path : str
        Ruta al archivo CSV con los datos (ahora hour.csv)
    """
    self.data_path = data_path
    self.df = None
    self.models = {}
    self.best_model = None
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
    self.y_pred = {}
    self.metrics = {}
    self.feature_names = None

    logger.info("=" * 80)
    logger.info("INICIO DEL PROYECTO: Predicción de Demanda de Bicicletas (Horaria)")
    logger.info("=" * 80)

  def load_data(self):
    """Carga y valida los datos del CSV"""
    try:
      logger.info(f"Cargando datos desde: {self.data_path}")
      self.df = pd.read_csv(self.data_path)

      # Limpia los nombres de las columnas
      self.df.columns = self.df.columns.str.strip()

      logger.info(f"Datos cargados exitosamente")
      logger.info(f"Dimensiones: {self.df.shape[0]} filas x {self.df.shape[1]} columnas")
      logger.info(f"Columnas: {list(self.df.columns)}")

      # Validar columnas necesarias
      required_cols = ['cnt', 'temp', 'atemp', 'hum', 'windspeed', 'hr']
      missing_cols = [col for col in required_cols if col not in self.df.columns]

      if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

      return True

    except FileNotFoundError:
      logger.error(f"Error: No se encontró el archivo {self.data_path}")
      return False
    except Exception as e:
      logger.error(f"Error al cargar datos: {str(e)}")
      return False

  def feature_engineering(self):
    """Ingeniería de características: Codificación cíclica para 'hr'"""
    logger.info("\nIngeniería de características...")
    # Codificación cíclica para hora (hr)
    self.df['sin_hr'] = np.sin(2 * np.pi * self.df['hr'] / 24)
    self.df['cos_hr'] = np.cos(2 * np.pi * self.df['hr'] / 24)
    logger.info("[OK] Codificación cíclica aplicada a 'hr' (sin y cos)")

  def exploratory_analysis(self):
    """Realiza análisis exploratorio de datos"""
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS EXPLORATORIO DE DATOS")
    logger.info("=" * 80)

    try:
      # Información básica
      logger.info("\n--- INFORMACIÓN DEL DATASET ---")
      logger.info(f"\nPrimeras filas:\n{self.df.head()}")
      logger.info(f"\nInformación de tipos de datos:\n{self.df.dtypes}")
      logger.info(f"\nEstadísticas descriptivas:\n{self.df.describe()}")

      # Valores nulos
      null_counts = self.df.isnull().sum()
      if null_counts.sum() > 0:
        logger.warning(f"\nValores nulos encontrados:\n{null_counts[null_counts > 0]}")
      else:
        logger.info("\n[OK] No hay valores nulos en el dataset")

      # Estadísticas de la variable objetivo
      logger.info("\n--- ESTADÍSTICAS DE LA VARIABLE OBJETIVO (cnt) ---")
      logger.info(f"Media: {self.df['cnt'].mean():.2f}")
      logger.info(f"Mediana: {self.df['cnt'].median():.2f}")
      logger.info(f"Desviación estándar: {self.df['cnt'].std():.2f}")
      logger.info(f"Mínimo: {self.df['cnt'].min():.2f}")
      logger.info(f"Máximo: {self.df['cnt'].max():.2f}")

      # Crear directorio para gráficos
      os.makedirs('plots', exist_ok=True)

      # Gráfico 1: Distribución de la variable objetivo
      plt.figure(figsize=(10, 6))
      plt.hist(self.df['cnt'], bins=50, edgecolor='black', alpha=0.7)
      plt.xlabel('Número de bicicletas rentadas (cnt)', fontsize=12)
      plt.ylabel('Frecuencia', fontsize=12)
      plt.title('Distribución de la Demanda de Bicicletas (Horaria)', fontsize=14, fontweight='bold')
      plt.grid(True, alpha=0.3)
      plt.savefig('plots/01_distribucion_cnt.png', dpi=300, bbox_inches='tight')
      plt.close()
      logger.info("[OK] Gráfico guardado: plots/01_distribucion_cnt.png")

      # Gráfico 2: Matriz de correlación (sin sin_hr y cos_hr, ya que se crean después)
      numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
      correlation_matrix = self.df[numeric_cols].corr()

      plt.figure(figsize=(10, 8))
      sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                  center=0, square=True, linewidths=1)
      plt.title('Matriz de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
      plt.tight_layout()
      plt.savefig('plots/02_correlacion.png', dpi=300, bbox_inches='tight')
      plt.close()
      logger.info("[OK] Gráfico guardado: plots/02_correlacion.png")

      # Gráfico 3: Grafo de correlación (Teoría de Grafos)
      G = nx.Graph()
      for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
          if abs(correlation_matrix.iloc[i, j]) > 0.3:  # Umbral para aristas significativas
            G.add_edge(numeric_cols[i], numeric_cols[j], weight=correlation_matrix.iloc[i, j])

      plt.figure(figsize=(12, 8))
      pos = nx.spring_layout(G)
      nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500,
              edge_color='gray', width=2, font_size=10)
      edge_labels = nx.get_edge_attributes(G, 'weight')
      nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
      plt.title('Grafo de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
      plt.axis('off')
      plt.savefig('plots/03_grafo_correlacion.png', dpi=300, bbox_inches='tight')
      plt.close()
      logger.info("[OK] Gráfico guardado: plots/03_grafo_correlacion.png")

      # Gráfico 4: Pairplot de variables principales
      logger.info("Generando pairplot (puede tardar un momento)...")
      main_vars = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'cnt']
      pairplot_data = self.df[main_vars].sample(min(1000, len(self.df)), random_state=42)
      sns.pairplot(pairplot_data, height=2.5, plot_kws={'alpha': 0.6})
      plt.savefig('plots/04_pairplot.png', dpi=300, bbox_inches='tight')
      plt.close()
      logger.info("[OK] Gráfico guardado: plots/04_pairplot.png")

      # Gráfico 5: Boxplots por categorías
      fig, axes = plt.subplots(2, 4, figsize=(20, 10))
      categorical_vars = ['season', 'mnth', 'weathersit', 'holiday', 'weekday', 'workingday', 'yr', 'hr']

      for idx, var in enumerate(categorical_vars):
        row = idx // 4
        col = idx % 4
        self.df.boxplot(column='cnt', by=var, ax=axes[row, col])
        axes[row, col].set_title(f'Demanda por {var}')
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('cnt')

      plt.suptitle('')
      plt.tight_layout()
      plt.savefig('plots/05_boxplots_categorias.png', dpi=300, bbox_inches='tight')
      plt.close()
      logger.info("[OK] Gráfico guardado: plots/05_boxplots_categorias.png")

      return True

    except Exception as e:
      logger.error(f"Error en análisis exploratorio: {str(e)}")
      return False

  def prepare_data(self, test_size=0.2, random_state=42):
    """
    Prepara los datos para entrenamiento

    Parameters:
    -----------
    test_size : float
        Proporción de datos para test (default: 0.2)
    random_state : int
        Semilla para reproducibilidad
    """
    logger.info("\n" + "=" * 80)
    logger.info("PREPARACIÓN DE DATOS")
    logger.info("=" * 80)

    try:
      # Ingeniería de características
      self.feature_engineering()

      # Eliminar columnas no necesarias
      columns_to_drop = ['instant', 'dteday', 'casual', 'registered', 'hr']  # Drop hr original, use sin/cos
      X = self.df.drop(columns_to_drop + ['cnt'], axis=1, errors='ignore')
      y = self.df['cnt']

      logger.info(f"\nVariables predictoras: {list(X.columns)}")
      logger.info(f"Variable objetivo: cnt")

      # Dividir datos
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
      )

      logger.info(f"\nDivisión de datos:")
      logger.info(f"  - Entrenamiento: {len(self.X_train)} muestras ({(1 - test_size) * 100:.0f}%)")
      logger.info(f"  - Prueba: {len(self.X_test)} muestras ({test_size * 100:.0f}%)")

      return True

    except Exception as e:
      logger.error(f"Error en preparación de datos: {str(e)}")
      return False

  def build_pipelines(self):
    """Construye pipelines para múltiples modelos"""
    logger.info("\n" + "=" * 80)
    logger.info("CONSTRUCCIÓN DE PIPELINES")
    logger.info("=" * 80)

    try:
      # Definir columnas numéricas y categóricas
      numeric_features = ['temp', 'atemp', 'hum', 'windspeed', 'sin_hr', 'cos_hr']
      categorical_features = ['season', 'yr', 'mnth', 'holiday',
                              'weekday', 'workingday', 'weathersit']

      logger.info(f"\nCaracterísticas numéricas: {numeric_features}")
      logger.info(f"Características categóricas: {categorical_features}")

      # Crear transformadores
      numeric_transformer = StandardScaler()
      categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

      # Preprocessor común
      preprocessor = ColumnTransformer(
        transformers=[
          ('num', numeric_transformer, numeric_features),
          ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
      )

      # Pipeline para Linear Regression (usa mínimos cuadrados ordinarios)
      self.models['Linear'] = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression(fit_intercept=True))  # Mínimos cuadrados explícito
      ])

      # Pipeline para Random Forest
      self.models['RF'] = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
      ])

      # Pipeline para Gradient Boosting
      self.models['GB'] = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
      ])

      logger.info("\n[OK] Pipelines construidos para Linear, RF y GB")

      return True

    except Exception as e:
      logger.error(f"Error al construir pipelines: {str(e)}")
      return False

  def build_nn_model(self):
    """Construye el modelo de Red Neuronal con Keras"""
    logger.info("\nConstruyendo modelo de Red Neuronal...")

    # Preprocesar datos para NN (usar el mismo preprocessor)
    preprocessor = self.models['Linear'].named_steps['preprocessor']
    self.X_train_scaled = preprocessor.fit_transform(self.X_train)
    self.X_test_scaled = preprocessor.transform(self.X_test)

    # Modelo Sequential
    model = Sequential()
    model.add(Dense(128, input_dim=self.X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Salida para regresión

    # Compilar
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    self.models['NN'] = model

    logger.info("[OK] Modelo NN construido")

  def train_models(self):
    """Entrena todos los modelos"""
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENAMIENTO DE MODELOS")
    logger.info("=" * 80)

    try:
      for name, model in self.models.items():
        if name == 'NN':
          continue  # Manejar NN por separado
        logger.info(f"\nEntrenando {name}...")
        model.fit(self.X_train, self.y_train)
        self.y_pred[name] = model.predict(self.X_test)
        logger.info(f"[OK] {name} entrenado")

      # Entrenar NN en dos fases
      logger.info("\nEntrenando NN en fases...")
      nn_model = self.models['NN']

      # Fase 1: Entrenamiento inicial con LR alto
      nn_model.optimizer.learning_rate = 0.01
      nn_model.fit(self.X_train_scaled, self.y_train, epochs=50, batch_size=64, validation_split=0.2,
                   callbacks=[EarlyStopping(patience=10)], verbose=0)
      logger.info("[OK] Fase 1 completada")

      # Fase 2: Fine-tuning con LR bajo
      nn_model.optimizer.learning_rate = 0.001
      nn_model.fit(self.X_train_scaled, self.y_train, epochs=100, batch_size=32, validation_split=0.2,
                   callbacks=[EarlyStopping(patience=15)], verbose=0)
      logger.info("[OK] Fase 2 completada")

      self.y_pred['NN'] = nn_model.predict(self.X_test_scaled).flatten()

      return True

    except Exception as e:
      logger.error(f"Error durante el entrenamiento: {str(e)}")
      return False

  def tune_hyperparams(self):
    """Ajuste de hiperparámetros para RF y GB usando GridSearchCV"""
    logger.info("\nAjuste de hiperparámetros para RF y GB...")

    # Para RF
    rf_param_grid = {
      'regressor__n_estimators': [100, 200],
      'regressor__max_depth': [10, 20]
    }
    rf_grid = GridSearchCV(self.models['RF'], rf_param_grid, cv=3, scoring='r2')
    rf_grid.fit(self.X_train, self.y_train)
    self.models['RF'] = rf_grid.best_estimator_
    logger.info(f"Mejores params RF: {rf_grid.best_params_}")

    # Para GB
    gb_param_grid = {
      'regressor__n_estimators': [100, 200],
      'regressor__learning_rate': [0.01, 0.1],
      'regressor__max_depth': [3, 5]
    }
    gb_grid = GridSearchCV(self.models['GB'], gb_param_grid, cv=3, scoring='r2')
    gb_grid.fit(self.X_train, self.y_train)
    self.models['GB'] = gb_grid.best_estimator_
    logger.info(f"Mejores params GB: {gb_grid.best_params_}")

  def evaluate_models(self):
    """Evalúa todos los modelos y selecciona el mejor"""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUACIÓN DE MODELOS")
    logger.info("=" * 80)

    try:
      for name in self.models.keys():
        y_pred = self.y_pred[name]
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        self.metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        logger.info(f"\n--- {name} ---")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")

      # Seleccionar el mejor basado en R²
      self.best_model = max(self.metrics, key=lambda k: self.metrics[k]['R2'])
      logger.info(f"\nMejor modelo: {self.best_model} con R² = {self.metrics[self.best_model]['R2']:.4f}")

      return True

    except Exception as e:
      logger.error(f"Error en evaluación: {str(e)}")
      return False

  def plot_results(self):
    """Genera gráficos de resultados para el mejor modelo"""
    logger.info("\n" + "=" * 80)
    logger.info("GENERACIÓN DE GRÁFICOS DE RESULTADOS")
    logger.info("=" * 80)

    try:
      y_pred = self.y_pred[self.best_model]

      # Gráfico 1: Predicciones vs Valores Reales
      plt.figure(figsize=(10, 6))
      plt.scatter(self.y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
      plt.plot([self.y_test.min(), self.y_test.max()],
               [self.y_test.min(), self.y_test.max()],
               'r--', lw=2, label='Predicción perfecta')
      plt.xlabel('Valores Reales', fontsize=12)
      plt.ylabel('Valores Predichos', fontsize=12)
      plt.title(f'Predicciones vs Valores Reales ({self.best_model})', fontsize=14, fontweight='bold')
      plt.legend()
      plt.grid(True, alpha=0.3)
      plt.savefig('plots/06_pred_vs_real.png', dpi=300, bbox_inches='tight')
      plt.close()
      logger.info("[OK] Gráfico guardado: plots/06_pred_vs_real.png")

      return True

    except Exception as e:
      logger.error(f"Error al generar gráficos: {str(e)}")
      return False

  def save_model(self, filename='best_bike_model.pkl'):
    """Guarda el mejor modelo y el preprocessor"""
    try:
      import joblib
      os.makedirs('models', exist_ok=True)
      filepath = f'models/{filename}'

      # Guardar el preprocessor del modelo lineal (se usa como referencia)
      preprocessor = self.models['Linear'].named_steps['preprocessor']
      joblib.dump(preprocessor, 'models/preprocessor.pkl')
      logger.info(f"[OK] Preprocessor guardado en: models/preprocessor.pkl")

      # Guardar el mejor modelo
      if self.best_model == 'NN':
        self.models['NN'].save(filepath.replace('.pkl', '.h5'))
      else:
        joblib.dump(self.models[self.best_model], filepath)
      logger.info(f"[OK] Mejor modelo ({self.best_model}) guardado en: {filepath}")

      return True
    except Exception as e:
      logger.error(f"Error al guardar modelo: {str(e)}")
      return False

  def predict_new_case(self, new_data):
    """
    Predice la demanda para un nuevo caso usando el mejor modelo

    Parameters:
    -----------
    new_data : dict
        Diccionario con las características del nuevo caso
        Ejemplo: {'temp': 0.6, 'atemp': 0.6, 'hum': 0.5, 'windspeed': 0.2,
                 'season': 3, 'yr': 1, 'mnth': 7, 'holiday': 0, 'weekday': 4,
                 'workingday': 1, 'weathersit': 1, 'hr': 17}

    Returns:
    --------
    float
        Predicción de la demanda (cnt)
    """
    logger.info("\n" + "=" * 80)
    logger.info("PREDICCIÓN PARA NUEVO CASO")
    logger.info("=" * 80)

    try:
      # Convertir diccionario a DataFrame y preprocesar
      new_df = pd.DataFrame([new_data])
      new_df['sin_hr'] = np.sin(2 * np.pi * new_df['hr'] / 24)
      new_df['cos_hr'] = np.cos(2 * np.pi * new_df['hr'] / 24)
      X_new = new_df.drop(['hr'], axis=1, errors='ignore')  # Drop hr original

      # Usar el mismo preprocessor del modelo lineal
      preprocessor = self.models['Linear'].named_steps['preprocessor']
      X_new_processed = preprocessor.transform(X_new)

      if self.best_model == 'NN':
        prediction = self.models['NN'].predict(X_new_processed).flatten()[0]
      else:
        prediction = self.models[self.best_model].predict(X_new_processed)[0]

      logger.info(f"Predicción para el nuevo caso: {prediction:.2f} bicicletas")
      return prediction

    except Exception as e:
      logger.error(f"Error en predicción: {str(e)}")
      return None


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
  """Función principal para ejecutar el pipeline completo"""

  # Ruta al archivo CSV (ajustar a hour.csv)
  DATA_PATH = 'hour.csv'  # Cambia esto a tu ruta

  # Crear instancia del modelo
  bike_model = BikeShareModel(DATA_PATH)

  # Pipeline completo
  if not bike_model.load_data():
    return

  if not bike_model.exploratory_analysis():
    return

  if not bike_model.prepare_data(test_size=0.2, random_state=42):
    return

  if not bike_model.build_pipelines():
    return

  bike_model.build_nn_model()

  bike_model.tune_hyperparams()  # Tune RF y GB

  if not bike_model.train_models():
    return

  if not bike_model.evaluate_models():
    return

  if not bike_model.plot_results():
    return

  bike_model.save_model()

  # Ejemplo de predicción para un caso nuevo (ajusta los valores según necesidad)
  new_case = {
    'temp': 0.6, 'atemp': 0.6, 'hum': 0.5, 'windspeed': 0.2,
    'season': 3, 'yr': 1, 'mnth': 7, 'holiday': 0, 'weekday': 4,
    'workingday': 1, 'weathersit': 1, 'hr': 17
  }
  bike_model.predict_new_case(new_case)

  logger.info("\n" + "=" * 80)
  logger.info("PROYECTO FINALIZADO EXITOSAMENTE")
  logger.info("=" * 80)


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
  main()
