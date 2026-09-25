import logging
import os

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BikePredictor:
  def __init__(self, model_path=None):
    """
    Inicializa el predictor con el modelo guardado.

    Parameters:
    -----------
    model_path : str, optional
        Ruta al modelo guardado (pkl para RF/GB, h5 para NN)
        Si no se especifica, intentará cargar primero .h5 y luego .pkl
    """
    if model_path is None:
      # Intentar primero con .h5, luego con .pkl
      if os.path.exists('models/best_bike_model.h5'):
        model_path = 'models/best_bike_model.h5'
      elif os.path.exists('models/best_bike_model.pkl'):
        model_path = 'models/best_bike_model.pkl'
      else:
        raise FileNotFoundError("No se encontró ningún modelo en models/best_bike_model.[h5|pkl]")
    
    self.model_path = model_path
    self.model_path = model_path
    self.model = self._load_model()
    self.preprocessor = self._load_preprocessor()

  def _load_model(self):
    """Carga el modelo guardado."""
    try:
      if self.model_path.endswith('.pkl'):
        model = joblib.load(self.model_path)
        logger.info(f"Modelo cargado desde {self.model_path} (RF/GB)")
      elif self.model_path.endswith('.h5'):
        model = load_model(self.model_path)
        logger.info(f"Modelo cargado desde {self.model_path} (NN)")
      else:
        raise ValueError("Formato de modelo no soportado. Use .pkl o .h5")
      return model
    except Exception as e:
      logger.error(f"Error al cargar el modelo: {str(e)}")
      return None

  def _load_preprocessor(self):
    """Carga o reconstruye el preprocessor desde el modelo lineal guardado."""
    try:
      if self.model_path.endswith('.pkl'):
        pipeline = joblib.load(self.model_path)
        return pipeline.named_steps['preprocessor']
      else:
        preprocessor_path = 'models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
          return joblib.load(preprocessor_path)
        else:
          logger.warning("Preprocessor no encontrado. Asegúrate de guardarlo previamente.")
          return None
    except Exception as e:
      logger.error(f"Error al cargar preprocessor: {str(e)}")
      return None

  def normalize_temperature(self, temp_input, unit):
    """Convierte temperatura a la escala normalizada del modelo (0 a 1, basada en Celsius)."""
    try:
      temp_c = float(temp_input)
      if unit.lower() == 'f':
        temp_c = (temp_c - 32) * 5 / 9  # Fahrenheit a Celsius
      elif unit.lower() == 'k':
        temp_c = temp_c - 273.15  # Kelvin a Celsius

      # Rango aproximado del dataset: -8°C a 39°C
      min_temp, max_temp = -8, 39
      normalized_temp = (temp_c - min_temp) / (max_temp - min_temp)
      return max(0, min(1, normalized_temp))  # Asegurar dentro de [0, 1]
    except ValueError:
      return None

  def get_user_input(self):
    """Obtiene y valida la entrada del usuario con lenguaje natural."""
    logger.info("¡Bienvenido! Ingresa los datos para predecir cuántas bicicletas se rentarán hoy.")

    while True:
      try:
        # Temperatura
        temp = input(
          "¿Cuál es la temperatura actual? (Escribe un número, ej. 25 para 25°C, o 77f para 77°F, o 298k para 298K): ")
        unit = 'c'  # Default a Celsius
        if 'f' in temp.lower():
          unit = 'f'
          temp = temp.replace('f', '').strip()
        elif 'k' in temp.lower():
          unit = 'k'
          temp = temp.replace('k', '').strip()
        temp = self.normalize_temperature(temp, unit)
        if temp is None or not 0 <= temp <= 1:
          raise ValueError("Temperatura inválida. Intenta un valor entre -8°C y 39°C (o equivalente).")

        # Sensación térmica
        atemp = input(
          "¿Cómo se siente la temperatura? (Escribe un número, ej. 25 para 25°C, o 77f para 77°F, o 298k para 298K): ")
        unit = 'c'
        if 'f' in atemp.lower():
          unit = 'f'
          atemp = atemp.replace('f', '').strip()
        elif 'k' in atemp.lower():
          unit = 'k'
          atemp = atemp.replace('k', '').strip()
        atemp = self.normalize_temperature(atemp, unit)
        if atemp is None or not 0 <= atemp <= 1:
          raise ValueError("Sensación térmica inválida. Intenta un valor entre -8°C y 39°C (o equivalente).")

        # Humedad (simplificado a porcentaje)
        hum = float(input("¿Qué tan húmedo está? (Escribe un porcentaje, ej. 50 para 50%): ")) / 100
        if not 0 <= hum <= 1:
          raise ValueError("Humedad debe estar entre 0% y 100%.")

        # Velocidad del viento (simplificado a escala cualitativa)
        wind_input = input("¿Qué tan fuerte sopla el viento? (débil, moderado, fuerte) - Ejemplo: moderado: ").lower()
        wind_map = {'débil': 0.2, 'moderado': 0.5, 'fuerte': 0.8}
        if wind_input not in wind_map:
          raise ValueError("Elige 'débil', 'moderado' o 'fuerte'.")
        windspeed = wind_map[wind_input]

        # Temporada
        season_input = input(
          "¿En qué temporada estamos? (invierno, primavera, verano, otoño) - Ejemplo: verano: ").lower()
        season_map = {'invierno': 1, 'primavera': 2, 'verano': 3, 'otoño': 4}
        if season_input not in season_map:
          raise ValueError("Elige 'invierno', 'primavera', 'verano' o 'otoño'.")
        season = season_map[season_input]

        # Año
        yr_input = input("¿En qué año estamos? (ejemplo: 2025): ")
        try:
          year = int(yr_input)
          if year < 1900 or year > 2100:  # Validación básica del año
            raise ValueError("Por favor ingresa un año válido (entre 1900 y 2100)")
          # Usamos 2020 como punto de referencia: 
          # - años >= 2020 se consideran "nuevos" (1)
          # - años < 2020 se consideran "antiguos" (0)
          yr = 1 if year >= 2020 else 0
        except ValueError as e:
          if "válido" not in str(e):
            raise ValueError("Por favor ingresa un año válido (ejemplo: 2025)")

        # Mes
        mnth = int(input("¿Qué mes es? (ej. 1 para enero, 7 para julio) - Ejemplo: 7: "))
        if not 1 <= mnth <= 12:
          raise ValueError("Mes debe estar entre 1 (enero) y 12 (diciembre).")

        # Día festivo
        holiday_input = input("¿Es hoy un día festivo? (sí o no) - Ejemplo: no: ").lower()
        holiday = 1 if holiday_input in ['sí', 'si'] else 0 if holiday_input == 'no' else None
        if holiday is None:
          raise ValueError("Elige 'sí' o 'no'.")

        # Día de la semana
        weekday_input = input(
          "¿Qué día de la semana es? (domingo, lunes, martes, miércoles, jueves, viernes, sábado) - Ejemplo: jueves: ").lower()
        weekday_map = {'domingo': 0, 'lunes': 1, 'martes': 2, 'miércoles': 3, 'jueves': 4, 'viernes': 5, 'sábado': 6}
        if weekday_input not in weekday_map:
          raise ValueError("Elige un día válido (domingo a sábado).")
        weekday = weekday_map[weekday_input]

        # Día laborable
        workingday_input = input("¿Es hoy un día laborable? (sí o no) - Ejemplo: sí: ").lower()
        workingday = 1 if workingday_input in ['sí', 'si'] else 0 if workingday_input == 'no' else None
        if workingday is None:
          raise ValueError("Elige 'sí' o 'no'.")

        # Condición climática
        weathersit_input = input(
          "¿Qué tiempo hace? (claro, nublado, lluvia ligera, lluvia fuerte) - Ejemplo: claro: ").lower()
        weathersit_map = {'claro': 1, 'nublado': 2, 'lluvia ligera': 3, 'lluvia fuerte': 4}
        if weathersit_input not in weathersit_map:
          raise ValueError("Elige 'claro', 'nublado', 'lluvia ligera' o 'lluvia fuerte'.")
        weathersit = weathersit_map[weathersit_input]

        # Hora
        hr = int(input("¿A qué hora es? (ej. 17 para 5:00 PM) - Ejemplo: 17: "))
        if not 0 <= hr <= 23:
          raise ValueError("Hora debe estar entre 0 (12:00 AM) y 23 (11:00 PM).")

        # Crear diccionario con datos
        new_case = {'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed, 'season': season, 'yr': yr,
                    'mnth': mnth, 'holiday': holiday, 'weekday': weekday, 'workingday': workingday,
                    'weathersit': weathersit, 'hr': hr, 'sin_hr': np.sin(2 * np.pi * hr / 24),
                    'cos_hr': np.cos(2 * np.pi * hr / 24)}

        # Agregar características cíclicas

        return new_case

      except ValueError as e:
        logger.error(f"Entrada inválida: {str(e)}. Intenta de nuevo.")
        continue

  def predict(self, new_case):
    """Realiza la predicción con los datos ingresados."""
    if self.model is None or self.preprocessor is None:
      logger.error("Modelo o preprocessor no cargados correctamente.")
      return None

    try:
      # Convertir a DataFrame
      new_df = pd.DataFrame([new_case])
      X_new = new_df.drop(['hr'], axis=1, errors='ignore')  # Drop hr original

      # Preprocesar datos
      X_new_processed = self.preprocessor.transform(X_new)

      # Predecir
      if self.model_path.endswith('.h5'):  # NN
        prediction = self.model.predict(X_new_processed).flatten()[0]
      else:  # RF/GB
        prediction = self.model.predict(X_new_processed)[0]

      logger.info(f"Predicción: Se espera que se renten unas {prediction:.2f} bicicletas.")
      return prediction

    except Exception as e:
      logger.error(f"Error en la predicción: {str(e)}")
      return None


def main():
  """Función principal para la predicción interactiva."""
  try:
    predictor = BikePredictor()
    if predictor.model is None:
      logger.error("No se pudo cargar el modelo. Verifica que exista el archivo del modelo y el preprocessor.")
      return
    if predictor.preprocessor is None:
      logger.error("No se pudo cargar el preprocessor. Verifica que exista models/preprocessor.pkl")
      return

    logger.info("Modelo y preprocessor cargados correctamente.")
    
    while True:
      new_case = predictor.get_user_input()
      if new_case:
        prediction = predictor.predict(new_case)
        if prediction is not None:
          logger.info(f"¿Quieres probar otra predicción? (sí/no): ")
          if input().lower() not in ['sí', 'si']:
            break
      else:
        logger.info("No se pudo procesar la predicción. Intenta de nuevo.")
  
  except FileNotFoundError as e:
    logger.error(f"Error al inicializar el predictor: {str(e)}")
  except Exception as e:
    logger.error(f"Error inesperado: {str(e)}")

  logger.info("¡Gracias por usar el predictor! Programa terminado.")


if __name__ == "__main__":
  main()
