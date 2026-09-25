C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\Scripts\python.exe C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\main..py
2025-10-13 01:48:37.370875: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

2025-10-13 01:48:50,781 - INFO - ================================================================================
2025-10-13 01:48:50,781 - INFO - INICIO DEL PROYECTO: Predicción de Demanda de Bicicletas (Horaria)
2025-10-13 01:48:50,781 - INFO - ================================================================================
2025-10-13 01:48:50,782 - INFO - Cargando datos desde: hour.csv
2025-10-13 01:48:50,839 - INFO - Datos cargados exitosamente
2025-10-13 01:48:50,840 - INFO - Dimensiones: 17379 filas x 17 columnas
2025-10-13 01:48:50,840 - INFO - Columnas: ['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
2025-10-13 01:48:50,854 - INFO -
================================================================================
2025-10-13 01:48:50,854 - INFO - ANÁLISIS EXPLORATORIO DE DATOS
2025-10-13 01:48:50,854 - INFO - ================================================================================
2025-10-13 01:48:50,855 - INFO -
--- INFORMACIÓN DEL DATASET ---
2025-10-13 01:48:50,884 - INFO -
Primeras filas:
   instant       dteday  season  yr  ...  windspeed  casual  registered  cnt
0        1   2011-01-01       1   0  ...        0.0       3          13   16
1        2   2011-01-01       1   0  ...        0.0       8          32   40
2        3   2011-01-01       1   0  ...        0.0       5          27   32
3        4   2011-01-01       1   0  ...        0.0       3          10   13
4        5   2011-01-01       1   0  ...        0.0       0           1    1

[5 rows x 17 columns]
2025-10-13 01:48:50,886 - INFO -
Información de tipos de datos:
instant         int64
dteday         object
season          int64
yr              int64
mnth            int64
hr              int64
holiday         int64
weekday         int64
workingday      int64
weathersit      int64
temp          float64
atemp         float64
hum           float64
windspeed     float64
casual          int64
registered      int64
cnt             int64
dtype: object
2025-10-13 01:48:50,966 - INFO -
Estadísticas descriptivas:
          instant        season  ...    registered           cnt
count  17379.0000  17379.000000  ...  17379.000000  17379.000000
mean    8690.0000      2.501640  ...    153.786869    189.463088
std     5017.0295      1.106918  ...    151.357286    181.387599
min        1.0000      1.000000  ...      0.000000      1.000000
25%     4345.5000      2.000000  ...     34.000000     40.000000
50%     8690.0000      3.000000  ...    115.000000    142.000000
75%    13034.5000      3.000000  ...    220.000000    281.000000
max    17379.0000      4.000000  ...    886.000000    977.000000

[8 rows x 16 columns]
2025-10-13 01:48:50,969 - INFO -
[OK] No hay valores nulos en el dataset
2025-10-13 01:48:50,969 - INFO -
--- ESTADÍSTICAS DE LA VARIABLE OBJETIVO (cnt) ---
2025-10-13 01:48:50,970 - INFO - Media: 189.46
2025-10-13 01:48:50,971 - INFO - Mediana: 142.00
2025-10-13 01:48:50,971 - INFO - Desviación estándar: 181.39
2025-10-13 01:48:50,971 - INFO - Mínimo: 1.00
2025-10-13 01:48:50,972 - INFO - Máximo: 977.00
2025-10-13 01:48:51,876 - INFO - [OK] Gráfico guardado: plots/01_distribucion_cnt.png
2025-10-13 01:48:53,301 - INFO - [OK] Gráfico guardado: plots/02_correlacion.png
2025-10-13 01:48:54,434 - INFO - [OK] Gráfico guardado: plots/03_grafo_correlacion.png
2025-10-13 01:48:54,434 - INFO - Generando pairplot (puede tardar un momento)...
2025-10-13 01:49:12,661 - INFO - [OK] Gráfico guardado: plots/04_pairplot.png
2025-10-13 01:49:17,704 - INFO - [OK] Gráfico guardado: plots/05_boxplots_categorias.png
2025-10-13 01:49:17,705 - INFO -
================================================================================
2025-10-13 01:49:17,705 - INFO - PREPARACIÓN DE DATOS
2025-10-13 01:49:17,705 - INFO - ================================================================================
2025-10-13 01:49:17,705 - INFO -
Ingeniería de características...
2025-10-13 01:49:17,708 - INFO - [OK] Codificación cíclica aplicada a 'hr' (sin y cos)
2025-10-13 01:49:17,710 - INFO -
Variables predictoras: ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'sin_hr', 'cos_hr']
2025-10-13 01:49:17,710 - INFO - Variable objetivo: cnt
2025-10-13 01:49:17,716 - INFO -
División de datos:
2025-10-13 01:49:17,716 - INFO -   - Entrenamiento: 13903 muestras (80%)
2025-10-13 01:49:17,716 - INFO -   - Prueba: 3476 muestras (20%)
2025-10-13 01:49:17,716 - INFO -
================================================================================
2025-10-13 01:49:17,716 - INFO - CONSTRUCCIÓN DE PIPELINES
2025-10-13 01:49:17,717 - INFO - ================================================================================
2025-10-13 01:49:17,717 - INFO -
Características numéricas: ['temp', 'atemp', 'hum', 'windspeed', 'sin_hr', 'cos_hr']
2025-10-13 01:49:17,717 - INFO - Características categóricas: ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
2025-10-13 01:49:17,717 - INFO -
[OK] Pipelines construidos para Linear, RF y GB
2025-10-13 01:49:17,717 - INFO -
Construyendo modelo de Red Neuronal...
WARNING:tensorflow:From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

2025-10-13 01:49:18,058 - WARNING - From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

2025-10-13 01:49:18.062385: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-10-13 01:49:18,283 - INFO - [OK] Modelo NN construido
2025-10-13 01:49:18,283 - INFO -
Ajuste de hiperparámetros para RF y GB...
2025-10-13 01:51:22,588 - INFO - Mejores params RF: {'regressor__max_depth': 10, 'regressor__n_estimators': 100}
2025-10-13 01:52:57,073 - INFO - Mejores params GB: {'regressor__learning_rate': 0.01, 'regressor__max_depth': 3, 'regressor__n_estimators': 100}
2025-10-13 01:52:57,073 - INFO -
================================================================================
2025-10-13 01:52:57,073 - INFO - ENTRENAMIENTO DE MODELOS
2025-10-13 01:52:57,073 - INFO - ================================================================================
2025-10-13 01:52:57,073 - INFO -
Entrenando Linear...
2025-10-13 01:52:57,144 - INFO - [OK] Linear entrenado
2025-10-13 01:52:57,145 - INFO -
Entrenando RF...
2025-10-13 01:53:03,386 - INFO - [OK] RF entrenado
2025-10-13 01:53:03,386 - INFO -
Entrenando GB...
2025-10-13 01:53:06,304 - INFO - [OK] GB entrenado
2025-10-13 01:53:06,304 - INFO -
Entrenando NN en fases...
WARNING:tensorflow:From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

2025-10-13 01:53:06,661 - WARNING - From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

2025-10-13 01:53:07,077 - WARNING - From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

2025-10-13 01:53:36,232 - INFO - [OK] Fase 1 completada
2025-10-13 01:54:27,737 - INFO - [OK] Fase 2 completada
109/109 [==============================] - 0s 2ms/step
2025-10-13 01:54:28,300 - INFO -
================================================================================
2025-10-13 01:54:28,301 - INFO - EVALUACIÓN DE MODELOS
2025-10-13 01:54:28,301 - INFO - ================================================================================
2025-10-13 01:54:28,304 - INFO -
--- Linear ---
2025-10-13 01:54:28,304 - INFO - MAE: 91.3775
2025-10-13 01:54:28,304 - INFO - RMSE: 124.3051
2025-10-13 01:54:28,304 - INFO - R²: 0.5120
2025-10-13 01:54:28,306 - INFO -
--- RF ---
2025-10-13 01:54:28,307 - INFO - MAE: 31.5161
2025-10-13 01:54:28,307 - INFO - RMSE: 50.4787
2025-10-13 01:54:28,307 - INFO - R²: 0.9195
2025-10-13 01:54:28,310 - INFO -
--- GB ---
2025-10-13 01:54:28,310 - INFO - MAE: 89.3165
2025-10-13 01:54:28,310 - INFO - RMSE: 118.6158
2025-10-13 01:54:28,310 - INFO - R²: 0.5557
2025-10-13 01:54:28,313 - INFO -
--- NN ---
2025-10-13 01:54:28,313 - INFO - MAE: 31.7761
2025-10-13 01:54:28,313 - INFO - RMSE: 49.2667
2025-10-13 01:54:28,313 - INFO - R²: 0.9233
2025-10-13 01:54:28,314 - INFO -
Mejor modelo: NN con R² = 0.9233
2025-10-13 01:54:28,314 - INFO -
================================================================================
2025-10-13 01:54:28,314 - INFO - GENERACIÓN DE GRÁFICOS DE RESULTADOS
2025-10-13 01:54:28,314 - INFO - ================================================================================
2025-10-13 01:54:29,315 - INFO - [OK] Gráfico guardado: plots/06_pred_vs_real.png
2025-10-13 01:54:29,320 - INFO - [OK] Preprocessor guardado en: models/preprocessor.pkl
2025-10-13 01:54:29,380 - INFO - [OK] Mejor modelo (NN) guardado en: models/best_bike_model.pkl
2025-10-13 01:54:29,380 - INFO -
================================================================================
2025-10-13 01:54:29,380 - INFO - PREDICCIÓN PARA NUEVO CASO
2025-10-13 01:54:29,380 - INFO - ================================================================================
1/1 [==============================] - 0s 40ms/step
2025-10-13 01:54:29,501 - INFO - Predicción para el nuevo caso: 702.87 bicicletas
2025-10-13 01:54:29,501 - INFO -
================================================================================
2025-10-13 01:54:29,501 - INFO - PROYECTO FINALIZADO EXITOSAMENTE
2025-10-13 01:54:29,502 - INFO - ================================================================================

Process finished with exit code 0
