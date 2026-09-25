C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\Scripts\python.exe C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\predict_interactive.py
2025-10-13 02:04:16.239556: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

2025-10-13 02:04:28.826818: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\backend.py:1398: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

2025-10-13 02:04:29,153 - WARNING - From C:\Users\JavicSoftCode-01\PycharmProjects\Prediccion_Alquiler_Bicicleta\.venv\lib\site-packages\keras\src\backend.py:1398: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

2025-10-13 02:04:29,442 - INFO - Modelo cargado desde models/best_bike_model.h5 (NN)
2025-10-13 02:04:30,806 - INFO - Modelo y preprocessor cargados correctamente.
2025-10-13 02:04:30,806 - INFO - ¡Bienvenido! Ingresa los datos para predecir cuántas bicicletas se rentarán hoy.
¿Cuál es la temperatura actual? (Escribe un número, ej. 25 para 25°C, o 77f para 77°F, o 298k para 298K): 25
¿Cómo se siente la temperatura? (Escribe un número, ej. 25 para 25°C, o 77f para 77°F, o 298k para 298K): 10
¿Qué tan húmedo está? (Escribe un porcentaje, ej. 50 para 50%): 23
¿Qué tan fuerte sopla el viento? (débil, moderado, fuerte) - Ejemplo: moderado: moderado
¿En qué temporada estamos? (invierno, primavera, verano, otoño) - Ejemplo: verano: verano
¿En qué año estamos? (ejemplo: 2025): 2025
¿Qué mes es? (ej. 1 para enero, 7 para julio) - Ejemplo: 7: 4
¿Es hoy un día festivo? (sí o no) - Ejemplo: no: si
¿Qué día de la semana es? (domingo, lunes, martes, miércoles, jueves, viernes, sábado) - Ejemplo: jueves: lunes
¿Es hoy un día laborable? (sí o no) - Ejemplo: sí: si
¿Qué tiempo hace? (claro, nublado, lluvia ligera, lluvia fuerte) - Ejemplo: claro: nublado
¿A qué hora es? (ej. 17 para 5:00 PM) - Ejemplo: 17: 14
1/1 [==============================] - 0s 225ms/step
2025-10-13 02:07:26,415 - INFO - Predicción: Se espera que se renten unas 202.14 bicicletas.
2025-10-13 02:07:26,415 - INFO - ¿Quieres probar otra predicción? (sí/no):
