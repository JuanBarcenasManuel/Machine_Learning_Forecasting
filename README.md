# Machine Learning Forecasting

Pronóstico de ventas diarias de **carne y pollo** con gradient boosting (XGBoost), validado contra **líneas base simples** y evaluado al horizonte que importa para planificar compras: **30 días**, con pronóstico recursivo y sin ver el futuro.

La idea central: un modelo sofisticado solo se justifica si le gana a repetir lo que pasó la semana pasada. Por eso cada resultado se compara contra un *naive estacional* y se mide el **sesgo** además del error.

## Qué incluye

- **Análisis exploratorio** de las ventas (variables categóricas y numéricas, correlaciones, serie de tiempo).
- **Modelos XGBoost con `TimeSeriesSplit`** para Pollo y Carne, con RMSE y R².
- **Figuras interactivas con Plotly**: validación de enero 2025, y la división pre-2025 / test 2025 con la proyección de noviembre.
- **Exportación a Excel** de real, ajuste, validación y proyección.
- **Evaluación contra línea base** (sección 8 del notebook): naive, naive estacional, WAPE, sesgo, valor agregado, y validación de origen móvil a 30 días.
- **Módulo `src/`** con la misma lógica en funciones reutilizables, un script de línea de comandos y pruebas unitarias.

## Hallazgo principal

El primer enfoque predecía la venta en bruto y **perdía contra la línea base**: a 1 día tenía ~4,2 % de WAPE frente a ~3,0 % de repetir la semana pasada, en los 8 folds.

La causa: las ventas crecen de forma sostenida y un árbol no puede predecir por encima de lo que vio al entrenar. La solución fue predecir `venta / nivel`, donde `nivel` es el promedio de los 28 días anteriores, y luego reescalar. Los atributos (lags y media móvil) también se expresan relativos a ese nivel.

El R² no habría mostrado este problema: en la validación de 2025 a 1 día, repetir la semana pasada ya da R² de 0,94 y el modelo en bruto 0,94, frente a 0,999 del modelo en razón. Con WAPE la diferencia sí se ve: 2,8 % del naive frente a 0,5 % del modelo en razón.

## Resultados

Validación de origen móvil: 8 cortes, pronóstico recursivo a 30 días. Serie de 2.861 días por producto (2018-01-01 a 2025-10-31), sin huecos.

**Pollo**

| Método | WAPE | Sesgo |
|---|---|---|
| **Modelo (objetivo en razón)** | **0,44 %** | +0,08 % |
| Modelo en bruto (sin razón) | 4,31 % | +2,30 % |
| Naive estacional (misma semana pasada) | 7,56 % | +2,35 % |
| Naive (último valor) | 11,62 % | −1,10 % |

**Carne**

| Método | WAPE | Sesgo |
|---|---|---|
| **Modelo (objetivo en razón)** | **0,33 %** | +0,06 % |
| Modelo en bruto (sin razón) | 4,64 % | +2,54 % |
| Naive estacional (misma semana pasada) | 7,60 % | +2,41 % |
| Naive (último valor) | 11,59 % | −1,19 % |

Los hiperparámetros conservadores (`max_depth=6`, `learning_rate=0.05`, submuestreo y regularización) mejoran frente a los primeros (`max_depth=12`, `learning_rate=0.3`): 0,44 % vs. 0,57 % de WAPE en Pollo y 0,33 % vs. 0,44 % en Carne.

### Nota importante sobre los datos

Un WAPE de 0,3–0,4 % a 30 días es demasiado bajo para ventas reales de retail. El promedio anual de cada producto sube en escalones casi iguales (Pollo: 24.455, 29.338, 34.207, 39.124, 44.022, 48.911 y 53.751 de 2018 a 2024) y el ruido diario es mínimo, lo que sugiere que el archivo es sintético o muy regular. Estos números demuestran que el método funciona y le gana a la línea base, pero **no deben leerse como el error que se tendría con ventas reales**.

Origen de los datos: archivo de ventas diarias entregado para el proyecto (143.050 filas y 22 columnas). No se incluye en el repositorio.

### Nota sobre el modelo

Las cifras de la sección 8 y de `reports/` se generaron con `HistGradientBoostingRegressor` de scikit-learn (el respaldo del código), porque XGBoost no estaba instalado en el entorno donde se ejecutaron. Con XGBoost instalado, el código lo usa automáticamente; los números pueden variar un poco.

## Estructura

```
.
├── README.md
├── requirements.txt
├── data/raw/                      # coloca aquí el Excel de ventas (no se versiona)
├── notebooks/
│   └── proyecto_pronostico_completo.ipynb   # análisis completo, paso a paso
├── src/
│   ├── pronostico.py              # datos, atributos, modelos, pronóstico y métricas
│   └── run.py                     # ejecuta la evaluación y guarda resultados en reports/
├── tests/
│   └── test_pronostico.py
└── reports/                       # tablas y gráficos generados
```

## Datos

El Excel no se incluye. Debe tener una hoja `DATA` y, como mínimo, las columnas:

| Columna | Descripción |
|---|---|
| `Fecha` | Fecha de la venta |
| `Producto` | `Pollo` o `Carne` |
| `Ventas_Netas` | Venta neta del día (por ubicación; el código suma todas) |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt

# 1. Coloca ventas_diarias_carne_pollo_143k_filas.xlsx en data/raw/
# 2. Notebook completo
jupyter notebook notebooks/proyecto_pronostico_completo.ipynb

# 3. O la evaluación por línea de comandos
python -m src.run --datos data/raw/ventas_diarias_carne_pollo_143k_filas.xlsx --producto Pollo
python -m src.run --datos data/raw/ventas_diarias_carne_pollo_143k_filas.xlsx --producto Carne

# 4. Pruebas
pytest
```

El notebook asume que el Excel está en `data/raw/`. Al ejecutarlo genera archivos en su carpeta (`*.pkl`, `*.sav`, `ultimos_datos.csv` y los Excel de análisis), que `.gitignore` excluye.

## Decisiones de diseño

- **Objetivo en razón.** El modelo predice `venta / nivel` y reescala, con atributos relativos al nivel. Hay pruebas que verifican que estos atributos no dependen de la escala ni ven el futuro, y que el modelo en razón le gana al de nivel en una serie con tendencia.
- **WAPE y sesgo, no solo R².** R² suele salir alto en series con estacionalidad semanal aunque el pronóstico sea mediocre. WAPE se lee directo como porcentaje de error sobre el volumen, y el sesgo indica si el modelo sobreestima (sobrestock) o subestima (quiebres).
- **Sin fuga de datos.** Las medias móviles usan `shift(1)`, de modo que el valor del día nunca entra en su propio atributo.
- **Sin el año como atributo en el modelo en razón.** Un árbol no extrapola fuera de lo que vio.
- **Respaldo de modelo.** Si XGBoost no está instalado, el código usa `HistGradientBoostingRegressor` (misma familia). `nombre_modelo()` indica cuál se usó.

## Limitaciones y siguientes pasos

- Solo usa la propia historia y el calendario. **Promociones, precios y festivos** son los siguientes atributos a incorporar.
- Una venta no es demanda: un **quiebre de stock** censura la demanda real. Conviene depurar o imputar esos periodos antes de entrenar.
- Se pronostica el total por producto. Un siguiente paso es pronosticar por ubicación y **reconciliar** la jerarquía.
- Entrega un pronóstico puntual. Para fijar stock de seguridad conviene producir **cuantiles** o usar el error histórico del pronóstico.

## Autor

Juan Manuel Barcenas
