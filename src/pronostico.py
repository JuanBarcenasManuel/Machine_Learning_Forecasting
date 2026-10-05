"""Pronóstico de demanda diaria con gradient boosting y línea base estacional.

Flujo:
    1. Cargar el Excel de ventas y construir una serie diaria por producto.
    2. Crear atributos usando SOLO información pasada (calendario, lags, medias móviles).
    3. Evaluar con validación temporal contra líneas base simples (naive y naive estacional).
    4. Pronosticar varios días hacia adelante de forma recursiva.

Decisión clave: el modelo principal predice la RAZÓN entre la venta del día y el nivel reciente
(media de los 28 días anteriores), no la venta en bruto. Los árboles no pueden predecir valores
mayores a los que vieron en entrenamiento, y una serie con tendencia sostenida siempre trae días
fuera de ese rango. Con la razón, el patrón (día de la semana, festivo, estacionalidad) queda libre
de escala y la tendencia la aporta el nivel.

Las métricas principales son WAPE y sesgo (bias). R² se deja fuera a propósito: en series
de tiempo con tendencia y estacionalidad es una métrica muy optimista.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

try:  # XGBoost es el modelo principal; si no está instalado se usa un respaldo de scikit-learn.
    from xgboost import XGBRegressor

    HAY_XGBOOST = True
except ImportError:  # pragma: no cover - depende del entorno
    XGBRegressor = None
    HAY_XGBOOST = False

LAGS = (1, 7, 14, 30)
VENTANAS = (7, 30)
OBJETIVO = "Ventas_Netas"
VENTANA_NIVEL = 28  # días usados para el nivel reciente en el modelo de razón
ATRIBUTOS_CALENDARIO = ["día", "díadelasemana", "mes", "trimestre", "díadelaño"]
ATRIBUTOS = (
    ATRIBUTOS_CALENDARIO
    + [f"{OBJETIVO}_lag{k}" for k in LAGS]
    + [f"{OBJETIVO}_media{w}" for w in VENTANAS]
)

# Hiperparámetros conservadores (por defecto) y los del primer intento, para compararlos.
PARAMS_BASE = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)
PARAMS_ORIGINALES = dict(
    n_estimators=1000,
    max_depth=12,
    learning_rate=0.3,
    reg_lambda=0.001,
    random_state=42,
)


# --------------------------------------------------------------------------- datos
def cargar_ventas(ruta: str | Path, hoja: str = "DATA") -> pd.DataFrame:
    """Lee el Excel de ventas. Debe tener al menos las columnas Fecha, Producto y Ventas_Netas."""
    df = pd.read_excel(ruta, sheet_name=hoja)
    faltan = {"Fecha", "Producto", OBJETIVO} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el archivo: {sorted(faltan)}")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def serie_diaria(df: pd.DataFrame, producto: str) -> pd.Series:
    """Suma las ventas de todas las ubicaciones por día para un producto.

    Si faltan días en el calendario se rellenan con 0 (se asume que no hubo venta) y se avisa.
    """
    sub = df[df["Producto"] == producto]
    if sub.empty:
        raise ValueError(f"No hay filas para el producto {producto!r}. Disponibles: {sorted(df['Producto'].unique())}")
    y = sub.groupby("Fecha")[OBJETIVO].sum().sort_index()
    completo = pd.date_range(y.index.min(), y.index.max(), freq="D")
    if len(completo) != len(y):
        warnings.warn(f"Faltan {len(completo) - len(y)} días en la serie; se rellenan con 0.", stacklevel=2)
        y = y.reindex(completo, fill_value=0.0)
    y.index.name = "Fecha"
    return y.astype(float)


# ------------------------------------------------------------------------ atributos
def construir_atributos(y: pd.Series) -> pd.DataFrame:
    """Atributos de calendario, lags y medias móviles. Solo usan valores pasados de y.

    La media móvil usa shift(1) para que el valor del día t nunca entre en su propio atributo.
    """
    X = pd.DataFrame(index=y.index)
    X["día"] = y.index.day
    X["díadelasemana"] = y.index.dayofweek
    X["mes"] = y.index.month
    X["trimestre"] = y.index.quarter
    X["díadelaño"] = y.index.dayofyear
    for k in LAGS:
        X[f"{OBJETIVO}_lag{k}"] = y.shift(k)
    for w in VENTANAS:
        X[f"{OBJETIVO}_media{w}"] = y.shift(1).rolling(w).mean()
    return X[ATRIBUTOS]


def construir_atributos_razon(y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Atributos libres de escala: lags y media de 7 días divididos por el nivel reciente.

    nivel_t = media de los VENTANA_NIVEL días anteriores a t (shift(1), sin usar el día t).
    Devuelve (atributos, nivel).
    """
    nivel = y.shift(1).rolling(VENTANA_NIVEL).mean()
    X = pd.DataFrame(index=y.index)
    for c in ATRIBUTOS_CALENDARIO:
        X[c] = construir_atributos(y)[c]
    for k in LAGS:
        X[f"lag{k}_razon"] = y.shift(k) / nivel
    X["media7_razon"] = y.shift(1).rolling(7).mean() / nivel
    return X, nivel


# --------------------------------------------------------------------------- modelos
def crear_modelo(params: dict | None = None):
    """XGBoost si está instalado; si no, HistGradientBoostingRegressor (misma idea, otra librería)."""
    p = dict(PARAMS_BASE if params is None else params)
    if HAY_XGBOOST:
        return XGBRegressor(objective="reg:squarederror", n_jobs=-1, **p)
    equivalencias = {
        "n_estimators": "max_iter",
        "learning_rate": "learning_rate",
        "max_depth": "max_depth",
        "reg_lambda": "l2_regularization",
        "random_state": "random_state",
    }
    return HistGradientBoostingRegressor(**{equivalencias[k]: v for k, v in p.items() if k in equivalencias})


def nombre_modelo() -> str:
    return "XGBoost" if HAY_XGBOOST else "HistGradientBoosting (respaldo, XGBoost no instalado)"


class Pronosticador:
    """Gradient boosting sobre atributos pasados.

    objetivo="razon" (recomendado): predice venta / nivel reciente y reescala. Maneja tendencia.
    objetivo="nivel": predice la venta en bruto. Se conserva para comparar; falla con tendencia sostenida.
    """

    def __init__(self, objetivo: str = "razon", params: dict | None = None):
        if objetivo not in ("razon", "nivel"):
            raise ValueError("objetivo debe ser 'razon' o 'nivel'")
        self.objetivo, self.params = objetivo, params
        self.modelo = None

    def _atributos(self, y: pd.Series) -> tuple[pd.DataFrame, pd.Series | None]:
        if self.objetivo == "razon":
            return construir_atributos_razon(y)
        return construir_atributos(y), None

    def ajustar(self, y: pd.Series) -> "Pronosticador":
        X, nivel = self._atributos(y)
        X = X.dropna()
        objetivo = y.loc[X.index] / nivel.loc[X.index] if self.objetivo == "razon" else y.loc[X.index]
        self.modelo = crear_modelo(self.params)
        self.modelo.fit(X, objetivo)
        return self

    def predecir_siguiente(self, historia: pd.Series) -> float:
        """Predice la última fila de `historia` (que debe venir como NaN) usando solo las anteriores."""
        X, nivel = self._atributos(historia)
        pred = float(self.modelo.predict(X.iloc[[-1]])[0])
        if self.objetivo == "razon":
            pred *= float(nivel.iloc[-1])
        return max(0.0, pred)

    def predecir_filas(self, y: pd.Series, indice: pd.Index) -> np.ndarray:
        """Predicción a 1 día para las fechas dadas, usando los valores reales anteriores como atributos."""
        X, nivel = self._atributos(y)
        pred = self.modelo.predict(X.loc[indice])
        return pred * nivel.loc[indice].to_numpy() if self.objetivo == "razon" else pred


def ajustar(y: pd.Series, params: dict | None = None, objetivo: str = "razon") -> Pronosticador:
    """Entrena con toda la serie dada."""
    return Pronosticador(objetivo, params).ajustar(y)


# ------------------------------------------------------------------------ pronóstico
def pronosticar(modelo: Pronosticador, y: pd.Series, horizonte: int) -> pd.Series:
    """Pronóstico recursivo: cada predicción entra como lag del día siguiente."""
    historia = y.copy()
    fechas = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq="D")
    predicciones = []
    ventana = max(max(LAGS), VENTANA_NIVEL, max(VENTANAS)) + 3  # filas necesarias para los atributos del último día
    for fecha in fechas:
        historia.loc[fecha] = np.nan
        pred = modelo.predecir_siguiente(historia.iloc[-ventana:])
        historia.loc[fecha] = pred
        predicciones.append(pred)
    return pd.Series(predicciones, index=fechas, name="pronostico")


def naive(y: pd.Series, horizonte: int) -> pd.Series:
    """Repite el último valor observado."""
    fechas = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq="D")
    return pd.Series(float(y.iloc[-1]), index=fechas, name="naive")


def naive_estacional(y: pd.Series, horizonte: int, periodo: int = 7) -> pd.Series:
    """Repite la última semana completa (cada día toma el valor del mismo día de la semana anterior)."""
    fechas = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq="D")
    ultimos = y.iloc[-periodo:].to_numpy()
    valores = [ultimos[i % periodo] for i in range(horizonte)]
    return pd.Series(valores, index=fechas, name="naive_estacional")


# --------------------------------------------------------------------------- métricas
def wape(real, pred) -> float:
    """Error absoluto ponderado: suma|error| / suma|real|. Se lee como % de error sobre el volumen."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.abs(real - pred).sum() / np.abs(real).sum())


def sesgo(real, pred) -> float:
    """Sesgo relativo: (suma pred - suma real) / suma real. Positivo = sobrestima (riesgo de sobrestock)."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float((pred.sum() - real.sum()) / real.sum())


def rmse(real, pred) -> float:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def resumen(real, pred) -> dict:
    return {"WAPE": wape(real, pred), "Sesgo": sesgo(real, pred), "RMSE": rmse(real, pred)}


# -------------------------------------------------------------------------- validación
def evaluar_un_paso(y: pd.Series, n_splits: int = 8, params: dict | None = None) -> pd.DataFrame:
    """Validación temporal a 1 día: usa los valores reales de ayer como lag.

    Es la versión "fácil" (la que suele dar los R² altos). Sirve para comparar con el naive de 1 día
    y con el naive estacional de 7 días, no para decidir compras.
    Métodos: modelo (objetivo razón), modelo_nivel (venta en bruto), naive_1d, naive_estacional_7d.
    """
    validos = construir_atributos(y).dropna().index.intersection(construir_atributos_razon(y)[0].dropna().index)
    filas = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(validos), start=1):
        idx_tr, idx_te = validos[tr], validos[te]
        # Se entrena solo con lo anterior al primer día de prueba
        historia = y.loc[: idx_tr[-1]]
        real = y.loc[idx_te]
        candidatos = {
            "modelo": ajustar(historia, params, "razon").predecir_filas(y, idx_te),
            "modelo_nivel": ajustar(historia, params, "nivel").predecir_filas(y, idx_te),
            "naive_1d": y.shift(1).loc[idx_te].to_numpy(),
            "naive_estacional_7d": y.shift(7).loc[idx_te].to_numpy(),
        }
        for nombre, pred in candidatos.items():
            filas.append({"fold": fold, "método": nombre, **resumen(real, pred)})
    return pd.DataFrame(filas)


def evaluar_horizonte(
    y: pd.Series,
    horizonte: int = 30,
    n_origenes: int = 8,
    separacion: int = 30,
    params: dict | None = None,
) -> pd.DataFrame:
    """Validación de origen móvil a `horizonte` días, con pronóstico recursivo.

    Para cada origen se entrena solo con lo anterior y se pronostica `horizonte` días sin ver el futuro.
    Es la evaluación relevante para planificar compras: no se conocen las ventas reales de los próximos días.
    Métodos: modelo (objetivo razón), modelo_nivel, naive, naive_estacional.
    """
    filas = []
    ultimo = len(y) - horizonte
    for i in range(n_origenes):
        corte = ultimo - (n_origenes - 1 - i) * separacion
        historia, real = y.iloc[:corte], y.iloc[corte : corte + horizonte]
        candidatos = {
            "modelo": pronosticar(ajustar(historia, params, "razon"), historia, horizonte),
            "modelo_nivel": pronosticar(ajustar(historia, params, "nivel"), historia, horizonte),
            "naive": naive(historia, horizonte),
            "naive_estacional": naive_estacional(historia, horizonte),
        }
        for nombre, pred in candidatos.items():
            filas.append(
                {
                    "origen": historia.index[-1].date(),
                    "método": nombre,
                    **resumen(real.to_numpy(), pred.to_numpy()),
                }
            )
    return pd.DataFrame(filas)


def valor_agregado(tabla: pd.DataFrame, base: str = "naive_estacional", metodo: str = "modelo") -> float:
    """Forecast Value Added: cuánto mejora el WAPE medio del modelo frente a la línea base (1 - WAPE_modelo / WAPE_base)."""
    medias = tabla.groupby("método")["WAPE"].mean()
    return float(1 - medias[metodo] / medias[base])
