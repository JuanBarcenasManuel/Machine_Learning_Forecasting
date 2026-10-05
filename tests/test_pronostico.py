import numpy as np
import pandas as pd

from src import pronostico as pr


def serie_prueba(n=400, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    valores = 100 + 20 * np.sin(2 * np.pi * idx.dayofweek / 7) + rng.normal(0, 3, n)
    return pd.Series(valores, index=idx)


def test_wape_y_sesgo_valores_conocidos():
    real, pred = [100, 100], [110, 90]
    assert pr.wape(real, pred) == 0.1
    assert pr.sesgo(real, pred) == 0.0
    assert pr.sesgo([100], [120]) == 0.2  # sobrestima 20 %


def test_atributos_no_ven_el_futuro():
    y = serie_prueba()
    X1 = pr.construir_atributos(y)
    y2 = y.copy()
    y2.iloc[300:] += 1000  # cambiar el futuro no debe alterar atributos anteriores
    X2 = pr.construir_atributos(y2)
    pd.testing.assert_frame_equal(X1.iloc[:300], X2.iloc[:300])


def test_naive_estacional_repite_la_ultima_semana():
    y = serie_prueba()
    f = pr.naive_estacional(y, 14)
    np.testing.assert_allclose(f.iloc[:7].to_numpy(), y.iloc[-7:].to_numpy())
    np.testing.assert_allclose(f.iloc[7:].to_numpy(), y.iloc[-7:].to_numpy())
    assert f.index[0] == y.index[-1] + pd.Timedelta(days=1)


def test_pronostico_recursivo_longitud_y_no_negativo():
    y = serie_prueba()
    modelo = pr.ajustar(y)
    f = pr.pronosticar(modelo, y, 30)
    assert len(f) == 30
    assert (f >= 0).all()


def test_modelo_supera_al_naive_en_serie_con_patron_semanal():
    y = serie_prueba(n=600)
    tabla = pr.evaluar_horizonte(y, horizonte=14, n_origenes=4)
    assert set(tabla["método"]) == {"modelo", "modelo_nivel", "naive", "naive_estacional"}
    assert tabla[tabla["método"] == "modelo"]["WAPE"].mean() < tabla[tabla["método"] == "naive"]["WAPE"].mean()


def serie_con_tendencia(n=500, semilla=1):
    rng = np.random.default_rng(semilla)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    base = 100 * (1 + 0.004 * np.arange(n))
    patron = 1 + 0.2 * np.sin(2 * np.pi * np.asarray(idx.dayofweek) / 7)
    return pd.Series(base * patron * (1 + rng.normal(0, 0.02, n)), index=idx)


def test_atributos_razon_no_dependen_de_la_escala():
    y = serie_con_tendencia()
    X1, _ = pr.construir_atributos_razon(y)
    X2, _ = pr.construir_atributos_razon(y * 5)
    pd.testing.assert_frame_equal(X1, X2)


def test_atributos_razon_no_ven_el_futuro():
    y = serie_con_tendencia()
    X1, n1 = pr.construir_atributos_razon(y)
    y2 = y.copy()
    y2.iloc[300:] += 1000
    X2, n2 = pr.construir_atributos_razon(y2)
    pd.testing.assert_frame_equal(X1.iloc[:300], X2.iloc[:300])
    pd.testing.assert_series_equal(n1.iloc[:300], n2.iloc[:300])


def test_modelo_razon_supera_al_de_nivel_con_tendencia():
    y = serie_con_tendencia()
    t = pr.evaluar_un_paso(y, n_splits=4)
    m = t.groupby("método")["WAPE"].mean()
    assert m["modelo"] < m["modelo_nivel"]
