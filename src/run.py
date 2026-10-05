"""Ejecuta la evaluación completa y guarda tablas y gráficos en reports/.

Uso (desde la raíz del repositorio):
    python -m src.run --datos data/raw/ventas_diarias_carne_pollo_143k_filas.xlsx --producto Pollo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import pronostico as pr

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", required=True, help="Ruta al Excel de ventas")
    ap.add_argument("--hoja", default="DATA")
    ap.add_argument("--producto", default="Pollo")
    ap.add_argument("--horizonte", type=int, default=30, help="Días a pronosticar")
    ap.add_argument("--origenes", type=int, default=8, help="Cortes de la validación de origen móvil")
    args = ap.parse_args()

    salida = RAIZ / "reports"
    sufijo = args.producto.lower()
    (salida / "figuras").mkdir(parents=True, exist_ok=True)

    y = pr.serie_diaria(pr.cargar_ventas(args.datos, args.hoja), args.producto)
    print(f"Modelo: {pr.nombre_modelo()}")
    print(f"Serie '{args.producto}': {len(y)} días ({y.index.min().date()} a {y.index.max().date()})\n")

    # 1. Validación a 1 día (referencia, no para decidir compras)
    un_paso = pr.evaluar_un_paso(y)
    print("Validación a 1 día (media de folds):")
    print(un_paso.groupby("método")[["WAPE", "Sesgo", "RMSE"]].mean().round(3), "\n")

    # 2. Validación a horizonte (la relevante para planificar)
    horizonte = pr.evaluar_horizonte(y, horizonte=args.horizonte, n_origenes=args.origenes)
    resumen_h = horizonte.groupby("método")[["WAPE", "Sesgo", "RMSE"]].mean()
    print(f"Validación a {args.horizonte} días con pronóstico recursivo (media de {args.origenes} orígenes):")
    print(resumen_h.round(3))
    print(f"\nValor agregado vs naive estacional: modelo {pr.valor_agregado(horizonte):+.1%}"
          f" | modelo_nivel {pr.valor_agregado(horizonte, metodo='modelo_nivel'):+.1%}\n")

    # 3. Comparación de hiperparámetros: los primeros vs. los conservadores
    comp = {}
    for etiqueta, params in {"originales": pr.PARAMS_ORIGINALES, "conservadores": pr.PARAMS_BASE}.items():
        t = pr.evaluar_horizonte(y, horizonte=args.horizonte, n_origenes=args.origenes, params=params)
        comp[etiqueta] = t[t["método"] == "modelo"][["WAPE", "Sesgo", "RMSE"]].mean()
    comp = pd.DataFrame(comp).T
    print("Hiperparámetros (modelo, validación a horizonte):")
    print(comp.round(3), "\n")

    # 4. Pronóstico final con toda la historia
    modelo = pr.ajustar(y)
    futuro = pr.pronosticar(modelo, y, args.horizonte)
    base = pr.naive_estacional(y, args.horizonte)

    resumen_h.round(4).to_csv(salida / f"metricas_horizonte_{sufijo}.csv")
    un_paso.to_csv(salida / f"metricas_un_paso_{sufijo}.csv", index=False)
    comp.round(4).to_csv(salida / f"comparacion_hiperparametros_{sufijo}.csv")
    pd.DataFrame({"pronostico": futuro, "naive_estacional": base}).round(0).to_csv(salida / f"pronostico_final_{sufijo}.csv")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    y.iloc[-120:].plot(ax=ax, color="gray", lw=1.2, label="Venta real")
    futuro.plot(ax=ax, color="tab:green", lw=2.2, label=f"Pronóstico {args.horizonte} días")
    base.plot(ax=ax, color="tab:orange", lw=1, ls="--", label="Naive estacional")
    ax.set_title(f"{args.producto}: últimos 120 días y pronóstico")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(pr.OBJETIVO)
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "figuras" / f"pronostico_{args.producto.lower()}.png", dpi=140)
    print(f"Resultados guardados en {salida}")


if __name__ == "__main__":
    main()
